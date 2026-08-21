//! Axis reduction kernels — subsample, envelope (interleaved `min,max` per bin) and area — over
//! a frame's f32 LE bytes. Each returns `None` when it would not shrink the axis.

use crate::{Coord, Data, MetaValue, Value};
use goofi_view::{MergedViewSpec, ReduceMethod};
use std::borrow::Cow;
use std::collections::BTreeMap;

fn method_name(m: ReduceMethod) -> &'static str {
    match m {
        ReduceMethod::Envelope => "envelope",
        ReduceMethod::Subsample => "subsample",
        ReduceMethod::Area => "area",
    }
}

/// Evaluate a merged view plan against a frame: reduce each planned axis, co-reduce the coords,
/// and record `meta.reduced`. Fail-open — an unreconstructible result returns the source frame.
pub fn reduce_for_view(frame: &Data, plan: &MergedViewSpec) -> Data {
    let Value::Array(store) = frame.value() else {
        return frame.clone();
    };
    if plan.axes.is_empty() {
        return frame.clone();
    }
    // Borrowed until an axis actually shrinks: a plan whose axes all fit is the common path.
    let mut bytes = Cow::Borrowed(store.as_bytes());
    let mut shape = store.shape().to_vec();
    let mut axes = frame.meta().channels().clone();
    let mut reduced: BTreeMap<String, MetaValue> = BTreeMap::new();

    // Descending dim so a reduction never invalidates a not-yet-processed lower dim.
    let mut planned = plan.axes.clone();
    planned.sort_by_key(|ax| std::cmp::Reverse(ax.dim));
    for ax in &planned {
        let Some(r) = reduce_axis(&bytes, &shape, ax.dim, ax.max, ax.method) else {
            continue;
        };
        let orig_len = shape[ax.dim];
        // Capture the ORIGINAL coords before slicing, so a small subsampled axis keeps exact labels.
        let verbatim = (ax.method == ReduceMethod::Subsample && orig_len <= 4096)
            .then(|| axes.get(ax.dim).and_then(|a| a.coords.clone()))
            .flatten();
        bytes = Cow::Owned(r.bytes);
        shape[ax.dim] = r.new_len;
        axes = axes.sliced(ax.dim, &r.centers);
        let mut entry = BTreeMap::new();
        entry.insert("orig_len".to_string(), MetaValue::Uint(orig_len as u64));
        entry.insert("method".to_string(), MetaValue::Str(method_name(ax.method).to_string()));
        if let Some(coords) = verbatim {
            let list = coords
                .iter()
                .map(|c| match c {
                    Coord::Num(n) => MetaValue::Float(*n),
                    Coord::Str(s) => MetaValue::Str(s.to_string()),
                })
                .collect();
            entry.insert("orig_coord".to_string(), MetaValue::List(list));
        }
        reduced.insert(ax.dim.to_string(), MetaValue::Map(entry));
    }
    if reduced.is_empty() {
        return frame.clone();
    }
    let mut meta = frame.meta().clone();
    meta.set_channels(axes);
    meta.set_reduced(Some(MetaValue::Map(reduced)));
    Data::array_f32(shape, bytes.into_owned(), meta).unwrap_or_else(|_| frame.clone())
}

/// `m` evenly-spaced indices into `0..n` (inclusive endpoints, like `np.linspace(0,n-1,m)`).
pub fn subsample_idx(n: usize, m: usize) -> Vec<usize> {
    if n == 0 || m == 0 {
        return Vec::new();
    }
    if m >= n {
        return (0..n).collect();
    }
    if m == 1 {
        return vec![0];
    }
    (0..m)
        .map(|i| ((i as f64) * (n - 1) as f64 / (m - 1) as f64).round() as usize)
        .collect()
}

/// Panics only on a malformed buffer — the length is validated at frame construction.
fn read_f32(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}

fn write_f32(v: f32, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// One axis's reduction: new bytes, the new axis length, and the original index each entry maps
/// to. For envelope, a bin's midpoint appears twice, so `centers` still matches the new length.
pub struct AxisReduction {
    pub bytes: Vec<u8>,
    pub new_len: usize,
    pub centers: Vec<usize>,
}

/// Row-major strides for dimension `dim`: (outer count, axis length, inner element count), so
/// element `(o, a, i)` sits at flat index `(o*axis + a)*inner + i`.
fn strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let axis = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, axis, inner)
}

/// Reduce one axis to at most `max` entries; `None` when it would not shrink the axis.
pub fn reduce_axis(
    bytes: &[u8],
    shape: &[usize],
    dim: usize,
    max: usize,
    method: ReduceMethod,
) -> Option<AxisReduction> {
    if dim >= shape.len() || max == 0 {
        return None;
    }
    match method {
        ReduceMethod::Subsample => subsample_axis(bytes, shape, dim, max),
        ReduceMethod::Envelope => envelope_axis(bytes, shape, dim, max),
        ReduceMethod::Area => area_axis(bytes, shape, dim, max),
    }
}

fn subsample_axis(bytes: &[u8], shape: &[usize], dim: usize, max: usize) -> Option<AxisReduction> {
    let (outer, axis, inner) = strides(shape, dim);
    let idx = subsample_idx(axis, max);
    if idx.len() >= axis {
        return None;
    }
    let block = inner * 4;
    let mut out = Vec::with_capacity(outer * idx.len() * block);
    for o in 0..outer {
        for &a in &idx {
            let start = (o * axis + a) * block;
            out.extend_from_slice(&bytes[start..start + block]);
        }
    }
    Some(AxisReduction { bytes: out, new_len: idx.len(), centers: idx })
}

/// Integer bin edges: `bins+1` boundaries evenly spanning `0..axis`.
fn bin_edges(axis: usize, bins: usize) -> Vec<usize> {
    (0..=bins).map(|b| ((b as u64 * axis as u64) / bins as u64) as usize).collect()
}

fn envelope_axis(bytes: &[u8], shape: &[usize], dim: usize, max: usize) -> Option<AxisReduction> {
    let (outer, axis, inner) = strides(shape, dim);
    let w = max.min(axis);
    // Envelope doubles the axis (min,max per bin); only worth it if it still shrinks ≥2×.
    if w == 0 || axis < 2 * w {
        return None;
    }
    let edges = bin_edges(axis, w);
    let mut out = Vec::with_capacity(outer * 2 * w * inner * 4);
    let mut centers = Vec::with_capacity(2 * w);
    for o in 0..outer {
        for b in 0..w {
            let (lo, hi) = (edges[b], edges[b + 1].max(edges[b] + 1).min(axis));
            let mut mn = vec![f32::INFINITY; inner];
            let mut mx = vec![f32::NEG_INFINITY; inner];
            for a in lo..hi {
                for i in 0..inner {
                    let v = read_f32(bytes, (o * axis + a) * inner + i);
                    if v < mn[i] {
                        mn[i] = v;
                    }
                    if v > mx[i] {
                        mx[i] = v;
                    }
                }
            }
            // An all-NaN bin wins no comparison, so both seeds survive — emit NaN rather than
            // the (+INF, -INF) pair, which would wreck a viewer's autoscale.
            let all_nan = |i: usize| mn[i] == f32::INFINITY && mx[i] == f32::NEG_INFINITY;
            for (i, &v) in mn.iter().enumerate() {
                write_f32(if all_nan(i) { f32::NAN } else { v }, &mut out);
            }
            for (i, &v) in mx.iter().enumerate() {
                write_f32(if all_nan(i) { f32::NAN } else { v }, &mut out);
            }
            if o == 0 {
                let mid = (lo + hi.saturating_sub(1)) / 2;
                centers.push(mid);
                centers.push(mid);
            }
        }
    }
    Some(AxisReduction { bytes: out, new_len: 2 * w, centers })
}

fn area_axis(bytes: &[u8], shape: &[usize], dim: usize, max: usize) -> Option<AxisReduction> {
    let (outer, axis, inner) = strides(shape, dim);
    let m = max.min(axis);
    if m == 0 || m >= axis {
        return None;
    }
    let edges = bin_edges(axis, m);
    let mut out = Vec::with_capacity(outer * m * inner * 4);
    let mut centers = Vec::with_capacity(m);
    for o in 0..outer {
        for b in 0..m {
            let (lo, hi) = (edges[b], edges[b + 1].max(edges[b] + 1).min(axis));
            for i in 0..inner {
                // Accumulate in f64 for precision, emit f32.
                let mut sum = 0.0f64;
                for a in lo..hi {
                    sum += read_f32(bytes, (o * axis + a) * inner + i) as f64;
                }
                write_f32((sum / (hi - lo) as f64) as f32, &mut out);
            }
            if o == 0 {
                centers.push((lo + hi.saturating_sub(1)) / 2);
            }
        }
    }
    Some(AxisReduction { bytes: out, new_len: m, centers })
}
