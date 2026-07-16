//! Signal reduction kernels — the pixel-capacity reduction of an array frame along one
//! axis, dtype-generic over raw little-endian bytes. These are the numerical heart of the
//! ViewSpec data plane (the signal side of Seam B `ViewEncodable`); the shape-level merge
//! that decides WHICH axes to reduce lives in the payload-free `goofi-view` crate.
//!
//! Three kernels, each reducing ONE axis of a row-major array:
//! - **subsample** — pick `m` evenly-spaced indices (dtype-agnostic byte gather); keeps
//!   exact samples (channels, trajectories).
//! - **envelope** — split into `W` bins and emit interleaved `min,max` per bin (axis → `2W`);
//!   preserves a waveform's peaks. Skipped unless it shrinks the axis ≥2×.
//! - **area** — block-mean into `M` bins; smooth downscale for images/spectra.
//!
//! Each returns `None` when it would not actually reduce the axis, so the caller leaves that
//! axis untouched.

use crate::DType;
use goofi_view::ReduceMethod;

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

/// Read the `i`-th element of a raw byte buffer as `f64` (all dtypes except F16, which the
/// caller degrades to subsample). Panics only on a malformed buffer (length re-validated at
/// frame construction upstream).
fn read_f64(b: &[u8], dt: DType, i: usize) -> f64 {
    let sz = dt.itemsize();
    let s = &b[i * sz..i * sz + sz];
    match dt {
        DType::F32 => f32::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::F64 => f64::from_le_bytes(s.try_into().unwrap()),
        DType::I8 => s[0] as i8 as f64,
        DType::I16 => i16::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::I32 => i32::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::I64 => i64::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::U8 | DType::Bool => s[0] as f64,
        DType::U16 => u16::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::U32 => u32::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::U64 => u64::from_le_bytes(s.try_into().unwrap()) as f64,
        DType::F16 => unreachable!("F16 degrades to subsample"),
    }
}

/// Append the `f64` value `v` as `dt`'s little-endian bytes (integers rounded to nearest).
fn write_f64(dt: DType, v: f64, out: &mut Vec<u8>) {
    match dt {
        DType::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
        DType::F64 => out.extend_from_slice(&v.to_le_bytes()),
        DType::I8 => out.push(v.round() as i8 as u8),
        DType::I16 => out.extend_from_slice(&(v.round() as i16).to_le_bytes()),
        DType::I32 => out.extend_from_slice(&(v.round() as i32).to_le_bytes()),
        DType::I64 => out.extend_from_slice(&(v.round() as i64).to_le_bytes()),
        DType::U8 | DType::Bool => out.push(v.round().clamp(0.0, 255.0) as u8),
        DType::U16 => out.extend_from_slice(&(v.round() as u16).to_le_bytes()),
        DType::U32 => out.extend_from_slice(&(v.round() as u32).to_le_bytes()),
        DType::U64 => out.extend_from_slice(&(v.round() as u64).to_le_bytes()),
        DType::F16 => unreachable!("F16 degrades to subsample"),
    }
}

/// The result of reducing one axis: new bytes (row-major), the new length along the reduced
/// axis, and the original-axis indices each output entry maps to (for coordinate
/// co-reduction). For envelope, `centers` holds each bin's midpoint index repeated twice
/// (the min,max pair), so its length equals the new axis length.
pub struct AxisReduction {
    pub bytes: Vec<u8>,
    pub new_len: usize,
    pub centers: Vec<usize>,
}

/// Row-major strides for reducing dimension `dim` of `shape`: (outer count, axis length,
/// inner element count). An element `(o, a, i)` sits at flat element index
/// `(o*axis + a)*inner + i`.
fn strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let axis = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, axis, inner)
}

/// Reduce one axis of a row-major byte array to at most `max` entries via `method`. Returns
/// `None` when it would not shrink the axis (the caller leaves it untouched). F16 envelope/
/// area degrade to subsample (no native f16 arithmetic).
pub fn reduce_axis(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
    dim: usize,
    max: usize,
    method: ReduceMethod,
) -> Option<AxisReduction> {
    if dim >= shape.len() || max == 0 {
        return None;
    }
    let effective = match (method, dtype) {
        // No f64 view of F16 — fall back to exact-sample subsampling.
        (ReduceMethod::Envelope | ReduceMethod::Area, DType::F16) => ReduceMethod::Subsample,
        (m, _) => m,
    };
    match effective {
        ReduceMethod::Subsample => subsample_axis(bytes, shape, dtype, dim, max),
        ReduceMethod::Envelope => envelope_axis(bytes, shape, dtype, dim, max),
        ReduceMethod::Area => area_axis(bytes, shape, dtype, dim, max),
    }
}

fn subsample_axis(bytes: &[u8], shape: &[usize], dtype: DType, dim: usize, max: usize) -> Option<AxisReduction> {
    let (outer, axis, inner) = strides(shape, dim);
    let idx = subsample_idx(axis, max);
    if idx.len() >= axis {
        return None; // no shrink
    }
    let sz = dtype.itemsize();
    let block = inner * sz; // bytes of one (o, a) inner slab
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

fn envelope_axis(bytes: &[u8], shape: &[usize], dtype: DType, dim: usize, max: usize) -> Option<AxisReduction> {
    let (outer, axis, inner) = strides(shape, dim);
    let w = max.min(axis);
    // Envelope doubles the axis (min,max per bin); only worth it if it still shrinks ≥2×.
    if w == 0 || axis < 2 * w {
        return None;
    }
    let edges = bin_edges(axis, w);
    let mut out = Vec::with_capacity(outer * 2 * w * inner * dtype.itemsize());
    let mut centers = Vec::with_capacity(2 * w);
    for o in 0..outer {
        for b in 0..w {
            let (lo, hi) = (edges[b], edges[b + 1].max(edges[b] + 1).min(axis));
            let mut mn = vec![f64::INFINITY; inner];
            let mut mx = vec![f64::NEG_INFINITY; inner];
            for a in lo..hi {
                for i in 0..inner {
                    let v = read_f64(bytes, dtype, (o * axis + a) * inner + i);
                    if v < mn[i] {
                        mn[i] = v;
                    }
                    if v > mx[i] {
                        mx[i] = v;
                    }
                }
            }
            for &v in &mn {
                write_f64(dtype, v, &mut out);
            }
            for &v in &mx {
                write_f64(dtype, v, &mut out);
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

fn area_axis(bytes: &[u8], shape: &[usize], dtype: DType, dim: usize, max: usize) -> Option<AxisReduction> {
    let (outer, axis, inner) = strides(shape, dim);
    let m = max.min(axis);
    if m == 0 || m >= axis {
        return None;
    }
    let edges = bin_edges(axis, m);
    let mut out = Vec::with_capacity(outer * m * inner * dtype.itemsize());
    let mut centers = Vec::with_capacity(m);
    for o in 0..outer {
        for b in 0..m {
            let (lo, hi) = (edges[b], edges[b + 1].max(edges[b] + 1).min(axis));
            for i in 0..inner {
                let mut sum = 0.0;
                for a in lo..hi {
                    sum += read_f64(bytes, dtype, (o * axis + a) * inner + i);
                }
                write_f64(dtype, sum / (hi - lo) as f64, &mut out);
            }
            if o == 0 {
                centers.push((lo + hi.saturating_sub(1)) / 2);
            }
        }
    }
    Some(AxisReduction { bytes: out, new_len: m, centers })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn as_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    #[test]
    fn subsample_idx_is_evenly_spaced_with_endpoints() {
        assert_eq!(subsample_idx(10, 3), vec![0, 5, 9]);
        assert_eq!(subsample_idx(5, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(subsample_idx(5, 9), vec![0, 1, 2, 3, 4], "m>=n returns all");
        assert_eq!(subsample_idx(100, 1), vec![0]);
    }

    #[test]
    fn subsample_axis_gathers_rows_dtype_agnostic() {
        // (3 channels, 2 samples) u8 — subsample the channel axis to 2 → rows 0 and 2.
        let bytes: Vec<u8> = vec![10, 11, /*c0*/ 20, 21, /*c1*/ 30, 31 /*c2*/];
        let r = reduce_axis(&bytes, &[3, 2], DType::U8, 0, 2, ReduceMethod::Subsample).unwrap();
        assert_eq!(r.new_len, 2);
        assert_eq!(r.centers, vec![0, 2]);
        assert_eq!(r.bytes, vec![10, 11, 30, 31], "kept channels 0 and 2, both samples each");
    }

    #[test]
    fn envelope_axis_emits_min_max_per_bin() {
        // 1-D f32 of 8 samples → W=2 bins → 4 outputs: [min,max] of [1,4,2,3] and [8,5,7,6].
        let d = f32_bytes(&[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0]);
        let r = reduce_axis(&d, &[8], DType::F32, 0, 2, ReduceMethod::Envelope).unwrap();
        assert_eq!(r.new_len, 4);
        assert_eq!(as_f32(&r.bytes), vec![1.0, 4.0, 5.0, 8.0], "min,max per bin");
        assert_eq!(r.centers.len(), 4, "one center per output entry");
    }

    #[test]
    fn envelope_skips_when_it_would_not_shrink_twofold() {
        // 6 samples, max 4 → W=4, 2W=8 > 6 → no reduction.
        let d = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(reduce_axis(&d, &[6], DType::F32, 0, 4, ReduceMethod::Envelope).is_none());
    }

    #[test]
    fn envelope_per_channel_on_2d() {
        // (2 channels, 4 samples), W=2 → each channel → [min,max]×2 = (2,4).
        let d = f32_bytes(&[0.0, 2.0, 1.0, 3.0, /*c0*/ 9.0, 5.0, 8.0, 6.0 /*c1*/]);
        let r = reduce_axis(&d, &[2, 4], DType::F32, 1, 2, ReduceMethod::Envelope).unwrap();
        assert_eq!(r.new_len, 4);
        // c0: [min(0,2),max(0,2), min(1,3),max(1,3)] = [0,2,1,3]; c1: [5,9,6,8]
        assert_eq!(as_f32(&r.bytes), vec![0.0, 2.0, 1.0, 3.0, 5.0, 9.0, 6.0, 8.0]);
    }

    #[test]
    fn area_axis_is_block_mean() {
        // 1-D f32 of 6 → 3 bins of 2 → means [1.5, 3.5, 5.5].
        let d = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reduce_axis(&d, &[6], DType::F32, 0, 3, ReduceMethod::Area).unwrap();
        assert_eq!(r.new_len, 3);
        assert_eq!(as_f32(&r.bytes), vec![1.5, 3.5, 5.5]);
    }

    #[test]
    fn area_2d_block_mean_over_the_inner_axis() {
        // (2, 4) mean the last axis to 2 → per channel means of pairs.
        let d = f32_bytes(&[1.0, 3.0, 5.0, 7.0, /*c0*/ 0.0, 0.0, 10.0, 10.0 /*c1*/]);
        let r = reduce_axis(&d, &[2, 4], DType::F32, 1, 2, ReduceMethod::Area).unwrap();
        assert_eq!(as_f32(&r.bytes), vec![2.0, 6.0, 0.0, 10.0]);
    }

    #[test]
    fn no_reduction_when_already_small() {
        let d = f32_bytes(&[1.0, 2.0, 3.0]);
        assert!(reduce_axis(&d, &[3], DType::F32, 0, 10, ReduceMethod::Subsample).is_none());
        assert!(reduce_axis(&d, &[3], DType::F32, 0, 10, ReduceMethod::Area).is_none());
    }

    #[test]
    fn f16_envelope_degrades_to_subsample() {
        // Two f16 elements (raw bytes irrelevant) — envelope on F16 must not panic; it
        // subsamples instead. 4 elems → subsample to 2.
        let d: Vec<u8> = vec![0, 0, 1, 0, 2, 0, 3, 0];
        let r = reduce_axis(&d, &[4], DType::F16, 0, 2, ReduceMethod::Envelope).unwrap();
        assert_eq!(r.new_len, 2, "degraded to a 2-element subsample");
        assert_eq!(r.bytes, vec![0, 0, 3, 0], "gathered elements 0 and 3");
    }
}
