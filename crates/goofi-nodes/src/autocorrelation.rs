//! Autocorrelation — similarity of a float32 signal with lagged copies of itself
//! along one axis, keeping the non-negative-lag half. Reveals periodicity/repeats.
//! Ported from the Python `nodes/signal/autocorrelation.py`.
//!
//! Per line along the axis this computes `np.correlate(m, m, "full")[N-1:]`, i.e.
//! `R[l] = Σ_{n} m[n]·m[n+l]` for lags `l = 0..N-1` (direct O(N·lag) — no FFT
//! needed here). It then applies the biased (`/N`) or unbiased (`/(N-lag)`)
//! estimator, an optional lag-0 normalization, and an optional tail cutoff.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Autocorrelation {
    normalize: bool,
    biased: bool,
    cutoff: i64,
    axis: i64,
}

impl Node for Autocorrelation {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("signal") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }
        let shape = store.shape();
        let ndim = shape.len();
        if ndim == 0 {
            return Ok(());
        }
        // Python bails on `x.size == 0`; a zero-length dim would also make the
        // per-line loops degenerate, so treat an empty array as a no-op.
        let nelem: usize = shape.iter().product();
        if nelem == 0 {
            return Ok(());
        }

        // Negative axes count from the end (numpy semantics); out-of-range no-ops.
        let axis = if self.axis < 0 { self.axis + ndim as i64 } else { self.axis };
        if axis < 0 || axis as usize >= ndim {
            return Ok(());
        }
        let a = axis as usize;

        let data: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // View the row-major buffer as [outer, axis_len, inner]; each (o, i) is one
        // line along the axis that we autocorrelate independently.
        let axis_len = shape[a];
        let outer: usize = shape[..a].iter().product();
        let inner: usize = shape[a + 1..].iter().product();

        // The non-negative-lag half has `axis_len` lags. A cutoff crops it via
        // numpy `slice(0, cutoff)`; the param range is negative, so a set cutoff
        // trims that many lags off the tail (`N + cutoff`).
        let out_axis_len = if self.cutoff == -1 {
            axis_len
        } else if self.cutoff < 0 {
            (axis_len as i64 + self.cutoff).max(0) as usize
        } else {
            (self.cutoff as usize).min(axis_len)
        };

        let n_f = axis_len as f32;
        let mut result = vec![0f32; outer * out_axis_len * inner];
        let mut m = vec![0f32; axis_len];
        let mut r = vec![0f32; axis_len];
        for o in 0..outer {
            for i in 0..inner {
                // Gather this line along the axis.
                for k in 0..axis_len {
                    m[k] = data[o * axis_len * inner + k * inner + i];
                }
                // Raw non-negative-lag autocorrelation:
                // R[l] = Σ_{n=0}^{N-1-l} m[n]·m[n+l]  == np.correlate(m,m,'full')[N-1:].
                for l in 0..axis_len {
                    let mut s = 0f32;
                    for n in 0..(axis_len - l) {
                        s += m[n] * m[n + l];
                    }
                    r[l] = s;
                }
                // Biased (/N) vs unbiased (/(N-lag)) estimator.
                if self.biased {
                    for v in r.iter_mut() {
                        *v /= n_f;
                    }
                } else {
                    for (l, v) in r.iter_mut().enumerate() {
                        *v /= (axis_len - l) as f32;
                    }
                }
                // Normalize by the (already-scaled) lag-0 value; a zero divisor
                // becomes 1 (matching numpy's `np.where(norm == 0, 1, norm)`).
                if self.normalize {
                    let nf = if r[0] == 0.0 { 1.0 } else { r[0] };
                    for v in r.iter_mut() {
                        *v /= nf;
                    }
                }
                // Scatter the (possibly cropped) lags back into the output line.
                for l in 0..out_axis_len {
                    result[o * out_axis_len * inner + l * inner + i] = r[l];
                }
            }
        }

        let out_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == a { out_axis_len } else { s })
            .collect();

        // Preserve the full meta (sfreq, index, other axes' coords) but drop the
        // processed axis's channel labels — the lag axis no longer maps to the
        // original coordinates, and after a cutoff its length changed too.
        let mut meta = d.meta().clone();
        meta.channels.0.remove(&a);

        let buf: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data =
            Data::from_array_bytes(DType::F32, out_shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("autocorr", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("autocorrelation", "normalize") => {
                if let Some(b) = v.as_bool() {
                    self.normalize = b;
                }
            }
            ("autocorrelation", "biased") => {
                if let Some(b) = v.as_bool() {
                    self.biased = b;
                }
            }
            ("autocorrelation", "cutoff") => {
                if let Some(x) = v.as_i64() {
                    self.cutoff = x;
                }
            }
            ("autocorrelation", "axis") => {
                if let Some(x) = v.as_i64() {
                    self.axis = x;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("normalize".to_string(), Param::boolean(true));
    g.insert("biased".to_string(), Param::boolean(false));
    g.insert("cutoff".to_string(), Param::int(-1, -500, -1));
    g.insert("axis".to_string(), Param::int(-1, -8, 8));
    let mut groups = ParamGroups::new();
    groups.insert("autocorrelation".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let normalize = param(p, "autocorrelation", "normalize")
        .and_then(Param::as_bool)
        .unwrap_or(true);
    let biased = param(p, "autocorrelation", "biased")
        .and_then(Param::as_bool)
        .unwrap_or(false);
    let cutoff = param(p, "autocorrelation", "cutoff")
        .and_then(Param::as_i64)
        .unwrap_or(-1);
    let axis = param(p, "autocorrelation", "axis")
        .and_then(Param::as_i64)
        .unwrap_or(-1);
    Box::new(Autocorrelation {
        normalize,
        biased,
        cutoff,
        axis,
    })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "signal",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "autocorr",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Autocorrelation",
        category: "signal",
        doc: "Autocorrelation along an axis (non-negative lags; biased/unbiased, normalize, cutoff).",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Coord, Data, DType, Meta, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;
    use std::sync::Arc;

    /// Independent f64 reference for one line, mirroring the Python pipeline.
    fn ref_line(m: &[f64], normalize: bool, biased: bool, cutoff: i64) -> Vec<f64> {
        let n = m.len();
        let mut r = vec![0.0f64; n];
        for (l, slot) in r.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in 0..(n - l) {
                s += m[k] * m[k + l];
            }
            *slot = s;
        }
        if biased {
            for v in r.iter_mut() {
                *v /= n as f64;
            }
        } else {
            for (l, v) in r.iter_mut().enumerate() {
                *v /= (n - l) as f64;
            }
        }
        if normalize {
            let nf = if r[0] == 0.0 { 1.0 } else { r[0] };
            for v in r.iter_mut() {
                *v /= nf;
            }
        }
        let out_len = if cutoff == -1 {
            n
        } else if cutoff < 0 {
            (n as i64 + cutoff).max(0) as usize
        } else {
            (cutoff as usize).min(n)
        };
        r.truncate(out_len);
        r
    }

    /// Drive the node with given params + input, returning (shape, values).
    fn run(
        normalize: bool,
        biased: bool,
        cutoff: i64,
        axis: i64,
        shape: Vec<usize>,
        input: &[f32],
    ) -> (Vec<usize>, Vec<f32>) {
        let d = run_meta(normalize, biased, cutoff, axis, shape, input, Meta::empty());
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    /// Same as `run` but with a caller-supplied input meta, returning the Data
    /// (so a test can inspect the output meta).
    fn run_meta(
        normalize: bool,
        biased: bool,
        cutoff: i64,
        axis: i64,
        shape: Vec<usize>,
        input: &[f32],
        meta: Meta,
    ) -> Data {
        let m = goofi_node::find("Autocorrelation").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("autocorrelation", "normalize"), &goofi_core::Param::boolean(normalize)).unwrap();
        node.on_param_changed(&ParamKey::new("autocorrelation", "biased"), &goofi_core::Param::boolean(biased)).unwrap();
        node.on_param_changed(&ParamKey::new("autocorrelation", "cutoff"), &goofi_core::Param::int(cutoff, -500, -1)).unwrap();
        node.on_param_changed(&ParamKey::new("autocorrelation", "axis"), &goofi_core::Param::int(axis, -8, 8)).unwrap();
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, shape, buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("signal", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        outbuf.get("autocorr").unwrap().as_ref().unwrap().clone()
    }

    #[test]
    fn unbiased_normalized_matches_reference() {
        // Default params: normalize=true, biased=false, no cutoff, last axis.
        let sig = [1.0f32, 2.0, 3.0, 4.0];
        let (shape, vals) = run(true, false, -1, -1, vec![4], &sig);
        assert_eq!(shape, vec![4]);
        let want = ref_line(&[1.0, 2.0, 3.0, 4.0], true, false, -1);
        for (g, w) in vals.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 1e-5, "got {g} want {w}");
        }
        // Lag-0 is exactly 1 after normalization, and here the sequence decays.
        assert!((vals[0] - 1.0).abs() < 1e-6);
        for v in &vals[1..] {
            assert!(*v < vals[0], "lag-0 must be the peak: {vals:?}");
        }
    }

    #[test]
    fn biased_unnormalized_exact_values() {
        // R = [30,20,11,4]; biased divides by N=4 -> [7.5, 5.0, 2.75, 1.0].
        let (shape, vals) = run(false, true, -1, -1, vec![4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![4]);
        let want = [7.5f32, 5.0, 2.75, 1.0];
        for (g, w) in vals.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g} want {w}");
        }
    }

    #[test]
    fn normalized_biased_peaks_at_lag_zero() {
        // For a biased+normalized estimator every |R[l]| <= R[0] = 1, so lag-0 is
        // the global max — a known analytic property of autocorrelation.
        let sig = [0.5f32, -1.0, 2.0, -0.3, 1.7, 0.9];
        let (_, vals) = run(true, true, -1, -1, vec![6], &sig);
        assert!((vals[0] - 1.0).abs() < 1e-6);
        for v in &vals {
            assert!(*v <= vals[0] + 1e-6, "lag-0 not the max: {vals:?}");
        }
    }

    #[test]
    fn constant_signal_unbiased_is_flat_ones() {
        // A constant signal's unbiased autocorrelation is exactly flat (each lag l
        // sums N-l equal products, divided by N-l), so normalized it is all ones.
        let (_, vals) = run(true, false, -1, -1, vec![5], &[5.0, 5.0, 5.0, 5.0, 5.0]);
        for v in &vals {
            assert!((v - 1.0).abs() < 1e-5, "constant autocorr should be 1.0: {vals:?}");
        }
    }

    #[test]
    fn cutoff_trims_the_tail() {
        // slice(0, -2) on a length-4 lag axis keeps the first 2 lags.
        let (shape, vals) = run(true, false, -2, -1, vec![4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![2]);
        let want = ref_line(&[1.0, 2.0, 3.0, 4.0], true, false, -2);
        assert_eq!(want.len(), 2);
        for (g, w) in vals.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 1e-5, "got {g} want {w}");
        }
    }

    #[test]
    fn two_d_autocorrelates_per_line_along_axis() {
        // shape [2,3]: [[1,2,3],[4,5,6]], axis 1 -> autocorrelate each row.
        let sig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (shape, vals) = run(true, false, -1, 1, vec![2, 3], &sig);
        assert_eq!(shape, vec![2, 3]);
        let row0 = ref_line(&[1.0, 2.0, 3.0], true, false, -1);
        let row1 = ref_line(&[4.0, 5.0, 6.0], true, false, -1);
        let want: Vec<f64> = row0.into_iter().chain(row1).collect();
        for (g, w) in vals.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 1e-5, "got {g} want {w}");
        }
        // Each row is independently normalized, so both lag-0 entries are 1.
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn two_d_axis0_matches_reference() {
        // shape [3,2]: columns are the lines when axis=0.
        let sig = [1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0];
        let (shape, vals) = run(false, true, -1, 0, vec![3, 2], &sig);
        assert_eq!(shape, vec![3, 2]);
        let col0 = ref_line(&[1.0, 2.0, 3.0], false, true, -1); // strided along axis 0
        let col1 = ref_line(&[10.0, 20.0, 30.0], false, true, -1);
        // Interleave columns back into row-major [3,2] order.
        for l in 0..3 {
            assert!((vals[l * 2] as f64 - col0[l]).abs() < 1e-4);
            assert!((vals[l * 2 + 1] as f64 - col1[l]).abs() < 1e-3);
        }
    }

    #[test]
    fn preserves_sfreq_drops_processed_axis_channels_keeps_others() {
        // 2-D input with coords on both dims + an sfreq; autocorrelating axis 1
        // must drop dim1's coords, keep dim0's, and carry sfreq through.
        let mut ch = std::collections::BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("c0".into()), Coord::Str("c1".into())]));
        ch.insert(1usize, Arc::new(vec![Coord::Num(0.0), Coord::Num(1.0), Coord::Num(2.0)]));
        let meta = Meta { sfreq: Some(256.0), channels: goofi_core::Channels(ch), ..Default::default() };
        let d = run_meta(true, false, -1, 1, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], meta);
        assert_eq!(d.meta().sfreq, Some(256.0), "sfreq preserved");
        assert!(d.meta().channels.0.contains_key(&0), "dim0 coords kept");
        assert!(!d.meta().channels.0.contains_key(&1), "processed dim1 coords dropped");
    }

    #[test]
    fn non_f32_and_missing_and_empty_are_no_ops() {
        let m = goofi_node::find("Autocorrelation").unwrap();

        // Missing input slot -> no emit.
        let mut node = (m.make)(&(m.default_params)());
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("signal", None);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("autocorr").unwrap().is_none());

        // Non-f32 input -> no emit.
        let mut node = (m.make)(&(m.default_params)());
        let frame = Data::from_array_bytes(DType::I32, vec![3], vec![0u8; 12], Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("signal", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("autocorr").unwrap().is_none());

        // Empty array (a zero-length dim) -> no emit.
        let mut node = (m.make)(&(m.default_params)());
        let frame = Data::from_array_bytes(DType::F32, vec![0], Vec::new(), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("signal", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("autocorr").unwrap().is_none());
    }
}
