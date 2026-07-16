//! FrequencyShift — translate a signal's frequency content by a fixed number of
//! Hz using the FFT bin-shift method: FFT the (flattened) time-domain signal,
//! slide the spectrum by `round(freq_shift * n / sfreq)` bins (zero-filling the
//! vacated bins), inverse-FFT, and keep the real part. Ported from the Python
//! `nodes/signal/frequencyshift.py`. The heavy lifting lives in
//! `goofi_dsp::frequency_shift`; this node only computes the bin count and keeps
//! `sfreq` on the output (the shift is length-preserving, so `index` rides along).

use goofi_core::SlotType;
use goofi_core::{Data, DType, Meta, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct FrequencyShift {
    freq_shift: f64,
    // Kept across ticks so a steady window length never re-plans the FFT/IFFT.
    planner: goofi_dsp::FftPlanner<f32>,
}

impl Node for FrequencyShift {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }
        // Flatten to 1-D: the row-major byte buffer IS the flattened signal.
        let signal: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let n = signal.len();
        if n == 0 {
            return Ok(());
        }
        // The Python raises when sfreq is absent; here we no-op (the repo idiom).
        // sfreq <= 0 would make the bin count non-finite, so guard it too.
        let Some(sfreq) = d.meta().sfreq else {
            return Ok(());
        };
        if sfreq <= 0.0 {
            return Ok(());
        }

        // Round-HALF-TO-EVEN to match Python's built-in round() (banker's), so a
        // shift landing exactly on a half-bin (e.g. 0.5 Hz over a 1 s window) picks
        // the same bin as Python — `f64::round` rounds half AWAY from zero and would
        // be off by one there.
        let delta_bins = (self.freq_shift * n as f64 / sfreq).round_ties_even() as i64;
        let shifted = goofi_dsp::frequency_shift(&mut self.planner, &signal, delta_bins);

        let buf: Vec<u8> = shifted.iter().flat_map(|v| v.to_le_bytes()).collect();
        // A genuinely 1-D input is unchanged by the flatten, so its full meta (incl.
        // dim0 channel labels, which stay length-n) carries through, matching the
        // Python `return data.meta`. A multi-dim input's per-axis coords would no
        // longer fit the 1-D output (Python's Data ctor raises there), so drop them.
        let meta = if store.shape().len() == 1 {
            d.meta().clone()
        } else {
            Meta {
                sfreq: Some(sfreq),
                index: d.meta().index,
                ..Default::default()
            }
        };
        let data = Data::from_array_bytes(DType::F32, vec![n], buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if (key.group.as_str(), key.name.as_str()) == ("shift", "frequency_shift") {
            if let Some(x) = v.as_f64() {
                self.freq_shift = x;
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("frequency_shift".to_string(), Param::float(1.0, -1000.0, 1000.0));
    let mut groups = ParamGroups::new();
    groups.insert("shift".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let freq_shift = param(p, "shift", "frequency_shift").and_then(Param::as_f64).unwrap_or(1.0);
    Box::new(FrequencyShift {
        freq_shift,
        planner: goofi_dsp::FftPlanner::new(),
    })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "FrequencyShift",
        category: "signal",
        doc: "Shift a signal's frequency content by a fixed number of Hz (FFT bin-shift).",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use goofi_core::{Data, DType, Meta, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;

    /// Drive the node once: shift `input` (shape `shape`, sample rate `fs`) by
    /// `freq_shift` Hz and return the emitted `(shape, values, out_meta)`.
    fn shift(
        freq_shift: f64,
        fs: Option<f64>,
        index: Option<u64>,
        shape: Vec<usize>,
        input: &[f32],
    ) -> Option<(Vec<usize>, Vec<f32>, Meta)> {
        let m = goofi_node::find("FrequencyShift").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(
            &ParamKey::new("shift", "frequency_shift"),
            &goofi_core::Param::float(freq_shift, -1000.0, 1000.0),
        )
        .unwrap();
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let meta = Meta { sfreq: fs, index, ..Default::default() };
        let frame = Data::from_array_bytes(DType::F32, shape, buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let out = outbuf.get("out").unwrap().as_ref()?.clone();
        match out.value() {
            Value::Array(s) => Some((
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                out.meta().clone(),
            )),
            _ => panic!("expected array"),
        }
    }

    fn sine(f: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * PI * f * i as f32 / fs).sin()).collect()
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }

    #[test]
    fn zero_shift_is_identity_and_keeps_meta() {
        // freq_shift 0 -> delta_bins 0 -> the FFT round-trip recovers the signal.
        // sfreq and the continuity index must survive on the output.
        let sig = sine(16.0, 128.0, 128);
        let (shape, out, meta) = shift(0.0, Some(128.0), Some(7), vec![128], &sig).unwrap();
        assert_eq!(shape, vec![128]);
        assert_eq!(meta.sfreq, Some(128.0));
        assert_eq!(meta.index, Some(7), "length-preserving node propagates index");
        for (o, s) in out.iter().zip(&sig) {
            assert!((o - s).abs() < 1e-3, "identity drifted: {o} vs {s}");
        }
    }

    #[test]
    fn positive_shift_moves_spectral_peak_up() {
        // Shift a 16 Hz sine up by 8 Hz at fs=128, n=128 (1 Hz per bin). The shift
        // splits it into tones at 8 and 24 Hz; the higher, newly-created 24 Hz tone
        // is what "shift up" produced — assert real energy landed there, and NONE
        // remained at the original 16 Hz bin.
        let (_shape, out, _meta) = shift(8.0, Some(128.0), None, vec![128], &sine(16.0, 128.0, 128)).unwrap();
        let mag = goofi_dsp::magnitude_spectrum(&mut goofi_dsp::FftPlanner::new(), &out);
        assert!(mag[24] > 10.0, "expected a strong 24 Hz component, got {}", mag[24]);
        assert!(mag[16] < 1.0, "original 16 Hz bin must be vacated, got {}", mag[16]);
        // Both created tones (8 and 24) dominate everything else.
        assert!(matches!(argmax(&mag), 8 | 24));
    }

    #[test]
    fn flattens_multidim_input_to_1d() {
        // A 2x64 input is flattened to length 128 before the transform; the output
        // is 1-D and channel labels are dropped (they no longer describe the axis).
        let sig = sine(10.0, 128.0, 128);
        let (shape, out, meta) = shift(0.0, Some(128.0), None, vec![2, 64], &sig).unwrap();
        assert_eq!(shape, vec![128]);
        assert!(meta.channels.0.is_empty(), "flatten drops per-axis coords");
        for (o, s) in out.iter().zip(&sig) {
            assert!((o - s).abs() < 1e-3);
        }
    }

    #[test]
    fn no_emit_without_sfreq() {
        // Missing sfreq is a silent no-op (the Python raises; the engine no-ops).
        assert!(shift(5.0, None, None, vec![64], &sine(10.0, 128.0, 64)).is_none());
    }

    #[test]
    fn no_emit_on_non_f32_or_empty() {
        // Non-f32 dtype -> no emit.
        let m = goofi_node::find("FrequencyShift").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let frame = Data::from_array_bytes(
            DType::I32,
            vec![4],
            (0i32..4).flat_map(|v| v.to_le_bytes()).collect(),
            Meta { sfreq: Some(128.0), ..Default::default() },
        )
        .unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("out").unwrap().is_none(), "non-f32 must not emit");

        // Absent input slot -> no emit, no panic.
        let empty: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        let mut outbuf = m.output_buffer();
        node.process(&Inputs::new(&empty), &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("out").unwrap().is_none());
    }

    #[test]
    fn half_bin_shift_rounds_ties_to_even_like_python() {
        // freq_shift=0.5, n=128, sfreq=128 -> exactly 0.5 bins. Python round(0.5)=0
        // (banker's), so delta_bins=0 and the output is the identity. f64::round would
        // give 1 (a real 1-bin shift) — this locks in the banker's-rounding fix.
        let sig = sine(16.0, 128.0, 128);
        let (_shape, out, _m) = shift(0.5, Some(128.0), None, vec![128], &sig).unwrap();
        for (o, s) in out.iter().zip(&sig) {
            assert!((o - s).abs() < 1e-3, "0.5-bin shift must round to 0 (identity), got {o} vs {s}");
        }
    }

    #[test]
    fn one_d_input_keeps_its_channels() {
        // A genuinely 1-D input is unchanged by the flatten, so its dim0 labels (still
        // length n) carry through, matching Python's `return data.meta`.
        use goofi_core::{Channels, Coord};
        use std::collections::BTreeMap;
        let m = goofi_node::find("FrequencyShift").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("shift", "frequency_shift"), &goofi_core::Param::float(0.0, -1000.0, 1000.0)).unwrap();
        let mut ch = BTreeMap::new();
        ch.insert(0usize, std::sync::Arc::new((0..4).map(|i| Coord::Num(i as f64)).collect::<Vec<_>>()));
        let meta = Meta { sfreq: Some(8.0), channels: Channels(ch), ..Default::default() };
        let frame = Data::from_array_bytes(DType::F32, vec![4], [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect(), meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        assert_eq!(d.meta().channels.0.get(&0).map(|c| c.len()), Some(4), "1-D input keeps dim0 labels");
    }

    #[test]
    fn huge_shift_zeros_the_signal_without_panicking() {
        // delta_bins far exceeding n zeros the whole spectrum -> an all-zero output
        // (matching numpy's empty slice assignment), and must not panic.
        let (shape, out, _m) = shift(1000.0, Some(1.0), None, vec![64], &sine(10.0, 128.0, 64)).unwrap();
        assert_eq!(shape, vec![64]);
        assert!(out.iter().all(|&v| v.abs() < 1e-4), "over-large shift -> all zeros");
    }
}
