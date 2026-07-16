//! PowerBandEEG — split a PSD into the six standard EEG bands (delta, theta,
//! alpha, low/high beta, gamma), emitting the total (or relative) power in each
//! as its own output. Reads the frequency axis from `meta["channels"]` (dim0 for
//! a 1-D PSD, dim1 for a 2-D [channels, freqs] PSD), sums the bins inside each
//! band along the last axis, and — for `power_type = "relative"` — divides by the
//! full-row power. Every output drops the summed-out frequency axis from its meta
//! and tags on `freq_min`/`freq_max`. Ported from
//! `nodes/analysis/powerbandeeg.py`.

use goofi_core::SlotType;
use goofi_core::{Coord, Data, DType, MetaValue, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

/// The fixed EEG bands (name, f_min, f_max in Hz). Bands share their boundary Hz
/// (delta ends and theta starts at 3): the selection is inclusive on both ends,
/// so a bin at exactly the boundary counts toward both — faithful to the Python.
const BANDS: [(&str, i64, i64); 6] = [
    ("delta", 1, 3),
    ("theta", 3, 7),
    ("alpha", 7, 12),
    ("lowbeta", 12, 20),
    ("highbeta", 20, 30),
    ("gamma", 30, 50),
];

struct PowerBandEeg {
    relative: bool,
}

impl Node for PowerBandEeg {
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
        let shape = store.shape();
        let ndim = shape.len();
        // Only 1-D and 2-D [channels, freqs] PSDs are defined (like the Python port).
        if ndim != 1 && ndim != 2 {
            return Ok(());
        }

        // The Python keys the frequency axis strictly off ndim: dim0 for a 1-D PSD,
        // dim1 for a 2-D [C, F] PSD. A missing axis is a KeyError there -> no-emit here.
        let freq_dim = if ndim == 1 { 0usize } else { 1usize };
        let Some(coords) = d.meta().channels.0.get(&freq_dim) else {
            return Ok(());
        };
        let mut freq = Vec::with_capacity(coords.len());
        for c in coords.iter() {
            match c {
                Coord::Num(n) => freq.push(*n),
                Coord::Str(_) => return Ok(()), // a string axis is not a frequency axis
            }
        }

        // The band is summed along the LAST axis (the freq axis of a well-formed
        // PSD); its coord list must label that axis 1:1 or selection is ill-defined.
        let f = shape[ndim - 1];
        if freq.len() != f {
            return Ok(());
        }
        // Nudge a DC bin sitting at exactly 0 Hz off zero, matching the Python guard.
        if let Some(first) = freq.first_mut() {
            if *first == 0.0 {
                *first = 1e-8;
            }
        }
        // Treat a 1-D PSD as a single row so both cases share one loop.
        let rows = if ndim == 1 { 1 } else { shape[0] };

        let data: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Relative power divides by the full-row power (Python np.sum(data, axis=-1)).
        // Computed once per row and reused across all six bands.
        let totals: Option<Vec<f32>> = self.relative.then(|| {
            (0..rows)
                .map(|row| {
                    let base = row * f;
                    data[base..base + f].iter().sum()
                })
                .collect()
        });

        let out_shape: Vec<usize> = if ndim == 1 { vec![1] } else { vec![rows] };
        // Base meta shared by every band: input meta with the summed-out frequency
        // axis's coords dropped (its dim is the last one, so lower axes keep their
        // index — no shift needed). Each band clones this and adds its Hz bounds.
        let mut base_meta = d.meta().clone();
        base_meta.channels.0.remove(&freq_dim);

        for (name, f_min, f_max) in BANDS {
            let (fmin, fmax) = (f_min as f64, f_max as f64);
            let selected: Vec<usize> =
                (0..f).filter(|&i| freq[i] >= fmin && freq[i] <= fmax).collect();

            let mut result = vec![0f32; rows];
            for (row, slot) in result.iter_mut().enumerate() {
                let base = row * f;
                let band: f32 = selected.iter().map(|&i| data[base + i]).sum();
                *slot = match &totals {
                    Some(t) => band / t[row],
                    None => band,
                };
            }

            let mut meta = base_meta.clone();
            meta.extra.insert("freq_min".to_string(), MetaValue::Int(f_min));
            meta.extra.insert("freq_max".to_string(), MetaValue::Int(f_max));
            let buf: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
            let band_data = Data::from_array_bytes(DType::F32, out_shape.clone(), buf, meta)
                .map_err(|e| e.to_string())?;
            out.set(name, band_data);
        }
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if (key.group.as_str(), key.name.as_str()) == ("powerband", "power_type") {
            if let Some(s) = v.as_str() {
                self.relative = s == "relative";
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert(
        "power_type".to_string(),
        Param::Str {
            value: "absolute".to_string(),
            options: Some(vec!["absolute".to_string(), "relative".to_string()]),
            refresh: None,
        },
    );
    let mut groups = ParamGroups::new();
    groups.insert("powerband".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let relative = param(p, "powerband", "power_type").and_then(Param::as_str) == Some("relative");
    Box::new(PowerBandEeg { relative })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
static OUTPUTS: &[OutputDecl] = &[
    OutputDecl { name: "delta", kind: SlotType::Array },
    OutputDecl { name: "theta", kind: SlotType::Array },
    OutputDecl { name: "alpha", kind: SlotType::Array },
    OutputDecl { name: "lowbeta", kind: SlotType::Array },
    OutputDecl { name: "highbeta", kind: SlotType::Array },
    OutputDecl { name: "gamma", kind: SlotType::Array },
];

inventory::submit! {
    NodeManifest {
        type_name: "PowerBandEEG",
        category: "analysis",
        doc: "Split a PSD into the six standard EEG bands (delta/theta/alpha/low+high beta/gamma), power per band.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Channels, Coord, Data, DType, Meta, MetaValue, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Drive PowerBandEEG on a frame; returns the whole output buffer so a test can
    /// inspect each of the six band slots (each `None` when the node no-ops).
    fn run(power_type: &str, frame: Option<Data>) -> IndexMap<&'static str, Option<Data>> {
        let m = goofi_node::find("PowerBandEEG").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("powerband", "power_type"), &Param::str_free(power_type))
            .unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", frame);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        outbuf
    }

    /// A frame with the given shape, f32 values and channel coords.
    fn frame(shape: Vec<usize>, data: &[f32], channels: BTreeMap<usize, Arc<Vec<Coord>>>) -> Data {
        let buf: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let meta = Meta { channels: Channels(channels), ..Default::default() };
        Data::from_array_bytes(DType::F32, shape, buf, meta).unwrap()
    }

    fn nums(v: &[f64]) -> Arc<Vec<Coord>> {
        Arc::new(v.iter().map(|&f| Coord::Num(f)).collect())
    }

    /// Shape + values of one output band slot (panics if that band did not emit).
    fn band(buf: &IndexMap<&'static str, Option<Data>>, name: &str) -> (Vec<usize>, Vec<f32>) {
        let d = buf.get(name).unwrap().as_ref().expect("band emitted");
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    fn all_none(buf: &IndexMap<&'static str, Option<Data>>) -> bool {
        buf.values().all(|v| v.is_none())
    }

    #[test]
    fn splits_1d_psd_into_bands() {
        // freq [0..8] Hz on dim0 (note the DC 0-Hz bin), power[i] = i+1.
        // delta [1,3] -> freqs 1,2,3 -> power 2+3+4 = 9 (the 0-Hz DC bin is excluded).
        // theta [3,7] -> freqs 3,4,5,6,7 -> power 4+5+6+7+8 = 30 (3 Hz counts in BOTH).
        // alpha [7,12] -> freqs 7,8 -> power 8+9 = 17.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
        let f = frame(vec![9], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], ch);
        let buf = run("absolute", Some(f));
        assert_eq!(band(&buf, "delta"), (vec![1], vec![9.0]));
        assert_eq!(band(&buf, "theta"), (vec![1], vec![30.0]));
        assert_eq!(band(&buf, "alpha"), (vec![1], vec![17.0]));
        // Bins above 8 Hz never appear, so the higher bands sum to 0.
        assert_eq!(band(&buf, "lowbeta"), (vec![1], vec![0.0]));
        assert_eq!(band(&buf, "gamma"), (vec![1], vec![0.0]));
    }

    #[test]
    fn all_six_bands_emit_with_range_meta() {
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[1.0, 2.0, 3.0]));
        let f = frame(vec![3], &[10.0, 20.0, 30.0], ch);
        let buf = run("absolute", Some(f));
        // Every band slot must carry data and its Hz bounds; the freq axis is gone.
        for (name, fmin, fmax) in
            [("delta", 1, 3), ("theta", 3, 7), ("alpha", 7, 12), ("lowbeta", 12, 20), ("highbeta", 20, 30), ("gamma", 30, 50)]
        {
            let d = buf.get(name).unwrap().as_ref().expect("band emitted");
            assert_eq!(d.meta().extra.get("freq_min"), Some(&MetaValue::Int(fmin)), "{name} freq_min");
            assert_eq!(d.meta().extra.get("freq_max"), Some(&MetaValue::Int(fmax)), "{name} freq_max");
            assert!(d.meta().channels.0.is_empty(), "{name} freq axis dropped");
        }
    }

    #[test]
    fn relative_divides_by_total_power_1d() {
        // total = 1+..+9 = 45; delta band = 9 -> 0.2; alpha band = 17 -> 17/45.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
        let f = frame(vec![9], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], ch);
        let buf = run("relative", Some(f));
        assert!((band(&buf, "delta").1[0] - 9.0 / 45.0).abs() < 1e-6);
        assert!((band(&buf, "alpha").1[0] - 17.0 / 45.0).abs() < 1e-6);
    }

    #[test]
    fn splits_2d_psd_per_channel() {
        // shape [2,4]: freq on dim1 = [1,2,3,4], electrode labels on dim0.
        // delta [1,3] -> cols 0,1,2 -> row0 10+20+30=60, row1 1+2+3=6.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("Fp1".into()), Coord::Str("Fp2".into())]));
        ch.insert(1usize, nums(&[1.0, 2.0, 3.0, 4.0]));
        let data = [10.0f32, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0];
        let f = frame(vec![2, 4], &data, ch);
        let buf = run("absolute", Some(f));
        assert_eq!(band(&buf, "delta"), (vec![2], vec![60.0, 6.0]));
        // No bin lands in [7,12] -> per-channel zeros.
        assert_eq!(band(&buf, "alpha"), (vec![2], vec![0.0, 0.0]));
        // dim1 (freq) dropped; dim0 (electrode labels) preserved on the output axis.
        let d = buf.get("delta").unwrap().as_ref().unwrap();
        assert!(!d.meta().channels.0.contains_key(&1), "freq axis dropped");
        let kept = d.meta().channels.0.get(&0).expect("electrode labels kept");
        assert_eq!(kept[0], Coord::Str("Fp1".into()));
    }

    #[test]
    fn relative_divides_per_channel_2d() {
        // row0 total = 100, delta(cols 0,1,2) = 60 -> 0.6; row1 total = 10, delta = 6 -> 0.6.
        let mut ch = BTreeMap::new();
        ch.insert(1usize, nums(&[1.0, 2.0, 3.0, 4.0]));
        let data = [10.0f32, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0];
        let f = frame(vec![2, 4], &data, ch);
        let buf = run("relative", Some(f));
        let (_, v) = band(&buf, "delta");
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.6).abs() < 1e-6, "got {v:?}");
    }

    #[test]
    fn preserves_non_channel_meta() {
        // sfreq (and other input meta) survive the reduction; only the freq axis goes.
        let buf: Vec<u8> = [10.0f32, 20.0, 30.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[1.0, 2.0, 3.0]));
        let meta = Meta { sfreq: Some(256.0), channels: Channels(ch), ..Default::default() };
        let f = Data::from_array_bytes(DType::F32, vec![3], buf, meta).unwrap();
        let out = run("absolute", Some(f));
        let d = out.get("delta").unwrap().as_ref().unwrap();
        assert_eq!(d.meta().sfreq, Some(256.0), "sfreq preserved through the reduction");
    }

    #[test]
    fn no_emit_on_missing_input() {
        assert!(all_none(&run("absolute", None)));
    }

    #[test]
    fn no_emit_on_non_f32_dtype() {
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[1.0, 2.0, 3.0]));
        let meta = Meta { channels: Channels(ch), ..Default::default() };
        let f = Data::from_array_bytes(DType::U8, vec![3], vec![1u8, 2, 3], meta).unwrap();
        assert!(all_none(&run("absolute", Some(f))));
    }

    #[test]
    fn no_emit_without_a_freq_axis() {
        // No channel coords at all -> can't locate the frequency axis -> no-op.
        let f = frame(vec![3], &[1.0, 2.0, 3.0], BTreeMap::new());
        assert!(all_none(&run("absolute", Some(f))));
    }

    #[test]
    fn no_emit_on_non_numeric_freq_axis() {
        // A string coord list can't be a frequency axis -> no-op (Python would error).
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("a".into()), Coord::Str("b".into())]));
        let f = frame(vec![2], &[1.0, 2.0], ch);
        assert!(all_none(&run("absolute", Some(f))));
    }

    #[test]
    fn no_emit_on_2d_without_dim1_freq_axis() {
        // A 2-D PSD keys the freq axis off dim1; only dim0 present -> no-op (Python KeyError).
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[1.0, 2.0]));
        let f = frame(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ch);
        assert!(all_none(&run("absolute", Some(f))));
    }

    #[test]
    fn no_emit_on_unsupported_ndim() {
        // 3-D is outside the {1,2} the Python handles -> no-op.
        let mut ch = BTreeMap::new();
        ch.insert(1usize, nums(&[1.0, 2.0]));
        let f = frame(vec![1, 2, 1], &[1.0, 2.0], ch);
        assert!(all_none(&run("absolute", Some(f))));
    }
}
