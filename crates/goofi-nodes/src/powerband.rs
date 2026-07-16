//! PowerBand — total power inside a [f_min, f_max] frequency band of a PSD.
//! Reads the frequency axis from `meta["channels"]` (dim1 for a 2-D
//! [channels, freqs] PSD, else dim0 for a 1-D PSD), sums the bins that fall in
//! the band along the last axis, and (for `power_type = "relative"`) divides by
//! the full-row power. The summed-out frequency axis is dropped from the output
//! meta; every other meta field is preserved. Ported from
//! `nodes/analysis/powerband.py`.

use goofi_core::SlotType;
use goofi_core::{Coord, Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct PowerBand {
    f_min: f64,
    f_max: f64,
    relative: bool,
}

impl Node for PowerBand {
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
        // Only 1-D PSDs and 2-D [channels, freqs] PSDs are defined (like the Python port).
        if ndim != 1 && ndim != 2 {
            return Ok(());
        }

        // The frequency axis lives in channels: dim1 for a 2-D [C, F] PSD, else
        // dim0 (a 1-D PSD stores it there). Faithful to the Python key-presence check.
        let freq_dim = if d.meta().channels.0.contains_key(&1) { 1usize } else { 0usize };
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
        // Treat a 1-D PSD as a single row so both cases share one loop; a 1-D result
        // is a scalar that Data construction promotes back to a length-1 vector.
        let rows = if ndim == 1 { 1 } else { shape[0] };

        let data: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Bins whose frequency falls inside the (inclusive) band.
        let selected: Vec<usize> = (0..f)
            .filter(|&i| freq[i] >= self.f_min && freq[i] <= self.f_max)
            .collect();

        let mut result = vec![0f32; rows];
        for (row, slot) in result.iter_mut().enumerate() {
            let base = row * f;
            let band: f32 = selected.iter().map(|&i| data[base + i]).sum();
            *slot = if self.relative {
                // relative = band power / full-row power (Python np.sum(data, axis=-1)).
                let total: f32 = data[base..base + f].iter().sum();
                band / total
            } else {
                band
            };
        }

        let out_shape: Vec<usize> = if ndim == 1 { vec![1] } else { vec![rows] };
        // Preserve the input meta but drop the now-summed-out frequency axis's coords
        // (its dim is the last one, so lower axes keep their index — no shift needed).
        let mut meta = d.meta().clone();
        meta.channels.0.remove(&freq_dim);
        let buf: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data =
            Data::from_array_bytes(DType::F32, out_shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("power", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("powerband", "f_min") => {
                if let Some(x) = v.as_f64() {
                    self.f_min = x;
                }
            }
            ("powerband", "f_max") => {
                if let Some(x) = v.as_f64() {
                    self.f_max = x;
                }
            }
            ("powerband", "power_type") => {
                if let Some(s) = v.as_str() {
                    self.relative = s == "relative";
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("f_min".to_string(), Param::float(1.0, 0.01, 9999.0));
    g.insert("f_max".to_string(), Param::float(60.0, 1.0, 10000.0));
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
    let f_min = param(p, "powerband", "f_min").and_then(Param::as_f64).unwrap_or(1.0);
    let f_max = param(p, "powerband", "f_max").and_then(Param::as_f64).unwrap_or(60.0);
    let relative = param(p, "powerband", "power_type").and_then(Param::as_str) == Some("relative");
    Box::new(PowerBand { f_min, f_max, relative })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "power",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "PowerBand",
        category: "analysis",
        doc: "Sum PSD power in a [f_min, f_max] band (absolute or relative), per channel.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Channels, Coord, Data, DType, Meta, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Drive PowerBand on a pre-built frame; returns the emitted `power` Data (or
    /// `None` when the node no-ops).
    fn run_frame(f_min: f64, f_max: f64, power_type: &str, frame: Option<Data>) -> Option<Data> {
        let m = goofi_node::find("PowerBand").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("powerband", "f_min"), &Param::float(f_min, 0.01, 9999.0))
            .unwrap();
        node.on_param_changed(&ParamKey::new("powerband", "f_max"), &Param::float(f_max, 1.0, 10000.0))
            .unwrap();
        node.on_param_changed(&ParamKey::new("powerband", "power_type"), &Param::str_free(power_type))
            .unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", frame);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        outbuf.get("power").unwrap().as_ref().cloned()
    }

    /// A frame with the given dtype, shape, values and channel coords.
    fn frame(
        dtype: DType,
        shape: Vec<usize>,
        data: &[f32],
        channels: BTreeMap<usize, Arc<Vec<Coord>>>,
    ) -> Data {
        let buf: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let meta = Meta { channels: Channels(channels), ..Default::default() };
        Data::from_array_bytes(dtype, shape, buf, meta).unwrap()
    }

    fn nums(v: &[f64]) -> Arc<Vec<Coord>> {
        Arc::new(v.iter().map(|&f| Coord::Num(f)).collect())
    }

    fn vals(d: &Data) -> (Vec<usize>, Vec<f32>) {
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn sums_band_of_1d_psd() {
        // freq [0,1,2,3,4] Hz on dim0, power [10..50]; band [1,3] -> bins 1,2,3 -> 90.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[0.0, 1.0, 2.0, 3.0, 4.0]));
        let f = frame(DType::F32, vec![5], &[10.0, 20.0, 30.0, 40.0, 50.0], ch);
        let d = run_frame(1.0, 3.0, "absolute", Some(f)).expect("emits");
        let (shape, v) = vals(&d);
        assert_eq!(shape, vec![1], "1-D PSD collapses to a length-1 scalar");
        assert_eq!(v, vec![90.0]);
        // The freq axis was summed out -> its dim0 coord list is dropped.
        assert!(d.meta().channels.0.is_empty(), "frequency channel dropped");
    }

    #[test]
    fn relative_divides_by_total_power_1d() {
        // Same band (90) over total 150 -> 0.6.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[0.0, 1.0, 2.0, 3.0, 4.0]));
        let f = frame(DType::F32, vec![5], &[10.0, 20.0, 30.0, 40.0, 50.0], ch);
        let d = run_frame(1.0, 3.0, "relative", Some(f)).expect("emits");
        let (shape, v) = vals(&d);
        assert_eq!(shape, vec![1]);
        assert!((v[0] - 0.6).abs() < 1e-6, "got {}", v[0]);
    }

    #[test]
    fn sums_band_per_channel_of_2d_psd() {
        // shape [2,4]: freq on dim1, electrode labels on dim0. Band [1,2] -> cols 1,2.
        // ch0 = 2+3 = 5; ch1 = 20+30 = 50. Output is one scalar per channel.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("Fp1".into()), Coord::Str("Fp2".into())]));
        ch.insert(1usize, nums(&[0.0, 1.0, 2.0, 3.0]));
        let data = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let f = frame(DType::F32, vec![2, 4], &data, ch);
        let d = run_frame(1.0, 2.0, "absolute", Some(f)).expect("emits");
        let (shape, v) = vals(&d);
        assert_eq!(shape, vec![2]);
        assert_eq!(v, vec![5.0, 50.0]);
        // dim1 (freq) dropped; dim0 (electrode labels) preserved on the output axis.
        assert!(!d.meta().channels.0.contains_key(&1), "freq axis dropped");
        let kept = d.meta().channels.0.get(&0).expect("electrode labels kept");
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], Coord::Str("Fp1".into()));
    }

    #[test]
    fn relative_divides_per_channel_2d() {
        // ch0 band 5 / total 10 = 0.5; ch1 band 50 / total 100 = 0.5.
        let mut ch = BTreeMap::new();
        ch.insert(1usize, nums(&[0.0, 1.0, 2.0, 3.0]));
        let data = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let f = frame(DType::F32, vec![2, 4], &data, ch);
        let d = run_frame(1.0, 2.0, "relative", Some(f)).expect("emits");
        let (_, v) = vals(&d);
        assert!((v[0] - 0.5).abs() < 1e-6 && (v[1] - 0.5).abs() < 1e-6, "got {v:?}");
    }

    #[test]
    fn empty_band_sums_to_zero() {
        // No bin lies in [10,20] -> every channel sums to 0 (numpy-empty-sum semantics).
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[0.0, 1.0, 2.0, 3.0, 4.0]));
        let f = frame(DType::F32, vec![5], &[10.0, 20.0, 30.0, 40.0, 50.0], ch);
        let d = run_frame(10.0, 20.0, "absolute", Some(f)).expect("emits");
        assert_eq!(vals(&d).1, vec![0.0]);
    }

    #[test]
    fn preserves_non_channel_meta() {
        // deepcopy semantics: sfreq (and other fields) survive; only the freq axis goes.
        let buf: Vec<u8> = [10.0f32, 20.0, 30.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[0.0, 1.0, 2.0]));
        let meta = Meta { sfreq: Some(256.0), channels: Channels(ch), ..Default::default() };
        let f = Data::from_array_bytes(DType::F32, vec![3], buf, meta).unwrap();
        let d = run_frame(0.0, 5.0, "absolute", Some(f)).expect("emits");
        assert_eq!(d.meta().sfreq, Some(256.0), "sfreq preserved through the reduction");
    }

    #[test]
    fn no_emit_on_missing_input() {
        assert!(run_frame(1.0, 3.0, "absolute", None).is_none());
    }

    #[test]
    fn no_emit_on_non_f32_dtype() {
        // A u8 array carries the right shape but the wrong dtype -> silent no-op.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[0.0, 1.0, 2.0]));
        let meta = Meta { channels: Channels(ch), ..Default::default() };
        let f = Data::from_array_bytes(DType::U8, vec![3], vec![1u8, 2, 3], meta).unwrap();
        assert!(run_frame(0.0, 5.0, "absolute", Some(f)).is_none());
    }

    #[test]
    fn no_emit_on_non_numeric_freq_axis() {
        // A string coord list can't be a frequency axis -> no-op (Python would error).
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("a".into()), Coord::Str("b".into())]));
        let f = frame(DType::F32, vec![2], &[1.0, 2.0], ch);
        assert!(run_frame(0.0, 5.0, "absolute", Some(f)).is_none());
    }

    #[test]
    fn no_emit_when_freq_len_mismatches_last_axis() {
        // 2-D [2,3] with a freq axis wrongly on dim0 (len 2 != last-axis len 3) -> no-op.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, nums(&[0.0, 1.0]));
        let f = frame(DType::F32, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ch);
        assert!(run_frame(0.0, 5.0, "absolute", Some(f)).is_none());
    }

    #[test]
    fn no_emit_without_a_freq_axis() {
        // No channel coords at all -> can't locate the frequency axis -> no-op.
        let f = frame(DType::F32, vec![3], &[1.0, 2.0, 3.0], BTreeMap::new());
        assert!(run_frame(0.0, 5.0, "absolute", Some(f)).is_none());
    }
}
