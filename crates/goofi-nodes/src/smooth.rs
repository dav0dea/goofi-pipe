//! Smooth — 1-D Gaussian smoothing of a float32 array along one axis (noise
//! reduction). Length-preserving, so the full input meta carries through. Ported
//! from the Python `nodes/signal/smooth.py` (scipy `gaussian_filter1d`, reflect
//! boundary); applied per line via the same `[outer, axis, inner]` view as Reduce.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Smooth {
    sigma: f32,
    axis: i64,
}

impl Node for Smooth {
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
        if ndim == 0 {
            return Ok(());
        }
        let axis = if self.axis < 0 { self.axis + ndim as i64 } else { self.axis };
        if axis < 0 || axis as usize >= ndim {
            return Ok(());
        }
        let a = axis as usize;

        let mut data: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // Smooth each 1-D line along the axis in place ([outer, axis, inner] view).
        let axis_len = shape[a];
        let outer: usize = shape[..a].iter().product();
        let inner: usize = shape[a + 1..].iter().product();
        let mut line = vec![0f32; axis_len];
        for o in 0..outer {
            for i in 0..inner {
                for k in 0..axis_len {
                    line[k] = data[o * axis_len * inner + k * inner + i];
                }
                let sm = goofi_dsp::gaussian_smooth1d(&line, self.sigma);
                for (k, &v) in sm.iter().enumerate() {
                    data[o * axis_len * inner + k * inner + i] = v;
                }
            }
        }

        let buf: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Length-preserving: shape + input meta stay valid.
        let out_data = Data::from_array_bytes(DType::F32, shape.to_vec(), buf, d.meta().clone())
            .map_err(|e| e.to_string())?;
        out.set("out", out_data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("smooth", "sigma") => {
                if let Some(x) = v.as_f64() {
                    self.sigma = x as f32;
                }
            }
            ("smooth", "axis") => {
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
    g.insert("sigma".to_string(), Param::float(2.0, 0.1, 20.0));
    g.insert("axis".to_string(), Param::int(-1, -8, 8));
    let mut groups = ParamGroups::new();
    groups.insert("smooth".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let sigma = param(p, "smooth", "sigma").and_then(Param::as_f64).unwrap_or(2.0) as f32;
    let axis = param(p, "smooth", "axis").and_then(Param::as_i64).unwrap_or(-1);
    Box::new(Smooth { sigma, axis })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Smooth",
        category: "signal",
        doc: "Gaussian 1-D smoothing along an axis (noise reduction).",
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

    fn smooth(sigma: f64, axis: i64, shape: Vec<usize>, input: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let m = goofi_node::find("Smooth").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("smooth", "sigma"), &goofi_core::Param::float(sigma, 0.1, 20.0)).unwrap();
        node.on_param_changed(&ParamKey::new("smooth", "axis"), &goofi_core::Param::int(axis, -8, 8)).unwrap();
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, shape, buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        match outbuf.get("out").unwrap().as_ref().unwrap().value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn smooths_1d_reducing_variance_length_preserved() {
        let sig: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let (shape, out) = smooth(2.0, 0, vec![16], &sig);
        assert_eq!(shape, vec![16], "length preserved");
        let ptp = |v: &[f32]| v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min);
        assert!(ptp(&out) < ptp(&sig), "smoothing reduces peak-to-peak");
    }

    #[test]
    fn smooths_each_row_independently_along_last_axis() {
        // shape [2,4]: row0 constant (stays), row1 alternating (smooths). axis -1.
        let v = [5.0f32, 5.0, 5.0, 5.0, 1.0, -1.0, 1.0, -1.0];
        let (shape, out) = smooth(1.5, -1, vec![2, 4], &v);
        assert_eq!(shape, vec![2, 4]);
        for x in &out[0..4] {
            assert!((x - 5.0).abs() < 1e-3, "constant row unchanged, got {x}");
        }
        // Row 1 is pulled toward its mean (0): peak magnitude drops below 1.
        assert!(out[4..8].iter().all(|x| x.abs() < 1.0), "alternating row smoothed: {:?}", &out[4..8]);
    }

    #[test]
    fn preserves_meta_and_channels() {
        let m = goofi_node::find("Smooth").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut meta = Meta::empty();
        meta.sfreq = Some(250.0);
        meta.channels.0.insert(0, Arc::new(vec![Coord::Str("Fz".into()), Coord::Str("Cz".into())]));
        let buf: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![2, 4], buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        assert_eq!(d.meta().sfreq, Some(250.0));
        assert_eq!(d.meta().channels.0.get(&0).unwrap().len(), 2, "dim0 channels preserved");
    }
}
