//! Clip — clamp every element of a float32 array to `[min, max]`. A staple
//! bio-signal preprocessing step (reject amplifier saturation / motion spikes
//! before downstream analysis). Length-preserving, so the full input meta
//! (sfreq/channels/index) carries through unchanged.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Clip {
    min: f32,
    max: f32,
}

impl Node for Clip {
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
        // A reversed [min,max] would clamp everything to a single value; guard it
        // so the output is well-defined (lo <= hi) rather than surprising.
        let (lo, hi) = if self.min <= self.max {
            (self.min, self.max)
        } else {
            (self.max, self.min)
        };
        let mut buf = Vec::with_capacity(store.as_bytes().len());
        for chunk in store.as_bytes().chunks_exact(4) {
            let a = f32::from_le_bytes(chunk.try_into().unwrap());
            buf.extend_from_slice(&a.clamp(lo, hi).to_le_bytes());
        }
        // Length-preserving: the input meta stays valid (shape unchanged).
        let data = Data::from_array_bytes(DType::F32, store.shape().to_vec(), buf, d.meta().clone())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("clip", "min") => {
                if let Some(x) = v.as_f64() {
                    self.min = x as f32;
                }
            }
            ("clip", "max") => {
                if let Some(x) = v.as_f64() {
                    self.max = x as f32;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("min".to_string(), Param::float(-1.0, -1.0e9, 1.0e9));
    g.insert("max".to_string(), Param::float(1.0, -1.0e9, 1.0e9));
    let mut groups = ParamGroups::new();
    groups.insert("clip".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let min = param(p, "clip", "min").and_then(Param::as_f64).unwrap_or(-1.0) as f32;
    let max = param(p, "clip", "max").and_then(Param::as_f64).unwrap_or(1.0) as f32;
    Box::new(Clip { min, max })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
    length_preserving: true,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Clip",
        category: "array",
        doc: "Clamp every element to [min, max].",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Data, DType, Meta, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;

    fn run(min: f32, max: f32, input: &[f32]) -> Vec<f32> {
        let m = goofi_node::find("Clip").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("clip", "min"), &Param::float(min as f64, -1e9, 1e9))
            .unwrap();
        node.on_param_changed(&ParamKey::new("clip", "max"), &Param::float(max as f64, -1e9, 1e9))
            .unwrap();
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![input.len()], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        match d.value() {
            Value::Array(s) => s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn clamps_to_range() {
        assert_eq!(run(-1.0, 1.0, &[-2.0, -0.5, 0.5, 3.0]), vec![-1.0, -0.5, 0.5, 1.0]);
    }

    #[test]
    fn reversed_bounds_are_normalized() {
        // max < min must not produce NaN/garbage; treat it as the same range.
        assert_eq!(run(1.0, -1.0, &[-2.0, 0.0, 3.0]), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn length_preserving_shape_and_meta() {
        let m = goofi_node::find("Clip").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut meta = Meta::empty();
        meta.sfreq = Some(250.0);
        let buf: Vec<u8> = [0.0f32, 5.0, -5.0, 2.0, 1.0, 9.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![2, 3], buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        match d.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[2, 3], "shape preserved"),
            _ => panic!("expected array"),
        }
        assert_eq!(d.meta().sfreq, Some(250.0), "meta carried through");
    }
}
