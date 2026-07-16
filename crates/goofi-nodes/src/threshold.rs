//! Threshold — binarize a float32 array against a threshold (x >= t -> 1, else 0).
//! Length-preserving.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Threshold {
    threshold: f32,
}

impl Node for Threshold {
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
        let buf: Vec<u8> = store
            .as_bytes()
            .chunks_exact(4)
            .flat_map(|c| {
                let x = f32::from_le_bytes(c.try_into().unwrap());
                let y = if x >= self.threshold { 1.0f32 } else { 0.0f32 };
                y.to_le_bytes()
            })
            .collect();
        let data = Data::from_array_bytes(DType::F32, store.shape().to_vec(), buf, d.meta().clone())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if key.group == "threshold" && key.name == "threshold" {
            if let Some(x) = v.as_f64() {
                self.threshold = x as f32;
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("threshold".to_string(), Param::float(0.5, -1.0e9, 1.0e9));
    let mut groups = ParamGroups::new();
    groups.insert("threshold".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let threshold = param(p, "threshold", "threshold")
        .and_then(Param::as_f64)
        .unwrap_or(0.5) as f32;
    Box::new(Threshold { threshold })
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
        type_name: "Threshold",
        category: "signal",
        doc: "Binarize a float32 array against a threshold (x >= t -> 1 else 0).",
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

    #[test]
    fn threshold_binarizes() {
        let m = goofi_node::find("Threshold").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("threshold", "threshold"), &Param::float(2.0, -1e9, 1e9))
            .unwrap();
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![3], buf, Meta::empty()).unwrap();
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
        if let Value::Array(s) = d.value() {
            let v: Vec<f32> = s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(v, vec![0.0, 1.0, 1.0]); // 1<2 ->0, 2>=2 ->1, 3>=2 ->1
        } else {
            panic!("expected array");
        }
    }
}
