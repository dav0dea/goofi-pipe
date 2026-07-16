//! ConstantArray — emits a constant-valued float32 array each tick. The simplest
//! source node; used to bring up the engine and data plane end-to-end.

use goofi_core::{Data, DType, Meta, Param};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, Outputs, OutputDecl,
    ParamGroups, ParamKey, SlotDecl,
};
use goofi_core::SlotType;
use indexmap::IndexMap;

struct ConstantArray {
    value: f32,
    length: usize,
}

impl Node for ConstantArray {
    fn process(&mut self, _inp: &Inputs<'_>, out: &mut Outputs<'_>, _ctx: &mut NodeCtx) -> NodeResult {
        let mut buf = Vec::with_capacity(self.length * 4);
        for _ in 0..self.length {
            buf.extend_from_slice(&self.value.to_le_bytes());
        }
        let data = Data::from_array_bytes(DType::F32, vec![self.length], buf, Meta::empty())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("constant", "value") => {
                if let Some(x) = v.as_f64() {
                    self.value = x as f32;
                }
            }
            ("constant", "length") => {
                if let Some(x) = v.as_i64() {
                    self.length = x.max(1) as usize;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut group = IndexMap::new();
    group.insert("value".to_string(), Param::float(0.0, -1.0e9, 1.0e9));
    group.insert("length".to_string(), Param::int(1, 1, 1_000_000));
    let mut groups = ParamGroups::new();
    groups.insert("constant".to_string(), group);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let value = param(p, "constant", "value").and_then(Param::as_f64).unwrap_or(0.0) as f32;
    let length = param(p, "constant", "length")
        .and_then(Param::as_i64)
        .unwrap_or(1)
        .max(1) as usize;
    Box::new(ConstantArray { value, length })
}

static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
    length_preserving: false,
}];
static INPUTS: &[SlotDecl] = &[];

inventory::submit! {
    NodeManifest {
        type_name: "ConstantArray",
        category: "inputs",
        doc: "Emit a constant-valued float32 array each tick.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{DType, Param, Value};
    use goofi_node::{param, Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;

    #[test]
    fn constant_array_emits_expected_array() {
        let m = goofi_node::find("ConstantArray").expect("ConstantArray registered in catalog");
        assert_eq!(m.category, "inputs");
        assert_eq!(m.outputs.len(), 1);

        // Override defaults: value 2.5, length 4.
        let mut params = (m.default_params)();
        params["constant"].insert("value".into(), Param::float(2.5, -1.0e9, 1.0e9));
        params["constant"].insert("length".into(), Param::int(4, 1, 1_000_000));

        let mut node = (m.make)(&params);

        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }

        let data = outbuf.get("out").unwrap().as_ref().expect("emitted a frame");
        match data.value() {
            Value::Array(store) => {
                assert_eq!(store.dtype(), DType::F32);
                assert_eq!(store.shape(), &[4]);
                let bytes = store.as_bytes();
                let first = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
                assert_eq!(first, 2.5);
                assert_eq!(bytes.len(), 16);
            }
            _ => panic!("expected an array output"),
        }
        let _ = param(&params, "constant", "value");
    }

    #[test]
    fn param_change_updates_output_length() {
        let m = goofi_node::find("ConstantArray").unwrap();
        let params = (m.default_params)();
        let mut node = (m.make)(&params);
        node.on_param_changed(&ParamKey::new("constant", "length"), &Param::int(3, 1, 10))
            .unwrap();

        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let data = outbuf.get("out").unwrap().as_ref().unwrap();
        if let Value::Array(store) = data.value() {
            assert_eq!(store.shape(), &[3]);
        } else {
            panic!("expected array");
        }
    }
}
