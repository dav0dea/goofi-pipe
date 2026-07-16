//! `_TestConst` — a constant-valued float32 array source (value + length via the
//! `constant` param group). The `_` prefix hides it from the palette (like the
//! engine's other `_Test*` nodes): it is **test/bench scaffolding**, not part of the
//! user-facing node library, kept here so the engine, bridge, and goofi-py test
//! suites share one simple deterministic source instead of each defining its own.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Meta, Param};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct TestConst {
    value: f32,
    length: usize,
}

impl Node for TestConst {
    fn process(&mut self, _inp: &Inputs<'_>, out: &mut Outputs<'_>, _ctx: &mut NodeCtx) -> NodeResult {
        let buf: Vec<u8> = (0..self.length).flat_map(|_| self.value.to_le_bytes()).collect();
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
    Box::new(TestConst { value, length })
}

static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];
static INPUTS: &[SlotDecl] = &[];

inventory::submit! {
    NodeManifest {
        type_name: "_TestConst",
        category: "test",
        doc: "constant float32 array source (value+length) — hidden test/bench scaffolding.",
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
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;

    #[test]
    fn emits_a_constant_array_and_reacts_to_params() {
        let m = goofi_node::find("_TestConst").expect("_TestConst registered");
        assert_eq!(m.category, "test");
        let mut params = (m.default_params)();
        params["constant"].insert("value".into(), Param::float(2.5, -1.0e9, 1.0e9));
        params["constant"].insert("length".into(), Param::int(4, 1, 1_000_000));
        let mut node = (m.make)(&params);

        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let data = outbuf.get("out").unwrap().as_ref().expect("emitted a frame");
        match data.value() {
            Value::Array(s) => {
                assert_eq!(s.dtype(), DType::F32);
                assert_eq!(s.shape(), &[4]);
                assert_eq!(f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()), 2.5);
            }
            _ => panic!("expected array"),
        }

        node.on_param_changed(&ParamKey::new("constant", "length"), &Param::int(3, 1, 10)).unwrap();
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        if let Value::Array(s) = outbuf.get("out").unwrap().as_ref().unwrap().value() {
            assert_eq!(s.shape(), &[3]);
        } else {
            panic!("expected array");
        }
    }
}
