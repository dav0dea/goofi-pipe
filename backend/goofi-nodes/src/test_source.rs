//! `_TestConst` — a constant-valued float32 array source (value + length via the
//! `constant` param group). The `_` prefix hides it from the palette (like the
//! engine's other `_Test*` nodes): it is **test/bench scaffolding**, not part of the
//! user-facing node library, kept here so the engine, bridge, and goofi-python test
//! suites share one simple deterministic source instead of each defining its own.
//!
//! Both params are cold — read live from `p` — so the node is stateless (no fields,
//! no `on_param_changed`).

use goofi_core::SlotType;
use goofi_core::{Data, Meta};
use goofi_node::{
    default_factory, ExprMode, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult,
    OutputDecl, Outputs, ParamDecl, ParamSpec, Params, SlotDecl,
};

#[derive(Default)]
struct TestConst;

impl Node for TestConst {
    fn process(&mut self, _inp: &Inputs<'_>, out: &mut Outputs<'_>, _ctx: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        let value = p.f64("constant", "value").unwrap_or(0.0) as f32;
        let length = p.i64("constant", "length").unwrap_or(1).max(1) as usize;
        let buf: Vec<u8> = (0..length).flat_map(|_| value.to_le_bytes()).collect();
        let data = Data::array_f32(vec![length], buf, Meta::empty())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "constant",
        name: "value",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        expression_mode: ExprMode::Off,
        trigger: false,
        doc: Some("The value every element of the emitted array carries."),
    },
    ParamDecl {
        group: "constant",
        name: "length",
        spec: ParamSpec::Int { default: 1, min: 1, max: 1_000_000 },
        expression: None,
        expression_mode: ExprMode::Off,
        trigger: false,
        doc: Some("How many elements the emitted array has."),
    },
];
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
        params: PARAMS,
        isolation: Isolation::InProcess,
        producer: false,
        factory: default_factory::<TestConst>,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamGroups, Params};
    use indexmap::IndexMap;

    fn emit(node: &mut Box<dyn goofi_node::Node>, m: &goofi_node::NodeManifest, params: &ParamGroups) -> Value {
        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new(), &Params::new(params)).unwrap();
        outbuf.get("out").unwrap().as_ref().expect("emitted a frame").value().clone()
    }

    #[test]
    fn emits_a_constant_array_and_reacts_to_params() {
        let m = goofi_node::find("_TestConst").expect("_TestConst registered");
        assert_eq!(m.category, "test");
        let mut params = m.default_params();
        params["constant"].insert("value".into(), Param::float(2.5, -1.0e9, 1.0e9));
        params["constant"].insert("length".into(), Param::int(4, 1, 1_000_000));
        let mut node = (m.factory)();

        match emit(&mut node, m, &params) {
            Value::Array(s) => {
                assert_eq!(s.shape(), &[4]);
                assert_eq!(f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()), 2.5);
            }
            _ => panic!("expected array"),
        }

        // `length` is read live — changing the param map changes the next emit.
        params["constant"].insert("length".into(), Param::int(3, 1, 10));
        match emit(&mut node, m, &params) {
            Value::Array(s) => assert_eq!(s.shape(), &[3]),
            _ => panic!("expected array"),
        }
    }
}
