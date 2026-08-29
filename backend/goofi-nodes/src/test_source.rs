//! `_TestConst` — a constant float32 array source; the `_` prefix keeps it out of the palette.

use goofi_core::SlotType;
use goofi_core::{Data, Meta};
use goofi_node::{NodeManifest, OutputDecl, ParamDecl, ParamSpec, Params, SlotDecl};
use goofi_signal::{default_factory, Inputs, Node, NodeClass, NodeCtx, NodeResult, Outputs};

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
        doc: Some("The value every element of the emitted array carries."),
    },
    ParamDecl {
        group: "constant",
        name: "length",
        spec: ParamSpec::Int { default: 1, min: 1, max: 1_000_000 },
        expression: None,
        doc: Some("How many elements the emitted array has."),
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];
static INPUTS: &[SlotDecl] = &[];

inventory::submit! {
    NodeClass {
        manifest: NodeManifest {
            type_name: "_TestConst",
            category: "test",
            doc: "constant float32 array source (value+length) — hidden test/bench scaffolding.",
            inputs: INPUTS,
            outputs: OUTPUTS,
            params: PARAMS,
            producer: true,
        },
        isolation: &goofi_node::NATIVE,
        factory: default_factory::<TestConst>,
    }
}
