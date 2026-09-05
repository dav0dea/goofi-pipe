//! Text — a string. With `value` in expression mode it is a full f-string over `nd()`, so a
//! number from anywhere in the patch becomes text here, with a Python format spec.

use goofi_core::{Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, Tag};

#[derive(Default)]
struct Text;

impl Node for Text {
    fn process(
        &mut self,
        _inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let value = p.str("text", "value").unwrap_or("");
        out.set("out", Data::string(value, Meta::new()));
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "text",
    name: "value",
    spec: ParamSpec::Str { default: "", options: &[], refresh: false },
    expression: None,
    doc: Some("The text to emit; in expression mode it is an f-string over the rest of the patch."),
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::String }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Generator, Tag::Text],
    doc: "A string, which an expression can build out of anything else in the patch.",
    inputs: &[],
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Text, MANIFEST);
