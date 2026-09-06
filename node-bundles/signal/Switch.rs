//! Switch — one of several wires passes, chosen by number. With `index` in reference mode a
//! signal chooses the route.

use goofi_core::{Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Switch;

impl Node for Switch {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let wires = inp.get_multi("input");
        if wires.is_empty() {
            return Err("`input` needs at least one wire".to_string().into());
        }
        let at = p.i64("switch", "index").unwrap_or(0).max(0) as usize;
        let (_, chosen) = wires
            .get(at)
            .ok_or_else(|| format!("`index` is {at} and there are {} wires", wires.len()))?;
        out.set("out", chosen.clone());
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "switch",
    name: "index",
    spec: ParamSpec::Int { default: 0, min: 0, max: 63 },
    expression: None,
    doc: Some(
        "Which wire passes, counted in the order they were connected. In reference mode another \
         node's output chooses the route.",
    ),
}];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Array,
    trigger_process: true,
    multi: true,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Control],
    doc: "Let one of several wires through, chosen by number.\n\
          So a signal can pick the route.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Switch, MANIFEST);
