//! TableSelect — one field out of a table, on the output that matches what the field holds.
//! `a.b.c` reaches through a table inside a table.

use goofi_core::{Data, SlotType, Value};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct TableSelect;

impl Node for TableSelect {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let key = p.str("table", "key").unwrap_or("").trim();
        if key.is_empty() {
            return Err("`key` names the field to take".to_string().into());
        }
        let mut here: &Data = d;
        for step in key.split('.') {
            let table = here.as_table()?;
            here = table.get(step).ok_or_else(|| {
                format!("no `{step}` here; this table holds {:?}", table.keys().collect::<Vec<_>>())
            })?;
        }
        let slot = match here.value() {
            Value::Array(_) => "array",
            Value::Str(_) => "string",
            Value::Table(_) => "table",
        };
        out.set(slot, here.clone());
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "table",
    name: "key",
    spec: ParamSpec::Str { default: "", options: &[], refresh: false },
    expression: None,
    doc: Some("Which field to take. `a.b.c` reaches through a table inside a table."),
}];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Table,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[
    OutputDecl { name: "array", kind: SlotType::Array },
    OutputDecl { name: "string", kind: SlotType::String },
    OutputDecl { name: "table", kind: SlotType::Table },
];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "One field out of a table, on the output that matches what the field holds.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(TableSelect, MANIFEST);
