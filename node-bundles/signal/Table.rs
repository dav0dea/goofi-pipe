//! Table — several wires become one table. Arrays and strings take a key each; a table wire
//! merges its own entries in, and a later wire wins a key a earlier one already holds.

use goofi_core::indexmap::IndexMap;
use goofi_core::{Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Table;

impl Node for Table {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let named: Vec<&str> = p
            .str("table", "keys")
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .collect();

        let mut map: IndexMap<String, Data> = IndexMap::new();
        let leaves = inp.get_multi("arrays").iter().chain(inp.get_multi("strings"));
        for (at, (source, d)) in leaves.enumerate() {
            let key = named.get(at).map_or(source.as_str(), |k| *k);
            map.insert(key.to_string(), d.clone());
        }
        for (source, d) in inp.get_multi("tables") {
            let inner = d.as_table().map_err(|e| format!("`{source}`: {e}"))?;
            for (k, v) in inner {
                map.insert(k.clone(), v.clone());
            }
        }
        if map.is_empty() {
            return Err("a table needs at least one wire".to_string().into());
        }
        out.set("out", Data::table(map, Meta::new()));
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "table",
    name: "keys",
    spec: ParamSpec::Str { default: "", options: &[], refresh: false },
    expression: None,
    doc: Some(
        "The key for each wire, comma-separated, over the arrays and then the strings. A wire \
         with no key here takes the name of the node it comes from.",
    ),
}];
static INPUTS: &[SlotDecl] = &[
    SlotDecl { name: "arrays", kind: SlotType::Array, trigger_process: true, multi: true, required: false },
    SlotDecl { name: "strings", kind: SlotType::String, trigger_process: true, multi: true, required: false },
    SlotDecl { name: "tables", kind: SlotType::Table, trigger_process: true, multi: true, required: false },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Table }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "Several wires become one table, each under a key.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Table, MANIFEST);
