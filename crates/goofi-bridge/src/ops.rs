//! The op registry — one row per `/control` op, and the single place the op SET is declared.
//!
//! `dispatch` is a string-keyed match, which has no way to say "this arm is missing" and has bitten
//! this project twice. The registry closes both directions of that hazard:
//!
//! * an op **not** in this table is refused before the match is reached, so a dispatch arm without a
//!   row is unreachable rather than a second, invisible definition of the op set;
//! * a row without an arm answers `unknown op`, which [`tests::every_registry_op_has_a_dispatch_arm`]
//!   catches;
//! * the frontend's `OpName` union is GENERATED from these names, so a call site cannot name an op
//!   that does not exist — a compile error on one side, a refusal on the other;
//! * `writes` replaces the parallel `read_only` list dispatch used to carry, so classifying a new op
//!   is part of declaring it rather than a second edit somewhere else;
//! * Task 4's MCP tool list is generated from the `Surface::Mcp` rows.

/// Where an op is offered.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Surface {
    /// Mirrored as an MCP tool for an agent.
    Mcp,
    /// `/control` only. An agent calling `load` would replace the patch it is working in — and
    /// itself, once the harness lives inside that patch's workspace; `save`/`serialize`/`list_dir`
    /// are the human file-browser's half of the same door. `set_layout` is here because the flat
    /// layout ops supersede it (plan Task 2) and it should never become an agent tool on the way.
    ControlOnly,
}

/// One op's contract.
pub struct Op {
    /// MCP-safe by invariant: `[a-z0-9_]+`, short enough that `mcp__goofi__<name>` fits in 64
    /// characters. A longer or dotted name makes a model provider reject the WHOLE tool list.
    pub name: &'static str,
    pub surface: Surface,
    /// Whether a successful call may have changed the graph — the gate on the post-dispatch CRDT
    /// re-mirror (and, with the exceptions dispatch names, on the unsaved-changes flag).
    pub writes: bool,
    /// The params schema: space-separated `name:type`, `!` marking a required one. Types are
    /// `uid`, `string`, `float`, `int`, `bool`, `float2` (an `[x, y]` pair), `json` (an opaque
    /// value the engine round-trips), and the `[]` suffix for a list.
    pub args: &'static str,
    pub doc: &'static str,
    /// The result schema, as the shape a caller gets back.
    pub result: &'static str,
}

impl Op {
    /// The params schema, parsed: `(name, type, required)` per argument.
    pub fn args(&self) -> impl Iterator<Item = (&'static str, &'static str, bool)> {
        self.args.split_whitespace().filter_map(|a| {
            let (name, ty) = a.split_once(':')?;
            Some((name, ty.trim_end_matches('!'), ty.ends_with('!')))
        })
    }
}

/// The prefix a model provider gives every tool of this server. Its length is the budget the op
/// names are checked against.
pub const MCP_PREFIX: &str = "mcp__goofi__";

use Surface::{ControlOnly, Mcp};

pub static REGISTRY: &[Op] = &[
    Op { name: "list_dir", surface: ControlOnly, writes: false, args: "path:string",
         doc: "List a directory on the goofi host — the save/load browser's read.",
         result: "{path, parent, entries: [{name, dir}], roots}" },
    Op { name: "list_nodes", surface: Mcp, writes: false, args: "",
         doc: "The node palette: every registered type with its slots, params, docs and availability.",
         result: "{types: [{type, category, doc, input_slots, output_slots, params, available}]}" },
    Op { name: "rescan_nodes", surface: Mcp, writes: true, args: "",
         doc: "Re-read the shipped and patch node directories; live instances of a changed type restart onto the new code. Call after writing a node file.",
         result: "{added: [type], changed: [type], removed: [type]}" },
    Op { name: "add_node", surface: Mcp, writes: true,
         args: "type:string! pos:float2 name:string inst_id:uid member_uid:uid params:json",
         doc: "Create a node of `type`. `inst_id` births it inside that sub-patch; absent = root. `params` is `{group: {name: value}}` applied at birth.",
         result: "the new node's uid" },
    Op { name: "remove_node", surface: Mcp, writes: true, args: "node:uid!",
         doc: "Delete a node, a sub-patch member, or a whole collapsed sub-patch instance.",
         result: "{ok: true}" },
    Op { name: "restart_node", surface: Mcp, writes: true, args: "node:uid!",
         doc: "Respawn a node in place, keeping its uid, name, params, links and scope. Recovery, not an edit — `setup()` runs again.",
         result: "{ok: true}" },
    Op { name: "add_link", surface: Mcp, writes: true,
         args: "node_out:uid! slot_out:string! node_in:uid! slot_in:string!",
         doc: "Wire an output slot to an input slot. Either end may name a sub-patch boundary port. Refuses a dtype mismatch, naming both ends.",
         result: "{ok: true}" },
    Op { name: "remove_link", surface: Mcp, writes: true,
         args: "node_out:uid! slot_out:string! node_in:uid! slot_in:string!",
         doc: "Remove one wire, addressed by both of its endpoints.",
         result: "{ok: true}" },
    Op { name: "refresh_param", surface: Mcp, writes: true, args: "node:uid! group:string! name:string!",
         doc: "Re-enumerate a refreshable string param's options (a device or stream picker).",
         result: "{options: [string] | null}" },
    Op { name: "update_param", surface: Mcp, writes: true,
         args: "node:uid! group:string! name:string! value:json!",
         doc: "Set one param's literal value; coerced to the param's declared type and range.",
         result: "{ok: true}" },
    Op { name: "set_expression", surface: Mcp, writes: true,
         args: "node:uid! group:string! name:string! expression:string enabled:bool triggers:bool",
         doc: "Bind a param to an expression — `nd('name').sfreq`, `globals.x`, `t`. An empty expression clears the binding.",
         result: "{error: string | null} — the compile/binding error, or null when it took" },
    Op { name: "set_node_pos", surface: Mcp, writes: true, args: "node:uid! pos:float2!",
         doc: "Move a node on the canvas.",
         result: "{ok: true}" },
    Op { name: "set_layout", surface: ControlOnly, writes: true, args: "layout:json! intent:string",
         doc: "Store the editor's panel arrangement verbatim. `intent:\"navigation\"` means the user only looked around, so it must not dirty the patch.",
         result: "{ok: true}" },
    Op { name: "set_node_viewers", surface: Mcp, writes: true, args: "node:uid! viewers:json!",
         doc: "Store a node's per-slot viewer view-state (chosen kind, settings, collapse).",
         result: "{ok: true}" },
    Op { name: "rename_node", surface: Mcp, writes: true, args: "node:uid! name:string!",
         doc: "Rename a node. Display names are unique patch-wide, and `nd()` references are rewritten to follow.",
         result: "{ok: true}" },
    Op { name: "add_global", surface: Mcp, writes: true, args: "name:string! value:json! type:string!",
         doc: "Create a patch global. `type` is one of float/int/bool/string. Refuses an existing name.",
         result: "{ok: true}" },
    Op { name: "set_global", surface: Mcp, writes: true, args: "name:string! value:json! type:string!",
         doc: "Set an existing global's value. Refuses an unknown name — use add_global to create one.",
         result: "{ok: true}" },
    Op { name: "remove_global", surface: Mcp, writes: true, args: "name:string!",
         doc: "Delete a user global. System globals cannot be removed.",
         result: "{ok: true}" },
    Op { name: "rename_global", surface: Mcp, writes: true, args: "old:string! new:string!",
         doc: "Rename a user global, as one undo step.",
         result: "{ok: true}" },
    Op { name: "group_nodes", surface: Mcp, writes: true, args: "members:uid[]! pos:float2",
         doc: "Collapse nodes into a new sub-patch, returning its instance uid.",
         result: "{inst_id: uid}" },
    Op { name: "expand_instance", surface: Mcp, writes: true, args: "inst_id:uid!",
         doc: "Dissolve a sub-patch, returning its members to the parent scope.",
         result: "{ok: true}" },
    Op { name: "add_boundary", surface: Mcp, writes: true,
         args: "inst_id:uid! dir:string! dtype:string pos:float2",
         doc: "Add a boundary port to a sub-patch. `dir` is \"in\" or \"out\"; `dtype` one of ARRAY/STRING/TABLE.",
         result: "{bnd_id: string}" },
    Op { name: "wire_boundary", surface: Mcp, writes: true,
         args: "inst_id:uid! bnd_id:string! inner_node:uid inner_slot:string",
         doc: "Point a boundary port at an inner member's slot. Naming neither inner half unwires it.",
         result: "{ok: true}" },
    Op { name: "remove_boundary", surface: Mcp, writes: true, args: "inst_id:uid! bnd_id:string!",
         doc: "Remove a sub-patch boundary port.",
         result: "{ok: true}" },
    Op { name: "rename_boundary", surface: Mcp, writes: true, args: "inst_id:uid! bnd_id:string! name:string!",
         doc: "Relabel a boundary port. Its id is stable, so external wires survive.",
         result: "{ok: true}" },
    Op { name: "set_boundary_pos", surface: Mcp, writes: true, args: "inst_id:uid! bnd_id:string! pos:float2!",
         doc: "Move a boundary port's pill inside the entered sub-patch.",
         result: "{ok: true}" },
    Op { name: "serialize", surface: ControlOnly, writes: false, args: "",
         doc: "The patch manifest as YAML — a debug read, not a save path.",
         result: "{yaml: string}" },
    Op { name: "open_workspace", surface: Mcp, writes: false, args: "",
         doc: "Where this patch's workspace files live right now. The mount is a per-run temp directory, so asking is the only way to find it.",
         result: "{path: string}" },
    Op { name: "save", surface: ControlOnly, writes: false, args: "path:string!",
         doc: "Pack the patch and its workspace to a `.gfi` at `path`, and remember it as the patch's home.",
         result: "{path: string}" },
    Op { name: "load_text", surface: ControlOnly, writes: true, args: "content:string!",
         doc: "Replace the open patch from an inline YAML manifest. Carries no workspace.",
         result: "{ok: true}" },
    Op { name: "load", surface: ControlOnly, writes: true, args: "path:string!",
         doc: "Replace the open patch with the `.gfi` at `path`, workspace and all.",
         result: "{ok: true}" },
    Op { name: "new", surface: Mcp, writes: true, args: "",
         doc: "Replace the open patch with an empty one. Unsaved work is lost.",
         result: "{ok: true}" },
    Op { name: "undo", surface: Mcp, writes: true, args: "",
         doc: "Undo this session's last graph command. Each caller's session has its own stack.",
         result: "{changed: [uid], can_undo: bool, can_redo: bool}" },
    Op { name: "redo", surface: Mcp, writes: true, args: "",
         doc: "Redo this session's last undone graph command.",
         result: "{changed: [uid], can_undo: bool, can_redo: bool}" },
    Op { name: "inspect_patch", surface: Mcp, writes: false, args: "scope:uid",
         doc: "Read one scope as a mermaid flowchart — nodes, sub-patches, boundary ports and wires — plus the whole patch's standing errors and how long each has stood. No arg = the root scope.",
         result: "{text: string}" },
    Op { name: "inspect_node", surface: Mcp, writes: false,
         args: "node:uid! slot:string params:bool meta:bool error:bool",
         doc: "Read one node: its params (values, ranges, expression bindings), the health of each output slot's latest frame (shape, finite count, range), that frame's meta, and its error. `slot` narrows to one output.",
         result: "{text: string}" },
    Op { name: "get_patch", surface: Mcp, writes: false, args: "",
         doc: "Where the open patch lives, where its workspace is, and whether it differs from disk.",
         result: "{save_path: string | null, workspace: string, dirty: bool}" },
    Op { name: "list_globals", surface: Mcp, writes: false, args: "",
         doc: "Every patch global — what an expression can read and set_global can write.",
         result: "{globals: [{name, type, value, system: bool}]}" },
    Op { name: "read_node_source", surface: Mcp, writes: false, args: "type:string!",
         doc: "A node type's source and provenance. A native type has no source text — copy a Python node into the patch workspace to modify it.",
         result: "{type, language, tier, source: string | null, path: string | null, provenance, doc, inputs, outputs}" },
];

/// The row for `name`, if the op exists.
pub fn find(name: &str) -> Option<&'static Op> {
    REGISTRY.iter().find(|o| o.name == name)
}

/// The frontend's `OpName` union, generated from the registry. Checked into the tree (see
/// [`tests::the_frontend_op_union_is_generated_from_the_registry`]) rather than emitted by a build
/// script: the artifact is small, reviewable in a diff, and needs no new build machinery.
pub fn typescript() -> String {
    let names: Vec<String> = REGISTRY.iter().map(|o| format!("\t| '{}'", o.name)).collect();
    format!(
        "// GENERATED from crates/goofi-bridge/src/ops.rs — do not edit by hand.\n\
         // The manager's op registry is the only place an op name is declared: naming one that is\n\
         // not in it is a type error here and an `unknown op` refusal there. Regenerate by running\n\
         // `cargo test -p goofi-bridge`, which rewrites this file when it drifts.\n\
         export type OpName =\n{};\n",
        names.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The argument types the schema DSL admits. A type outside this set is a typo, which
    /// [`every_row_declares_a_well_formed_schema`] refuses.
    const ARG_TYPES: &[&str] =
        &["uid", "string", "float", "int", "bool", "float2", "json", "uid[]", "string[]"];

    /// A name outside `[a-z0-9_]+`, or one long enough to push `mcp__goofi__<name>` past 64
    /// characters, makes Claude and OpenAI reject the ENTIRE tool list with a 400 — every tool,
    /// not just the offending one. So this is a build-stopping invariant, not a lint.
    #[test]
    fn every_op_name_fits_the_mcp_budget() {
        for op in REGISTRY {
            assert!(
                op.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                    && !op.name.is_empty(),
                "`{}` is not [a-z0-9_]+",
                op.name
            );
            assert!(
                MCP_PREFIX.len() + op.name.len() <= 64,
                "`{}{}` is {} characters — over the 64 a tool name may have",
                MCP_PREFIX,
                op.name,
                MCP_PREFIX.len() + op.name.len(),
            );
        }
    }

    /// The params schema is a string, so a typo in it is only a fact at read time. Parse every
    /// row here instead, where a malformed one stops the build.
    #[test]
    fn every_row_declares_a_well_formed_schema() {
        for op in REGISTRY {
            assert_eq!(
                op.args().count(),
                op.args.split_whitespace().count(),
                "`{}` has an argument with no `name:type`: {:?}",
                op.name,
                op.args
            );
            for (arg, ty, _) in op.args() {
                assert!(ARG_TYPES.contains(&ty), "`{}`'s `{arg}` has unknown type `{ty}`", op.name);
            }
            assert!(!op.doc.is_empty() && !op.result.is_empty(), "`{}` is undocumented", op.name);
        }
    }

    /// Uniqueness matters twice over: two rows of one name would give the MCP tool list a
    /// duplicate (a 400, like a bad name) and make `find` silently prefer the first.
    #[test]
    fn op_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for op in REGISTRY {
            assert!(seen.insert(op.name), "`{}` is declared twice", op.name);
        }
    }

    /// The other half of the coverage claim. A row without a dispatch arm falls through to the
    /// match's catch-all and answers `unknown op` — an op the palette, the MCP tool list and the
    /// frontend's `OpName` union all advertise and nothing can actually call. (The converse — an
    /// arm without a row — needs no test: the gate in `dispatch` refuses the op before the match
    /// is reached, so such an arm is unreachable rather than silently live.)
    #[test]
    fn every_registry_op_has_a_dispatch_arm() {
        let state = crate::AppState::new();
        for op in REGISTRY {
            let req = serde_json::json!({ "id": 1, "op": op.name }).to_string();
            let reply = crate::dispatch(&state, &req).expect("a numeric id is always answered");
            assert!(
                !reply.contains(&format!("unknown op `{}`", op.name)),
                "`{}` is in the registry but `dispatch` has no arm for it: {reply}",
                op.name,
            );
        }
        state.release_mount();
    }

    /// The generated frontend union, kept honest. On drift the file is REWRITTEN and the test
    /// fails once, so the fix is to re-run and commit rather than to hand-transcribe a list.
    #[test]
    fn the_frontend_op_union_is_generated_from_the_registry() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../frontend/src/lib/api/ops.ts");
        let want = typescript();
        if std::fs::read_to_string(&path).ok().as_deref() != Some(want.as_str()) {
            std::fs::write(&path, &want).expect("rewriting the generated op union");
            panic!("{} was stale; it has been regenerated — review and commit it", path.display());
        }
    }
}
