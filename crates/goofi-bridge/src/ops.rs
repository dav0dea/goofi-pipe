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
    /// are the human file-browser's half of the same door. `new` shares that very arm: it empties
    /// the patch AND clears the undo history, so an agent that called it could not take it back
    /// (user, 2026-08-10 — a human who wants a fresh patch makes one). `set_viewpoint` is here
    /// because a viewpoint belongs to a client that has a screen — an agent has no camera to move,
    /// and moving the human's would be the whole hazard.
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
    /// value the engine round-trips), `panel_type` (a string out of [`crate::vocab`], advertised
    /// as a JSON-Schema `enum`), and the `[]` suffix for a list.
    pub args: &'static str,
    /// The doc TEMPLATE. `{panel_types}` and `{viewer_kinds}` expand to the vocabularies — see
    /// [`Op::doc`]. Read it through that, never as the raw field.
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

    /// The doc with its vocabulary placeholders expanded. A caller reading this never has to GUESS
    /// a panel type or a viewer kind — which is the point: the teachable refusal is the fallback,
    /// and a description that enumerates the choices is the mechanism.
    pub fn doc(&self) -> String {
        self.doc
            .replace("{panel_types}", &crate::vocab::panel_types_help())
            .replace("{viewer_kinds}", &crate::vocab::viewer_kinds_help())
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
         doc: "Delete a node, a sub-patch member, or a whole collapsed sub-patch instance. Idempotent: a uid naming no node succeeds having deleted nothing, so confirm with inspect_patch rather than reading `ok` as proof the node existed.",
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
    Op { name: "set_viewpoint", surface: ControlOnly, writes: true, args: "viewpoint:json!",
         doc: "Store where this client is looking — active page, maximize, camera, each panel's sub-patch path. Persisted, never converged, never dirtying.",
         result: "{ok: true}" },
    Op { name: "inspect_layout", surface: Mcp, writes: false, args: "",
         doc: "The whole arrangement as a tree: every page, split and panel with its id, order and share. How a caller discovers the split id it needs to name as a move target.",
         result: "{text: string}" },
    Op { name: "session_list_pages", surface: Mcp, writes: false, args: "",
         doc: "The layout pages, in order, with the panel count of each. Pages are addressed by name everywhere else.",
         result: "{pages: [{name, id, index, panels}]}" },
    Op { name: "session_add_page", surface: Mcp, writes: true,
         args: "name:string! index:int subtree:string",
         doc: "Add a layout page at `index` in the tab strip. It holds one node-editor panel — or, with `subtree`, is built AROUND an existing panel or split, which is the drag-onto-the-tab-bar gesture. The name must be free: it is how every page op addresses it.",
         result: "{page, panel} — the new page's id and its root panel's" },
    Op { name: "session_remove_page", surface: Mcp, writes: true, args: "name:string!",
         doc: "Remove a page and every panel on it. The last page stays.",
         result: "{ok: true}" },
    Op { name: "session_rename_page", surface: Mcp, writes: true, args: "from:string! to:string!",
         doc: "Rename a page. A field edit: its id and every panel on it stand.",
         result: "{ok: true}" },
    Op { name: "session_reorder_page", surface: Mcp, writes: true, args: "name:string! to_index:int!",
         doc: "Move a page to a position in the tab strip.",
         result: "{ok: true}" },
    Op { name: "page_list_panels", surface: Mcp, writes: false, args: "page:string!",
         doc: "The panels on a page as a table: uid, type, the node each is bound to, and its share of the page.",
         result: "{text: string}" },
    Op { name: "page_split_panel", surface: Mcp, writes: true,
         args: "page:string! panel:string! direction:string place_before:bool ratio:float",
         doc: "Split a panel along `row`/`column`, birthing an EMPTY panel that takes `ratio` of its space (default half) after the target, or before it with `place_before`. Give the new panel content with page_set_panel.",
         result: "the new panel's uid" },
    Op { name: "page_set_panel", surface: Mcp, writes: true,
         args: "page:string! panel:string! type:panel_type state:json",
         doc: "Set a panel's type and/or its state (a viewer's `{node, slot, kind}`). State MERGES key by key — send only what changes, and null to clear a key. A new type clears the old type's state, so send both together to rebind. Sizing is page_resize_split's.\n\n`type` is one of: {panel_types}.\n\nA viewer panel's `state.kind` is one of: {viewer_kinds}; a STRING or TABLE slot ignores it and uses its own.",
         result: "the resulting arrangement, as inspect_layout draws it" },
    Op { name: "page_move_panel", surface: Mcp, writes: true,
         args: "page:string! panel:string! new_parent:string! order_index:int",
         doc: "Move a panel — or the whole subtree under a split id — to sit at `order_index` inside another split, on any page. Identity and every descendant are preserved.",
         result: "{ok: true}" },
    Op { name: "page_insert_at_panel", surface: Mcp, writes: true,
         args: "page:string! subtree:string! target:string! direction:string place_before:bool ratio:float",
         doc: "Move an existing panel or split to sit beside `target` on `page`, splitting it along `row`/`column`. One op, so a drag is one undo step; taking a page's last panel takes the page with it.",
         result: "{ok: true}" },
    Op { name: "page_resize_split", surface: Mcp, writes: true,
         args: "page:string! split:string! fractions:float[]!",
         doc: "Set the shares of ALL of a split's children at once, in child order — what a resize drag commits. Renormalized to fill the slot.",
         result: "{ok: true}" },
    Op { name: "page_remove_panel", surface: Mcp, writes: true, args: "page:string! panel:string!",
         doc: "Close a panel (or a whole split's subtree). Its space goes to its siblings; a page keeps its last panel.",
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
    Op { name: "new", surface: ControlOnly, writes: true, args: "",
         doc: "Replace the open patch with an empty one. Unsaved work is lost.",
         result: "{ok: true}" },
    Op { name: "undo", surface: Mcp, writes: true, args: "",
         doc: "Undo this session's last graph command. Each caller's session has its own stack.",
         result: "{changed: bool, can_undo: bool, can_redo: bool}" },
    Op { name: "redo", surface: Mcp, writes: true, args: "",
         doc: "Redo this session's last undone graph command.",
         result: "{changed: bool, can_undo: bool, can_redo: bool}" },
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
    Op { name: "list_harnesses", surface: ControlOnly, writes: false, args: "",
         doc: "The agent harnesses installed on this machine, and the ones goofi has running.",
         result: "{instances: [{id, harness, state, exit_code}], detected: [{harness, path, version}]}" },
    Op { name: "spawn_harness", surface: ControlOnly, writes: false, args: "harness:string!",
         doc: "Launch an agent harness on a PTY with the patch workspace as its cwd, minting the MCP address it is handed. Read its terminal at /term/<instance_id>.",
         result: "{instance_id: string}" },
    Op { name: "stop_harness", surface: ControlOnly, writes: false, args: "instance:string!",
         doc: "Stop a running harness (SIGTERM, then SIGKILL), or dismiss one that already exited. Its MCP address drops immediately; the exit code arrives on harness_changed.",
         result: "{ok: true}" },
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
    const ARG_TYPES: &[&str] = &[
        "uid", "string", "float", "int", "bool", "float2", "json", "panel_type", "uid[]",
        "string[]", "float[]",
    ];

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
        // The `!` itself has to reach the parse, or Task 4 would advertise every argument as
        // optional and a model would omit the one the op cannot run without.
        let add: Vec<_> = find("add_node").expect("add_node is registered").args().collect();
        assert_eq!(add[0], ("type", "string", true));
        assert_eq!(add[1], ("pos", "float2", false));
    }

    /// A caller that has to guess a vocabulary word gets it wrong (`params` for `parameters`), and
    /// the guess used to be answered `{ok: true}`. So the description ENUMERATES both vocabularies,
    /// and it does it by expansion rather than by a hand-copied list — which would be the very
    /// duplication `vocab.rs` exists to remove.
    #[test]
    fn the_panel_op_names_the_vocabularies_a_caller_would_otherwise_guess() {
        let doc = find("page_set_panel").expect("page_set_panel is registered").doc();
        for word in ["parameters", "node-editor", "viewer", "line", "trajectory", "topomap"] {
            assert!(doc.contains(word), "`{word}` is not offered by page_set_panel's doc: {doc}");
        }
        for op in REGISTRY {
            let doc = op.doc();
            assert!(
                !doc.contains("{panel_types}") && !doc.contains("{viewer_kinds}"),
                "`{}` has an unexpanded placeholder — a model would read it verbatim",
                op.name
            );
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

    /// `surface` is the one column with a SAFETY consequence, and Task 4 generates the agent's
    /// whole tool list from it — so it is pinned as a set, not as a property. Every name here
    /// either replaces the patch an agent is working inside (and, for the three that share the
    /// `load` arm, its undo history with it), is the human file browser's half of that door, or is
    /// a harness op: an agent that could spawn or kill a harness could spawn itself a peer, or
    /// terminate the very process it is speaking through (user, 2026-08-10).
    /// Adding a row to this list is a decision; the test is where it gets made deliberately.
    #[test]
    fn only_the_self_terminating_and_file_browser_ops_are_kept_off_the_agent_surface() {
        let control_only: Vec<&str> =
            REGISTRY.iter().filter(|o| o.surface == ControlOnly).map(|o| o.name).collect();
        assert_eq!(
            control_only,
            [
                "list_dir",
                "set_viewpoint",
                "serialize",
                "save",
                "load_text",
                "load",
                "new",
                "list_harnesses",
                "spawn_harness",
                "stop_harness"
            ]
        );
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
