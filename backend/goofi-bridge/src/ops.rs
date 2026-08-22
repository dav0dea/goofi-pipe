//! The op registry — one row per `/control` op, and the single place the op SET is declared. An op
//! not in this table is refused before `dispatch`, and the MCP tool list and the frontend's
//! `OpName` union are both generated from it.

/// Where an op is offered.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Surface {
    /// Mirrored as an MCP tool for an agent.
    Mcp,
    /// `/control` only, and internal: state a test observes, which no product surface consumes.
    Internal,
    /// `/control` only: an op an agent must not reach, because it would replace the patch — or the
    /// camera — it is working in.
    ControlOnly,
}

/// One op's contract.
pub struct Op {
    /// `[a-z0-9_]+`, short enough that `mcp__goofi__<name>` fits in 64 characters: a longer or
    /// dotted name makes a model provider reject the WHOLE tool list.
    pub name: &'static str,
    pub surface: Surface,
    /// Whether a successful call may have changed the graph.
    pub writes: bool,
    /// The params schema: space-separated `name:type`, `!` marking a required one. Types are
    /// `uid`, `string`, `float`, `int`, `bool`, `float2`, `json`, `panel_type`, and `[]` for a list.
    pub args: &'static str,
    /// The doc TEMPLATE; read it through [`Op::doc`], which expands the vocabularies.
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

    /// The doc with its vocabulary placeholders expanded.
    pub fn doc(&self) -> String {
        self.doc
            .replace("{panel_types}", &crate::vocab::panel_types_help())
            .replace("{viewer_kinds}", &crate::vocab::viewer_kinds_help())
    }
}

/// The prefix a model provider gives every tool of this server; its length is the name budget.
pub const MCP_PREFIX: &str = "mcp__goofi__";

use Surface::{ControlOnly, Internal, Mcp};

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
         doc: "Create a node of `type`. `inst_id` births it inside that sub-patch; absent = root. `params` is edit_node's bag, applied at birth.",
         result: "{uid, name, input_slots, output_slots, params} — the node as born, so it can be wired and tuned without a follow-up read. `name` is what nd() addresses it by." },
    Op { name: "remove_node", surface: Mcp, writes: true, args: "node:uid!",
         doc: "Delete a node, a sub-patch member, or a whole collapsed sub-patch instance. Idempotent: a uid naming no node succeeds having deleted nothing, and says so.",
         result: "{removed: bool} — false when the uid named nothing" },
    Op { name: "restart_node", surface: Mcp, writes: true, args: "node:uid!",
         doc: "Respawn a node in place, keeping its uid, name, params, links and scope. Recovery, not an edit — `setup()` runs again.",
         result: "{ok: true}" },
    Op { name: "add_link", surface: Mcp, writes: true,
         args: "node_out:uid! slot_out:string! node_in:uid! slot_in:string!",
         doc: "Wire an output slot to an input slot. Either end may name a sub-patch boundary port. Refuses a dtype mismatch, naming both ends; refuses an end that names no node, or a boundary port with no inner slot behind it — so a reply means the wire is really there.",
         result: "{node_out, slot_out, node_in, slot_in, dtype} — the wire as made, with a boundary endpoint resolved to the inner leaf it exposes." },
    Op { name: "remove_link", surface: Mcp, writes: true,
         args: "node_out:uid! slot_out:string! node_in:uid! slot_in:string!",
         doc: "Remove one wire, addressed by both of its endpoints. Idempotent, like remove_node.",
         result: "{removed: bool} — false when there was no such wire" },
    Op { name: "refresh_param", surface: Mcp, writes: true, args: "node:uid! group:string! name:string!",
         doc: "Ask a node to re-enumerate a refreshable string param's options (a device or stream picker). The scan runs on the node's own thread, so this reply only says the request was dispatched — read the fresh options back with inspect_node.",
         result: "{ok: true} — the options land on the node; inspect_node reports them" },
    Op { name: "edit_node", surface: Mcp, writes: true,
         args: "node:uid! name:string pos:float2 params:json viewers:json",
         doc: "Edit a node: rename it, move it, set params, set viewers — any of them, in one step and one undo. An omitted field is left alone.\n\n\
               `params` is `{group: {param: …}}`, the shape add_node takes. A param entry is either a bare value or `{value, expression, mode, triggers}`: `mode` is `constant`/`expression` and defaults to `expression` when an expression is given, so binding one is a single field. An empty expression clears the binding. Only the params named are touched.\n\n\
               `triggers` defaults false, and that is almost always right: a binding re-evaluates on its own — when a referenced node emits, or on each of the node's own runs for a ref-less one — and the node reads the fresh value on its next normal run. `triggers: true` ALSO wakes the node's process() on every evaluation, making the reference its clock. Reach for it only when the node would otherwise not run (a trigger input with no wire into it) and you want the referenced node to drive it. Never on a ref-less expression (`t`, `globals.x`): that free-runs the node at its common.max_frequency.\n\n\
               A value is coerced to the param's declared type — a fraction into an int rounds, a value of the wrong kind falls back to that type's zero. The declared min/max are the editor's range, NOT a clamp.\n\n\
               `viewers` is `{slot: {kind, settings}}`, merged key by key, so only the slots named move. `kind` is one of: {viewer_kinds}.",
         result: "{params} — every param touched, as STORED, with its binding error if the expression did not compile." },
    Op { name: "set_viewpoint", surface: ControlOnly, writes: true, args: "viewpoint:json!",
         doc: "Store where this client is looking — active tab, maximize, camera, each panel's sub-patch path. Persisted, never converged, never dirtying.",
         result: "{ok: true}" },
    Op { name: "inspect_layout", surface: Mcp, writes: false, args: "from:string",
         doc: "The arrangement as a tree: every tab group, split and panel with its id, order and share of its parent. How a caller discovers the ids every layout op addresses. `from` narrows it to one subtree; no arg = the whole of it. A `tabs` entry shows ONE child at a time and draws the rest as tabs — the topmost one is the workspace's page strip, so a page is a child of it and nothing else.",
         result: "{text: string}" },
    Op { name: "add_panel", surface: Mcp, writes: true,
         args: "at:uid! direction:string place_before:bool ratio:float index:int",
         doc: "Add a panel. With `direction` (`row`/`column`) it SPLITS `at`, taking `ratio` of its space (default half) after it, or before it with `place_before`; the new panel starts empty, so give it content with edit_panel. Without a direction it joins `at` as a TAB, at `index` in that group's strip, starting as a node editor — which is what adding a page is, with `at` naming the topmost group.",
         result: "the new panel's id" },
    Op { name: "edit_panel", surface: Mcp, writes: true, args: "panel:uid! type:panel_type state:json fractions:float[]",
         doc: "Edit one entry: a panel's type and/or state, or a split's shares. State MERGES key by key — send only what changes, and null to clear a key. A new type clears the old type's state, so send both together to rebind.\n\n\
               `type` is one of: {panel_types}.\n\n\
               A viewer panel's `state.kind` is one of: {viewer_kinds}; a STRING or TABLE slot ignores it and uses its own.\n\n\
               `fractions` sets the shares of ALL of a split's children at once, in child order — what a resize drag commits. Renormalized to fill the slot.",
         result: "{text} — the resulting arrangement, as inspect_layout draws it" },
    Op { name: "move_panel", surface: Mcp, writes: true,
         args: "panel:uid! to:uid! direction:string place_before:bool ratio:float index:int",
         doc: "Move a panel — or the whole subtree under a split or tab-group id — somewhere else. With `direction` it lands BESIDE `to`, splitting it; without one it lands INSIDE `to` as a tab, at `index`. Dropping onto a plain panel that way groups the two. One op, so a drag is one undo step; taking a page's last panel takes the page with it.",
         result: "{text} — the resulting arrangement, as inspect_layout draws it" },
    Op { name: "remove_panel", surface: Mcp, writes: true, args: "panel:uid!",
         doc: "Close a panel, a tab group, or a whole split's subtree. Its space goes to its siblings; the workspace keeps its last panel.",
         result: "{text} — the resulting arrangement, as inspect_layout draws it" },
    Op { name: "set_global", surface: Mcp, writes: true, args: "name:string! value:json type:string",
         doc: "Write a patch global: create it, change its value, or — with NO value — delete it. `type` is one of float/int/bool/string, required only when the global is new; giving a different one than it holds is refused, because every expression reading it depends on its type. System globals cannot be deleted. To rename one, compound a set of the new name with a delete of the old.",
         result: "{value} — the value as stored, type-coerced — or {removed: true}" },
    Op { name: "compound", surface: Mcp, writes: true, args: "ops:json!",
         doc: "Run several write ops in order as ONE undo step. `ops` is a list of `{op, payload}`. A refused step takes back the ones that already landed, so the call either happens whole or not at all. A step must be an undoable write: undo, redo, compound itself and the ops that replace the patch are refused.",
         result: "{results} — each step's own reply, in order" },
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
    Op { name: "load", surface: ControlOnly, writes: true, args: "path:string! adopt:bool",
         doc: "Replace the open patch with the `.gfi` at `path`, workspace and all. `adopt` \
               (default true) decides whether the patch takes that path as its home, which is what \
               a later silent Save overwrites; `/patch.gfi` passes false, because the file a \
               browser upload came from lives on the user's machine and the staged copy this \
               reads is deleted immediately.",
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
         args: "node:uid! slot:string params:bool error:bool",
         doc: "Read one node: its params (values, ranges, expression bindings), each output slot's name and kind and whether the node is emitting on it, and its error. `slot` narrows to one output. The FRAMES are not here and cannot be: subscribe to `/data/<node>/<slot>` to see a node's data, exactly as a viewer does.",
         result: "{text: string}" },
    Op { name: "get_state", surface: Internal, writes: false, args: "",
         doc: "The replicated control-plane projection — nodes, links, instances, globals, arrangement — as plain JSON. What every client mirrors, read without the sync protocol that carries it.",
         result: "{nodes, links, instances, globals, arrangement} — each an object keyed by id." },
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
         doc: "Launch an agent harness on a PTY with the patch workspace as its cwd, minting the MCP address it is handed. Read its terminal at /term/<instance_id>. An unknown name is refused with the set this build knows; list_harnesses says which of them are installed.",
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

/// The frontend's `OpName` union, generated from the registry and checked into the tree.
pub fn typescript() -> String {
    let names: Vec<String> = REGISTRY
        .iter()
        .filter(|o| o.surface != Surface::Internal)
        .map(|o| format!("\t| '{}'", o.name))
        .collect();
    format!(
        "// GENERATED from backend/goofi-bridge/src/ops.rs — do not edit by hand.\n\
         // The manager's op registry is the only place an op name is declared: naming one that is\n\
         // not in it is a type error here and an `unknown op` refusal there. Regenerate by running\n\
         // `cargo test -p goofi-bridge`, which rewrites this file when it drifts.\n\
         export type OpName =\n{};\n",
        names.join("\n")
    )
}
