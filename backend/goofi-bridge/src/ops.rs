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
    Op { name: "list_nodes", surface: Mcp, writes: false, args: "type:string",
         doc: "The node palette: every registered type with its slots, params, docs and availability.\n\n\
               Name a `type` and you get that ONE entry instead, in full — the same fields plus where it came from and its source text. A native type has no source to read; copy a python node into the patch workspace to modify one.",
         result: "{types: [{type, category, doc, input_slots, output_slots, params, available}]} — or, for one `type`, that entry plus {language, tier, provenance, path, source}." },
    Op { name: "rescan_nodes", surface: Mcp, writes: true, args: "",
         doc: "Re-read the shipped and patch node directories; live instances of a changed type restart onto the new code. Call after writing a node file.",
         result: "{added: [type], changed: [type], removed: [type]}" },
    Op { name: "add_node", surface: Mcp, writes: true,
         args: "type:string! pos:float2 name:string inst_id:uid member_uid:uid params:json",
         doc: "Create a node of `type`. `inst_id` births it inside that sub-patch; absent = root. `params` is edit_node's bag, applied at birth. `member_uid` asks for a CHOSEN uid, so a caller rebuilding a graph it already knows keeps its uid-keyed bindings; naming one the patch already holds answers with that node rather than a second one.\n\n\
               The boundary types (InArray/InString/InTable and the Out trio) create a PORT of the sub-patch named by `inst_id`, which is required for them. A port is a node in every way an op can see — it is named, moved, wired and removed by the same five ops — but it never runs, so it takes no params and no chosen uid. To COPY a node rather than build one, read it with copy_nodes and put it back with paste_nodes.",
         result: "{uid, name, input_slots, output_slots, params} — the node as born, so it can be wired and tuned without a follow-up read. `name` is what nd() addresses it by." },
    Op { name: "copy_nodes", surface: Mcp, writes: false, args: "nodes:uid[]!",
         doc: "Read `nodes` and everything they hold — a sub-patch's members, their ports and the nested sub-patches below them, to any depth — as a self-contained fragment. A link rides only when BOTH its ends are in the fragment. The shape is the `.gfi`'s own, so a fragment is a patch's worth of nodes in the format a patch is written in, and paste_nodes is what puts one back.",
         result: "{doc: {nodes, links}} — the fragment, keyed by the uids it was read from" },
    Op { name: "paste_nodes", surface: Mcp, writes: true, args: "doc:json! pos:float2 inst_id:uid",
         doc: "Add a copy_nodes fragment on FRESH uids and fresh names, so it lands beside whatever it was copied from rather than colliding with it. `pos` shifts the whole fragment by that offset; `inst_id` puts its roots inside that sub-patch, absent = root. A record naming a scope that is IN the fragment keeps the shape it was copied with. One command, so it is one undo step.",
         result: "{rename: {old_uid: new_uid}} — every record's uid in the fragment mapped to the one it was created at" },
    Op { name: "remove_node", surface: Mcp, writes: true, args: "node:uid!",
         doc: "Delete whatever the uid names — a leaf, a boundary port or a whole sub-patch. A sub-patch takes everything inside it, to any depth: nested sub-patches, their members and their ports. A port of an enclosing sub-patch that existed only to expose a deleted one goes with it. Idempotent: a uid naming no node succeeds having deleted nothing, and says so.",
         result: "{removed: bool} — false when the uid named nothing" },
    Op { name: "restart_node", surface: Mcp, writes: true, args: "node:uid!",
         doc: "Respawn a node in place, keeping its uid, name, params, links and scope. Recovery, not an edit — `setup()` runs again.",
         result: "{ok: true}" },
    Op { name: "add_link", surface: Mcp, writes: true,
         args: "node_out:uid! slot_out:string! node_in:uid! slot_in:string!",
         doc: "Wire an output slot to an input slot. Refuses a dtype mismatch, naming both ends; refuses an end that names no node — so a reply means the wire is really there.\n\n\
               A link never crosses a sub-patch boundary, and the two acts that look like it are ordinary links in different scopes. From the OUTSIDE you wire a node to the sub-patch's facade, naming a port's uid as the slot; the wire is stored against the inner leaf that port exposes, so a facade port with nothing behind it is refused. From the INSIDE you wire a port to a member, both of them in that sub-patch; a port carries one slot, `value`, and one inner wire, so a second is refused rather than replacing the first.",
         result: "{node_out, slot_out, node_in, slot_in, dtype} — the wire as made, with a facade endpoint resolved to the inner leaf it exposes." },
    Op { name: "remove_link", surface: Mcp, writes: true,
         args: "node_out:uid! slot_out:string! node_in:uid! slot_in:string!",
         doc: "Remove one wire, addressed by both of its endpoints — a boundary port's inner wire included. Idempotent, like remove_node.",
         result: "{removed: bool} — false when there was no such wire" },
    Op { name: "refresh_param", surface: Mcp, writes: true, args: "node:uid! group:string! name:string!",
         doc: "Ask a node to re-enumerate a refreshable string param's options (a device or stream picker). The scan runs on the node's own thread, so this reply only says the request was dispatched — read the fresh options back with inspect_node.",
         result: "{ok: true} — the options land on the node; inspect_node reports them" },
    Op { name: "edit_node", surface: Mcp, writes: true,
         args: "node:uid! name:string pos:float2 params:json viewers:json",
         doc: "Edit a node: rename it, move it, set params, set viewers — any of them, in one step and one undo. An omitted field is left alone. A sub-patch boundary port takes every field but `params`, which it has no thread to hold: its name is in the one namespace nd() reads, so a collision is refused exactly as a leaf's is, and its `value` slot takes a viewer exactly as a leaf's output does.\n\n\
               `params` is `{group: {param: …}}`, the shape add_node takes. A param entry is either a bare value or `{value, expression, mode, triggers}`: `mode` is `constant`/`expression` and defaults to `expression` when an expression is given, so binding one is a single field. An empty expression clears the binding. Only the params named are touched.\n\n\
               `triggers` defaults false, and that is almost always right: a binding re-evaluates on its own — when a referenced node emits, or on each of the node's own runs for a ref-less one — and the node reads the fresh value on its next normal run. `triggers: true` ALSO wakes the node's process() on every evaluation, making the reference its clock. Reach for it only when the node would otherwise not run (a trigger input with no wire into it) and you want the referenced node to drive it. Never on a ref-less expression (`t`, `globals.x`): that free-runs the node at its common.max_frequency.\n\n\
               A value is coerced to the param's declared type — a fraction into an int rounds, a value of the wrong kind falls back to that type's zero. The declared min/max are the editor's range, NOT a clamp.\n\n\
               `viewers` is `{slot: {kind, settings}}`, merged key by key, so only the slots named move. `kind` is one of: {viewer_kinds}.",
         result: "{params} — every param touched, as STORED, with its binding error if the expression did not compile." },
    Op { name: "set_viewpoint", surface: ControlOnly, writes: true, args: "viewpoint:json!",
         doc: "Store where this client is looking — active tab, maximize, camera, each panel's sub-patch path. Persisted, never converged, never dirtying.",
         result: "{ok: true}" },
    Op { name: "inspect_layout", surface: Mcp, writes: false, args: "tab:string",
         doc: "The arrangement as a tree: every tab, split and panel with its id, order and share of its parent. How a caller discovers the ids every layout op addresses. `tab` narrows it to one tab; no arg = all of them.",
         result: "{text: string}" },
    Op { name: "edit_panel", surface: Mcp, writes: true,
         args: "panel:uid! name:string type:panel_type state:json fractions:float[]",
         doc: "Edit ONE entry's fields — a tab, a split or a panel, whichever the id names. Any mix of them is one call and one undo, and an omitted field is left alone.\n\n\
               `name` relabels a TAB. Its id and every panel on it stand; the strip index is where it sits, which place_panel owns.\n\n\
               `type` and `state` are a PANEL's. State MERGES key by key — send only what changes, and null to clear a key. A new type clears the old type's state, so send both together to rebind. `type` is one of: {panel_types}. A viewer panel's `state.kind` is one of: {viewer_kinds}; a STRING or TABLE slot ignores it and uses its own.\n\n\
               `fractions` sets the shares of ALL of a SPLIT's children at once, in child order — what a resize drag commits. Renormalized to fill the slot.",
         result: "{text} — the resulting arrangement, as inspect_layout draws it" },
    Op { name: "place_panel", surface: Mcp, writes: true,
         args: "panel:uid to:uid index:int direction:string ratio:float name:string",
         doc: "Put a panel somewhere. WHAT is placed: with `panel`, that entry — a panel, a whole split's subtree, or a tab — moves; with no `panel`, a fresh empty one is born. WHERE it lands is the same grammar either way, so a drag and a birth are one op and one undo step; taking a tab's last panel takes the tab with it.\n\n\
               With `to` and `direction` it lands BESIDE that panel, on its `left`/`right`/`top`/`bottom`, taking `ratio` of its space (default half) — a drop on a panel's edge, or a split.\n\n\
               With `to` and no direction it lands INSIDE that split, at `index` among its children — a MOVE only, since a fresh panel has nothing to put inside one; born, it simply divides `to` on the default side.\n\n\
               With no `to` it lands on a tab of its own at `index` in the strip — `name` labels it, and is minted (`Tab 2`, `Tab 3`, …) unless you give one. A `panel` that IS a tab just moves to `index` instead, because it already has one.",
         result: "{id, tab, text} — what was placed, the tab it is on, and the arrangement as inspect_layout draws it" },
    Op { name: "remove_panel", surface: Mcp, writes: true, args: "panel:uid!",
         doc: "Close a panel, a whole split's subtree, or a tab and every panel on it. Its space goes to its siblings; a tab keeps its last panel, and the last tab stays.",
         result: "{text} — the resulting arrangement, as inspect_layout draws it" },
    Op { name: "set_global", surface: Mcp, writes: true, args: "name:string! value:json type:string",
         doc: "Write a patch global: create it, change its value, or — with NO value — delete it. `type` is one of float/int/bool/string, required only when the global is new; giving a different one than it holds is refused, because every expression reading it depends on its type. System globals cannot be deleted. To rename one, compound a set of the new name with a delete of the old.",
         result: "{value} — the value as stored, type-coerced — or {removed: true}" },
    Op { name: "compound", surface: Mcp, writes: true, args: "ops:json!",
         doc: "Run several write ops in order as ONE undo step. `ops` is a list of `{op, payload}`. A refused step takes back the ones that already landed, so the call either happens whole or not at all. A step must be an undoable write: undo, redo, compound itself and the ops that replace the patch are refused.",
         result: "{results} — each step's own reply, in order" },
    Op { name: "group_nodes", surface: Mcp, writes: true, args: "members:uid[]! pos:float2",
         doc: "Collapse nodes into a new sub-patch, returning its instance uid. `members` must share one scope, and one of them may itself be a sub-patch. Every wire that ends up CROSSING the new boundary mints a port to carry it, so nothing is disconnected and nothing stops running; a wire buried in a nested member mints a port there too, so it can reach the new boundary.",
         result: "{inst_id: uid}" },
    Op { name: "expand_instance", surface: Mcp, writes: true, args: "inst_id:uid!",
         doc: "Dissolve a sub-patch, returning its members to the parent scope. Its ports go with it and every wire they carried stands, because a port keeps its wire against the node behind it. A port of an ENCLOSING sub-patch that exposed one of these follows down onto what it exposed.",
         result: "{ok: true}" },
    Op { name: "serialize", surface: ControlOnly, writes: false, args: "",
         doc: "The patch manifest as YAML — a debug read, not a save path.",
         result: "{yaml: string}" },
    Op { name: "save", surface: ControlOnly, writes: false, args: "path:string!",
         doc: "Pack the patch and its workspace to a `.gfi` at `path`, and remember it as the patch's home.",
         result: "{path: string}" },
    Op { name: "load", surface: ControlOnly, writes: true, args: "path:string content:string adopt:bool",
         doc: "Replace the open patch, losing unsaved work. `path` names a `.gfi` and brings its \
               workspace with it; `content` is an inline YAML manifest and carries no workspace; \
               NEITHER is an empty patch, which is a New. At most one of the two. `adopt` (default \
               true) decides whether a loaded FILE becomes the patch's home, which is what a later \
               silent Save overwrites; `/patch.gfi` passes false, because the file a browser upload \
               came from lives on the user's machine and the staged copy this reads is deleted \
               immediately.",
         result: "{ok: true}" },
    Op { name: "undo", surface: Mcp, writes: true, args: "",
         doc: "Undo this session's last graph command. Each caller's session has its own stack.",
         result: "{changed: bool, can_undo: bool, can_redo: bool}" },
    Op { name: "redo", surface: Mcp, writes: true, args: "",
         doc: "Redo this session's last undone graph command.",
         result: "{changed: bool, can_undo: bool, can_redo: bool}" },
    Op { name: "inspect_patch", surface: Mcp, writes: false, args: "scope:uid",
         doc: "Read one scope as a mermaid flowchart — nodes, sub-patches, boundary ports and wires. No arg = the root scope. Scope-wide and nothing more: what is broken is the whole patch's business, so get_patch answers that.",
         result: "{text: string}" },
    Op { name: "inspect_node", surface: Mcp, writes: false,
         args: "node:uid! slot:string params:bool error:bool",
         doc: "Read one node: its params (values, ranges, expression bindings), each output slot's name and kind and whether the node is emitting on it, and its error. `slot` narrows to one output. The FRAMES are not here and cannot be: subscribe to `/data/<node>/<slot>` to see a node's data, exactly as a viewer does.",
         result: "{text: string}" },
    Op { name: "get_state", surface: Internal, writes: false, args: "",
         doc: "The replicated control-plane projection — nodes, links, globals, arrangement — as plain JSON. What every client mirrors, read without the sync protocol that carries it. ONE `nodes` map carries leaves, sub-patch facades and boundary ports alike, each naming its scope, and a port's inner wire is in `links` like any other cable.",
         result: "{nodes, links, globals, arrangement} — nodes and globals keyed by id, links a list." },
    Op { name: "get_patch", surface: Mcp, writes: false, args: "",
         doc: "The open patch itself: where it lives, where its workspace is, whether it differs from disk, and every standing error with how long it has stood. One read for `is my patch healthy, and have I saved it`.",
         result: "{save_path: string | null, workspace: string, dirty: bool, errors: [{node, path, error, standing}]}" },
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
