//! The op registry — a phrase TREE, and the single place the op SET is declared. Every op is on
//! every transport: the socket names a flat row directly, the phrase layer walks the tree word by
//! word, and the frontend's `OpName` union is generated from the flat rows. Each group word is
//! written ONCE, with a doc line of its own — what `help` and completion show for that word.

/// One op's contract. In the tree a leaf authors `name` as its FINAL word alone; the flat rows
/// [`registry`] derives carry the full phrase, words joined with single spaces.
pub struct Op {
    pub name: &'static str,
    /// What calling the op IS — see [`Handler`]. The batch gate, the dirty decision and the
    /// re-mirror are all READ off this kind, never declared beside it.
    pub handler: Handler,
    /// The params schema: space-separated `name:type`, `!` marking a required one. Types are
    /// `uid` (a node's NAME, which is unique across the patch, or the uid behind it), `string`,
    /// `float`, `int`, `bool`, `float2`, `json`, `any`, `param_addr`, `endpoint` (`node/slot`,
    /// the node half a name or a uid),
    /// `panel_type`, and `[]` for a list.
    pub args: &'static str,
    /// How many of the LEADING declared args a command line takes as positionals (0..=2). A
    /// list-typed positional is variadic; every positional stays reachable as a flag too.
    pub positional: usize,
    /// The doc TEMPLATE; read it through [`Op::doc`], which expands the vocabularies.
    pub doc: &'static str,
    /// The result schema, as the shape a caller gets back.
    pub result: &'static str,
}

/// One node of the phrase tree: a group word with its own one-line doc, or a leaf op.
pub enum Entry {
    Group(&'static str, &'static str, &'static [Entry]),
    Leaf(Op),
}

/// An op's handler and its KIND in one field. A bool column beside a handler would be a second
/// declaration of what the handler already is, and the one that drifts — so the columns the kind
/// replaced (`writes`, the dirty name-match) are readings of it instead.
#[derive(Clone, Copy)]
pub enum Handler {
    /// Reads state, changes nothing: never dirties, never re-mirrors.
    Read(OpFn),
    /// Routes every mutation through the command history, so it has an exact inverse. The shared
    /// tail in [`crate::AppState::call`] re-mirrors and raises the unsaved dot; a `compound`
    /// step is a Write or a Read.
    Write(OpFn),
    /// Owns its consequences itself — re-mirror, events and dirty transitions — because they are
    /// not a graph command's: a save, a process, a restart, the history ops.
    Effect(OpFn),
}

pub type OpFn = fn(
    &crate::AppState,
    &serde_json::Value,
    &str,
    &mut Vec<String>,
) -> Result<serde_json::Value, String>;

impl Handler {
    pub fn run(
        &self,
        state: &crate::AppState,
        payload: &serde_json::Value,
        actor: &str,
        events: &mut Vec<String>,
    ) -> Result<serde_json::Value, String> {
        let (Handler::Read(f) | Handler::Write(f) | Handler::Effect(f)) = self;
        f(state, payload, actor, events)
    }
    pub fn is_write(&self) -> bool {
        matches!(self, Handler::Write(_))
    }
    pub fn is_read(&self) -> bool {
        matches!(self, Handler::Read(_))
    }
    /// The kind as the word `list_ops` answers with.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Handler::Read(_) => "read",
            Handler::Write(_) => "write",
            Handler::Effect(_) => "effect",
        }
    }
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
            .replace("{boundary_types}", &crate::vocab::boundary_types_help())
    }
}

use crate::arms;
use Entry::{Group, Leaf};
use Handler::{Effect, Read, Write};

pub static TREE: &[Entry] = &[
    Group("session", "the goofi instance as a whole — identity, the open patch, save and load", &[
        Leaf(Op { name: "status", handler: Read(arms::session_status), args: "", positional: 0,
             doc: "The session's identity AND its health: which instance this is, where the patch lives, whether it differs from disk, and every standing error with how long it has stood. One read for `is my patch healthy, and have I saved it`.",
             result: "{instance_id, save_path: string | null, workspace, dirty: bool, errors: [{node, path, error, standing}]} — `node` is the name to pass back, `path` where it sits" }),
        Leaf(Op { name: "state", handler: Read(arms::session_state), args: "", positional: 0,
             doc: "The whole replicated document, exact and ATOMIC: nodes, links, globals and arrangement in one read — what every client mirrors, read without the sync protocol that carries it. ONE `nodes` map carries leaves, sub-patch facades and boundary ports alike, each naming its scope, and a port's inner wire is in `links` like any other cable. Narrowing is the caller's: pipe it through `jq`.",
             result: "{nodes, links, globals, arrangement} — nodes and globals keyed by id, links a list." }),
        Leaf(Op { name: "manifest", handler: Read(arms::session_manifest), args: "", positional: 0,
             doc: "The open patch as YAML — the manifest a `.gfi` holds, diffable and versionable.",
             result: "{yaml: string}" }),
        Leaf(Op { name: "save", handler: Effect(arms::session_save), args: "path:string", positional: 1,
             doc: "Pack the patch and its workspace to a `.gfi`. With no `path` it saves to the patch's home — refused when the patch has never been saved — and a given `path` becomes the new home.",
             result: "{path: string}" }),
        Leaf(Op { name: "load", handler: Effect(arms::session_load), args: "path:string content:string adopt:bool", positional: 1,
             doc: "Replace the open patch, losing unsaved work. `path` names a `.gfi` and brings its \
                   workspace with it; `--content` is an inline YAML manifest and carries no workspace. \
                   Exactly ONE of the two — the empty patch is `session new`. `adopt` (default true) \
                   decides whether a loaded FILE becomes the patch's home, which is what a later \
                   silent save overwrites; `/patch.gfi` passes false, because the file a browser \
                   upload came from lives on the user's machine and the staged copy this reads is \
                   deleted immediately.",
             result: "{ok: true, layout_warning: string | null}" }),
        Leaf(Op { name: "new", handler: Effect(arms::session_new), args: "", positional: 0,
             doc: "Replace the open patch with the empty one, losing unsaved work. The undo history is cleared, so this cannot be taken back.",
             result: "{ok: true, layout_warning: string | null}" }),
    ]),
    Group("node", "one node instance — read it, build it, tune it, remove it", &[
        Leaf(Op { name: "state", handler: Read(arms::node_state),
             args: "node:uid! slot:string params:bool error:bool", positional: 1,
             doc: "Read one node: its params (values, ranges, expression bindings), each output slot's name and kind and whether the node is emitting on it, and its error. `slot` narrows to one output; `--no-params` and `--no-error` drop a section. The FRAMES are not here: `node snapshot` reads one raw, and `/data/<node>/<slot>` streams them exactly as a viewer sees them.",
             result: "{text: string}" }),
        Leaf(Op { name: "snapshot", handler: Read(arms::node_snapshot), args: "output:endpoint! raw:bool",
             positional: 1,
             doc: "The output's latest frame, once — the analysis read, addressed `node/slot`. A facade or a boundary port resolves to the stream behind it, exactly as a viewer's does. It reads the cache the slot's reducer already keeps, so it never wakes the node and never touches the viewers' shared stream. An ARRAY answers its shape and its range, which is what tells silence from signal; `--raw` answers the numbers themselves as base64 NPY. STRING and TABLE answer plain JSON, a table's ARRAY members reading the same way. A slot asked about before anything was cached answers `{frame: null}` with the reason — asking is also what opens the slot's feed, so ask again after the node's next emit.",
             result: "{meta, shape, range: {min, max, mean}} for ARRAY, or {meta, npy_b64} under `--raw`; {meta, value} for STRING/TABLE; {frame: null, reason} before the first cached frame" }),
        Leaf(Op { name: "add", handler: Write(arms::node_add),
             args: "type:string! pos:float2 name:string inst_id:uid member_uid:uid param:json[]", positional: 1,
             doc: "Create a node of `type`. `inst_id` births it inside that sub-patch; absent = root. `name` is a letter then letters or digits and not a Python keyword — one that is taken or illegal is refused, never silently swapped — and an omitted one is minted. Each `--param` is one birth param, self-addressed: `{\"name\": \"group/param\", …}` carrying `node param edit`'s fields — inside a JSON flag under bash, spell nested strings with ESCAPED double quotes (`\"nd(\\\"other\\\").out.sfreq\"`); a single-quoted `nd('x')` inside a single-quoted shell token loses its quotes silently. `member_uid` asks for a CHOSEN uid, so a caller rebuilding a graph it already knows — or wiring a batch it is still building — keeps its uid-keyed bindings; naming one the patch already holds answers with that node rather than a second one.\n\n\
                   The boundary types ({boundary_types}) create a PORT of the sub-patch named by `inst_id`, which is required for them. A port is a node in every way an op can see — it is named, moved, wired and removed by the same ops — but it never runs, so it takes no params. To COPY a node rather than build one, read it with `nodes copy` and put it back with `nodes paste`.",
             result: "{name, uid, input_slots, output_slots, params} — the node as born, so it can be wired and tuned without a follow-up read. `name` is what every op and nd() address it by; the uid is for a caller keying its own records." }),
        Leaf(Op { name: "edit", handler: Write(arms::node_edit),
             args: "node:uid! name:string pos:float2 viewer:json[]", positional: 1,
             doc: "Edit a node's own record: rename it, move it, set viewers — any of them, in one step and one undo. An omitted field is left alone. Params are `node param edit`'s. A sub-patch boundary port takes every field: its name is in the one namespace nd() reads, so a collision is refused exactly as a leaf's is, and its `value` slot takes a viewer exactly as a leaf's output does.\n\n\
                   A `name` is a letter then letters or digits, and not a Python keyword, for every kind of node. An expression reads a name as an ATTRIBUTE — a sub-patch's slot in `nd('chain').out.drain` — and a reference spells `name.slot`, so one that cannot be read there breaks every source naming it, and the rewrite that follows the NEXT rename can no longer find what it broke.\n\n\
                   Each `--viewer` is one slot's inline view, `{\"slot\": \"out\", \"kind\": …, \"settings\": …}`, merged slot by slot so only the slots named move; `{\"slot\": \"out\", \"clear\": true}` removes that slot's stored view. `kind` is one of: {viewer_kinds}.",
             result: "{ok: true}" }),
        Group("param", "one param of a node, addressed `group/param`", &[
            Leaf(Op { name: "edit", handler: Write(arms::node_param_edit),
                 args: "node:uid! param:param_addr! value:string expression:string reference:string mode:string triggers:bool",
                 positional: 2,
                 doc: "Set ONE param, addressed `group/param`. `value` is coerced to the param's declared type — a fraction into an int rounds, a value of the wrong kind falls back to that type's zero; the declared min/max are the editor's range, NOT a clamp. A param has ONE active source, named by `mode`: `constant` (the value), `expression` (Python over nd(), globals and me, at control rate), or `reference` (one producer output spelled `node.slot`, no Python, at the producer's rate). Giving an `expression` or a `reference` implies its mode, so binding one is a single flag; the other two are RETAINED across a mode switch, an empty text clears that text (and the mode, if it was the active one), and a mode or trigger given alone edits what is already there. A `value` on a driven param switches it to `constant`. A reference's producer slot must match the param: a number or bool references an ARRAY or AUDIO output holding one element, a string references a STRING output.\n\n\
                       `triggers` defaults false, and that is almost always right: a binding re-evaluates on its own — when a referenced node emits, when a referenced param (`nd('x').params.<group>.<param>`, `me.params.…`) is edited, or on each of the node's own runs for a ref-less one — and the node reads the fresh value on its next normal run. `triggers: true` ALSO wakes the node's process() on every evaluation, making the reference its clock. Reach for it only when the node would otherwise not run (a trigger input with no wire into it) and you want the referenced node to drive it. Never on a ref-less expression (`t`, `globals.x`): that free-runs the node at its common.max_frequency.",
                 result: "{value, error} — the value as STORED, with its bind error: a compile failure, an unknown producer, or a slot of the wrong kind." }),
            Leaf(Op { name: "refresh", handler: Effect(arms::node_param_refresh),
                 args: "node:uid! param:param_addr!", positional: 2,
                 doc: "Ask a node to re-enumerate a refreshable string param's options (a device or stream picker), addressed `group/param`. The scan runs on the node's own thread, so this reply only says the request was dispatched — read the fresh options back with `node state`.",
                 result: "{ok: true} — the options land on the node; `node state` reports them" }),
            Leaf(Op { name: "pulse", handler: Effect(arms::node_param_pulse),
                 args: "node:uid! param:param_addr!", positional: 2,
                 doc: "Fire a pulse param once, addressed `group/param`: a reset, a trigger, a clear. A pulse \
                       holds no value, so this is a request to the node, never an edit of the document.",
                 result: "{ok: true} — the pulse was dispatched to the node's own thread" }),
        ]),
        Leaf(Op { name: "remove", handler: Write(arms::node_remove), args: "node:uid!", positional: 1,
             doc: "Delete whatever `node` names — a leaf, a boundary port or a whole sub-patch. A sub-patch takes everything inside it, to any depth: nested sub-patches, their members and their ports. A port of an enclosing sub-patch that exposed the deleted node STAYS, unwired — a port is a node, and it outlives what was behind it exactly as an unconnected node outlives the cable it lost. Idempotent: a uid naming no node succeeds having deleted nothing, and says so.",
             result: "{removed: bool} — false when it named nothing" }),
        Leaf(Op { name: "restart", handler: Effect(arms::node_restart), args: "node:uid!", positional: 1,
             doc: "Respawn a node in place, keeping its uid, name, params, links and scope. Recovery, not an edit — `setup()` runs again.",
             result: "{ok: true}" }),
        Leaf(Op { name: "editor", handler: Effect(arms::node_editor), args: "node:uid! show:bool", positional: 1,
             doc: "Open a node's own editor window — a plugin's GUI — on the machine goofi runs on, never in the page; `--no-show` closes it. Only a type whose palette row says `editor: true` has one, and a machine with no display has none.",
             result: "{changed: bool} — false when the editor was already open, or already closed" }),
    ]),
    Group("nodes", "the graph of several nodes — inspect, copy, paste, group", &[
        Leaf(Op { name: "inspect", handler: Read(arms::nodes_inspect), args: "scope:uid", positional: 1,
             doc: "Read one scope as a mermaid flowchart — nodes, sub-patches, boundary ports and wires. A node's mermaid id is its NAME, which is what every other op takes. No arg = the root scope. Scope-wide and nothing more: what is broken is the whole patch's business, so `session status` answers that.",
             result: "{text: string}" }),
        Leaf(Op { name: "copy", handler: Read(arms::nodes_copy), args: "nodes:uid[]!", positional: 1,
             doc: "Read `nodes` and everything they hold — a sub-patch's members, their ports and the nested sub-patches below them, to any depth — as a self-contained fragment. A link rides only when BOTH its ends are in the fragment. The shape is the `.gfi`'s own, so a fragment is a patch's worth of nodes in the format a patch is written in, and `nodes paste` is what puts one back.",
             result: "{doc: {nodes, links}} — the fragment, keyed by the uids it was read from" }),
        Leaf(Op { name: "paste", handler: Write(arms::nodes_paste),
             args: "doc:json! pos:float2 inst_id:uid", positional: 0,
             doc: "Add a `nodes copy` fragment on FRESH uids and fresh names, so it lands beside whatever it was copied from rather than colliding with it. `pos` shifts the whole fragment by that offset; `inst_id` puts its roots inside that sub-patch, absent = root. A record naming a scope that is IN the fragment keeps the shape it was copied with. One command, so it is one undo step.",
             result: "{rename: {old_uid: new_uid}} — every record's uid in the fragment mapped to the one it was created at" }),
        Leaf(Op { name: "group", handler: Write(arms::nodes_group), args: "nodes:uid[]! pos:float2", positional: 1,
             doc: "Collapse nodes into a new sub-patch, returning it. `nodes` must share one scope, and one of them may itself be a sub-patch. Every wire that ends up CROSSING the new boundary mints a port to carry it, so nothing is disconnected and nothing stops running; a wire buried in a nested member mints a port there too, so it can reach the new boundary.",
             result: "{name, inst_id} — the sub-patch as born; `name` is what every op takes, `inst_id` the uid behind it" }),
        Leaf(Op { name: "ungroup", handler: Write(arms::nodes_ungroup), args: "subpatch:uid!", positional: 1,
             doc: "Dissolve a sub-patch, returning its members to the parent scope. Its ports go with it and every wire they carried stands, because a port keeps its wire against the node behind it. A port of an ENCLOSING sub-patch that exposed one of these follows down onto what it exposed.",
             result: "{ok: true}" }),
    ]),
    Group("link", "one wire between an output and an input", &[
        Leaf(Op { name: "add", handler: Write(arms::link_add),
             args: "from:endpoint! to:endpoint!", positional: 2,
             doc: "Wire `from` (an output, as `node/slot`) to `to` (an input). Refuses a dtype mismatch, naming both ends; refuses an end that names no node — so a reply means the wire is really there.\n\n\
                   A link never crosses a sub-patch boundary, and the two acts that look like it are ordinary links in different scopes. From the OUTSIDE you wire a node to the sub-patch's facade, naming a port's uid as the slot; the wire is stored against the PORT, whether or not anything is behind it yet. From the INSIDE you wire a port to a member, both of them in that sub-patch. A port carries one slot, `value`, on both of its sides.",
             result: "{from, to, dtype} — the wire as made, both ends named, with a facade endpoint resolved to the PORT it named." }),
        Leaf(Op { name: "remove", handler: Write(arms::link_remove),
             args: "from:endpoint! to:endpoint!", positional: 2,
             doc: "Remove one wire, addressed by both of its endpoints — a boundary port's inner wire included. Idempotent, like `node remove`.",
             result: "{removed: bool} — false when there was no such wire" }),
    ]),
    Group("global", "the patch globals — what an expression reads as `globals.group.element`", &[
        Leaf(Op { name: "list", handler: Read(arms::global_list), args: "", positional: 0,
             doc: "Every patch global — what an expression can read and the global writes can set — each with the lock that holds it (its own and its group's together), and every group that carries a lock. The `system` group is goofi's own: config-locked for life, and `system.goofi_home` is the machine's value, never saved into a patch.",
             result: "{globals: [{name, type, value, lock: {config, value}, control?, source?}], groups: {group: {lock}}}" }),
        Group("entry", "one global, addressed `group.element`", &[
            Leaf(Op { name: "add", handler: Write(arms::global_add),
                 args: "name:string! type:string! value:any! control:json", positional: 1,
                 doc: "Create a patch global. `name` is `group.element` — every global is in a group. `type` is one of float/int/bool/string; a name the patch already holds is refused — `global entry edit` changes one. `control` makes it a control-panel element: {kind, min, max, step, options, x, y, w, h}, where kind is knob/slider/number/field/toggle/dropdown and must be able to draw the type.",
                 result: "{value} — the value as stored, type-coerced" }),
            Leaf(Op { name: "edit", handler: Write(arms::global_edit), args: "name:string! value:any control:json", positional: 1,
                 doc: "Change an existing global's value, type-coerced to the type it holds. The type is immutable, because every expression reading a global depends on it: re-typing is a remove and an add. A value-locked global refuses the edit, and so does the machine's own (system.goofi_home). `control` sets the control-panel widget and its place, `null` clears it, and giving one makes `value` optional — which is what a panel sends when it moves a widget; a config-locked global refuses it.",
                 result: "{value} — the value as stored, type-coerced" }),
            Leaf(Op { name: "remove", handler: Write(arms::global_remove), args: "name:string!", positional: 1,
                 doc: "Delete a patch global. A config-locked one refuses, and a system global always is.",
                 result: "{removed: true}" }),
            Leaf(Op { name: "source", handler: Write(arms::global_source), args: "name:string! reference:string! index:int", positional: 2,
                 doc: "Make a global FOLLOW one producer output, `node.slot`, at that producer's rate: the manager writes the global on every frame that changes it, and nobody else may set it until the source is cleared with an empty reference. `index` picks one number out of a frame wider than one — a MIDI controller's `cc` is 128 of them — and a frame that holds one number needs none. The reference follows a node rename exactly as a param's does. A config-locked global refuses; a value-locked one holds its value and takes nothing.",
                 result: "{source: {reference, index} | null}" }),
            Leaf(Op { name: "lock", handler: Write(arms::global_lock), args: "name:string! config:bool value:bool", positional: 1,
                 doc: "Lock or unlock one global on its own account: `config` freezes its name, its widget and its place in a group, `value` freezes its value alone. An axis not named keeps what it has, and its group's lock holds it besides. The system group's locks are goofi's own.",
                 result: "{lock: {config, value}} — the global's own lock as stored" }),
            Leaf(Op { name: "rename", handler: Write(arms::global_rename), args: "name:string! to:string!", positional: 2,
                 doc: "Rename a global, and rewrite every expression that reads it. `to` is a full `group.element`, so one op both renames an element and moves it to another group.",
                 result: "{name} — the name as stored" }),
        ]),
        Group("group", "a whole group of globals — a control panel is one", &[
            Leaf(Op { name: "rename", handler: Write(arms::global_group_rename), args: "from:string! to:string!", positional: 2,
                 doc: "Rename a group, moving every member with it and rewriting every expression that reads one. A config-locked group refuses, and the system group always does.",
                 result: "{group} — the group as stored" }),
            Leaf(Op { name: "lock", handler: Write(arms::global_group_lock), args: "group:string! config:bool value:bool", positional: 1,
                 doc: "Lock or unlock a whole group, reaching every member: `config` freezes every name, widget and the membership itself — nothing is added, renamed or removed — and `value` freezes every value. An axis not named keeps what it has. A control panel's edit mode is this lock, inverted. The system group's lock is goofi's own.",
                 result: "{lock: {config, value}} — the group's lock as stored" }),
        ]),
    ]),
    Group("control", "a control panel: one group of globals drawn as widgets, and the door that edits it locked", &[
        Leaf(Op { name: "list", handler: Read(arms::control_list), args: "", positional: 0,
             doc: "Every control panel and the group it draws, and every group holding a widget: each element with its value, its widget (`control`), its lock and what it follows (`source`).",
             result: "{panels: [{panel, group}], groups: {group: {lock, elements: [{name, element, type, value, control, lock, source?}]}}}" }),
        Leaf(Op { name: "add", handler: Write(arms::control_add), args: "group:string! kind:string! element:string x:float y:float w:float h:float value:any min:float max:float step:float options:json", positional: 2,
             doc: "Bear a widget in a control panel's group: a global of the kind's own type, carrying the widget. `kind` is knob/slider/number/field/toggle/dropdown. `element` is minted `knob0`, `knob1`, … when not given, and the cell is the first free one when `x`/`y` are not. Works on a config-locked group and leaves it locked — a control panel owns its group — as ONE undo step.",
             result: "{name, control} — the global's full name and the widget as stored" }),
        Leaf(Op { name: "edit", handler: Write(arms::control_edit), args: "group:string! element:string! name:string kind:string min:float max:float step:float options:json x:float y:float w:float h:float", positional: 2,
             doc: "Change a widget: `name` renames the element (every expression reading it follows), and the rest re-shape the widget, its range, its options or its cell. Through the lock, as `control add` is.",
             result: "{name} — the element's full name after the edit" }),
        Leaf(Op { name: "remove", handler: Write(arms::control_remove), args: "group:string! element:string!", positional: 2,
             doc: "Delete a widget and the global under it. Through the lock, as `control add` is.",
             result: "{removed: true}" }),
        Leaf(Op { name: "source", handler: Write(arms::control_source), args: "group:string! element:string! reference:string! index:int", positional: 2,
             doc: "Make a widget FOLLOW one producer output, `node.slot`, with `index` picking one number out of a wide frame — a MIDI controller's `cc` is 128 of them — so a knob on a controller drives the widget. An empty reference clears it. Through the lock, as `control add` is.",
             result: "{source: {reference, index} | null}" }),
    ]),
    Group("library", "the node types — what `node add` can build", &[
        Leaf(Op { name: "list", handler: Read(arms::library_list), args: "full:bool", positional: 0,
             doc: "The node library as an INDEX: every registered type, and the first line of its doc. Everything else about the one type you then pick — where it came from, its slots, its params, the rest of its doc — is `library get`'s, so choosing from this list costs a catalog and not a manual. `--full` answers the palette a client draws the add-menu from instead.",
             result: "{types: [{type, doc}]}, an unloadable one also carrying `available: false` and saying so in its doc. With `--full`, every entry carries {source, bundle, tags, available, missing_deps, editor, input_slots, input_multi, output_slots, params} and the WHOLE doc." }),
        Leaf(Op { name: "get", handler: Read(arms::library_get), args: "type:string! source:bool", positional: 1,
             doc: "ONE library entry in full: the palette fields — slots, params, availability — plus where the type came from. `--source` reads the file itself too, under `text`. Copy a node into the patch workspace to modify one.",
             result: "the `library list --full` entry plus {language, tier, provenance, path}, and `text` under `--source`" }),
        Leaf(Op { name: "refresh", handler: Effect(arms::library_refresh), args: "", positional: 0,
             doc: "Re-read the shipped and patch node directories; live instances of a changed type restart onto the new code. Call after writing a node file.",
             result: "{added: [type], changed: [type], removed: [type]}" }),
    ]),
    Group("dir", "the goofi host's filesystem", &[
        Leaf(Op { name: "list", handler: Read(arms::dir_list), args: "path:string hidden:bool", positional: 1,
             doc: "List a directory on the goofi host — the save/load browser's read. Dot-names are left out; `--hidden` includes them.",
             result: "{path, parent, entries: [{name, kind, is_gfi}], roots}" }),
    ]),
    Group("op", "the vocabulary itself", &[
        Leaf(Op { name: "list", handler: Read(arms::op_list), args: "doc:bool", positional: 0,
             doc: "Every op this server speaks: its name, its arguments (`!` marks a required one) and its kind — a `write` is undoable and may ride in a batch, an `effect` runs alone. Every argument is reachable as `--name value`, which is all a caller needs to write one. What an op DOES is `<op> --help`; `--doc` answers the whole vocabulary explained, which is the manual and costs like one.",
             result: "{ops: [{op, args, kind}]}, each row also carrying {positional, doc, result} under `--doc`" }),
        Leaf(Op { name: "complete", handler: Read(arms::op_complete), args: "line:string", positional: 1,
             doc: "What can come NEXT on a partial command line — the shell completion read. Each candidate is a word with a one-line doc: a group or op word mid-phrase, a flag once the op is named, or a value for a flag with a known vocabulary (a panel type, a live node's name). The line's last word, when partial, filters the candidates.",
             result: "{text: string} — one candidate per line, `word<TAB>doc`" }),
    ]),
    Group("agent", "the agent harnesses goofi launches", &[
        Leaf(Op { name: "list", handler: Read(arms::agent_list), args: "", positional: 0,
             doc: "The agents the config offers to launch, and the ones goofi has running.",
             result: "{instances: [{id, harness, state, exit_code}], agents: [{name, command}], config_error: string | null}" }),
        Leaf(Op { name: "start", handler: Effect(arms::agent_start), args: "name:string!", positional: 1,
             doc: "Launch a config-listed agent on a PTY with the patch workspace as its cwd, under a login shell — a command that cannot launch fails on the PTY itself. Read its terminal at /term/<instance_id>. An unknown name is refused with the config's list.",
             result: "{instance_id: string}" }),
        Leaf(Op { name: "stop", handler: Effect(arms::agent_stop), args: "instance:string!", positional: 1,
             doc: "Stop a running agent (SIGTERM, then SIGKILL), or dismiss one that already exited. The shell's undo stack dies with it; the exit code arrives on harness_changed.",
             result: "{ok: true}" }),
    ]),
    Leaf(Op { name: "undo", handler: Effect(arms::undo), args: "", positional: 0,
         doc: "Undo this actor's last graph command. Each actor — a browser tab, a shell, the MCP — has its own stack.",
         result: "{changed: bool, can_undo: bool, can_redo: bool}" }),
    Leaf(Op { name: "redo", handler: Effect(arms::redo), args: "", positional: 0,
         doc: "Redo this actor's last undone graph command.",
         result: "{changed: bool, can_undo: bool, can_redo: bool}" }),
    Leaf(Op { name: "compound", handler: Effect(arms::compound), args: "ops:json!", positional: 0,
         doc: "Run several steps in order as ONE undo step and one settled decision: viewers see no intermediate document, and the unsaved dot moves once. `ops` is a list of `{op, payload}`; a step is a read or an undoable write, and an effect is refused — it runs as its own call. A refused step takes back the ones that already landed, so the call either happens whole or not at all.\n\n\
               A read step sees the earlier steps' writes on the GRAPH — but the document settles only when the batch does, so `session state` and `session status` inside a batch answer the document the batch found. Read them after it, not inside it.",
         result: "the steps' own replies, as a bare JSON list in order" }),
    Group("layout", "panels, tabs and splits — absent under headless", &[
        Leaf(Op { name: "inspect", handler: Read(arms::layout_inspect), args: "tab:string", positional: 1,
             doc: "The arrangement as a tree: every tab, split and panel with its id, order and share of its parent. How a caller discovers the ids every layout op addresses. `tab` narrows it to one tab; no arg = all of them.",
             result: "{text: string}" }),
        Group("panel", "one panel — the unit a viewer or tool lives in", &[
            Leaf(Op { name: "add", handler: Write(arms::layout_panel_add),
                 args: "beside:string side:string ratio:float name:string index:int", positional: 0,
                 doc: "A fresh empty panel. With `--beside` it divides that panel, on its `left`/`right`/`top`/`bottom` (`--side`, default right), taking `--ratio` of its space (default half). Bare, it lands on a new tab at `--index` in the strip, labelled `--name` — minted (`Tab 2`, `Tab 3`, …) unless you give one.",
                 result: "{id, tab, text} — the born panel, the tab it is on, and the arrangement as `layout inspect` draws it" }),
            Leaf(Op { name: "edit", handler: Write(arms::layout_panel_edit),
                 args: "panel:string! type:panel_type state:json", positional: 1,
                 doc: "Edit a PANEL's content: its type, its state, or both in one call and one undo. State MERGES key by key — send only what changes, and null to clear a key. A new type clears the old type's state, so send both together to rebind. `type` is one of: {panel_types}. A viewer panel's `state.kind` is one of: {viewer_kinds}; a STRING or TABLE slot ignores it and uses its own.",
                 result: "{text} — the resulting arrangement, as `layout inspect` draws it" }),
        ]),
        Leaf(Op { name: "move", handler: Write(arms::layout_move),
             args: "entry:string! beside:string side:string ratio:float in:string index:int name:string", positional: 1,
             doc: "Move a layout entry — a panel, a whole split's subtree, or a tab; one op per drag gesture, so a drop is one undo step. With `--beside` (and `--side`, `--ratio`) it lands beside that panel. With `--in` it lands inside that split, at `--index` among its children. Bare, a TAB moves to `--index` in the strip, and anything else wraps onto a tab of its own, labelled `--name`. Taking a tab's last panel takes the tab with it.",
             result: "{id, tab, text} — what was moved, the tab it is on, and the arrangement as `layout inspect` draws it" }),
        Leaf(Op { name: "remove", handler: Write(arms::layout_remove), args: "entry:string!", positional: 1,
             doc: "Close a layout entry: a panel, a whole split's subtree, or a tab and every panel on it. Its space goes to its siblings; a tab keeps its last panel, and the last tab stays.",
             result: "{text} — the resulting arrangement, as `layout inspect` draws it" }),
        Group("tab", "one tab in the strip", &[
            Leaf(Op { name: "edit", handler: Write(arms::layout_tab_edit),
                 args: "tab:string! name:string!", positional: 1,
                 doc: "Relabel a TAB. Its id and every panel on it stand; the strip index is where it sits, which `layout move` owns.",
                 result: "{text} — the resulting arrangement, as `layout inspect` draws it" }),
        ]),
        Group("split", "one split's children and their shares", &[
            Leaf(Op { name: "edit", handler: Write(arms::layout_split_edit),
                 args: "split:string! fraction:float[]!", positional: 1,
                 doc: "Set the shares of ALL of a SPLIT's children at once, in child order — what a resize drag commits. Renormalized to fill the slot.",
                 result: "{text} — the resulting arrangement, as `layout inspect` draws it" }),
        ]),
        Group("viewpoint", "where this client is looking", &[
            Leaf(Op { name: "edit", handler: Effect(arms::layout_viewpoint_edit),
                 args: "value:json!", positional: 0,
                 doc: "Store where this client is looking — active tab, maximize, camera, each panel's sub-patch path. ONE stored value, replaced whole, last writer wins; persisted in the `.gfi`, never converged, never dirtying.",
                 result: "{ok: true}" }),
        ]),
    ]),
];

/// The phrases the CLIENT owns — `serve`, the door words, and the future `plugin` prefix. Never
/// registrable, and prefix-free with the registry: the contracts invariant checks both together.
pub static RESERVED: &[&str] =
    &["serve", "help", "session list", "agent term", "plugin", "completions"];

/// The flat rows the tree spells, full phrases joined once and leaked once per process — what the
/// socket, `op list` and the generated `OpName` union read.
pub fn registry() -> &'static [Op] {
    use std::sync::OnceLock;
    static FLAT: OnceLock<Vec<Op>> = OnceLock::new();
    fn walk(prefix: &str, entries: &[Entry], out: &mut Vec<Op>) {
        for e in entries {
            match e {
                Entry::Group(word, _, children) => {
                    walk(&format!("{prefix}{word} "), children, out);
                }
                Entry::Leaf(op) => out.push(Op {
                    name: Box::leak(format!("{prefix}{}", op.name).into_boxed_str()),
                    handler: op.handler,
                    args: op.args,
                    positional: op.positional,
                    doc: op.doc,
                    result: op.result,
                }),
            }
        }
    }
    FLAT.get_or_init(|| {
        let mut out = Vec::new();
        walk("", TREE, &mut out);
        out
    })
}

/// The row for `name`, if the op exists.
pub fn find(name: &str) -> Option<&'static Op> {
    registry().iter().find(|o| o.name == name)
}

/// The rows one server serves. A mode does not REGISTER what it withholds — the one spelling of
/// each mode, so `op list`, the phrase resolver and the MCP all shrink with it.
pub fn table(mode: crate::Mode) -> Vec<&'static Op> {
    // What a demo drops: the host's filesystem, the agents it would spawn, and the two ops that
    // read or write a `.gfi` beside them. `session new` stays — it is the visitor's reset.
    const DEMO_DROPS: [&str; 4] = ["dir", "agent", "session save", "session load"];
    let dropped = |name: &str, group: &str| {
        name == group || name.strip_prefix(group).is_some_and(|rest| rest.starts_with(' '))
    };
    registry()
        .iter()
        .filter(|o| !mode.headless || !dropped(o.name, "layout"))
        .filter(|o| !mode.demo || !DEMO_DROPS.iter().any(|d| dropped(o.name, d)))
        .collect()
}

/// The frontend's `OpName` union, generated from the registry and checked into the tree.
pub fn typescript() -> String {
    let names: Vec<String> =
        registry().iter().map(|o| format!("\t| '{}'", o.name)).collect();
    format!(
        "// GENERATED from backend/goofi-bridge/src/ops.rs — do not edit by hand.\n\
         // The manager's op registry is the only place an op name is declared: naming one that is\n\
         // not in it is a type error here and an `unknown op` refusal there. Regenerate by running\n\
         // `cargo test -p goofi-bridge`, which rewrites this file when it drifts.\n\
         export type OpName =\n{};\n",
        names.join("\n")
    )
}
