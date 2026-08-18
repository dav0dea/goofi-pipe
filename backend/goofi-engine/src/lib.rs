//! goofi-engine — the graph and its tick scheduler.
//!
//! `tick()` walks the graph in topological LEVELS, running each level's nodes in
//! parallel (rayon) and then moving their outputs into their consumers' inputs
//! (latest-wins), so one pass propagates through an acyclic graph. Nodes land on
//! one of two execution tiers (see `make_exec`): inline, or detached onto an
//! off-tick worker. Each node's latest output frame is exposed for the data plane.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;

use goofi_core::Param;
use goofi_node::{
    ExprMode, NodeCtx, NodeManifest, ParamGroups, ParamKey,
};
use indexmap::IndexMap;

/// The `.gfi` zip container: pack and unpack (see `archive.rs`).
pub mod archive;

/// Sub-patch forest model + stub resolution (see `subpatch.rs`).
pub mod subpatch;
/// The flat, id-keyed editor arrangement (pages, splits, panels) — the fifth CRDT doc root.
pub mod layout;

/// Semantic patch commands with exact inverses — the manager's undo/redo unit.
pub mod command;
pub use command::{Command, CommandHistory, ExprState, Outcome};

/// The isolated-node execution tier (off-tick detached workers + latest-wins mailboxes).

/// The expression rewrite: an authored source becomes a variable-keyed one plus its variable map.
pub mod expr_rewrite;

/// The per-node runtime: the wake loop, the three run paths, and a node's faults (see `runtime/`).
/// Public because it is a standalone module today — nothing in [`Graph`] drives it yet, and the
/// cutover from `tick_at` is its own step.
pub mod runtime;
pub mod testing;

/// A stable node identity. Encoded as a 12-hex string for the `.gfi` / frontend
/// (the same key those use), a `u64` internally.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Uid(pub u64);

impl Uid {
    pub fn to_hex(self) -> String {
        format!("{:012x}", self.0)
    }
    /// Parse the canonical 12-hex identity `to_hex` writes, and nothing wider. Accepting any
    /// radix-16 `u64` let a hand-edited `.gfi` (or a non-browser control client) carry
    /// `ffffffffffffffff`, whose `+ 1` in `restore_uid` / `add_node_at` overflows — a panic under
    /// overflow checks, a corrupted `next_uid` in release. Bounding the DOMAIN is what makes that
    /// arithmetic total, rather than checking each site.
    pub fn from_hex(s: &str) -> Option<Uid> {
        if s.len() != 12 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        u64::from_str_radix(s, 16).ok().map(Uid)
    }
}

impl std::fmt::Display for Uid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// The `.gfi` manifest version: written by [`Graph::serialize`], the sole version
/// [`Graph::load_doc`] accepts, and the number its refusal quotes. One literal for all three, so a
/// bump cannot leave the error message lying about what this build actually reads.
const MANIFEST_VERSION: i64 = 7;

/// One node's manager-side thread, and the graph's end of its wires (§5).
///
/// A node is *known* the moment `add_node` answers and *addressable* only once it has published
/// [`runtime::Status::Ready`] — §4's birth barrier, which exists because pub/sub has no history and
/// a `Control` sent before the node's subscriber exists is simply lost.
struct NodeHost {
    /// What stops the thread. See `runtime::wire`'s note on why the stop is a flag rather than a
    /// `Control::Terminate`: a node removed before it was addressable has no sink to receive one.
    halt: Arc<runtime::Halt>,
    /// The control publisher, status subscriber and doorbell. `None` when this node's services could
    /// not be created — the node then exists in the patch carrying its boot error and nothing else.
    channel: Option<Arc<runtime::NodeChannel>>,
}

impl Drop for NodeHost {
    /// Removing a node stops its thread — and rings it, so a parked node notices now rather than at
    /// the end of its park. Never joined: the thread may be inside a long `process()`, and this runs
    /// under the graph mutex the bridge holds.
    fn drop(&mut self) {
        self.halt.stop();
        if let Some(channel) = &self.channel {
            channel.wake();
        }
    }
}

struct NodeEntry {
    manifest: &'static NodeManifest,
    host: NodeHost,
    /// The param RECORD — literals and, through `bindings`, expression source. Held behind an
    /// [`ArcSwap`] so the graph's readers never block on a write (spec §5.1). The node's own thread
    /// keeps its own copy, fed by the `SetParam` every write announces: an evaluated value must not
    /// reach this record, because this is what `serialize` writes.
    params: Arc<ArcSwap<ParamGroups>>,
    /// Param-expression bindings on this node, keyed by `(group, name)`. The graph resolves each
    /// one's references and ships it; the NODE evaluates it (spec §5.3).
    bindings: HashMap<ParamKey, ExprBinding>,
    /// The evaluated values of this node's bound params, as it last reported them
    /// ([`runtime::Status::ParamValues`]). Kept apart from `params`, which holds the literal
    /// RECORD — the number the user authored and the `.gfi` persists. `serialize` writes that
    /// record, so folding evaluated values into it would save whatever a since-deleted reference
    /// last happened to give the param, and leave nothing to fall back TO when the binding broke
    /// (§2.1).
    evaluated: IndexMap<ParamKey, Param>,
    /// Errors the node reported for params it holds no BINDING for — an `on_param_changed` that
    /// refused or panicked on a literal the user typed. Kept apart from `ExprBinding::error`
    /// because the two have different LIFETIMES, not because they mean different things: a
    /// binding's error may be a compile failure the graph itself recorded and a restart must
    /// preserve, while this one is only ever the running instance's report. Before the cutover it
    /// rode the `update_param` reply; the hook runs on the node's own thread now, so a
    /// binding-keyed projection alone drops it on the floor and the node draws healthy.
    param_errors: IndexMap<ParamKey, String>,
    /// `Some(msg)` when this node's INITIALIZATION failed — the `on_param_changed` replay and
    /// `setup()` together, which are one unit (D3). Deliberately NOT `last_error`, which a process
    /// failure overwrites — that is how a bootstrap failure used to erase itself on the first clean
    /// run.
    setup_error: Option<String>,
    last_error: Option<String>,
    /// The message [`Graph::last_error`] last derived for this node, and WHEN it first read that
    /// way. Re-stamped only when the message changes, so the instant is the error's *onset* — the
    /// difference between a pipeline settling and one that is broken. Derived from
    /// [`entry_error`] at every status application rather than written at each site an error can
    /// arise, so the two can never disagree about what the node's error is.
    error_since: Option<(String, Instant)>,
    /// The lifecycle stage the node last reported. `creating` until it reports anything — a node
    /// the graph has built and not yet heard from. The `error` the editor draws is DERIVED from the
    /// fault ([`Graph::node_stage`]) and is never stored here, so the two cannot disagree.
    stage: &'static str,
    /// The measured update rate the node last reported ([`runtime::Status::Ufreq`]), which is the
    /// same number it stamps as `meta["ufreq"]`. `None` until it has emitted twice.
    ufreq: Option<f64>,
    /// Globally-unique display name (type-numbered), for the frontend/`.gfi`.
    name: String,
    /// Editor position `[x, y]`.
    pos: [f64; 2],
    /// Per-slot viewer view-state (chosen kind + settings + collapsed), an OPAQUE JSON
    /// blob the backend persists and round-trips but never interprets — view-state is
    /// cross-cutting UI state, not pillar logic. Empty object until the editor sets it.
    viewers: serde_json::Value,
}

/// A resolved link (uids + `&'static` slot names), for snapshot projection.
#[derive(Clone, Copy, Debug)]
pub struct LinkView {
    pub node_out: Uid,
    pub slot_out: &'static str,
    pub node_in: Uid,
    pub slot_in: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Link {
    node_out: Uid,
    slot_out: &'static str,
    node_in: Uid,
    slot_in: &'static str,
}

/// Extract a readable message from a caught panic payload.
/// Run a node LIFECYCLE hook, converting a panic into an error string — the same trade
/// [`execute_node`] already makes for `process`, and for the same reason: a node is third-party
/// code (a native crate registered through `inventory`, or a `.py` the user just edited).
///
/// The difference that makes this load-bearing is WHERE these run. `process` executes on the tick
/// thread; `setup`, `on_param_changed` and `on_param_refreshed` execute under the graph mutex the
/// bridge is holding, and this codebase locks with `.lock().unwrap()` throughout. An unguarded
/// panic there poisons the mutex, so every subsequent lock in the bridge AND the tick thread
/// panics too — one node's bug becomes total, permanent loss of the control plane.
fn guard_lifecycle<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(panic_message)
}

/// Fold a caught lifecycle panic into the `NodeResult` the hook would have returned, so the two
/// failure modes travel the one channel a caller already handles.
fn fold_panic(panicked: String) -> goofi_node::NodeResult {
    Err(goofi_node::NodeError(panicked))
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = p.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic in node".to_string()
    }
}

/// The persisted scalar value of a param (flat form; triggers persist `false`).
fn param_value_json(p: &Param) -> serde_json::Value {
    use serde_json::json;
    match p {
        Param::Float { value, .. } => json!(value),
        Param::Int { value, .. } => json!(value),
        Param::Bool { value } => json!(value),
        Param::Trigger { .. } => json!(false),
        Param::Str { value, .. } => json!(value),
    }
}

/// Coerce a JSON scalar into a `Param` of `existing`'s type, preserving its bounds/options — the
/// inverse of [`param_value_json`], and the SSOT for the engine load path (`Graph::load_doc`)
/// and the bridge's RPC/CRDT param writes. An Int rounds a fractional value to nearest (rather than
/// zeroing it, which a hand-edited `.gfi` used to do). `fire_triggers` gates the Trigger arm: a live
/// UI edit passes `true` (the trigger button fires), a `.gfi` load passes `false` (a persisted or
/// hand-edited value must never trip a node's trigger on load).
pub fn param_from_json(existing: &Param, v: &serde_json::Value, fire_triggers: bool) -> Param {
    match existing {
        Param::Float { vmin, vmax, .. } => Param::Float { value: v.as_f64().unwrap_or(0.0), vmin: *vmin, vmax: *vmax },
        Param::Int { vmin, vmax, .. } => Param::Int {
            value: v.as_i64().or_else(|| v.as_f64().map(|f| f.round() as i64)).unwrap_or(0),
            vmin: *vmin,
            vmax: *vmax,
        },
        Param::Bool { .. } => Param::Bool { value: v.as_bool().unwrap_or(false) },
        Param::Trigger { .. } => Param::Trigger { fired: fire_triggers && v.as_bool().unwrap_or(false) },
        Param::Str { options, refresh, .. } => Param::Str {
            value: v.as_str().unwrap_or("").to_string(),
            options: options.clone(),
            refresh: *refresh,
        },
    }
}

/// A global's value as a `{value, type}` JSON object — the shape used in the `.gfi` and the CRDT
/// doc. The `type` tag preserves float-vs-int after JSON's whole-float normalization. SSOT reused by
/// the bridge's CRDT mirror (like [`param_from_json`]).
pub fn global_to_json(v: &goofi_core::globals::GlobalValue) -> serde_json::Value {
    use goofi_core::globals::GlobalValue;
    use serde_json::json;
    let value = match v {
        GlobalValue::Float(x) => json!(x),
        GlobalValue::Int(x) => json!(x),
        GlobalValue::Bool(x) => json!(x),
        GlobalValue::Str(s) => json!(s),
    };
    json!({ "value": value, "type": v.type_tag() })
}

/// Parse a `{value, type}` JSON entry into a [`goofi_core::globals::GlobalValue`] (type-directed);
/// `None` if malformed. Inverse of [`global_to_json`].
pub fn global_from_json(entry: &serde_json::Value) -> Option<goofi_core::globals::GlobalValue> {
    use goofi_core::globals::GlobalValue;
    let value = entry.get("value")?;
    match entry.get("type").and_then(|t| t.as_str())? {
        "float" => Some(GlobalValue::Float(value.as_f64()?)),
        "int" => Some(GlobalValue::Int(
            value.as_i64().or_else(|| value.as_f64().map(|f| f.round() as i64))?,
        )),
        "bool" => Some(GlobalValue::Bool(value.as_bool()?)),
        "string" => Some(GlobalValue::Str(value.as_str()?.to_string())),
        _ => None,
    }
}

/// A node factory that can capture runtime state (a Python class handle, a device
/// descriptor). Used for node types discovered at runtime rather than compiled
/// into the `inventory` catalog — a bare `fn` pointer can't close over such state.
/// One definition in goofi-node, shared with every discovery backend.
pub use goofi_node::discover::NodeFactory;

/// A runtime-registered node type: its (leaked-`'static`) manifest plus the
/// factory that builds instances of it. Its `manifest.factory` is never called.
struct DynType {
    manifest: &'static NodeManifest,
    factory: NodeFactory,
}

/// What one [`Graph::register_dyn_type`] call did to the runtime registry. The three are kept
/// apart because only the CALLER can read them: a rescan re-registers every type it finds, so
/// `Replaced` is an ordinary refresh there — while a boot scan starts from an empty registry, so
/// the same value can only mean two node files claiming one name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Registration {
    /// The name was free; the type entered the registry.
    Added,
    /// A runtime type of that name was already registered and has been replaced.
    Replaced,
    /// A built-in owns the name; the registry is unchanged.
    Refused,
}

/// A param bound to an expression (engine-side record; the node stays oblivious — the
/// engine writes the evaluated value into its params before it runs). See the
/// param-expressions design.
struct ExprBinding {
    /// The AUTHORED source — `nd('lfo').out * globals.gain`. This is the SSOT: it is what the
    /// `.gfi` stores, what the doc and inspector show, and what a rename edits. Everything below it
    /// is DERIVED and re-derived whenever it, or a name the graph resolves, changes (spec §5.3).
    source: String,
    enabled: bool,
    triggers_process: bool,
    /// Compiled handle owned by the evaluator (`None` if compile failed / no evaluator). Compiled
    /// from [`Self::rewritten`], never from [`Self::source`] — the node's evaluator is handed
    /// variables, not names.
    id: Option<goofi_node::BindingId>,
    /// Derived: `source` with every reference replaced by a generated variable.
    rewritten: String,
    /// Derived: one entry per variable `rewritten` names, resolved against the graph.
    vars: Vec<BoundVar>,
    /// Derived: the rewrite's own variable list, BEFORE resolution. Kept beside `vars` because a
    /// variable that failed to resolve no longer says what it was looking for — and those are
    /// exactly the bindings a node being added, or a global being defined, has to re-resolve. One
    /// record rather than a name list and a key list beside it: both are questions about this.
    terms: Vec<expr_rewrite::VarRef>,
    /// This binding's identity in the wire planner, stable across a rebind — its index into
    /// [`Graph::bind_keys`].
    bind_id: usize,
    /// The current expression error (field indicator), or `None` when healthy.
    error: Option<String>,
}

/// One resolved expression variable, graph-side (spec §5.3). The wire projection
/// ([`runtime::Var`]) drops the uid and keeps the service name, because a node addresses a producer
/// by service and never by uid; the graph keeps the uid because it is what re-planning is keyed on.
#[derive(Clone, Debug)]
enum BoundVar {
    /// A producer's output slot, and the doorbell id the producer rings this consumer with. §3.2
    /// budgets `65..=128` for expression channels, so a node may hold at most 64 of them.
    Stream { var: String, producer: Uid, slot: &'static str, event_id: runtime::EventId },
    /// A `globals.*` read, resolved and shipped inline — a globals edit re-sends the binding.
    Value { var: String, value: Param },
    /// The graph could not resolve it: an unknown node, a slot that does not exist, an ambiguous
    /// bare `nd()` on a multi-output producer, a global that is not defined.
    Missing { var: String, reason: String },
}

impl BoundVar {
    /// The producer wire this variable subscribes to, if it resolved to one.
    fn wire(&self) -> Option<runtime::plan::Wire> {
        match self {
            BoundVar::Stream { producer, slot, .. } => Some((*producer, *slot)),
            _ => None,
        }
    }
}

/// A param's expression binding, projected for the bridge/`.gfi` (the internal
/// [`ExprBinding`] is private). `error` drives the per-param field indicator.
pub struct ExprInfo {
    pub source: String,
    pub enabled: bool,
    pub triggers_process: bool,
    pub error: Option<String>,
}

/// The authoritative graph + scheduler.
pub struct Graph {
    nodes: IndexMap<Uid, NodeEntry>,
    links: Vec<Link>,
    next_uid: u64,
    /// Node types registered at runtime (e.g. discovered Python nodes), keyed by
    /// type name. Survives `clear()`/`load_doc` — these are catalog, not content.
    dyn_types: HashMap<&'static str, DynType>,
    /// The editor's panel arrangement, held FLAT and interpreted — the fifth CRDT doc root. Every
    /// mutation is an ordinary command over it, which is what ends the frontend's parallel write
    /// authority: there is exactly ONE projection of the layout, as there is for nodes and links.
    arrangement: layout::Layout,
    /// Why a stored arrangement was refused, if it was — read once by the load reply so a fallback
    /// to the default is stated rather than silent. Cleared by every load.
    arrangement_warning: Option<String>,
    /// Where a client is LOOKING: active page, panel maximize, editor camera, and each panel's
    /// sub-patch path. Opaque and per-client, so it is deliberately NOT a doc root (converging it
    /// would drag peers and dirty the patch on mere navigation) — but persistence is the other
    /// axis, so it rides the `.gfi` and the snapshot all the same.
    viewpoint: serde_json::Value,
    /// Node types that EXIST on disk but cannot load here, keyed by type name → reason (a
    /// missing module name, or the exception line). They appear in the palette, greyed, so a
    /// node that needs an uninstalled dependency explains itself instead of silently not
    /// existing. Catalog-only: they can never be instantiated.
    unavailable: std::collections::BTreeMap<String, String>,
    /// The runtime types that came from the open patch's own workspace rather than the shipped
    /// node directory — the palette's provenance badge, and the one thing about a type that only
    /// the scan can know. Re-derived wholesale by each scan (see [`Graph::set_patch_types`]).
    patch_types: std::collections::HashSet<String>,
    /// Wall-clock reference, anchored when the patch begins, so every node's `NodeCtx::now` is
    /// seconds-since-start — one clock across every node thread rather than one per birth.
    start: Instant,
    /// The injected param-expression evaluator (pyo3, from goofi-python). `None` → bindings
    /// are stored + round-trip but can't evaluate (graceful degrade to the literal).
    evaluator: Option<Arc<dyn goofi_node::ExprEvaluator>>,
    /// ── Sub-patch scopes: a purely organizational overlay over the flat `nodes`/`links`. A scope
    /// references its member nodes (via `scope_of`) + holds boundary stubs; the members stay live +
    /// flat. Empty ⇒ a plain flat graph. Keyed by the scope uid (== its collapsed facade node's uid).
    scopes: IndexMap<Uid, subpatch::Scope>,
    /// node/scope uid → its parent scope (`None`/absent = ROOT). The ONE tree SSOT for parentage +
    /// membership; an ordinary flat graph needs no entries.
    scope_of: HashMap<Uid, Option<Uid>>,
    /// Patch-scoped globals (system + user). System globals are seeded here; a `clear`/load
    /// re-asserts them. Read by param expressions + node setup/process; persisted to `.gfi`.
    globals: goofi_core::globals::GlobalStore,
    /// The same globals as the node threads read them (§5.2) — the store is the graph's writable
    /// record, this is the lock-free view every node holds a handle to. Re-published by every
    /// mutator, which is why `globals` is written ONLY through [`Graph::globals_mut`].
    globals_record: Arc<ArcSwap<goofi_core::globals::GlobalsSnapshot>>,
    /// The async runtime's wire plane: each live node's control channel, the per-slot sequence in
    /// flight, and every uid's birth generation.
    wire: runtime::plan::WirePlanner,
    /// Every `(node, param)` this graph has ever bound an expression to, so the wire planner can
    /// name a binding by index (`Slot::Bind`) and keep a `Copy` key. Append-only within a patch:
    /// an unbind's own three-phase sequence still has to compose the `SetParam` that announces it,
    /// and by then the binding is gone. `clear` resets it with the rest of the patch.
    bind_keys: Vec<(Uid, ParamKey)>,
    /// What this graph's service names are scoped by — random, not the bridge's instance id: a
    /// service name has to be unique on the MACHINE, across this process's own graphs and across
    /// every stale record a previous run left behind.
    instance: String,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Write a node's param record: load, clone, mutate, swap. Copy-on-write because the record is
/// SHARED with the node's own thread (§5.1) — a reader mid-`process` keeps the version it loaded,
/// and the next load sees the whole edit or none of it. Param edits are user-paced, so the clone
/// costs nothing that matters.
fn edit_params(entry: &NodeEntry, edit: impl FnOnce(&mut ParamGroups)) {
    let mut next = (*entry.params.load_full()).clone();
    edit(&mut next);
    entry.params.store(Arc::new(next));
}

/// A global's value as the [`Param`] an expression variable carries (§5.3: the graph resolves a
/// global and ships it inline). The numeric bounds are a carrier's, not a control's — nothing
/// clamps a local, and the evaluator coerces the RESULT to the target param's own type and range.
fn global_as_param(value: &goofi_core::globals::GlobalValue) -> Param {
    use goofi_core::globals::GlobalValue as G;
    match value {
        G::Float(v) => Param::float(*v, f64::NEG_INFINITY, f64::INFINITY),
        G::Int(v) => Param::int(*v, i64::MIN, i64::MAX),
        G::Bool(v) => Param::boolean(*v),
        G::Str(v) => Param::str_free(v.clone()),
    }
}

/// The lowest free doorbell id in §3.2's expression range, or `None` when a node has spent all 64.
fn next_event_id(taken: &[runtime::EventId]) -> Option<runtime::EventId> {
    (65..=128).find(|id| !taken.contains(id))
}

impl Graph {
    pub fn new() -> Graph {
        // Reference goofi-nodes so the linker keeps its inventory registrations.
        let _ = goofi_nodes::native_node_count();
        Graph {
            nodes: IndexMap::new(),
            links: Vec::new(),
            next_uid: 1,
            dyn_types: HashMap::new(),
            unavailable: std::collections::BTreeMap::new(),
            patch_types: std::collections::HashSet::new(),
            arrangement: layout::Layout::default(),
            arrangement_warning: None,
            viewpoint: serde_json::Value::Null,
            start: Instant::now(),
            evaluator: None,
            scopes: IndexMap::new(),
            scope_of: HashMap::new(),
            globals: goofi_core::globals::GlobalStore::new(),
            globals_record: Arc::new(ArcSwap::from_pointee(
                goofi_core::globals::GlobalStore::new().snapshot(),
            )),
            wire: runtime::plan::WirePlanner::default(),
            bind_keys: Vec::new(),
            instance: runtime::service_instance(),
        }
    }

    // ── Globals ─────────────────────────────────────────────────────────────────────────────
    // Patch-scoped named scalars. System globals (`default_ufreq`) are seeded + delete-protected;
    // user globals are add/edit/remove/rename. Read by expressions (`globals.<name>`) + node ctx.

    /// The authoritative globals store — its `entries()`/`snapshot()` serve the CRDT mirror, the
    /// `.gfi`, and (via `snapshot()`) expression eval + node setup/process.
    pub fn globals(&self) -> &goofi_core::globals::GlobalStore {
        &self.globals
    }

    /// Apply one mirrored client global change (`Some` = set/add, `None` = remove). System deletes
    /// are rejected. Every expression binding that READS this global is re-resolved and re-sent, so
    /// a producer bound to `globals.default_ufreq` re-rates live — and only those bindings pay (an
    /// unrelated global edit touches nothing). §5.2: there is no invalidation message, because the
    /// graph resolves a global's value and ships it inline.
    pub fn apply_global_change(
        &mut self,
        name: &str,
        value: Option<goofi_core::globals::GlobalValue>,
    ) -> Result<(), String> {
        self.globals_mut(|g| g.apply_change(name, value))?;
        self.invalidate_bindings_reading(name);
        Ok(())
    }

    /// Re-add a previously-removed user global at its ORIGINAL ordered position — the
    /// position-preserving inverse of a delete/rename (order feeds the `.gfi`, mirror, and panel).
    pub fn insert_global_at(
        &mut self,
        name: &str,
        value: goofi_core::globals::GlobalValue,
        at: usize,
    ) -> Result<(), String> {
        self.globals_mut(|g| g.add_at(name, value, at))?;
        self.invalidate_bindings_reading(name);
        Ok(())
    }

    /// Re-resolve and re-send every expression binding that reads global `name`, so its new value
    /// reaches the nodes reading it (only those bindings pay). Shared by the global mutators.
    fn invalidate_bindings_reading(&mut self, name: &str) {
        let reading = self.bindings_where(|b| {
            b.terms.iter().any(|t| matches!(t, expr_rewrite::VarRef::Global { key, .. } if key == name))
        });
        self.rebind(&reading);
    }

    /// Re-resolve and re-send every binding whose source references the node display name `name`.
    /// §5.3's "renamed, added, removed or restarted", stated once — what all four have in common is
    /// that a NAME started or stopped meaning what it did, and the authored source is written
    /// against names.
    fn rebind_naming(&mut self, name: &str) {
        let naming = self.bindings_where(|b| {
            b.terms.iter().any(|t| matches!(t, expr_rewrite::VarRef::Node { name: n, .. } if n == name))
        });
        self.rebind(&naming);
    }

    /// Every binding matching a predicate, as `(node, param)` — the addressing `rebind` takes.
    fn bindings_where(&self, want: impl Fn(&ExprBinding) -> bool) -> Vec<(Uid, ParamKey)> {
        self.nodes
            .iter()
            .flat_map(|(uid, e)| e.bindings.iter().map(move |(k, b)| (*uid, k.clone(), b)))
            .filter(|(_, _, b)| want(b))
            .map(|(uid, key, _)| (uid, key))
            .collect()
    }

    /// Re-run `set_expression` on each of these bindings from its AUTHORED source — the one
    /// operation that re-derives everything a resolution depends on (the rewrite, the variables,
    /// the compiled handle, the wire plan). Every "the graph changed under a binding" path funnels
    /// here rather than patching a resolved field in place, so there is one re-resolution.
    fn rebind(&mut self, bindings: &[(Uid, ParamKey)]) {
        for (uid, key) in bindings {
            let Some(b) = self.nodes.get(uid).and_then(|e| e.bindings.get(key)) else { continue };
            let (source, enabled, triggers) = (b.source.clone(), b.enabled, b.triggers_process);
            let _ = self.set_expression(*uid, &key.group, &key.name, &source, enabled, triggers);
        }
    }

    /// Inject the param-expression evaluator (pyo3, from goofi-python). Wired by the CLI at
    /// startup; without it, expression bindings are stored but not evaluated.
    pub fn set_evaluator(&mut self, evaluator: Arc<dyn goofi_node::ExprEvaluator>) {
        self.evaluator = Some(evaluator);
    }

    /// Register a node type discovered at runtime. `manifest` must be `'static`
    /// (runtime types leak one manifest per type — bounded, catalog-lifetime); its
    /// `make` field is unused (instances come from `factory`).
    ///
    /// A name that collides with a built-in catalog type is refused (with a warning): a built-in
    /// always wins `add_node`/`load_doc` resolution, so a runtime type of that name could never be
    /// reached. A name held by another RUNTIME type is **replaced**, because a rescan re-registers
    /// every type it finds — refusing would make the second scan a silent no-op. The loser's
    /// manifest stays leaked (the accepted price of `&'static` manifests) and LIVE INSTANCES of it
    /// keep running: an entry owns its own manifest + node, so only the NEXT instance is built from
    /// the new factory.
    ///
    /// A replace is deliberately silent here — see [`Registration`]: the engine cannot tell a
    /// rescan's refresh from a boot-time name collision, and the caller can.
    pub fn register_dyn_type(
        &mut self,
        manifest: &'static NodeManifest,
        factory: NodeFactory,
    ) -> Registration {
        let name = manifest.type_name;
        if goofi_node::find(name).is_some() {
            eprintln!("warning: runtime node type `{name}` collides with a built-in; ignoring it");
            return Registration::Refused;
        }
        // A name that loads now is not unloadable any more: `unavailable` had no removal, so a
        // rescan after a `pip install` would otherwise leave the greyed row standing beside the
        // working type — two palette rows for one name.
        self.unavailable.remove(name);
        match self.dyn_types.insert(name, DynType { manifest, factory }) {
            Some(_) => Registration::Replaced,
            None => Registration::Added,
        }
    }

    /// Forget a runtime type ENTIRELY — a rescan whose file has vanished. ONE door for both
    /// registries, because that caller knows only that the file is gone, not which of the two the
    /// last scan put it in. Returns whether anything was removed (so it can report the diff).
    /// LIVE INSTANCES ARE UNTOUCHED, for the same reason a replace leaves them alone: removal only
    /// stops the NEXT `add_node` and the `.gfi` load gate.
    pub fn remove_dyn_type(&mut self, type_name: &str) -> bool {
        let had_dyn = self.dyn_types.remove(type_name).is_some();
        self.unavailable.remove(type_name).is_some() || had_dyn
    }

    /// Whether a type name resolves to either the compile-time catalog or a
    /// runtime-registered type.
    fn known_type(&self, type_name: &str) -> bool {
        goofi_node::find(type_name).is_some() || self.dyn_types.contains_key(type_name)
    }

    /// The refusal message for a type `known_type` rejects — the ONE phrasing, shared by
    /// `build_node` and the `.gfi` load gate so they cannot word the same rejection two ways. An
    /// unavailable type names its missing dependency; anything else reads as a typo, which is what
    /// it is. (`unavailable` is deliberately outside `known_type` — that is what makes such a type
    /// unaddable — so this is the only place the two registries meet.)
    fn reject_type(&self, type_name: &str) -> String {
        match self.unavailable.get(type_name) {
            Some(reason) => format!("node type `{type_name}` is unavailable: {reason}"),
            None => format!("unknown node type `{type_name}`"),
        }
    }

    /// A node's lifecycle stage for the editor: `creating` / `setup` / `ready` / `error`.
    ///
    /// `creating` is the graph's own — a node it has built and not yet heard from; `setup` and
    /// `ready` are the node's, reported as it passes them (§6.2). `error` is DERIVED from the
    /// fault rather than stored, so a node cannot report itself healthy while carrying one — and
    /// the uninitialized state is precisely the one the editor must not paint healthy.
    pub fn node_stage(&self, uid: Uid) -> &'static str {
        let Some(entry) = self.nodes.get(&uid) else { return "error" };
        if entry.setup_error.is_some() || entry.last_error.is_some() {
            return "error";
        }
        entry.stage
    }

    /// The node's current measured update frequency (Hz) — the same value it stamps as
    /// `meta["ufreq"]` on its output, as it last reported it ([`runtime::Status::Ufreq`]). `None`
    /// until it has been measured (≥2 emits).
    pub fn node_ufreq(&self, uid: Uid) -> Option<f64> {
        self.nodes.get(&uid).and_then(|e| e.ufreq)
    }

    /// The flat arrangement — pages, splits and panels. Reads plan against this; writes go through
    /// a command, so undo/redo and the CRDT mirror come for free.
    pub fn arrangement(&self) -> &layout::Layout {
        &self.arrangement
    }

    /// The one write door, held by [`command::Command::EditLayoutEntry`] and by a load.
    pub fn arrangement_mut(&mut self) -> &mut layout::Layout {
        &mut self.arrangement
    }

    /// Why the last load fell back to the default arrangement, if it did.
    pub fn arrangement_warning(&self) -> Option<&str> {
        self.arrangement_warning.as_deref()
    }

    /// The client-local viewpoint blob (see the field).
    pub fn viewpoint(&self) -> &serde_json::Value {
        &self.viewpoint
    }

    pub fn set_viewpoint(&mut self, viewpoint: serde_json::Value) {
        self.viewpoint = viewpoint;
    }

    /// Record a node type that could not be loaded, with the reason. Refused if a BUILT-IN owns the
    /// name — a compiled-in node always wins resolution, so a broken file of that name could never
    /// be reached. A runtime type of the same name is displaced, the mirror of the clearing
    /// `register_dyn_type` does: both registries answer "what is on disk under this name", and the
    /// latest scan of that name is the answer.
    pub fn register_unavailable(&mut self, type_name: String, reason: String) -> bool {
        if goofi_node::find(&type_name).is_some() {
            return false;
        }
        self.dyn_types.remove(type_name.as_str());
        self.unavailable.insert(type_name, reason);
        true
    }

    /// The unloadable types, `(type_name, reason)`, sorted by name.
    pub fn unavailable_types(&self) -> impl Iterator<Item = (&str, &str)> {
        self.unavailable.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Declare which runtime types came from the open patch's own workspace rather than the shipped
    /// node directory — the palette's provenance badge. Written WHOLESALE by the scan, because the
    /// scan is the only thing that knows which directory a type came from, and a rescan re-derives
    /// the answer for every name at once.
    pub fn set_patch_types(&mut self, names: std::collections::HashSet<String>) {
        self.patch_types = names;
    }

    /// Whether `type_name` came from the open patch (see [`Graph::set_patch_types`]). Everything
    /// else — built-ins and the shipped node directory alike — reads as shipped.
    pub fn is_patch_type(&self, type_name: &str) -> bool {
        self.patch_types.contains(type_name)
    }

    /// The manifests of all runtime-registered node types, sorted by type name
    /// (the compile-time catalog is enumerated separately via `goofi_node::catalog`).
    /// Used by the bridge to include runtime types in the editor palette.
    pub fn dyn_type_manifests(&self) -> Vec<&'static NodeManifest> {
        let mut ms: Vec<&'static NodeManifest> =
            self.dyn_types.values().map(|dt| dt.manifest).collect();
        ms.sort_by_key(|m| m.type_name);
        ms
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains(&self, uid: Uid) -> bool {
        self.nodes.contains_key(&uid)
    }

    /// Node uids in insertion order.
    pub fn node_uids(&self) -> Vec<Uid> {
        self.nodes.keys().copied().collect()
    }

    pub fn type_name(&self, uid: Uid) -> Option<&'static str> {
        self.nodes.get(&uid).map(|e| e.manifest.type_name)
    }

    pub fn manifest(&self, uid: Uid) -> Option<&'static NodeManifest> {
        self.nodes.get(&uid).map(|e| e.manifest)
    }

    /// The node's current error, derived fresh on read so recovery is always surfaced.
    /// A detached worker's bootstrap failure wins, then a process error (`last_error`), then the
    /// errored expression binding with the smallest `ParamKey` — a deterministic pick, since
    /// `bindings` is a `HashMap` whose iteration order is randomized. Deriving on read (rather
    /// than caching into `last_error`) means a binding that recovers on a node that never runs
    /// again still clears, and the channels can't drift apart.
    pub fn last_error(&self, uid: Uid) -> Option<&str> {
        entry_error(self.nodes.get(&uid)?)
    }

    /// How long this node's CURRENT error has been standing, or `None` when it is healthy. The
    /// clock restarts when the message changes, so a node cycling through different failures
    /// always reads young — which is exactly the signal a reader wants.
    pub fn error_age(&self, uid: Uid) -> Option<Duration> {
        let (_, since) = self.nodes.get(&uid)?.error_since.as_ref()?;
        Some(since.elapsed())
    }

    fn mint(&mut self) -> Uid {
        let u = Uid(self.next_uid);
        self.next_uid += 1;
        u
    }

    /// The uid a loaded record restores at: the one the archive named — a `.gfi`'s node and scope
    /// KEYS have always been uid hex — unless that key is unreadable or already `claimed` by an
    /// earlier record of this same load, in which case it mints a fresh one so an odd file still
    /// opens. `next_uid` is advanced past every restored uid, or the next `add_node` would mint
    /// straight onto a node the load just brought back.
    ///
    /// Restoring rather than reminting is what makes a load a restore of IDENTITY. Everything keyed
    /// by uid that the load does not itself remap depends on it: a viewer panel's `state.node`, an
    /// editor panel's `subpatchPath`, the viewpoint. Reminting repointed all of them — and only
    /// ever in an instance that had already held nodes, since a load into a fresh one renumbers to
    /// the very values it saved.
    fn restore_uid(&mut self, key: &str, claimed: &HashSet<Uid>) -> Uid {
        match Uid::from_hex(key).filter(|u| !claimed.contains(u)) {
            Some(u) => {
                // `from_hex` admits only the 48-bit canonical domain, so `+ 1` cannot overflow.
                self.next_uid = self.next_uid.max(u.0 + 1);
                u
            }
            None => self.mint(),
        }
    }

    /// The manifest for `type_name` — the compile-time catalog, else a runtime-registered type.
    fn manifest_of(&self, type_name: &str) -> Result<&'static NodeManifest, String> {
        goofi_node::find(type_name)
            .or_else(|| self.dyn_types.get(type_name).map(|dt| dt.manifest))
            .ok_or_else(|| self.reject_type(type_name))
    }

    /// The params a fresh instance of `type_name` starts from, resolved WITHOUT constructing the
    /// node. The `.gfi` load path needs these first: it folds the saved values in before building,
    /// so `setup()` sees what the user saved rather than the type's defaults.
    fn default_params_of(&self, type_name: &str) -> Result<ParamGroups, String> {
        let m = self.manifest_of(type_name)?;
        Ok(goofi_node::with_common(m.default_params(), m))
    }

    /// Construct (but do not insert) a node by type name — the shared front half of `add_node` /
    /// `add_node_at`. Resolves the compile-time catalog or a runtime-registered type and builds its
    /// params (defaulting to the type's defaults).
    fn build_node(
        &self,
        type_name: &str,
        params: Option<ParamGroups>,
    ) -> Result<(&'static NodeManifest, ParamGroups, Box<dyn goofi_node::Node>), String> {
        let p = match params {
            // Supplied params still get the `common` group NORMALIZED, since a caller may hand
            // over a partial group (an MCP `add_node` payload, a hand-edited `.gfi`) — and the
            // type is what decides a missing key's default. Not a load-path concern: `serialize`
            // writes every param unconditionally, so a saved patch has all three keys and its own
            // values win here.
            Some(p) => goofi_node::with_common(p, self.manifest_of(type_name)?),
            None => self.default_params_of(type_name)?,
        };
        if let Some(m) = goofi_node::find(type_name) {
            Ok((m, p, (m.factory)()))
        } else if let Some(dt) = self.dyn_types.get(type_name) {
            let n = (dt.factory)(&p);
            Ok((dt.manifest, p, n))
        } else {
            Err(self.reject_type(type_name))
        }
    }

    /// Instantiate a node by type name (compile-time catalog or a
    /// runtime-registered type). `params` defaults to the type's defaults.
    pub fn add_node(
        &mut self,
        type_name: &str,
        params: Option<ParamGroups>,
    ) -> Result<Uid, String> {
        // A fresh mint + an empty name is exactly what `add_node_at` treats as "pick them for me",
        // so the two paths share one body. (An unknown type now burns the minted uid — harmless:
        // uids are u64 and never user-visible.)
        let uid = self.mint();
        self.add_node_at(type_name, params, uid, "")
    }

    /// Instantiate a node at a SPECIFIC uid + display name — the undo/redo restoration path, so
    /// uid-keyed links and panels reconnect to the same node (redo-of-add, undo-of-delete). The uid
    /// must be free; `next_uid` is advanced past it so a later mint can never collide. A requested
    /// name already in use falls back to a fresh unique name (the uniqueness invariant wins).
    pub fn add_node_at(
        &mut self,
        type_name: &str,
        params: Option<ParamGroups>,
        uid: Uid,
        name: &str,
    ) -> Result<Uid, String> {
        if self.contains(uid) {
            return Err(format!("add_node_at: uid {} already in use", uid.to_hex()));
        }
        let params_arg_was_none = params.is_none();
        let (manifest, params, node) = self.build_node(type_name, params)?;
        let name = if name.is_empty() || self.name_in_use(name) {
            self.fresh_name(&manifest.type_name.to_lowercase())
        } else {
            name.to_string()
        };
        let seed = params_arg_was_none;
        let born = name.clone();
        self.insert_node_at(uid, name, manifest, node, params);
        if uid.0 >= self.next_uid {
            self.next_uid = uid.0 + 1;
        }
        if seed {
            self.seed_default_expressions(uid, manifest);
        }
        // A name that meant nothing a moment ago now names a producer (§5.3). This also covers
        // undo-of-delete, which is how a binding survives its reference being deleted and restored.
        self.rebind_naming(&born);
        Ok(uid)
    }

    /// Seed an expression binding for each of the type's declared `expression` params — the
    /// fresh-add analogue of a literal default. The declaration decides whether that binding starts
    /// live: an `ExprMode::Off` expression is *carried*, stored so the inspector's fx toggle has a
    /// source to turn on while the `spec` literal stands. Skipped entirely without an evaluator (the
    /// literal is the graceful fallback, never an errored "no evaluator" binding). Only fresh adds
    /// (`params == None`) call this; a restore/load supplies explicit params + its own captured
    /// expressions.
    fn seed_default_expressions(&mut self, uid: Uid, manifest: &'static NodeManifest) {
        if self.evaluator.is_none() {
            return;
        }
        // The manifest's own declarations, then the universal `common` group — and the node's own
        // win, exactly as `with_common`'s `or_insert_with` makes them win on the VALUE side. A node
        // that declares a `common` param has said what it means by it; nothing here overwrites a
        // manifest's param definition. Read through `common_decls`, which is the one place
        // `producer` is interpreted, so the two halves cannot disagree about who is a producer.
        let declared = manifest.params.iter().map(|d| (d.group, d.name, d.expression));
        let universal = goofi_node::common_decls(manifest)
            .filter(|d| !manifest.params.iter().any(|o| o.group == d.group && o.name == d.name))
            .map(|d| (d.group, d.name, d.expression))
            .collect::<Vec<_>>();
        for (group, name, expression) in declared.chain(universal) {
            if let Some(e) = expression {
                let enabled = matches!(e.mode, ExprMode::On);
                let _ = self.set_expression(uid, group, name, e.source, enabled, e.trigger);
            }
        }
    }

    /// Insert a constructed node at a SPECIFIC uid + display name — the reconcile path, which
    /// spawns sub-patch members at their deterministic uids. The uid must be free.
    ///
    /// This is where a node gets its manager-side thread (§5). The transport is created HERE rather
    /// than on that thread because it is the one step whose failure has nowhere to be reported to —
    /// without services there is no status service to carry a fault — so it becomes a
    /// [`runtime::NodeFault::Boot`] on the entry directly. Everything after it, `setup()` included,
    /// runs on the node's own thread and off the graph lock.
    fn insert_node_at(
        &mut self,
        uid: Uid,
        name: String,
        manifest: &'static NodeManifest,
        node: Box<dyn goofi_node::Node>,
        params: ParamGroups,
    ) {
        // This IS the birth §3.1 counts, whichever door it came through — a fresh add, a restart,
        // an undo of a delete, a load.
        let generation = self.wire.bump_generation(uid);
        let (host, boot_error) = self.spawn_host(uid, generation, manifest, node, &params);
        self.nodes.insert(
            uid,
            NodeEntry {
                manifest,
                host,
                params: Arc::new(ArcSwap::from_pointee(params)),
                bindings: HashMap::new(),
                evaluated: IndexMap::new(),
                param_errors: IndexMap::new(),
                setup_error: boot_error,
                last_error: None,
                error_since: None,
                stage: "creating",
                ufreq: None,
                name,
                pos: [0.0, 0.0],
                viewers: serde_json::json!({}),
            },
        );
    }

    /// Create one node's services, open the graph's end of them, and start its thread.
    ///
    /// Answers the host and the boot error, if any. A node whose services could not be created is
    /// still INSERTED — it holds its place in the patch, its links and its params, and says why it
    /// is not running — rather than failing an `add_node` the user cannot act on.
    fn spawn_host(
        &self,
        uid: Uid,
        generation: u64,
        manifest: &'static NodeManifest,
        node: Box<dyn goofi_node::Node>,
        params: &ParamGroups,
    ) -> (NodeHost, Option<String>) {
        let halt = Arc::new(runtime::Halt::default());
        let base = runtime::service_base(&self.instance, uid, generation);
        let started = runtime::IoxTransport::create(&self.instance, uid, generation, manifest)
            .and_then(|transport| Ok((transport, runtime::NodeChannel::open(&base)?)))
            .and_then(|(transport, channel)| {
                let env = runtime::NodeEnv {
                    evaluator: self.evaluator.clone(),
                    globals: self.globals_record.clone(),
                    started: self.start,
                };
                // The join handle is dropped on purpose: a node's thread is stopped by its `Halt`
                // and reaped by the OS, and holding one would tempt a caller into joining under
                // the graph mutex while the node is inside a long `process()`.
                runtime::spawn(manifest, node, params.clone(), Arc::new(transport), env, halt.clone())
                    .map(|_| channel)
                    .map_err(|e| format!("could not start the node's thread: {e}"))
            });
        match started {
            Ok(channel) => (NodeHost { halt, channel: Some(Arc::new(channel)) }, None),
            Err(e) => (NodeHost { halt, channel: None }, Some(e)),
        }
    }

    /// Whether a display name is taken by any live leaf node OR sub-patch scope facade. The two
    /// share one display-name namespace (a scope facade renders as a node), so uniqueness must span
    /// both — else a leaf renamed onto a scope's `subpatch{N}` name would collide on the canvas.
    fn name_in_use(&self, name: &str) -> bool {
        self.nodes.values().any(|e| e.name == name) || self.scopes.values().any(|s| s.name == name)
    }

    /// Is `name` already a display name of a node OR scope facade OTHER than `except`? The bridge
    /// pre-validates a forward rename with this: `Command::EditNode` tolerates a rename collision as
    /// a no-op (so a stale undo-replay converges instead of wedging the stack), so the user-facing
    /// duplicate-name error must be raised up front at the RPC boundary.
    pub fn name_taken(&self, name: &str, except: Uid) -> bool {
        self.nodes.iter().any(|(u, e)| *u != except && e.name == name)
            || self.scopes.iter().any(|(u, s)| *u != except && s.name == name)
    }

    /// Lowest `{base}{N}` display name not already in use (globally unique).
    fn fresh_name(&self, base: &str) -> String {
        for n in 0.. {
            let cand = format!("{base}{n}");
            if !self.name_in_use(&cand) {
                return cand;
            }
        }
        unreachable!()
    }

    /// A display name for a freshly-minted sub-patch instance. Prefers the dense `subpatch{uid}`
    /// form, but falls back to a uniqueness-checked `fresh_name` if a leaf was already renamed onto
    /// it (leaves + instances share one display-name namespace — a collision would collapse two
    /// members onto one local key on a later group and orphan one). The instance isn't inserted yet,
    /// so it can't self-collide.
    fn mint_subpatch_name(&self, uid: Uid) -> String {
        let base = format!("subpatch{}", uid.0);
        if self.name_in_use(&base) {
            self.fresh_name("subpatch")
        } else {
            base
        }
    }

    /// Display name of a node OR a scope facade (a collapsed sub-patch instance). Uniform so
    /// `EditNode` reads either through one seam.
    pub fn name(&self, uid: Uid) -> Option<&str> {
        self.nodes
            .get(&uid)
            .map(|e| e.name.as_str())
            .or_else(|| self.scopes.get(&uid).map(|s| s.name.as_str()))
    }

    /// Position of a node OR a scope facade (whose pos lives in `scopes[uid].pos`, not a live node).
    pub fn pos(&self, uid: Uid) -> Option<[f64; 2]> {
        self.nodes
            .get(&uid)
            .map(|e| e.pos)
            .or_else(|| self.scopes.get(&uid).map(|s| s.pos))
    }

    /// A node's params as of now. An owned snapshot rather than a borrow, because the record is an
    /// [`ArcSwap`] a node thread writes nothing to and the graph replaces wholesale (§5.1) — cloning
    /// the `Arc` is what makes the read lock-free, and holding a `&` into it would pin the version.
    pub fn params(&self, uid: Uid) -> Option<Arc<ParamGroups>> {
        self.nodes.get(&uid).map(|e| e.params.load_full())
    }

    /// The node's param record itself — the handle its own thread keeps, so it reads params without
    /// ever taking the graph lock. The graph writes through it; nobody else does.
    pub fn param_record(&self, uid: Uid) -> Option<Arc<ArcSwap<ParamGroups>>> {
        self.nodes.get(&uid).map(|e| e.params.clone())
    }

    /// The globals as the node threads read them (§5.2), by the same rule and for the same reason.
    pub fn globals_record(&self) -> Arc<ArcSwap<goofi_core::globals::GlobalsSnapshot>> {
        self.globals_record.clone()
    }

    /// Write the globals store and re-publish the node-side view. The ONE writer, so the two can
    /// never drift — a store mutated anywhere else would leave every node reading the old values.
    fn globals_mut(&mut self, edit: impl FnOnce(&mut goofi_core::globals::GlobalStore) -> Result<(), String>) -> Result<(), String> {
        let out = edit(&mut self.globals);
        self.globals_record.store(Arc::new(self.globals.snapshot()));
        out
    }

    /// Rename a node's display name (globally unique). On a successful rename, every
    /// `nd('old')` reference in the graph's param expressions follows to `nd('new')` —
    /// they resolve producers by name — re-binding each rewritten expression. Returns the
    /// referrer uids whose source changed, so the bridge can rebroadcast them. Mirrors
    /// Python's `manager.rename_node`; the rewrite happens ONLY when the rename succeeds.
    pub fn rename_node(&mut self, uid: Uid, name: &str) -> Result<Vec<Uid>, String> {
        if self.name_in_use(name) {
            return Err(format!("display name `{name}` already in use"));
        }
        // A scope facade (collapsed sub-patch instance) carries its own display name; `nd()`
        // expressions only reference leaf-node names, so a scope rename rewrites nothing.
        if let Some(s) = self.scopes.get_mut(&uid) {
            s.name = name.to_string();
            return Ok(vec![]);
        }
        let old_name = self
            .nodes
            .get(&uid)
            .ok_or_else(|| format!("no such node {uid}"))?
            .name
            .clone();
        self.nodes.get_mut(&uid).unwrap().name = name.to_string();
        // `name_in_use` guarantees `name != old_name`, so the rename genuinely moved the
        // display name — propagate it into every expression that referenced it.
        let touched = self.rewrite_nd_refs_for_rename(&old_name, name);
        // …and re-resolve the ones that were ALREADY written against the new name. A binding
        // authored as `nd('src')` before any node was called `src` is unresolved, and this rename
        // is what makes it resolvable — the rewrite above cannot see it, since there is no
        // `nd('<old>')` in it to follow (§5.3: the graph re-resolves on a rename, an add and a
        // removal alike).
        self.rebind_naming(name);
        Ok(touched)
    }

    /// Rewrite `nd('old')` -> `nd('new')` across all nodes' param expressions, re-binding
    /// each changed source (recompiling so its extracted refs track the new name). Returns
    /// the distinct referrer uids whose source changed.
    fn rewrite_nd_refs_for_rename(&mut self, old: &str, new: &str) -> Vec<Uid> {
        let mut edits: Vec<(Uid, ParamKey, String, bool, bool)> = Vec::new();
        for (&ruid, entry) in &self.nodes {
            for (key, b) in &entry.bindings {
                let rewritten = goofi_node::rewrite_nd_refs(&b.source, |n| {
                    (n == old).then(|| new.to_string())
                });
                if let Some(src) = rewritten {
                    edits.push((ruid, key.clone(), src, b.enabled, b.triggers_process));
                }
            }
        }
        let mut touched: Vec<Uid> = Vec::new();
        for (ruid, key, src, enabled, triggers) in edits {
            if self.set_expression(ruid, &key.group, &key.name, &src, enabled, triggers).is_ok()
                && !touched.contains(&ruid)
            {
                touched.push(ruid);
            }
        }
        // Expressions live only on the live flat nodes now (no def templates) — the loop above has
        // already followed the rename into every one.
        touched
    }

    pub fn set_node_pos(&mut self, uid: Uid, pos: [f64; 2]) -> Result<(), String> {
        // A scope facade's pos lives in `scopes[uid].pos` (it is not a live node) — move it there.
        if let Some(s) = self.scopes.get_mut(&uid) {
            s.pos = pos;
            return Ok(());
        }
        let e = self
            .nodes
            .get_mut(&uid)
            .ok_or_else(|| format!("no such node {uid}"))?;
        e.pos = pos;
        Ok(())
    }

    /// Replace a node's opaque viewer view-state blob (persisted to `.gfi`, echoed in node
    /// info). The backend never interprets it — it is the editor's per-slot kind/settings.
    pub fn set_node_viewers(&mut self, uid: Uid, viewers: serde_json::Value) -> Result<(), String> {
        let e = self
            .nodes
            .get_mut(&uid)
            .ok_or_else(|| format!("no such node {uid}"))?;
        e.viewers = viewers;
        Ok(())
    }

    /// A node's viewer view-state blob (empty object if never set).
    pub fn viewers(&self, uid: Uid) -> Option<&serde_json::Value> {
        self.nodes.get(&uid).map(|e| &e.viewers)
    }

    // ── Sub-patch forest: accessors + group/expand (bookkeeping-only) ─────────────
    // Grouping never touches the flat runtime — the members stay the exact live nodes they
    // were; only their membership re-tags. So there is no respawn, no data gap, and undo is
    // just the inverse tag flip. `reconcile` (Phase 5) is what SPAWNS subtrees.

    /// The parent scope of a node/scope (`None` = ROOT). Absent ⇒ ROOT, so a plain flat graph
    /// needs no entries.
    pub fn scope_of(&self, uid: Uid) -> Option<Uid> {
        self.scope_of.get(&uid).copied().flatten()
    }

    /// All scope uids (each == its collapsed facade node's uid).
    pub fn scope_uids(&self) -> Vec<Uid> {
        self.scopes.keys().copied().collect()
    }

    pub fn scope(&self, uid: Uid) -> Option<&subpatch::Scope> {
        self.scopes.get(&uid)
    }

    /// The direct member uids of a scope (leaf nodes + child scopes), in flat-`nodes` then scope
    /// insertion order — a deterministic display/serialization order derived from the `scope_of`
    /// tree (the SSOT), so there is no parallel member list to keep in sync.
    pub fn scope_members(&self, scope: Uid) -> Vec<Uid> {
        let mut out: Vec<Uid> = self
            .nodes
            .keys()
            .copied()
            .filter(|u| self.scope_of(*u) == Some(scope))
            .collect();
        out.extend(self.scopes.keys().copied().filter(|u| self.scope_of(*u) == Some(scope)));
        out
    }

    /// Chain-resolve a scope's stub port to the single physical inner leaf `(uid, slot)` it exposes
    /// (walking nested scopes); `None` if unwired. Used by the snapshot projection and the data
    /// plane (a viewer on `scope/stub` subscribes to this leaf) + link authoring.
    pub fn resolve_stub(&self, scope: Uid, stub: &str) -> Option<(Uid, String)> {
        subpatch::resolve_stub(&self.scopes, scope, stub)
    }

    /// The output slot decl named `slot` on node `uid`, if any — the shared lookup behind
    /// `output_slot_type` / `resolve_output`.
    fn find_output(&self, uid: Uid, slot: &str) -> Option<&'static goofi_node::OutputDecl> {
        self.nodes.get(&uid)?.manifest.outputs.iter().find(|o| o.name == slot)
    }

    /// The input slot decl named `slot` on node `uid`, if any — the shared lookup behind
    /// `input_slot_type` / `resolve_input` / `is_multi_input`.
    fn find_input(&self, uid: Uid, slot: &str) -> Option<&'static goofi_node::SlotDecl> {
        self.nodes.get(&uid)?.manifest.inputs.iter().find(|s| s.name == slot)
    }

    fn output_slot_type(&self, uid: Uid, slot: &str) -> Option<goofi_core::SlotType> {
        self.find_output(uid, slot).map(|o| o.kind)
    }

    fn input_slot_type(&self, uid: Uid, slot: &str) -> Option<goofi_core::SlotType> {
        self.find_input(uid, slot).map(|s| s.kind)
    }

    /// Move a node or scope into `scope` (`None` = ROOT), returning its prior membership. The one
    /// validated re-parent seam a `SetScope` command drives (restoring a member back inside its
    /// scope on a delete-undo). Errors on an unknown uid or a `scope` that is not a live scope.
    pub fn reparent(&mut self, uid: Uid, scope: Option<Uid>) -> Result<Option<Uid>, String> {
        if !self.nodes.contains_key(&uid) && !self.scopes.contains_key(&uid) {
            return Err(format!("reparent: no such node/scope {uid}"));
        }
        if let Some(s) = scope {
            if !self.scopes.contains_key(&s) {
                return Err(format!("reparent: no such scope {s}"));
            }
        }
        let old = self.scope_of(uid);
        self.set_member_scope(uid, scope);
        Ok(old)
    }

    /// Re-tag a member's scope. `scope_of` is the single source of truth for parentage, so this is
    /// the one place membership changes. `None` = ROOT scope.
    fn set_member_scope(&mut self, member: Uid, scope: Option<Uid>) {
        match scope {
            Some(p) => {
                self.scope_of.insert(member, Some(p));
            }
            None => {
                self.scope_of.remove(&member);
            }
        }
    }

    /// The member of `member_set` that transitively contains `uid` — `uid` itself if it is a direct
    /// member, else the ancestor scope (walking up `scope_of`) that is a member. `None` if `uid`
    /// lies outside every member. Lets link classification treat a leaf buried in a nested member
    /// scope as "inside the group".
    fn containing_member(&self, uid: Uid, member_set: &std::collections::HashSet<Uid>) -> Option<Uid> {
        let mut cur = uid;
        loop {
            if member_set.contains(&cur) {
                return Some(cur);
            }
            cur = self.scope_of(cur)?;
        }
    }

    /// The stub id on nested scope `scope` whose chain-to-leaf resolution is exactly `(leaf, slot)`
    /// in direction `dir`. Used to name the interior endpoint of a link that crosses into a nested
    /// member: the stub references the nested scope's PORT, not the buried leaf. Handles arbitrary
    /// nesting depth (`resolve_stub` recurses down).
    fn stub_exposing(&self, scope: Uid, leaf: Uid, slot: &str, dir: subpatch::Dir) -> Option<subpatch::StubId> {
        let s = self.scopes.get(&scope)?;
        s.stubs
            .iter()
            .filter(|(_, st)| st.dir == dir)
            .find(|(id, _)| self.resolve_stub(scope, id).is_some_and(|(u, sl)| u == leaf && sl == slot))
            .map(|(id, _)| id.clone())
    }

    /// The direct member of `scope` on the path from `leaf` up the `scope_of` tree — the child of
    /// `scope` that (transitively) contains `leaf`, or `leaf` itself when it is a direct member.
    /// `None` if `leaf` is not inside `scope`.
    fn direct_child_containing(&self, scope: Uid, leaf: Uid) -> Option<Uid> {
        let mut cur = leaf;
        loop {
            let parent = self.scope_of(cur)?;
            if parent == scope {
                return Some(cur);
            }
            cur = parent;
        }
    }

    /// Lowest `in{n}`/`out{n}` stub id not already used on `scope`.
    fn fresh_stub_id(&self, scope: Uid, dir: subpatch::Dir) -> subpatch::StubId {
        let stubs = self.scopes.get(&scope).map(|s| &s.stubs);
        for n in 0.. {
            let cand = format!("{}{n}", dir.name());
            if stubs.map(|st| !st.contains_key(&cand)).unwrap_or(true) {
                return cand;
            }
        }
        unreachable!()
    }

    /// The inner-slot key that a group boundary stub should reference for a crossing link whose
    /// direct group member is `member` and whose buried leaf endpoint is `(leaf, slot)`:
    /// * `member == leaf` — the member IS the leaf endpoint: the real slot.
    /// * `member` is a nested scope already exposing the leaf: its existing stub id.
    /// * otherwise — MINT the missing chain of stubs (one fresh port per nesting level) down to the
    ///   leaf and return the top one's id.
    ///
    /// The last case keeps `group_nodes` TOTAL. A crossing flat link can outlive the boundary port
    /// that once exposed it — `remove_boundary` drops the stub but LEAVES the leaf→leaf link — so a
    /// later re-group must reconstruct the port rather than assert a now-broken invariant (the old
    /// `debug_assert` panicked in dev/CI, poisoning the graph mutex, and minted a dangling stub in
    /// release). Every port MINTED here is recorded in `minted` so the group's inverse can un-mint it
    /// (else group→undo would leave the reconstructed port resurrected on the nested member).
    fn expose_in_nested_member(
        &mut self,
        member: Uid,
        leaf: Uid,
        slot: &str,
        dir: subpatch::Dir,
        minted: &mut Vec<(Uid, subpatch::StubId)>,
    ) -> subpatch::StubId {
        if member == leaf {
            return slot.to_string();
        }
        if let Some(id) = self.stub_exposing(member, leaf, slot, dir) {
            return id;
        }
        // No port exposes the leaf — mint one. Its inner is the leaf directly when the leaf is a
        // direct member, else the (recursively ensured) port on the intermediate nested scope.
        let child = self.direct_child_containing(member, leaf).unwrap_or(leaf);
        let inner = if child == leaf {
            (leaf, slot.to_string())
        } else {
            let child_stub = self.expose_in_nested_member(child, leaf, slot, dir, minted);
            (child, child_stub)
        };
        let dtype = match dir {
            subpatch::Dir::Out => self.output_slot_type(leaf, slot),
            subpatch::Dir::In => self.input_slot_type(leaf, slot),
        }
        .unwrap_or(goofi_core::SlotType::Array);
        let id = self.fresh_stub_id(member, dir);
        let base = self.pos(member).unwrap_or([0.0, 0.0]);
        let pos = match dir {
            subpatch::Dir::Out => [base[0] + 220.0, base[1]],
            subpatch::Dir::In => [base[0] - 40.0, base[1]],
        };
        if let Some(s) = self.scopes.get_mut(&member) {
            s.stubs
                .insert(id.clone(), subpatch::Stub { dir, dtype, inner: Some(inner), pos, name: id.clone() });
            minted.push((member, id.clone()));
        }
        id
    }

    /// The single common parent scope of `members` (each must exist as a node or scope), or an error
    /// if the set is empty or spans multiple scopes. Shared validation for `group_nodes` (mint) and
    /// `restore_scope` (undo/redo) so the check lives in one place.
    fn common_parent(&self, members: &[Uid]) -> Result<Option<Uid>, String> {
        if members.is_empty() {
            return Err("group: empty selection".into());
        }
        let mut parent: Option<Option<Uid>> = None;
        for &m in members {
            if !self.nodes.contains_key(&m) && !self.scopes.contains_key(&m) {
                return Err(format!("group: no such member {m}"));
            }
            let s = self.scope_of(m);
            match parent {
                None => parent = Some(s),
                Some(prev) if prev != s => return Err("group: members span multiple scopes".into()),
                _ => {}
            }
        }
        Ok(parent.unwrap())
    }

    /// Group `members` (leaf nodes and/or existing scopes, all in ONE scope) into a new sub-patch
    /// scope. Pure reference-move bookkeeping: mint a scope, mint a stub for every flat link that
    /// crosses the new boundary (its `inner` = the crossing member + slot), and re-tag membership.
    /// Returns the new scope uid. The flat `nodes`/`links` and every member's uid are UNCHANGED →
    /// uid-stable by construction; the crossing flat links stay verbatim (they resolve through the
    /// new stubs).
    pub fn group_nodes(&mut self, members: &[Uid], pos: [f64; 2]) -> Result<Uid, String> {
        self.group_nodes_capturing(members, pos, &mut Vec::new())
    }

    /// Like [`Self::group_nodes`], but records into `minted` every stub it has to MINT on a
    /// pre-existing nested member (to re-expose an orphaned crossing link). The `Group` command
    /// threads this into its inverse so undo un-mints them — else group→undo would leave those ports
    /// resurrected on the nested member (an inexact inverse).
    pub fn group_nodes_capturing(
        &mut self,
        members: &[Uid],
        pos: [f64; 2],
        minted: &mut Vec<(Uid, subpatch::StubId)>,
    ) -> Result<Uid, String> {
        use subpatch::{Dir, Scope, Stub};
        // 1. Validate BEFORE any mutation: each exists, and all share one parent scope.
        let parent = self.common_parent(members)?;
        let member_set: std::collections::HashSet<Uid> = members.iter().copied().collect();
        let scope_uid = self.mint();

        // 2. Classify each flat link by TRANSITIVE containment — an endpoint buried inside a nested
        //    member scope counts as inside the group. Exactly one endpoint inside → a boundary,
        //    minted as a stub whose `inner` names the DIRECT member + slot (a nested member's stub
        //    id when buried, so `resolve_stub` chains to the leaf). Both inside / both outside → the
        //    flat link stays verbatim, no stub. One stub per inner (node, slot).
        let mut stubs: IndexMap<subpatch::StubId, Stub> = IndexMap::new();
        let mut seen: std::collections::HashSet<(Uid, &'static str, bool)> = std::collections::HashSet::new();
        let (mut in_n, mut out_n) = (0usize, 0usize);
        // Snapshot the links: `expose_in_nested_member` may MINT an intermediate stub on a nested
        // member (re-exposing a leaf whose port was dropped by `remove_boundary`), which needs
        // `&mut self` — so the classification can't hold a borrow on `self.links`.
        let links = self.links.clone();
        for l in &links {
            let out_m = self.containing_member(l.node_out, &member_set);
            let in_m = self.containing_member(l.node_in, &member_set);
            match (out_m, in_m) {
                (Some(om), None) => {
                    if !seen.insert((l.node_out, l.slot_out, true)) {
                        continue;
                    }
                    let dtype = self.output_slot_type(l.node_out, l.slot_out).unwrap_or(goofi_core::SlotType::Array);
                    let inner_slot = self.expose_in_nested_member(om, l.node_out, l.slot_out, Dir::Out, minted);
                    let id = format!("out{out_n}");
                    stubs.insert(
                        id.clone(),
                        Stub {
                            dir: Dir::Out,
                            dtype,
                            inner: Some((om, inner_slot)),
                            pos: [pos[0] + 220.0, pos[1] + 40.0 * out_n as f64],
                            name: id,
                        },
                    );
                    out_n += 1;
                }
                (None, Some(im)) => {
                    if !seen.insert((l.node_in, l.slot_in, false)) {
                        continue;
                    }
                    let dtype = self.input_slot_type(l.node_in, l.slot_in).unwrap_or(goofi_core::SlotType::Array);
                    let inner_slot = self.expose_in_nested_member(im, l.node_in, l.slot_in, Dir::In, minted);
                    let id = format!("in{in_n}");
                    stubs.insert(
                        id.clone(),
                        Stub {
                            dir: Dir::In,
                            dtype,
                            inner: Some((im, inner_slot)),
                            pos: [pos[0] - 40.0, pos[1] + 40.0 * in_n as f64],
                            name: id,
                        },
                    );
                    in_n += 1;
                }
                _ => {}
            }
        }

        // 3. Register the scope + re-tag membership. Members stay live; only `scope_of` changes.
        let disp = self.mint_subpatch_name(scope_uid);
        self.scopes.insert(scope_uid, Scope { name: disp, pos, stubs });
        for &m in members {
            self.set_member_scope(m, Some(scope_uid));
        }
        self.scope_of.insert(scope_uid, parent);
        Ok(scope_uid)
    }

    /// Recreate a scope EXACTLY — a specific `scope_id`, name, pos, and stubs — re-tagging `members`
    /// into it. The inverse of `expand_instance` (undo-of-expand / redo-of-group): the members are
    /// currently in the grandparent scope (where expand left them); this moves them back under
    /// `scope_id` with the captured stubs verbatim, so undo/redo is uid-stable. A `scope_id` already
    /// live is rejected (the command layer guards redo races before calling).
    pub fn restore_scope(
        &mut self,
        scope_id: Uid,
        name: String,
        pos: [f64; 2],
        members: &[Uid],
        stubs: IndexMap<subpatch::StubId, subpatch::Stub>,
        parent: Option<Uid>,
    ) -> Result<Uid, String> {
        if self.scopes.contains_key(&scope_id) {
            return Err(format!("restore_scope: scope {scope_id} already live"));
        }
        // The parent is captured explicitly (not derived from members via `common_parent`), so an
        // EMPTY scope restores fine and a subtree restore need not thread parentage through member
        // placement. Re-tag only members that actually exist (a redo-race may have dropped one).
        self.scopes.insert(scope_id, subpatch::Scope { name, pos, stubs });
        for &m in members {
            if self.nodes.contains_key(&m) || self.scopes.contains_key(&m) {
                self.set_member_scope(m, Some(scope_id));
            }
        }
        // A peer may have dissolved the captured parent since this restore was recorded (a nested
        // scope's delete-undo racing an expand). Writing it verbatim would install a dangling-parent
        // orphan — a scope whose parentage names a scope that no longer exists, which no member walk
        // can reach. Degrade to ROOT, exactly as the membership-restoring `SetScope` child does.
        self.set_member_scope(scope_id, parent.filter(|p| self.scopes.contains_key(p)));
        Ok(scope_id)
    }

    /// The parent-scope stubs that currently expose `scope` (each as `(parent, stub_id, inner)`).
    /// `Expand` captures these BEFORE dissolving so its `Group` inverse can re-point them back
    /// exactly (Expand re-points them forward to the child stub's inner). Empty if `scope` is at ROOT
    /// or no parent stub references it.
    pub fn parent_stubs_referencing(&self, scope: Uid) -> Vec<(Uid, subpatch::StubId, Option<(Uid, String)>)> {
        let Some(p) = self.scope_of(scope) else {
            return vec![];
        };
        self.scopes
            .get(&p)
            .map(|ps| {
                ps.stubs
                    .iter()
                    .filter(|(_, st)| st.inner.as_ref().map(|(u, _)| *u == scope).unwrap_or(false))
                    .map(|(id, st)| (p, id.clone(), st.inner.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Directly set a stub's `inner` with NO validation — the restore path for re-pointing a parent
    /// stub during a Group/Expand round-trip, where the target is a known-good captured state (which
    /// may name a nested scope, unlike the validated `set_stub_inner` wire path).
    pub fn restore_stub_inner(&mut self, scope: Uid, stub_id: &str, inner: Option<(Uid, String)>) {
        if let Some(st) = self.scopes.get_mut(&scope).and_then(|s| s.stubs.get_mut(stub_id)) {
            st.inner = inner;
        }
    }

    /// Inline a scope back into its parent: re-tag each member to the parent scope, then drop the
    /// scope + its stubs. The crossing flat links already point at the members leaf→leaf, so they
    /// survive verbatim — nothing to reconnect. Returns the restored member uids. Uid-stable.
    pub fn expand_instance(&mut self, scope: Uid) -> Result<Vec<Uid>, String> {
        if !self.scopes.contains_key(&scope) {
            return Err(format!("expand_instance: no such scope {scope}"));
        }
        let restored = self.scope_members(scope);
        let parent = self.scope_of(scope); // the grandparent scope members fall back to
        // Re-point any PARENT-scope stub that exposed this scope's port. The scope dissolves but its
        // members survive (they move up to `parent`), so a parent stub whose inner==(scope, child_id)
        // must FOLLOW to the physical leaf that child resolved to — else it dangles at a scope that no
        // longer exists. (remove_member PRUNES the analogous stub because ITS member is deleted; here
        // the leaf lives on, so we re-point.)
        if let Some(p) = parent {
            let targets: Vec<(subpatch::StubId, String)> = self
                .scopes
                .get(&p)
                .map(|ps| {
                    ps.stubs
                        .iter()
                        .filter_map(|(id, st)| {
                            st.inner.as_ref().and_then(|(u, cid)| (*u == scope).then(|| (id.clone(), cid.clone())))
                        })
                        .collect()
                })
                .unwrap_or_default();
            for (id, cid) in targets {
                // Re-point to the child stub's OWN inner (ONE level down) — that direct member of
                // `scope` becomes a direct member of `p` after expand, so the parent stub stays
                // structurally valid. (Using the fully-resolved leaf would be wrong when the leaf is
                // buried in a NESTED scope that only moves up one level.)
                let child_inner = self.scopes.get(&scope).and_then(|s| s.stubs.get(&cid)).and_then(|st| st.inner.clone());
                if let Some(st) = self.scopes.get_mut(&p).and_then(|ps| ps.stubs.get_mut(&id)) {
                    st.inner = child_inner;
                }
            }
        }
        for &m in &restored {
            self.set_member_scope(m, parent);
        }
        self.scopes.shift_remove(&scope);
        self.scope_of.remove(&scope);
        Ok(restored)
    }

    /// Delete a whole sub-patch scope: tear down every member (recursing into nested scopes, and
    /// removing leaves + their flat links), then drop the scope. The frontend routes
    /// Delete-on-a-collapsed-sub-patch here.
    pub fn remove_instance(&mut self, scope: Uid) -> Result<(), String> {
        if !self.scopes.contains_key(&scope) {
            return Err(format!("remove_instance: no such scope {scope}"));
        }
        for m in self.scope_members(scope) {
            if self.scopes.contains_key(&m) {
                self.remove_instance(m)?; // nested scope subtree
            } else {
                let _ = self.remove_node(m); // leaf (tolerate an already-gone member)
            }
        }
        self.scopes.shift_remove(&scope);
        self.scope_of.remove(&scope);
        Ok(())
    }

    /// Remove a MEMBER of a sub-patch (a leaf or nested scope living inside a scope). Tears down the
    /// live entity (a nested scope subtree via `remove_instance`, else a leaf via `remove_node`,
    /// which also drops the external flat links into it), then drops any stub of the enclosing scope
    /// whose `inner` referenced it — so a dangling port is never left, and a save/reload has nothing
    /// to resurrect. The enclosing scope survives its remaining members; a uid with no scope never
    /// reaches here.
    pub fn remove_member(&mut self, member: Uid) -> Result<(), String> {
        let scope = self
            .scope_of(member)
            .ok_or_else(|| format!("remove_member: {member} is not a sub-patch member"))?;
        if self.scopes.contains_key(&member) {
            self.remove_instance(member)?;
        } else {
            let _ = self.remove_node(member);
        }
        if let Some(s) = self.scopes.get_mut(&scope) {
            s.stubs.retain(|_, st| st.inner.as_ref().map(|(u, _)| *u != member).unwrap_or(true));
        }
        Ok(())
    }

    // ── Stub authoring (boundary ports on a scope; never live nodes) ──────────────
    // A stub is a naming indirection over an inner member slot (its `inner` child side). External
    // wires stay flat leaf→leaf links that resolve through the stub; the stub itself stores only the
    // inner side. All edits mutate the scope's stubs directly (no sharing, no def).

    /// Is `uid` a direct member of `scope`?
    fn is_member_of(&self, scope: Uid, uid: Uid) -> bool {
        self.scope_of(uid) == Some(scope)
    }

    /// Add an UNWIRED stub to a scope; returns its stable `StubId` (`in{n}`/`out{n}`). `dtype` is
    /// the caller's provisional type until the port is wired.
    pub fn add_boundary(
        &mut self,
        scope: Uid,
        dir: subpatch::Dir,
        dtype: goofi_core::SlotType,
        pos: [f64; 2],
    ) -> Result<subpatch::StubId, String> {
        if !self.scopes.contains_key(&scope) {
            return Err(format!("add_boundary: no such scope {scope}"));
        }
        // StubIds are persisted into the `.gfi`, so the two minting sites must agree forever —
        // hence the shared `fresh_stub_id` rather than a second inline scan.
        let id = self.fresh_stub_id(scope, dir);
        let s = self.scopes.get_mut(&scope).expect("checked above");
        s.stubs.insert(id.clone(), subpatch::Stub { dir, dtype, inner: None, pos, name: id.clone() });
        Ok(id)
    }

    /// Validate a candidate stub wire and resolve the port dtype it would take, without mutating.
    /// Extracted from [`Graph::set_stub_inner`] so the forward-RPC precondition (`Command::
    /// precondition`) and the mutation share ONE algebra — a second copy of "is this inner target
    /// legal" is exactly the drift this codebase spends its unification budget avoiding.
    pub fn stub_wire_dtype(
        &self,
        scope: Uid,
        stub: &str,
        inner: &(Uid, String),
    ) -> Result<goofi_core::SlotType, String> {
        let (inner_node, inner_slot) = inner;
        if !self.is_member_of(scope, *inner_node) {
            return Err("set_stub_inner: inner is not a member of this scope".into());
        }
        let dir = self
            .scopes
            .get(&scope)
            .and_then(|s| s.stubs.get(stub))
            .map(|st| st.dir)
            .ok_or("set_stub_inner: no such stub")?;
        // A member may itself be a sub-patch, in which case its ports are that scope's own
        // stubs, not slot decls — the `(facade uid, StubId)` shape `Stub.inner` documents
        // and that `group_nodes` mints itself. `is_member_of` above already proved a DIRECT
        // child, so chaining one port onto another cannot close a cycle.
        let dtype = match self.scopes.get(inner_node) {
            Some(nested) => nested.stubs.get(inner_slot.as_str()).filter(|st| st.dir == dir).map(|st| st.dtype),
            None => match dir {
                subpatch::Dir::In => self.input_slot_type(*inner_node, inner_slot),
                subpatch::Dir::Out => self.output_slot_type(*inner_node, inner_slot),
            },
        }
        .ok_or("set_stub_inner: no such inner slot")?;
        let s = self.scopes.get(&scope).ok_or("set_stub_inner: no such scope")?;
        if s.stubs.iter().any(|(id, st)| id != stub && st.inner.as_ref() == Some(inner)) {
            return Err("set_stub_inner: that inner slot is already exposed by another stub".into());
        }
        Ok(dtype)
    }

    /// Set (`Some`) or clear (`None`) a stub's inner target — the canonical wire/unwire. Wiring
    /// validates membership + one-stub-per-inner-slot and resolves the port dtype from the slot
    /// (via [`Graph::stub_wire_dtype`], check-then-mutate so a refused attempt leaves the stub
    /// untouched); unwiring just clears it. The command layer captures the old inner for the
    /// exact inverse.
    pub fn set_stub_inner(&mut self, scope: Uid, stub: &str, inner: Option<(Uid, String)>) -> Result<(), String> {
        match inner {
            Some(target) => {
                let dtype = self.stub_wire_dtype(scope, stub, &target)?;
                let st = self
                    .scopes
                    .get_mut(&scope)
                    .and_then(|s| s.stubs.get_mut(stub))
                    .ok_or("set_stub_inner: no such stub")?;
                st.inner = Some(target);
                st.dtype = dtype;
                Ok(())
            }
            None => {
                let st = self
                    .scopes
                    .get_mut(&scope)
                    .and_then(|s| s.stubs.get_mut(stub))
                    .ok_or("set_stub_inner: no such stub")?;
                st.inner = None;
                Ok(())
            }
        }
    }

    /// Insert a full captured stub at a specific id — the restore inverse of removing a stub.
    pub fn insert_stub(&mut self, scope: Uid, stub_id: subpatch::StubId, stub: subpatch::Stub) -> Result<(), String> {
        let s = self.scopes.get_mut(&scope).ok_or("insert_stub: no such scope")?;
        s.stubs.insert(stub_id, stub);
        Ok(())
    }

    /// Force a stub's advertised dtype. Wiring resolves the dtype from the inner slot, so the
    /// `WireStub` inverse uses this to restore the EXACT pre-wire dtype on unwire (else an unwired
    /// pill would keep the wired slot's type instead of its provisional one).
    pub fn set_stub_dtype(&mut self, scope: Uid, stub: &str, dtype: goofi_core::SlotType) -> Result<(), String> {
        let st = self
            .scopes
            .get_mut(&scope)
            .and_then(|s| s.stubs.get_mut(stub))
            .ok_or("set_stub_dtype: no such stub")?;
        st.dtype = dtype;
        Ok(())
    }

    /// Drop a stub. External flat links stay valid leaf→leaf links (they never referenced the stub
    /// at runtime), so they are left in place.
    pub fn remove_boundary(&mut self, scope: Uid, stub: &str) -> Result<(), String> {
        let s = self.scopes.get_mut(&scope).ok_or("remove_boundary: no such scope")?;
        s.stubs.shift_remove(stub).ok_or("remove_boundary: no such stub")?;
        Ok(())
    }

    /// Relabel a stub's display name. The `StubId` is unchanged, so external wires survive.
    pub fn rename_boundary(&mut self, scope: Uid, stub: &str, name: &str) -> Result<(), String> {
        let st = self
            .scopes
            .get_mut(&scope)
            .and_then(|s| s.stubs.get_mut(stub))
            .ok_or("rename_boundary: no such stub")?;
        st.name = name.to_string();
        Ok(())
    }

    /// Move a stub pill inside the entered view.
    pub fn set_boundary_pos(&mut self, scope: Uid, stub: &str, pos: [f64; 2]) -> Result<(), String> {
        let st = self
            .scopes
            .get_mut(&scope)
            .and_then(|s| s.stubs.get_mut(stub))
            .ok_or("set_boundary_pos: no such stub")?;
        st.pos = pos;
        Ok(())
    }

    // The `update_member_param` / `set_member_pos` / `set_member_expression` wrappers are gone
    // (B3a): with sharing dropped a member is just a live node, and every mutation now routes through
    // an `EditParam` / `EditNode` command over `update_param` / `set_node_pos` / `set_expression`
    // directly — the client-doc-write leaf path they served was retired with `apply_client_write`.

    /// All links as resolved views (snapshot projection).
    pub fn links_view(&self) -> Vec<LinkView> {
        self.links
            .iter()
            .map(|l| LinkView {
                node_out: l.node_out,
                slot_out: l.slot_out,
                node_in: l.node_in,
                slot_in: l.slot_in,
            })
            .collect()
    }

    /// Release every compiled expression handle a node entry holds, so the evaluator's
    /// registry doesn't leak across a node/graph teardown.
    fn release_entry_bindings(&self, entry: &NodeEntry) {
        if let Some(ev) = &self.evaluator {
            for b in entry.bindings.values() {
                if let Some(id) = b.id {
                    ev.release(id);
                }
            }
        }
    }

    pub fn remove_node(&mut self, uid: Uid) -> Result<(), String> {
        let Some(removed) = self.nodes.shift_remove(&uid) else {
            return Err(format!("no such node {uid}"));
        };
        self.release_entry_bindings(&removed);
        // §5.3: every binding that referenced this node by name is now unresolvable, and must be
        // told so — a variable still naming a dead producer's service is one the node waits on
        // forever.
        let name = removed.name.clone();
        self.rebind_naming(&name);
        // Drop any membership tag: a removed node has no scope. Leaving it dangling would make a
        // reused uid (a delete→undo that restores the scope) self-parent via `common_parent`.
        self.scope_of.remove(&uid);
        // Drop links touching the node, then re-plan every consumer slot one of them fed (§4:
        // removal is a wire change like any other). Links INTO the removed node need no re-plan —
        // its thread is already halted and its services are going with it.
        let dropped: Vec<Link> = self
            .links
            .iter()
            .filter(|l| l.node_out == uid || l.node_in == uid)
            .cloned()
            .collect();
        self.links
            .retain(|l| l.node_out != uid && l.node_in != uid);
        for l in dropped.iter().filter(|l| l.node_in != uid) {
            self.replan_slot(l.node_in, l.slot_in);
        }
        Ok(())
    }

    /// Respawn a node's live instance IN PLACE — the recovery action behind the inspector's
    /// restart button, for a node that crashed or whose backing `.py` was fixed on disk.
    ///
    /// Everything that identifies the node *in the patch* survives: uid (so uid-keyed links and
    /// panels stay connected), display name (expressions reference it), position, params,
    /// expression bindings, viewer state, and scope membership. Only the instance and its
    /// per-run state are replaced. Remove+add is NOT a substitute — it drops the links, the
    /// bindings and the sub-patch membership, and would land the node back at root.
    ///
    /// Deliberately **not** a `Command`: it changes no persisted patch state, so it has no
    /// meaningful inverse, and the client records no history entry for it.
    ///
    /// One limit worth knowing: a Python node re-runs the SOURCE CAPTURED AT DISCOVERY, so editing
    /// the `.py` and restarting does not pick up the edit — a rescan (which re-registers the type)
    /// does, and that is what drives the auto-restart. The `index_counters` carried over below are
    /// engine-side, so a Subprocess node's own child-side numbering still restarts with its process.
    pub fn restart_node(&mut self, uid: Uid) -> Result<(), String> {
        let entry = self.nodes.get(&uid).ok_or_else(|| format!("no such node {uid}"))?;
        let type_name = entry.manifest.type_name;
        let held = entry.params.load_full();
        // Fold what the node HAS onto what its type declares NOW, rather than replaying the old map
        // verbatim: a rescan restart is usually prompted by an edit to the file, and an edit that
        // adds a param would otherwise leave the instance without it while the palette advertises
        // it. Same order and same rule as the `.gfi` load — defaults first, and then only the
        // saved VALUE over each: the declaration's bounds, options, `refresh` flag and variant are
        // the edited file's to state. Replacing the whole `Param` would silently keep the instance
        // on the old spec while the inspector already draws the new one from the catalog.
        let mut params = self.default_params_of(type_name)?;
        for (group, held) in &*held {
            let Some(g) = params.get_mut(group) else { continue };
            for (name, value) in held {
                if let Some(slot) = g.get_mut(name) {
                    // `fire_triggers: false` — a rescan must not trip a node's trigger.
                    *slot = param_from_json(slot, &param_value_json(value), false);
                }
            }
        }
        // Construct BEFORE touching the entry: a type that no longer resolves leaves the old
        // instance running rather than half-killing the node.
        let (manifest, params, node) = self.build_node(type_name, Some(params))?;

        // A restart is a BIRTH at this uid (§3.1) — the first case the generation counter names,
        // because the corpse's teardown does not block: without the bump the reborn node re-opens
        // its predecessor's service names while its predecessor's ports are still registered, and
        // `max_publishers(1)` refuses the new publisher. This is the one birth that does not go
        // through `insert_node_at`, which is exactly why it has to be said here too.
        let generation = self.wire.bump_generation(uid);
        let (host, boot_error) = self.spawn_host(uid, generation, manifest, node, &params);

        let entry = self.nodes.get_mut(&uid).expect("looked up above");
        // Replacing the host halts the corpse's thread (`Drop for NodeHost`), which never waits:
        // the dying node notices at its next wake and this runs under the graph mutex.
        //
        // The MANIFEST goes with the instance. It is the graph's whole description of this node —
        // link validation, schema projection, `/data` target checks and the scheduler's trigger
        // policy all read it — and the rescan path re-registers a stable `type_name` over a
        // possibly-reshaped interface. Keeping the old one here (which this did until the
        // boundary-hardening pass) left the graph describing a node that is no longer running:
        // a slot the edit added was unlinkable, and one it removed still accepted wires.
        entry.manifest = manifest;
        entry.host = host;
        // A swap, not a new record: the graph's readers hold this very handle, so replacing it
        // would leave them reading the corpse's params.
        entry.params.store(Arc::new(params));
        // A fresh generation boots healthy and reports its own state; the corpse's error and stage
        // describe a node that no longer exists (§4). `boot_error` is this birth's own.
        entry.setup_error = boot_error;
        entry.last_error = None;
        entry.stage = "creating";
        entry.ufreq = None;
        // The evaluated values are the CORPSE's report (§6.2): a fresh instance has evaluated
        // nothing, and leaving them would let the inspector preview show a dead node's numbers
        // until the new one reports its own.
        entry.evaluated.clear();
        entry.param_errors.clear();
        // `bindings` are left untouched — their compiled handles are evaluator-owned and may only
        // be dropped through `release_entry_bindings`.

        // A wire into or out of a slot the reshape retired can never propagate, and cannot be
        // repaired by the user either — the slot is not in the palette any more. Silently keeping
        // it is the least diagnosable outcome: the editor draws a cable the runtime ignores. It
        // goes here, under the same lock as the swap, through `remove_link` so the scheduler's
        // derived state stays consistent.
        let orphaned: Vec<(Uid, &'static str, Uid, &'static str)> = self
            .links
            .iter()
            .filter(|l| {
                (l.node_in == uid && self.input_slot_type(uid, l.slot_in).is_none())
                    || (l.node_out == uid && self.output_slot_type(uid, l.slot_out).is_none())
            })
            .map(|l| (l.node_out, l.slot_out, l.node_in, l.slot_in))
            .collect();
        for (out, so, into, si) in orphaned {
            let _ = self.remove_link(out, so, into, si);
        }
        // The rebirth renamed every one of this node's services (§3.1), so a binding reading one
        // is holding a name that no longer resolves. Re-resolved, not patched: the reshape may
        // also have retired the very output slot the reference named.
        let name = self.nodes[&uid].name.clone();
        self.rebind_naming(&name);
        Ok(())
    }

    pub fn update_param(
        &mut self,
        uid: Uid,
        group: &str,
        name: &str,
        value: Param,
    ) -> Result<(), String> {
        let entry = self
            .nodes
            .get_mut(&uid)
            .ok_or_else(|| format!("no such node {uid}"))?;
        if entry.params.load().get(group).is_none() {
            return Err(format!("no such param group `{group}`"));
        }
        edit_params(entry, |p| {
            p.entry(group.to_string()).or_default().insert(name.to_string(), value.clone());
        });
        // §3.4: a LITERAL on a param the node is DRIVING unbinds it, which is what the node does
        // with the `SetParam` this write sends — so the graph must mean the same by it, or the two
        // records disagree about whether the param is driven. Unbinding also drops this node from
        // the producer's target set (§5.3: an expression reference IS a link), so the producer
        // stops ringing a doorbell nobody reads.
        //
        // An ENABLED binding only. A disabled one drives nothing — it is source the fx toggle is
        // holding for the user, and every node in the patch carries one on `common.max_frequency`
        // (`globals.default_ufreq`, waiting to be switched on). Unbinding those would make typing a
        // number into a consumer's rate cap permanently delete the patch-rate expression, and
        // persist the loss to the `.gfi`.
        let key = ParamKey::new(group, name);
        if self.nodes[&uid].bindings.get(&key).is_some_and(|b| b.enabled) {
            self.unbind(uid, &key);
        }
        // The record has moved and the node has been told (§5.1) — nothing else happens here.
        // `on_param_changed` runs where the node instance lives, which is its own thread: a hook
        // that mirrors the value onto a field, or reopens a device, no longer runs under the graph
        // mutex and no longer rides this reply. Its failure arrives as a fault (§8.4).
        self.notify_param(uid, &key);
        Ok(())
    }

    /// Ask the node to re-enumerate a refreshable `Str` param's options — the ⟳ button behind a
    /// device or stream picker, whose choices are only knowable at runtime.
    ///
    /// It ALWAYS answers `Ok(None)` (§8.5). The hook runs on the node's own thread, which is the
    /// whole point of the move — a multi-second device scan no longer stalls anything — so the RPC
    /// that asked cannot carry the list back. The options arrive as
    /// [`runtime::Status::RefreshOptions`] and reach the client on the doc re-mirror the status
    /// worker drives. `Err` is still a real refusal: an unknown node, an unknown param, or one the
    /// type never declared refreshable (the UI shows no button for one).
    ///
    /// Not a command: nothing persisted changes (options never reach the `.gfi`), so there is
    /// nothing to undo.
    pub fn refresh_param(
        &mut self,
        uid: Uid,
        group: &str,
        name: &str,
    ) -> Result<Option<Vec<String>>, String> {
        let entry = self.nodes.get(&uid).ok_or_else(|| format!("no such node {uid}"))?;
        let live = entry.params.load_full();
        let param = goofi_node::param(&live, group, name)
            .ok_or_else(|| format!("no such param `{group}.{name}`"))?;
        if !matches!(param, Param::Str { refresh: true, .. }) {
            return Err(format!("param `{group}.{name}` is not refreshable"));
        }
        self.wire.send(uid, runtime::Control::RefreshParam { key: ParamKey::new(group, name) });
        Ok(None)
    }

    /// Bind (or unbind) a param to an expression. An **empty** `source` unbinds (the stored
    /// literal is used again). A non-empty source with `enabled == false` PRESERVES the
    /// authored binding, disabled — so a UI fx toggle-off then -on keeps the user's code —
    /// but is not compiled or evaluated. An enabled non-empty source is (re)compiled via the
    /// injected evaluator; a compile error is stored as the binding's field error (surfaced
    /// on the node) rather than rejecting the RPC — the frontend keeps the source so the
    /// user can fix it. Returns Err for an unknown node or an unknown `(group, name)` param.
    pub fn set_expression(
        &mut self,
        uid: Uid,
        group: &str,
        name: &str,
        source: &str,
        enabled: bool,
        triggers_process: bool,
    ) -> Result<(), String> {
        if !self.nodes.contains_key(&uid) {
            return Err(format!("no such node {uid}"));
        }
        let key = ParamKey::new(group, name);
        // Only an empty source is a true unbind, and `unbind` owns the release on that path — so it
        // goes FIRST, above the release below. Releasing here as well handed the evaluator two
        // `release` calls for one handle, and `ExprEvaluator` is a public trait an implementation
        // may reasonably treat as a refcount.
        if source.trim().is_empty() {
            self.unbind(uid, &key);
            self.notify_param(uid, &key);
            return Ok(());
        }
        // Release any prior compiled handle first — this path REPLACES it.
        if let Some(prev) = self.nodes.get(&uid).and_then(|e| e.bindings.get(&key)) {
            if let (Some(ev), Some(id)) = (&self.evaluator, prev.id) {
                ev.release(id);
            }
        }
        // A non-empty source binds a real param — reject a dangling binding (invisible in
        // the descriptor, unclearable from the UI, phantom scheduling edges), like
        // update_param guards param existence.
        if goofi_node::param(&self.nodes[&uid].params.load(), group, name).is_none() {
            return Err(format!("no such param `{group}/{name}`"));
        }
        let bind_id = self.bind_id(uid, &key);
        // §5.3's four steps, in order: rewrite, resolve, compile the REWRITTEN source, ship. The
        // scan runs even for a DISABLED binding, because `terms` is what says which bindings a
        // later rename or globals edit has to re-resolve — a disabled binding that an
        // fx toggle re-enables must come back resolved against the graph as it is then. What a
        // disabled binding does not get is variables, a handle, or a place in anyone's target set.
        let scanned = expr_rewrite::rewrite(source);
        let terms = scanned.as_ref().map(|(_, vars)| vars.clone()).unwrap_or_default();
        let (rewritten, vars, mut error) = match (enabled, scanned) {
            (true, Ok((rewritten, refs))) => {
                let vars = self.resolve_vars(uid, &key, &refs);
                let error = vars.iter().find_map(|v| match v {
                    BoundVar::Missing { reason, .. } => Some(reason.clone()),
                    _ => None,
                });
                (rewritten, vars, error)
            }
            (true, Err(e)) => (source.to_string(), Vec::new(), Some(e.0)),
            (false, _) => (source.to_string(), Vec::new(), None),
        };
        let id = match (&self.evaluator, enabled, error.is_none()) {
            (Some(ev), true, true) => match ev.compile(&rewritten) {
                Ok(c) => Some(c.id),
                Err(e) => {
                    error = Some(e.0);
                    None
                }
            },
            (None, true, _) => {
                error = Some("no expression evaluator available".to_string());
                None
            }
            _ => None,
        };
        let binding = ExprBinding {
            source: source.to_string(),
            enabled,
            triggers_process,
            id,
            rewritten,
            vars,
            terms,
            bind_id,
            error,
        };
        if let Some(e) = self.nodes.get_mut(&uid) {
            e.bindings.insert(key, binding);
        }
        self.replan_binding(uid, bind_id);
        Ok(())
    }

    /// Drop a binding and release its compiled handle — the shared tail of an empty
    /// `set_expression` and of a literal write over a bound param (§3.4: both mean unbind, and the
    /// graph must mean by it what the node does).
    ///
    /// It does NOT re-plan: its callers do, exactly once, through [`Self::notify_param`]. A second
    /// `begin` on the same key cancels the first mid-sequence, so a re-plan here would leave the
    /// producer-shrink it had already sent waiting on an ack nothing listens for.
    fn unbind(&mut self, uid: Uid, key: &ParamKey) {
        let Some(binding) = self.nodes.get_mut(&uid).and_then(|e| e.bindings.remove(key)) else {
            return;
        };
        if let (Some(ev), Some(id)) = (&self.evaluator, binding.id) {
            ev.release(id);
        }
    }

    /// Tell the node what this param is now (§5.1). Storing the record is only HALF of a param
    /// edit: the `ArcSwap` is the read path, and a node parked with `next_wake() == None` is never
    /// rung by a bare pointer swap, so the write has to be announced as well.
    ///
    /// A param's [`runtime::plan::Slot::Bind`] subscription is that channel whether or not it
    /// currently holds variables — with none, §4's phases 1 and 3 have no recipients and phase 2
    /// carries the literal. ONE re-plan per edit: a second `begin` on the same key cancels the
    /// first mid-sequence, so the producer-shrink it had already sent would be waiting on an ack
    /// nothing is listening for.
    fn notify_param(&mut self, uid: Uid, key: &ParamKey) {
        let bind_id = self.bind_id(uid, key);
        self.replan_binding(uid, bind_id);
    }

    /// This PARAM's index into [`Self::bind_keys`], minting one the first time the graph has
    /// anything to say about it. Keyed by param rather than by binding because [`runtime::plan::Slot::Bind`]
    /// is the param's notification channel, which outlives any one binding on it: an unbind's own
    /// wire sequence still has to compose the `SetParam` that says the param is a literal again, and
    /// a param that was never bound still has to hear its edits. Append-only and cleared only by a
    /// whole-graph `clear`, for the same reason.
    fn bind_id(&mut self, uid: Uid, key: &ParamKey) -> usize {
        if let Some(b) = self.nodes.get(&uid).and_then(|e| e.bindings.get(key)) {
            return b.bind_id;
        }
        if let Some(at) = self.bind_keys.iter().position(|(u, k)| *u == uid && k == key) {
            return at;
        }
        self.bind_keys.push((uid, key.clone()));
        self.bind_keys.len() - 1
    }

    /// Resolve a rewrite's variables against the graph: a producer output, a global's value, or the
    /// reason neither could be found. Event ids are drawn from §3.2's `65..=128` expression budget,
    /// lowest free first among the ids this node's OTHER bindings already hold — `key`'s own ids are
    /// being replaced and are therefore free.
    fn resolve_vars(&self, consumer: Uid, key: &ParamKey, refs: &[expr_rewrite::VarRef]) -> Vec<BoundVar> {
        let mut taken: Vec<runtime::EventId> = self
            .nodes
            .get(&consumer)
            .into_iter()
            .flat_map(|e| e.bindings.iter().filter(|(k, _)| *k != key))
            .flat_map(|(_, b)| &b.vars)
            .filter_map(|v| match v {
                BoundVar::Stream { event_id, .. } => Some(*event_id),
                _ => None,
            })
            .collect();
        refs.iter()
            .map(|r| match r {
                expr_rewrite::VarRef::Global { var, key } => match self.globals.get(key) {
                    Some(value) => BoundVar::Value { var: var.clone(), value: global_as_param(value) },
                    None => BoundVar::Missing {
                        var: var.clone(),
                        reason: format!("global `{key}` is not defined"),
                    },
                },
                expr_rewrite::VarRef::Node { var, name, slot } => {
                    match self.resolve_stream(name.as_str(), slot.as_deref()) {
                        Err(reason) => BoundVar::Missing { var: var.clone(), reason },
                        Ok((producer, slot)) => match next_event_id(&taken) {
                            None => BoundVar::Missing {
                                var: var.clone(),
                                reason: "too many expression references on this node".to_string(),
                            },
                            Some(event_id) => {
                                taken.push(event_id);
                                BoundVar::Stream { var: var.clone(), producer, slot, event_id }
                            }
                        },
                    }
                }
            })
            .collect()
    }

    /// The producer output a `nd('name')` / `nd('name').slot` term names, or why it names none. A
    /// bare reference to a multi-output node is refused HERE rather than at eval, where it used to
    /// raise from inside the proxy — the graph is what knows how many outputs a node has.
    fn resolve_stream(&self, name: &str, slot: Option<&str>) -> Result<(Uid, &'static str), String> {
        let uid = self.uid_by_name(name).ok_or_else(|| format!("no node named `{name}`"))?;
        let outputs = self.nodes[&uid].manifest.outputs;
        match slot {
            Some(slot) => outputs
                .iter()
                .find(|o| o.name == slot)
                .map(|o| (uid, o.name))
                .ok_or_else(|| format!("node `{name}` has no output `{slot}`")),
            None if outputs.len() == 1 => Ok((uid, outputs[0].name)),
            None if outputs.is_empty() => Err(format!("node `{name}` has no outputs")),
            None => Err(format!("nd('{name}') is ambiguous: it has multiple outputs; use nd('{name}').slot")),
        }
    }

    /// The expression binding on a param, for the bridge descriptor + `.gfi` (or `None`
    /// if the param is a plain literal).
    pub fn param_expression(&self, uid: Uid, group: &str, name: &str) -> Option<ExprInfo> {
        let b = self.nodes.get(&uid)?.bindings.get(&ParamKey::new(group, name))?;
        Some(ExprInfo {
            source: b.source.clone(),
            enabled: b.enabled,
            triggers_process: b.triggers_process,
            error: b.error.clone(),
        })
    }

    /// Every expression binding on a node as `(group, name, source, enabled, triggers)` — the
    /// bindings a delete's inverse must re-apply (params alone carry only the literal value, so
    /// without this a restored node loses its live-driven params).
    pub fn param_bindings(&self, uid: Uid) -> Vec<(String, String, String, bool, bool)> {
        self.nodes
            .get(&uid)
            .map(|e| {
                e.bindings
                    .iter()
                    .map(|(k, b)| {
                        (k.group.clone(), k.name.clone(), b.source.clone(), b.enabled, b.triggers_process)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The current values of the params driven by an ENABLED expression binding on `uid`,
    /// as `(group, name, value)`. Empty when the node has no active expressions. Feeds the
    /// live inspector preview (`param_values` event) so a bound param's displayed value
    /// tracks each re-evaluation, not just the value captured at edit time. A disabled
    /// binding is excluded — its value is the static literal, already on the descriptor.
    pub fn expression_values(&self, uid: Uid) -> Vec<(&str, &str, &Param)> {
        let Some(entry) = self.nodes.get(&uid) else {
            return Vec::new();
        };
        entry
            .evaluated
            .iter()
            .filter(|(key, _)| entry.bindings.get(key).is_some_and(|b| b.enabled))
            .map(|(key, p)| (key.group.as_str(), key.name.as_str(), p))
            .collect()
    }

    /// Resolve a node display name to its uid (for `nd('name')` references).
    fn uid_by_name(&self, name: &str) -> Option<Uid> {
        self.nodes.iter().find(|(_, e)| e.name == name).map(|(u, _)| *u)
    }

    /// Resolve an output slot name to its `&'static` manifest name.
    fn resolve_output(&self, uid: Uid, slot: &str) -> Option<&'static str> {
        self.find_output(uid, slot).map(|o| o.name)
    }
    fn resolve_input(&self, uid: Uid, slot: &str) -> Option<&'static str> {
        self.find_input(uid, slot).map(|i| i.name)
    }

    /// Whether input `slot` on node `uid` is a `multi` (variadic) slot — i.e. it
    /// accepts many wires and lives in `multi_inputs` rather than `inputs`.
    fn is_multi_input(&self, uid: Uid, slot: &str) -> bool {
        self.find_input(uid, slot).is_some_and(|i| i.multi)
    }

    /// The wire currently feeding a SINGLE input `(node_in, slot)` — the wire an `add_link` would
    /// evict. `None` for a multi input (append, no eviction) or an empty input. Lets the `AddLink`
    /// command capture the displaced wire so its inverse restores it.
    pub fn single_input_source(&self, node_in: Uid, slot: &str) -> Option<(Uid, &'static str)> {
        let slot = self.resolve_input(node_in, slot)?;
        if self.is_multi_input(node_in, slot) {
            return None;
        }
        self.links
            .iter()
            .find(|l| l.node_in == node_in && l.slot_in == slot)
            .map(|l| (l.node_out, l.slot_out))
    }

    /// Does this exact (resolved) wire already exist? False if either slot fails to resolve
    /// (`add_link` will surface the real error). Lets a command detect an idempotent AddLink no-op
    /// so its inverse can be a no-op too, instead of destroying the pre-existing wire.
    pub fn has_link(&self, node_out: Uid, slot_out: &str, node_in: Uid, slot_in: &str) -> bool {
        let (Some(slot_out), Some(slot_in)) =
            (self.resolve_output(node_out, slot_out), self.resolve_input(node_in, slot_in))
        else {
            return false;
        };
        self.links.contains(&Link { node_out, slot_out, node_in, slot_in })
    }

    pub fn add_link(
        &mut self,
        node_out: Uid,
        slot_out: &str,
        node_in: Uid,
        slot_in: &str,
    ) -> Result<(), String> {
        // Each slot's DECL, taken once: it carries both the `&'static` name a link is keyed by and
        // the dtype the check below needs, so there is no second lookup that could fail on its own.
        let out = self
            .find_output(node_out, slot_out)
            .ok_or_else(|| format!("no output slot `{slot_out}` on {node_out}"))?;
        let inp = self
            .find_input(node_in, slot_in)
            .ok_or_else(|| format!("no input slot `{slot_in}` on {node_in}"))?;
        let (slot_out, slot_in) = (out.name, inp.name);
        // A cross-dtype cable can never carry data — propagation writes the producer's frame into
        // an input the consumer reads with the wrong accessor, so the consumer sits empty forever.
        // Refuse it here, at the one door every link authoring path goes through (the canvas, the
        // boundary resolution, a `.gfi` restore, an agent), naming both ends and both dtypes.
        if out.kind != inp.kind {
            let label = |uid: Uid, slot: &str| {
                format!("{}.{slot}", self.name(uid).unwrap_or("?"))
            };
            return Err(format!(
                "cannot link {} ({}) to {} ({}): the slots carry different data types",
                label(node_out, slot_out),
                out.kind.name(),
                label(node_in, slot_in),
                inp.kind.name(),
            ));
        }

        let new = Link {
            node_out,
            slot_out,
            node_in,
            slot_in,
        };
        if self.links.contains(&new) {
            return Ok(()); // idempotent
        }
        // A multi slot accepts many wires and keeps them in connection order, which IS `links`'
        // own order; a single input takes one, so a second wire EVICTS the first. The node hears
        // both as one declarative set — §4's "a displaced single-input wire needs no special case".
        if !self.is_multi_input(node_in, slot_in) {
            self.links
                .retain(|l| !(l.node_in == node_in && l.slot_in == slot_in));
        }
        self.links.push(new);
        self.replan_slot(node_in, slot_in);
        Ok(())
    }

    pub fn remove_link(
        &mut self,
        node_out: Uid,
        slot_out: &str,
        node_in: Uid,
        slot_in: &str,
    ) -> Result<(), String> {
        let before = self.links.len();
        self.links.retain(|l| {
            !(l.node_out == node_out
                && l.slot_out == slot_out
                && l.node_in == node_in
                && l.slot_in == slot_in)
        });
        if self.links.len() == before {
            return Err("no such link".into());
        }
        if let Some(slot_in) = self.resolve_input(node_in, slot_in) {
            self.replan_slot(node_in, slot_in);
        }
        Ok(())
    }

    // ── The wire plane (spec §3.1, §4) ──────────────────────────────────────────────────────
    // The async runtime's topology side: what each node is told about its slots, and in what order.
    // `add_link` and `remove_link` above already replan through it; it stays inert until a node's
    // control channel is attached, which nothing does until the cutover.

    /// Register the graph's end of one node's control channel. §4's birth barrier: this happens on
    /// [`runtime::Status::Ready`] and never at birth, because a `Control` published before the
    /// node's own subscriber exists is lost and pub/sub has no history.
    ///
    /// Attaching RE-PLANS every slot this node touches, from an empty base. A node that was not
    /// addressable when those slots were planned had its message dropped — `dispatch` skips a uid
    /// with no channel so a partially attached graph converges instead of stalling — while the diff
    /// base moved anyway, so nothing would ever resend it. A node that has just become addressable
    /// knows nothing, whatever the graph planned meanwhile. This is the shape the birth barrier
    /// takes when it lands: `Status::Ready` is the moment a node becomes addressable.
    pub fn attach_control_sink(&mut self, uid: Uid, sink: Arc<dyn runtime::ControlSink>) {
        self.wire.attach(uid, sink);
        for (consumer, slot) in self.slots_touching(uid) {
            self.wire.forget_planned((consumer, slot));
            self.replan(consumer, slot);
        }
    }

    /// Every consumer subscription whose wiring names `uid` — the input slots it consumes on and
    /// feeds, and the expression bindings on either end of it. A subscription is named once however
    /// many wires it has.
    fn slots_touching(&self, uid: Uid) -> Vec<runtime::plan::SlotKey> {
        let mut slots: Vec<runtime::plan::SlotKey> = Vec::new();
        for link in self.links.iter().filter(|l| l.node_in == uid || l.node_out == uid) {
            let key = (link.node_in, runtime::plan::Slot::In(link.slot_in));
            if !slots.contains(&key) {
                slots.push(key);
            }
        }
        // Every param channel this graph has ever spoken on for `uid`, bound or not (§3.4: a
        // literal edit is announced too). A node becoming addressable is the FIRST moment anything
        // it was told can actually arrive — `add_node` answers before the barrier lifts, so the
        // ordinary `add_node(); update_param()` pair falls entirely inside the window where a
        // `Control` is published to a subscriber that does not exist yet and is lost.
        for (at, (owner, _)) in self.bind_keys.iter().enumerate() {
            let key = (*owner, runtime::plan::Slot::Bind(at));
            if *owner == uid && !slots.contains(&key) {
                slots.push(key);
            }
        }
        // §5.3: an expression reference is a link, so a node becoming addressable owes its bindings
        // the same re-plan its input slots get — both ends of one, since a producer that could not
        // be reached was never told to ring the reader.
        for (consumer, entry) in &self.nodes {
            for binding in entry.bindings.values() {
                let touches = *consumer == uid
                    || binding.vars.iter().filter_map(BoundVar::wire).any(|(p, _)| p == uid);
                let key = (*consumer, runtime::plan::Slot::Bind(binding.bind_id));
                if touches && !slots.contains(&key) {
                    slots.push(key);
                }
            }
        }
        slots
    }

    /// The generation the node at `uid` was born at — the third component of its service names, and
    /// what keeps a rebirth clear of a predecessor whose teardown does not block.
    pub fn node_generation(&self, uid: Uid) -> u64 {
        self.wire.generation(uid)
    }

    /// One output slot's data service name — the whole of a wire's identity, which is why a slot
    /// message carries names and never a source uid. Public because it is also the `/data` plane's
    /// subscribe address: a viewer resolves `(uid, slot)` here once and is lock-free after (§7).
    pub fn output_service_of(&self, uid: Uid, slot: &str) -> runtime::ServiceName {
        runtime::output_service(&self.service_base_of(uid), slot)
    }

    /// One node's doorbell service name.
    pub(crate) fn door_of(&self, uid: Uid) -> runtime::ServiceName {
        runtime::door_service(&self.service_base_of(uid))
    }

    fn service_base_of(&self, uid: Uid) -> String {
        runtime::service_base(&self.instance, uid, self.wire.generation(uid))
    }

    /// Plan an input slot's full desired wire set and run the three-phase sequence (§4). Every link
    /// change to a slot comes through here, which is why a displaced single-input wire needs no
    /// special case anywhere: the consumer's new set is simply the new producer.
    pub(crate) fn replan_slot(&mut self, uid: Uid, slot: &'static str) {
        self.replan(uid, runtime::plan::Slot::In(slot));
    }

    /// The same for an expression binding (§5.3), whose subscription set is the producers its
    /// variables resolved to. Keyed by the binding's id rather than by its `ParamKey`, so an unbind
    /// can still be planned after the binding itself is gone.
    fn replan_binding(&mut self, uid: Uid, bind_id: usize) {
        self.replan(uid, runtime::plan::Slot::Bind(bind_id));
    }

    fn replan(&mut self, uid: Uid, slot: runtime::plan::Slot) {
        let key = (uid, slot);
        let desired = self.desired_wires(key);
        let previous = self.wire.planned(key);
        let removed = previous.iter().copied().filter(|w| !desired.contains(w)).collect();
        let added = desired.iter().copied().filter(|w| !previous.contains(w)).collect();
        self.wire.begin(key, desired, removed, added);
        self.advance_wire(key);
    }

    /// Answer one node's ack — the status-drain worker's door into the sequence. Completing a phase
    /// is the only thing that starts the next one.
    pub fn wire_ack(&mut self, seq: u64, ok: Result<(), String>) {
        if let Some(key) = self.wire.ack(seq, ok) {
            self.advance_wire(key);
        }
    }

    /// Take every report waiting on every live node's status service and apply it — the
    /// status-drain worker's engine-side half (§6.2). Answers how many landed, so a caller can tell
    /// a quiet graph from one it has stopped hearing from.
    ///
    /// The worker owns the loop and the events; this owns the graph. A test drives the same door,
    /// which is what makes "the node reported it" observable without a bridge.
    pub fn drain_status(&mut self) -> usize {
        let channels: Vec<(Uid, Arc<runtime::NodeChannel>)> = self
            .nodes
            .iter()
            .filter_map(|(uid, e)| e.host.channel.clone().map(|c| (*uid, c)))
            .collect();
        let mut applied = 0;
        for (uid, channel) in channels {
            for status in channel.drain_status() {
                self.apply_status(uid, status);
                applied += 1;
            }
        }
        applied
    }

    /// Apply one node's report to the graph — the status-drain worker's whole graph-side job
    /// (§6.2). Every variant is a TRANSITION the node stamped itself, so nothing here diffs.
    ///
    /// The faults land in the very fields `last_error`/`node_stage` already read, because §6 is
    /// explicit that `runtime_overlay` keeps working verbatim: what changes is how the graph LEARNS
    /// a node's state, not how it projects it.
    pub fn apply_status(&mut self, uid: Uid, status: runtime::Status) {
        // An ack is the PLANNER's, not an entry's, and it must still land after the node it came
        // from is gone — or a sequence parks forever on a message nobody will answer.
        if let runtime::Status::Ack { seq, ok } = status {
            self.wire_ack(seq, ok);
            return;
        }
        // …and so is `Ready`: it is what makes a node addressable, and the sink it attaches is the
        // graph's, not the entry's.
        if matches!(status, runtime::Status::Ready) {
            if let Some(channel) = self.nodes.get(&uid).and_then(|e| e.host.channel.clone()) {
                self.attach_control_sink(uid, channel);
            }
            return;
        }
        let Some(entry) = self.nodes.get_mut(&uid) else { return };
        match status {
            // Consumed above. An inert arm rather than an `unreachable!`: this runs under the mutex
            // the bridge locks with `.lock().unwrap()` throughout, so a panic site here would
            // poison the control plane rather than cost one report — and "genuinely unreachable"
            // is a claim about today's callers, which is what B's hardening pass stopped trusting.
            runtime::Status::Ack { .. } | runtime::Status::Ready => {}
            runtime::Status::Stage { stage } => entry.stage = stage.as_str(),
            runtime::Status::Ufreq { hz } => entry.ufreq = Some(hz),
            // The options are the node's answer to a refresh (§8.5), and they land in the RECORD
            // rather than in a reply: the RPC that asked has already returned.
            runtime::Status::RefreshOptions { key, options } => {
                if let Some(options) = options {
                    edit_params(entry, |p| {
                        if let Some(Param::Str { options: slot, .. }) =
                            p.get_mut(&key.group).and_then(|g| g.get_mut(&key.name))
                        {
                            *slot = Some(options);
                        }
                    });
                }
            }
            runtime::Status::Fault { fault } => match fault {
                // A clean run clears Setup/Process/Boot together and never touches a binding
                // error, which only that binding evaluating successfully clears (§6).
                None => {
                    entry.setup_error = None;
                    entry.last_error = None;
                }
                Some(runtime::NodeFault::Setup { msg, .. }) => entry.setup_error = Some(msg),
                // `Boot` shares `last_error` with `Process` on purpose: the graph projects one
                // node-level error string, and the two differ only in which side of the manager
                // thread failed — which the node has already said in the message.
                Some(runtime::NodeFault::Process { msg, .. } | runtime::NodeFault::Boot { msg, .. }) => {
                    entry.last_error = Some(msg)
                }
                // The roll-up, not the record: a node reports `Expr` only as the badge-level
                // derivation of its binding-error map, and that map arrives as `BindingErrors`.
                Some(runtime::NodeFault::Expr { .. }) => {}
            },
            runtime::Status::BindingErrors { errors } => {
                for (key, msg) in errors {
                    // A bound param renders its error on its own inspector field; an unbound one
                    // has no such field, so it goes to the node-level channel instead of nowhere.
                    match entry.bindings.get_mut(&key) {
                        Some(b) => b.error = msg,
                        None => match msg {
                            Some(msg) => {
                                entry.param_errors.insert(key, msg);
                            }
                            None => {
                                entry.param_errors.shift_remove(&key);
                            }
                        },
                    }
                }
            }
            runtime::Status::ParamValues { evaluated } => {
                entry.evaluated = evaluated.into_iter().collect();
            }
        }
        self.stamp_error_onset(uid);
    }

    /// Note when this node's error first read the way it does now — the clock [`Graph::error_age`]
    /// reports. Derived from [`entry_error`] rather than written at each site that can set one, so
    /// a process failure, a setup failure and a binding failure are all stamped by the same rule,
    /// and the stamp cannot outlive the error it belongs to. Run after every applied report, which
    /// is the only thing that can change any of the three.
    fn stamp_error_onset(&mut self, uid: Uid) {
        let Some(e) = self.nodes.get_mut(&uid) else { return };
        let current = entry_error(e);
        if e.error_since.as_ref().map(|(m, _)| m.as_str()) != current {
            e.error_since = current.map(|m| (m.to_string(), Instant::now()));
        }
    }

    /// Walk the phases until one has something to send. A phase with no recipients is skipped rather
    /// than sent empty, or the sequence would park on an ack for a message that says nothing.
    fn advance_wire(&mut self, key: runtime::plan::SlotKey) {
        while let Some(phase) = self.wire.step(key) {
            let messages = self.compose_wire(key, phase);
            if self.wire.dispatch(key, messages) {
                return;
            }
        }
    }

    /// One phase's messages. The `OutSlot` phases are built from the graph as it stands NOW rather
    /// than at plan time — a producer's target set can be changed by another slot's sequence between
    /// two phases of this one, and the message that goes out must carry the truth at the moment it
    /// goes. `Apply` carries the sequence's own stored `desired`, which is what the phases are
    /// ordered around and must not shift underneath them.
    fn compose_wire(
        &self,
        key: runtime::plan::SlotKey,
        phase: runtime::plan::Phase,
    ) -> Vec<(Uid, runtime::Control)> {
        match phase {
            // Phase 2 is the SUBSCRIBE, whichever kind of consumer this is: an input slot receives
            // its full service set, a binding receives the whole re-resolved expression. Both are
            // declarative, and both are what the producer phases are ordered around.
            runtime::plan::Phase::Apply => match key.1 {
                runtime::plan::Slot::In(slot) => {
                    let services = self
                        .wire
                        .desired(key)
                        .iter()
                        .map(|(uid, slot)| self.output_service_of(*uid, slot))
                        .collect();
                    vec![(key.0, runtime::Control::InSlot { slot: slot.to_string(), services })]
                }
                runtime::plan::Slot::Bind(id) => self.compose_set_param(key.0, id).into_iter().collect(),
            },
            runtime::plan::Phase::Shrink | runtime::plan::Phase::Grow => self
                .wire
                .recipients(key, phase)
                .into_iter()
                .map(|(uid, slot)| {
                    let targets = self.out_targets(uid, slot);
                    (uid, runtime::Control::OutSlot { slot: slot.to_string(), targets })
                })
                .collect(),
        }
    }

    /// The `SetParam` a binding's phase 2 carries: the rewritten source with its resolved variables
    /// while the binding stands, and the param's LITERAL once it does not — an unbind is a param
    /// going back to its authored number, and §3.4 makes that the message that says so.
    fn compose_set_param(&self, uid: Uid, bind_id: usize) -> Option<(Uid, runtime::Control)> {
        let (owner, key) = self.bind_keys.get(bind_id)?;
        if *owner != uid {
            return None;
        }
        let entry = self.nodes.get(&uid)?;
        let value = match entry.bindings.get(key).filter(|b| b.enabled) {
            Some(b) => runtime::ParamValue::Expr {
                source: b.rewritten.clone(),
                vars: b.vars.iter().map(|v| self.wire_var(v)).collect(),
                trigger: b.triggers_process,
                // The graph compiled it, the node evaluates it (§2.1) — one handle, so the two ends
                // can never be evaluating different source.
                id: b.id,
            },
            None => {
                runtime::ParamValue::Literal(goofi_node::param(&entry.params.load(), &key.group, &key.name)?.clone())
            }
        };
        Some((uid, runtime::Control::SetParam { key: key.clone(), value }))
    }

    /// A resolved variable as the NODE sees it: a service name rather than a uid, because a node
    /// addresses a producer by service and cannot resolve anything for itself (§5.3).
    fn wire_var(&self, var: &BoundVar) -> runtime::Var {
        match var {
            BoundVar::Stream { var, producer, slot, event_id } => runtime::Var::Stream {
                name: var.clone(),
                service: self.output_service_of(*producer, slot),
                event_id: *event_id,
            },
            BoundVar::Value { var, value } => runtime::Var::Value { name: var.clone(), value: value.clone() },
            BoundVar::Missing { var, reason } => {
                runtime::Var::Missing { name: var.clone(), reason: reason.clone() }
            }
        }
    }

    /// Every producer a consumer subscription feeds from, in wire order.
    ///
    /// For an input slot: many for a `multi` slot, at most one for a single one. `links` is the
    /// order for both — a wire is appended there as it is added, which IS `Inputs::get_multi`'s
    /// connection order, and the per-wire cells the tick path keeps are rebuilt from this same list
    /// by `restart_node`. Reading those cells here instead would be a second record of one order.
    ///
    /// For a binding: the producers its variables resolved to, in variable order.
    ///
    /// Stubs never appear — a link's endpoints are resolved real nodes by the time it is recorded.
    /// A slot with no event id does not appear either: §3.2 budgets 1..=64 for input slots, and a
    /// wire subscribed by a consumer that no producer can ring is worse than no wire at all. No
    /// manifest in this codebase declares anything like 64 inputs.
    fn desired_wires(&self, key: runtime::plan::SlotKey) -> Vec<runtime::plan::Wire> {
        match key.1 {
            runtime::plan::Slot::In(slot) => {
                if self.input_event_id(key.0, slot).is_none() {
                    return Vec::new();
                }
                self.links
                    .iter()
                    .filter(|l| l.node_in == key.0 && l.slot_in == slot)
                    .map(|l| (l.node_out, l.slot_out))
                    .collect()
            }
            runtime::plan::Slot::Bind(id) => self
                .binding_of(key.0, id)
                .filter(|b| b.enabled)
                .map(|b| b.vars.iter().filter_map(BoundVar::wire).collect())
                .unwrap_or_default(),
        }
    }

    /// The binding a planner id names, if it still exists.
    fn binding_of(&self, uid: Uid, bind_id: usize) -> Option<&ExprBinding> {
        let (owner, key) = self.bind_keys.get(bind_id)?;
        (*owner == uid).then(|| self.nodes.get(&uid)?.bindings.get(key)).flatten()
    }

    /// Every doorbell one output slot rings, with the event id that says why the far node woke —
    /// the UNION of this slot's wire consumers and its expression subscribers (§5.3). One set,
    /// because a producer cannot tell an `nd()` reader from a wired consumer and does not need to.
    fn out_targets(&self, producer: Uid, slot: &'static str) -> Vec<(runtime::ServiceName, runtime::EventId)> {
        // §4's ordering guarantee is per TARGET, not per sequence: a consumer whose own sequence
        // has not applied this wire yet does not exist as a subscriber, so naming it here — because
        // some other consumer's sequence reached phase 3 first — is exactly the "told to notify a
        // subscriber that does not exist yet" the phases are ordered to prevent. Its own phase 3
        // names it, against the same live target set.
        let wired = self
            .links
            .iter()
            .filter(|l| l.node_out == producer && l.slot_out == slot)
            .filter(|l| {
                !self.wire.unapplied((l.node_in, runtime::plan::Slot::In(l.slot_in)), (producer, slot))
            })
            .filter_map(|l| Some((self.door_of(l.node_in), self.input_event_id(l.node_in, l.slot_in)?)));
        let bound = self.nodes.iter().flat_map(|(consumer, entry)| {
            entry.bindings.values().filter(|b| b.enabled).flat_map(move |b| {
                b.vars
                    .iter()
                    .filter(move |v| v.wire() == Some((producer, slot)))
                    .filter(move |_| {
                        !self.wire.unapplied((*consumer, runtime::plan::Slot::Bind(b.bind_id)), (producer, slot))
                    })
                    .filter_map(move |v| match v {
                        BoundVar::Stream { event_id, .. } => Some((self.door_of(*consumer), *event_id)),
                        _ => None,
                    })
            })
        });
        wired.chain(bound).collect()
    }

    /// An input slot's event id: its position in the manifest's inputs, past `EventId(0)`, which is
    /// the control channel's (§3.2). `None` beyond the 64 ids the budget gives input slots.
    fn input_event_id(&self, uid: Uid, slot: &str) -> Option<runtime::EventId> {
        let at = self.nodes.get(&uid)?.manifest.inputs.iter().position(|s| s.name == slot)?;
        (at < 64).then_some(at as runtime::EventId + 1)
    }

    /// Remove all nodes and links.
    pub fn clear(&mut self) {
        // Release each node's compiled expression handles before dropping them (load_doc
        // goes through here, so a File→Open cycle can't leak the evaluator's registry).
        for e in self.nodes.values() {
            self.release_entry_bindings(e);
        }
        self.nodes.clear();
        self.links.clear();
        self.scopes.clear();
        self.scope_of.clear();
        // The channels addressed nodes that no longer exist; the generations stay, because they are
        // what keeps whatever is born at those uids next clear of what just died.
        self.wire.reset_channels();
        // …and the binding ids went with the sequences that named them.
        self.bind_keys.clear();
        // Globals are patch content: a load starts from a fresh system-seeded store (load_doc then
        // repopulates user globals from the `.gfi`). `dyn_types` stays (catalog, not content).
        self.globals_mut(|g| {
            *g = goofi_core::globals::GlobalStore::new();
            Ok(())
        })
        .expect("re-seeding cannot fail");
        // The node clock belongs to the PATCH, not the process: a patch loaded an hour in must
        // compute what it would have computed at boot. Safe only because every node — and every
        // ufreq meter reading this clock — was just dropped above; nothing survives to see the
        // discontinuity.
        self.start = Instant::now();
    }

    fn force_set_name(&mut self, uid: Uid, name: &str) {
        if let Some(e) = self.nodes.get_mut(&uid) {
            e.name = name.to_string();
        }
    }

    /// Serialize the graph to a `.gfi` v7 document (YAML text) — the `patch.yaml` manifest inside the
    /// archive container. v7 nests `nodes`/`links` under `root` alongside a flat `root.scopes` block
    /// (the organizational sub-patch overlay — scope metadata + member uid lists + stubs); top-level
    /// `globals`. A plain flat patch has an empty `scopes` block.
    pub fn serialize(&self) -> String {
        use serde_json::{json, Map, Value};
        let mut nodes = Map::new();
        for uid in self.node_uids() {
            let e = &self.nodes[&uid];
            let mut params = Map::new();
            let live = e.params.load_full();
            for (group, names) in &*live {
                let mut gmap = Map::new();
                for (name, p) in names {
                    gmap.insert(name.clone(), param_value_json(p));
                }
                params.insert(group.clone(), Value::Object(gmap));
            }
            let mut node_obj = Map::new();
            node_obj.insert("type".into(), json!(e.manifest.type_name));
            node_obj.insert("name".into(), json!(e.name));
            node_obj.insert("pos".into(), json!(e.pos));
            node_obj.insert("params".into(), Value::Object(params));
            // Persist expression bindings (sorted for a stable diff) — else a save/load
            // silently freezes every live-driven param to its last evaluated literal.
            if !e.bindings.is_empty() {
                let mut binds: Vec<(&ParamKey, &ExprBinding)> = e.bindings.iter().collect();
                binds.sort_by(|a, b| a.0.cmp(b.0));
                let arr: Vec<Value> = binds
                    .iter()
                    .map(|(k, b)| {
                        json!({ "group": k.group, "name": k.name, "source": b.source,
                                "enabled": b.enabled, "triggers_process": b.triggers_process })
                    })
                    .collect();
                node_obj.insert("expressions".into(), Value::Array(arr));
            }
            // Persist viewer view-state (per-slot kind/settings) when the editor has set
            // any — an empty blob stays out of the file so a fresh patch has no noise.
            if e.viewers.as_object().is_some_and(|m| !m.is_empty()) {
                node_obj.insert("viewers".into(), e.viewers.clone());
            }
            nodes.insert(uid.to_hex(), Value::Object(node_obj));
        }
        let links: Vec<Value> = self
            .links
            .iter()
            .map(|l| json!([l.node_out.to_hex(), l.slot_out, l.node_in.to_hex(), l.slot_in]))
            .collect();

        // Sub-patch scopes (the flat overlay). Each scope emits its own metadata + its direct member
        // uids + its stubs. The flat nodes/links above already hold the runtime, so there are no def
        // bodies to persist — the scope block is purely the organizational tree.
        let mut scope_map = Map::new();
        for (uid, scope) in &self.scopes {
            let mut stubs = Map::new();
            for (id, st) in &scope.stubs {
                stubs.insert(
                    id.clone(),
                    json!({
                        "dir": st.dir.name(),
                        "dtype": st.dtype.name(),
                        "inner_uid": st.inner.as_ref().map(|(u, _)| u.to_hex()),
                        "inner_slot": st.inner.as_ref().map(|(_, s)| s.clone()),
                        "pos": st.pos,
                        "name": st.name,
                    }),
                );
            }
            let members: Vec<Value> = self.scope_members(*uid).iter().map(|m| json!(m.to_hex())).collect();
            scope_map.insert(
                uid.to_hex(),
                json!({
                    "name": scope.name,
                    "parent": self.scope_of(*uid).map(|p| p.to_hex()),
                    "pos": scope.pos,
                    "nodes": members,
                    "stubs": Value::Object(stubs),
                }),
            );
        }

        let root = json!({ "nodes": Value::Object(nodes), "links": links, "scopes": Value::Object(scope_map) });
        // Globals (system + user) as an ORDERED array of `{name, value, type}` — order is observable
        // (panel / eval iteration), and a keyed serde_json::Map (a BTreeMap here) would alphabetize
        // the keys and silently lose it. On load, entries `set` existing system globals and
        // `add` user ones in file order, then `reassert_system` back-fills; so a system global always
        // round-trips and an older patch simply picks up any new system default.
        let globals: Vec<Value> = self
            .globals
            .entries()
            .map(|(name, value, _is_system)| {
                let mut e = global_to_json(value); // {value, type}
                if let Value::Object(ref mut m) = e {
                    m.insert("name".to_string(), Value::String(name.to_string()));
                }
                e
            })
            .collect();
        let mut doc = json!({
            "version": MANIFEST_VERSION,
            "pillar_default": "signal",
            "globals": Value::Array(globals),
            "root": root,
        });
        if let Value::Object(ref mut m) = doc {
            // The flat arrangement always exists (at worst the default), so it always rides.
            m.insert("arrangement".to_string(), self.arrangement.to_json());
            if !self.viewpoint.is_null() {
                m.insert("viewpoint".to_string(), self.viewpoint.clone());
            }
        }
        serde_yaml_ng::to_string(&doc).unwrap_or_default()
    }

    /// Replace the graph from a `.gfi` v7 manifest. Node types are validated before the current
    /// graph is torn down (a rejected load is a no-op).
    pub fn load_doc(&mut self, text: &str) -> Result<(), String> {
        let doc: serde_json::Value = serde_yaml_ng::from_str(text).map_err(|e| e.to_string())?;
        let (nodes_v, links_v) = match doc.get("version").and_then(|v| v.as_i64()) {
            // v7 is the archive era: nodes/links nested under `root`, a flat `root.scopes`
            // overlay, top-level `globals`, opaque top-level `layout`. The bare-YAML v3-v6
            // files predate the zip container and are deliberately not read (spec Decision 3).
            Some(MANIFEST_VERSION) => {
                let root = doc.get("root");
                (root.and_then(|r| r.get("nodes")), root.and_then(|r| r.get("links")))
            }
            _ => {
                return Err(format!(
                    "unsupported .gfi version (this build reads version {MANIFEST_VERSION})"
                ))
            }
        };
        let nodes = nodes_v.and_then(|v| v.as_object()).ok_or("missing `nodes`")?;
        for rec in nodes.values() {
            let ty = rec.get("type").and_then(|v| v.as_str()).ok_or("node missing `type`")?;
            if !self.known_type(ty) {
                return Err(self.reject_type(ty));
            }
        }

        self.clear();
        // Globals load BEFORE nodes so a node's `globals.*` param default-expression resolves at
        // instantiation. `clear()` already re-seeded the system globals; each entry sets an existing
        // (system) global or adds a user one, IN FILE ORDER (so the observable order round-trips).
        // Malformed entries are skipped (best-effort load).
        if let Some(serde_json::Value::Array(arr)) = doc.get("globals") {
            for entry in arr {
                if let (Some(name), Some(value)) =
                    (entry.get("name").and_then(|v| v.as_str()), global_from_json(entry))
                {
                    let _ = self.globals_mut(|g| g.apply_change(name, Some(value)));
                }
            }
        }
        // Every uid this load hands out, restored or minted — what keeps two records from landing
        // on one uid when a hand-written file spells the same number two ways.
        let mut claimed: HashSet<Uid> = HashSet::new();
        let mut idmap: HashMap<String, Uid> = HashMap::new();
        for (old, rec) in nodes {
            let ty = rec["type"].as_str().unwrap();
            // NON-seeding instantiation: load is a restore, so the doc is authoritative for BOTH
            // params and expressions. Going through `add_node` (which seeds `default_expr` bindings)
            // would re-synthesize a binding for any `default_expr` param the user had UNBOUND to a
            // literal — the reseed would then clobber the saved literal on the next tick. The doc's
            // own `expressions` block (restored below) round-trips every binding the user actually has.
            //
            // The saved params are folded in BEFORE construction, because `insert_node` runs the
            // node's `setup()` — a one-time init that reads its params (allocate a buffer of `size`,
            // open device `name`) and never runs again. Applying them afterwards would boot every
            // node against the type's defaults; on the detached tier, where a param edit is an
            // explicit no-op, the child would never see them at all. This is the same order the
            // undo/redo restore path uses (`Command::AddNode` carries the captured params).
            let mut params = self.default_params_of(ty)?;
            if let Some(groups) = rec.get("params").and_then(|v| v.as_object()) {
                for (group, names) in groups {
                    let Some(nm) = names.as_object() else { continue };
                    let Some(g) = params.get_mut(group) else { continue };
                    for (name, val) in nm {
                        if let Some(existing) = g.get_mut(name) {
                            // Never fire a trigger on load: a persisted or hand-edited value must
                            // not trip the node's trigger as the patch opens.
                            *existing = param_from_json(existing, val, false);
                        }
                    }
                }
            }
            let (manifest, params, node) = self.build_node(ty, Some(params))?;
            // The record's KEY is its uid — restored, not reminted (see `restore_uid`). The name is
            // the type's fresh one only until the record's own `name` lands, just below.
            let uid = self.restore_uid(old, &claimed);
            claimed.insert(uid);
            let name = self.fresh_name(&manifest.type_name.to_lowercase());
            self.insert_node_at(uid, name, manifest, node, params);
            idmap.insert(old.clone(), uid);
            if let Some(name) = rec.get("name").and_then(|v| v.as_str()) {
                self.force_set_name(uid, name);
            }
            if let Some(p) = rec.get("pos").and_then(|v| v.as_array()) {
                if p.len() == 2 {
                    if let (Some(x), Some(y)) = (p[0].as_f64(), p[1].as_f64()) {
                        let _ = self.set_node_pos(uid, [x, y]);
                    }
                }
            }
            if let Some(v) = rec.get("viewers").filter(|v| v.is_object()) {
                let _ = self.set_node_viewers(uid, v.clone());
            }
            // Reconstruct expression bindings (after literal params are applied).
            if let Some(exprs) = rec.get("expressions").and_then(|v| v.as_array()) {
                for ex in exprs {
                    let g = ex.get("group").and_then(|v| v.as_str());
                    let n = ex.get("name").and_then(|v| v.as_str());
                    let src = ex.get("source").and_then(|v| v.as_str()).unwrap_or("");
                    let en = ex.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
                    let tp = ex.get("triggers_process").and_then(|v| v.as_bool()).unwrap_or(false);
                    if let (Some(g), Some(n)) = (g, n) {
                        let _ = self.set_expression(uid, g, n, src, en, tp);
                    }
                }
            }
        }
        if let Some(links) = links_v.and_then(|v| v.as_array()) {
            for l in links {
                if let Some(a) = l.as_array() {
                    if a.len() == 4 {
                        let no = a[0].as_str().and_then(|s| idmap.get(s)).copied();
                        let ni = a[2].as_str().and_then(|s| idmap.get(s)).copied();
                        if let (Some(no), Some(ni)) = (no, ni) {
                            let _ = self.add_link(
                                no,
                                a[1].as_str().unwrap_or(""),
                                ni,
                                a[3].as_str().unwrap_or(""),
                            );
                        }
                    }
                }
            }
        }
        // Reconstruct the flat sub-patch scopes. The members are already live flat nodes; here
        // we restore each scope's uid, re-tag membership from its `nodes` list, and rebuild its
        // stubs (resolving the stored inner uid). No def bodies to rehydrate — the runtime is flat.
        let scopes_v = doc.get("root").and_then(|r| r.get("scopes")).and_then(|v| v.as_object());
        self.reload_scopes(scopes_v, &idmap, &mut claimed);
        self.viewpoint = doc.get("viewpoint").cloned().unwrap_or(serde_json::Value::Null);
        // A corrupt arrangement costs the CHROME, never the patch — the graph is the value, and a
        // file that cannot be opened is the one outcome worse than a lost layout. The reason is kept
        // for the load reply so the fallback is stated rather than silent. An ABSENT arrangement is
        // not a corrupt one (a patch saved before this shape existed), so it warns about nothing.
        let (arrangement, warning) = match doc.get("arrangement") {
            None => (layout::Layout::default(), None),
            Some(v) => match layout::Layout::from_json(v) {
                Ok(l) => (l, None),
                Err(e) => (layout::Layout::default(), Some(e)),
            },
        };
        self.arrangement = arrangement;
        self.arrangement_warning = warning;
        Ok(())
    }

    /// Rebuild `scopes`/`scope_of` from a loaded v7 document, after the flat nodes/links are live.
    /// A scope uid restores from its key like a node's does — an editor panel's `subpatchPath` names
    /// scopes, and it is persisted beside the very scopes it points at. A member uid resolves through
    /// `idmap` (a flat leaf) or the scope map (a nested-scope member); a stub's stored `inner_uid`
    /// resolves the same way.
    fn reload_scopes(
        &mut self,
        scopes_v: Option<&serde_json::Map<String, serde_json::Value>>,
        idmap: &HashMap<String, Uid>,
        claimed: &mut HashSet<Uid>,
    ) {
        use subpatch::{Dir, Scope, Stub};
        let Some(scopes) = scopes_v else { return };

        // Resolve every scope uid first, so parent refs + nested-member refs + stub inner refs
        // resolve regardless of iteration order.
        let mut scopemap: HashMap<String, Uid> = HashMap::new();
        for old in scopes.keys() {
            let uid = self.restore_uid(old, claimed);
            claimed.insert(uid);
            scopemap.insert(old.clone(), uid);
        }
        let resolve_uid = |s: &str| idmap.get(s).copied().or_else(|| scopemap.get(s).copied());

        for (old, rec) in scopes {
            let uid = scopemap[old];
            let name = rec.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let parent = rec.get("parent").and_then(|v| v.as_str()).and_then(|s| scopemap.get(s)).copied();
            let pos = rec
                .get("pos")
                .and_then(|v| v.as_array())
                .and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?]))
                .unwrap_or([0.0, 0.0]);
            // Re-tag membership from the scope's `nodes` list (leaf uids + child-scope uids).
            if let Some(members) = rec.get("nodes").and_then(|v| v.as_array()) {
                for mv in members {
                    if let Some(ru) = mv.as_str().and_then(resolve_uid) {
                        self.scope_of.insert(ru, Some(uid));
                    }
                }
            }
            // Rebuild stubs (remapping each stored inner uid through idmap/scopemap).
            let mut stubs: IndexMap<subpatch::StubId, Stub> = IndexMap::new();
            if let Some(sm) = rec.get("stubs").and_then(|v| v.as_object()) {
                for (id, st) in sm {
                    let dir = if st.get("dir").and_then(|v| v.as_str()) == Some("in") { Dir::In } else { Dir::Out };
                    // The save side writes `st.dtype.name()`; read it back with that function's own
                    // inverse. An unknown/absent tag still degrades silently to Array, as before.
                    let dtype = st
                        .get("dtype")
                        .and_then(|v| v.as_str())
                        .and_then(goofi_core::SlotType::from_name)
                        .unwrap_or(goofi_core::SlotType::Array);
                    let inner = match (
                        st.get("inner_uid").and_then(|v| v.as_str()).and_then(resolve_uid),
                        st.get("inner_slot").and_then(|v| v.as_str()),
                    ) {
                        (Some(u), Some(s)) => Some((u, s.to_string())),
                        _ => None,
                    };
                    let pos = st
                        .get("pos")
                        .and_then(|v| v.as_array())
                        .and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?]))
                        .unwrap_or([0.0, 0.0]);
                    let sname = st.get("name").and_then(|v| v.as_str()).unwrap_or(id).to_string();
                    stubs.insert(id.clone(), Stub { dir, dtype, inner, pos, name: sname });
                }
            }
            self.scope_of.insert(uid, parent);
            self.scopes.insert(uid, Scope { name, pos, stubs });
        }
    }

}

/// One node's current error, derived fresh from the three places one can arise — see
/// [`Graph::last_error`], whose contract this is. A free function so the per-tick onset sweep can
/// read it while holding a `&mut NodeEntry`, which keeps derivation and stamping on one rule.
fn entry_error(e: &NodeEntry) -> Option<&str> {
    // The node's initialization failure outranks a process error, and D3 makes it the only thing
    // that CAN be true beside one: if `setup` failed, `process` never runs. The order therefore
    // encodes which of the two is possible, not which one wins a contest. A node whose services
    // could not be created carries its boot failure here too — it is the same "this node never
    // started" fact, one layer further out.
    // An inline node's initialization failure, for the same reason and one tier down: it is the
    // root cause, and D3 makes it the only thing that CAN be true here — if `setup` failed,
    // `process` never runs, so a process error cannot arise beside it. The order therefore encodes
    // which of the two is possible, not which one wins a contest.
    if let Some(err) = e.setup_error.as_deref() {
        return Some(err);
    }
    if let Some(err) = e.last_error.as_deref() {
        return Some(err);
    }
    // Both param-keyed error records, ordered by key together — the node rolls its own map up the
    // same way (`NodeRuntime::node_fault`), and folding only one of them here would make which
    // record an error landed in decide whether the badge ever shows it.
    e.bindings
        .iter()
        .filter_map(|(k, b)| b.error.as_deref().map(|s| (k, s)))
        .chain(e.param_errors.iter().map(|(k, m)| (k, m.as_str())))
        .min_by(|a, b| a.0.cmp(b.0))
        .map(|(_, s)| s)
}

pub(crate) fn seed_node(
    node: &mut dyn goofi_node::Node,
    params: &ParamGroups,
    ctx: &mut NodeCtx,
) -> Option<String> {
    let mut last_error = None;
    for (group, entries) in params {
        if group == "common" {
            continue;
        }
        for (name, value) in entries {
            let key = ParamKey::new(group.as_str(), name.as_str());
            if let Err(e) = guard_lifecycle(|| node.on_param_changed(&key, value)).unwrap_or_else(fold_panic) {
                last_error.get_or_insert(e.0);
            }
        }
    }
    // A panic in `setup` is this node's boot error, exactly as a returned `Err` is. Unguarded it
    // unwound through `Graph::add_node` and out through the graph lock the caller was holding.
    let started =
        guard_lifecycle(|| node.setup(ctx, &goofi_node::Params::new(params))).unwrap_or_else(fold_panic);
    if let Err(e) = started {
        last_error.get_or_insert(e.0);
    }
    last_error
}

/// How long the TICK waits between retries of a failed initialization (D3). The retry is the whole
/// [`seed_node`] unit — every param's `on_param_changed` plus `setup()` — and it runs on the tick
/// thread inside the mutex the bridge holds across a whole tick. A `setup()` that fails is exactly
/// the kind that BLOCKS first (opening a device, dialling a socket) and the kind that leaks a
/// handle per attempt, since `Drop` never fires between them. One second bounds both to roughly one
/// per second, and is still well inside the time a user watching the node would wait for it to heal.
const SETUP_RETRY_INTERVAL: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{stays, wait_for, OutputProbe};
    use goofi_core::{Data, Meta, SlotType, Value};
    use goofi_node::{
        default_factory, ExprDecl, ExprMode, Inputs, Isolation, Node, NodeManifest, NodeResult,
        OutputDecl, Outputs, ParamDecl, ParamSpec, Params, SlotDecl,
    };

    /// Empty param declaration, shared by the many test nodes with no own params.
    static NO_PARAMS: &[ParamDecl] = &[];

    #[test]
    fn uid_from_hex_accepts_only_the_canonical_12_hex_domain() {
        // `to_hex` formats `{:012x}` and every comment calls this a 12-hex identity, but the parser
        // took any radix-16 u64 — so a hand-edited `.gfi` keyed by 16 hex digits reached
        // `restore_uid`'s `u.0 + 1` and overflowed. Bounding the domain here is what makes that
        // arithmetic total everywhere downstream.
        assert_eq!(Uid::from_hex("000000000001"), Some(Uid(1)));
        assert_eq!(Uid::from_hex("ffffffffffff"), Some(Uid(0xffff_ffff_ffff)));
        assert_eq!(Uid::from_hex("ffffffffffffffff"), None, "16 hex is outside the identity domain");
        assert_eq!(Uid::from_hex("1"), None, "a short uid is not canonical either");
        assert_eq!(Uid::from_hex("zzzzzzzzzzzz"), None);
        assert_eq!(Uid::from_hex(""), None);
        // Round-trip: whatever `to_hex` writes, `from_hex` must read.
        let u = Uid(0x0000_0000_002a);
        assert_eq!(Uid::from_hex(&u.to_hex()), Some(u));
    }

    #[test]
    fn graph_seeds_and_edits_globals() {
        use goofi_core::globals::GlobalValue;
        let mut g = Graph::new();
        // A fresh graph carries the system globals.
        assert_eq!(g.globals().get("default_ufreq"), Some(&GlobalValue::Float(30.0)));
        assert!(g.globals().is_system("default_ufreq"));
        // Edit a system global's value; add + remove a user global.
        g.apply_global_change("default_ufreq", Some(GlobalValue::Int(60))).unwrap(); // coerces to Float
        assert_eq!(g.globals().snapshot().f64("default_ufreq"), Some(60.0));
        g.apply_global_change("subject", Some(GlobalValue::Str("P07".into()))).unwrap();
        assert_eq!(g.globals().snapshot().str("subject"), Some("P07"));
        g.apply_global_change("subject", None).unwrap();
        assert!(g.globals().get("subject").is_none());
        // System globals can't be deleted.
        assert!(g.apply_global_change("default_ufreq", None).is_err());
        // A load (clear) resets globals to the fresh system-seeded store.
        g.apply_global_change("u", Some(GlobalValue::Bool(true))).unwrap();
        g.clear();
        assert!(g.globals().get("u").is_none(), "user globals cleared on load");
        assert_eq!(g.globals().get("default_ufreq"), Some(&GlobalValue::Float(30.0)), "system re-seeded");
    }

    #[test]
    fn node_process_reads_live_globals_from_ctx() {
        use goofi_core::globals::GlobalValue;
        let mut g = Graph::new();
        let n = g.add_node("_TestGlobal", None).unwrap();
        let out = OutputProbe::open(&g, n, "out");
        out.wait_until(&mut g, "reads the seeded default_ufreq", |d| first_f32(d) == 30.0);
        // An edit is visible on the node's next run — `process` reads the globals RECORD live
        // through the handle it holds (§5.2), not a snapshot latched at birth.
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(45.0))).unwrap();
        out.wait_until(&mut g, "sees the edited value", |d| first_f32(d) == 45.0);
    }

    /// What a binding's variables resolved to, spelled out — the graph half of §5.3, which is now
    /// the whole of what the graph does with an expression. Written out as strings so a wrong
    /// producer, a wrong slot or a lost value is visible in the failure.
    fn resolved(g: &Graph, uid: Uid, group: &str, name: &str) -> Vec<String> {
        g.nodes[&uid].bindings[&ParamKey::new(group, name)]
            .vars
            .iter()
            .map(|v| match v {
                BoundVar::Stream { var, producer, slot, event_id } => {
                    format!("{var}={}.{slot}#{event_id}", g.nodes[producer].name)
                }
                BoundVar::Value { var, value } => format!("{var}={:?}", value.as_f64()),
                BoundVar::Missing { var, reason } => format!("{var}!{reason}"),
            })
            .collect()
    }

    #[test]
    fn a_global_is_resolved_into_the_binding_and_re_resolved_on_an_edit() {
        // §5.2/§5.3: a global is an ordinary variable — the graph resolves its VALUE and ships it
        // inline, and an edit re-sends the binding. There is no invalidation message and nothing
        // for the node to look up, which is why the value has to be here rather than a promise.
        use goofi_core::globals::GlobalValue;
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "globals.default_ufreq", true, false).unwrap();
        assert_eq!(resolved(&g, n, "constant", "value"), ["__v0=Some(30.0)"]);

        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(48.0))).unwrap();
        assert_eq!(resolved(&g, n, "constant", "value"), ["__v0=Some(48.0)"], "the edit re-resolved it");

        // A global the patch does not define is refused HERE, at bind time. It used to be an
        // eval-time NameError raised by the evaluator's `_Globals` proxy; with the rewrite the
        // graph is what resolves a global, so this is where the rule lives now.
        g.set_expression(n, "constant", "value", "globals.nope", true, false).unwrap();
        assert_eq!(resolved(&g, n, "constant", "value"), ["__v0!global `nope` is not defined"]);
        assert_eq!(
            g.param_expression(n, "constant", "value").unwrap().error.as_deref(),
            Some("global `nope` is not defined"),
        );

        // …and defining it later resolves the binding, which is the other half of the same rule.
        g.apply_global_change("nope", Some(GlobalValue::Float(2.0))).unwrap();
        assert_eq!(resolved(&g, n, "constant", "value"), ["__v0=Some(2.0)"]);
    }

    #[test]
    fn editing_a_referenced_global_re_resolves_only_the_bindings_that_read_it() {
        // The targeted half: a binding that does NOT read the edited global is not re-resolved.
        //
        // Its value cannot show that — `globals.other` resolves to 7.0 whether it is re-resolved or
        // not, so a "does it still hold 7.0?" oracle passes against a blanket re-resolution. What
        // does show it is the compiled HANDLE: re-resolving a binding releases the old one and
        // compiles the rewritten source again, so the untouched binding's id must never appear in
        // the evaluator's release log.
        use goofi_core::globals::GlobalValue;
        let mock = Arc::new(MockEval::default());
        let mut g = Graph::new();
        g.set_evaluator(mock.clone());
        g.apply_global_change("other", Some(GlobalValue::Float(7.0))).unwrap();
        let (a, b) = (g.add_node("_TestConst", None).unwrap(), g.add_node("_TestConst", None).unwrap());
        g.set_expression(a, "constant", "value", "globals.default_ufreq", true, false).unwrap();
        g.set_expression(b, "constant", "value", "globals.other", true, false).unwrap();
        let key = ParamKey::new("constant", "value");
        let (a_id, b_id) = (g.nodes[&a].bindings[&key].id.unwrap(), g.nodes[&b].bindings[&key].id.unwrap());

        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(50.0))).unwrap();

        let released = mock.releases.lock().unwrap().clone();
        assert!(released.contains(&a_id), "the binding that reads it WAS re-resolved");
        assert!(!released.contains(&b_id), "and the one that does not read it was not");
        // …and the re-resolution actually landed the new value, so "nothing was re-resolved" is not
        // a way to pass the line above.
        assert_eq!(resolved(&g, a, "constant", "value"), ["__v0=Some(50.0)"]);
        assert_eq!(resolved(&g, b, "constant", "value"), ["__v0=Some(7.0)"]);
    }

    #[test]
    fn an_expression_a_user_can_type_never_panics_under_the_graph_lock() {
        // `set_expression` is an MCP surface AND the RPC the inspector's fx field calls, and it runs
        // holding the `Arc<Mutex<Graph>>` the bridge locks with `.lock().unwrap()` everywhere. A
        // panic here is not one failed edit — it poisons the mutex and every later RPC, worker and
        // `/data` subscribe dies with it. So a string a user can type must ANSWER, always.
        //
        // Driven through the RPC entry point rather than through `rewrite`, because that is where
        // the mutex is: a pure-function test proves the arithmetic and not the containment.
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        for source in [
            "nd('globals.gain') + 1",  // a globals read nested inside a node NAME
            "globals.g * nd('a')",     // the two scans arriving out of source order
            "nd('globals.a').out * globals.b",
            "nd('') + 1",
            "nd('a', 2) + globals.x",
            "nd(",
            "globals.",
            "",
        ] {
            // Ok or Err, both fine — the point is that it RETURNS. `set_expression` answers Err only
            // for an unknown node or param, so a bad expression rides the binding's error field.
            let _ = g.set_expression(n, "constant", "value", source, true, false);
        }
        assert!(g.contains(n), "the graph survived every one of them");
    }

    #[test]
    fn a_reported_fault_surfaces_on_the_projections_that_already_exist() {
        // §6: `NodeFault` and `Stage` change only how the graph LEARNS a node's state, not how it
        // is projected — `runtime_overlay` keeps working verbatim, and it reads `last_error` and
        // `node_stage`. So the four variants have to land in the fields those two already read;
        // storing them anywhere else would leave a node reporting an error the editor never draws.
        let mut g = eval_graph();
        let uid = g.add_node("_TestConst", None).unwrap();
        // The node reports its own stage, so the graph learns it by draining — `creating` until it
        // has (§6.2), which is the projection's own way of saying "built, not yet heard from".
        wait_for(&mut g, "the node to report itself ready", |g| g.node_stage(uid) == "ready");

        for (fault, msg) in [
            (runtime::NodeFault::Process { msg: "boom".into(), since: 1.0 }, "boom"),
            (runtime::NodeFault::Boot { msg: "no worker".into(), since: 1.0 }, "no worker"),
        ] {
            g.apply_status(uid, runtime::Status::Fault { fault: Some(fault) });
            assert_eq!(g.last_error(uid), Some(msg));
            assert_eq!(g.node_stage(uid), "error");
            // The roll-ups fold BOTH error channels, so which field a fault landed in is exactly
            // what they cannot show — and the two fields are not interchangeable (below).
            assert_eq!(g.nodes[&uid].last_error.as_deref(), Some(msg));
            assert!(g.nodes[&uid].setup_error.is_none(), "a run failure is not a setup failure");
        }

        // `Setup` is the one that must be told apart, and `last_error`/`node_stage` cannot tell it:
        // `setup_error` is `ensure_initialized`'s gate (D3), so a Setup fault written to
        // `last_error` instead leaves the node "initialized" — nothing retries it, and correcting
        // the param that broke it stops being the door back. That closes silently.
        g.apply_status(
            uid,
            runtime::Status::Fault {
                fault: Some(runtime::NodeFault::Setup { msg: "no device".into(), since: 1.0, last_attempt: 1.0 }),
            },
        );
        assert_eq!(g.last_error(uid), Some("no device"));
        assert_eq!(g.node_stage(uid), "error");
        assert_eq!(g.nodes[&uid].setup_error.as_deref(), Some("no device"), "the RETRY gate is set");
        // That the gate is LIVE is not assertable from here any more: the retry runs on the node's
        // own thread against its own state (D3), and the fault above is one this test injected —
        // the real `_TestConst` never failed `setup()`, so it has nothing to retry and nothing to
        // report. The end-to-end door is pinned where it now lives, on a node whose `setup` really
        // does fail: `correcting_the_param_that_broke_setup_reinitializes_the_node`.

        // A clean run clears Setup/Process/Boot TOGETHER — the node stamped one fault at a time,
        // and clearing only the last one reported would leave the earlier field standing forever.
        g.apply_status(uid, runtime::Status::Fault { fault: None });
        assert_eq!(g.last_error(uid), None);
        assert_eq!(g.node_stage(uid), "ready");

        // A binding error is a MAP, not a fault: it arrives on its own channel and rolls up through
        // the same `last_error` the badge reads. It is deliberately NOT cleared by a clean run.
        let key = ParamKey::new("constant", "value");
        g.set_expression(uid, "constant", "value", "5", true, false).unwrap();
        g.apply_status(
            uid,
            runtime::Status::BindingErrors { errors: vec![(key.clone(), Some("nope".to_string()))] },
        );
        assert_eq!(g.last_error(uid), Some("nope"));
        g.apply_status(uid, runtime::Status::Fault { fault: None });
        assert_eq!(g.last_error(uid), Some("nope"), "a clean run does not fix a broken expression");
        g.apply_status(uid, runtime::Status::BindingErrors { errors: vec![(key, None)] });
        assert_eq!(g.last_error(uid), None, "only the binding evaluating clears it");
    }

    #[test]
    fn a_param_read_is_never_blocked_by_a_concurrent_edit() {
        // §5.1: the record is an `ArcSwap` the graph and the node hold TOGETHER. That is the READ
        // path — a node's `process()` loads it without the graph mutex, so an edit storm cannot
        // stall a run — while `Control::SetParam` stays the NOTIFICATION path, because a bare swap
        // cannot say which key changed and a node parked with `next_wake() == None` is never rung.
        //
        // The reader deliberately runs while the WRITER holds the graph mutex for its whole burst:
        // a handle that needed that lock would deadlock here rather than merely be slow.
        use std::sync::Mutex;
        let graph = Arc::new(Mutex::new(Graph::new()));
        let uid = graph.lock().unwrap().add_node("_TestConst", None).unwrap();
        let record = graph.lock().unwrap().param_record(uid).expect("the node's own handle");

        let reader = {
            let record = record.clone();
            std::thread::spawn(move || {
                (0..10_000).filter(|_| record.load().contains_key("constant")).count()
            })
        };
        let mut g = graph.lock().unwrap();
        for i in 0..10_000 {
            g.update_param(uid, "constant", "value", Param::float(i as f64, -1e9, 1e9)).unwrap();
        }
        drop(g);
        assert_eq!(reader.join().unwrap(), 10_000, "every read completed, none torn");

        // …and the handle IS the record rather than a copy of it: a snapshot handed out once would
        // pass everything above and never see an edit.
        assert_eq!(
            goofi_node::param(&record.load(), "constant", "value").and_then(Param::as_f64),
            Some(9_999.0),
        );
    }

    #[test]
    fn a_restart_swaps_the_record_rather_than_replacing_it() {
        // A restart replaces the INSTANCE, not the node — and the node's own thread is holding this
        // handle (§5.1). Installing a fresh `ArcSwap` would leave every holder reading the corpse's
        // params forever, while `params(uid)` — which reads the entry — went on looking right.
        // Driven by an edit made AFTER the restart, because the restart carries the held values
        // over and a stale handle answers those correctly.
        let mut g = Graph::new();
        let uid = g.add_node("_TestConst", None).unwrap();
        let record = g.param_record(uid).unwrap();
        g.restart_node(uid).unwrap();
        g.update_param(uid, "constant", "value", Param::float(9.0, -1e9, 1e9)).unwrap();
        assert_eq!(
            goofi_node::param(&record.load(), "constant", "value").and_then(Param::as_f64),
            Some(9.0),
            "a handle taken before the restart still sees the graph",
        );
    }

    #[test]
    fn the_globals_record_is_a_handle_too() {
        // §5.2: globals are shared exactly as params are, for the direct reads a node makes through
        // `ctx.globals`. Same property, same reason: a node thread must not need the graph mutex to
        // read one.
        use goofi_core::globals::GlobalValue;
        let mut g = Graph::new();
        let record = g.globals_record();
        assert_eq!(record.load().f64("default_ufreq"), Some(30.0));
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(45.0))).unwrap();
        assert_eq!(record.load().f64("default_ufreq"), Some(45.0), "the edit reached the handle");
        g.apply_global_change("subject", Some(GlobalValue::Str("P07".into()))).unwrap();
        assert_eq!(record.load().str("subject"), Some("P07"), "and so does an ADD");
    }

    #[test]
    fn the_universal_common_declarations_are_seeded_too() {
        // `common.max_frequency` CARRIES `globals.default_ufreq` on every node — live on a producer,
        // carried (disabled) on a consumer, so one inspector toggle paces anything. The seeding walk
        // read only `manifest.params`, and the universal declarations are in no manifest's params,
        // so a producer that did not redeclare the key got no binding at all and never re-rated.
        //
        // Read through `common_decls`, the ONE place `producer` is interpreted, so the value half
        // (`with_common`) and this half cannot disagree about who is a producer.
        let mut g = eval_graph();
        let producer = g.add_node("_TestGated", None).unwrap();
        let consumer = g.add_node("_TestSink", None).unwrap();

        let live = g.param_expression(producer, "common", "max_frequency").expect("a producer is paced");
        assert_eq!(live.source, "globals.default_ufreq");
        assert!(live.enabled, "a source is what the patch rate is for");
        assert_eq!(resolved(&g, producer, "common", "max_frequency"), ["__v0=Some(30.0)"]);

        let carried =
            g.param_expression(consumer, "common", "max_frequency").expect("a consumer carries it");
        assert!(!carried.enabled, "carried for the fx toggle, not imposed");
        assert!(carried.error.is_none(), "and healthy — a carried binding is not a broken one");
    }

    #[test]
    fn a_node_that_declares_a_common_param_itself_keeps_its_own_declaration() {
        // The owner's rule, verbatim: "when a node declares a common param in its manifest, we will
        // not touch this param. We will not overwrite the manifest's param definitions." Oscillator
        // declares `common.max_frequency` with its own literal ceiling, and seeding it twice — once
        // from the manifest, once from the universal list — would leave whichever ran last.
        let mut g = eval_graph();
        let osc = g.add_node("Oscillator", None).unwrap();
        let info = g.param_expression(osc, "common", "max_frequency").expect("seeded");
        assert_eq!(info.source, "globals.default_ufreq");
        assert!(info.enabled);
        // The literal underneath is the manifest's 30.0-with-a-1000-ceiling, not the universal
        // declaration's 0.0-with-a-100-ceiling. `with_common` is what keeps it, and this is the
        // half that could quietly replace it.
        assert_eq!(
            g.params(osc).unwrap()["common"]["max_frequency"],
            Param::float(30.0, 0.0, 1000.0),
            "the manifest's own declaration stands",
        );

        // Oscillator happens to declare a `common.max_frequency` expression IDENTICAL to the
        // universal one in every field, so it cannot show which of the two was applied. `_TestOwnCommon`
        // is where they differ: it is a CONSUMER — universal mode Off — that declares mode On.
        let own = g.add_node("_TestOwnCommon", None).unwrap();
        let info = g.param_expression(own, "common", "max_frequency").expect("seeded");
        assert!(info.enabled, "the node said On; the universal declaration does not get to say Off");
        assert!(!info.triggers_process, "and its own `trigger`, not the universal `true`");
        assert_eq!(g.params(own).unwrap()["common"]["max_frequency"], Param::float(7.0, 0.0, 500.0));
    }

    #[test]
    fn fresh_add_seeds_a_default_expr_binding_resolved_against_the_globals() {
        use goofi_core::globals::GlobalValue;
        let mut g = eval_graph();
        let n = g.add_node("_TestDefaultExpr", None).unwrap();
        // The declared default_expr became a real, live binding (not a literal).
        let info = g.param_expression(n, "control", "rate").expect("default_expr seeded a binding");
        assert_eq!(info.source, "globals.default_ufreq");
        assert!(info.enabled && info.error.is_none(), "seeded binding is enabled + healthy");
        assert_eq!(resolved(&g, n, "control", "rate"), ["__v0=Some(30.0)"]);
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(42.0))).unwrap();
        assert_eq!(resolved(&g, n, "control", "rate"), ["__v0=Some(42.0)"], "and it re-rates live");
    }

    #[test]
    fn seeding_carries_the_declarations_mode_and_trigger_rather_than_assuming_them() {
        // A declared expression states whether it starts live and whether it wakes `process`.
        // Seeding used to hard-code `enabled = true, triggers_process = false`, which made a
        // carried (Off) expression live the moment it was declared — the opposite of what
        // `ExprMode::Off` documents — and left `trigger` unable to mean anything at all.
        let mut g = eval_graph();
        let n = g.add_node("_TestCarriedExpr", None).unwrap();

        let carried = g.param_expression(n, "control", "carried").expect("the source is stored");
        assert_eq!(carried.source, "globals.default_ufreq");
        assert!(!carried.enabled, "mode Off is carried for the fx toggle, not imposed");
        assert!(!carried.triggers_process);

        let paced = g.param_expression(n, "control", "paced").expect("seeded");
        assert!(paced.enabled, "mode On is live");
        assert!(paced.triggers_process, "and `trigger: true` reaches the binding");

        // …and only the LIVE one subscribes: a carried binding resolves no variables, so nothing
        // is shipped for it and the spec literal stands until the fx toggle turns it on.
        assert!(resolved(&g, n, "control", "carried").is_empty(), "a carried binding resolves nothing");
        assert_eq!(resolved(&g, n, "control", "paced"), ["__v0=Some(30.0)"]);
        assert_eq!(g.params(n).unwrap()["control"]["carried"].as_f64(), Some(5.0), "literal stands");
    }

    #[test]
    fn default_expr_falls_back_to_the_literal_without_an_evaluator() {
        // No evaluator wired ⇒ no binding is minted; the param keeps its spec-default literal (5.0),
        // never an errored "no evaluator" binding. Graceful degrade for eval-less runs (a build
        // without the `python` feature, or an interpreter the evaluator could not initialize).
        let mut g = Graph::new();
        let n = g.add_node("_TestDefaultExpr", None).unwrap();
        assert!(g.param_expression(n, "control", "rate").is_none(), "no binding without an evaluator");
        let out = OutputProbe::open(&g, n, "out");
        assert_eq!(first_f32(&out.expect_frame(&mut g, "the node to emit")), 5.0, "the literal fallback is used");
    }

    #[test]
    fn binding_common_max_frequency_to_a_global_re_rates_the_run_policy() {
        // The producer story, graph-side: a `common.max_frequency` bound to `globals.default_ufreq`
        // carries the global's VALUE, and a global edit re-resolves it — which is what re-paces
        // every Oscillator. The node's half (re-deriving `RunPolicy` from the arrival without
        // running) is `runtime::tests::a_common_arrival_repaces_without_running`.
        use goofi_core::globals::GlobalValue;
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "common", "max_frequency", "globals.default_ufreq", true, false).unwrap();
        assert_eq!(resolved(&g, n, "common", "max_frequency"), ["__v0=Some(30.0)"], "rated by the global");
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(12.0))).unwrap();
        assert_eq!(resolved(&g, n, "common", "max_frequency"), ["__v0=Some(12.0)"], "re-rates on an edit");
    }

    #[test]
    fn restore_path_does_not_reseed_default_expr() {
        // A restore/load supplies explicit params (the doc is authoritative) → NO auto-binding; the
        // doc's own captured expressions are what get restored (separately). `add_node_at(Some(..))`
        // models the restore entry point.
        let mut g = eval_graph();
        let m = goofi_node::find("_TestDefaultExpr").unwrap();
        let params = goofi_node::with_common(m.default_params(), m);
        let n = g.add_node_at("_TestDefaultExpr", Some(params), Uid(0xD15C), "restored").unwrap();
        assert!(
            g.param_expression(n, "control", "rate").is_none(),
            "restore must not auto-bind — the doc is the source of truth"
        );
    }

    #[test]
    fn default_expr_binding_round_trips_and_load_does_not_reseed() {
        // A seeded default_expr binding must persist through the .gfi like any expression, and load
        // must RESTORE it from the doc — not re-synthesize it (which would risk a double-seed). One
        // binding out, one binding in, still evaluating to the global.
        let mut g = eval_graph();
        let n = g.add_node("_TestDefaultExpr", None).unwrap();
        assert_eq!(g.param_expression(n, "control", "rate").unwrap().source, "globals.default_ufreq");
        let doc = g.serialize();

        let mut g2 = eval_graph();
        g2.load_doc(&doc).unwrap();
        let restored = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.type_name(*u) == Some("_TestDefaultExpr"))
            .expect("node restored");
        let info = g2.param_expression(restored, "control", "rate").expect("binding restored from the doc");
        assert_eq!(info.source, "globals.default_ufreq");
        assert!(info.error.is_none(), "restored binding is healthy (not double-seeded / errored)");
        assert_eq!(
            resolved(&g2, restored, "control", "rate"),
            ["__v0=Some(30.0)"],
            "and it comes back resolved against the loaded patch's globals",
        );
    }

    #[test]
    fn load_preserves_a_literal_that_overrode_a_default_expr_binding() {
        // A user who UNBINDS a default_expr param (clearing the binding) and sets a fixed literal must
        // keep that literal across save/load — load must NOT re-seed the default_expr binding, which
        // would clobber the literal on the next tick and silently re-rate the node to the global.
        let mut g = eval_graph();
        let n = g.add_node("_TestDefaultExpr", None).unwrap();
        // Clear the seeded binding (empty source removes it) and pin a fixed literal.
        g.set_expression(n, "control", "rate", "", false, false).unwrap();
        assert!(g.param_expression(n, "control", "rate").is_none(), "binding cleared");
        g.update_param(n, "control", "rate", Param::float(100.0, 0.0, 1000.0)).unwrap();
        let doc = g.serialize();

        let mut g2 = eval_graph();
        g2.load_doc(&doc).unwrap();
        let restored = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.type_name(*u) == Some("_TestDefaultExpr"))
            .expect("node restored");
        assert!(
            g2.param_expression(restored, "control", "rate").is_none(),
            "load must not re-seed a binding the user removed — the doc is authoritative"
        );
        let out = OutputProbe::open(&g2, restored, "out");
        assert_eq!(
            first_f32(&out.expect_frame(&mut g2, "the restored node to emit")),
            100.0,
            "the saved literal survives (not re-rated to the global)"
        );
    }

    #[test]
    fn load_runs_setup_against_the_saved_params_not_the_type_defaults() {
        // A node's one-time init reads its params — it allocates a buffer of `size`, opens device
        // `name`. On load, `setup()` must therefore see the params the user SAVED. The load path
        // built every node from the type's DEFAULTS and applied the saved values only afterwards,
        // and nothing re-runs `setup`. The undo/redo restore path already gets this right
        // (`Command::AddNode` carries the captured params) — the two paths must agree, and they are
        // one path now that a node's params travel with it to the thread that runs its `setup`.
        let mut g = Graph::new();
        let n = g.add_node("_TestSetupLatch", None).unwrap();
        g.update_param(n, "control", "value", Param::float(42.0, 0.0, 100.0)).unwrap();
        let doc = g.serialize();

        let mut g2 = Graph::new();
        g2.load_doc(&doc).unwrap();
        let restored = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.type_name(*u) == Some("_TestSetupLatch"))
            .expect("node restored");
        let out = OutputProbe::open(&g2, restored, "out");
        assert_eq!(
            first_f32(&out.expect_frame(&mut g2, "the restored node to emit")),
            42.0,
            "setup() latched the saved value, not the type default"
        );
    }

    #[test]
    fn param_from_json_coerces_each_type_and_gates_trigger_firing() {
        use serde_json::json;
        // Float: takes as_f64, preserves bounds.
        assert!(matches!(
            param_from_json(&Param::float(0.0, -1.0, 2.0), &json!(0.5), true),
            Param::Float { value, vmin, vmax } if value == 0.5 && vmin == -1.0 && vmax == 2.0
        ));
        // Int: rounds a fractional value to nearest (not zero); a plain int passes through.
        assert!(matches!(param_from_json(&Param::int(0, -10, 10), &json!(5.5), true), Param::Int { value: 6, .. }));
        assert!(matches!(param_from_json(&Param::int(0, -10, 10), &json!(5.4), true), Param::Int { value: 5, .. }));
        assert!(matches!(param_from_json(&Param::int(0, -10, 10), &json!(7), true), Param::Int { value: 7, .. }));
        // Bool.
        assert!(matches!(param_from_json(&Param::boolean(false), &json!(true), true), Param::Bool { value: true }));
        // Str preserves options + refresh.
        let s = Param::Str { value: "a".into(), options: Some(vec!["a".into(), "b".into()]), refresh: true };
        assert!(matches!(
            param_from_json(&s, &json!("b"), true),
            Param::Str { value, options: Some(o), refresh: true } if value == "b" && o == vec!["a".to_string(), "b".to_string()]
        ));
        // Trigger fires only on a live edit (`fire_triggers`); a load (false) never fires, even if the
        // value says true — a persisted/hand-edited `.gfi` must not trip a node's trigger on load.
        assert!(matches!(param_from_json(&Param::Trigger { fired: false }, &json!(true), true), Param::Trigger { fired: true }));
        assert!(matches!(param_from_json(&Param::Trigger { fired: false }, &json!(true), false), Param::Trigger { fired: false }));
    }

    // A test-only passthrough node (ARRAY "in" -> ARRAY "out") to exercise links.
    #[derive(Default)]
    struct Echo;
    impl Node for Echo {
        fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            if let Some(d) = inp.get("in") {
                out.set("out", d.clone());
            }
            Ok(())
        }
    }
    static E_IN: &[SlotDecl] = &[SlotDecl {
        name: "in",
        kind: SlotType::Array,
        trigger_process: true,
        multi: false,
        required: false,
    }];
    static E_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestEcho",
            category: "test",
            doc: "test passthrough",
            inputs: E_IN,
            outputs: E_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<Echo>,
        }
    }

    // A trigger-gated sink WITH a bindable param — used to exercise the "never runs while
    // unwired" path (its expression can error/recover on a node whose process never runs).
    #[derive(Default)]
    struct Sink;
    impl Node for Sink {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Ok(())
        }
    }
    static SINK_IN: &[SlotDecl] = &[SlotDecl {
        name: "in",
        kind: SlotType::Array,
        trigger_process: true,
        multi: false,
        required: false,
    }];
    static SINK_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "control",
        name: "value",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: None,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestSink",
            category: "test",
            doc: "trigger-gated param sink",
            inputs: SINK_IN,
            outputs: &[],
            params: SINK_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<Sink>,
        }
    }

    // A source that only emits on every other run (to exercise trigger arbitration).
    #[derive(Default)]
    struct GatedSource {
        n: i64,
    }
    impl Node for GatedSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let emit = self.n % 2 == 0;
            self.n += 1;
            if emit {
                let d = Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                    .map_err(|e| e.to_string())?;
                out.set("out", d);
            }
            Ok(())
        }
    }
    static G_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestGated",
            category: "test",
            doc: "gated source",
            inputs: &[],
            outputs: G_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<GatedSource>,
        }
    }

    // A CONSUMER that declares `common.max_frequency` itself, with a LIVE expression. The universal
    // declaration would give a non-producer a carried (Off) one, so this is where "we will not
    // overwrite the manifest's param definitions" is observable rather than a coincidence: for a
    // producer the two declarations happen to agree in every field.
    static OWN_COMMON_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "common",
        name: "max_frequency",
        spec: ParamSpec::Float { default: 7.0, min: 0.0, max: 500.0 },
        expression: Some(goofi_node::ExprDecl {
            source: "globals.default_ufreq",
            mode: ExprMode::On,
            trigger: false,
        }),
        doc: None,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestOwnCommon",
            category: "test",
            doc: "a consumer that declares its own common.max_frequency",
            inputs: SINK_IN,
            outputs: G_OUT,
            params: OWN_COMMON_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<GatedSource>,
        }
    }

    // A triggered node that counts the number of times it actually ran.
    #[derive(Default)]
    struct Counter {
        runs: i64,
    }
    impl Node for Counter {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.runs += 1;
            let d = Data::array_f32(vec![1], (self.runs as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static C_IN: &[SlotDecl] = &[SlotDecl {
        name: "in",
        kind: SlotType::Array,
        trigger_process: true,
        multi: false,
        required: false,
    }];
    static C_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestCounter",
            category: "test",
            doc: "run counter",
            inputs: C_IN,
            outputs: C_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<Counter>,
        }
    }

    // A two-input node summing a[0]+b[0] — exercises fan-in convergence, where a
    // consumer at a later level must receive fresh frames from two producers that
    // ran (in parallel) at the same earlier level.
    #[derive(Default)]
    struct Adder;
    impl Node for Adder {
        fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let (Some(a), Some(b)) = (inp.get("a"), inp.get("b")) else {
                return Ok(());
            };
            let sum = first_f32(a) + first_f32(b);
            let d = Data::array_f32(vec![1], sum.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static ADD_IN: &[SlotDecl] = &[
        SlotDecl { name: "a", kind: SlotType::Array, trigger_process: true, multi: false, required: false },
        SlotDecl { name: "b", kind: SlotType::Array, trigger_process: true, multi: false, required: false },
    ];
    static ADD_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestAdder",
            category: "test",
            doc: "a[0] + b[0]",
            inputs: ADD_IN,
            outputs: ADD_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<Adder>,
        }
    }

    // A source that sleeps in process() — used to prove independent nodes at the
    // same topological level actually run concurrently (wall-clock < sum).
    struct Slow {
        ms: u64,
    }
    impl Default for Slow {
        fn default() -> Slow {
            Slow { ms: 20 }
        }
    }
    impl Node for Slow {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            std::thread::sleep(std::time::Duration::from_millis(self.ms));
            let d = Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static SLOW_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestSlow",
            category: "test",
            doc: "sleeps 20ms then emits",
            inputs: &[],
            outputs: SLOW_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<Slow>,
        }
    }

    // A node that panics in process() — to verify the engine survives it.
    #[derive(Default)]
    struct Panicky;
    impl Node for Panicky {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            panic!("boom");
        }
    }
    static P_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestPanic",
            category: "test",
            doc: "panics",
            inputs: &[],
            outputs: P_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<Panicky>,
        }
    }

    // The only STRING-slotted node in the catalog — it exists so a test can build the
    // cross-dtype cable `add_link` must refuse.
    #[derive(Default)]
    struct Text;
    impl Node for Text {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Ok(())
        }
    }
    static TXT_IN: &[SlotDecl] = &[SlotDecl {
        name: "words",
        kind: SlotType::String,
        trigger_process: true,
        multi: false,
        required: false,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestText",
            category: "test",
            doc: "string sink",
            inputs: TXT_IN,
            outputs: &[],
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<Text>,
        }
    }

    // A free-running counter capped at 10 Hz via a `common` group — exercises the
    // wall-clock rate gate. Emits its run count so a test can read how often it ran.
    #[derive(Default)]
    struct CappedSource {
        runs: i64,
    }
    impl Node for CappedSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.runs += 1;
            let d = Data::array_f32(vec![1], (self.runs as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    // 10 Hz (-> 0.1s), autotriggering. `frequency_mode` is filled by `with_common`.
    static CAPPED_PARAMS: &[ParamDecl] = &[
        ParamDecl { group: "common", name: "autotrigger", spec: ParamSpec::Bool { default: true },
            expression: None, doc: None },
        ParamDecl { group: "common", name: "max_frequency", spec: ParamSpec::Float { default: 10.0, min: 0.0, max: 60.0 },
            expression: None, doc: None },
    ];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestCapped",
            category: "test",
            doc: "10 Hz free-running counter",
            inputs: &[],
            outputs: G_OUT,
            params: CAPPED_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<CappedSource>,
        }
    }

    // A node with a TRIGGERING "data" input and a NON-triggering "ref" (control)
    // input, emitting a length-1 frame. Used to prove index propagation ignores a
    // control input even when its length coincidentally matches the output's.
    #[derive(Default)]
    struct RefLenChange;
    impl Node for RefLenChange {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let d = Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static REF_IN: &[SlotDecl] = &[
        SlotDecl { name: "data", kind: SlotType::Array, trigger_process: true, multi: false, required: false },
        SlotDecl { name: "ref", kind: SlotType::Array, trigger_process: false, multi: false, required: false },
    ];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestRefLenChange",
            category: "test",
            doc: "triggering data + non-triggering ref; emits len-1",
            inputs: REF_IN,
            outputs: C_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<RefLenChange>,
        }
    }

    // A source that MIRRORS its param to a field in `on_param_changed` and emits the FIELD — the
    // documented hot-param authoring pattern (`Oscillator.sfreq` is the shipped instance). A node
    // that read `p` live in `process` could not tell a dispatched hook from a skipped one; this one
    // can. Slot 0 is the mirrored value, slot 1 the number of hook calls, so a test can also assert
    // that a settled binding stops re-dispatching.
    #[derive(Default)]
    struct MirrorSource {
        mirrored: f32,
        calls: f32,
    }
    impl Node for MirrorSource {
        fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
            if key.group == "mirror" && key.name == "value" {
                self.mirrored = v.as_f64().unwrap_or(f64::NAN) as f32;
                self.calls += 1.0;
            }
            Ok(())
        }
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let mut buf = self.mirrored.to_le_bytes().to_vec();
            buf.extend_from_slice(&self.calls.to_le_bytes());
            let d = Data::array_f32(vec![2], buf, Meta::empty()).map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static MIRROR_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "mirror",
        name: "value",
        spec: ParamSpec::Float { default: 1.0, min: -1e9, max: 1e9 },
        expression: None,
        doc: None,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestMirror",
            category: "test",
            doc: "mirrors mirror.value to a field via on_param_changed and emits [field, hook_calls]",
            inputs: &[],
            outputs: G_OUT,
            params: MIRROR_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<MirrorSource>,
        }
    }

    // A source that emits the engine-supplied wall clock (ctx.now) as its value,
    // to prove NodeCtx::now advances deterministically under an injected clock.
    #[derive(Default)]
    struct NowSource;
    impl Node for NowSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let d = Data::array_f32(vec![1], (c.now as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    inventory::submit! {
        NodeManifest {
            type_name: "_TestNow",
            category: "test",
            doc: "emits ctx.now",
            inputs: &[],
            outputs: G_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<NowSource>,
        }
    }

    // A source that emits the live `default_ufreq` global from its NodeCtx — proving the
    // engine feeds `process` the current globals snapshot each tick (a mid-run edit is seen
    // on the next run, not latched at setup).
    #[derive(Default)]
    struct GlobalSource;
    impl Node for GlobalSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let v = c.globals.f64("default_ufreq").unwrap_or(-1.0) as f32;
            let d = Data::array_f32(vec![1], v.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    inventory::submit! {
        NodeManifest {
            type_name: "_TestGlobal",
            category: "test",
            doc: "emits ctx.globals['default_ufreq']",
            inputs: &[],
            outputs: G_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<GlobalSource>,
        }
    }

    // A source whose `control.rate` param declares a `default_expr` — proving a fresh add seeds a
    // live binding (not a plain literal). It emits the param's current value so a test can watch the
    // binding evaluate + re-rate; the 5.0 spec default is the no-evaluator fallback.
    #[derive(Default)]
    struct DefaultExprSource;
    impl Node for DefaultExprSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
            let v = p.f64("control", "rate").unwrap_or(-1.0) as f32;
            let d = Data::array_f32(vec![1], v.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static DEFAULT_EXPR_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "control",
        name: "rate",
        spec: ParamSpec::Float { default: 5.0, min: 0.0, max: 1000.0 },
        expression: Some(ExprDecl {
            source: "globals.default_ufreq",
            mode: ExprMode::On,
            trigger: false,
        }),
        doc: None,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestDefaultExpr",
            category: "test",
            doc: "control.rate has a default_expr binding",
            inputs: &[],
            outputs: G_OUT,
            params: DEFAULT_EXPR_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<DefaultExprSource>,
        }
    }

    // A source declaring the two `ExprDecl` shapes `_TestDefaultExpr` cannot express — a CARRIED
    // expression (mode Off) and a TRIGGERING one. Both are needed as a fixture: seeding that
    // hard-codes `enabled = true, triggers_process = false` reproduces `_TestDefaultExpr` exactly
    // and would pass against it, so only a node declaring the other values can catch that.
    static CARRIED_PARAMS: &[ParamDecl] = &[
        ParamDecl {
            group: "control",
            name: "carried",
            spec: ParamSpec::Float { default: 5.0, min: 0.0, max: 1000.0 },
            expression: Some(ExprDecl {
                source: "globals.default_ufreq",
                mode: ExprMode::Off,
                trigger: false,
            }),
            doc: None,
        },
        ParamDecl {
            group: "control",
            name: "paced",
            spec: ParamSpec::Float { default: 5.0, min: 0.0, max: 1000.0 },
            expression: Some(ExprDecl {
                source: "globals.default_ufreq",
                mode: ExprMode::On,
                trigger: true,
            }),
            doc: None,
        },
    ];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestCarriedExpr",
            category: "test",
            doc: "one carried (Off) expression and one triggering one",
            inputs: &[],
            outputs: G_OUT,
            params: CARRIED_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<DefaultExprSource>,
        }
    }

    // A source that LATCHES a param in `setup()` and emits the latched value forever — the shape of
    // every real one-time init (allocate a buffer of `size`, open device `name`), and the only way
    // to observe which params `setup` actually saw.
    #[derive(Default)]
    struct SetupLatch {
        latched: f32,
    }
    impl Node for SetupLatch {
        fn setup(&mut self, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
            self.latched = p.f64("control", "value").unwrap_or(-1.0) as f32;
            Ok(())
        }
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let d = Data::array_f32(vec![1], self.latched.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static SETUP_LATCH_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "control",
        name: "value",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 100.0 },
        expression: None,
        doc: None,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestSetupLatch",
            category: "test",
            doc: "latches control.value in setup() and emits it",
            inputs: &[],
            outputs: G_OUT,
            params: SETUP_LATCH_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<SetupLatch>,
        }
    }

    // A pure source with two output slots at different cadences: "fast" emits every
    // run, "slow" every other run — to prove the node-level ufreq is stamped identically
    // on every slot (not each slot's own cadence).
    #[derive(Default)]
    struct TwoRate {
        n: i64,
    }
    impl Node for TwoRate {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.n += 1;
            let mk = || {
                Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                    .map_err(|e| e.to_string())
            };
            out.set("fast", mk()?);
            if self.n % 2 == 0 {
                out.set("slow", mk()?);
            }
            Ok(())
        }
    }
    static TWO_OUT: &[OutputDecl] = &[
        OutputDecl { name: "fast", kind: SlotType::Array },
        OutputDecl { name: "slow", kind: SlotType::Array },
    ];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestTwoRate",
            category: "test",
            doc: "fast slot every run, slow slot every other run",
            inputs: &[],
            outputs: TWO_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: default_factory::<TwoRate>,
        }
    }

    // A node with a MULTI triggering input "ins". Emits [count, v0, v1, …] where vi
    // is the first element of each received frame in connection order — so a test can
    // read the fan-in count, order, and latest-wins. Autotriggers so the 0-wire
    // (empty-list) case still runs.
    #[derive(Default)]
    struct Collect;
    impl Node for Collect {
        fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let items = inp.get_multi("ins");
            let mut vals: Vec<f32> = vec![items.len() as f32];
            vals.extend(items.iter().map(first_f32));
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            let d = Data::array_f32(vec![vals.len()], bytes, Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static COLLECT_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "common",
        name: "autotrigger",
        spec: ParamSpec::Bool { default: true },
        expression: None,
        doc: None,
    }];
    static COLLECT_IN: &[SlotDecl] = &[SlotDecl {
        name: "ins",
        kind: SlotType::Array,
        trigger_process: true,
        multi: true,
        required: false,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestCollect",
            category: "test",
            doc: "multi-input: emits [count, v0, v1, …] of its wires in connection order",
            inputs: COLLECT_IN,
            outputs: G_OUT,
            params: COLLECT_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<Collect>,
        }
    }

    // A node with a REQUIRED input slot that counts and emits its own `process` calls. The
    // counter is the load-bearing part: an assertion on the error message alone would also
    // hold for a check placed AFTER `node.process`, which is the bug the contract exists to
    // prevent, so the tests read the count back to prove `process` was never entered.
    #[derive(Default)]
    struct Required {
        runs: i64,
    }
    impl Node for Required {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.runs += 1;
            let d = Data::array_f32(vec![1], (self.runs as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static REQ_IN: &[SlotDecl] = &[SlotDecl {
        name: "data",
        kind: SlotType::Array,
        trigger_process: true,
        multi: false,
        required: true,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestRequired",
            category: "test",
            doc: "one required input; emits its own run count",
            inputs: REQ_IN,
            outputs: G_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<Required>,
        }
    }

    // The multi analogue: a required VARIADIC slot, whose frames live in `multi_inputs` rather
    // than `inputs` — so the presence check has to look in the other place.
    #[derive(Default)]
    struct RequiredMulti;
    impl Node for RequiredMulti {
        fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let n = inp.get_multi("ins").len() as f32;
            let d = Data::array_f32(vec![1], n.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static REQ_MULTI_IN: &[SlotDecl] = &[SlotDecl {
        name: "ins",
        kind: SlotType::Array,
        trigger_process: true,
        multi: true,
        required: true,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestRequiredMulti",
            category: "test",
            doc: "one required multi input; emits its wire count",
            inputs: REQ_MULTI_IN,
            outputs: G_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<RequiredMulti>,
        }
    }

    // The D1 shape — `Required`'s run counter behind a two-slot interface: a required slot that
    // does NOT trigger, beside a triggering one that does. That is what lets a test wire the
    // required slot to a producer and still tick the node while the slot is empty.
    static REQ_PAIR_IN: &[SlotDecl] = &[
        SlotDecl { name: "data", kind: SlotType::Array, trigger_process: false, multi: false, required: true },
        SlotDecl { name: "tick", kind: SlotType::Array, trigger_process: true, multi: false, required: false },
    ];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestRequiredPair",
            category: "test",
            doc: "a required non-triggering slot beside a triggering one",
            inputs: REQ_PAIR_IN,
            outputs: G_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: false,
            factory: default_factory::<Required>,
        }
    }

    // A node whose INITIALIZATION the test controls: `setup()` fails unless its `boot.ok` param
    // says otherwise, so correcting that param is a real retry door (D3) rather than a
    // fixture-only hook. Every `process` entry, every `on_param_changed` and every `setup` it
    // receives is counted in a cell the test reads directly — the counters are the load-bearing
    // part. An assertion on the error MESSAGE alone holds just as well for a gate that reports the
    // failure and then runs `process` anyway, which is exactly what D3 forbids.
    #[derive(Default)]
    struct GateCounts {
        runs: usize,
        param_calls: usize,
        setups: usize,
    }
    type SharedCounts = std::sync::Arc<std::sync::Mutex<GateCounts>>;
    struct GatedSetup {
        counts: SharedCounts,
    }
    impl Node for GatedSetup {
        fn setup(&mut self, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
            self.counts.lock().unwrap().setups += 1;
            match p.bool("boot", "ok") {
                Some(true) => Ok(()),
                _ => Err("device is not open".into()),
            }
        }
        fn on_param_changed(&mut self, _k: &ParamKey, _v: &Param) -> NodeResult {
            self.counts.lock().unwrap().param_calls += 1;
            Ok(())
        }
        fn on_param_refreshed(&mut self, _k: &ParamKey, _p: &Params<'_>) -> Option<Vec<String>> {
            Some(vec!["dev0".to_string()])
        }
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let mut c = self.counts.lock().unwrap();
            c.runs += 1;
            let d = Data::array_f32(vec![1], (c.runs as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static GATED_PARAMS: &[ParamDecl] = &[
        ParamDecl {
            group: "boot",
            name: "ok",
            spec: ParamSpec::Bool { default: false },
            expression: None,
            doc: None,
        },
        ParamDecl {
            group: "boot",
            name: "device",
            spec: ParamSpec::Str { default: "none", options: &["none"], refresh: true },
            expression: None,
            doc: None,
        },
    ];
    static GATED_MANIFEST: NodeManifest = NodeManifest {
        type_name: "_GatedSetup",
        category: "test",
        doc: "setup() fails until its `boot.ok` param says otherwise",
        inputs: &[],
        outputs: G_OUT,
        params: GATED_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    /// Register the gated type and add one instance, with the counters its node writes to. The
    /// instance arrives UNINITIALIZED — `boot.ok` defaults false — which is the state every one of
    /// these tests starts from.
    fn gated_setup_node(g: &mut Graph) -> (Uid, SharedCounts) {
        let counts: SharedCounts = Default::default();
        let mine = counts.clone();
        g.register_dyn_type(
            &GATED_MANIFEST,
            Box::new(move |_p| Box::new(GatedSetup { counts: mine.clone() })),
        );
        let uid = g.add_node("_GatedSetup", None).unwrap();
        (uid, counts)
    }

    // `_TestConst` (the constant-array source these tests use as a generic value
    // source) is the hidden test node in goofi-nodes — one shared definition across
    // the engine, bridge, and goofi-python suites.

    fn first_f32(d: &Data) -> f32 {
        if let Value::Array(s) = d.value() {
            f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap())
        } else {
            panic!("not an array")
        }
    }

    fn as_f32_vec(d: &Data) -> Vec<f32> {
        if let Value::Array(s) = d.value() {
            s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        } else {
            panic!("not an array")
        }
    }

    fn const_src(g: &mut Graph, v: f32) -> Uid {
        let u = g.add_node("_TestConst", None).unwrap();
        g.update_param(u, "constant", "value", Param::float(v as f64, -1e9, 1e9)).unwrap();
        u
    }

    /// Add a node BORN capped at `hz`. A param edit cannot do this: a node free-runs from birth
    /// until the `SetParam` reaches it, and uncapped that is thousands of runs — so a test about a
    /// node's FIRST frame, or its first failure, has to cap it before it has one. The probe's cell
    /// is one deep and latest-wins, and a 30 kHz producer overwrites frame 1 long before anything
    /// looks.
    fn capped(g: &mut Graph, type_name: &str, hz: f64) -> Uid {
        let mut params = g.default_params_of(type_name).unwrap();
        params
            .entry("common".to_string())
            .or_default()
            .insert("max_frequency".to_string(), Param::float(hz, 0.0, 1e9));
        g.add_node(type_name, Some(params)).unwrap()
    }

    #[test]
    fn a_param_edited_before_its_node_was_addressable_still_reaches_it() {
        // §4's birth barrier: a `Control` sent to a node that has not yet reported `Ready` is
        // published to a subscriber that does not exist, and pub/sub has no history — so it is
        // simply lost. `add_node` answers long before that report is drained, which makes the
        // window every ordinary `add_node(); update_param()` pair falls into.
        //
        // `attach_control_sink` re-plans what it finds by walking links and BINDINGS, and a plain
        // literal param is neither. Measured with that gap open: 20 of this crate's own tests, each
        // one watching a node emit its type default forever.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let out = OutputProbe::open(&g, src, "out");
        // The graph attaches a channel in `apply_status` and nowhere else, and nothing has drained
        // one yet — which `node_stage` states race-free, since it reads what the graph has APPLIED
        // rather than what the node has sent.
        assert_eq!(g.node_stage(src), "creating", "the graph has heard nothing, so there is no sink");
        g.update_param(src, "constant", "value", Param::float(3.0, -1e9, 1e9)).unwrap();

        out.wait_until(&mut g, "carries the value written before the node was addressable", |d| {
            first_f32(d) == 3.0
        });
    }

    #[test]
    fn source_streams_latest_frame() {
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(7.0, -1e9, 1e9))
            .unwrap();
        let out = OutputProbe::open(&g, src, "out");
        // `wait_until` rather than "the next frame": the node was already running when the param
        // was written, so the frame in flight may still carry the old value. What the test means is
        // that the edit reaches the stream, not which emit carries it.
        out.wait_until(&mut g, "carries the edited value", |d| first_f32(d) == 7.0);
    }

    #[test]
    fn link_propagates_in_one_tick() {
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(5.0, -1e9, 1e9))
            .unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10))
            .unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        let out = OutputProbe::open(&g, echo, "out");
        g.add_link(src, "out", echo, "in").unwrap();
        // The link is carried by the three-phase sequence, which advances on acks — so the graph
        // has to be drained for the wire to exist at all.
        wait_for(&mut g, "the wire to attach", |_| out.latest().is_some());

        let f = out.expect_frame(&mut g, "the echo to emit");
        if let Value::Array(s) = f.value() {
            assert_eq!(s.shape(), &[2]);
        } else {
            panic!("expected array");
        }
        assert_eq!(first_f32(&f), 5.0);
    }

    #[test]
    fn one_wire_per_input_evicts_prior_source() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestConst", None).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        g.update_param(a, "constant", "value", Param::float(1.0, -1e9, 1e9))
            .unwrap();
        g.update_param(b, "constant", "value", Param::float(2.0, -1e9, 1e9))
            .unwrap();
        let out = OutputProbe::open(&g, echo, "out");
        g.add_link(a, "out", echo, "in").unwrap();
        g.add_link(b, "out", echo, "in").unwrap(); // evicts a
        wait_for(&mut g, "the second wire to displace the first", |_| {
            out.latest().is_some_and(|d| first_f32(&d) == 2.0)
        });
        assert!(stays(&mut g, |_| out.latest().is_some_and(|d| first_f32(&d) == 2.0)), "and a is gone");
    }

    // ---- multi-input slots -------------------------------------------------

    /// Drain the graph until the collector emits exactly `want`, then hold it there for a settle
    /// window. Both halves matter: a wire change lands asynchronously, so the first frame after it
    /// may still be the old set — and a set that is merely PASSED THROUGH on the way to another one
    /// would satisfy a bare "eventually" on its own.
    fn collects(g: &mut Graph, out: &OutputProbe, want: &[f32]) {
        let matches = |d: &Data| as_f32_vec(d) == want;
        out.wait_until(g, &format!("collects {want:?}"), matches);
        assert!(
            stays(g, |_| out.latest().is_some_and(|d| matches(&d))),
            "and settles there rather than passing through: last was {:?}",
            out.latest().map(|d| as_f32_vec(&d)),
        );
    }

    #[test]
    fn multi_input_collects_wires_in_connection_order() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        let out = OutputProbe::open(&g, col, "out");
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();
        // [count=3, then each wire's value in connection order].
        collects(&mut g, &out, &[3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn multi_input_remove_link_drops_one_wire_keeping_order() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        let out = OutputProbe::open(&g, col, "out");
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();
        g.remove_link(b, "out", col, "ins").unwrap();
        collects(&mut g, &out, &[2.0, 1.0, 3.0]);
    }

    #[test]
    fn multi_input_remove_node_drops_its_wires() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        let out = OutputProbe::open(&g, col, "out");
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();
        g.remove_node(b).unwrap();
        collects(&mut g, &out, &[2.0, 1.0, 3.0]);
    }

    #[test]
    fn multi_input_latest_wins_per_wire() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        let out = OutputProbe::open(&g, col, "out");
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        collects(&mut g, &out, &[2.0, 1.0, 2.0]);
        // a's next frame overwrites its cell (latest-wins); b is retained; order stable.
        g.update_param(a, "constant", "value", Param::float(9.0, -1e9, 1e9)).unwrap();
        collects(&mut g, &out, &[2.0, 9.0, 2.0]);
    }

    #[test]
    fn multi_input_empty_slot_is_empty_list() {
        let mut g = Graph::new();
        let col = g.add_node("_TestCollect", None).unwrap(); // autotriggers with 0 wires
        let out = OutputProbe::open(&g, col, "out");
        collects(&mut g, &out, &[0.0]);
    }

    #[test]
    fn multi_input_wires_round_trip_in_connection_order() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();

        let yaml = g.serialize();
        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_count(), 4);

        let col2 = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.type_name(*u) == Some("_TestCollect"))
            .expect("collect restored");
        let out = OutputProbe::open(&g2, col2, "out");
        // All 3 wires restored, in connection order (a=1, b=2, c=3).
        collects(&mut g2, &out, &[3.0, 1.0, 2.0, 3.0]);
    }

    // ---- required input slots ----------------------------------------------

    #[test]
    fn a_required_input_with_no_frame_errors_before_process_is_entered() {
        let mut g = Graph::new();
        // Born capped, so the run that first HAS data is still on the wire when the probe looks —
        // the counter is the oracle and an uncapped node has overwritten `1` thousands of times.
        let n = capped(&mut g, "_TestRequired", 5.0);
        // Autotrigger is what makes an unwired node run at all — D1: the check fires on a RUN,
        // never on the configuration.
        g.update_param(n, "common", "autotrigger", Param::boolean(true)).unwrap();
        let out = OutputProbe::open(&g, n, "out");
        wait_for(&mut g, "the empty required slot to be named", |g| {
            g.last_error(n) == Some("required input slot `data` has no data")
        });
        assert!(out.silent(&mut g), "a refused run emits nothing");
        // …and `process` was never ENTERED, not merely denied its output. The node counts its own
        // calls, so once the slot is fed the FIRST frame must read 1; a check placed AFTER
        // `node.process` would have counted every refused run before it.
        let src = const_src(&mut g, 4.0);
        g.add_link(src, "out", n, "data").unwrap();
        assert_eq!(
            first_f32(&out.expect_frame(&mut g, "the fed node to run")),
            1.0,
            "process was entered exactly once, on the run that had data",
        );
        wait_for(&mut g, "the fed slot to clear the error", |g| g.last_error(n).is_none());
    }

    #[test]
    fn a_required_input_holding_a_frame_runs_cleanly() {
        let mut g = Graph::new();
        let src = const_src(&mut g, 7.0);
        let n = capped(&mut g, "_TestRequired", 5.0);
        let out = OutputProbe::open(&g, n, "out");
        g.add_link(src, "out", n, "data").unwrap();
        assert_eq!(first_f32(&out.expect_frame(&mut g, "the node to run")), 1.0, "process ran");
        wait_for(&mut g, "a satisfied required slot to report no error", |g| g.last_error(n).is_none());
    }

    #[test]
    fn a_required_multi_input_with_no_frames_errors() {
        let mut g = Graph::new();
        let n = capped(&mut g, "_TestRequiredMulti", 5.0);
        g.update_param(n, "common", "autotrigger", Param::boolean(true)).unwrap();
        let out = OutputProbe::open(&g, n, "out");
        wait_for(&mut g, "the unwired variadic slot to be named", |g| {
            g.last_error(n) == Some("required input slot `ins` has no data")
        });
        assert!(out.silent(&mut g), "a refused run emits nothing");
        // Wire one source and the same node runs, seeing its one frame.
        let src = const_src(&mut g, 1.0);
        g.add_link(src, "out", n, "ins").unwrap();
        assert_eq!(first_f32(&out.expect_frame(&mut g, "the node to run")), 1.0, "one wire present");
        wait_for(&mut g, "the error to clear", |g| g.last_error(n).is_none());
    }

    #[test]
    fn a_required_input_on_a_node_that_never_ticks_is_silent() {
        // D1 again, from the other side: an unwired node with no autotrigger is "a disconnected
        // node floating in space" — we never asked it to run, so it has nothing to report.
        let mut g = Graph::new();
        let n = g.add_node("_TestRequired", None).unwrap();
        let out = OutputProbe::open(&g, n, "out");
        assert!(out.silent(&mut g), "it emitted nothing");
        assert!(
            stays(&mut g, |g| g.last_error(n).is_none()),
            "a node that never ran cannot be missing an input",
        );
    }

    #[test]
    fn a_required_slot_wired_to_a_producer_that_has_emitted_nothing_is_still_refused() {
        // Invariant 1, which nothing else pins: the check reads the LAST-STORE, never the link
        // table. Every other test here contrasts unwired-and-empty against wired-and-fed, and those
        // two are indistinguishable under either rule — so a wiring test would pass them all.
        //
        // The discriminating part is the producer: `silent` is an autotriggered `_TestRequired`
        // whose OWN required slot is empty, so it is refused every tick and emits nothing. `data` is
        // therefore wired and empty at once, and the second slot is what makes the node tick at all
        // (D1's headline case).
        let mut g = Graph::new();
        let silent = g.add_node("_TestRequired", None).unwrap();
        g.update_param(silent, "common", "autotrigger", Param::boolean(true)).unwrap();
        let n = g.add_node("_TestRequiredPair", None).unwrap();
        g.add_link(silent, "out", n, "data").unwrap();
        let src = const_src(&mut g, 1.0);
        let out = OutputProbe::open(&g, n, "out");
        g.add_link(src, "out", n, "tick").unwrap();

        wait_for(&mut g, "the wired-but-empty slot to be named", |g| {
            g.last_error(n) == Some("required input slot `data` has no data")
        });
        assert!(out.silent(&mut g), "so `process` was never entered");
    }

    // ---- the initialization gate (D3) --------------------------------------

    #[test]
    fn a_failed_setup_stands_and_the_node_never_enters_process() {
        // D3 end to end, across the status service: the failure the node reported at birth is what
        // the graph keeps reporting, and `process` is unreachable underneath it. The run COUNTER is
        // the load-bearing half — a gate that reports the failure and runs `process` anyway passes
        // an assertion on the message alone, which is the bug the contract exists to prevent.
        let mut g = Graph::new();
        let (n, counts) = gated_setup_node(&mut g);
        let out = OutputProbe::open(&g, n, "out");
        wait_for(&mut g, "the setup failure to reach the graph", |g| {
            g.last_error(n) == Some("device is not open")
        });
        assert!(
            stays(&mut g, |g| g.node_stage(n) == "error"),
            "the editor draws it errored, not ready, and it stays that way",
        );
        assert!(out.silent(&mut g), "the node emitted nothing");
        assert_eq!(counts.lock().unwrap().runs, 0, "process was never entered");
    }

    #[test]
    fn correcting_the_param_that_broke_setup_reinitializes_the_node() {
        // D3's retry door, end to end: `update_param` stores the new value and announces it, and
        // the node's own retry replays the record — which is what makes fixing the param the fix.
        //
        // The counters are read as DELTAS across the edit, not as absolutes: the node retries its
        // own initialization on a wall-clock backoff, so how many attempts it has made by the time
        // the edit lands is a function of how long the harness took to get here.
        let mut g = Graph::new();
        let (n, counts) = gated_setup_node(&mut g);
        let out = OutputProbe::open(&g, n, "out");
        wait_for(&mut g, "the setup failure to reach the graph", |g| {
            g.last_error(n) == Some("device is not open")
        });
        let before = { let c = counts.lock().unwrap(); (c.setups, c.param_calls) };

        assert!(g.update_param(n, "boot", "ok", Param::boolean(true)).is_ok(), "the edit is accepted");
        wait_for(&mut g, "the retry to initialize the node", |g| g.last_error(n).is_none());
        assert_eq!(g.node_stage(n), "ready");
        // "For the first time" is read off the node's own counter rather than off a frame: the
        // healed node free-runs, so the probe's one-deep cell has long stopped holding run 1.
        wait_for(&mut g, "the healed node to run", |_| counts.lock().unwrap().runs > 0);
        assert!(out.frame(&mut g).is_some(), "and what it ran reached the data plane");
        let c = counts.lock().unwrap();
        assert_eq!(c.setups, before.0 + 1, "the edit retried the initialization exactly once");
        // The retry replayed BOTH params through `on_param_changed`. `update_param` delivering its
        // own edit on top would add a third — and double-apply the handler's side effect.
        assert_eq!(c.param_calls, before.1 + 2, "the edit reached the node once, through the replay");
    }

    #[test]
    fn a_failed_retry_keeps_the_node_uninitialized_and_still_accepts_the_edit() {
        // An edit that does NOT fix what broke `setup()`. Refusing it (returning Err) would refuse
        // the very interaction that is the retry door, and `update_param` is a command whose
        // inverse must stay in step with the session's history.
        let mut g = Graph::new();
        let (n, counts) = gated_setup_node(&mut g);
        wait_for(&mut g, "the setup failure to reach the graph", |g| {
            g.last_error(n) == Some("device is not open")
        });
        let picked = Param::Str { value: "hw:1".into(), options: None, refresh: true };
        assert!(g.update_param(n, "boot", "device", picked).is_ok(), "the edit is stored, not refused");
        assert!(
            stays(&mut g, |g| g.last_error(n) == Some("device is not open") && g.node_stage(n) == "error"),
            "the node is still uninitialized",
        );
        assert_eq!(counts.lock().unwrap().runs, 0, "and nothing ran against it");
    }

    #[test]
    fn a_refresh_on_an_uninitialized_node_enumerates_nothing() {
        // §8.5 moved the answer off the RPC: `refresh_param` always answers `Ok(None)` because the
        // hook runs on the node's own thread. D3 still gates it there — a picker whose node failed
        // `setup()` has nothing to scan — so the observable is the OPTIONS in the record, which the
        // node's report writes. The node's hook does return a list, so a missing gate reads as a
        // successful scan.
        let mut g = Graph::new();
        let (n, _counts) = gated_setup_node(&mut g);
        assert_eq!(g.refresh_param(n, "boot", "device").unwrap(), None, "the answer never rides the RPC");
        assert!(
            stays(&mut g, |g| device_options(g, n) == vec!["none".to_string()]),
            "an uninitialized node enumerated nothing",
        );

        // Once it initializes, the same call reaches the hook and the record moves.
        g.update_param(n, "boot", "ok", Param::boolean(true)).unwrap();
        wait_for(&mut g, "the node to initialize", |g| g.last_error(n).is_none());
        g.refresh_param(n, "boot", "device").unwrap();
        wait_for(&mut g, "the scanned options to reach the record", |g| {
            device_options(g, n) == vec!["dev0".to_string()]
        });
    }

    /// The `boot.device` param's current options — what a refresh rewrites, and nothing else does.
    fn device_options(g: &Graph, uid: Uid) -> Vec<String> {
        match goofi_node::param(&g.params(uid).unwrap(), "boot", "device") {
            Some(Param::Str { options: Some(o), .. }) => o.clone(),
            other => panic!("expected a Str param with options, got {other:?}"),
        }
    }

    #[test]
    fn remove_node_drops_links() {
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        g.add_link(src, "out", echo, "in").unwrap();
        let out = OutputProbe::open(&g, echo, "out");
        g.remove_node(src).unwrap();
        assert!(!g.contains(src));
        assert!(out.silent(&mut g), "the echo has no input left, so nothing triggers it");
    }

    #[test]
    fn trigger_arbitration_gates_downstream() {
        // A consumer runs once per frame its producer emits, and on nothing else. `_TestGated`
        // emits on every other run of its own, so the counter must stay strictly behind it —
        // a consumer that free-ran would overtake it immediately.
        let mut g = Graph::new();
        let src = g.add_node("_TestGated", None).unwrap();
        let cnt = g.add_node("_TestCounter", None).unwrap(); // triggered
        let gated = OutputProbe::open(&g, src, "out");
        let counted = OutputProbe::open(&g, cnt, "out");
        g.add_link(src, "out", cnt, "in").unwrap();
        wait_for(&mut g, "the wire to carry a frame", |_| counted.latest().is_some());

        // The producer's own index counts its EMITS; the counter counts its RUNS. One run per emit
        // means the counter can never be ahead, and the emits it skipped keep it behind.
        let emits = gated.expect_frame(&mut g, "the gated source to emit").meta().index().unwrap_or(0) + 1;
        let runs = first_f32(&counted.expect_frame(&mut g, "the counter to run")) as u64;
        assert!(runs <= emits, "the counter ran {runs} times for {emits} emits");
    }

    #[test]
    fn unwired_triggered_node_never_runs() {
        let mut g = Graph::new();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        let out = OutputProbe::open(&g, cnt, "out");
        assert!(out.silent(&mut g), "a triggered node with no wired input must never run");
    }

    #[test]
    fn gfi_serialize_load_roundtrip() {
        let mut g = Graph::new();
        let c = g.add_node("_TestConst", None).unwrap();
        g.update_param(c, "constant", "value", Param::float(7.5, -1e9, 1e9))
            .unwrap();
        g.rename_node(c, "myconst").unwrap();
        g.set_node_pos(c, [11.0, 22.0]).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        g.add_link(c, "out", echo, "in").unwrap();

        let yaml = g.serialize();
        assert!(yaml.contains("version: 7"));

        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_count(), 2);

        let restored = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.name(*u) == Some("myconst"))
            .expect("named node restored");
        assert_eq!(g2.type_name(restored), Some("_TestConst"));
        assert_eq!(g2.pos(restored), Some([11.0, 22.0]));
        assert_eq!(
            goofi_node::param(&g2.params(restored).unwrap(), "constant", "value")
                .unwrap()
                .as_f64(),
            Some(7.5)
        );

        // The link round-trips: the restored source drives the restored echo.
        let echo2 = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.type_name(*u) == Some("_TestEcho"))
            .unwrap();
        let out = OutputProbe::open(&g2, echo2, "out");
        wait_for(&mut g2, "the restored wire to carry data", |_| out.latest().is_some());
        assert_eq!(first_f32(&out.expect_frame(&mut g2, "the echo to emit")), 7.5);
    }

    #[test]
    fn load_doc_rejects_unknown_type_before_teardown() {
        let mut g = Graph::new();
        g.add_node("_TestConst", None).unwrap();
        let before = g.node_count();
        let bad = "version: 7\nroot:\n  nodes:\n    \"00000000000a\":\n      type: NotAReal Node\n      pos: [0, 0]\n  links: []\n";
        let err = g.load_doc(bad).unwrap_err();
        // Name the type, so a future version-gate change can't make this pass for the wrong reason.
        assert!(err.contains("NotAReal Node"), "rejected on the type, not the version: {err}");
        // validate-before-teardown: the existing graph is untouched on failure.
        assert_eq!(g.node_count(), before);
    }

    #[test]
    fn independent_branches_both_produce_correctly() {
        // Two disjoint _TestConst -> Echo branches, each carrying its OWN value. Distinct values
        // are the whole oracle: a propagation that crossed the branches, or one wire feeding both
        // consumers, reads identically to a correct one if both branches carry the same number.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let ea = g.add_node("_TestEcho", None).unwrap();
        g.update_param(a, "constant", "value", Param::float(3.0, -1e9, 1e9)).unwrap();
        let out_a = OutputProbe::open(&g, ea, "out");
        g.add_link(a, "out", ea, "in").unwrap();

        let b = g.add_node("_TestConst", None).unwrap();
        let eb = g.add_node("_TestEcho", None).unwrap();
        g.update_param(b, "constant", "value", Param::float(4.0, -1e9, 1e9)).unwrap();
        let out_b = OutputProbe::open(&g, eb, "out");
        g.add_link(b, "out", eb, "in").unwrap();

        wait_for(&mut g, "both branches to carry data", |_| {
            out_a.latest().is_some() && out_b.latest().is_some()
        });
        out_a.wait_until(&mut g, "carries its own branch's value", |d| first_f32(d) == 3.0);
        out_b.wait_until(&mut g, "carries its own branch's value", |d| first_f32(d) == 4.0);
    }

    // ---- node lifecycle ----------------------------------------------------

    /// A latch a node's `setup()` parks on. `Condvar` rather than a poll because the point is that
    /// the node's thread is genuinely INSIDE `setup` while the test looks at its stage — a sleep
    /// loop would leave a window where it is not.
    type SetupGate = std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>;

    fn setup_gate() -> SetupGate {
        std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()))
    }

    trait Openable {
        fn open(&self);
    }

    impl Openable for SetupGate {
        fn open(&self) {
            *self.0.lock().unwrap() = true;
            self.1.notify_all();
        }
    }

    /// Releases a [`SetupGate`] however the test ends, so a failed assertion never leaves a node
    /// thread parked inside `setup`.
    struct OnDrop(SetupGate);
    impl Drop for OnDrop {
        fn drop(&mut self) {
            self.0.open();
        }
    }

    struct SlowSetup(SetupGate);
    impl Node for SlowSetup {
        fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut open = self.0 .0.lock().unwrap();
            while !*open && Instant::now() < deadline {
                // Bounded, so a failing test leaves no thread parked here for the rest of the run.
                (open, _) = self.0 .1.wait_timeout(open, Duration::from_millis(50)).unwrap();
            }
            Ok(())
        }
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            out.set("out", Data::array_f32(vec![1], 7.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap());
            Ok(())
        }
    }
    static SLOW_SETUP: NodeManifest = NodeManifest {
        type_name: "_TestSlowSetup",
        category: "test",
        doc: "blocks inside setup()",
        inputs: &[],
        outputs: DROP_COUNTED_OUT,
        params: NO_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    /// A node that counts its own drops. The counter is what makes "the thread stopped" observable
    /// from the test's side: a halt that is never noticed leaves the instance alive on a thread
    /// nothing is watching, and no graph-side read can tell that apart from a clean teardown.
    struct DropCounted(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for DropCounted {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    impl Node for DropCounted {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            out.set("out", Data::array_f32(vec![1], 7.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap());
            Ok(())
        }
    }
    static DROP_COUNTED_OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
    static DROP_COUNTED: NodeManifest = NodeManifest {
        type_name: "_TestDropCounted",
        category: "test",
        doc: "counts its own drops",
        inputs: &[],
        outputs: DROP_COUNTED_OUT,
        params: NO_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    /// Register the drop-counting type and add one instance, with the counter it writes to.
    fn drop_counted_node(g: &mut Graph) -> (Uid, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let drops: std::sync::Arc<std::sync::atomic::AtomicUsize> = Default::default();
        let mine = drops.clone();
        g.register_dyn_type(&DROP_COUNTED, Box::new(move |_p| Box::new(DropCounted(mine.clone()))));
        let uid = g.add_node("_TestDropCounted", None).unwrap();
        (uid, drops)
    }

    /// Wait for `count` drops, or fail. The halt is fire-and-forget — a node inside a long
    /// `process()` notices at its next wake — so this is a bounded poll rather than a join.
    fn wait_drops(drops: &std::sync::atomic::AtomicUsize, count: usize, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if drops.load(std::sync::atomic::Ordering::SeqCst) == count {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("{what}: {} drops, expected {count}", drops.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn removing_a_node_stops_its_thread_and_drops_its_instance() {
        // Every node has a manager-side thread (§5), and removal is what stops it. Nothing joins
        // it — a node inside a long `process()` would hold the graph lock hostage — so the halt has
        // to be a flag the loop reads, and the instance's own `Drop` is the proof it did.
        let mut g = Graph::new();
        let (n, drops) = drop_counted_node(&mut g);
        let out = OutputProbe::open(&g, n, "out");
        out.expect_frame(&mut g, "the node to run at all");
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 0, "still alive on its thread");

        g.remove_node(n).unwrap();
        wait_drops(&drops, 1, "the removed instance was never dropped");
    }

    #[test]
    fn dropping_the_graph_stops_every_node_thread() {
        // The other end of the same flag: a `Graph` going out of scope — which every test does, and
        // which `clear`/`load_doc` do wholesale — must not leave threads publishing into shared
        // memory for the rest of the process. There is no channel to send a terminate on here,
        // which is why the halt is a flag.
        let drops: std::sync::Arc<std::sync::atomic::AtomicUsize> = Default::default();
        {
            let mut g = Graph::new();
            let mine = drops.clone();
            g.register_dyn_type(&DROP_COUNTED, Box::new(move |_p| Box::new(DropCounted(mine.clone()))));
            let a = g.add_node("_TestDropCounted", None).unwrap();
            g.add_node("_TestDropCounted", None).unwrap();
            OutputProbe::open(&g, a, "out").expect_frame(&mut g, "the nodes to be running");
            assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        }
        wait_drops(&drops, 2, "a dropped graph left threads running");
    }


    // ---- restart_node (in-place respawn) ----

    #[test]
    fn restart_rebuilds_the_instance_and_clears_the_error() {
        // A dyn type standing in for a Python node: its factory counts constructions and the
        // FIRST instance fails setup, so the restart's fresh instance is observably different.
        static BOOT: NodeManifest = NodeManifest {
            type_name: "_RestartBoot",
            category: "runtime",
            doc: "fails setup once, then succeeds",
            inputs: &[],
            outputs: RT_OUT,
            params: RT_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: rt_stub_factory,
        };
        struct Boot {
            fail: bool,
        }
        impl Node for Boot {
            fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
                if self.fail {
                    return Err("boot failed".into());
                }
                Ok(())
            }
            fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
                let d = Data::array_f32(vec![1], 7.0f32.to_le_bytes().to_vec(), Meta::empty())
                    .map_err(|e| e.to_string())?;
                out.set("out", d);
                Ok(())
            }
        }
        let builds = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let b = builds.clone();
        let mut g = Graph::new();
        g.register_dyn_type(
            &BOOT,
            Box::new(move |_p| {
                let n = b.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::new(Boot { fail: n == 0 })
            }),
        );

        let uid = g.add_node("_RestartBoot", None).unwrap();
        wait_for(&mut g, "the first instance's boot failure", |g| {
            g.last_error(uid) == Some("boot failed")
        });

        g.restart_node(uid).unwrap();

        assert_eq!(builds.load(std::sync::atomic::Ordering::SeqCst), 2, "a fresh instance was built");
        let out = OutputProbe::open(&g, uid, "out");
        assert_eq!(first_f32(&out.expect_frame(&mut g, "the new instance to run")), 7.0, "the new instance runs");
        wait_for(&mut g, "the restart to clear the recovered node's error", |g| g.last_error(uid).is_none());
    }

    #[test]
    fn restart_preserves_identity_position_viewers_and_scope() {
        let mut g = Graph::new();
        let uid = const_src(&mut g, 5.0);
        g.rename_node(uid, "my source").unwrap();
        g.set_node_pos(uid, [12.0, 34.0]).unwrap();
        g.set_node_viewers(uid, serde_json::json!({ "out": { "kind": "line" } })).unwrap();
        let scope = g.group_nodes(&[uid], [0.0, 0.0]).unwrap();

        g.restart_node(uid).unwrap();

        assert_eq!(g.name(uid), Some("my source"), "the display name survives (nd() refs it)");
        assert_eq!(g.pos(uid), Some([12.0, 34.0]));
        assert_eq!(g.viewers(uid), Some(&serde_json::json!({ "out": { "kind": "line" } })));
        assert_eq!(g.scope_of(uid), Some(scope), "a sub-patch member stays in its scope");
        // The param edit const_src made must reach the fresh instance, not the type default.
        let out = OutputProbe::open(&g, uid, "out");
        assert_eq!(first_f32(&out.expect_frame(&mut g, "the restarted node to run")), 5.0, "params carried over");
    }

    #[test]
    fn restart_keeps_every_wire_of_a_multi_input_in_connection_order() {
        // A reborn node has all-new service names (§3.1), so every wire into it has to be planned
        // again — a restart that does not re-plan leaves the slot silently dead while the editor
        // still draws three cables.
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        let out = OutputProbe::open(&g, col, "out");
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();
        collects(&mut g, &out, &[3.0, 1.0, 2.0, 3.0]);

        g.restart_node(col).unwrap();
        // The probe is on the OLD generation's service, which the reborn node never publishes to.
        let out = OutputProbe::open(&g, col, "out");
        collects(&mut g, &out, &[3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn restart_keeps_a_param_expression_binding_live() {
        // A restart replaces the instance, not the patch: the authored binding and its resolution
        // both survive. The reference is to ANOTHER node, so the assertion can see the resolution
        // and not merely the stored text.
        let mut g = eval_graph();
        let src = const_src(&mut g, 1.0);
        g.rename_node(src, "src").unwrap();
        let uid = const_src(&mut g, 1.0);
        g.set_expression(uid, "constant", "value", "nd('src')", true, false).unwrap();

        g.restart_node(uid).unwrap();

        assert_eq!(
            g.param_expression(uid, "constant", "value").map(|e| e.source),
            Some("nd('src')".to_string()),
            "the authored source is untouched by a restart"
        );
        assert_eq!(resolved(&g, uid, "constant", "value"), ["__v0=src.out#65"], "and still resolved");
    }

    #[test]
    fn restarting_a_node_reaps_its_predecessor() {
        // A restart replaces the instance, and the corpse's thread has to stop — its services are
        // already unreachable (the generation moved), so a thread left running would publish into
        // shared memory nobody reads for the rest of the process.
        let mut g = Graph::new();
        let (n, drops) = drop_counted_node(&mut g);
        OutputProbe::open(&g, n, "out").expect_frame(&mut g, "the first instance to run");

        g.restart_node(n).unwrap();
        wait_drops(&drops, 1, "the replaced instance was never dropped");
    }

    #[test]
    fn a_node_reports_its_bootstrap_stage_while_setup_is_still_running() {
        // The spinner exists for this, and it is no longer a subprocess-only window: `setup()` runs
        // on the node's own thread for every kind of node now, so any node whose init blocks —
        // opening a device, importing numpy — is observably `setup` before it is `ready`.
        let gate = setup_gate();
        let mut g = Graph::new();
        let mine = gate.clone();
        g.register_dyn_type(&SLOW_SETUP, Box::new(move |_p| Box::new(SlowSetup(mine.clone()))));
        let n = g.add_node("_TestSlowSetup", None).unwrap();
        let _release = OnDrop(gate.clone()); // so a failing assertion never parks the node's thread

        // `creating` until the node says otherwise, then `setup` while it is inside its own.
        wait_for(&mut g, "the node to report that it is initializing", |g| g.node_stage(n) == "setup");
        assert!(stays(&mut g, |g| g.node_stage(n) == "setup"), "and it stays there while setup blocks");

        gate.open();
        wait_for(&mut g, "the bootstrap to finish", |g| g.node_stage(n) == "ready");
    }

    #[test]
    fn an_unknown_node_is_not_ready() {
        let mut g = Graph::new();
        let n = g.add_node("_TestConst", None).unwrap();
        wait_for(&mut g, "the node to report ready", |g| g.node_stage(n) == "ready");
        assert_eq!(g.node_stage(Uid(9999)), "error", "an unknown node is not `ready`");
    }

    #[test]
    fn restarting_an_unknown_node_is_an_error() {
        let mut g = Graph::new();
        assert!(g.restart_node(Uid(999)).is_err());
    }

    // ---- refresh_param (the UI's re-enumerate button) ----

    /// A node whose refreshable `device` param re-enumerates a growing device list, so a test
    /// can tell a first refresh from a second.
    static PICKER: NodeManifest = NodeManifest {
        type_name: "_RefreshPicker",
        category: "runtime",
        doc: "a refreshable string param",
        inputs: &[],
        outputs: RT_OUT,
        params: PICKER_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };
    static PICKER_PARAMS: &[ParamDecl] = &[
        ParamDecl {
            group: "audio",
            name: "device",
            spec: ParamSpec::Str { default: "none", options: &["none"], refresh: true },
            expression: None,
            doc: None,
        },
        ParamDecl {
            group: "audio",
            name: "fixed",
            spec: ParamSpec::Str { default: "a", options: &["a", "b"], refresh: false },
            expression: None,
            doc: None,
        },
    ];
    #[derive(Default)]
    struct Picker {
        scans: usize,
    }
    impl Node for Picker {
        fn on_param_refreshed(&mut self, key: &goofi_node::ParamKey, _p: &Params<'_>) -> Option<Vec<String>> {
            if key.name != "device" {
                return None;
            }
            self.scans += 1;
            Some((0..self.scans).map(|i| format!("dev{i}")).collect())
        }
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Ok(())
        }
    }

    fn picker_graph() -> (Graph, Uid) {
        let mut g = Graph::new();
        g.register_dyn_type(&PICKER, Box::new(|_p| Box::<Picker>::default()));
        let uid = g.add_node("_RefreshPicker", None).unwrap();
        (g, uid)
    }

    fn options_of(g: &Graph, uid: Uid, group: &str, name: &str) -> Option<Vec<String>> {
        match g.params(uid).unwrap().get(group).unwrap().get(name).unwrap() {
            Param::Str { options, .. } => options.clone(),
            other => panic!("not a string param: {other:?}"),
        }
    }

    #[test]
    fn refresh_param_asks_the_node_and_stores_the_fresh_options() {
        let (mut g, uid) = picker_graph();
        assert_eq!(options_of(&g, uid, "audio", "device"), Some(vec!["none".to_string()]));

        // §8.5: the answer never rides the RPC — the hook runs on the node's own thread, so a
        // multi-second device scan cannot stall the caller. The options arrive as
        // `Status::RefreshOptions` and land in the record the inspector reads.
        assert_eq!(g.refresh_param(uid, "audio", "device").unwrap(), None);
        wait_for(&mut g, "the scanned options to reach the record", |g| {
            options_of(g, uid, "audio", "device") == Some(vec!["dev0".to_string()])
        });

        // A second click re-scans rather than replaying a cached list.
        g.refresh_param(uid, "audio", "device").unwrap();
        wait_for(&mut g, "the second scan to answer with its own list", |g| {
            options_of(g, uid, "audio", "device") == Some(vec!["dev0".into(), "dev1".into()])
        });
    }

    #[test]
    fn refresh_param_keeps_the_selected_value_and_the_refreshable_flag() {
        let (mut g, uid) = picker_graph();
        let declared = Param::Str { value: "none".into(), options: Some(vec!["none".into()]), refresh: true };
        g.update_param(uid, "audio", "device", declared).unwrap();

        g.refresh_param(uid, "audio", "device").unwrap();

        match g.params(uid).unwrap().get("audio").unwrap().get("device").unwrap() {
            Param::Str { value, refresh, .. } => {
                assert_eq!(value, "none", "a refresh re-enumerates options; it never changes the selection");
                assert!(*refresh, "the param stays refreshable");
            }
            other => panic!("not a string param: {other:?}"),
        }
    }

    #[test]
    fn refresh_param_rejects_a_param_that_is_not_refreshable() {
        let (mut g, uid) = picker_graph();

        assert!(g.refresh_param(uid, "audio", "fixed").is_err(), "the UI shows no button for it");
        assert!(g.refresh_param(uid, "audio", "nope").is_err(), "unknown param");
        assert!(g.refresh_param(Uid(999), "audio", "device").is_err(), "unknown node");
    }

    #[test]
    fn refresh_param_rejects_a_string_param_a_node_never_declared_refreshable() {
        let mut g = Graph::new();
        let uid = g.add_node("Oscillator", None).unwrap();
        // Oscillator's `waveform` is a string param with a FIXED list — the UI shows no button.
        assert!(g.refresh_param(uid, "oscillator", "waveform").is_err());
    }

    #[test]
    fn refresh_param_on_a_node_without_the_hook_succeeds_with_no_options() {
        // A node may declare a param refreshable and implement no hook (the default returns
        // None). That is a successful "nothing new to offer", NOT an error: the param keeps the
        // options it had, and the caller still reports completion so the UI's spinner clears.
        static NO_HOOK: NodeManifest = NodeManifest {
            type_name: "_RefreshNoHook",
            category: "runtime",
            doc: "declares a refreshable param but implements no hook",
            inputs: &[],
            outputs: RT_OUT,
            params: PICKER_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: rt_stub_factory,
        };
        #[derive(Default)]
        struct NoHook;
        impl Node for NoHook {
            fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
                Ok(())
            }
        }
        let mut g = Graph::new();
        g.register_dyn_type(&NO_HOOK, Box::new(|_p| Box::<NoHook>::default()));
        let uid = g.add_node("_RefreshNoHook", None).unwrap();

        assert_eq!(g.refresh_param(uid, "audio", "device"), Ok(None));
        assert_eq!(
            options_of(&g, uid, "audio", "device"),
            Some(vec!["none".to_string()]),
            "the declared options are left exactly as they were"
        );
    }

    // A runtime source built by a captured closure (not a bare fn pointer) —
    // stands in for a pyo3 node whose factory captures a Python class handle.
    struct RtSource {
        base: f32,
    }
    impl Node for RtSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let d = Data::array_f32(vec![1], self.base.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static RT_PARAMS: &[ParamDecl] = &[];
    fn rt_stub_factory() -> Box<dyn Node> {
        unreachable!("a runtime dyn type is constructed by its registered factory, not manifest.factory")
    }
    static RT_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    static RT_MANIFEST: NodeManifest = NodeManifest {
        type_name: "_RuntimeDyn",
        category: "runtime",
        doc: "runtime-registered node type",
        inputs: &[],
        outputs: RT_OUT,
        params: RT_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    // The SAME runtime type after its file was edited to declare a param — what a rescan
    // re-registers under a name it already holds.
    static GAINED_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "shape",
        name: "gain",
        spec: ParamSpec::Float { default: 3.0, min: 0.0, max: 10.0 },
        expression: None,
        doc: None,
    }];
    static RT_GAINED_MANIFEST: NodeManifest = NodeManifest {
        type_name: "_RuntimeDyn",
        category: "runtime",
        doc: "runtime-registered node type, edited to declare a param",
        inputs: &[],
        outputs: RT_OUT,
        params: GAINED_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    // The same type again, edited to WIDEN the param's bound rather than to add a param — the
    // spec changed while the name did not.
    static WIDENED_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "shape",
        name: "gain",
        spec: ParamSpec::Float { default: 3.0, min: 0.0, max: 100.0 },
        expression: None,
        doc: None,
    }];
    static RT_WIDENED_MANIFEST: NodeManifest = NodeManifest {
        type_name: "_RuntimeDyn",
        category: "runtime",
        doc: "runtime-registered node type, edited to widen a param's bound",
        inputs: &[],
        outputs: RT_OUT,
        params: WIDENED_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    // A runtime manifest whose name collides with a built-in catalog type.
    static COLLIDE_MANIFEST: NodeManifest = NodeManifest {
        type_name: "Oscillator",
        category: "runtime",
        doc: "collides with the built-in Oscillator",
        inputs: &[],
        outputs: RT_OUT,
        params: RT_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    #[test]
    fn an_unavailable_type_appears_in_the_catalog_but_cannot_be_added() {
        // A node file whose dependency is missing must not vanish: the palette shows it with the
        // reason, and trying to add it says why rather than "unknown node type".
        let mut g = Graph::new();
        assert!(g.register_unavailable("PsdScipy".into(), "scipy".into()));
        assert_eq!(g.unavailable_types().collect::<Vec<_>>(), [("PsdScipy", "scipy")]);

        let err = g.add_node("PsdScipy", None).unwrap_err();
        assert!(err.contains("unavailable"), "names the state: {err}");
        assert!(err.contains("scipy"), "names the missing dependency: {err}");
    }

    #[test]
    fn loading_a_patch_using_an_unavailable_type_names_the_missing_dependency() {
        // The same refusal, from the OTHER door: `load_doc` pre-validates types through
        // `known_type`, which deliberately excludes the unavailable registry (that is what makes an
        // unavailable type unaddable) — so its own message must still explain WHY, not send the user
        // hunting for a typo. Lose a dependency, restart, reopen a patch that used the node.
        let mut g = Graph::new();
        assert!(g.register_unavailable("PsdScipy".into(), "scipy".into()));
        let doc = "version: 7\nroot:\n  nodes:\n    n0:\n      type: PsdScipy\n  links: []\n";

        let err = g.load_doc(doc).unwrap_err();
        assert!(err.contains("unavailable"), "names the state: {err}");
        assert!(err.contains("scipy"), "names the missing dependency: {err}");
    }

    #[test]
    fn a_working_type_wins_over_an_unavailable_one_of_the_same_name() {
        let mut g = Graph::new();
        assert!(!g.register_unavailable("Oscillator".into(), "nope".into()));
        assert_eq!(g.unavailable_types().count(), 0);
        assert!(g.add_node("Oscillator", None).is_ok(), "the real type still instantiates");
    }

    // The same runtime type after its file was edited to RESHAPE it — the case `restart_node`
    // exists for since live patch-node editing shipped. v1 takes a triggering `alpha`; v2 renames
    // it to `beta`, drops the trigger, and adds an output. Same `type_name` throughout, because
    // that is what a rescan re-registers under.
    static RESHAPE_IN_V1: &[SlotDecl] = &[SlotDecl {
        name: "alpha",
        kind: SlotType::Array,
        trigger_process: true,
        multi: false,
        required: false,
    }];
    static RESHAPE_IN_V2: &[SlotDecl] = &[SlotDecl {
        name: "beta",
        kind: SlotType::Array,
        trigger_process: false,
        multi: false,
        required: false,
    }];
    static RESHAPE_OUT_V1: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
    static RESHAPE_OUT_V2: &[OutputDecl] = &[
        OutputDecl { name: "out", kind: SlotType::Array },
        OutputDecl { name: "extra", kind: SlotType::Array },
    ];
    static RESHAPE_V1: NodeManifest = NodeManifest {
        type_name: "_Reshaper",
        category: "runtime",
        doc: "a runtime type whose interface changes between rescans",
        inputs: RESHAPE_IN_V1,
        outputs: RESHAPE_OUT_V1,
        params: RT_PARAMS,
        isolation: Isolation::InProcess,
        producer: false,
        factory: rt_stub_factory,
    };
    // `producer` because V2's only slot is `trigger_process: false` — the node has an input and
    // still free-runs, which is the case `inputs.is_empty()` misses and `!any(trigger_process)`
    // catches. Inert today (`has_trigger_inputs` is already false), and what keeps it running
    // once the implicit free-run term goes.
    static RESHAPE_V2: NodeManifest = NodeManifest {
        type_name: "_Reshaper",
        category: "runtime",
        doc: "a runtime type whose interface changes between rescans",
        inputs: RESHAPE_IN_V2,
        outputs: RESHAPE_OUT_V2,
        params: RT_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    #[test]
    fn restart_node_adopts_the_new_manifest_after_a_rescan_reshapes_the_type() {
        // `restart_node` served crash recovery, where "same registered type ⇒ same interface" held.
        // Live patch-node editing broke that assumption without changing this code: rescan_nodes →
        // register_dyn_type(Replaced) → restart_changed → restart_node, with a type_name that is
        // stable while the INTERFACE is not. Carrying the old manifest through leaves the graph's
        // link validation, schema projection and /data target checks describing a node that is no
        // longer running.
        let mut g = Graph::new();
        g.register_dyn_type(&RESHAPE_V1, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        let uid = g.add_node("_Reshaper", None).unwrap();
        let src = g.add_node("Oscillator", None).unwrap();
        g.add_link(src, "out", uid, "alpha").unwrap();

        g.register_dyn_type(&RESHAPE_V2, Box::new(|_| Box::new(RtSource { base: 2.0 })));
        g.restart_node(uid).unwrap();

        let m = g.manifest(uid).expect("a restarted node still has a manifest");
        assert!(m.inputs.iter().any(|i| i.name == "beta"), "the graph reads the NEW manifest");
        assert!(!m.inputs.iter().any(|i| i.name == "alpha"), "and not the old one");
        assert!(m.outputs.iter().any(|o| o.name == "extra"), "a slot the rescan added is present");

        // The manifest-derived caches followed it, which is what makes the new shape usable.
        assert!(g.add_link(src, "out", uid, "beta").is_ok(), "the new input accepts a wire");
        assert!(g.add_link(src, "out", uid, "alpha").is_err(), "the retired input does not");
        // The wire into the retired slot cannot propagate and cannot be repaired from the palette,
        // so it goes with the slot rather than lingering as a cable the runtime ignores.
        assert!(
            !g.links_view().iter().any(|l| l.node_in == uid && l.slot_in == "alpha"),
            "the link into the retired slot was pruned"
        );
    }

    // A node that panics in its LIFECYCLE hooks — the shape a third-party native crate or a
    // freshly-edited `.py` can take. `process` panics were already contained (`_TestPanic`
    // above); these were not.
    struct PanickyHooks;
    impl Node for PanickyHooks {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Ok(())
        }
        fn on_param_changed(&mut self, _k: &ParamKey, v: &Param) -> NodeResult {
            if matches!(v, Param::Bool { value: true }) {
                panic!("hook exploded");
            }
            Ok(())
        }
    }
    static PANICKY_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "danger",
        name: "boom",
        spec: ParamSpec::Bool { default: false },
        expression: None,
        doc: None,
    }];
    static PANICKY_MANIFEST: NodeManifest = NodeManifest {
        type_name: "_Panicky",
        category: "runtime",
        doc: "panics in on_param_changed when danger.boom is set",
        inputs: &[],
        outputs: RT_OUT,
        params: PANICKY_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: rt_stub_factory,
    };

    #[test]
    fn a_panicking_lifecycle_hook_becomes_a_node_error_rather_than_killing_its_thread() {
        // A node is third-party code, and `guard_lifecycle` wraps every hook for it. What the guard
        // BUYS moved with the runtime: the hooks no longer run under the graph mutex, so a panic
        // can no longer poison it — it kills the node's wake loop instead, silently and forever.
        // So the load-bearing half is that the node is still THERE afterwards, answering the next
        // message; the error reaching the graph alone holds just as well for a dead thread.
        let mut g = Graph::new();
        g.register_dyn_type(&PANICKY_MANIFEST, Box::new(|_| Box::new(PanickyHooks)));
        let uid = g.add_node("_Panicky", None).unwrap();

        // The RPC succeeds — the hook runs on the node's thread, so its failure cannot ride this
        // reply (§8.4). It arrives as the node's own error instead.
        g.update_param(uid, "danger", "boom", Param::boolean(true)).unwrap();
        wait_for(&mut g, "the panic to reach the graph as this node's error", |g| {
            g.last_error(uid).is_some_and(|e| e.contains("panic"))
        });

        // Still running, and still listening. This is the load-bearing half: only a live wake loop
        // can apply the second edit, and only applying it clears the error the first one left. A
        // thread the panic killed would sit on that error for ever.
        g.update_param(uid, "danger", "boom", Param::boolean(false)).unwrap();
        wait_for(&mut g, "the node to answer the next edit", |g| g.last_error(uid).is_none());
    }

    #[test]
    fn a_panicking_setup_is_the_nodes_boot_error_not_a_lost_process() {
        // `seed_node` runs `on_param_changed` then `setup` at construction. That construction moved
        // ONTO the node's thread (§5) — `add_node` answers before it has happened — so an unguarded
        // panic there no longer unwinds through `Graph::add_node`; it takes the thread down before
        // the node has published anything, and the node draws "creating" for ever.
        let mut g = Graph::new();
        g.register_dyn_type(&PANICKY_MANIFEST, Box::new(|_| Box::new(PanickyHooks)));
        // A node born with the param already set panics during its seed replay.
        let mut params = g.default_params_of("_Panicky").unwrap();
        params.get_mut("danger").unwrap().insert("boom".into(), Param::boolean(true));
        let uid = g.add_node("_Panicky", Some(params)).unwrap();

        // On the INITIALIZATION channel: the replay is half of `seed_node`, so a panic in it leaves
        // the node uninitialized exactly as a failed `setup()` does — which is what `setup_error`
        // gates the retry on, and what `last_error` alone cannot tell apart.
        wait_for(&mut g, "the panicking seed to be reported as a boot failure", |g| {
            g.nodes[&uid].setup_error.as_deref().is_some_and(|e| e.contains("panic"))
        });
        assert_eq!(g.node_stage(uid), "error");
    }

    #[test]
    fn register_dyn_type_refuses_a_built_in_collision() {
        let mut g = Graph::new();
        // Collides with the built-in "Oscillator": refused, and add_node still
        // resolves the native node (the dyn factory would panic via rt_stub_make).
        let r = g.register_dyn_type(&COLLIDE_MANIFEST, Box::new(|_| unreachable!()));
        assert_eq!(r, Registration::Refused);
        assert!(g.dyn_type_manifests().is_empty());
        let osc = g.add_node("Oscillator", None).unwrap();
        assert_eq!(g.manifest(osc).unwrap().category, "inputs"); // the native one
    }

    #[test]
    fn re_registering_a_dyn_type_swaps_the_factory_for_later_instances() {
        // The rescan contract: edit a node file, re-register, and the NEXT instance runs the new
        // code. An instance built BEFORE the swap keeps the old factory's node — nothing here
        // restarts it (that is the auto-restart step, deliberately not this one).
        let mut g = Graph::new();
        let r = g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        assert_eq!(r, Registration::Added);
        let old = g.add_node("_RuntimeDyn", None).unwrap();
        let before = OutputProbe::open(&g, old, "out");

        let r = g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 2.0 })));
        assert_eq!(r, Registration::Replaced);
        assert_eq!(g.dyn_type_manifests().len(), 1, "a replace does not add a second entry");

        let new = g.add_node("_RuntimeDyn", None).unwrap();
        let after = OutputProbe::open(&g, new, "out");
        assert_eq!(first_f32(&after.expect_frame(&mut g, "the new instance to run")), 2.0, "new factory");
        assert_eq!(
            first_f32(&before.expect_frame(&mut g, "the old instance to still be running")),
            1.0,
            "live instance untouched",
        );
    }

    #[test]
    fn removing_a_dyn_type_takes_it_out_of_the_catalog_and_out_of_resolution() {
        // The other half of the rescan: the file vanished, so the type must stop being addable —
        // while the instance that is already running stays running.
        let mut g = Graph::new();
        g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        let live = g.add_node("_RuntimeDyn", None).unwrap();
        let out = OutputProbe::open(&g, live, "out");

        assert!(g.remove_dyn_type("_RuntimeDyn"));
        assert!(g.dyn_type_manifests().is_empty(), "gone from the palette");
        // `known_type` is private, and `add_node` is its door: the refusal must read as the
        // vanished type it is, not as a dependency-missing "unavailable" one.
        assert_eq!(g.add_node("_RuntimeDyn", None).unwrap_err(), "unknown node type `_RuntimeDyn`");
        assert!(!g.remove_dyn_type("_RuntimeDyn"), "nothing left to remove");

        assert_eq!(first_f32(&out.expect_frame(&mut g, "the live instance to keep running")), 1.0, "the instance still runs");
    }

    /// The two registries are one answer to "what is on disk under this name", so the LATEST scan
    /// of a name wins in both directions. Without this a rescan that fixes a broken node's
    /// dependency leaves the greyed row standing beside the working type — two palette rows for
    /// one name — because `unavailable` had no removal at all.
    #[test]
    fn the_latest_scan_of_a_name_wins_across_both_registries() {
        let mut g = Graph::new();
        // Deps installed: the file that could not load now loads.
        assert!(g.register_unavailable("_RuntimeDyn".into(), "numpy".into()));
        assert_eq!(
            g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 }))),
            Registration::Added
        );
        assert_eq!(g.unavailable_types().count(), 0, "no stale greyed row beside the working type");
        assert_eq!(g.dyn_type_manifests().len(), 1);

        // …and back: the file is edited into something that no longer loads.
        assert!(g.register_unavailable("_RuntimeDyn".into(), "SyntaxError".into()));
        assert!(g.dyn_type_manifests().is_empty(), "a type that stopped loading is not addable");
        assert_eq!(g.unavailable_types().collect::<Vec<_>>(), [("_RuntimeDyn", "SyntaxError")]);

        // The file vanishes: ONE removal door, because a rescan knows only that it is gone —
        // not which of the two registries the last scan put it in.
        assert!(g.remove_dyn_type("_RuntimeDyn"));
        assert_eq!(g.unavailable_types().count(), 0);
    }

    /// Auto-restart's sharp edge: the edit that prompts a rescan is often "I added a param". The
    /// restart must therefore rebuild the instance against the type's CURRENT declarations, keeping
    /// the values the user had set — exactly what the `.gfi` load does — rather than replaying a
    /// param map captured before the file changed, which would leave the new param missing from the
    /// instance while the palette advertises it.
    #[test]
    fn a_restart_rebuilds_against_the_types_current_params() {
        let mut g = Graph::new();
        g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        let uid = g.add_node("_RuntimeDyn", None).unwrap();
        g.update_param(uid, "common", "max_frequency", Param::float(11.0, 0.0, 100.0)).unwrap();

        // The file gained a param; the rescan re-registered the type and restarts the instance.
        g.register_dyn_type(&RT_GAINED_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        g.restart_node(uid).unwrap();

        let p = g.params(uid).unwrap();
        assert_eq!(
            p.get("shape").and_then(|g| g.get("gain")).and_then(Param::as_f64),
            Some(3.0),
            "the param the edited file declares is on the restarted instance"
        );
        assert_eq!(
            p.get("common").unwrap().get("max_frequency").and_then(Param::as_f64),
            Some(11.0),
            "a value the user set survives the restart"
        );

        // The next edit changed the param's SPEC, not the set of names. Carrying the held `Param`
        // over wholesale would revert the bound (and, for a `Str`, the `refresh` flag the palette
        // now advertises) — so the fold must take the value only, exactly as the `.gfi` load does.
        g.update_param(uid, "shape", "gain", Param::float(5.0, 0.0, 10.0)).unwrap();
        g.register_dyn_type(&RT_WIDENED_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        g.restart_node(uid).unwrap();

        let gain = g.params(uid).unwrap().get("shape").unwrap().get("gain").cloned();
        assert!(
            matches!(gain, Some(Param::Float { value, vmax, .. }) if value == 5.0 && vmax == 100.0),
            "the restart rebuilt against the type's current spec, carrying only the value: {gain:?}"
        );
    }

    #[test]
    fn hosts_a_runtime_registered_dyn_type() {
        let mut g = Graph::new();
        // Register a node TYPE that is not in the compile-time inventory. The
        // factory captures state (base = 42.0), which a fn pointer could not.
        let base = 42.0f32;
        g.register_dyn_type(
            &RT_MANIFEST,
            Box::new(move |_params| Box::new(RtSource { base })),
        );
        // add_node resolves it transparently, like any catalog node.
        let uid = g.add_node("_RuntimeDyn", None).unwrap();
        assert_eq!(g.type_name(uid), Some("_RuntimeDyn"));
        assert_eq!(g.manifest(uid).unwrap().category, "runtime");
        let out = OutputProbe::open(&g, uid, "out");
        assert_eq!(first_f32(&out.expect_frame(&mut g, "the dyn node to run")), 42.0);
    }

    #[test]
    fn dyn_type_manifests_enumerates_registered_runtime_types() {
        let mut g = Graph::new();
        assert!(g.dyn_type_manifests().is_empty());
        g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        let ms = g.dyn_type_manifests();
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].type_name, "_RuntimeDyn");
        assert_eq!(ms[0].category, "runtime");
    }

    #[test]
    fn dyn_type_survives_gfi_roundtrip() {
        // A .gfi referencing a runtime type must load into a graph that has the
        // type registered (validation consults both inventory and dyn types).
        let mut g = Graph::new();
        g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        g.add_node("_RuntimeDyn", None).unwrap();
        let yaml = g.serialize();

        let mut g2 = Graph::new();
        g2.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_count(), 1);

        // Loading a .gfi with an *unregistered* runtime type is rejected up front.
        let mut g3 = Graph::new();
        assert!(g3.load_doc(&yaml).is_err());
        assert_eq!(g3.node_count(), 0);
    }

    #[test]
    fn a_diamond_converges_on_both_branches() {
        // src -> echoA, src -> echoB, {echoA,echoB} -> adder. The adder reads BOTH branches, so a
        // sum of 10 means each one carried the source's 5 — one branch feeding both of the adder's
        // inputs, or a branch stuck empty, reads as a different number.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(5.0, -1e9, 1e9)).unwrap();
        let ea = g.add_node("_TestEcho", None).unwrap();
        let eb = g.add_node("_TestEcho", None).unwrap();
        let add = g.add_node("_TestAdder", None).unwrap();
        let out = OutputProbe::open(&g, add, "out");
        g.add_link(src, "out", ea, "in").unwrap();
        g.add_link(src, "out", eb, "in").unwrap();
        g.add_link(ea, "out", add, "a").unwrap();
        g.add_link(eb, "out", add, "b").unwrap();

        wait_for(&mut g, "the diamond to attach", |_| out.latest().is_some());
        out.wait_until(&mut g, "sums both branches", |d| first_f32(d) == 10.0);
    }

    #[test]
    fn a_cycle_is_tolerated_rather_than_policed() {
        // §4: `add_link` has no cycle check and gains none — latest-wins delivery makes a cycle
        // correct by construction. A pure 2-cycle of triggered nodes has nothing to seed it, so it
        // settles at rest: the property is that nothing hangs, spins or refuses, and that an
        // unseeded cycle produces nothing rather than a stream of empty frames.
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let out_a = OutputProbe::open(&g, a, "out");
        let out_b = OutputProbe::open(&g, b, "out");
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", a, "in").unwrap();
        wait_for(&mut g, "both wires to attach", |g| g.links_view().len() == 2);
        assert!(out_a.silent(&mut g) && out_b.silent(&mut g), "an unseeded cycle produces nothing");
    }

    #[test]
    fn sustained_load_reference_stress_shape_stays_stable() {
        // The reference stress-patch shape: one Oscillator fanning out to 8 Buffers, each on its
        // own thread. Let it run and assert every consumer keeps producing with a clean error
        // channel — sustained stability, no drift into a faulted state.
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let mut buffers = Vec::new();
        for _ in 0..8 {
            let b = g.add_node("Buffer", None).unwrap();
            g.add_link(osc, "out", b, "data").unwrap();
            buffers.push((b, OutputProbe::open(&g, b, "out")));
        }

        // Every buffer has to reach a SECOND frame, not merely a first: a node that emitted once
        // and then faulted, or one whose wire went away under it, passes a first-frame check.
        for (b, out) in &buffers {
            wait_for(&mut g, "each buffer to start producing", |_| out.latest().is_some());
            let first = out.expect_frame(&mut g, "a buffer frame").meta().index().unwrap_or(0);
            out.wait_until(&mut g, "keeps producing", |d| d.meta().index().unwrap_or(0) > first);
            assert!(g.last_error(*b).is_none(), "buffer faulted: {:?}", g.last_error(*b));
        }
        assert!(g.last_error(osc).is_none(), "oscillator faulted: {:?}", g.last_error(osc));
    }

    #[test]
    fn generator_stamps_fresh_incrementing_index() {
        // A source (no index-bearing input) gets a fresh per-output counter that advances once per
        // emit. The oracle is CONSECUTIVE indices, not a final number: a free-running node emits
        // whatever its thread manages in the window, and what the counter promises is that no emit
        // is skipped and none is repeated.
        //
        // Born capped, because CONSECUTIVE is exactly what a one-deep latest-wins cell cannot show
        // of an uncapped 30 kHz producer: every look lands thousands of emits on, and waiting for
        // `first + 1` waits for an index that went past before the poll returned.
        let mut g = Graph::new();
        let src = capped(&mut g, "_TestConst", 20.0);
        let out = OutputProbe::open(&g, src, "out");
        let first = out.expect_frame(&mut g, "the source to emit").meta().index().expect("stamped");
        for step in 1..4 {
            out.wait_until(&mut g, "advances one per emit", |d| d.meta().index() == Some(first + step));
        }
    }

    // A deterministic stand-in for the pyo3 evaluator, so the engine's binding lifecycle +
    // resolution are testable without a Python interpreter. It is handed the REWRITTEN source
    // (§5.3), so what it recognizes is a bare variable (`__v0` — read from the locals the graph
    // resolved), a bare number (a constant), and `ERR` (a compile failure). It resolves no names,
    // exactly as the real evaluator no longer does.
    #[derive(Default)]
    struct MockEval {
        exprs: std::sync::Mutex<HashMap<u64, MockExpr>>,
        next: std::sync::atomic::AtomicU64,
        /// Every `release` CALL, in order — not the surviving handles. `ExprEvaluator` is a public
        /// trait an implementation may reasonably treat as a refcount, so releasing one handle twice
        /// is a defect a "how many are left?" oracle cannot see: the second remove is a no-op.
        releases: std::sync::Mutex<Vec<u64>>,
    }
    #[derive(Clone)]
    enum MockExpr {
        Var(String),
        Const(f64),
    }
    impl goofi_node::ExprEvaluator for MockEval {
        fn compile(&self, source: &str) -> Result<goofi_node::Compiled, goofi_node::ExprError> {
            if source == "ERR" {
                return Err("mock compile error".into());
            }
            let expr = if source.starts_with("__v") {
                MockExpr::Var(source.to_string())
            } else {
                let v: f64 = source.parse().map_err(|_| goofi_node::ExprError("mock: not a number".into()))?;
                MockExpr::Const(v)
            };
            let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            self.exprs.lock().unwrap().insert(id, expr);
            Ok(goofi_node::Compiled { id })
        }
        fn eval(&self, id: u64, ctx: &goofi_node::EvalCtx<'_>) -> Result<Param, goofi_node::ExprError> {
            let expr = self.exprs.lock().unwrap().get(&id).cloned().ok_or_else(|| goofi_node::ExprError("mock: no such id".into()))?;
            let v: f64 = match expr {
                MockExpr::Const(c) => c,
                MockExpr::Var(name) => match ctx.locals.get(&name) {
                    Some(Some(goofi_node::Local::Frame(d))) => first_f32(d) as f64,
                    Some(Some(goofi_node::Local::Value(p))) => p.as_f64().unwrap_or(f64::NAN),
                    _ => return Err(goofi_node::ExprError(format!("mock: {name} missing"))),
                },
            };
            Ok(match ctx.target {
                Param::Int { vmin, vmax, .. } => Param::Int { value: v.round() as i64, vmin: *vmin, vmax: *vmax },
                _ => Param::Float { value: v, vmin: 0.0, vmax: 0.0 },
            })
        }
        fn release(&self, id: u64) {
            self.releases.lock().unwrap().push(id);
            self.exprs.lock().unwrap().remove(&id);
        }
    }

    fn eval_graph() -> Graph {
        let mut g = Graph::new();
        g.set_evaluator(Arc::new(MockEval::default()));
        g
    }

    #[test]
    fn an_nd_reference_resolves_to_the_producers_output_stream() {
        // §5.3, the graph's whole job with a reference: `nd('src')` becomes a variable naming a
        // PRODUCER OUTPUT, with a doorbell id out of the expression budget. The node resolves
        // nothing — it is handed the service and subscribes.
        //
        // (The tick used to evaluate this in-place, lifting the reference into the DAG so `host`
        // read `src`'s value in the same tick. That path is gone: §2.1 evaluates in the node, in
        // the same breath as the run that reads it.)
        let mut g = eval_graph();
        let src = g.add_node("_TestConst", None).unwrap();
        g.rename_node(src, "src").unwrap();
        let host = g.add_node("_TestConst", None).unwrap();
        g.set_expression(host, "constant", "value", "nd('src')", true, false).unwrap();
        assert_eq!(resolved(&g, host, "constant", "value"), ["__v0=src.out#65"]);
        assert!(g.param_expression(host, "constant", "value").unwrap().error.is_none());

        // A reference nothing answers is an error the moment it is authored, and it names what it
        // could not find rather than failing silently at some later run.
        g.set_expression(host, "constant", "value", "nd('ghost')", true, false).unwrap();
        assert_eq!(resolved(&g, host, "constant", "value"), ["__v0!no node named `ghost`"]);
    }

    /// The doorbell entry a producer would ring `uid` on, for `event`.
    fn doorbell_of(g: &Graph, uid: Uid, event: runtime::EventId) -> (runtime::ServiceName, runtime::EventId) {
        (g.door_of(uid), event)
    }

    #[test]
    fn setting_a_literal_on_a_bound_param_unbinds_it_and_unlinks_the_producer() {
        // §5.3: expression references ARE links. A producer's `OutSlot` target set is the union of
        // its wire consumers and its expression subscribers, so unbinding must drop this node from
        // it — otherwise the producer keeps ringing a doorbell nobody reads.
        //
        // In-module because `out_targets` is the graph's own, and widening it for a test would be
        // publishing an internal to prove something about it.
        let mut g = eval_graph();
        let lfo = g.add_node("_TestConst", None).unwrap();
        g.rename_node(lfo, "lfo").unwrap();
        let osc = g.add_node("_TestConst", None).unwrap();
        assert!(g.out_targets(lfo, "out").is_empty(), "nothing reads it yet");

        g.set_expression(osc, "constant", "value", "nd('lfo')", true, false).unwrap();
        assert_eq!(g.out_targets(lfo, "out"), [doorbell_of(&g, osc, 65)], "the reader joined the set");

        g.update_param(osc, "constant", "value", Param::float(5.0, -1e9, 1e9)).unwrap();
        assert!(g.out_targets(lfo, "out").is_empty(), "unbind unlinked it");
        assert!(g.param_expression(osc, "constant", "value").is_none(), "and the binding is gone");
    }

    #[test]
    fn a_handle_is_released_exactly_once_however_the_binding_ends() {
        // One owner for the release. `set_expression` released the previous handle at the top and
        // then handed the empty-source path to `unbind`, which released it again — invisible to a
        // "how many handles survive?" check, because the second remove is a no-op.
        let mock = Arc::new(MockEval::default());
        let mut g = Graph::new();
        g.set_evaluator(mock.clone());
        let n = g.add_node("_TestConst", None).unwrap();
        let released = || mock.releases.lock().unwrap().clone();

        g.set_expression(n, "constant", "value", "5", true, false).unwrap();
        let id = g.nodes[&n].bindings[&ParamKey::new("constant", "value")].id.expect("compiled");
        assert!(released().is_empty(), "nothing released while it stands");

        g.set_expression(n, "constant", "value", "", false, false).unwrap();
        assert_eq!(released(), [id], "the empty source released it, once");

        // The two other doors onto the same handle, each also exactly once.
        g.set_expression(n, "constant", "value", "6", true, false).unwrap();
        let rebound = g.nodes[&n].bindings[&ParamKey::new("constant", "value")].id.unwrap();
        g.set_expression(n, "constant", "value", "7", true, false).unwrap();
        assert_eq!(released(), [id, rebound], "a REBIND releases what it replaces, once");
        let literal = g.nodes[&n].bindings[&ParamKey::new("constant", "value")].id.unwrap();
        g.update_param(n, "constant", "value", Param::float(1.0, -1e9, 1e9)).unwrap();
        assert_eq!(released(), [id, rebound, literal], "and so does a literal write");
    }

    #[test]
    fn a_literal_unbinds_only_a_binding_that_is_actually_driving() {
        // §3.4 unbinds a param the node is DRIVEN on. A disabled binding drives nothing — it is
        // source the fx toggle is holding — and EVERY node in the patch carries one on
        // `common.max_frequency` (`globals.default_ufreq`, waiting to be switched on). Unbinding
        // those made typing a number into a consumer's rate cap permanently delete the patch-rate
        // expression, and `serialize` then persisted the loss.
        //
        // Both halves in one test: the enabled case is the rule and the disabled case is its edge,
        // and pinning only one is how this diff's other guards went hollow.
        let mut g = eval_graph();
        let consumer = g.add_node("_TestSink", None).unwrap();
        let carried = g.param_expression(consumer, "common", "max_frequency").expect("carried");
        assert!(!carried.enabled, "a consumer carries the patch rate, disabled");

        g.update_param(consumer, "common", "max_frequency", Param::float(7.0, 0.0, 100.0)).unwrap();
        let kept = g.param_expression(consumer, "common", "max_frequency").expect("still there");
        assert_eq!(kept.source, "globals.default_ufreq", "a disabled binding is not driving it");
        assert!(!kept.enabled);
        assert!(g.serialize().contains("globals.default_ufreq"), "and the `.gfi` still carries it");

        // The same write over an ENABLED binding still unbinds — that is the rule this is the edge of.
        let producer = g.add_node("_TestGated", None).unwrap();
        assert!(g.param_expression(producer, "common", "max_frequency").unwrap().enabled);
        g.update_param(producer, "common", "max_frequency", Param::float(7.0, 0.0, 100.0)).unwrap();
        assert!(g.param_expression(producer, "common", "max_frequency").is_none(), "unbound");
    }

    #[test]
    fn a_producers_targets_are_the_union_of_its_wires_and_its_readers() {
        // One set, because a producer cannot tell a wired consumer from an `nd()` reader — and the
        // two carry DIFFERENT event ids out of §3.2's two budgets, so a union that dropped either
        // half, or that reused one id for both, is visible here.
        let mut g = eval_graph();
        let src = g.add_node("_TestGated", None).unwrap();
        g.rename_node(src, "src").unwrap();
        let wired = g.add_node("_TestSink", None).unwrap();
        let reader = g.add_node("_TestConst", None).unwrap();

        g.add_link(src, "out", wired, "in").unwrap();
        g.set_expression(reader, "constant", "value", "nd('src')", true, false).unwrap();
        assert_eq!(
            g.out_targets(src, "out"),
            [doorbell_of(&g, wired, 1), doorbell_of(&g, reader, 65)],
            "an input slot's id comes from the manifest, an expression's from the 65.. budget",
        );

        // A DISABLED binding subscribes to nothing: the fx toggle is off, so the literal stands and
        // the producer has no one to ring for it.
        g.set_expression(reader, "constant", "value", "nd('src')", false, false).unwrap();
        assert_eq!(g.out_targets(src, "out"), [doorbell_of(&g, wired, 1)]);
    }

    #[test]
    fn a_reference_the_graph_cannot_resolve_is_refused_rather_than_guessed() {
        // The graph is what knows how many outputs a node has, so this is where a bare `nd()` on a
        // multi-output producer and a slot that does not exist are caught — the pyo3 proxy used to
        // raise for the first at eval time, and the second was never anyone's.
        //
        // The refusal is load-bearing for the REWRITE, not just tidy: `expr_rewrite` reads a
        // trailing non-call attribute as a slot on purpose, and the whole defence of that decision
        // is that an attribute which is not a slot comes back named. Resolving to `outputs[0]`
        // instead would make `nd('a').T` silently mean `nd('a').fast`.
        let mut g = eval_graph();
        let two = g.add_node("_TestTwoRate", None).unwrap();
        g.rename_node(two, "two").unwrap();
        let host = g.add_node("_TestConst", None).unwrap();

        g.set_expression(host, "constant", "value", "nd('two')", true, false).unwrap();
        let bare = g.param_expression(host, "constant", "value").unwrap().error;
        assert_eq!(
            bare.as_deref(),
            Some("nd('two') is ambiguous: it has multiple outputs; use nd('two').slot"),
            "a bare reference to a multi-output node names the problem, it does not pick one",
        );
        assert!(resolved(&g, host, "constant", "value")[0].contains("!"), "and resolves to nothing");

        g.set_expression(host, "constant", "value", "nd('two').nope", true, false).unwrap();
        assert_eq!(
            g.param_expression(host, "constant", "value").unwrap().error.as_deref(),
            Some("node `two` has no output `nope`"),
        );

        // The control: a slot that DOES exist resolves, and to the one it names — otherwise
        // "everything is refused" would pass both assertions above.
        g.set_expression(host, "constant", "value", "nd('two').slow", true, false).unwrap();
        assert_eq!(resolved(&g, host, "constant", "value"), ["__v0=two.slow#65"]);
    }

    #[test]
    fn a_reference_follows_its_producer_being_removed_and_restored() {
        // §5.3's "added" and "removed": a NAME started or stopped meaning what it did, and every
        // binding written against it has to be re-resolved. Both halves in one test, because they
        // are one rule and pinning one leaves the other free — the add case is undo-of-delete,
        // which is the whole reason a restore keeps the display name.
        let mut g = eval_graph();
        let src = g.add_node("_TestConst", None).unwrap();
        g.rename_node(src, "src").unwrap();
        let host = g.add_node("_TestConst", None).unwrap();
        g.set_expression(host, "constant", "value", "nd('src')", true, false).unwrap();
        assert_eq!(resolved(&g, host, "constant", "value"), ["__v0=src.out#65"]);

        g.remove_node(src).unwrap();
        assert_eq!(
            resolved(&g, host, "constant", "value"),
            ["__v0!no node named `src`"],
            "a variable still naming a dead producer's service waits on it forever",
        );
        assert_eq!(g.last_error(host), Some("no node named `src`"));

        g.add_node_at("_TestConst", None, src, "src").unwrap();
        assert_eq!(resolved(&g, host, "constant", "value"), ["__v0=src.out#65"], "undo-of-delete");
        assert!(g.last_error(host).is_none(), "and the error cleared with it");
    }

    #[test]
    fn each_reference_on_a_node_gets_its_own_doorbell_id() {
        // §3.2 budgets `65..=128` for expression channels, and a producer rings ONE id per
        // variable — two references sharing an id would be one wake the node cannot attribute.
        // Driven across two bindings, because the ids are allocated per binding against what the
        // node's OTHER bindings already hold.
        let mut g = eval_graph();
        let a = g.add_node("_TestConst", None).unwrap();
        g.rename_node(a, "a").unwrap();
        let b = g.add_node("_TestConst", None).unwrap();
        g.rename_node(b, "b").unwrap();
        let host = g.add_node("_TestConst", None).unwrap();
        g.set_expression(host, "constant", "value", "nd('a')", true, false).unwrap();
        g.set_expression(host, "constant", "length", "nd('b')", true, false).unwrap();
        assert_eq!(resolved(&g, host, "constant", "value"), ["__v0=a.out#65"]);
        assert_eq!(resolved(&g, host, "constant", "length"), ["__v0=b.out#66"], "the next free id");

        // …and re-binding one frees its own id rather than stepping past it forever.
        g.set_expression(host, "constant", "value", "nd('b')", true, false).unwrap();
        assert_eq!(resolved(&g, host, "constant", "value"), ["__v0=b.out#65"], "its own id is free");
    }

    #[test]
    fn renaming_a_referenced_node_rewrites_nd_expressions() {
        // host.value = nd('src'); renaming `src` -> `signal` must follow the reference so
        // the expression still resolves (Python: manager.rename_node rewrites nd('old')).
        let mut g = eval_graph();
        let src = g.add_node("_TestConst", None).unwrap();
        g.rename_node(src, "src").unwrap();
        g.update_param(src, "constant", "value", Param::float(3.0, -1e9, 1e9)).unwrap();
        let host = g.add_node("_TestConst", None).unwrap();
        g.set_expression(host, "constant", "value", "nd('src')", true, false).unwrap();

        let touched = g.rename_node(src, "signal").unwrap();
        assert_eq!(touched, vec![host], "the referrer is reported for rebroadcast");
        assert_eq!(
            g.param_expression(host, "constant", "value").unwrap().source,
            "nd('signal')",
            "the reference followed the rename"
        );
        // And it still resolves, through the new name, to the same producer.
        assert_eq!(resolved(&g, host, "constant", "value"), ["__v0=signal.out#65"]);
    }

    #[test]
    fn expression_values_report_live_evaluated_params() {
        // The live preview seam: the evaluated value of each ENABLED binding is reported (a plain
        // literal param is not), and a disabled binding drops out. The value now arrives as the
        // node's own `Status::ParamValues` (§6.2) rather than being computed here, so the record
        // it feeds must be the one `expression_values` reads — reading `params` instead would
        // report the LITERAL and look right for a binding that never evaluated.
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "7", true, false).unwrap();
        assert!(g.expression_values(n).is_empty(), "nothing is live until the node reports one");
        g.apply_status(
            n,
            runtime::Status::ParamValues {
                evaluated: vec![(ParamKey::new("constant", "value"), Param::float(7.0, -1e9, 1e9))],
            },
        );
        let vals = g.expression_values(n);
        assert_eq!(vals.len(), 1, "only the expression-bound param is reported");
        let (group, name, p) = vals[0];
        assert_eq!((group, name), ("constant", "value"));
        assert!(matches!(p, Param::Float { value, .. } if (value - 7.0).abs() < 1e-9), "carries the evaluated value");
        // Disabling the binding removes it from the live set (its value is now the literal).
        g.set_expression(n, "constant", "value", "7", false, false).unwrap();
        assert!(g.expression_values(n).is_empty(), "disabled binding is not a live value");

        // And a RESTART clears them: they are the corpse's report, and a fresh instance has
        // evaluated nothing yet. Left standing, the inspector preview shows a dead node's numbers
        // until the new one happens to report — indistinguishable from a live value.
        g.set_expression(n, "constant", "value", "7", true, false).unwrap();
        g.apply_status(
            n,
            runtime::Status::ParamValues {
                evaluated: vec![(ParamKey::new("constant", "value"), Param::float(7.0, -1e9, 1e9))],
            },
        );
        assert_eq!(g.expression_values(n).len(), 1);
        g.restart_node(n).unwrap();
        assert!(g.expression_values(n).is_empty(), "the new instance has reported nothing");
    }

    #[test]
    fn a_failed_rename_leaves_nd_expressions_untouched() {
        // The rewrite is gated on a SUCCESSFUL rename: renaming onto a taken name must
        // fail and touch no expression.
        let mut g = eval_graph();
        let a = g.add_node("_TestConst", None).unwrap();
        g.rename_node(a, "a").unwrap();
        let b = g.add_node("_TestConst", None).unwrap();
        g.rename_node(b, "b").unwrap();
        let host = g.add_node("_TestConst", None).unwrap();
        g.set_expression(host, "constant", "value", "nd('a')", true, false).unwrap();
        assert!(g.rename_node(a, "b").is_err(), "rename onto a taken name fails");
        assert_eq!(
            g.param_expression(host, "constant", "value").unwrap().source,
            "nd('a')",
            "a failed rename rewrites nothing"
        );
    }

    #[test]
    fn missing_ref_errors_and_keeps_last_value() {
        let mut g = eval_graph();
        let host = g.add_node("_TestConst", None).unwrap();
        let out = OutputProbe::open(&g, host, "out");
        g.set_expression(host, "constant", "value", "nd('ghost')", true, false).unwrap();
        wait_for(&mut g, "the unresolved reference to reach the node error channel", |g| {
            g.last_error(host).is_some()
        });
        let info = g.param_expression(host, "constant", "value").expect("binding present");
        assert!(info.error.is_some(), "field error indicator set");
        // The literal value (default 0) is kept.
        out.wait_until(&mut g, "falls back to the literal", |d| first_f32(d) == 0.0);
    }

    #[test]
    fn compile_error_is_stored_not_rejected() {
        let mut g = eval_graph();
        let host = g.add_node("_TestConst", None).unwrap();
        g.set_expression(host, "constant", "value", "ERR", true, false).unwrap(); // RPC ok
        let info = g.param_expression(host, "constant", "value").expect("binding stored");
        assert!(info.error.is_some(), "compile error stored as the field indicator");
    }

    #[test]
    fn disabling_preserves_the_source_and_only_empty_unbinds() {
        // fx toggle-off (non-empty source, enabled=false) must PRESERVE the authored
        // source, disabled — not destroy it. Only an EMPTY source truly unbinds.
        let mut g = eval_graph();
        let host = g.add_node("_TestConst", None).unwrap();
        g.set_expression(host, "constant", "value", "5", true, false).unwrap();
        g.set_expression(host, "constant", "value", "5", false, false).unwrap();
        let info = g.param_expression(host, "constant", "value").expect("binding preserved when disabled");
        assert!(!info.enabled, "disabled");
        assert_eq!(info.source, "5", "authored source survives the toggle-off");
        // A disabled binding is not evaluated (the param keeps its literal default 0).
        let out = OutputProbe::open(&g, host, "out");
        assert!(
            stays(&mut g, |_| out.latest().is_some_and(|d| first_f32(&d) == 0.0)),
            "disabled binding is inert",
        );
        // Empty source is the true unbind.
        g.set_expression(host, "constant", "value", "", false, false).unwrap();
        assert!(g.param_expression(host, "constant", "value").is_none(), "empty source unbinds");
    }

    #[test]
    fn set_expression_rejects_an_unknown_param() {
        // A non-empty source on a bogus (group, name) must be refused — no dangling,
        // unclearable, phantom-edge-injecting binding.
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        assert!(g.set_expression(n, "constant", "nope", "5", true, false).is_err());
        assert!(g.param_expression(n, "constant", "nope").is_none(), "no dangling binding stored");
    }

    #[test]
    fn expression_survives_a_gfi_roundtrip() {
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "5", true, true).unwrap();
        let yaml = g.serialize();
        let mut g2 = eval_graph();
        g2.load_doc(&yaml).unwrap();
        let uid2 = g2.node_uids()[0];
        let info = g2.param_expression(uid2, "constant", "value").expect("binding restored from .gfi");
        assert_eq!(info.source, "5");
        assert!(info.enabled);
        assert!(info.triggers_process, "triggers_process round-trips");
    }

    #[test]
    fn serialize_emits_v7_root_nested_and_roundtrips() {
        // .gfi v7: version 7, a `pillar_default`, nodes/links nested under `root`, plus a `globals`
        // block (and a flat `scopes` overlay). A signal-only patch round-trips.
        let mut g = Graph::new();
        let n = g.add_node("_TestConst", None).unwrap();
        g.update_param(n, "constant", "value", Param::float(7.0, -1.0e9, 1.0e9)).unwrap();
        let yaml = g.serialize();
        assert!(yaml.contains("version: 7"), "emits v7; got:\n{yaml}");
        assert!(yaml.contains("pillar_default: signal"), "carries the default pillar");
        assert!(yaml.contains("root:"), "nodes/links nested under root");
        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_uids().len(), 1, "node round-trips");
        let uid2 = g2.node_uids()[0];
        assert_eq!(
            goofi_node::param(&g2.params(uid2).unwrap(), "constant", "value").unwrap().as_f64(),
            Some(7.0),
            "param round-trips through v7",
        );

        // v3-v6 belonged to the bare-text era and go with it (spec Decision 3: no back-compat).
        // v8 pins the OTHER direction: the gate is an equality, so a future arm cannot widen it
        // silently — enumerating only versions below 7 would never notice.
        for other in [
            "version: 6\nroot:\n  nodes: {}\n  links: []\n",
            "version: 3\nnodes: {}\nlinks: []\n",
            "version: 8\nroot:\n  nodes: {}\n  links: []\n",
        ] {
            let err = g.load_doc(other).unwrap_err();
            assert!(err.contains('7'), "the error names the version this build reads, got: {err}");
        }
    }

    /// A patch must come back with the uids it was saved with — INCLUDING when it loads into an
    /// instance that has already held other nodes. Uid stability is what every by-uid reference a
    /// load does NOT remap rests on: a viewer panel's `state.node`, an editor panel's
    /// `subpatchPath`, the viewpoint. Reminting silently repointed all three — and only ever on the
    /// SECOND patch, because a load into a fresh instance renumbers to the very values it saved.
    #[test]
    fn a_load_into_a_used_instance_keeps_the_saved_uids() {
        let mut authored = Graph::new();
        let a = authored.add_node("_TestEcho", None).unwrap();
        let b = authored.add_node("_TestEcho", None).unwrap();
        authored.add_link(a, "out", b, "in").unwrap();
        let scope = authored.group_nodes(&[b], [0.0, 0.0]).unwrap();
        // A viewer panel bound to `a`, exactly as the editor writes it.
        let page = authored.arrangement().pages()[0].clone();
        let panel = authored.arrangement().children(&page)[0].clone();
        let w = authored
            .arrangement()
            .set_panel(
                &page,
                &panel,
                Some("viewer"),
                Some(serde_json::json!({ "node": a.to_hex(), "slot": "out" })),
            )
            .unwrap();
        authored.arrangement_mut().apply(w);
        let saved = authored.serialize();

        // …into an instance that has already held OTHER nodes. This is the case the user hit, and
        // the only one that can fail — a fixture loading into a fresh graph proves nothing here.
        let mut used = Graph::new();
        for _ in 0..3 {
            used.add_node("_TestEcho", None).unwrap();
        }
        used.load_doc(&saved).unwrap();

        let mut got = used.node_uids();
        got.sort_by_key(|u| u.0);
        assert_eq!(got, vec![a, b], "the saved node uids are restored, not reminted");
        assert_eq!(used.scope_uids(), vec![scope], "and so is the scope uid `subpatchPath` names");
        assert_eq!(used.scope_of(b), Some(scope), "membership still resolves");
        assert_eq!(used.links.len(), 1, "the link survives");
        let bound = match used.arrangement().get(&panel) {
            Some(layout::Entry::Panel { state, .. }) => state["node"].as_str().unwrap().to_string(),
            other => panic!("the viewer panel is gone: {other:?}"),
        };
        assert_eq!(Uid::from_hex(&bound), Some(a), "the panel still names a LIVE node");
        assert!(used.contains(a), "…which is exactly the node it was bound to");
    }

    /// Restoring uids must advance the mint past them, or the next add collides with a node the
    /// load just brought back.
    #[test]
    fn a_restored_uid_never_collides_with_the_next_mint() {
        let mut authored = Graph::new();
        for _ in 0..4 {
            authored.add_node("_TestEcho", None).unwrap();
        }
        let saved = authored.serialize();

        let mut fresh = Graph::new();
        fresh.load_doc(&saved).unwrap();
        let next = fresh.add_node("_TestEcho", None).unwrap();
        assert!(!authored.contains(next), "the fresh mint is past every restored uid: {next}");
        assert_eq!(fresh.node_count(), 5, "so the add landed rather than clobbering a restored node");
    }

    /// A manifest whose node keys are not uids — hand-written, or generated by another tool — still
    /// opens: those nodes are minted fresh and the links keyed on the old names still resolve. Uid
    /// restoration is an upgrade for files goofi itself wrote (their keys have always BEEN the uid),
    /// never a new requirement on the format.
    #[test]
    fn a_manifest_with_non_uid_keys_still_loads() {
        let doc = "version: 7\nroot:\n  nodes:\n    alpha: { type: _TestEcho }\n    beta: { type: _TestEcho }\n  links: [[alpha, out, beta, in]]\n";
        let mut g = Graph::new();
        g.add_node("_TestEcho", None).unwrap(); // a used instance here too
        g.load_doc(doc).unwrap();
        assert_eq!(g.node_count(), 2, "both nodes minted");
        assert_eq!(g.links.len(), 1, "and the link between them resolved");

        // Two keys spelling the SAME number are two records, so the second one mints rather than
        // landing on — and silently replacing — the node the first restored.
        let twinned = "version: 7\nroot:\n  nodes:\n    \"1\": { type: _TestEcho }\n    \"000000000001\": { type: _TestEcho }\n  links: []\n";
        g.load_doc(twinned).unwrap();
        assert_eq!(g.node_count(), 2, "both records survive a duplicated uid spelling");
    }

    #[test]
    fn globals_round_trip_through_gfi() {
        use goofi_core::globals::GlobalValue;
        let mut g = Graph::new();
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(60.0))).unwrap();
        g.apply_global_change("subject", Some(GlobalValue::Str("P07".into()))).unwrap();
        g.apply_global_change("trials", Some(GlobalValue::Int(12))).unwrap();
        g.apply_global_change("live", Some(GlobalValue::Bool(true))).unwrap();
        let yaml = g.serialize();

        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.globals().get("default_ufreq"), Some(&GlobalValue::Float(60.0)), "edited system value");
        assert!(g2.globals().is_system("default_ufreq"), "still system after load");
        assert_eq!(g2.globals().get("subject"), Some(&GlobalValue::Str("P07".into())));
        assert_eq!(g2.globals().get("trials"), Some(&GlobalValue::Int(12)), "int type preserved (not floated)");
        assert_eq!(g2.globals().get("live"), Some(&GlobalValue::Bool(true)));
        assert!(!g2.globals().is_system("subject"), "user global loads as user");
    }

    #[test]
    fn globals_keep_their_ordered_position_across_a_gfi_round_trip() {
        use goofi_core::globals::GlobalValue;
        // Global order is observable (the panel, the mirror, expression-eval iteration). Seed user
        // globals in a NON-alphabetical order and confirm serialize→load preserves it — a
        // serde_json::Map (BTreeMap) would silently alphabetize them.
        let mut g = Graph::new();
        g.apply_global_change("zebra", Some(GlobalValue::Int(1))).unwrap();
        g.apply_global_change("apple", Some(GlobalValue::Int(2))).unwrap();
        g.apply_global_change("mango", Some(GlobalValue::Int(3))).unwrap();
        let order = |g: &Graph| g.globals().entries().map(|(k, _, _)| k.to_string()).collect::<Vec<_>>();
        let before = order(&g);
        let mut alphabetical = before.clone();
        alphabetical.sort();
        assert_ne!(before, alphabetical, "the seed order must be non-alphabetical for this test to bite");

        let text = g.serialize();
        let mut g2 = Graph::new();
        g2.load_doc(&text).unwrap();
        assert_eq!(order(&g2), before, "user global order survives the round trip (not alphabetized)");
    }

    #[test]
    fn a_globals_less_patch_loads_with_system_globals_seeded() {
        // A patch with no `globals` block loads fine — the system defaults are seeded.
        let doc = "version: 7\npillar_default: signal\nroot:\n  nodes:\n    n0: { type: _TestConst, name: c0, pos: [1.0, 2.0], params: {} }\n  links: []\n";
        let mut g = Graph::new();
        g.load_doc(doc).unwrap();
        assert_eq!(g.node_uids().len(), 1, "nodes load");
        assert_eq!(
            g.globals().get("default_ufreq"),
            Some(&goofi_core::globals::GlobalValue::Float(30.0)),
            "system default seeded on a globals-less patch",
        );
    }

    #[test]
    fn the_arrangement_rides_the_gfi_and_a_corrupt_one_never_costs_the_patch() {
        use crate::layout::{Axis, Layout};
        let mut g = Graph::new();
        g.add_node("_TestConst", None).unwrap();
        let page = g.arrangement().pages()[0].clone();
        let panel = g.arrangement().children(&page)[0].clone();
        let (w, fresh) = g.arrangement().split_panel(&page, &panel, Axis::Column, false, 0.25).unwrap();
        g.arrangement_mut().apply(w);
        let text = g.serialize();

        let mut g2 = Graph::new();
        g2.load_doc(&text).unwrap();
        assert_eq!(g2.arrangement(), g.arrangement(), "the arrangement round-trips entry for entry");
        assert!(g2.arrangement_warning().is_none(), "a valid arrangement loads silently");

        // Flattening admits corruption the nested tree could not express. It must cost the CHROME,
        // never the patch: the graph is the value, the arrangement is how it is looked at.
        let mut doc: serde_json::Value = serde_yaml_ng::from_str(&text).unwrap();
        doc["arrangement"][&fresh]["parent"] = serde_json::json!("gone");
        let broken = serde_yaml_ng::to_string(&doc).unwrap();
        let mut g3 = Graph::new();
        g3.load_doc(&broken).expect("a corrupt arrangement does not refuse the patch");
        assert_eq!(g3.node_uids().len(), 1, "the graph loaded regardless");
        assert_eq!(g3.arrangement(), &Layout::default(), "the arrangement fell back to the default");
        let warning = g3.arrangement_warning().expect("the fallback is stated, not silent");
        assert!(warning.contains("reaches no page"), "and it says what was wrong: {warning}");

        // A patch saved before this shape existed simply opens on the default, with nothing to say.
        let mut g4 = Graph::new();
        g4.load_doc(&Graph::new().serialize()).unwrap();
        assert_eq!(g4.arrangement(), &Layout::default());
        assert!(g4.arrangement_warning().is_none(), "an absent arrangement is not a corrupt one");
        assert!(g4.load_doc(&text).is_ok() && g4.arrangement_warning().is_none(), "and the flag clears");
    }

    #[test]
    fn the_viewpoint_persists_but_is_not_arrangement() {
        // Where a client is LOOKING — active page, maximize, camera, and each panel's sub-patch path
        // — is per-client, so it stays out of the shared doc. Persistence is the separate axis: it
        // still rides the `.gfi`, so reopening a patch restores the saver's viewpoint.
        let mut g = Graph::new();
        assert_eq!(g.viewpoint(), &serde_json::Value::Null, "none until a client sets one");
        let vp = serde_json::json!({ "activePage": "page-1", "subpatchPath": { "panel-2": ["a1b2"] } });
        g.set_viewpoint(vp.clone());
        let mut g2 = Graph::new();
        g2.load_doc(&g.serialize()).unwrap();
        assert_eq!(g2.viewpoint(), &vp, "stored verbatim, like the layout blob it was carved out of");

        let mut g3 = Graph::new();
        g3.set_viewpoint(vp);
        g3.load_doc(&Graph::new().serialize()).unwrap();
        assert_eq!(g3.viewpoint(), &serde_json::Value::Null, "a patch without one clears it");
    }

    #[test]
    fn viewer_view_state_round_trips_through_gfi() {
        // The editor's per-slot viewer state is stored opaquely, echoed back, persisted to
        // .gfi, and reconstructed — the backend never interprets the blob.
        let mut g = Graph::new();
        let n = g.add_node("_TestConst", None).unwrap();
        assert_eq!(g.viewers(n), Some(&serde_json::json!({})), "empty until set");
        let vs = serde_json::json!({ "out": { "collapsed": false, "kind": "line", "settings": { "yScale": 2 } } });
        g.set_node_viewers(n, vs.clone()).unwrap();
        assert_eq!(g.viewers(n), Some(&vs), "stored verbatim");

        let yaml = g.serialize();
        assert!(yaml.contains("viewers"), "view-state persisted; got:\n{yaml}");
        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.viewers(g2.node_uids()[0]), Some(&vs), "view-state reconstructed");
    }

    #[test]
    fn empty_viewers_stay_out_of_the_gfi() {
        // A node whose view-state was never set (or cleared to {}) writes no `viewers` key —
        // a fresh patch carries no editor noise.
        let mut g = Graph::new();
        g.add_node("_TestConst", None).unwrap();
        assert!(!g.serialize().contains("viewers"), "no empty viewers blob in the file");
    }

    #[test]
    fn group_nodes_is_a_reference_move_only() {
        // Group a 2-node chain [a,b] (a→b internal, b→c cut): the flat graph is byte-identical
        // (same uids, same links), only `scope_of` re-tags. The cut b→c mints one Out stub whose
        // `inner` is (b, "out"); the internal a→b link stays a flat link (never captured).
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", c, "in").unwrap();
        let nodes_before = g.node_uids();
        let links_before = g.links_view().len();

        let s = g.group_nodes(&[a, b], [100.0, 100.0]).unwrap();

        assert_eq!(g.node_uids(), nodes_before, "no node minted/removed; uids identical");
        assert_eq!(g.links_view().len(), links_before, "both flat links untouched");
        assert_eq!(g.scope_of(a), Some(s), "a is now a member of the scope");
        assert_eq!(g.scope_of(b), Some(s));
        assert_eq!(g.scope_of(c), None, "c stays at ROOT (external)");
        assert_eq!(g.scope_of(s), None, "the scope sits at ROOT");
        let scope = g.scope(s).unwrap();
        assert_eq!(scope.stubs.len(), 1, "only the b→c cut mints a stub");
        let (_id, stub) = scope.stubs.iter().next().unwrap();
        assert_eq!(stub.dir, subpatch::Dir::Out, "downstream output boundary");
        assert_eq!(stub.inner, Some((b, "out".to_string())), "stub exposes b.out");
        assert_eq!(g.scope_members(s).len(), 2, "two members");
    }

    #[test]
    fn group_nodes_rejects_mixed_scope_without_mutating() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestConst", None).unwrap();
        let inner = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let scopes_before = g.scope_uids().len();
        let err = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap_err();
        assert!(err.contains("scope"), "mixed-scope error; got {err}");
        assert_eq!(g.scope_uids().len(), scopes_before, "no scope created on failure");
        assert_eq!(g.scope_of(a), Some(inner), "a's membership untouched");
        assert_eq!(g.scope_of(b), None, "b stays at ROOT");
    }

    #[test]
    fn stub_authoring_add_wire_rename_and_one_per_inner() {
        use subpatch::Dir;
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let s = g.group_nodes(&[a], [0.0, 0.0]).unwrap(); // a→b cut auto-exposes a.out as out0
        let before = g.scope(s).unwrap().stubs.len();

        let stub = g.add_boundary(s, Dir::In, goofi_core::SlotType::Array, [10.0, 10.0]).unwrap();
        assert_eq!(g.scope(s).unwrap().stubs.len(), before + 1, "stub added");
        assert!(g.scope(s).unwrap().stubs[&stub].inner.is_none(), "born unwired");

        g.set_stub_inner(s, &stub, Some((a, "in".to_string()))).unwrap();
        assert_eq!(g.resolve_stub(s, &stub), Some((a, "in".to_string())), "wired stub resolves to a.in");

        // One stub per inner slot: a.in is exposed by `stub`, a second wiring is rejected.
        let extra = g.add_boundary(s, Dir::In, goofi_core::SlotType::Array, [0.0, 0.0]).unwrap();
        let err = g.set_stub_inner(s, &extra, Some((a, "in".to_string()))).unwrap_err();
        assert!(err.contains("already exposed"), "one-stub-per-inner enforced; got {err}");

        // rename keeps the StubId (external wires survive), only the label changes.
        g.rename_boundary(s, &stub, "signal").unwrap();
        assert_eq!(g.scope(s).unwrap().stubs[&stub].name, "signal");
        assert!(g.scope(s).unwrap().stubs.contains_key(&stub), "StubId unchanged after rename");

        // wiring a non-member is rejected.
        let outsider = g.add_node("_TestConst", None).unwrap();
        assert!(g.set_stub_inner(s, &stub, Some((outsider, "in".to_string()))).is_err(), "non-member rejected");
    }

    #[test]
    fn a_boundary_can_be_wired_to_a_nested_scopes_port() {
        // `Stub.inner` is documented to hold a nested scope's `(facade uid, StubId)`, and
        // `group_nodes`/`expose_in_nested_member` mint exactly that — but the wire path
        // resolved the dtype through `self.nodes` only, so dragging a boundary pill onto a
        // nested sub-patch's facade port was refused (and `Command::WireStub` swallows the
        // refusal, so the cable just vanished on the next reconcile).
        use subpatch::Dir;
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let t = g.group_nodes(&[a], [0.0, 0.0]).unwrap(); // inner scope holding the leaf
        let s = g.group_nodes(&[t], [0.0, 0.0]).unwrap(); // outer scope holding that scope

        let t_out = g.add_boundary(t, Dir::Out, goofi_core::SlotType::Array, [0.0, 0.0]).unwrap();
        g.set_stub_inner(t, &t_out, Some((a, "out".to_string()))).unwrap();

        let s_out = g.add_boundary(s, Dir::Out, goofi_core::SlotType::String, [0.0, 0.0]).unwrap();
        g.set_stub_inner(s, &s_out, Some((t, t_out.clone()))).expect("wiring to a nested scope's port");
        assert_eq!(g.resolve_stub(s, &s_out), Some((a, "out".to_string())), "chains through T to the leaf");
        assert_eq!(g.scope(s).unwrap().stubs[&s_out].dtype, goofi_core::SlotType::Array, "dtype taken from T's port");

        // The nested port must exist and face the same way.
        let s_in = g.add_boundary(s, Dir::In, goofi_core::SlotType::Array, [0.0, 0.0]).unwrap();
        assert!(g.set_stub_inner(s, &s_in, Some((t, t_out.clone()))).is_err(), "an Out port cannot back an In pill");
        assert!(g.set_stub_inner(s, &s_in, Some((t, "in7".to_string()))).is_err(), "no such nested port");
    }

    #[test]
    fn expand_restores_membership_and_drops_the_scope() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let nodes_before = g.node_uids();
        let links_before = g.links_view().len();
        let s = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();

        let restored = g.expand_instance(s).unwrap();
        assert_eq!(restored.len(), 2, "two members restored");
        assert_eq!(g.scope_of(a), None, "a back at ROOT");
        assert_eq!(g.scope_of(b), None, "b back at ROOT");
        assert!(g.scope(s).is_none(), "scope dropped");
        assert!(g.scope_uids().is_empty(), "no scopes remain");
        assert_eq!(g.node_uids(), nodes_before, "uids identical after group→expand");
        assert_eq!(g.links_view().len(), links_before, "flat links intact");
    }

    #[test]
    fn group_then_expand_round_trips_uid_identical() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", c, "in").unwrap();
        let nodes0 = g.node_uids();
        let links0: Vec<_> = g.links_view().iter().map(|l| (l.node_out, l.slot_out, l.node_in, l.slot_in)).collect();

        let s = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();
        g.expand_instance(s).unwrap();

        assert_eq!(g.node_uids(), nodes0, "nodes uid-identical");
        let links1: Vec<_> = g.links_view().iter().map(|l| (l.node_out, l.slot_out, l.node_in, l.slot_in)).collect();
        assert_eq!(links1, links0, "links identical");
    }

    #[test]
    fn remove_instance_tears_down_the_whole_subtree() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let s = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();
        g.remove_instance(s).unwrap();
        assert!(g.scope(s).is_none(), "scope gone");
        assert!(g.node_uids().is_empty(), "both members torn down");
        assert!(g.links_view().is_empty(), "their link gone");
    }

    #[test]
    fn remove_member_drops_a_leaf_and_its_stubs() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap(); // a external → b (an In stub once grouped)
        g.add_link(b, "out", c, "in").unwrap(); // b → c external (an Out stub)
        let s = g.group_nodes(&[b], [0.0, 0.0]).unwrap();
        assert_eq!(g.scope(s).unwrap().stubs.len(), 2, "in + out stubs, both inner == b");

        g.remove_member(b).unwrap();
        assert!(g.name(b).is_none(), "b torn down");
        assert_eq!(g.scope(s).unwrap().stubs.len(), 0, "both stubs referencing b dropped");
        assert!(g.name(a).is_some() && g.name(c).is_some(), "external nodes survive");
        assert!(g.links_view().is_empty(), "b's flat links removed");
    }

    #[test]
    fn a_removed_member_is_not_resurrected_on_reload() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let s = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();
        let _ = s;
        g.remove_member(b).unwrap();
        let yaml = g.serialize();
        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_uids().len(), 1, "only a survives; b not resurrected");
        assert_eq!(g2.scope_uids().len(), 1, "the scope persists with its remaining member");
    }

    #[test]
    fn grouping_a_nested_member_maps_interior_links_to_the_nested_stub() {
        // a → b → c. Group [b] → s1 (in0=b.in, out0=b.out). Then group [a, s1]: a→b is wholly
        // inside (no stub), and b→c crosses via s1 → an Out stub on s2 whose inner is s1's out0,
        // chain-resolving to (b, "out").
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", c, "in").unwrap();
        let s1 = g.group_nodes(&[b], [0.0, 0.0]).unwrap();
        let s2 = g.group_nodes(&[a, s1], [0.0, 0.0]).unwrap();

        let out_stubs: Vec<_> =
            g.scope(s2).unwrap().stubs.values().filter(|st| st.dir == subpatch::Dir::Out).collect();
        assert_eq!(out_stubs.len(), 1, "one Out stub for the b→c cut");
        assert_eq!(out_stubs[0].inner, Some((s1, "out0".to_string())), "inner is the nested scope's stub");
        let (id, _) =
            g.scope(s2).unwrap().stubs.iter().find(|(_, st)| st.dir == subpatch::Dir::Out).unwrap();
        assert_eq!(g.resolve_stub(s2, id), Some((b, "out".to_string())), "chain resolves to b.out");
    }

    #[test]
    fn grouping_a_scope_with_an_orphaned_crossing_link_re_exposes_the_buried_leaf() {
        // remove_boundary drops a boundary port but LEAVES the external flat link (leaf→leaf, by
        // design — the data path is untouched). Re-grouping that scope used to trip the
        // capture-invariant debug_assert (a panic poisoning the graph mutex → a dev/CI DoS) and mint
        // a dangling stub in release. group_nodes must instead re-mint the missing exposing stub on
        // the nested member so the outer boundary chain-resolves to the buried leaf.
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let x = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", x, "in").unwrap();

        let s = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        assert_eq!(g.resolve_stub(s, "out0"), Some((a, "out".to_string())), "s exposes a.out as out0");

        // Drop the port; the flat link a.out→x.in survives leaf→leaf (documented remove_boundary).
        g.remove_boundary(s, "out0").unwrap();
        assert!(g.scope(s).unwrap().stubs.is_empty(), "boundary port removed");
        assert!(
            g.links_view().iter().any(|l| l.node_out == a && l.node_in == x),
            "the external flat link survives the port removal"
        );

        // Re-group the orphaned scope — must NOT panic, and the new boundary must chain to a.out.
        let t = g.group_nodes(&[s], [1.0, 1.0]).unwrap();
        let tid = g
            .scope(t)
            .unwrap()
            .stubs
            .iter()
            .find(|(_, st)| st.dir == subpatch::Dir::Out)
            .map(|(id, _)| id.clone())
            .expect("t exposes an Out boundary for the crossing link");
        assert_eq!(
            g.resolve_stub(t, &tid),
            Some((a, "out".to_string())),
            "t's boundary chain-resolves to the buried leaf a.out (a fresh stub was minted on s)"
        );
    }

    #[test]
    fn expanding_an_outer_scope_reparents_a_nested_scope_to_the_grandparent() {
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let s1 = g.group_nodes(&[b], [0.0, 0.0]).unwrap();
        let s2 = g.group_nodes(&[a, s1], [0.0, 0.0]).unwrap();
        assert_eq!(g.scope_of(s1), Some(s2), "s1 nested in s2");
        g.expand_instance(s2).unwrap();
        assert_eq!(g.scope_of(s1), None, "s1 re-parented to ROOT");
        assert_eq!(g.scope_of(a), None, "a re-parented to ROOT");
        assert!(g.scope(s1).is_some(), "the nested scope itself survives");
    }

    #[test]
    fn expanding_an_inner_scope_re_points_the_parent_stub_that_referenced_it() {
        // a→b→c; group[b]→s1; group[a,s1]→s2 with an Out stub inner=(s1, out0) for the b→c cut.
        // Expanding the INNER scope s1 dissolves it and moves b UP into s2 — s2's stub must FOLLOW to
        // (b, out), not dangle at the vanished s1. (remove_member PRUNES such a stub because its
        // member is deleted; expand RE-POINTS because the leaf survives.)
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", c, "in").unwrap();
        let s1 = g.group_nodes(&[b], [0.0, 0.0]).unwrap();
        let s2 = g.group_nodes(&[a, s1], [0.0, 0.0]).unwrap();
        let sid = g
            .scope(s2)
            .unwrap()
            .stubs
            .iter()
            .find(|(_, st)| st.dir == subpatch::Dir::Out)
            .map(|(id, _)| id.clone())
            .unwrap();
        assert_eq!(g.scope(s2).unwrap().stubs[&sid].inner, Some((s1, "out0".to_string())), "points at s1's port");
        assert_eq!(g.resolve_stub(s2, &sid), Some((b, "out".to_string())), "chain resolves to b.out");

        g.expand_instance(s1).unwrap();
        assert_eq!(g.scope_of(b), Some(s2), "b moved up into s2");
        assert_eq!(
            g.scope(s2).unwrap().stubs[&sid].inner,
            Some((b, "out".to_string())),
            "the parent stub re-pointed to the direct member (leaf b), not the vanished scope s1"
        );
        assert_eq!(g.resolve_stub(s2, &sid), Some((b, "out".to_string())), "still resolves after expand");
    }

    #[test]
    fn expand_command_undo_restores_the_parent_stub_exactly() {
        // The Command::Expand round-trip must be EXACT: expand_instance re-points a parent stub, so
        // Expand's Group inverse must re-point it BACK to (scope, child_id) — not leave it at the
        // resolved leaf (which resolves the same but is structurally non-canonical).
        use crate::Command;
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", c, "in").unwrap();
        let s1 = g.group_nodes(&[b], [0.0, 0.0]).unwrap();
        let s2 = g.group_nodes(&[a, s1], [0.0, 0.0]).unwrap();
        let sid = g
            .scope(s2)
            .unwrap()
            .stubs
            .iter()
            .find(|(_, st)| st.dir == subpatch::Dir::Out)
            .map(|(id, _)| id.clone())
            .unwrap();
        let before = g.scope(s2).unwrap().stubs[&sid].inner.clone(); // Some((s1, out0))

        let (_r, undo) = Command::Expand { scope: s1 }.execute(&mut g).unwrap();
        assert_ne!(g.scope(s2).unwrap().stubs[&sid].inner, before, "expand re-pointed the parent stub");

        let (_r2, redo) = undo.execute(&mut g).unwrap();
        assert_eq!(
            g.scope(s2).unwrap().stubs[&sid].inner,
            before,
            "undo restored the parent stub EXACTLY to (s1, out0)"
        );

        // Redo re-expands and re-points again (the cycle is stable).
        redo.execute(&mut g).unwrap();
        assert_eq!(g.scope_of(b), Some(s2), "redo re-expanded s1");
        assert_eq!(g.resolve_stub(s2, &sid), Some((b, "out".to_string())), "and still resolves");
    }

    #[test]
    fn rename_rejects_collision_with_a_scope_name() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestConst", None).unwrap();
        let s = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let scope_name = g.scope(s).unwrap().name.clone();
        let err = g.rename_node(b, &scope_name).unwrap_err();
        assert!(err.contains("in use"), "a leaf can't take a scope's display name; got {err}");
    }

    #[test]
    fn set_node_pos_moves_a_scope_facade() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let s = g.group_nodes(&[a], [1.0, 2.0]).unwrap();
        g.set_node_pos(s, [7.0, 8.0]).unwrap();
        assert_eq!(g.scope(s).unwrap().pos, [7.0, 8.0], "scope facade pos updated in place");
    }

    #[test]
    fn flat_scopes_survive_a_gfi_roundtrip() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", c, "in").unwrap();
        let s = g.group_nodes(&[a, b], [10.0, 20.0]).unwrap();
        let stub_count = g.scope(s).unwrap().stubs.len();

        let yaml = g.serialize();
        assert!(yaml.contains("scopes:"), "flat scope overlay persisted");
        assert!(!yaml.contains("definitions:"), "no def bodies persisted");

        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_uids().len(), 3, "all leaves restored");
        assert_eq!(g2.scope_uids().len(), 1, "the scope restored");
        let s2 = g2.scope_uids()[0];
        assert_eq!(g2.scope(s2).unwrap().stubs.len(), stub_count, "stubs restored");
        assert_eq!(g2.scope_members(s2).len(), 2, "membership restored");
        let (id, _) =
            g2.scope(s2).unwrap().stubs.iter().find(|(_, st)| st.dir == subpatch::Dir::Out).unwrap();
        assert!(g2.resolve_stub(s2, id).is_some(), "restored stub still chain-resolves");
    }

    #[test]
    fn three_deep_nesting_survives_a_gfi_roundtrip() {
        // a → b → c → d. Nest c three scopes deep; the outermost Out stub chains s3→s2→s1→c.out.
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        let d = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", c, "in").unwrap();
        g.add_link(c, "out", d, "in").unwrap();
        let s1 = g.group_nodes(&[c], [0.0, 0.0]).unwrap();
        let s2 = g.group_nodes(&[b, s1], [0.0, 0.0]).unwrap();
        let s3 = g.group_nodes(&[a, s2], [0.0, 0.0]).unwrap();
        let (out3, _) =
            g.scope(s3).unwrap().stubs.iter().find(|(_, st)| st.dir == subpatch::Dir::Out).unwrap();
        assert_eq!(g.resolve_stub(s3, out3), Some((c, "out".to_string())), "3-deep chain → c.out");

        let mut g2 = Graph::new();
        g2.load_doc(&g.serialize()).unwrap();
        assert_eq!(g2.node_uids().len(), 4, "all four leaves restored");
        assert_eq!(g2.scope_uids().len(), 3, "all three scopes restored");
        // Find the outermost scope (parent = ROOT) and confirm its Out stub still resolves 3-deep.
        let root_scope = g2.scope_uids().into_iter().find(|s| g2.scope_of(*s).is_none()).unwrap();
        let (out3b, _) =
            g2.scope(root_scope).unwrap().stubs.iter().find(|(_, st)| st.dir == subpatch::Dir::Out).unwrap();
        let (leaf, slot) = g2.resolve_stub(root_scope, out3b).expect("3-deep chain resolves after reload");
        assert_eq!(slot, "out", "resolves to a leaf's out slot after reload");
        assert!(g2.name(leaf).is_some(), "the resolved endpoint is a live leaf");
    }

    #[test]
    fn a_half_wired_stub_survives_a_gfi_roundtrip() {
        use subpatch::Dir;
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let s = g.group_nodes(&[a], [0.0, 0.0]).unwrap(); // auto out0 (a.out → b cut)
        let unwired = g.add_boundary(s, Dir::In, goofi_core::SlotType::Array, [0.0, 0.0]).unwrap();
        assert!(g.scope(s).unwrap().stubs[&unwired].inner.is_none(), "born unwired");

        let mut g2 = Graph::new();
        g2.load_doc(&g.serialize()).unwrap();
        let s2 = g2.scope_uids()[0];
        assert_eq!(g2.scope(s2).unwrap().stubs.len(), 2, "wired + half-wired stubs both restored");
        let dangling = g2.scope(s2).unwrap().stubs.values().filter(|st| st.inner.is_none()).count();
        assert_eq!(dangling, 1, "the half-wired stub round-trips as present-but-dangling");
    }

    #[test]
    fn teardown_releases_compiled_handles() {
        // remove_node and clear (hence load_doc) must release the evaluator's handles. Counted
        // against a live baseline rather than against 1: a fresh producer also seeds its universal
        // `common.max_frequency` binding, so the node under test holds two handles, and the point
        // here is that NONE of them survive their node.
        let mock = Arc::new(MockEval::default());
        let mut g = Graph::new();
        g.set_evaluator(mock.clone());
        let live = || mock.exprs.lock().unwrap().len();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "5", true, false).unwrap();
        assert!(live() >= 1, "the binding compiled");
        g.remove_node(n).unwrap();
        assert_eq!(live(), 0, "released on remove_node");
        let n2 = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n2, "constant", "value", "7", true, false).unwrap();
        assert!(live() >= 1);
        g.clear();
        assert_eq!(live(), 0, "released on clear");
    }

    #[test]
    fn binding_error_clears_on_recovery_even_for_a_never_running_node() {
        // _TestSink has a trigger input, autotrigger off, and (unwired) never runs — so
        // run_node never fires for it. The node-level error must still clear when its
        // expression recovers, because last_error() derives the binding error on read.
        let mut g = eval_graph();
        let sink = g.add_node("_TestSink", None).unwrap();
        g.set_expression(sink, "control", "value", "nd('src')", true, false).unwrap();
        wait_for(&mut g, "the missing ref to error while idle", |g| g.last_error(sink).is_some());
        let src = g.add_node("_TestConst", None).unwrap();
        g.rename_node(src, "src").unwrap();
        wait_for(&mut g, "recovery to clear the node error on a never-running node", |g| {
            g.last_error(sink).is_none()
        });
    }

    #[test]
    fn multiple_binding_errors_surface_deterministically() {
        // Two errored bindings on one node -> the smaller ParamKey (constant/length <
        // constant/value) wins, deterministically (not HashMap order).
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "nd('gv')", true, false).unwrap();
        g.set_expression(n, "constant", "length", "nd('gl')", true, false).unwrap();
        wait_for(&mut g, "both binding errors to reach the graph", |g| {
            g.last_error(n).is_some_and(|e| e.contains("gl"))
        });
        let err = g.last_error(n).expect("a binding error surfaces");
        assert!(err.contains("gl"), "deterministic min-ParamKey selection, got: {err}");
    }

    #[test]
    fn length_preserving_node_propagates_source_index() {
        // _TestConst(len 2) -> Echo (echoes -> len 2). The echo's output frame
        // count matches its single index-bearing input, so it PROPAGATES the source's origin index
        // rather than starting a fresh counter — an upstream drop stays visible at the sink. The
        // source runs unwired for a while first, so its index is well past 0 and a fresh-from-0
        // counter is distinguishable from a propagated one.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10)).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        let source = OutputProbe::open(&g, src, "out");
        let out = OutputProbe::open(&g, echo, "out");
        source.wait_until(&mut g, "the source to be well past its first emit", |d| {
            d.meta().index().unwrap_or(0) > 10
        });
        g.add_link(src, "out", echo, "in").unwrap();
        let echoed = out.expect_frame(&mut g, "the echo to run").meta().index().expect("stamped");
        assert!(echoed > 10, "propagates the source's index, not a fresh 0: {echoed}");
    }

    #[test]
    fn accumulating_length_changing_node_keeps_index_monotonic() {
        // Reference-patch shape: a source -> Buffer. Buffer's FIRST output frame (ring
        // holds one input's worth) coincidentally equals the input length, then grows.
        // The first frame must not propagate/duplicate the source index; the slot stays a
        // monotonic fresh timeline (regression for the [0,0,1,2] stamp_meta bug).
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10)).unwrap();
        // Born capped: the FIRST emit is the whole subject, and a one-deep latest-wins cell holds
        // it for as long as the next run is away.
        let buf = capped(&mut g, "Buffer", 20.0);
        let out = OutputProbe::open(&g, buf, "out");
        g.add_link(src, "out", buf, "data").unwrap();
        // The bug this pins stamped [0, 0, 1, 2]: the buffer's first output length equals its
        // input's, so the index PROPAGATED, and the fresh counter then restarted from 0 and
        // repeated it. A repeat is invisible to an index-only oracle — through a one-deep
        // latest-wins cell, "the same index again" and "no new frame yet" read identically — so
        // the emits are told apart by the ring's GROWING length, which is two more each time.
        //
        // The starting number is deliberately not pinned: it is whatever the source had reached
        // when the wire attached, and asserting 0 was asserting the old tick's lockstep.
        let first = out
            .wait_until(&mut g, "the buffer's first emit", |d| as_f32_vec(d).len() == 2)
            .meta()
            .index()
            .expect("stamped");
        for step in 1..4u64 {
            let want = 2 * (step as usize + 1);
            let f = out.wait_until(&mut g, "the next, longer buffered frame", |d| as_f32_vec(d).len() == want);
            assert_eq!(f.meta().index(), Some(first + step), "one index per emit, never repeated");
        }
    }

    #[test]
    fn length_changing_node_uses_fresh_index() {
        // _TestConst(len 2) -> Counter (emits len 1). The output frame count (1)
        // never matches the input (2), so no input is the same timeline: the counter
        // starts its OWN fresh index at 0, independent of the source's index (3).
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10)).unwrap();
        // Born capped, so its FIRST frame — the one carrying index 0 — is still in the probe's
        // one-deep cell when the assertion looks.
        let cnt = capped(&mut g, "_TestCounter", 5.0);
        let source = OutputProbe::open(&g, src, "out");
        let out = OutputProbe::open(&g, cnt, "out");
        source.wait_until(&mut g, "the source to be well past its first emit", |d| {
            d.meta().index().unwrap_or(0) > 10
        });
        g.add_link(src, "out", cnt, "in").unwrap();
        let f = out.expect_frame(&mut g, "the counter to run");
        assert_eq!(f.meta().index(), Some(0), "fresh counter, not the source's index");
    }

    #[test]
    fn every_type_that_free_runs_says_so_in_its_own_declaration() {
        // The tick free-ran any node with no *triggering* input, whatever its params said. §1
        // removed that implicit rule — a node that declares no trigger input and leaves autotrigger
        // off never runs, and that is correct — so a type that relied on it has to declare the
        // pacing itself, via `producer` or by declaring `common.autotrigger` in its own params.
        //
        // The operative predicate is `!any(trigger_process)`, NOT `inputs.is_empty()`: a node can
        // declare a held reference input and still free-run.
        //
        // REACH: the `inventory` catalog only — goofi-nodes plus the test types this crate submits.
        // That is every type goofi SHIPS, which is what matters. It does NOT cover a manifest handed
        // to `register_dyn_type`, since those are per-`Graph` and have no registry to walk: today
        // that means `RESHAPE_V2` here, plus this crate's and `goofi-bridge`'s fixture statics —
        // all of them test fixtures. A discovered Python node is also outside it by nature — its
        // `producer` comes from the author's class attribute, so a missing one is an authoring
        // mistake this process cannot diagnose, not an invariant to assert.
        for m in goofi_node::catalog() {
            if m.inputs.iter().any(|s| s.trigger_process) {
                continue;
            }
            let common = goofi_node::with_common(m.default_params(), m);
            assert_eq!(
                goofi_node::param(&common, "common", "autotrigger").and_then(Param::as_bool),
                Some(true),
                "`{}` has no triggering input, so it free-runs on the implicit rule alone — set \
                 `producer: true` on its manifest (or declare `common.autotrigger` yourself)",
                m.type_name,
            );
        }
    }

    #[test]
    fn every_node_gets_a_common_group() {
        // The engine merges a universal `common` scheduling group into every node
        // (like Python's DEFAULT_PARAMS), so rate controls exist uniformly.
        let mut g = Graph::new();
        let c = g.add_node("_TestConst", None).unwrap();
        let p = g.params(c).unwrap();
        let common = p.get("common").expect("common group injected");
        assert!(common.contains_key("autotrigger"));
        assert!(common.contains_key("max_frequency"));
        assert!(common.contains_key("frequency_mode"));
        // Unbounded by default. `_TestConst` is a producer, so its autotrigger is on; a node
        // driven by its input gets the same group with the opposite default, from the same
        // declaration — the manifest's `producer` is the only thing that differs.
        assert_eq!(common["max_frequency"].as_f64(), Some(0.0));
        assert_eq!(common["autotrigger"].as_bool(), Some(true), "a source paces itself");
        let b = g.add_node("Buffer", None).unwrap();
        let consumer = g.params(b).unwrap();
        assert_eq!(consumer["common"]["autotrigger"].as_bool(), Some(false), "a transform does not");
    }

    #[test]
    fn common_max_frequency_caps_a_node_by_wall_clock() {
        // The rate cap is what stops an uncapped producer saturating a core, and it is enforced by
        // the node against its own clock — there is no scheduler left to enforce it for anyone.
        //
        // The gate is a CEILING, not a target: a machine under load admits fewer runs, never more,
        // so a `<=` bound cannot flake upward. The floor is deliberately loose and exists only to
        // fail a node that stopped running altogether, which the ceiling alone would pass.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "common", "max_frequency", Param::float(10.0, 0.0, 1e9)).unwrap();
        let out = OutputProbe::open(&g, src, "out");
        // Take the index AFTER the cap has landed, or the uncapped emits from before it is applied
        // are counted against the window.
        std::thread::sleep(Duration::from_millis(200));
        let first = out.expect_frame(&mut g, "the capped source to emit").meta().index().expect("stamped");
        std::thread::sleep(Duration::from_millis(1000));
        let emitted = out.expect_frame(&mut g, "the capped source to keep emitting").meta().index().unwrap() - first;
        assert!(emitted <= 20, "a 10 Hz cap emitted {emitted} frames in a second");
        assert!(emitted >= 3, "and it did not stop: {emitted} frames in a second");
    }

    #[test]
    fn run_policy_survives_gfi_roundtrip() {
        // A saved max_frequency must re-derive into the loaded node's run gate.
        let mut g = Graph::new();
        let c = g.add_node("_TestConst", None).unwrap();
        g.update_param(c, "common", "max_frequency", Param::float(10.0, 0.0, 60.0)).unwrap();
        let yaml = g.serialize();

        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        let c2 = g2.node_uids()[0];
        assert_eq!(
            goofi_node::param(&g2.params(c2).unwrap(), "common", "max_frequency").unwrap().as_f64(),
            Some(10.0),
            "max_frequency round-trips"
        );
        // …and the gate it re-derives is ACTIVE, not merely stored: an uncapped node would run
        // hundreds of times in this window.
        let out = OutputProbe::open(&g2, c2, "out");
        let first = out.expect_frame(&mut g2, "the loaded node to emit").meta().index().expect("stamped");
        std::thread::sleep(Duration::from_millis(500));
        let emitted = out.expect_frame(&mut g2, "the loaded node to keep emitting").meta().index().unwrap() - first;
        assert!(emitted <= 15, "the 10 Hz gate is active post-load: {emitted} frames in half a second");
    }

    #[test]
    fn autotrigger_free_runs_an_unwired_trigger_node() {
        // The faithful counterpart: a node that DECLARES a trigger input but has it
        // UNWIRED, with autotrigger=true, free-runs every tick (Python:
        // `_has_no_triggering_inputs()` is true when the slot has no source). This
        // guards the fix from over-correcting the wired case into this one.
        let mut g = Graph::new();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        let out = OutputProbe::open(&g, cnt, "out");
        assert!(out.silent(&mut g), "with autotrigger off and nothing wired, it never runs");
        g.update_param(cnt, "common", "autotrigger", Param::boolean(true)).unwrap();
        out.wait_until(&mut g, "free-runs past its third call", |d| first_f32(d) >= 3.0);
    }

    #[test]
    fn ctx_now_is_seconds_since_the_patch_started() {
        // One clock for the whole patch, not one per node thread: `NodeCtx::now` is the time since
        // the graph was created, so two nodes born a minute apart agree about what time it is.
        let mut g = Graph::new();
        let a = g.add_node("_TestNow", None).unwrap();
        let out_a = OutputProbe::open(&g, a, "out");
        out_a.wait_until(&mut g, "the clock to advance past 200 ms", |d| first_f32(d) > 0.2);

        let b = g.add_node("_TestNow", None).unwrap();
        let out_b = OutputProbe::open(&g, b, "out");
        let born_late = first_f32(&out_b.expect_frame(&mut g, "the second node to run"));
        assert!(
            born_late > 0.2,
            "a node born into a running patch reads the patch's clock, not its own age: {born_late}",
        );
    }

    #[test]
    fn a_load_restarts_the_node_clock() {
        // A patch loaded into a running session must behave like the same patch loaded at boot.
        // The load happens at a genuinely ADVANCED clock — at t≈0 a graph that re-anchors and one
        // that does not agree, so a fixture that loads immediately proves nothing.
        let mut g = Graph::new();
        let n = g.add_node("_TestNow", None).unwrap();
        let out = OutputProbe::open(&g, n, "out");
        out.wait_until(&mut g, "the session clock to genuinely advance", |d| first_f32(d) > 0.3);

        let doc = g.serialize();
        g.load_doc(&doc).unwrap();
        let loaded = g.node_uids()[0];
        let out = OutputProbe::open(&g, loaded, "out");
        assert!(
            first_f32(&out.expect_frame(&mut g, "the loaded node to run")) < 0.3,
            "a loaded patch starts its clock at zero, whenever it was loaded",
        );
        out.wait_until(&mut g, "and advances from there", |d| first_f32(d) > 0.1);
    }

    #[test]
    fn ufreq_measures_a_capped_source_rate() {
        // The measured rate has to be the node's ACTUAL emit rate, and a rate cap is the one way a
        // test can name a number for it. A tolerance rather than an equality: the meter is an EMA
        // over real intervals and the node's thread is scheduled by the OS.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "common", "max_frequency", Param::float(20.0, 0.0, 1e9)).unwrap();
        let out = OutputProbe::open(&g, src, "out");
        out.wait_until(&mut g, "the meter to settle near the cap", |d| {
            d.meta().ufreq().is_some_and(|hz| (hz - 20.0).abs() < 6.0)
        });
    }

    #[test]
    fn ufreq_is_node_level_same_on_every_slot() {
        // "fast" emits every run; "slow" every other one. ufreq is measured PER NODE, so BOTH slots
        // carry the same number — the slow slot must not report its own halved cadence. Equality
        // rather than a target rate: what this pins is that there is ONE meter, and a per-slot
        // meter differs by a factor of two here whatever the node's real rate turns out to be.
        //
        // Born capped, and read as a RATIO once the meter has settled there: the two slots hold
        // whatever emit each last caught, and an EMA still climbing off a node's first runs differs
        // between any two of them for a reason that has nothing to do with how many meters there
        // are. What a per-slot meter cannot survive is the ratio — it halves the slow slot.
        let mut g = Graph::new();
        let src = capped(&mut g, "_TestTwoRate", 50.0);
        let fast = OutputProbe::open(&g, src, "fast");
        let slow = OutputProbe::open(&g, src, "slow");
        let fast_hz = fast
            .wait_until(&mut g, "the meter to settle at the cap", |d| {
                d.meta().ufreq().is_some_and(|hz| (hz - 50.0).abs() < 10.0)
            })
            .meta()
            .ufreq()
            .unwrap();
        let slow_hz = slow
            .expect_frame(&mut g, "the slow slot to have emitted")
            .meta()
            .ufreq()
            .expect("stamped");
        assert!(
            (fast_hz / slow_hz - 1.0).abs() < 0.25,
            "one meter per node: a per-slot meter halves the slow slot ({slow_hz} vs {fast_hz})",
        );
    }

    #[test]
    fn node_ufreq_exposes_the_measured_rate() {
        // The control-plane accessor the bridge forwards to the node header.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "common", "max_frequency", Param::float(20.0, 0.0, 1e9)).unwrap();
        assert_eq!(g.node_ufreq(src), None, "no rate before the node has reported one");
        wait_for(&mut g, "the reported rate to reach the graph", |g| {
            g.node_ufreq(src).is_some_and(|hz| (hz - 20.0).abs() < 6.0)
        });
    }

    #[test]
    fn ufreq_survives_the_data_plane_wire() {
        // End-to-end through the bridge's exact seam: an engine-stamped frame,
        // encoded as `goofi_codec::encode(latest_frame(..))` (see bridge/lib.rs),
        // carries ufreq across the wire so the browser inspector shows it.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let out = OutputProbe::open(&g, src, "out");
        let frame = out.wait_until(&mut g, "a measured rate", |d| d.meta().ufreq().is_some());
        let measured = frame.meta().ufreq().unwrap();

        let wire = goofi_codec::encode(&frame);
        let back = goofi_codec::decode(&wire).expect("data-plane frame decodes");
        assert_eq!(back.meta().ufreq(), Some(measured), "ufreq round-trips the data plane");
    }

    #[test]
    fn control_input_is_not_an_index_timeline() {
        // A non-triggering "ref" (control) input must NOT drive meta["index"], even
        // when its length coincidentally equals the output's. `ref`'s index is
        // advanced to 3 while the consumer is dormant (its "data" trigger unwired),
        // then a length-4 data frame triggers the consumer, which emits length 1 —
        // matching only the length-1 ref. The output index must be a FRESH 0, not
        // ref's 3 (which a naive length-only match would wrongly propagate).
        let mut g = Graph::new();
        let rs = g.add_node("_TestConst", None).unwrap(); // ref source, len 1
        let ds = g.add_node("_TestConst", None).unwrap();
        g.update_param(ds, "constant", "length", Param::int(4, 1, 10)).unwrap(); // data source, len 4
        let c = capped(&mut g, "_TestRefLenChange", 5.0);
        let refs = OutputProbe::open(&g, rs, "out");
        let out = OutputProbe::open(&g, c, "out");
        g.add_link(rs, "out", c, "ref").unwrap();
        // The ref source runs well past 0 while the consumer is dormant (its `data` trigger is
        // unwired), so a wrongly-propagated ref index is a big number and a fresh one is 0.
        refs.wait_until(&mut g, "the ref source to be well past its first emit", |d| {
            d.meta().index().unwrap_or(0) > 10
        });
        g.add_link(ds, "out", c, "data").unwrap();
        let f = out.expect_frame(&mut g, "the consumer to run");
        assert_eq!(f.meta().index(), Some(0), "control input must not be the timeline");
    }

    #[test]
    fn panicking_node_does_not_crash_the_engine() {
        // Silence the default panic backtrace during this test.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let mut g = Graph::new();
        let boom = g.add_node("_TestPanic", None).unwrap();
        let ok = g.add_node("_TestConst", None).unwrap();
        g.update_param(ok, "constant", "value", Param::float(9.0, -1e9, 1e9))
            .unwrap();
        let out = OutputProbe::open(&g, ok, "out");

        // The panic is captured as the node's error, and the healthy node — on its own thread —
        // keeps running. Neither the graph lock nor the other node's thread may be taken with it.
        wait_for(&mut g, "the panic to be captured as an error", |g| {
            g.last_error(boom).unwrap_or("").contains("panic")
        });
        std::panic::set_hook(prev);
        out.wait_until(&mut g, "the healthy node to keep running", |d| first_f32(d) == 9.0);
    }

    /// A cable between two slots of different dtypes can never carry data — the consumer would
    /// read an empty slot forever. Refuse it at the source, naming both ends AND both dtypes, so
    /// the message teaches which end to change. Protects the canvas as much as an agent.
    #[test]
    fn add_link_refuses_a_dtype_mismatch_and_says_which_ends_disagree() {
        let mut g = Graph::new();
        let echo = g.add_node("_TestEcho", None).unwrap(); // out: ARRAY
        let text = g.add_node("_TestText", None).unwrap(); // words: STRING

        let err = g.add_link(echo, "out", text, "words").unwrap_err();
        for needle in ["_testecho0.out", "ARRAY", "_testtext0.words", "STRING"] {
            assert!(err.contains(needle), "the refusal must name `{needle}`: {err}");
        }
        assert!(!g.has_link(echo, "out", text, "words"), "and nothing was wired");

        // The matching pair still links — the check refuses a mismatch, not a link.
        let echo2 = g.add_node("_TestEcho", None).unwrap();
        g.add_link(echo, "out", echo2, "in").unwrap();
    }

    /// An agent reading a patch has to tell a pipeline that is *settling* (a node that errored
    /// 40 ms ago while its upstream warms up) from one that is *broken* (the same error standing
    /// for a minute). That is only answerable if the engine records WHEN the error appeared.
    #[test]
    fn an_errored_node_records_how_long_its_error_has_stood() {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let mut g = Graph::new();
        let boom = g.add_node("_TestPanic", None).unwrap();
        let ok = g.add_node("_TestEcho", None).unwrap();
        wait_for(&mut g, "the panicking node to report", |g| g.error_age(boom).is_some());
        std::panic::set_hook(prev);

        let first = g.error_age(boom).expect("the error is stamped the moment it appears");
        std::thread::sleep(Duration::from_millis(120));
        g.drain_status();
        let later = g.error_age(boom).expect("still errored");
        assert!(
            later >= first + Duration::from_millis(100),
            "the age is the error's standing, not a constant: {first:?} then {later:?}",
        );
        assert_eq!(g.error_age(ok), None, "a healthy node has no error to age");
    }

    /// …and the clock belongs to the MESSAGE, not to the mere presence of an error. A node that
    /// has been failing one way for a minute and one that just started failing a different way are
    /// different diagnoses, and reporting the older onset for the newer fault would send a reader
    /// looking at whatever changed a minute ago.
    #[test]
    fn a_different_failure_restarts_the_error_clock() {
        struct Changing(usize);
        impl Node for Changing {
            fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
                self.0 += 1;
                Err(format!("failure {}", self.0).into())
            }
        }
        static CHANGING: NodeManifest = NodeManifest {
            type_name: "_TestChangingError",
            category: "test",
            doc: "fails differently every tick",
            inputs: &[],
            outputs: P_OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer: true,
            factory: || Box::new(Changing(0)),
        };
        let mut g = Graph::new();
        g.register_dyn_type(&CHANGING, Box::new(|_| Box::new(Changing(0))));
        // Born at 2 Hz: the failures have to be far enough apart that the FIRST one is still what
        // the graph is holding when the clock is read. Uncapped, this node reaches failure 30000
        // before anything drains, and "failure 1" never appears at all.
        let uid = capped(&mut g, "_TestChangingError", 2.0);
        wait_for(&mut g, "the first failure", |g| g.last_error(uid) == Some("failure 1"));

        // The node reports only TRANSITIONS, so the second message arrives when its complaint
        // changes — and nothing between the two touches the clock.
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(g.last_error(uid), Some("failure 1"), "still the first one, 300 ms on");
        wait_for(&mut g, "the node to fail differently", |g| g.last_error(uid) == Some("failure 2"));
        let age = g.error_age(uid).expect("still errored");
        assert!(
            age < Duration::from_millis(150),
            "a new message is a new error, not the old one still standing: {age:?}",
        );
    }

    // ------------------------------------------------------------------
    // The headline property (spec §1): a node's rate is its own.
    // ------------------------------------------------------------------

    /// Counts every `process()` entry in a cell the test reads. The COUNT is the whole oracle —
    /// asserting that the node "ran" would hold just as well for one run in half a second.
    struct Counting(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl Node for Counting {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out.set("out", Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap());
            Ok(())
        }
    }

    /// Sleeps 50 ms inside `process()` — a device read, a subprocess round-trip, a slow FFT.
    struct Sleeping;
    impl Node for Sleeping {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            std::thread::sleep(Duration::from_millis(50));
            out.set("out", Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap());
            Ok(())
        }
    }

    static COUNTING: NodeManifest = NodeManifest {
        type_name: "_TestCounting",
        category: "test",
        doc: "counts its runs",
        inputs: &[],
        outputs: P_OUT,
        params: NO_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: || Box::new(Counting(Default::default())),
    };
    static SLEEPING: NodeManifest = NodeManifest {
        type_name: "_TestSleeping",
        category: "test",
        doc: "sleeps 50ms per run",
        inputs: &[],
        outputs: P_OUT,
        params: NO_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: || Box::new(Sleeping),
    };

    /// How many times the counting source runs in `window`, alone or beside the sleeper. The two
    /// nodes are NEVER linked: whatever one does to the other travelled through the scheduler.
    fn runs_in(window: Duration, with_sleeper: bool) -> usize {
        let runs: std::sync::Arc<std::sync::atomic::AtomicUsize> = Default::default();
        let mine = runs.clone();
        let mut g = Graph::new();
        g.register_dyn_type(&COUNTING, Box::new(move |_| Box::new(Counting(mine.clone()))));
        g.register_dyn_type(&SLEEPING, Box::new(|_| Box::new(Sleeping)));
        g.add_node("_TestCounting", None).unwrap();
        if with_sleeper {
            g.add_node("_TestSleeping", None).unwrap();
        }
        // Nothing to drive: each node runs itself on its own thread. What the test measures is the
        // rate that arrangement gives the counter, so the measurement is a wall-clock window and
        // the graph is simply alive for it.
        std::thread::sleep(window);
        runs.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[test]
    fn a_slow_node_does_not_throttle_an_unrelated_one() {
        // THE headline property. Measured on the old runtime: 18411 runs alone, 10 beside a 50ms
        // node with no link between them — a 1841x throttle, because rayon's per-level join barrier
        // and the graph mutex made every node inherit the slowest one's rate.
        let alone = runs_in(Duration::from_millis(500), false);
        let beside = runs_in(Duration::from_millis(500), true);
        assert!(beside as f64 / alone as f64 > 0.5,
                "an unrelated node must keep its rate: {beside} vs {alone}");
    }
}
