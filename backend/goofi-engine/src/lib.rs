//! The graph, and the nodes that schedule themselves.
//!
//! A node's DATA never comes back here: frames go out on that node's own shared-memory service and
//! a consumer subscribes to it, so there is no last-output cache and no privileged path in.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;

use goofi_core::Param;
use goofi_node::{
    ExprMode, NodeCtx, NodeManifest, ParamGroups, ParamKey,
};
use indexmap::IndexMap;

pub mod archive;

pub mod subpatch;
pub mod layout;

pub mod command;
pub use command::{Command, CommandHistory, ExprState, Outcome};

pub mod expr_rewrite;

/// Public because a host needs the wire vocabulary: service names, [`runtime::Status`] to drain,
pub mod runtime;
pub mod testing;

/// A `u64` internally, a 12-hex string in the `.gfi` and on the wire.
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

/// One literal for the writer, the reader and the refusal message, so a bump cannot leave the
/// message lying about what this build reads.
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
    /// `None` when this node's services could not be created — it then exists in the patch
    /// carrying its boot error and nothing else.
    channel: Option<Arc<runtime::NodeChannel>>,
}

impl NodeHost {
    /// Ask the thread to stop, and ring it so a parked node notices now rather than at the end of
    /// its park. Never joined here: the thread may be inside a long `process()`, and both callers
    /// run under the graph mutex the bridge holds.
    fn signal_stop(&self) {
        self.halt.stop();
        if let Some(channel) = &self.channel {
            channel.wake();
        }
    }
}

impl Drop for NodeHost {
    fn drop(&mut self) {
        self.signal_stop();
    }
}

struct NodeEntry {
    manifest: &'static NodeManifest,
    host: NodeHost,
    /// The param RECORD — literals and, through `bindings`, expression source. An `Arc` so a
    /// reader gets a cheap owned snapshot; every write happens under the graph mutex, so nothing
    /// shares it mutably. The node's own thread holds an unrelated copy, fed by the `SetParam`
    /// every write announces: an evaluated value must not reach this record, because this is what
    /// `serialize` writes.
    params: Arc<ParamGroups>,
    /// The graph resolves each binding's references and ships it; the NODE evaluates it.
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
    /// The same number the node stamps as `meta["ufreq"]`. `None` until it has emitted twice.
    ufreq: Option<f64>,
        name: String,
        pos: [f64; 2],
    /// Opaque: persisted and round-tripped, never interpreted.
    viewers: serde_json::Value,
}

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
/// Run a node lifecycle hook or `process`, converting a panic into an error string. A node is
/// third-party code — a native crate registered through `inventory`, or a `.py` the user just
/// edited — and every entry point into it goes through here.
///
/// What the guard buys MOVED with the cutover, and the old reason is worth keeping straight: these
/// hooks used to run under the graph mutex the bridge holds, where an unguarded panic poisoned it
/// and took the control plane down permanently. They now run on the node's own thread, which the
/// mutex never reaches — so what a panic costs today is that THREAD, silently and for good, and
/// the node stops answering anything for the rest of the session. The containment is the same; the
/// failure it prevents is not.
fn guard_lifecycle<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(panic_message)
}

/// So a panic and a returned error travel the one channel a caller already handles.
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
/// A param's VALUE as JSON — the inverse of [`param_from_json`], and the one definition of it.
///
/// `fire_triggers` decides what a fired momentary trigger reads as, and the two callers want
/// opposite answers: a PERSISTED value (`serialize`, `restart_node`) must never record a trigger as
/// fired, or reloading the patch would fire it again, while a UI PROJECTION (the param descriptors
/// and the CRDT mirror) must show the live state. The flag is named for its inverse's, which gates
/// the same distinction on the way back in.
///
/// Two copies of this used to exist — this one and `goofi_bridge::schemas`' — under the same name,
/// in two crates, differing in exactly that arm and nothing else.
pub fn param_value_json(p: &Param, fire_triggers: bool) -> serde_json::Value {
    use serde_json::json;
    match p {
        Param::Float { value, .. } => json!(value),
        Param::Int { value, .. } => json!(value),
        Param::Bool { value } => json!(value),
        Param::Trigger { fired } => json!(fire_triggers && *fired),
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
    /// The `(node, param)` pairs whose options a node has re-enumerated since anyone last looked
    /// ([`Graph::take_refreshed`]). Options are the one thing a node reports that the CRDT doc has
    /// no field for, so the status worker has to be TOLD which params to echo rather than being
    /// able to find out on the re-mirror.
    refreshed: Vec<(Uid, ParamKey)>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// How long [`Graph::shutdown`] waits for its node threads to release their shared memory. A node
/// looks at its halt flag at every wake, so the usual cost is well under `PARK_CEILING`; the
/// ceiling is for one inside a long `process()`. A ceiling and not a join, because a wedged node
/// must not be able to wedge the exit — the leftovers of one that misses it are exactly what
/// [`runtime::reclaim_stale_resources`] takes on the next startup.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(2);

impl Drop for Graph {
    /// Release this graph's shared memory before the process can leave without it. An iceoryx2 node
    /// removes its segments when it DROPS, and a node's transport is owned by the node's own
    /// thread — so raising the halt flags and returning leaves every segment allocated if the
    /// process exits before the threads notice. Measured after one such run: 82 leaked
    /// `/dev/shm/iox2_*` files holding 4.7 GB, and shared memory counts against RAM.
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Write a node's param record in place. `Arc::make_mut` clones only while a reader still holds
/// the previous snapshot, so a caller that took one keeps a consistent version and an unshared
/// record is edited without a copy at all.
fn edit_params(entry: &mut NodeEntry, edit: impl FnOnce(&mut ParamGroups)) {
    edit(Arc::make_mut(&mut entry.params));
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
        // Reference goofi-nodes so the linker keeps its inventory registrations. This is the ONLY
        // anchor: the bridge carried a second one, and dropping it left the catalog intact.
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
            refreshed: Vec::new(),
        }
    }

    // ── Globals ─────────────────────────────────────────────────────────────────────────────
    // Patch-scoped named scalars. System globals (`default_ufreq`) are seeded + delete-protected;
    // user globals are add/edit/remove/rename. Read by expressions (`globals.<name>`) + node ctx.

    /// Stop every node and WAIT for each to release its shared memory, up to [`SHUTDOWN_WAIT`].
    ///
    /// The waiting is the whole point, and it is why this is not what `clear()` does: a load
    /// destroys nodes under the graph mutex the bridge holds, and the segments of a node that has
    /// not noticed yet are released a moment later by its own thread. Only a process about to EXIT
    /// has no "a moment later".
    pub fn shutdown(&mut self) {
        for entry in self.nodes.values() {
            entry.host.signal_stop();
        }
        let deadline = Instant::now() + SHUTDOWN_WAIT;
        while self.nodes.values().any(|e| !e.host.halt.released()) {
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        // The graph's OWN end of each node goes here too: `NodeChannel` holds an iceoryx2 node of
        // its own — and the wire planner keeps a SECOND handle on it, which is why `remove_node`
        // forgets a uid there as well as dropping its entry. Clearing the entries alone released a
        // node's own transport and kept the graph's half of it for the process lifetime.
        self.nodes.clear();
        self.wire.reset_channels();
    }

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
    /// The node's initialization failure wins, then a process error (`last_error`), then the
    /// errored param with the smallest `ParamKey` — a deterministic pick, since
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
    /// without services there is no status service to carry a fault — so it is written straight
    /// into the entry's `setup_error`. Everything after it, `setup()` included,
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
                params: Arc::new(params),
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

    /// A node's params as of now. An owned snapshot rather than a borrow: cloning the `Arc` is
    /// cheap, and a `&` would borrow the whole graph for as long as the caller held it.
    pub fn params(&self, uid: Uid) -> Option<Arc<ParamGroups>> {
        self.nodes.get(&uid).map(|e| e.params.clone())
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

    /// One stub of a scope, mutable. Six near-identical setters used to sit here — insert, remove,
    /// set-dtype, rename, move, restore-inner — each re-doing this lookup and each minting its own
    /// error string. Those strings were unreachable: every `Command` that edits a boundary already
    /// guards on the stub existing and answers an idempotent no-op when it does not, so this
    /// returns `None` and lets the caller keep the guard it already has.
    pub fn stub_mut(&mut self, scope: Uid, stub: &str) -> Option<&mut subpatch::Stub> {
        self.scopes.get_mut(&scope)?.stubs.get_mut(stub)
    }

    /// A scope's whole stub map, for the two edits that are map-shaped rather than field-shaped.
    /// Deliberately NOT `&mut Scope`: that would hand out `Scope::name` and `Scope::pos` as well,
    /// and let a caller bypass the validated rename path.
    pub fn stubs_mut(&mut self, scope: Uid) -> Option<&mut IndexMap<subpatch::StubId, subpatch::Stub>> {
        self.scopes.get_mut(&scope).map(|s| &mut s.stubs)
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
        // The planner holds its OWN handle on this node's channel, and that handle is the graph's
        // end of its services. Dropping the entry alone leaves them allocated for the process
        // lifetime, where no sweep can reach them. `forget` rather than `detach`: this uid is
        // retired, so anything still queued for it addresses nobody.
        self.wire.forget(uid);
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
        let held = entry.params.clone();
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
                    *slot = param_from_json(slot, &param_value_json(value, false), false);
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
        // The corpse's channel goes with it, and it must go BEFORE the new generation reports
        // `Ready`: while it stands, the reborn node reads as addressable and anything sent to it is
        // published to services nothing is listening on any more.
        self.wire.detach(uid);
        // A swap, not a new record: the graph's readers hold this very handle, so replacing it
        // would leave them reading the corpse's params.
        entry.params = Arc::new(params);
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
        if entry.params.get(group).is_none() {
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
    /// It answers only that the request was DISPATCHED (§8.5). The hook runs on the node's own
    /// thread, which is the whole point of the move — a multi-second device scan no longer stalls
    /// anything — so the caller cannot carry the list back. The options arrive later as
    /// [`runtime::Status::RefreshOptions`], land in the param record, and reach the client on the
    /// echo the status worker drives off [`Graph::take_refreshed`]. `Err` is still a real refusal:
    /// an unknown node, an unknown param, or one the type never declared refreshable (the UI shows
    /// no button for one).
    ///
    /// Not a command: nothing persisted changes (options never reach the `.gfi`), so there is
    /// nothing to undo.
    pub fn refresh_param(&mut self, uid: Uid, group: &str, name: &str) -> Result<(), String> {
        let entry = self.nodes.get(&uid).ok_or_else(|| format!("no such node {uid}"))?;
        let live = entry.params.clone();
        let param = goofi_node::param(&live, group, name)
            .ok_or_else(|| format!("no such param `{group}.{name}`"))?;
        if !matches!(param, Param::Str { refresh: true, .. }) {
            return Err(format!("param `{group}.{name}` is not refreshable"));
        }
        self.wire.send(uid, runtime::Control::RefreshParam { key: ParamKey::new(group, name) });
        Ok(())
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
        if goofi_node::param(&self.nodes[&uid].params, group, name).is_none() {
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
    // Every link and binding change replans through it, and a node is addressable from the moment
    // it reports `Ready` — [`Graph::attach_control_sink`] — until the graph destroys it.

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
        // Set by the `RefreshOptions` arm and drained after the match: `entry` holds a mutable
        // borrow of `self` for the whole of it, so the queue cannot be pushed to from inside.
        let mut refreshed: Option<ParamKey> = None;
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
                // Queued whether or not there were any: this IS the answer to a ⟳, and the client
                // that asked lifts its spinner off the echo of it. A node with no hook for the
                // param answers `None`, and would otherwise spin until a 15 s safety timeout.
                refreshed = Some(key);
            }
            runtime::Status::Fault { fault } => match fault {
                // A clean run clears Setup/Process/Boot together and never touches a binding
                // error, which only that binding evaluating successfully clears (§6).
                None => {
                    entry.setup_error = None;
                    entry.last_error = None;
                }
                Some(runtime::NodeFault::Setup { msg, .. }) => entry.setup_error = Some(msg),
                Some(runtime::NodeFault::Process { msg, .. }) => entry.last_error = Some(msg),
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
        if let Some(key) = refreshed {
            self.refreshed.push((uid, key));
        }
        self.stamp_error_onset(uid);
    }

    /// Take the `(node, param)` pairs whose options have been re-enumerated since the last call —
    /// the status worker's cue to echo them.
    ///
    /// A queue rather than something derivable on the re-mirror, because a param's OPTIONS are the
    /// one part of a node the CRDT doc has no field for (they are runtime-only and never persisted,
    /// so they reach a client as a `state_update` and no other way). Draining it here is what makes
    /// each answered ⟳ echo exactly once.
    pub fn take_refreshed(&mut self) -> Vec<(Uid, ParamKey)> {
        std::mem::take(&mut self.refreshed)
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
                runtime::ParamValue::Literal(goofi_node::param(&entry.params, &key.group, &key.name)?.clone())
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
    /// connection order, and the per-wire cells the NODE keeps are set from this same list by the
    /// `InSlot` a re-plan sends. Reading those cells here instead would be a second record of one
    /// order, on the far side of a wire.
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
        // An un-echoed refresh names a node the patch no longer holds; a fresh patch may well mint
        // that uid again (a load restores the uids it was saved with), and the echo would then be
        // read as an answer about a param nobody asked about.
        self.refreshed.clear();
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
            let live = e.params.clone();
            for (group, names) in &*live {
                let mut gmap = Map::new();
                for (name, p) in names {
                    gmap.insert(name.clone(), param_value_json(p, false));
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
            // literal — the reseed would then clobber the saved literal on the node's next run. The doc's
            // own `expressions` block (restored below) round-trips every binding the user actually has.
            //
            // The saved params are folded in BEFORE construction, because `insert_node` runs the
            // node's `setup()` — a one-time init that reads its params (allocate a buffer of `size`,
            // open device `name`) and never runs again. Applying them afterwards would boot every
            // node against the type's defaults and leave the saved values to arrive as a `SetParam`
            // AFTER `setup()` had already read the wrong ones. This is the same order the undo/redo
            // restore path uses (`Command::AddNode` carries the captured params).
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

/// One node's current error, derived fresh from the places one can arise — see
/// [`Graph::last_error`], whose contract this is. A free function so [`Graph::stamp_error_onset`]
/// can read it while holding a `&mut NodeEntry`, which keeps derivation and stamping on one rule.
fn entry_error(e: &NodeEntry) -> Option<&str> {
    // The node's initialization failure outranks a process error, and D3 makes it the only thing
    // that CAN be true beside one: if `setup` failed, `process` never runs. The order therefore
    // encodes which of the two is possible, not which one wins a contest. A node whose services
    // could not be created carries its boot failure here too — it is the same "this node never
    // started" fact, one layer further out.
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

/// How long a NODE waits between retries of a failed initialization (D3). The retry is the whole
/// [`seed_node`] unit — every param's `on_param_changed` plus `setup()` — and it runs on that
/// node's own thread, at its own wake rate. A `setup()` that fails is exactly the kind that BLOCKS
/// first (opening a device, dialling a socket) and the kind that leaks a handle per attempt, since
/// `Drop` never fires between them; a free-running producer would attempt tens of times a second.
/// One second bounds both, and is still well inside the time a user watching the node would wait
/// for it to heal. Only a WAKE is paced: a param edit is a user asking, and retries at once.
const SETUP_RETRY_INTERVAL: f64 = 1.0;

