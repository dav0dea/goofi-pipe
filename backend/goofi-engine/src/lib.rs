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

use goofi_core::{Data, Param};
use goofi_node::{
    ExprMode, Inputs, NodeCtx, NodeManifest, Outputs, ParamGroups, ParamKey, Params, RunPolicy,
};
use indexmap::IndexMap;
use rayon::prelude::*;

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
mod detached;

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

/// EMA weight for the measured update-frequency (`ufreq`). Smooths the inter-emit
/// interval: time-constant ≈ `1/α` emits, so a steady slot reads exact from its 2nd
/// emit and a jittery one settles within ~10–15. Tunable in this one place.
const UFREQ_EMA_ALPHA: f64 = 0.2;

/// The `.gfi` manifest version: written by [`Graph::serialize`], the sole version
/// [`Graph::load_doc`] accepts, and the number its refusal quotes. One literal for all three, so a
/// bump cannot leave the error message lying about what this build actually reads.
const MANIFEST_VERSION: i64 = 7;

/// Per-NODE measured emit-rate state (see [`stamp_meta`]). Tracks the wall-clock
/// (`ctx.now`) of the node's previous productive emit and the smoothed inter-emit
/// interval; `ufreq = 1/ema`. `last_emit == None` until the first emit, `ema == None`
/// until the second gives one interval to seed it.
#[derive(Default)]
pub(crate) struct UfreqMeter {
    last_emit: Option<f64>,
    ema: Option<f64>,
}

/// One wire feeding a `multi` input slot: its source `(uid, out-slot)` identity and
/// that wire's latest-wins frame (`None` until it first emits).
type WireCell = (Uid, &'static str, Option<Data>);

/// How a node's `process()` is executed. An `Isolation::InProcess` node runs inline on
/// the tick's rayon pool (`Inline`); an `Isolation::Subprocess` node runs on a dedicated
/// off-tick worker (`Detached`) so a blocking backend can't stall
/// the tick or the graph lock. The tick decides *whether* every node runs identically —
/// only the execution site differs.
enum Execution {
    Inline(Box<dyn goofi_node::Node>),
    Detached(detached::DetachedHandle),
}

struct NodeEntry {
    manifest: &'static NodeManifest,
    exec: Execution,
    params: ParamGroups,
    inputs: IndexMap<&'static str, Option<Data>>,
    /// Per-wire latest-wins cells for each `multi` input slot, in connection order:
    /// `(src_uid, src_slot) -> latest frame`. Engine-owned; materialized to an ordered
    /// present-only `&[Data]` for the node at run time. Single slots live in `inputs`;
    /// the two maps partition the manifest's input slots (a slot is single XOR multi).
    multi_inputs: IndexMap<&'static str, Vec<WireCell>>,
    outputs: IndexMap<&'static str, Option<Data>>,
    /// The last frame this node EMITTED on each slot, persisted across ticks where it
    /// emitted nothing. `outputs` is reset to `None` every tick for emit detection +
    /// propagation, so viewers (`latest_frame`) read this instead — a sparse /
    /// wall-clock-paced producer (e.g. Oscillator ticked faster than its sample rate)
    /// keeps showing its latest data rather than blinking to None on silent ticks.
    last_outputs: IndexMap<&'static str, Data>,
    /// Param-expression bindings on this node, keyed by `(group, name)`. The engine
    /// resolves them into `params` before the node runs; the node never sees them.
    bindings: HashMap<ParamKey, ExprBinding>,
    ctx: NodeCtx,
    /// `Some(msg)` when this node's INITIALIZATION failed — the `on_param_changed` replay and
    /// `setup()` together, which are one unit: a node that did not finish either never
    /// initialized, so nothing may run against it (D3, [`ensure_initialized`]). Deliberately NOT
    /// `last_error`, which [`execute_node`] overwrites every run — that is how a bootstrap failure
    /// used to erase itself on the node's first clean tick.
    setup_error: Option<String>,
    /// The tick clock (`ctx.now`) as of this node's last INITIALIZATION attempt — the backoff
    /// [`run_node`] paces its retry by (see [`SETUP_RETRY_INTERVAL`]). Every attempt stamps it,
    /// construction's included; only the tick reads it, so an explicit interaction still retries
    /// at once. Meaningless (and never read) while `setup_error` is `None`.
    last_setup_attempt: f64,
    last_error: Option<String>,
    /// The message [`Graph::last_error`] last derived for this node, and WHEN it first read that
    /// way. Re-stamped only when the message changes, so the instant is the error's *onset* — the
    /// difference between a pipeline settling and one that is broken. Swept once per tick from the
    /// derived error rather than written at each of the three places an error can arise (a process
    /// failure, a binding, a detached bootstrap), so the two can never disagree about what the
    /// node's error is.
    error_since: Option<(String, Instant)>,
    /// Globally-unique display name (type-numbered), for the frontend/`.gfi`.
    name: String,
    /// Editor position `[x, y]`.
    pos: [f64; 2],
    /// Per-slot viewer view-state (chosen kind + settings + collapsed), an OPAQUE JSON
    /// blob the backend persists and round-trips but never interprets — view-state is
    /// cross-cutting UI state, not pillar logic. Empty object until the editor sets it.
    viewers: serde_json::Value,
    /// Whether this node has any triggering input (else it free-runs each tick).
    has_trigger_inputs: bool,
    /// Set when a triggering input received a fresh frame; cleared on process.
    trigger_pending: bool,
    /// Per-output-slot source-origin emit counter for `meta["index"]`. Advanced
    /// only when a slot's frame starts a *fresh* timeline (a generator, or a
    /// length-changing transform); a length-preserving emit mirrors its matching
    /// input's index instead. Engine-owned — the node never sees it.
    index_counters: HashMap<&'static str, u64>,
    /// Per-NODE measured update-rate state for `meta["ufreq"]`. Engine-owned; advanced
    /// once per productive tick (a tick emitting ≥1 output). The single node-level rate
    /// is stamped onto every output slot's meta — ufreq describes the node, not a slot.
    ufreq_meter: UfreqMeter,
    /// The node's run gate (from its `common` params), consulted each tick.
    run_policy: RunPolicy,
    /// Wall-clock instant the node last ran, for rate-cap gating (`None` = never).
    last_run: Option<Instant>,
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
    source: String,
    enabled: bool,
    triggers_process: bool,
    /// Compiled handle owned by the evaluator (`None` if compile failed / no evaluator).
    id: Option<goofi_node::BindingId>,
    /// Statically-extracted `nd()` node names (empty for a ref-less/time expression).
    refs: Vec<String>,
    /// Statically-extracted `globals.<name>` reads, so a change to one of those globals forces a
    /// re-eval of this binding (and only these) on the next tick — see [`Graph::apply_global_change`].
    global_refs: Vec<String>,
    /// The referenced producers' emit `index` seen at the last eval, for the dirty check.
    last_seen: HashMap<(String, Option<String>), Option<u64>>,
    /// Wall-clock of the last eval, for the per-node `max_frequency` eval gate.
    last_eval: Option<Instant>,
    /// The current expression error (field indicator), or `None` when healthy.
    error: Option<String>,
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
    /// Wall-clock reference, anchored at the first tick, so `NodeCtx::now` is
    /// seconds-since-start (deterministic under an injected clock).
    start: Option<Instant>,
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
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// The present frames on each `multi` input slot, in wire order — the shape both execution
/// tiers hand a node's `process` (inline here, or packed into a detached [`detached::Job`]).
/// Absent wires are dropped, so a node sees only the frames that actually arrived.
fn materialize_multis(entry: &NodeEntry) -> IndexMap<&'static str, Vec<Data>> {
    entry
        .multi_inputs
        .iter()
        .map(|(k, cells)| (*k, cells.iter().filter_map(|(_, _, o)| o.clone()).collect()))
        .collect()
}

/// The "this node wants to run" predicate, shared by the tick's two execution sites and the
/// pacer that decides how long to sleep before the next one. A pure source free-runs; a fresh
/// trigger fires; `autotrigger` free-runs a node only when it has no *WIRED* trigger input
/// (Python parity). The three callers MUST agree — the pacer sets the sleep while the other two
/// decide who runs, so a term added to one alone means spinning hot or sleeping through work.
fn wants_run(e: &NodeEntry, uid: &Uid, wired: &std::collections::HashSet<Uid>) -> bool {
    e.trigger_pending || !e.has_trigger_inputs || (e.run_policy.autotrigger && !wired.contains(uid))
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
            start: None,
            evaluator: None,
            scopes: IndexMap::new(),
            scope_of: HashMap::new(),
            globals: goofi_core::globals::GlobalStore::new(),
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
    /// are rejected. Every expression binding that reads this global is forced to re-evaluate on the
    /// next tick, so a producer bound to `globals.default_ufreq` re-rates live — and only those
    /// bindings pay (an unrelated global edit touches nothing). Resetting `last_eval` opens both the
    /// due check and the per-node eval-rate gate, giving exactly one immediate re-eval.
    pub fn apply_global_change(
        &mut self,
        name: &str,
        value: Option<goofi_core::globals::GlobalValue>,
    ) -> Result<(), String> {
        self.globals.apply_change(name, value)?;
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
        self.globals.add_at(name, value, at)?;
        self.invalidate_bindings_reading(name);
        Ok(())
    }

    /// Force every expression binding that reads global `name` to re-evaluate on the next tick, so a
    /// producer bound to it re-rates live (only those bindings pay). Shared by the global mutators.
    fn invalidate_bindings_reading(&mut self, name: &str) {
        for entry in self.nodes.values_mut() {
            for b in entry.bindings.values_mut() {
                if b.global_refs.iter().any(|g| g == name) {
                    b.last_eval = None;
                }
            }
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
    /// Only the DETACHED tier has a real bootstrap to report — an inline node is seeded
    /// synchronously before it is ever visible, so it is `ready` (or `error`) from the first
    /// frame the editor sees. A detached node's `setup()` runs on its worker and a Python child
    /// is spawn + import + setup, which is where the spinner earns its keep.
    pub fn node_stage(&self, uid: Uid) -> &'static str {
        let Some(entry) = self.nodes.get(&uid) else { return "error" };
        // Both error channels, or a node whose `setup()` failed would draw as `ready` — the
        // uninitialized state is precisely the one the editor must not paint healthy.
        if entry.setup_error.is_some() || entry.last_error.is_some() {
            return "error";
        }
        match &entry.exec {
            Execution::Inline(_) => "ready",
            Execution::Detached(h) => match h.stage() {
                detached::STAGE_CREATING => "creating",
                detached::STAGE_SETUP => "setup",
                // A bootstrap failure is latched on the handle, never in `entry.last_error`
                // (see [`Graph::last_error`]), so the bootstrapped arm must consult it — else a
                // worker whose `setup` failed draws as healthy.
                _ if h.boot_error().is_some() => "error",
                _ => "ready",
            },
        }
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
        self.insert_node_at(uid, name, manifest, node, params);
        if uid.0 >= self.next_uid {
            self.next_uid = uid.0 + 1;
        }
        if seed {
            self.seed_default_expressions(uid, manifest);
        }
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
        for decl in manifest.params {
            if let Some(e) = decl.expression {
                let enabled = matches!(e.mode, ExprMode::On);
                let _ = self.set_expression(uid, decl.group, decl.name, e.source, enabled, e.trigger);
            }
        }
    }

    /// Insert a constructed node at a SPECIFIC uid + display name — the reconcile path, which
    /// spawns sub-patch members at their deterministic uids. The uid must be free.
    fn insert_node_at(
        &mut self,
        uid: Uid,
        name: String,
        manifest: &'static NodeManifest,
        node: Box<dyn goofi_node::Node>,
        params: ParamGroups,
    ) {
        let mut ctx = NodeCtx::new();
        // `setup` latches the globals as of insert time (`process` reads them live each tick).
        ctx.globals = self.globals.snapshot();

        let inputs: IndexMap<&'static str, Option<Data>> =
            manifest.inputs.iter().filter(|s| !s.multi).map(|s| (s.name, None)).collect();
        let multi_inputs: IndexMap<&'static str, Vec<WireCell>> =
            manifest.inputs.iter().filter(|s| s.multi).map(|s| (s.name, Vec::new())).collect();
        let outputs = manifest.output_buffer();

        let has_trigger_inputs = manifest.inputs.iter().any(|i| i.trigger_process);
        let run_policy = RunPolicy::from_params(&params);

        let (exec, setup_error) = make_exec(manifest, node, &params, &mut ctx);
        // Construction IS the first initialization attempt, so it starts the retry backoff — else
        // the very next tick would re-run a `setup()` that has just failed.
        let last_setup_attempt = ctx.now;

        self.nodes.insert(
            uid,
            NodeEntry {
                manifest,
                exec,
                params,
                inputs,
                multi_inputs,
                outputs,
                last_outputs: IndexMap::new(),
                bindings: HashMap::new(),
                ctx,
                setup_error,
                last_setup_attempt,
                last_error: None,
                error_since: None,
                name,
                pos: [0.0, 0.0],
                viewers: serde_json::json!({}),
                has_trigger_inputs,
                trigger_pending: false,
                index_counters: HashMap::new(),
                ufreq_meter: UfreqMeter { last_emit: None, ema: None },
                run_policy,
                last_run: None,
            },
        );
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

    pub fn params(&self, uid: Uid) -> Option<&ParamGroups> {
        self.nodes.get(&uid).map(|e| &e.params)
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
        Ok(self.rewrite_nd_refs_for_rename(&old_name, name))
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
        // Drop any membership tag: a removed node has no scope. Leaving it dangling would make a
        // reused uid (a delete→undo that restores the scope) self-parent via `common_parent`.
        self.scope_of.remove(&uid);
        // Drop links touching the node; clear any downstream input it fed.
        let dropped: Vec<Link> = self
            .links
            .iter()
            .filter(|l| l.node_out == uid || l.node_in == uid)
            .cloned()
            .collect();
        self.links
            .retain(|l| l.node_out != uid && l.node_in != uid);
        for l in dropped {
            // Purge the removed node's wire from a downstream multi slot; else clear
            // the single input it fed. (Links into the removed node itself no-op —
            // its entry is already gone.)
            if self.is_multi_input(l.node_in, l.slot_in) {
                self.drop_multi_wire(l.node_in, l.slot_in, l.node_out, l.slot_out);
            } else {
                self.clear_input(l.node_in, l.slot_in);
            }
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
        for (group, held) in &held {
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
        let mut ctx = NodeCtx::new();
        ctx.globals = self.globals.snapshot();
        let (exec, setup_error) = make_exec(manifest, node, &params, &mut ctx);

        // The per-wire cells live in the entry while the wires themselves live on the graph, so
        // they must be rebuilt from `links` (in connection order) — a fresh empty map would
        // leave the multi slot silently dead with every link still shown in the editor. Each
        // wire KEEPS the frame it was last given: these cells are latest-wins caches, not
        // instance state, and dropping them stalls the node until every upstream happens to emit
        // again (for a slow or rate-capped producer, potentially a very long time).
        let held = &self.nodes.get(&uid).expect("looked up above").multi_inputs;
        let multi_inputs: IndexMap<&'static str, Vec<WireCell>> = manifest
            .inputs
            .iter()
            .filter(|s| s.multi)
            .map(|s| {
                let wires = self
                    .links
                    .iter()
                    .filter(|l| l.node_in == uid && l.slot_in == s.name)
                    .map(|l| {
                        let frame = held
                            .get(s.name)
                            .and_then(|cells| {
                                cells.iter().find(|(u, slot, _)| *u == l.node_out && *slot == l.slot_out)
                            })
                            .and_then(|(_, _, frame)| frame.clone());
                        (l.node_out, l.slot_out, frame)
                    })
                    .collect();
                (s.name, wires)
            })
            .collect();

        let entry = self.nodes.get_mut(&uid).expect("looked up above");
        // Dropping the old instance never waits: a `DetachedHandle`'s Drop only signals its
        // worker, which reaps itself off this thread.
        //
        // The MANIFEST goes with the instance. It is the graph's whole description of this node —
        // link validation, schema projection, `/data` target checks and the scheduler's trigger
        // policy all read it — and the rescan path re-registers a stable `type_name` over a
        // possibly-reshaped interface. Keeping the old one here (which this did until the
        // boundary-hardening pass) left the graph describing a node that is no longer running:
        // a slot the edit added was unlinkable, and one it removed still accepted wires.
        entry.manifest = manifest;
        entry.exec = exec;
        entry.params = params;
        // Manifest-derived caches, REBUILT rather than carried. A slot that survived the reshape
        // by name keeps its last frame (a live graph should not blink — same rationale as the
        // multi cells above); one that did not is dropped, because its `&'static str` key no
        // longer names anything the node will read.
        let mut prior = std::mem::take(&mut entry.inputs);
        entry.inputs =
            manifest.inputs.iter().filter(|s| !s.multi).map(|s| (s.name, prior.shift_remove(s.name).flatten())).collect();
        entry.multi_inputs = multi_inputs;
        entry.outputs = manifest.output_buffer();
        entry.last_outputs.retain(|slot, _| manifest.outputs.iter().any(|o| o.name == *slot));
        entry.has_trigger_inputs = manifest.inputs.iter().any(|i| i.trigger_process);
        entry.ctx = ctx;
        entry.setup_error = setup_error;
        entry.last_setup_attempt = entry.ctx.now;
        // A fresh instance carries none of the corpse's failures — its predecessor's process error
        // describes a node that no longer exists.
        entry.last_error = None;
        entry.trigger_pending = false;
        entry.ufreq_meter = UfreqMeter { last_emit: None, ema: None };
        entry.run_policy = RunPolicy::from_params(&entry.params);
        entry.last_run = None;
        // `index_counters` deliberately CARRY OVER: `meta["index"]` is a stream-position counter,
        // and restarting it at 0 would regress the index downstream consumers dirty-check on.
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
        if let Some(g) = entry.params.get_mut(group) {
            g.insert(name.to_string(), value.clone());
        } else {
            return Err(format!("no such param group `{group}`"));
        }
        // The `common` group is scheduler metadata, not a node param — re-derive
        // the cached run gate rather than dispatching it to the node.
        if group == "common" {
            entry.run_policy = RunPolicy::from_params(&entry.params);
            return Ok(());
        }
        // D3: the new value is stored ABOVE, so the retry's replay delivers it — correcting the
        // param that broke `setup()` is what re-initializes the node. Two consequences, both
        // deliberate. A SUCCESSFUL retry has already handed this edit to `on_param_changed`, so
        // calling it again below would double-apply the handler's side effect. And a FAILED retry
        // still answers `Ok`: the node's failure belongs on its error channel, not in this reply —
        // returning `Err` would refuse the very edit that is the retry door, and `update_param` is
        // a command whose inverse must stay in step with the session's history.
        if entry.setup_error.is_some() {
            let _ = ensure_initialized(entry);
            return Ok(());
        }
        match &mut entry.exec {
            Execution::Inline(node) => {
                guard_lifecycle(|| node.on_param_changed(&ParamKey::new(group, name), &value))
                    .and_then(|r| r.map_err(|e| e.0))
            }
            // A detached node's instance lives on its worker; the edit is stored in
            // `entry.params` and rides the next Job's cold read, which is how a param reaches
            // the subprocess tier at all. Nothing is lost by not forwarding the notification:
            // `RemoteNode` implements no `on_param_changed`.
            Execution::Detached(_) => Ok(()),
        }
    }

    /// Re-enumerate a refreshable `Str` param's options by asking the node — the ⟳ button behind
    /// a device or stream picker, whose choices are only knowable at runtime. Returns the fresh
    /// list, or `None` when the node declares the param refreshable but implements no hook.
    ///
    /// A refresh never changes the SELECTION: it rewrites `options` and nothing else, so a device
    /// that has disappeared stays selected (the UI keeps showing it) rather than silently
    /// re-pointing the node at a different one.
    ///
    /// Not a command: nothing persisted changes (options never reach the `.gfi` or the doc), so
    /// there is nothing to undo.
    ///
    /// NOTE: this runs under the graph lock, so a node that blocks here (a slow device scan, an
    /// LSL resolve) stalls the tick for that long. Node authors must keep the hook quick.
    pub fn refresh_param(
        &mut self,
        uid: Uid,
        group: &str,
        name: &str,
    ) -> Result<Option<Vec<String>>, String> {
        let entry = self.nodes.get_mut(&uid).ok_or_else(|| format!("no such node {uid}"))?;
        let param = entry
            .params
            .get(group)
            .and_then(|g| g.get(name))
            .ok_or_else(|| format!("no such param `{group}.{name}`"))?;
        // Refreshing a param the node never declared refreshable would call a hook it does not
        // implement and report success — reject it instead (the UI shows no button for one).
        if !matches!(param, Param::Str { refresh: true, .. }) {
            return Err(format!("param `{group}.{name}` is not refreshable"));
        }
        // D3: a refresh is an interaction, so it retries the initialization first — a picker whose
        // node failed `setup()` rescans as soon as that node comes up. If it does not, this DOES
        // refuse: the call answers the UI with a list and there is none, and `Ok(None)` would read
        // as "this node implements no hook", which is a different and misleading answer.
        if let Err(e) = ensure_initialized(entry) {
            return Err(format!("`{group}.{name}` cannot be refreshed — the node is uninitialized: {e}"));
        }
        // Disjoint field borrows: the instance mutably, its live params immutably.
        let live = Params::new(&entry.params);
        let fresh = match &mut entry.exec {
            Execution::Inline(node) => {
                // A device/stream scan is exactly the hook most likely to throw, and it runs under
                // the graph lock — so its panic has to become a refusal, not a poisoned mutex.
                guard_lifecycle(|| node.on_param_refreshed(&ParamKey::new(group, name), &live))?
            }
            // A detached node's instance lives on its worker and the request/response codec has
            // no refresh op — the same deferral as live `on_param_changed` propagation. SAY SO
            // rather than returning Ok with the old list: a silent success would present stale
            // options as freshly scanned, which is worse than a visible refusal.
            Execution::Detached(_) => {
                return Err(format!(
                    "`{group}.{name}` cannot be refreshed on the subprocess tier yet — its \
                     request/response codec has no refresh op"
                ))
            }
        };
        if let Some(options) = &fresh {
            if let Some(Param::Str { options: slot, .. }) =
                entry.params.get_mut(group).and_then(|g| g.get_mut(name))
            {
                *slot = Some(options.clone());
            }
        }
        Ok(fresh)
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
        // Release any prior compiled handle first.
        if let Some(prev) = self.nodes.get(&uid).and_then(|e| e.bindings.get(&key)) {
            if let (Some(ev), Some(id)) = (&self.evaluator, prev.id) {
                ev.release(id);
            }
        }
        // Only an empty source is a true unbind.
        if source.trim().is_empty() {
            if let Some(e) = self.nodes.get_mut(&uid) {
                e.bindings.remove(&key);
            }
            return Ok(());
        }
        // A non-empty source binds a real param — reject a dangling binding (invisible in
        // the descriptor, unclearable from the UI, phantom scheduling edges), like
        // update_param guards param existence.
        if goofi_node::param(&self.nodes[&uid].params, group, name).is_none() {
            return Err(format!("no such param `{group}/{name}`"));
        }
        // Compile only when enabled; a disabled binding is preserved (source round-trips)
        // but carries no handle/refs/error and is skipped by the scheduling + eval guards.
        let (id, refs, global_refs, error) = if enabled {
            match &self.evaluator {
                Some(ev) => match ev.compile(source) {
                    Ok(c) => (Some(c.id), c.refs, c.global_refs, None),
                    Err(e) => (None, Vec::new(), Vec::new(), Some(e.0)),
                },
                None => {
                    (None, Vec::new(), Vec::new(), Some("no expression evaluator available".to_string()))
                }
            }
        } else {
            (None, Vec::new(), Vec::new(), None)
        };
        let binding = ExprBinding {
            source: source.to_string(),
            enabled,
            triggers_process,
            id,
            refs,
            global_refs,
            last_seen: HashMap::new(),
            last_eval: None,
            error,
        };
        if let Some(e) = self.nodes.get_mut(&uid) {
            e.bindings.insert(key, binding);
        }
        Ok(())
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
            .bindings
            .iter()
            .filter(|(_, b)| b.enabled)
            .filter_map(|(key, _)| {
                let p = entry.params.get(&key.group)?.get(&key.name)?;
                Some((key.group.as_str(), key.name.as_str(), p))
            })
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
        if self.is_multi_input(node_in, slot_in) {
            // A multi slot accepts many wires: append this wire's latest-wins cell in
            // connection order (no eviction).
            if let Some(e) = self.nodes.get_mut(&node_in) {
                if let Some(cells) = e.multi_inputs.get_mut(slot_in) {
                    cells.push((node_out, slot_out, None));
                }
            }
        } else {
            // One wire per single input: evict any prior source of this (node_in, slot_in).
            self.links
                .retain(|l| !(l.node_in == node_in && l.slot_in == slot_in));
            self.clear_input(node_in, slot_in);
        }
        self.links.push(new);
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
        if self.is_multi_input(node_in, slot_in) {
            self.drop_multi_wire(node_in, slot_in, node_out, slot_out);
        } else {
            self.clear_input(node_in, slot_in);
        }
        Ok(())
    }

    /// Remove one wire `(src_uid, src_slot)` from a multi input slot, preserving the
    /// connection order of the survivors.
    fn drop_multi_wire(&mut self, node_in: Uid, slot_in: &str, src: Uid, src_slot: &str) {
        if let Some(e) = self.nodes.get_mut(&node_in) {
            if let Some(cells) = e.multi_inputs.get_mut(slot_in) {
                cells.retain(|(u, s, _)| !(*u == src && *s == src_slot));
            }
        }
    }

    fn clear_input(&mut self, uid: Uid, slot: &str) {
        if let Some(e) = self.nodes.get_mut(&uid) {
            if let Some(s) = e.inputs.get_mut(slot) {
                *s = None;
            }
        }
    }

    /// The latest output frame on `(uid, slot)`, if any (data plane read).
    pub fn latest_frame(&self, uid: Uid, slot: &str) -> Option<Data> {
        // The last EMITTED frame (persisted across silent ticks), not the per-tick
        // output that `run_node` resets to None — so a sparse producer still shows data.
        self.nodes
            .get(&uid)
            .and_then(|e| e.last_outputs.get(slot))
            .cloned()
    }

    /// The node's current measured update frequency (Hz) — the same value stamped as
    /// `meta["ufreq"]` on its output. `None` until it has been measured (≥2 emits).
    /// The control plane forwards this to the node-header update-rate readout.
    pub fn node_ufreq(&self, uid: Uid) -> Option<f64> {
        self.nodes.get(&uid).and_then(|e| e.ufreq_meter.ema.map(|ema| 1.0 / ema))
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
        // Globals are patch content: a load starts from a fresh system-seeded store (load_doc then
        // repopulates user globals from the `.gfi`). `dyn_types` stays (catalog, not content).
        self.globals = goofi_core::globals::GlobalStore::new();
        // The node clock belongs to the PATCH, not the process: a patch loaded an hour in must
        // compute what it would have computed at boot, so the next tick re-anchors it. Safe only
        // because every node — and every `UfreqMeter`/`last_emit` reading this clock — was just
        // dropped above; nothing survives to see the discontinuity.
        self.start = None;
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
            for (group, names) in &e.params {
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
                    let _ = self.globals.apply_change(name, Some(value));
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

    /// BFS topological layering (producers before consumers). Each returned level
    /// is a set of mutually-independent nodes — no edges run between them — and
    /// every node's predecessors lie in strictly earlier levels. Nodes trapped in
    /// a cycle form a final level (latest-wins tolerates their back-edges). This
    /// is what lets a level's nodes run concurrently while the graph as a whole
    /// still propagates end-to-end in a single tick.
    /// The scheduling dependency edges `(producer, consumer)`: wired links PLUS
    /// param-expression `nd()` references (a host depends on each node it references, so
    /// the referenced node runs first → the expression sees this-tick's value). A ref
    /// cycle is handled like a link cycle (the remainder runs last, reading prev-tick
    /// outputs — 1-tick feedback).
    fn scheduling_edges(&self) -> Vec<(Uid, Uid)> {
        let mut edges: Vec<(Uid, Uid)> = self
            .links
            .iter()
            .filter(|l| self.nodes.contains_key(&l.node_out) && self.nodes.contains_key(&l.node_in))
            .map(|l| (l.node_out, l.node_in))
            .collect();
        for (host, e) in &self.nodes {
            for b in e.bindings.values() {
                if !b.enabled {
                    continue;
                }
                for r in &b.refs {
                    if let Some(prod) = self.uid_by_name(r) {
                        if prod != *host {
                            edges.push((prod, *host));
                        }
                    }
                }
            }
        }
        edges
    }

    fn topo_levels(&self) -> Vec<Vec<Uid>> {
        let edges = self.scheduling_edges();
        let mut indeg: HashMap<Uid, usize> = self.nodes.keys().map(|k| (*k, 0)).collect();
        for (_from, to) in &edges {
            if let Some(d) = indeg.get_mut(to) {
                *d += 1;
            }
        }
        let mut levels: Vec<Vec<Uid>> = Vec::new();
        let mut placed: std::collections::HashSet<Uid> = std::collections::HashSet::new();
        // Level 0: insertion-order nodes with no incoming edges.
        let mut current: Vec<Uid> = self
            .nodes
            .keys()
            .copied()
            .filter(|u| indeg[u] == 0)
            .collect();
        while !current.is_empty() {
            for u in &current {
                placed.insert(*u);
            }
            // Relax edges out of this level; a successor whose indegree hits zero
            // joins the next level. Reorder by insertion order for determinism.
            let mut freed: std::collections::HashSet<Uid> = std::collections::HashSet::new();
            for u in &current {
                for (from, to) in &edges {
                    if from == u {
                        if let Some(d) = indeg.get_mut(to) {
                            if *d > 0 {
                                *d -= 1;
                                if *d == 0 {
                                    freed.insert(*to);
                                }
                            }
                        }
                    }
                }
            }
            levels.push(current);
            current = self
                .nodes
                .keys()
                .copied()
                .filter(|u| freed.contains(u))
                .collect();
        }
        // Any node never freed sits in a cycle; run them together, last.
        let remainder: Vec<Uid> = self
            .nodes
            .keys()
            .copied()
            .filter(|u| !placed.contains(u))
            .collect();
        if !remainder.is_empty() {
            levels.push(remainder);
        }
        levels
    }

    /// The set of nodes with at least one *wired* triggering input — a link feeds a
    /// `trigger_process` input slot. Mirrors Python's `_has_no_triggering_inputs`
    /// (negated): `autotrigger` free-runs a node only when this is empty for it, so
    /// a connected consumer runs on its producer's rate rather than every tick.
    fn wired_trigger_nodes(&self) -> std::collections::HashSet<Uid> {
        self.links
            .iter()
            .filter(|l| {
                self.nodes.get(&l.node_in).is_some_and(|e| {
                    e.manifest
                        .inputs
                        .iter()
                        .any(|i| i.name == l.slot_in && i.trigger_process)
                })
            })
            .map(|l| l.node_in)
            .collect()
    }

    /// Resolve the expression-bound params of this level's nodes into their concrete
    /// `params`, BEFORE they run — so `process` reads a finished value with no eval in
    /// its path. Called per level in topo order, so a referenced producer (an earlier
    /// level via `scheduling_edges`) has already emitted this tick; a cycle back-edge
    /// reads the producer's still-previous `last_outputs`. Two phases (read then apply)
    /// so the immutable cross-node reads don't collide with the per-node param write.
    fn resolve_level_bindings(
        &mut self,
        level: &[Uid],
        now: Instant,
        now_secs: f64,
        globals: &goofi_core::globals::GlobalsSnapshot,
    ) {
        let Some(ev) = self.evaluator.clone() else { return };

        enum Outcome {
            Value(Param, HashMap<(String, Option<String>), Option<u64>>),
            Error(String),
        }
        let mut results: Vec<(Uid, ParamKey, Outcome)> = Vec::new();

        // READ phase — immutable; decide each due binding's outcome.
        for &uid in level {
            let Some(entry) = self.nodes.get(&uid) else { continue };
            if entry.bindings.is_empty() {
                continue;
            }
            let period = entry.run_policy.period();
            for (key, b) in &entry.bindings {
                if !b.enabled {
                    continue;
                }
                let Some(id) = b.id else { continue }; // compile failed → keep its error
                // Eval-rate gate: at most one eval per `max_frequency` period.
                let gate_open = match (period, b.last_eval) {
                    (None, _) | (Some(_), None) => true,
                    (Some(p), Some(t)) => now.saturating_duration_since(t).as_secs_f64() >= p,
                };
                if !gate_open {
                    continue;
                }
                // Resolve refs: expose EVERY output slot of each referenced node keyed by
                // (name, Some(slot)), plus (name, None) = the single output for a bare
                // nd(). A multi-output node gets no (name, None) entry — a bare nd() on it
                // is caught at runtime by the proxy. Fresh if the producer ran an earlier
                // level this tick, else prev-tick (`last_outputs`) = 1-tick feedback.
                let mut refs_map: HashMap<(String, Option<String>), Option<Data>> = HashMap::new();
                let mut seen: HashMap<(String, Option<String>), Option<u64>> = HashMap::new();
                let mut names: Vec<&str> = b.refs.iter().map(|r| r.as_str()).collect();
                names.sort_unstable();
                names.dedup();
                for nm in names {
                    let mut put = |k: (String, Option<String>), data: Option<Data>| {
                        seen.insert(k.clone(), data.as_ref().and_then(|d| d.meta().index()));
                        refs_map.insert(k, data);
                    };
                    match self.uid_by_name(nm) {
                        None => put((nm.to_string(), None), None), // missing → bare is None
                        Some(pu) => {
                            let pe = &self.nodes[&pu];
                            for o in pe.manifest.outputs {
                                let d = pe.last_outputs.get(o.name).cloned();
                                put((nm.to_string(), Some(o.name.to_string())), d);
                            }
                            if pe.manifest.outputs.len() == 1 {
                                let d = pe.last_outputs.get(pe.manifest.outputs[0].name).cloned();
                                put((nm.to_string(), None), d);
                            }
                        }
                    }
                }
                // Due: a ref-less (time) expr every gated tick; else a ref emitted a new
                // frame, or a first eval, or an error to retry.
                let due = b.refs.is_empty()
                    || b.last_eval.is_none()
                    || b.error.is_some()
                    || seen.iter().any(|(k, idx)| b.last_seen.get(k) != Some(idx));
                if !due {
                    continue;
                }
                let Some(target) =
                    entry.params.get(&key.group).and_then(|g| g.get(&key.name)).cloned()
                else {
                    continue;
                };
                let ctx = goofi_node::EvalCtx { refs: &refs_map, t: now_secs, target: &target, globals };
                match ev.eval(id, &ctx) {
                    Ok(p) => results.push((uid, key.clone(), Outcome::Value(p, seen))),
                    Err(e) => results.push((uid, key.clone(), Outcome::Error(e.0))),
                }
            }
        }

        // APPLY phase — mutable.
        for (uid, key, outcome) in results {
            let Some(entry) = self.nodes.get_mut(&uid) else { continue };
            match outcome {
                Outcome::Value(p, seen) => {
                    // A settled binding re-evaluates to the same value most ticks; only a real
                    // change is worth propagating past `params`.
                    let changed = entry.params.get(&key.group).and_then(|g| g.get(&key.name)) != Some(&p);
                    if let Some(g) = entry.params.get_mut(&key.group) {
                        g.insert(key.name.clone(), p.clone());
                    }
                    let triggers = entry.bindings.get(&key).is_some_and(|b| b.triggers_process);
                    if let Some(b) = entry.bindings.get_mut(&key) {
                        b.last_seen = seen;
                        b.last_eval = Some(now);
                        b.error = None;
                    }
                    if key.group == "common" {
                        // Scheduler metadata, not a node param — re-derive the run gate, exactly as
                        // `update_param` does, and dispatch nothing. Nothing includes the trigger
                        // (spec §1.1): a `common.*` arrival is a RE-PACING, never a reason to run.
                        // Left to fall through, a ref-less `globals.` binding is due every gated
                        // tick, so a `trigger: true` on one would pin `trigger_pending` on forever
                        // and make `autotrigger` moot.
                        entry.run_policy = RunPolicy::from_params(&entry.params);
                    } else if changed && entry.setup_error.is_none() {
                        // The rest of `update_param`'s contract: `on_param_changed` is the SINGLE
                        // source of truth for param→field, so a node that mirrors a hot param to a
                        // field (Oscillator.sfreq) must hear its binding as well as a manual edit.
                        // A detached node's instance lives on its worker and reads params cold off
                        // each Job, so there is nothing to notify — same deferral as `update_param`.
                        //
                        // …and an UNINITIALIZED node hears nothing at all (D3, [`ensure_initialized`]),
                        // exactly as `update_param` above it. Nothing is lost: the evaluated value is
                        // already in `entry.params`, and the next successful initialization replays
                        // every param from there — so skipping the dispatch delivers it once instead
                        // of twice, since `run_node`'s retry would replay this same param moments
                        // later in this very tick.
                        if let Execution::Inline(node) = &mut entry.exec {
                            let hook =
                                guard_lifecycle(|| node.on_param_changed(&key, &p)).unwrap_or_else(fold_panic);
                            if let Err(e) = hook {
                                // The channel a runtime eval failure already uses, so a rejecting
                                // (or panicking) hook surfaces on the field rather than vanishing.
                                if let Some(b) = entry.bindings.get_mut(&key) {
                                    b.error = Some(e.0);
                                }
                            }
                        }
                    }
                    // Guarded by NAMESPACE, not by `changed`, and deliberately so. The value guard
                    // above is an optimization — a settled binding is not worth propagating — but a
                    // `common.*` binding must not wake the node even when its value really did
                    // change, because re-pacing is the whole of what a rate edit means.
                    //
                    // The converse case is left alone on purpose: a ref-less binding OUTSIDE
                    // `common` still triggers on every evaluation, where the honest predicate is
                    // *arrival* rather than re-evaluation. That distinction only becomes
                    // expressible in the async runtime (a mailbox delivery is an arrival; an
                    // evaluation is not), and `resolve_level_bindings` does not survive it — so
                    // the namespace guard is the whole fix here rather than an arrival tracker
                    // built to be deleted.
                    if triggers && key.group != "common" {
                        entry.trigger_pending = true;
                    }
                }
                Outcome::Error(msg) => {
                    if let Some(b) = entry.bindings.get_mut(&key) {
                        b.last_eval = Some(now);
                        b.error = Some(msg);
                    }
                    // The node-level error is derived from `b.error` on read (see
                    // `last_error()`), so recovery/selection stays consistent — nothing to
                    // cache here.
                }
            }
        }
    }

    /// Run one tick of the whole graph against the wall clock. See [`Self::tick_at`].
    pub fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    /// The wall-clock delay until the next node is due to run, as of `now` — the pacing
    /// signal for an adaptive tick loop that honors each node's `common.max_frequency`
    /// with NO extra hardcoded ceiling. `Some(ZERO)`: a node wants to run right now (an
    /// unbounded — `max_frequency <= 0` — or never-run/overdue producer) → tick again
    /// immediately, i.e. as fast as possible. `Some(d)`: the soonest a rate-capped
    /// producer's period elapses. `None`: nothing currently self-starts (the caller may
    /// idle-poll for control-plane edits). Only self-starting producers constrain the
    /// rate; a purely input-triggered node runs in the same tick as its producer.
    pub fn next_run_delay(&self, now: Instant) -> Option<Duration> {
        let wired = self.wired_trigger_nodes();
        let mut soonest: Option<Duration> = None;
        for (uid, e) in &self.nodes {
            if !wants_run(e, uid, &wired) {
                continue;
            }
            // A worker whose bootstrap failed is refused by the dispatch gate in [`Self::tick_at`]
            // (D3) — permanently, since the latch is write-once per worker and only `restart_node`
            // clears it by installing a fresh handle. Dispatch is also the sole writer of a
            // detached node's `last_run`, so leaving it in this scan means the `None` arm below
            // answers ZERO for the life of the patch and pins the loop at its floor, whatever the
            // node's cap says.
            if matches!(&e.exec, Execution::Detached(h) if h.boot_error().is_some()) {
                continue;
            }
            let remaining = match e.run_policy.period() {
                None => Duration::ZERO, // unbounded → as fast as possible
                Some(p) => match e.last_run {
                    None => Duration::ZERO, // never ran → due now
                    // `try_from_secs_f64` (not `from_secs_f64`, which PANICS on overflow —
                // poisoning the graph mutex and killing the server) — an out-of-range period
                // (a huge max_frequency from a .gfi / agent / expression) saturates to MAX,
                // which the caller then clamps to IDLE_POLL anyway.
                Some(t) => Duration::try_from_secs_f64(
                    (p - now.saturating_duration_since(t).as_secs_f64()).max(0.0),
                )
                .unwrap_or(Duration::MAX),
                },
            };
            if soonest.is_none_or(|s| remaining < s) {
                soonest = Some(remaining);
            }
            if remaining.is_zero() {
                break; // can't beat "now"
            }
        }
        soonest
    }

    /// Run one tick as of instant `now` (injectable so rate gating is
    /// deterministically testable). Nodes are grouped into topological levels
    /// ([`Self::topo_levels`]); each level's mutually-independent nodes execute
    /// concurrently on the rayon work-stealing pool, then their fresh outputs are
    /// propagated to the next level's inputs before it runs — so an acyclic graph
    /// still propagates end-to-end within a single tick. A node runs iff it *wants*
    /// to run — it's a pure source (no triggering inputs), a triggering input
    /// received a fresh frame, or it autotriggers *and has no wired trigger* — AND
    /// its [`RunPolicy`] rate cap has elapsed since it last ran. A skipped node
    /// keeps its outputs. With the default policy (`max_frequency == 0`) the rate
    /// cap is unbounded, so this reduces to pure trigger arbitration.
    fn tick_at(&mut self, now: Instant) {
        // Seconds since the first-ever tick — the monotonic wall clock nodes read.
        let start = *self.start.get_or_insert(now);
        let now_secs = now.duration_since(start).as_secs_f64();
        // One globals snapshot for the whole tick — an Arc-backed view every binding eval and every
        // running node's `ctx` shares (globals don't change mid-tick; edits land between ticks).
        let globals = self.globals.snapshot();
        let wired = self.wired_trigger_nodes();
        let levels = self.topo_levels();
        for level in levels {
            // Resolve this level's expression-bound params BEFORE it runs, using the
            // (already-run) earlier levels' fresh outputs. May set `trigger_pending` for
            // a `triggers_process` binding, so it must precede Phase A's run decision.
            self.resolve_level_bindings(&level, now, now_secs, &globals);
            let set: std::collections::HashSet<Uid> = level.iter().copied().collect();

            // Detached tier — drain each detached node's completed output (→ `ran`, so
            // Phase B propagates it like any fresh frame) and dispatch fresh work. The
            // SAME wants_run/should_run gate as inline; only the execution site differs, so
            // a detached node never enters Phase A. `last_run` is set on *dispatch*, so the
            // worker is never fed faster than the node's cap; a still-busy worker coalesces
            // to the newest inputs (the mailbox is latest-wins).
            let mut ran: Vec<Uid> = Vec::new();
            for &uid in &level {
                let Some(entry) = self.nodes.get_mut(&uid) else { continue };
                if !matches!(entry.exec, Execution::Detached(_)) {
                    continue;
                }
                let done = match &entry.exec {
                    Execution::Detached(h) => h.take_output(),
                    Execution::Inline(_) => None,
                };
                if let Some(done) = done {
                    entry.outputs = done.outputs;
                    for (slot, o) in entry.outputs.iter() {
                        if let Some(d) = o {
                            entry.last_outputs.insert(*slot, d.clone());
                        }
                    }
                    entry.last_error = done.error;
                    if entry.outputs.values().any(|o| o.is_some()) {
                        ran.push(uid);
                    }
                }
                // Feed the worker only once it has bootstrapped SUCCESSFULLY. A job built
                // mid-`setup` is a snapshot of PRE-setup state — stale by the time the worker could
                // run it — and it lands the instant `setup` returns, which is what used to race the
                // bootstrap error out of the latest-wins outbox. Skipping it costs nothing:
                // `last_run` and `trigger_pending` are left untouched, so the node runs on the
                // first tick after READY rather than losing its turn.
                //
                // The `boot_error` term is this tier's half of the initialization gate (D3): a
                // worker whose `setup` failed is uninitialized, and "ticks of a node that had a
                // setup() error should not be possible". Unlike the inline gate
                // ([`ensure_initialized`]) it never RETRIES — that would need a new worker
                // protocol op — so `restart_node` is the retry door for a detached node.
                if !matches!(&entry.exec, Execution::Detached(h)
                    if h.stage() == detached::STAGE_READY && h.boot_error().is_none())
                {
                    continue;
                }
                let since_last = entry.last_run.map(|t| now.saturating_duration_since(t).as_secs_f64());
                if entry.run_policy.should_run(since_last, wants_run(entry, &uid, &wired)) {
                    entry.last_run = Some(now);
                    entry.trigger_pending = false;
                    let multis = materialize_multis(entry);
                    let job = detached::Job {
                        inputs: entry.inputs.clone(),
                        multis,
                        params: entry.params.clone(),
                        now: now_secs,
                    };
                    if let Execution::Detached(h) = &entry.exec {
                        h.dispatch(job);
                    }
                }
            }

            // Phase A — run every runnable INLINE node in this level in parallel. Each
            // closure touches only its own entry (disjoint `&mut`), so there is no
            // shared state and the result is independent of thread scheduling.
            {
                let batch: Vec<(Uid, &mut NodeEntry)> = self
                    .nodes
                    .iter_mut()
                    .filter(|(uid, e)| {
                        if !set.contains(uid) || !matches!(e.exec, Execution::Inline(_)) {
                            return false;
                        }
                        let since_last = e.last_run.map(|t| now.saturating_duration_since(t).as_secs_f64());
                        e.run_policy.should_run(since_last, wants_run(e, uid, &wired))
                    })
                    .map(|(uid, e)| {
                        e.last_run = Some(now);
                        e.ctx.now = now_secs;
                        // Live globals for `process` (Arc bump); `setup` latched them at insert time.
                        e.ctx.globals = globals.clone();
                        (*uid, e)
                    })
                    .collect();
                ran.extend(batch.iter().map(|(u, _)| *u));
                batch.into_par_iter().for_each(|(_, entry)| run_node(entry));
            }

            // Phase B — propagate this level's fresh frames to their consumers
            // (serial; one-wire-per-input means each input has a single writer).
            for uid in ran {
                let produced: Vec<(&'static str, Data)> = self.nodes[&uid]
                    .outputs
                    .iter()
                    .filter_map(|(k, v)| v.as_ref().map(|d| (*k, d.clone())))
                    .collect();
                if produced.is_empty() {
                    continue;
                }
                let outgoing: Vec<(&'static str, Uid, &'static str)> = self
                    .links
                    .iter()
                    .filter(|l| l.node_out == uid)
                    .map(|l| (l.slot_out, l.node_in, l.slot_in))
                    .collect();
                for (slot_out, tgt, slot_in) in outgoing {
                    if let Some(d) = produced
                        .iter()
                        .find(|(s, _)| *s == slot_out)
                        .map(|(_, d)| d.clone())
                    {
                        if let Some(te) = self.nodes.get_mut(&tgt) {
                            if let Some(slot) = te.inputs.get_mut(slot_in) {
                                *slot = Some(d); // single slot: latest-wins
                            } else if let Some(cells) = te.multi_inputs.get_mut(slot_in) {
                                // multi slot: update THIS wire's latest-wins cell,
                                // keyed by its source (uid, slot_out) — position kept.
                                if let Some(cell) =
                                    cells.iter_mut().find(|(u, s, _)| *u == uid && *s == slot_out)
                                {
                                    cell.2 = Some(d);
                                }
                            }
                            // A fresh frame on a triggering input wakes the consumer.
                            if te
                                .manifest
                                .inputs
                                .iter()
                                .any(|i| i.name == slot_in && i.trigger_process)
                            {
                                te.trigger_pending = true;
                            }
                        }
                    }
                }
            }
        }
        self.stamp_error_onsets(now);
    }

    /// Note, for every node, when its current error first read the way it does now — the clock
    /// [`Graph::error_age`] reports. Derived from [`Graph::last_error`] rather than written at
    /// each site that can set one, so a process failure, a binding failure and a detached
    /// bootstrap failure are all stamped by the same rule, and the stamp cannot outlive the error
    /// it belongs to. Costs one comparison per node per tick and allocates only on a transition.
    fn stamp_error_onsets(&mut self, now: Instant) {
        for e in self.nodes.values_mut() {
            let current = entry_error(e);
            if e.error_since.as_ref().map(|(m, _)| m.as_str()) != current {
                let stamped = current.map(|m| (m.to_string(), now));
                e.error_since = stamped;
            }
        }
    }
}

/// One node's current error, derived fresh from the three places one can arise — see
/// [`Graph::last_error`], whose contract this is. A free function so the per-tick onset sweep can
/// read it while holding a `&mut NodeEntry`, which keeps derivation and stamping on one rule.
fn entry_error(e: &NodeEntry) -> Option<&str> {
    // A detached worker's bootstrap failure lives on its handle, not in `last_error`, because
    // the per-tick `Done` channel is latest-wins and a successful job erases an un-drained one
    // (see `detached::DetachedHandle::boot_error`). It outranks a process error deliberately:
    // it is the ROOT CAUSE, and it is a one-shot fact — a process error recurs on every tick
    // and can be observed again, a failed `setup` never can.
    if let Execution::Detached(h) = &e.exec {
        if let Some(err) = h.boot_error() {
            return Some(err);
        }
    }
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
    e.bindings
        .iter()
        .filter_map(|(k, b)| b.error.as_deref().map(|s| (k, s)))
        .min_by(|a, b| a.0.cmp(b.0))
        .map(|(_, s)| s)
}

/// Route a constructed node onto its execution tier — the ONE place the isolation split
/// lives, shared by insertion and [`Graph::restart_node`] so the two cannot diverge.
///
/// An InProcess node is seeded synchronously (replay `on_param_changed`, then `setup`) and
/// runs inline. A Subprocess node is detached onto an off-tick worker that seeds ITSELF (its
/// setup / first-tick spawn may block) and latches a bootstrap failure on its handle, where
/// [`Graph::last_error`] reads it — so its `last_error` starts, and stays, `None` here.
fn make_exec(
    manifest: &'static NodeManifest,
    mut node: Box<dyn goofi_node::Node>,
    params: &ParamGroups,
    ctx: &mut NodeCtx,
) -> (Execution, Option<String>) {
    match manifest.isolation {
        goofi_node::Isolation::InProcess => {
            let err = seed_node(&mut *node, params, ctx);
            (Execution::Inline(node), err)
        }
        goofi_node::Isolation::Subprocess => {
            let handle = detached::DetachedHandle::spawn(node, manifest, params.clone(), ctx.clone());
            (Execution::Detached(handle), None)
        }
    }
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

/// The initialization gate (D3). A node whose `setup()` failed is UNINITIALIZED, so nothing may
/// run against it — not `process`, not a param callback, not a refresh. Any of those interactions
/// RETRIES the initialization first ([`seed_node`]: the param replay and `setup()` together, which
/// are one unit), so a node whose device has since appeared, or whose bad param the user has just
/// corrected, comes back without an explicit restart. `Err` carries the standing failure: the gate
/// is shut and the caller must not proceed.
///
/// A DETACHED node is a no-op here: its `setup()` runs on the worker and its failure is latched
/// there (`detached::Channels::boot_error`), so `setup_error` is never set for one. Retrying it
/// would need a new worker protocol op — `update_param` does not reach the worker at all and
/// `refresh_param` refuses on that tier — so [`Graph::restart_node`] stays the retry door there.
/// Not forgotten, scoped out; the job-dispatch gate in [`Graph::tick_at`] is what keeps a failed
/// worker from being fed in the meantime.
fn ensure_initialized(entry: &mut NodeEntry) -> Result<(), String> {
    if entry.setup_error.is_none() {
        return Ok(());
    }
    if let Execution::Inline(node) = &mut entry.exec {
        // Every attempt stamps itself, so the tick's backoff restarts from an interaction's retry
        // too — the interaction is unthrottled, but it does not also hand the next tick a free one.
        entry.last_setup_attempt = entry.ctx.now;
        entry.setup_error = seed_node(&mut **node, &entry.params, &mut entry.ctx);
    }
    match &entry.setup_error {
        Some(e) => Err(e.clone()),
        None => Ok(()),
    }
}

fn run_node(entry: &mut NodeEntry) {
    entry.trigger_pending = false;
    // A tick is an interaction like any other: it retries the initialization, and runs nothing if
    // that fails. `trigger_pending` is consumed either way — the frame that asked for this run has
    // been seen, and holding it would make the node fire twice the moment it recovers.
    //
    // Unlike `update_param`/`refresh_param` it retries on a TIMER ([`SETUP_RETRY_INTERVAL`]): a
    // tick is not a user asking, it is one of however many the pacer admits — a rate-capped source
    // ticks every period and a `trigger_process` consumer once per delivered frame — and this runs
    // under the graph lock.
    if entry.setup_error.is_some() && entry.ctx.now - entry.last_setup_attempt < SETUP_RETRY_INTERVAL {
        return;
    }
    if ensure_initialized(entry).is_err() {
        return;
    }
    // Materialize each multi slot's present frames in connection order for the node
    // (Arc-bump clones). Empty for nodes with no multi slots — the common case pays
    // nothing beyond an empty map.
    let multis = materialize_multis(entry);
    // A detached node runs on its own worker (see `tick_at`), never inline here.
    let Execution::Inline(node) = &mut entry.exec else { return };
    entry.last_error = execute_node(
        entry.manifest,
        node,
        &entry.params,
        &entry.inputs,
        &multis,
        &mut entry.outputs,
        &mut entry.last_outputs,
        &mut entry.ctx,
        &mut entry.index_counters,
        &mut entry.ufreq_meter,
    );
}

/// Run a node's `process()` + engine meta-stamping in place against its live state.
/// Shared by the inline tick path ([`run_node`]) and the detached worker, so both stamp
/// index/ufreq identically. `catch_unwind` keeps a faulty node from unwinding the
/// scheduler (and, in the bridge, poisoning the graph mutex). Returns the process/panic
/// error (`None` on success); a binding error is NOT folded in here — it is derived on
/// read by `last_error()`, so a recovered binding surfaces even on a node that no longer
/// runs. The caller owns `trigger_pending`.
// The parts are the node's live per-tick state, passed individually so both a `NodeEntry`
// (inline) and a detached worker — which owns the same parts separately — can call it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_node(
    manifest: &'static NodeManifest,
    node: &mut Box<dyn goofi_node::Node>,
    params: &ParamGroups,
    inputs: &IndexMap<&'static str, Option<Data>>,
    multis: &IndexMap<&'static str, Vec<Data>>,
    outputs: &mut IndexMap<&'static str, Option<Data>>,
    last_outputs: &mut IndexMap<&'static str, Data>,
    ctx: &mut NodeCtx,
    index_counters: &mut HashMap<&'static str, u64>,
    ufreq_meter: &mut UfreqMeter,
) -> Option<String> {
    for v in outputs.values_mut() {
        *v = None;
    }
    // A required slot must HOLD data when the node ticks — presence, never wiring, so a slot
    // wired to a node that has emitted nothing reads the same as an unwired one (invariant 1).
    // Checked here, the one seam the inline tick path and the detached worker share, so all
    // three execution tiers answer identically. `last_outputs` is untouched: a viewer on a
    // previously-emitting slot keeps its frame.
    for slot in manifest.inputs.iter().filter(|s| s.required) {
        let absent = if slot.multi {
            multis.get(slot.name).is_none_or(|v| v.is_empty())
        } else {
            inputs.get(slot.name).and_then(Option::as_ref).is_none()
        };
        if absent {
            return Some(format!("required input slot `{}` has no data", slot.name));
        }
    }
    let inp = Inputs::with_multi(inputs, multis);
    let p = goofi_node::Params::new(params);
    // Scope the `Outputs` borrow so `outputs` is free again for stamping below.
    let result = {
        let mut out = Outputs::new(outputs);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| node.process(&inp, &mut out, ctx, &p)))
    };
    let err = match result {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e.0),
        Err(pnc) => Some(panic_message(pnc)),
    };
    stamp_meta_parts(manifest, inputs, outputs, ctx.now, index_counters, ufreq_meter);
    // Persist each freshly-emitted (stamped) frame so `latest_frame` keeps returning it
    // on later ticks where this node emits nothing — viewers of a sparse producer never
    // blink to None.
    for (slot, out) in outputs.iter() {
        if let Some(d) = out {
            last_outputs.insert(*slot, d.clone());
        }
    }
    err
}

/// The number of frames a `Data` spans — its total element count (numpy `.size`
/// for an array, `len` for a string/table). This, not a static per-slot flag, is
/// the timeline discriminator: a length-preserving transform's output matches its
/// input's frame count; a generator or length-changing transform does not.
fn frame_count(d: &Data) -> usize {
    match d.value() {
        goofi_core::Value::Array(s) => s.shape().iter().product(),
        goofi_core::Value::Str(s) => s.chars().count(),
        goofi_core::Value::Table(m) => m.len(),
    }
}

/// Stamp the engine-owned meta — `index` and `ufreq` — on every frame this node
/// just emitted (the node never touches either).
///
/// **index**: for each output, propagate the index of the SINGLE index-bearing
/// TRIGGERING input whose frame count equals the output's — that input is the same
/// data timeline, so an upstream drop stays visible downstream. A non-triggering
/// (control/reference) input — an oscillator's scalar frequency, say — is never a
/// timeline candidate even if its length happens to match. With zero, or more than
/// one, matching inputs (a generator, a length-changing transform, or an ambiguous
/// fan-in) the slot starts a fresh per-output counter that advances one per emit.
/// Ported from the Python node's `_next_index`/`_propagated_index`.
///
/// **ufreq**: the NODE's measured update rate (Hz) — an EMA of the inter-emit
/// interval keyed on `ctx.now`, `None` until a second emit gives one interval.
/// Measured PER NODE (one `ufreq_meter`), advanced once per productive tick (a tick
/// emitting ≥1 output), and the same value stamped onto every emitted slot — ufreq
/// describes how often the node updates, not a per-slot cadence. Authoritative —
/// overwritten every emit, never inherited from upstream meta.
fn stamp_meta_parts(
    manifest: &'static NodeManifest,
    inputs: &IndexMap<&'static str, Option<Data>>,
    outputs: &mut IndexMap<&'static str, Option<Data>>,
    now: f64,
    counters: &mut HashMap<&'static str, u64>,
    ufreq_meter: &mut UfreqMeter,
) {
    // Nothing emitted this tick → no meta to stamp, and the ufreq meter only advances
    // on a productive emit. Skip the whole index-timeline scan (the common case for a
    // rate-gated or idle node that ran but produced nothing).
    if outputs.values().all(|o| o.is_none()) {
        return;
    }
    // Only triggering inputs carry the data timeline; control inputs are excluded.
    let triggering: std::collections::HashSet<&str> = manifest
        .inputs
        .iter()
        .filter(|s| s.trigger_process)
        .map(|s| s.name)
        .collect();
    // Snapshot the index-bearing triggering inputs (index, frame_count) — no borrow held.
    let input_frames: Vec<(u64, usize)> = inputs
        .iter()
        .filter(|(name, _)| triggering.contains(*name))
        .filter_map(|(_, o)| o.as_ref())
        .filter_map(|d| d.meta().index().map(|i| (i, frame_count(d))))
        .collect();
    // Node-level ufreq: EMA of the inter-emit interval, inverted. `None` until the
    // second emit; a non-advancing clock (`dt <= 0`) keeps the prior estimate.
    let node_ufreq = {
        let m = ufreq_meter;
        match m.last_emit {
            None => {
                m.last_emit = Some(now); // first emit: no interval yet
                None
            }
            Some(prev) => {
                let dt = now - prev;
                m.last_emit = Some(now);
                if dt > 0.0 {
                    let ema = m.ema.map_or(dt, |p| UFREQ_EMA_ALPHA * dt + (1.0 - UFREQ_EMA_ALPHA) * p);
                    m.ema = Some(ema);
                    Some(1.0 / ema)
                } else {
                    m.ema.map(|e| 1.0 / e)
                }
            }
        }
    };
    // Rewrite outputs while advancing the index counters.
    for (slot, slot_opt) in outputs.iter_mut() {
        let Some(d) = slot_opt else { continue };
        let of = frame_count(d);
        // Exactly one index-bearing triggering input with a matching frame count → the
        // same timeline; zero or more than one → a fresh per-output counter.
        let mut matches = input_frames.iter().filter(|(_, f)| *f == of).map(|(i, _)| *i);
        let counter = counters.entry(*slot).or_insert(0);
        let index = match (matches.next(), matches.next()) {
            (Some(i), None) => i,
            _ => *counter,
        };
        // Keep the fresh counter monotonically past whatever we emitted. Without this, a
        // slot that MATCHES on one frame (an accumulator's first output length equals its
        // input length) then goes fresh would restart the counter at 0 — duplicating or
        // regressing the index at stream start (the Oscillator→Buffer reference patch).
        *counter = index + 1;
        *d = d.with_stamps(index, node_ufreq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Meta, SlotType, Value};
    use goofi_node::{
        default_factory, ExprDecl, ExprMode, Isolation, Node, NodeManifest, NodeResult,
        OutputDecl, ParamDecl, ParamSpec, Params, SlotDecl,
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
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 30.0, "reads the seeded default_ufreq");
        // A mid-run edit is visible on the next tick — `process` reads globals live, not latched.
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(45.0))).unwrap();
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 45.0, "sees the edited value next tick");
    }

    #[test]
    fn expression_binding_reads_globals_and_tracks_edits() {
        use goofi_core::globals::GlobalValue;
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "globals.default_ufreq", true, false).unwrap();
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 30.0, "the binding reads the global");
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(48.0))).unwrap();
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 48.0, "the bound param re-rates on a global edit");
    }

    #[test]
    fn editing_a_referenced_global_forces_only_that_bindings_reeval() {
        // White-box: a global edit resets `last_eval` (→ due + gate-open) for exactly the bindings
        // that read it, and leaves unrelated bindings untouched — the targeted dirty-tracking that
        // makes the mixed `nd('x') * globals.gain` case re-rate even when its ref hasn't re-emitted.
        use goofi_core::globals::GlobalValue;
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "globals.default_ufreq", true, false).unwrap();
        let key = ParamKey::new("constant", "value");
        g.tick();
        assert!(g.nodes.get(&n).unwrap().bindings.get(&key).unwrap().last_eval.is_some(), "evaluated once");
        // Editing the referenced global forces an immediate re-eval.
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(50.0))).unwrap();
        assert!(
            g.nodes.get(&n).unwrap().bindings.get(&key).unwrap().last_eval.is_none(),
            "a referenced-global edit resets the eval gate"
        );
        // An UNrelated global edit must not disturb the binding.
        g.tick(); // re-evaluates → last_eval Some again
        g.apply_global_change("unrelated", Some(GlobalValue::Float(1.0))).unwrap();
        assert!(
            g.nodes.get(&n).unwrap().bindings.get(&key).unwrap().last_eval.is_some(),
            "an unrelated global edit leaves the binding alone"
        );
    }

    #[test]
    fn fresh_add_seeds_a_default_expr_binding_that_tracks_globals() {
        use goofi_core::globals::GlobalValue;
        let mut g = eval_graph();
        let n = g.add_node("_TestDefaultExpr", None).unwrap();
        // The declared default_expr became a real, live binding (not a literal).
        let info = g.param_expression(n, "control", "rate").expect("default_expr seeded a binding");
        assert_eq!(info.source, "globals.default_ufreq");
        assert!(info.enabled && info.error.is_none(), "seeded binding is enabled + healthy");
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 30.0, "evaluates to the global");
        // Editing the referenced global re-rates the producer live.
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(42.0))).unwrap();
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 42.0, "re-rates on a global edit");
    }

    #[test]
    fn a_common_binding_re_paces_the_node_and_never_asks_it_to_run() {
        // Spec §1.1: a `common.*` arrival is a RE-PACING, never a reason to run. The APPLY phase
        // already treats a common binding as scheduler metadata that "dispatches nothing" — but
        // the trigger set sat outside that branch and fired for every group, so a `common.*`
        // binding with `trigger: true` pinned `trigger_pending` on permanently (a `globals.`-only
        // expression is ref-less, hence due on every gated tick, and the trigger was applied
        // whether or not the value changed).
        //
        // Driven through `resolve_level_bindings` directly, and asserted on `trigger_pending`
        // rather than on whether the node ran, because neither proxy survives contact:
        // `wants_run` still free-runs a source through `!has_trigger_inputs` (the term the async
        // runtime deletes — which is what would turn this into a live "autotrigger does nothing
        // for any producer" bug), and a full tick would consume the flag in the very run it
        // wrongly caused. The eval-rate gate is the same period as the run gate, so a rate cap
        // cannot separate them either.
        let mut g = eval_graph();
        // Oscillator binds `common.max_frequency` to `globals.default_ufreq` with `trigger: true`.
        let osc = g.add_node("Oscillator", None).unwrap();
        // The control: a triggering binding OUTSIDE `common` must still wake `process`, so the fix
        // has to be by namespace rather than a blanket removal of the trigger.
        let paced = g.add_node("_TestCarriedExpr", None).unwrap();

        let globals = g.globals.snapshot();
        g.resolve_level_bindings(&[osc, paced], Instant::now(), 0.0, &globals);

        assert!(!g.nodes[&osc].trigger_pending, "a common.* re-evaluation re-paces, it does not run");
        assert!(g.nodes[&paced].trigger_pending, "a triggering binding outside common still wakes it");
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

        // The carried one must not drive its param either: the spec literal stands.
        g.tick();
        assert_eq!(g.params(n).unwrap()["control"]["carried"].as_f64(), Some(5.0), "literal stands");
        assert_eq!(g.params(n).unwrap()["control"]["paced"].as_f64(), Some(30.0), "the global drives");
    }

    #[test]
    fn default_expr_falls_back_to_the_literal_without_an_evaluator() {
        // No evaluator wired ⇒ no binding is minted; the param keeps its spec-default literal (5.0),
        // never an errored "no evaluator" binding. Graceful degrade for eval-less runs (a build
        // without the `python` feature, or an interpreter the evaluator could not initialize).
        let mut g = Graph::new();
        let n = g.add_node("_TestDefaultExpr", None).unwrap();
        assert!(g.param_expression(n, "control", "rate").is_none(), "no binding without an evaluator");
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 5.0, "the literal fallback is used");
    }

    #[test]
    fn binding_common_max_frequency_to_a_global_re_rates_the_run_policy() {
        // The producer story end-to-end: a `common.max_frequency` bound to `globals.default_ufreq`
        // rates the scheduler at the global, and a global EDIT re-rates it immediately — even though
        // the node is under a rate cap (the dirty-reset opens the closed eval gate). This is exactly
        // how editing default_ufreq re-paces every Oscillator.
        use goofi_core::globals::GlobalValue;
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "common", "max_frequency", "globals.default_ufreq", true, false).unwrap();
        g.tick();
        assert_eq!(g.nodes.get(&n).unwrap().run_policy.max_frequency, 30.0, "rated by the global");
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(12.0))).unwrap();
        g.tick();
        assert_eq!(g.nodes.get(&n).unwrap().run_policy.max_frequency, 12.0, "re-rates on a global edit");
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
        g2.tick();
        assert_eq!(first_f32(&g2.latest_frame(restored, "out").unwrap()), 30.0, "evaluates to the global after load");
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
        g2.tick();
        assert_eq!(
            first_f32(&g2.latest_frame(restored, "out").unwrap()),
            100.0,
            "the saved literal survives (not re-rated to the global)"
        );
    }

    #[test]
    fn load_runs_setup_against_the_saved_params_not_the_type_defaults() {
        // A node's one-time init reads its params — it allocates a buffer of `size`, opens device
        // `name`. On load, `setup()` must therefore see the params the user SAVED. The load path
        // built every node from the type's DEFAULTS and applied the saved values only afterwards,
        // and nothing re-runs `setup`; on the detached tier `update_param` is an explicit no-op, so
        // the child never saw them at all. The undo/redo restore path already gets this right
        // (`Command::AddNode` carries the captured params) — the two paths must agree.
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
        g2.tick();
        assert_eq!(
            first_f32(&g2.latest_frame(restored, "out").unwrap()),
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

    // The DETACHED analogue of `_TestRequired`, registered at runtime the way every detached
    // fixture here is. The required check sits in `execute_node` BECAUSE that is the one seam the
    // inline tick path and the detached worker share — and every other required-slot fixture is
    // `Isolation::InProcess`, so lifting the check into `run_node` (the inline caller) would exempt
    // this whole tier without reddening a single test.
    struct RequiredDetached {
        /// One entry per `process` entry, carrying the value it saw on `data` (`NaN` when it saw
        /// none). Written on the worker, read by the test from the tick thread.
        arrivals: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    }
    impl Node for RequiredDetached {
        fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let seen = inp.get("data").map(first_f32).unwrap_or(f32::NAN);
            self.arrivals.lock().unwrap().push(seen);
            let d = Data::array_f32(vec![1], seen.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static REQ_DET_IN: &[SlotDecl] =
        &[SlotDecl { name: "data", kind: SlotType::Array, trigger_process: true, multi: false, required: true }];
    static REQ_DET_MANIFEST: NodeManifest = NodeManifest {
        type_name: "_TestRequiredDetached",
        category: "test",
        doc: "one required input, run on a detached worker",
        inputs: REQ_DET_IN,
        outputs: G_OUT,
        params: NO_PARAMS,
        isolation: Isolation::Subprocess,
        producer: false,
        factory: rt_stub_factory,
    };

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

    #[test]
    fn source_streams_latest_frame() {
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(7.0, -1e9, 1e9))
            .unwrap();
        g.tick();
        let f = g.latest_frame(src, "out").expect("frame");
        assert_eq!(first_f32(&f), 7.0);
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
        g.add_link(src, "out", echo, "in").unwrap();

        g.tick();

        let f = g.latest_frame(echo, "out").expect("echo produced a frame");
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
        g.add_link(a, "out", echo, "in").unwrap();
        g.add_link(b, "out", echo, "in").unwrap(); // evicts a
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(echo, "out").unwrap()), 2.0);
    }

    // ---- multi-input slots -------------------------------------------------

    #[test]
    fn multi_input_collects_wires_in_connection_order() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();
        g.tick();
        // [count=3, then each wire's value in connection order].
        assert_eq!(as_f32_vec(&g.latest_frame(col, "out").unwrap()), vec![3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn multi_input_remove_link_drops_one_wire_keeping_order() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();
        g.remove_link(b, "out", col, "ins").unwrap();
        g.tick();
        assert_eq!(as_f32_vec(&g.latest_frame(col, "out").unwrap()), vec![2.0, 1.0, 3.0]);
    }

    #[test]
    fn multi_input_remove_node_drops_its_wires() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();
        g.remove_node(b).unwrap();
        g.tick();
        assert_eq!(as_f32_vec(&g.latest_frame(col, "out").unwrap()), vec![2.0, 1.0, 3.0]);
    }

    #[test]
    fn multi_input_latest_wins_per_wire() {
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.tick();
        assert_eq!(as_f32_vec(&g.latest_frame(col, "out").unwrap()), vec![2.0, 1.0, 2.0]);
        // a's next frame overwrites its cell (latest-wins); b is retained; order stable.
        g.update_param(a, "constant", "value", Param::float(9.0, -1e9, 1e9)).unwrap();
        g.tick();
        assert_eq!(as_f32_vec(&g.latest_frame(col, "out").unwrap()), vec![2.0, 9.0, 2.0]);
    }

    #[test]
    fn multi_input_empty_slot_is_empty_list() {
        let mut g = Graph::new();
        let col = g.add_node("_TestCollect", None).unwrap(); // autotriggers with 0 wires
        g.tick();
        assert_eq!(as_f32_vec(&g.latest_frame(col, "out").unwrap()), vec![0.0]);
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
        g2.tick();
        // All 3 wires restored, in connection order (a=1, b=2, c=3).
        assert_eq!(as_f32_vec(&g2.latest_frame(col2, "out").unwrap()), vec![3.0, 1.0, 2.0, 3.0]);
    }

    // ---- required input slots ----------------------------------------------

    #[test]
    fn a_required_input_with_no_frame_errors_before_process_is_entered() {
        let mut g = Graph::new();
        let n = g.add_node("_TestRequired", None).unwrap();
        // Autotrigger is what makes an unwired node tick at all — D1: the check fires on a TICK,
        // never on the configuration.
        g.update_param(n, "common", "autotrigger", Param::boolean(true)).unwrap();
        g.tick();
        assert_eq!(
            g.last_error(n),
            Some("required input slot `data` has no data"),
            "the empty required slot is named"
        );
        assert!(g.latest_frame(n, "out").is_none(), "a refused tick emits nothing");
        // …and `process` was never ENTERED, not merely denied its output. The node counts its own
        // calls, so once the slot is fed the count must read 1; a check placed AFTER `node.process`
        // would leave it at 2.
        let src = const_src(&mut g, 4.0);
        g.add_link(src, "out", n, "data").unwrap();
        g.tick();
        assert_eq!(g.last_error(n), None, "the fed slot clears the error");
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 1.0, "process was entered exactly once");
    }

    #[test]
    fn a_required_input_holding_a_frame_runs_cleanly() {
        let mut g = Graph::new();
        let src = const_src(&mut g, 7.0);
        let n = g.add_node("_TestRequired", None).unwrap();
        g.add_link(src, "out", n, "data").unwrap();
        g.tick();
        assert_eq!(g.last_error(n), None, "a satisfied required slot is not an error");
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 1.0, "process ran");
    }

    #[test]
    fn a_required_multi_input_with_no_frames_errors() {
        let mut g = Graph::new();
        let n = g.add_node("_TestRequiredMulti", None).unwrap();
        g.update_param(n, "common", "autotrigger", Param::boolean(true)).unwrap();
        g.tick();
        assert_eq!(
            g.last_error(n),
            Some("required input slot `ins` has no data"),
            "an unwired variadic slot holds no frames either"
        );
        assert!(g.latest_frame(n, "out").is_none(), "a refused tick emits nothing");
        // Wire one source and the same node runs, seeing its one frame.
        let src = const_src(&mut g, 1.0);
        g.add_link(src, "out", n, "ins").unwrap();
        g.tick();
        assert_eq!(g.last_error(n), None);
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 1.0, "one wire present");
    }

    #[test]
    fn a_required_input_on_a_node_that_never_ticks_is_silent() {
        // D1 again, from the other side: an unwired node with no autotrigger is "a disconnected
        // node floating in space" — we never asked it to run, so it has nothing to report.
        let mut g = Graph::new();
        let n = g.add_node("_TestRequired", None).unwrap();
        for _ in 0..3 {
            g.tick();
        }
        assert_eq!(g.last_error(n), None, "a node that never ran cannot be missing an input");
        assert!(g.latest_frame(n, "out").is_none(), "and it emitted nothing");
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
        g.add_link(src, "out", n, "tick").unwrap();

        g.tick();

        assert_eq!(
            g.last_error(n),
            Some("required input slot `data` has no data"),
            "a wire is not a frame"
        );
        assert!(g.latest_frame(n, "out").is_none(), "so `process` was never entered");
    }

    #[test]
    fn a_required_input_is_refused_on_the_detached_tier_too() {
        // The check is placed in the seam BOTH tiers share, and this is the only test that holds it
        // there: hoist it into `run_node` and every inline required-slot test stays green while a
        // detached node pays a full dispatch and fails inside its worker instead of the engine
        // refusing the tick.
        let mut g = Graph::new();
        let arrivals: std::sync::Arc<std::sync::Mutex<Vec<f32>>> = Default::default();
        let mine = arrivals.clone();
        g.register_dyn_type(
            &REQ_DET_MANIFEST,
            Box::new(move |_p| Box::new(RequiredDetached { arrivals: mine.clone() })),
        );
        let n = g.add_node("_TestRequiredDetached", None).unwrap();
        g.update_param(n, "common", "autotrigger", Param::boolean(true)).unwrap();
        // One dispatch per second of the clock this test drives, so the positive control below can
        // hold that clock still and count exactly one job rather than however many ticks the drain
        // happened to take.
        g.update_param(n, "common", "max_frequency", Param::float(1.0, 0.0, 1e9)).unwrap();
        wait_bootstrapped(&g, n); // dispatch is gated on READY

        // The worker's answer comes back through `Done` and is drained on a LATER tick, so the
        // error is a bounded poll rather than the tick that dispatched it.
        let t0 = Instant::now();
        let mut err = None;
        for i in 0..200 {
            g.tick_at(t0 + Duration::from_millis(5 * i));
            if let Some(e) = g.last_error(n) {
                err = Some(e.to_string());
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(
            err.as_deref(),
            Some("required input slot `data` has no data"),
            "the detached tier names the empty required slot identically"
        );
        assert!(arrivals.lock().unwrap().is_empty(), "and `process` was never entered on the worker");

        // The positive control. Without it both assertions above hold just as well for a job that
        // was never dispatched at all — an unready worker, a `wants_run` that answered false — which
        // is a test that proves nothing.
        let src = const_src(&mut g, 4.0);
        g.add_link(src, "out", n, "data").unwrap();
        let t1 = t0 + Duration::from_secs(10); // held still: the 1 Hz cap admits exactly one dispatch
        let mut ran = false;
        for _ in 0..200 {
            g.tick_at(t1);
            if g.latest_frame(n, "out").is_some() {
                ran = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(ran, "the fed node ran on its worker and its frame came back");
        assert_eq!(g.last_error(n), None, "the fed slot clears the error");
        assert_eq!(
            *arrivals.lock().unwrap(),
            vec![4.0],
            "exactly one `process` entry, seeing the frame it was fed"
        );
    }

    // ---- the initialization gate (D3) --------------------------------------

    #[test]
    fn a_failed_setup_stands_through_the_ticks_that_used_to_erase_it() {
        // `run_node` assigned `execute_node`'s result straight into `last_error`, and a
        // construction failure lived in that SAME field — so a node whose `setup()` raised erased
        // its own bootstrap failure on its first clean tick and read `ready, no error` though it
        // had never initialized. The failure now has its own field, which `execute_node` cannot
        // reach.
        let mut g = Graph::new();
        let (n, _counts) = gated_setup_node(&mut g);
        assert_eq!(g.last_error(n), Some("device is not open"), "setup failed at construction");
        for _ in 0..5 {
            g.tick();
        }
        assert_eq!(g.last_error(n), Some("device is not open"), "and that is still what it reports");
        assert_eq!(g.node_stage(n), "error", "the editor draws it errored, not ready");
    }

    #[test]
    fn an_uninitialized_node_never_enters_process() {
        // The message is not enough: a gate that reports the failure and runs `process` anyway
        // passes every assertion above. The run counter is what fails it — "ticks of a node that
        // had a setup() error should not be possible".
        let mut g = Graph::new();
        let (n, counts) = gated_setup_node(&mut g);
        for _ in 0..5 {
            g.tick();
        }
        assert_eq!(counts.lock().unwrap().runs, 0, "process was never entered");
        assert!(g.latest_frame(n, "out").is_none(), "and the node emitted nothing");
    }

    #[test]
    fn correcting_the_param_that_broke_setup_reinitializes_the_node() {
        // D3's retry door. `update_param` stores the new value BEFORE the gate, so the replay
        // inside the retry delivers it — which is what makes fixing the param the fix.
        let mut g = Graph::new();
        let (n, counts) = gated_setup_node(&mut g);
        assert_eq!(g.last_error(n), Some("device is not open"));

        assert!(g.update_param(n, "boot", "ok", Param::boolean(true)).is_ok(), "the edit is accepted");
        assert_eq!(g.last_error(n), None, "the retry initialized the node");
        assert_eq!(g.node_stage(n), "ready");
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 1.0, "and it runs — for the first time");
        let c = counts.lock().unwrap();
        assert_eq!(c.setups, 2, "setup ran again");
        // The retry replayed BOTH params through `on_param_changed` (2 at construction, 2 on the
        // retry). `update_param` delivering its own edit on top would read 5 — and double-apply
        // whatever side effect the handler carries.
        assert_eq!(c.param_calls, 4, "the edit reached the node once, through the replay");
    }

    #[test]
    fn a_failed_retry_keeps_the_node_uninitialized_and_still_accepts_the_edit() {
        // An edit that does NOT fix what broke `setup()`. Refusing it (returning Err) would refuse
        // the very interaction that is the retry door, and `update_param` is a command whose
        // inverse must stay in step with the session's history.
        let mut g = Graph::new();
        let (n, counts) = gated_setup_node(&mut g);
        let picked = Param::Str { value: "hw:1".into(), options: None, refresh: true };
        assert!(g.update_param(n, "boot", "device", picked).is_ok(), "the edit is stored, not refused");
        assert_eq!(g.last_error(n), Some("device is not open"), "the node is still uninitialized");
        assert_eq!(g.node_stage(n), "error");
        {
            let c = counts.lock().unwrap();
            assert_eq!(c.setups, 2, "the interaction retried the initialization");
            assert_eq!(c.param_calls, 4, "the failed retry replayed the params; nothing was applied twice");
            assert_eq!(c.runs, 0, "and no callback ran against an uninitialized node");
        }
        g.tick();
        assert_eq!(counts.lock().unwrap().runs, 0, "a tick still cannot enter process");
    }

    #[test]
    fn refreshing_a_param_on_an_uninitialized_node_refuses_and_names_the_failure() {
        // A refresh answers the UI with a LIST, and there is none. `Ok(None)` would read as "this
        // node implements no hook" — a different, misleading answer — so the refusal says why.
        // The node's hook DOES return a list, so a missing gate reads as a successful scan.
        let mut g = Graph::new();
        let (n, _counts) = gated_setup_node(&mut g);
        let err = g
            .refresh_param(n, "boot", "device")
            .expect_err("an uninitialized node has no options to give");
        assert!(err.contains("device is not open"), "the refusal names the setup failure: {err}");

        // Once it initializes, the same call answers normally.
        g.update_param(n, "boot", "ok", Param::boolean(true)).unwrap();
        assert_eq!(g.refresh_param(n, "boot", "device").unwrap(), Some(vec!["dev0".to_string()]));
    }

    #[test]
    fn a_failed_setup_is_not_re_initialized_on_every_tick() {
        // The tick's retry is THROTTLED. Unthrottled, every admitted run re-ran the whole
        // `seed_node` unit — each param's `on_param_changed` plus `setup()` — on the tick thread,
        // inside the mutex the bridge holds across the entire tick. The counters are the assertion:
        // an error-message check passes identically with or without the backoff.
        let mut g = Graph::new();
        let (_n, counts) = gated_setup_node(&mut g);
        let t0 = Instant::now();
        for i in 0..5 {
            g.tick_at(t0 + Duration::from_millis(i));
        }
        let c = counts.lock().unwrap();
        assert_eq!(c.setups, 1, "only construction's setup ran; the ticks inside the window retried nothing");
        assert_eq!(c.param_calls, 2, "and no param handler was replayed against them");
    }

    #[test]
    fn a_failed_setup_retries_once_the_backoff_elapses() {
        // The throttle must not become a refusal: a node whose device appears later still heals on
        // its own, one attempt per interval — the recovery half of D3's tick retry.
        let mut g = Graph::new();
        let (_n, counts) = gated_setup_node(&mut g);
        let t0 = Instant::now();
        g.tick_at(t0);
        assert_eq!(counts.lock().unwrap().setups, 1, "inside the window");

        g.tick_at(t0 + Duration::from_secs(2));
        assert_eq!(counts.lock().unwrap().setups, 2, "the elapsed window admits exactly one retry");
        assert_eq!(counts.lock().unwrap().param_calls, 4, "which replayed the params once");

        // …and the retry restarts the window rather than opening it.
        g.tick_at(t0 + Duration::from_secs(2) + Duration::from_millis(1));
        assert_eq!(counts.lock().unwrap().setups, 2, "the tick right behind it retries nothing");
    }

    #[test]
    fn an_expression_binding_never_dispatches_into_an_uninitialized_node() {
        // The binding APPLY phase is a FOURTH caller of `on_param_changed`, and it sat outside the
        // gate — so a node whose `setup()` failed kept receiving param callbacks, against
        // `ensure_initialized`'s own contract ("nothing may run against it — not `process`, not a
        // param callback, not a refresh"). It also DOUBLE-APPLIES: the dispatch runs before
        // `run_node`'s retry in the same tick, and that retry's `seed_node` replays the same param.
        let mut g = eval_graph();
        let src = g.add_node("_TestConst", None).unwrap();
        g.rename_node(src, "src").unwrap();
        let (n, counts) = gated_setup_node(&mut g);
        g.set_expression(n, "boot", "device", "nd('src')", true, false).unwrap();

        // Three ticks inside the backoff window, each evaluating a NEW value for the bound param.
        let t0 = Instant::now();
        for i in 0..3 {
            g.update_param(src, "constant", "value", Param::float(i as f64 + 1.0, -1e9, 1e9)).unwrap();
            g.tick_at(t0 + Duration::from_millis(i));
        }
        {
            let c = counts.lock().unwrap();
            assert_eq!(c.setups, 1, "no retry inside the window");
            assert_eq!(c.param_calls, 2, "and the node heard none of the three evaluated values");
        }

        // The window elapses: the retry's replay is what finally delivers the bound value — ONCE.
        g.update_param(src, "constant", "value", Param::float(9.0, -1e9, 1e9)).unwrap();
        g.tick_at(t0 + Duration::from_secs(2));
        let c = counts.lock().unwrap();
        assert_eq!(c.setups, 2, "the elapsed window admitted the retry");
        assert_eq!(c.param_calls, 4, "each param replayed once — not the binding's dispatch on top");
    }

    #[test]
    fn remove_node_drops_links() {
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        g.add_link(src, "out", echo, "in").unwrap();
        g.remove_node(src).unwrap();
        assert!(!g.contains(src));
        g.tick(); // must not panic; echo has no input now
        assert!(g.latest_frame(echo, "out").is_none());
    }

    #[test]
    fn trigger_arbitration_gates_downstream() {
        let mut g = Graph::new();
        let src = g.add_node("_TestGated", None).unwrap(); // emits every other tick
        let cnt = g.add_node("_TestCounter", None).unwrap(); // triggered
        g.add_link(src, "out", cnt, "in").unwrap();
        for _ in 0..6 {
            g.tick();
        }
        // The gated source emits on 3 of 6 ticks, so the counter ran exactly 3 times.
        assert_eq!(first_f32(&g.latest_frame(cnt, "out").expect("counter ran")), 3.0);
    }

    #[test]
    fn unwired_triggered_node_never_runs() {
        let mut g = Graph::new();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        for _ in 0..5 {
            g.tick();
        }
        assert!(
            g.latest_frame(cnt, "out").is_none(),
            "a triggered node with no wired input must never run"
        );
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
            goofi_node::param(g2.params(restored).unwrap(), "constant", "value")
                .unwrap()
                .as_f64(),
            Some(7.5)
        );

        // The link round-trips: ticking drives the echo from the restored source.
        g2.tick();
        let echo2 = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.type_name(*u) == Some("_TestEcho"))
            .unwrap();
        assert!(g2.latest_frame(echo2, "out").is_some(), "restored link must carry data");
        assert_eq!(first_f32(&g2.latest_frame(echo2, "out").unwrap()), 7.5);
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
    fn independent_nodes_run_in_parallel() {
        // Eight sources with no edges between them all sit in topo level 0, so a
        // parallel scheduler runs them concurrently. Each sleeps 20ms: a
        // sequential tick would take >= 160ms; a parallel one must finish well
        // under that. Generous bound to stay robust on a loaded machine.
        let mut g = Graph::new();
        for _ in 0..8 {
            g.add_node("_TestSlow", None).unwrap();
        }
        g.tick(); // warm the rayon pool (first use pays thread-spawn cost)
        let t = std::time::Instant::now();
        g.tick();
        let elapsed = t.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "8 independent 20ms nodes took {elapsed:?}; expected concurrent execution (< 100ms)"
        );
    }

    #[test]
    fn independent_branches_both_produce_correctly() {
        // Two disjoint _TestConst -> Echo branches must both propagate in one
        // tick regardless of the parallel scheduling of their level-0 sources.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let ea = g.add_node("_TestEcho", None).unwrap();
        g.update_param(a, "constant", "value", Param::float(3.0, -1e9, 1e9)).unwrap();
        g.add_link(a, "out", ea, "in").unwrap();

        let b = g.add_node("_TestConst", None).unwrap();
        let eb = g.add_node("_TestEcho", None).unwrap();
        g.update_param(b, "constant", "value", Param::float(4.0, -1e9, 1e9)).unwrap();
        g.add_link(b, "out", eb, "in").unwrap();

        g.tick();
        assert_eq!(first_f32(&g.latest_frame(ea, "out").unwrap()), 3.0);
        assert_eq!(first_f32(&g.latest_frame(eb, "out").unwrap()), 4.0);
    }

    // ---- detached (Subprocess-isolated) execution scaffolding ----
    //
    // A blocking test node that runs on the detached worker WITHOUT a real subprocess: it
    // records each job's arrival (the input's first f32) then waits for a permit, so a test
    // controls exactly when the worker proceeds. `open()` releases the gate for good — used
    // at teardown so a blocked worker can drain, see the shutdown signal and exit.
    struct Gate {
        mtx: std::sync::Mutex<GateInner>,
        cv: std::sync::Condvar,
    }
    struct GateInner {
        permits: u32,
        calls: Vec<f32>,
    }
    impl Gate {
        fn new() -> std::sync::Arc<Gate> {
            std::sync::Arc::new(Gate {
                mtx: std::sync::Mutex::new(GateInner { permits: 0, calls: Vec::new() }),
                cv: std::sync::Condvar::new(),
            })
        }
        fn release(&self) {
            self.mtx.lock().unwrap().permits += 1;
            self.cv.notify_one();
        }
        fn open(&self) {
            self.mtx.lock().unwrap().permits = u32::MAX;
            self.cv.notify_all();
        }
        fn calls(&self) -> Vec<f32> {
            self.mtx.lock().unwrap().calls.clone()
        }
        /// Block the test thread until the worker has started at least `n` jobs.
        fn wait_calls(&self, n: usize) {
            for _ in 0..1000 {
                if self.calls().len() >= n {
                    return;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            panic!("worker never reached {n} calls (got {})", self.calls().len());
        }
    }

    struct GateNode {
        gate: std::sync::Arc<Gate>,
        // Bumped on Drop so a teardown test can observe the worker dropping its node.
        on_drop: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
        fail: bool,
    }
    impl Drop for GateNode {
        fn drop(&mut self) {
            if let Some(c) = &self.on_drop {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }
    impl Node for GateNode {
        fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let first = inp
                .get("data")
                .and_then(|d| match d.value() {
                    Value::Array(s) => Some(f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap())),
                    _ => None,
                })
                .unwrap_or(0.0);
            self.gate.mtx.lock().unwrap().calls.push(first); // record arrival before blocking
            if self.fail {
                return Err("gate failure".into());
            }
            {
                let mut g = self.gate.mtx.lock().unwrap();
                while g.permits == 0 {
                    g = self.gate.cv.wait(g).unwrap();
                }
                g.permits -= 1;
            }
            let d = Data::array_f32(vec![1], first.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }

    static GATE_IN: &[SlotDecl] =
        &[SlotDecl { name: "data", kind: SlotType::Array, trigger_process: true, multi: false, required: false }];
    static GATE_OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
    static GATE_MANIFEST: NodeManifest = NodeManifest {
        type_name: "GateSubproc",
        category: "test",
        doc: "blocking detached test node (no real subprocess)",
        inputs: GATE_IN,
        outputs: GATE_OUT,
        params: NO_PARAMS,
        isolation: Isolation::Subprocess,
        producer: false,
        factory: rt_stub_factory,
    };

    /// A detached node that blocks inside `setup()` — the shape of a Python child paying its
    /// spawn + import cost, which is what the boot spinner is for. `fail` makes the bootstrap
    /// end in an error once released, which is the other half of what that window is for.
    struct SeedingNode {
        gate: std::sync::Arc<Gate>,
        fail: bool,
    }
    impl Node for SeedingNode {
        fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.gate.mtx.lock().unwrap().calls.push(0.0); // announce arrival, then block
            self.gate.cv.notify_all();
            let mut g = self.gate.mtx.lock().unwrap();
            while g.permits == 0 {
                g = self.gate.cv.wait(g).unwrap();
            }
            g.permits -= 1;
            if self.fail {
                return Err("boot failed".into());
            }
            Ok(())
        }
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            // 1.0 marks a dispatched JOB, distinct from setup's 0.0, so a test can tell whether a
            // job ran at all — and the frame proves its `Done` was drained by a tick.
            self.gate.mtx.lock().unwrap().calls.push(1.0);
            let d = Data::array_f32(vec![1], 7.0f32.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static SEEDING_MANIFEST: NodeManifest = NodeManifest {
        type_name: "GateSeeding",
        category: "test",
        doc: "detached node that blocks in setup()",
        inputs: &[],
        outputs: GATE_OUT,
        params: NO_PARAMS,
        isolation: Isolation::Subprocess,
        producer: true,
        factory: rt_stub_factory,
    };

    fn register_gate_seeding(g: &mut Graph, gate: std::sync::Arc<Gate>, fail: bool) {
        g.register_dyn_type(
            &SEEDING_MANIFEST,
            Box::new(move |_p| Box::new(SeedingNode { gate: gate.clone(), fail })),
        );
    }

    /// Block until a detached node's worker has finished bootstrapping. Job dispatch is gated on
    /// `STAGE_READY`, so a test that ticks before its worker is up would silently skip its job.
    fn wait_bootstrapped(g: &Graph, uid: Uid) {
        for _ in 0..1000 {
            if !matches!(g.node_stage(uid), "creating" | "setup") {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("worker never bootstrapped (stage {})", g.node_stage(uid));
    }

    fn register_gate(
        g: &mut Graph,
        gate: std::sync::Arc<Gate>,
        on_drop: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
        fail: bool,
    ) {
        g.register_dyn_type(
            &GATE_MANIFEST,
            Box::new(move |_p| Box::new(GateNode { gate: gate.clone(), on_drop: on_drop.clone(), fail })),
        );
    }

    #[test]
    fn detached_node_does_not_block_the_tick() {
        let gate = Gate::new();
        let mut g = Graph::new();
        register_gate(&mut g, gate.clone(), None, false);
        let src = g.add_node("_TestConst", None).unwrap();
        let det = g.add_node("GateSubproc", None).unwrap();
        g.add_link(src, "out", det, "data").unwrap();
        wait_bootstrapped(&g, det); // dispatch is gated on READY

        let t0 = Instant::now();
        g.tick_at(t0); // dispatches a job; the worker will block on the permit
        assert!(t0.elapsed() < Duration::from_millis(50), "tick did not block on the busy worker");
        gate.wait_calls(1); // the worker took the job (proving it ran off-tick)

        gate.open(); // let it (and future jobs) complete
        let mut got = false;
        for i in 1..200 {
            g.tick_at(t0 + Duration::from_millis(10 * i));
            if g.latest_frame(det, "out").is_some() {
                got = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(3));
        }
        assert!(got, "the detached node's output propagated on a later tick");
    }

    #[test]
    fn detached_dispatch_coalesces_latest_wins() {
        // While the worker is blocked on job 1, three more dispatches with changing values
        // collapse in the latest-wins inbox — the worker runs the FIRST and the LAST only.
        let gate = Gate::new();
        let mut g = Graph::new();
        register_gate(&mut g, gate.clone(), None, false);
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(1.0, -1.0e9, 1.0e9)).unwrap();
        let det = g.add_node("GateSubproc", None).unwrap();
        g.add_link(src, "out", det, "data").unwrap();
        wait_bootstrapped(&g, det); // dispatch is gated on READY

        let t0 = Instant::now();
        g.tick_at(t0); // dispatch job(value=1); worker takes it and blocks
        gate.wait_calls(1);
        for (i, v) in [2.0f32, 3.0, 4.0].iter().enumerate() {
            g.update_param(src, "constant", "value", Param::float(*v as f64, -1.0e9, 1.0e9)).unwrap();
            g.tick_at(t0 + Duration::from_millis(10 * (i as u64 + 1))); // 2 and 3 coalesce into 4
        }
        gate.release(); // finish job 1 → worker takes the coalesced job(value=4)
        gate.wait_calls(2);
        assert_eq!(gate.calls(), vec![1.0, 4.0], "middle jobs coalesced; only first + last ran");
        gate.open(); // teardown: let the worker drain + idle so it sees the shutdown signal
    }

    #[test]
    fn removing_a_detached_node_reaps_its_worker() {
        // No tick → the worker seeds then idles on the inbox. remove_node drops the handle, which
        // signals shutdown; the idle worker wakes, exits and drops the node (reaping any child
        // process through its own Drop). The signal is fire-and-forget — the caller never waits on
        // the worker — so this is a bounded poll, like its restart sibling.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let gate = Gate::new();
        let dropped = std::sync::Arc::new(AtomicUsize::new(0));
        let mut g = Graph::new();
        register_gate(&mut g, gate.clone(), Some(dropped.clone()), false);
        let det = g.add_node("GateSubproc", None).unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 0, "node still alive on its worker");

        g.remove_node(det).unwrap();

        for _ in 0..500 {
            if dropped.load(Ordering::SeqCst) == 1 {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("the removed instance was never dropped (got {})", dropped.load(Ordering::SeqCst));
    }

    #[test]
    fn removing_a_busy_detached_node_does_not_block_the_graph() {
        // The sibling of `restarting_a_busy_detached_node_does_not_block_the_graph`, for the OTHER
        // three teardown paths (delete, batch delete, undo-of-add, load). The bridge holds the
        // graph mutex across `remove_node`, and the worker only observes shutdown between jobs —
        // so waiting on a worker parked inside a blocked `process()` would freeze the tick, every
        // viewer and every other RPC for the rest of that call.
        let gate = Gate::new();
        let mut g = Graph::new();
        register_gate(&mut g, gate.clone(), None, false);
        let src = g.add_node("_TestConst", None).unwrap();
        let det = g.add_node("GateSubproc", None).unwrap();
        g.add_link(src, "out", det, "data").unwrap();
        wait_bootstrapped(&g, det); // dispatch is gated on READY
        g.tick(); // dispatches a job; the worker blocks on the permit, inside process()
        gate.wait_calls(1);

        // Stand in for the backend's own timeout releasing the stuck call (a subprocess roundtrip
        // gives up after 10s). It bounds the failure so a regression is a measured wait rather
        // than a hung suite, and doubles as the worker's cleanup.
        let releaser = gate.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(800));
            releaser.open();
        });

        let t0 = Instant::now();
        g.remove_node(det).unwrap();
        let blocked_for = t0.elapsed();

        assert!(
            blocked_for < Duration::from_millis(500),
            "remove_node returned only after {blocked_for:?} — it waited on the busy worker"
        );
    }

    #[test]
    fn detached_process_error_surfaces_on_the_error_channel() {
        let gate = Gate::new();
        let mut g = Graph::new();
        register_gate(&mut g, gate.clone(), None, true); // process() returns Err immediately
        let src = g.add_node("_TestConst", None).unwrap();
        let det = g.add_node("GateSubproc", None).unwrap();
        g.add_link(src, "out", det, "data").unwrap();

        let t0 = Instant::now();
        let mut err = None;
        for i in 0..200 {
            g.tick_at(t0 + Duration::from_millis(5 * i));
            if let Some(e) = g.last_error(det) {
                err = Some(e.to_string());
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(err.as_deref(), Some("gate failure"), "the detached process error surfaced");
    }

    #[test]
    fn a_worker_whose_bootstrap_failed_is_never_given_a_job() {
        // Two halves of one contract. The dispatch gate must refuse a worker whose `setup` failed —
        // it is uninitialized, and "ticks of a node that had a setup() error should not be
        // possible" (D3). And the failure must STAND: `outbox` is a latest-wins single slot, so a
        // bootstrap failure posted there would be erased by any later `Done` before a tick drained
        // it, and the node would report healthy though its `setup` failed (the silent case: a param
        // `seed_node` folded in). It is latched off that channel for exactly that reason.
        //
        // Parking the worker inside `setup` makes the ordering exact rather than a race: the tick
        // below runs while the bootstrap is still in flight, and the release then fails it.
        let gate = Gate::new();
        let mut g = Graph::new();
        register_gate_seeding(&mut g, gate.clone(), true);
        let det = g.add_node("GateSeeding", None).unwrap();
        gate.wait_calls(1); // parked inside setup()

        let t0 = Instant::now();
        g.tick_at(t0);
        gate.open(); // setup returns Err and the worker reaches READY, failed
        wait_bootstrapped(&g, det);

        // Every tick from here is one the failed worker must NOT be fed.
        for i in 1..40 {
            g.tick_at(t0 + Duration::from_millis(10 * i));
            std::thread::sleep(Duration::from_millis(1));
        }
        // `SeedingNode` pushes 0.0 from `setup` and 1.0 from `process`, so the call log says
        // whether a job ever reached it — an assertion on the error alone would also pass while
        // jobs ran, which is the hole this used to be written around.
        assert_eq!(gate.calls(), vec![0.0], "the bootstrap ran; no process() job ever followed it");
        assert!(g.latest_frame(det, "out").is_none(), "so the node emitted nothing");
        assert_eq!(g.last_error(det), Some("boot failed"), "and the bootstrap failure stands");
        assert_eq!(g.node_stage(det), "error", "the editor sees it as errored, not ready");
    }

    #[test]
    fn no_job_is_dispatched_while_the_worker_is_still_in_setup() {
        // A job built while `setup` is still running was snapshotted from PRE-setup state, so it is
        // stale by the time the worker could run it — and racing it against the bootstrap is what
        // lets a `Done` erase the failure. The gate makes that window observable.
        let gate = Gate::new();
        let mut g = Graph::new();
        register_gate_seeding(&mut g, gate.clone(), false);
        let det = g.add_node("GateSeeding", None).unwrap();
        gate.wait_calls(1); // parked inside setup()
        assert_eq!(g.node_stage(det), "setup", "still booting");

        let t0 = Instant::now();
        g.tick_at(t0);
        g.tick_at(t0 + Duration::from_millis(10));

        gate.open(); // setup completes — nothing may be queued behind it
        wait_bootstrapped(&g, det);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(gate.calls(), vec![0.0], "the bootstrap ran; no process() job followed it");

        // ...and the node is not starved for it: the first tick after READY feeds the worker.
        g.tick_at(t0 + Duration::from_millis(500));
        gate.wait_calls(2);
    }

    #[test]
    fn a_healthy_detached_node_never_reports_a_bootstrap_error() {
        // The mirror-image of the latch's failure mode: making a failed `setup` visible must never
        // make a working node look broken.
        let gate = Gate::new();
        gate.open();
        let mut g = Graph::new();
        register_gate_seeding(&mut g, gate.clone(), false);
        let det = g.add_node("GateSeeding", None).unwrap();
        wait_bootstrapped(&g, det);
        assert_eq!(g.node_stage(det), "ready", "a clean bootstrap is ready, not errored");

        let t0 = Instant::now();
        for i in 0..400 {
            g.tick_at(t0 + Duration::from_millis(10 * i));
            if g.latest_frame(det, "out").is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(g.latest_frame(det, "out").is_some(), "the worker ran a job");
        assert_eq!(g.last_error(det), None, "a healthy worker reports no error");
    }

    /// A detached type whose FIRST instance fails `setup()` and whose later ones succeed — the
    /// shape `restart_node` exists to rescue.
    static BOOT_ONCE_MANIFEST: NodeManifest = NodeManifest {
        type_name: "GateBootOnce",
        category: "test",
        doc: "detached node whose first instance fails setup()",
        inputs: &[],
        outputs: GATE_OUT,
        params: NO_PARAMS,
        isolation: Isolation::Subprocess,
        producer: true,
        factory: rt_stub_factory,
    };
    struct BootOnceNode {
        fail: bool,
    }
    impl Node for BootOnceNode {
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

    #[test]
    fn a_worker_whose_bootstrap_failed_stops_asking_the_tick_loop_to_wake() {
        // `next_run_delay` paces the tick loop, and `entry.last_run = Some(now)` on DISPATCH is a
        // detached node's only writer of `last_run`. Since the dispatch gate refuses a boot-failed
        // worker permanently, `last_run` stays `None` for the life of the patch — and the scan's
        // `None => Duration::ZERO` arm then answers "run me now" forever, ignoring the node's own
        // cap. The bridge clamps ZERO to `LOCK_CEDE`, so one 30 Hz node pinned the loop at ~10 kHz
        // on the graph mutex.
        let mut g = Graph::new();
        g.register_dyn_type(&BOOT_ONCE_MANIFEST, Box::new(|_p| Box::new(BootOnceNode { fail: true })));
        let det = g.add_node("GateBootOnce", None).unwrap();
        wait_bootstrapped(&g, det);
        assert_eq!(g.last_error(det), Some("boot failed"), "the worker latched its bootstrap failure");

        g.update_param(det, "common", "max_frequency", Param::float(30.0, 0.0, 1e9)).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // refused by the dispatch gate, so `last_run` is still None
        assert_eq!(
            g.next_run_delay(t0 + Duration::from_millis(1)),
            None,
            "a node the tick permanently refuses must not ask the loop to wake for it"
        );
    }

    #[test]
    fn restarting_a_failed_detached_node_clears_its_bootstrap_error() {
        // A bootstrap error that outlived the instance that earned it would leave a respawned,
        // healthy node reporting a corpse's failure forever. It is sticky for the WORKER's
        // lifetime only — `restart_node` installs a fresh handle, whose latch starts empty.
        let builds = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let b = builds.clone();
        let mut g = Graph::new();
        g.register_dyn_type(
            &BOOT_ONCE_MANIFEST,
            Box::new(move |_p| {
                let n = b.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::new(BootOnceNode { fail: n == 0 })
            }),
        );
        let det = g.add_node("GateBootOnce", None).unwrap();
        wait_bootstrapped(&g, det);
        assert_eq!(g.last_error(det), Some("boot failed"), "the first instance failed to boot");

        g.restart_node(det).unwrap();
        wait_bootstrapped(&g, det);
        assert_eq!(builds.load(std::sync::atomic::Ordering::SeqCst), 2, "a fresh instance was built");
        assert_eq!(g.last_error(det), None, "the respawn does not inherit the corpse's error");
        assert_eq!(g.node_stage(det), "ready");

        let t0 = Instant::now();
        for i in 0..400 {
            g.tick_at(t0 + Duration::from_millis(10 * i));
            if g.latest_frame(det, "out").is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(first_f32(&g.latest_frame(det, "out").unwrap()), 7.0, "the new instance runs");
        assert_eq!(g.last_error(det), None, "and stays healthy");
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
        assert_eq!(g.last_error(uid), Some("boot failed"), "the first instance failed to boot");

        g.restart_node(uid).unwrap();

        assert_eq!(builds.load(std::sync::atomic::Ordering::SeqCst), 2, "a fresh instance was built");
        assert_eq!(g.last_error(uid), None, "restart clears the recovered node's error");
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(uid, "out").unwrap()), 7.0, "the new instance runs");
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
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(uid, "out").unwrap()), 5.0, "params carried over");
    }

    #[test]
    fn restart_keeps_every_wire_of_a_multi_input_in_connection_order() {
        // The per-wire cells live inside the node entry while the links live on the graph: a
        // restart that forgets to rebuild them leaves the slot silently dead.
        let mut g = Graph::new();
        let a = const_src(&mut g, 1.0);
        let b = const_src(&mut g, 2.0);
        let c = const_src(&mut g, 3.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        g.add_link(a, "out", col, "ins").unwrap();
        g.add_link(b, "out", col, "ins").unwrap();
        g.add_link(c, "out", col, "ins").unwrap();
        g.tick();
        assert_eq!(as_f32_vec(&g.latest_frame(col, "out").unwrap()), vec![3.0, 1.0, 2.0, 3.0]);

        g.restart_node(col).unwrap();
        g.tick();

        assert_eq!(
            as_f32_vec(&g.latest_frame(col, "out").unwrap()),
            vec![3.0, 1.0, 2.0, 3.0],
            "all three wires still feed the restarted node, in connection order"
        );
    }

    #[test]
    fn restart_keeps_a_param_expression_binding_live() {
        let mut g = eval_graph();
        let uid = const_src(&mut g, 1.0);
        g.set_expression(uid, "constant", "value", "42", true, false).unwrap();
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(uid, "out").unwrap()), 42.0);

        g.restart_node(uid).unwrap();
        g.tick();

        assert_eq!(
            g.param_expression(uid, "constant", "value").map(|e| e.source),
            Some("42".to_string()),
            "the compiled binding is untouched by a restart"
        );
        assert_eq!(first_f32(&g.latest_frame(uid, "out").unwrap()), 42.0);
    }

    #[test]
    fn restarting_a_detached_node_reaps_the_old_worker() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let gate = Gate::new();
        gate.open();
        let dropped = std::sync::Arc::new(AtomicUsize::new(0));
        let mut g = Graph::new();
        register_gate(&mut g, gate.clone(), Some(dropped.clone()), false);
        let det = g.add_node("GateSubproc", None).unwrap();

        g.restart_node(det).unwrap();

        // The replaced handle's worker still exits and drops the instance (reaping the child
        // process through the node's own Drop) — on its own thread rather than under the caller's
        // graph lock, so this is a bounded poll rather than an immediate read.
        for _ in 0..500 {
            if dropped.load(Ordering::SeqCst) == 1 {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("the replaced instance was never dropped (got {})", dropped.load(Ordering::SeqCst));
    }

    // A Subprocess-isolated type WITH a refreshable param, for the tier-refusal test.
    static GATE_PICKER_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "audio",
        name: "device",
        spec: ParamSpec::Str { default: "none", options: &["none"], refresh: true },
        expression: None,
        doc: None,
    }];
    static GATE_PICKER_MANIFEST: NodeManifest = NodeManifest {
        type_name: "GateSubprocPicker",
        category: "test",
        doc: "detached node with a refreshable param",
        inputs: GATE_IN,
        outputs: GATE_OUT,
        params: GATE_PICKER_PARAMS,
        isolation: Isolation::Subprocess,
        producer: false,
        factory: rt_stub_factory,
    };

    #[test]
    fn a_detached_node_reports_its_bootstrap_stage() {
        // The spinner exists for this: a Python child is spawn + import + setup. The gate holds
        // the worker inside setup, so the stage is observable rather than a race.
        let gate = Gate::new();
        let mut g = Graph::new();
        register_gate_seeding(&mut g, gate.clone(), false);
        let det = g.add_node("GateSeeding", None).unwrap();

        // The worker is parked in its `setup()`.
        gate.wait_calls(1);
        assert_eq!(g.node_stage(det), "setup", "still booting");

        gate.open();
        for _ in 0..500 {
            if g.node_stage(det) == "ready" {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(g.node_stage(det), "ready", "bootstrap finished");
    }

    #[test]
    fn an_inline_node_is_ready_immediately_and_errors_are_reported_as_such() {
        // Nothing to wait for: an inline node is seeded before it is visible.
        let mut g = Graph::new();
        let n = g.add_node("_TestConst", None).unwrap();
        assert_eq!(g.node_stage(n), "ready");
        assert_eq!(g.node_stage(Uid(9999)), "error", "an unknown node is not `ready`");
    }

    #[test]
    fn restarting_a_busy_detached_node_does_not_block_the_graph() {
        // The whole point of the restart button is rescuing a node that is stuck. The worker only
        // observes the shutdown signal between jobs, so waiting on the old handle would hold the
        // graph mutex for the rest of a blocked process() call (up to a subprocess roundtrip's 10s
        // timeout), freezing the tick, every viewer and every other RPC. The restart must never be
        // the thing that freezes the app it is rescuing.
        let gate = Gate::new();
        let mut g = Graph::new();
        register_gate(&mut g, gate.clone(), None, false);
        let src = g.add_node("_TestConst", None).unwrap();
        let det = g.add_node("GateSubproc", None).unwrap();
        g.add_link(src, "out", det, "data").unwrap();
        wait_bootstrapped(&g, det); // dispatch is gated on READY
        g.tick(); // dispatches a job; the worker blocks on the permit, inside process()
        gate.wait_calls(1);

        let t0 = Instant::now();
        g.restart_node(det).unwrap();
        let blocked_for = t0.elapsed();

        assert!(
            blocked_for < Duration::from_millis(500),
            "restart returned only after {blocked_for:?} — it waited on the busy worker"
        );
        gate.open(); // let the reaped worker finish rather than leaking it
    }

    #[test]
    fn restart_keeps_the_frames_already_delivered_to_its_inputs() {
        // Input cells are latest-wins caches, not instance state. Keep one producer running and
        // silence the other: if a restart dropped the cells, the silent producer's wire would
        // vanish from the fan-in and the node would emit a SHORTER list. (Asserting on
        // `latest_frame` alone cannot see this — it replays the last emitted frame when the node
        // does not run at all.)
        let mut g = Graph::new();
        let fast = const_src(&mut g, 1.0);
        let slow = const_src(&mut g, 2.0);
        let col = g.add_node("_TestCollect", None).unwrap();
        g.add_link(fast, "out", col, "ins").unwrap();
        g.add_link(slow, "out", col, "ins").unwrap();
        g.tick();
        assert_eq!(as_f32_vec(&g.latest_frame(col, "out").unwrap()), vec![2.0, 1.0, 2.0]);

        // Park the slow producer: a rate cap of 0.001 Hz means it will not run again in this
        // test. (Clearing `autotrigger` would NOT silence it — a node with no triggering input
        // has nothing to wait for and free-runs regardless.)
        g.update_param(slow, "common", "max_frequency", Param::float(0.001, 0.0, 1e9)).unwrap();
        g.restart_node(col).unwrap();
        g.tick(); // `fast` emits again and re-triggers the node; `slow` stays quiet

        assert_eq!(
            as_f32_vec(&g.latest_frame(col, "out").unwrap()),
            vec![2.0, 1.0, 2.0],
            "the silent wire kept the frame it was given before the restart"
        );
    }

    #[test]
    fn refreshing_a_param_on_the_subprocess_tier_reports_that_it_cannot() {
        // The request/response codec has no refresh op, so a detached node cannot answer one.
        // Reporting success would echo a stale list as though it had just been re-scanned.
        let gate = Gate::new();
        let mut g = Graph::new();
        g.register_dyn_type(
            &GATE_PICKER_MANIFEST,
            Box::new(move |_p| Box::new(GateNode { gate: gate.clone(), on_drop: None, fail: false })),
        );
        let det = g.add_node("GateSubprocPicker", None).unwrap();

        let err = g.refresh_param(det, "audio", "device").unwrap_err();

        assert!(err.contains("subprocess"), "the error names the tier that cannot answer: {err}");
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

        let fresh = g.refresh_param(uid, "audio", "device").unwrap();

        assert_eq!(fresh, Some(vec!["dev0".to_string()]));
        assert_eq!(options_of(&g, uid, "audio", "device"), Some(vec!["dev0".to_string()]));
        // A second click re-scans rather than replaying a cached list.
        g.refresh_param(uid, "audio", "device").unwrap();
        assert_eq!(options_of(&g, uid, "audio", "device"), Some(vec!["dev0".into(), "dev1".into()]));
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
        // `has_trigger_inputs` is a CACHED field (set once at construction), and the scheduler's
        // free-run decision reads it — so a stale one changes when the node runs. Read directly:
        // a child module sees its parent's private fields, and production gains no test accessor.
        assert!(!g.nodes[&uid].has_trigger_inputs, "the trigger flag was recomputed, not carried");
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
    fn a_panicking_lifecycle_hook_becomes_a_node_error_and_leaves_the_lock_usable() {
        // `execute_node` wraps `process` in catch_unwind precisely because a node is third-party
        // code. The other hooks run under the SAME graph lock the bridge holds, and this codebase
        // locks with `.lock().unwrap()` throughout — so an unguarded panic there poisons the mutex
        // and every later lock in the bridge and the tick thread fails from then on. A node bug
        // becomes total loss of the control plane. Containment has to be uniform, not per-hook.
        let g = std::sync::Arc::new(std::sync::Mutex::new(Graph::new()));
        let uid = {
            let mut gg = g.lock().unwrap();
            gg.register_dyn_type(&PANICKY_MANIFEST, Box::new(|_| Box::new(PanickyHooks)));
            gg.add_node("_Panicky", None).unwrap()
        };
        {
            let mut gg = g.lock().unwrap();
            let r = gg.update_param(uid, "danger", "boom", Param::boolean(true));
            assert!(r.is_err(), "the panic reaches the caller as an error: {r:?}");
            assert!(r.unwrap_err().contains("panic"), "and says it was a panic");
        }
        // The lock IS the assertion: a poisoned mutex makes every later `.lock().unwrap()` panic.
        assert!(g.lock().is_ok(), "the graph mutex is not poisoned");
        assert!(g.lock().unwrap().contains(uid), "and the graph is still readable");
    }

    #[test]
    fn a_panicking_setup_is_the_nodes_boot_error_not_a_lost_process() {
        // `seed_node` runs `on_param_changed` then `setup` at construction — inside `add_node`,
        // under the same lock. A panic here used to unwind straight through `Graph::add_node`.
        let g = std::sync::Arc::new(std::sync::Mutex::new(Graph::new()));
        {
            let mut gg = g.lock().unwrap();
            gg.register_dyn_type(&PANICKY_MANIFEST, Box::new(|_| Box::new(PanickyHooks)));
            // A node born with the param already set panics during its seed replay.
            let mut params = gg.default_params_of("_Panicky").unwrap();
            params.get_mut("danger").unwrap().insert("boom".into(), Param::boolean(true));
            let uid = gg.add_node("_Panicky", Some(params)).unwrap();
            // On the INITIALIZATION channel: the replay is half of `seed_node`, so a panic in it
            // leaves the node uninitialized exactly as a failed `setup()` does.
            assert!(
                gg.nodes[&uid].setup_error.as_deref().is_some_and(|e| e.contains("panic")),
                "the panic is the node's error: {:?}",
                gg.nodes[&uid].setup_error
            );
        }
        assert!(g.lock().is_ok(), "the graph mutex is not poisoned");
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

        let r = g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 2.0 })));
        assert_eq!(r, Registration::Replaced);
        assert_eq!(g.dyn_type_manifests().len(), 1, "a replace does not add a second entry");

        let new = g.add_node("_RuntimeDyn", None).unwrap();
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(new, "out").unwrap()), 2.0, "new factory");
        assert_eq!(first_f32(&g.latest_frame(old, "out").unwrap()), 1.0, "live instance untouched");
    }

    #[test]
    fn removing_a_dyn_type_takes_it_out_of_the_catalog_and_out_of_resolution() {
        // The other half of the rescan: the file vanished, so the type must stop being addable —
        // while the instance that is already running stays running.
        let mut g = Graph::new();
        g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        let live = g.add_node("_RuntimeDyn", None).unwrap();

        assert!(g.remove_dyn_type("_RuntimeDyn"));
        assert!(g.dyn_type_manifests().is_empty(), "gone from the palette");
        // `known_type` is private, and `add_node` is its door: the refusal must read as the
        // vanished type it is, not as a dependency-missing "unavailable" one.
        assert_eq!(g.add_node("_RuntimeDyn", None).unwrap_err(), "unknown node type `_RuntimeDyn`");
        assert!(!g.remove_dyn_type("_RuntimeDyn"), "nothing left to remove");

        g.tick();
        assert_eq!(first_f32(&g.latest_frame(live, "out").unwrap()), 1.0, "the instance still runs");
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
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(uid, "out").unwrap()), 42.0);
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
    fn diamond_converges_through_levels_in_one_tick() {
        // src -> echoA, src -> echoB, {echoA,echoB} -> adder. Levels: src(0),
        // {echoA,echoB}(1, parallel), adder(2). The adder must see BOTH branch
        // outputs — proving level-2 propagation waits for the whole level-1 batch.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(5.0, -1e9, 1e9)).unwrap();
        let ea = g.add_node("_TestEcho", None).unwrap();
        let eb = g.add_node("_TestEcho", None).unwrap();
        let add = g.add_node("_TestAdder", None).unwrap();
        g.add_link(src, "out", ea, "in").unwrap();
        g.add_link(src, "out", eb, "in").unwrap();
        g.add_link(ea, "out", add, "a").unwrap();
        g.add_link(eb, "out", add, "b").unwrap();

        g.tick();
        assert_eq!(first_f32(&g.latest_frame(add, "out").expect("adder produced")), 10.0);
    }

    #[test]
    fn cycle_is_tolerated_without_hanging() {
        // A pure 2-cycle of triggered nodes (echoA -> echoB -> echoA) has no
        // level-0 seed: both land in the cycle-remainder final level. tick() must
        // terminate (not spin) and, unseeded, produce nothing.
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", a, "in").unwrap();
        g.tick(); // must return
        assert!(g.latest_frame(a, "out").is_none());
        assert!(g.latest_frame(b, "out").is_none());
    }

    #[test]
    fn sustained_load_reference_stress_shape_stays_stable() {
        use std::time::Duration;
        // The reference stress-patch shape: one Oscillator fanning out to 8 Buffers —
        // all at topo level 1, so they run concurrently on the pool each tick. Drive it
        // hard and assert every consumer keeps producing with a clean error channel
        // (sustained parallel stability, no drift into a faulted state).
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let mut buffers = Vec::new();
        for _ in 0..8 {
            let b = g.add_node("Buffer", None).unwrap();
            g.add_link(osc, "out", b, "data").unwrap();
            buffers.push(b);
        }

        // Advance a synthetic clock 10 ms/tick so the wall-clock-paced Oscillator
        // emits a real block each tick (default 1 kHz -> ~10 samples) and keeps its
        // consumers fed — a tight `tick()` loop would pass no time and starve them.
        let t0 = Instant::now();
        for i in 0..5000u64 {
            g.tick_at(t0 + Duration::from_millis(10 * i));
        }

        assert!(g.last_error(osc).is_none(), "oscillator faulted: {:?}", g.last_error(osc));
        for b in &buffers {
            assert!(g.last_error(*b).is_none(), "buffer faulted: {:?}", g.last_error(*b));
            assert!(g.latest_frame(*b, "out").is_some(), "each buffer must keep producing");
        }
    }

    #[test]
    fn generator_stamps_fresh_incrementing_index() {
        // A source (no index-bearing input) gets a fresh per-output counter that
        // advances once per emit: after 3 ticks the latest frame carries index 2.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        for _ in 0..3 {
            g.tick();
        }
        let f = g.latest_frame(src, "out").expect("frame");
        assert_eq!(f.meta().index(), Some(2), "3 emits -> indices 0,1,2 (latest 2)");
    }

    #[test]
    fn next_run_delay_zero_for_unbounded_producer() {
        // A source with no rate cap (max_frequency <= 0) is always due — the adaptive
        // tick loop must run it as fast as possible, not ceiling it at a fixed rate.
        let mut g = Graph::new();
        g.add_node("_TestConst", None).unwrap(); // no `common` group -> unbounded, no inputs
        g.tick();
        assert_eq!(
            g.next_run_delay(Instant::now()),
            Some(Duration::ZERO),
            "unbounded producer -> zero delay (as fast as possible)"
        );
    }

    #[test]
    fn next_run_delay_respects_the_rate_cap() {
        // A 10 Hz autotrigger source: after running, the next run is within its 0.1s
        // period — the cap, not a hardcoded tick rate, sets the pace.
        let mut g = Graph::new();
        g.add_node("_TestCapped", None).unwrap();
        g.tick();
        let d = g.next_run_delay(Instant::now()).expect("a capped producer still wants to run");
        assert!(d <= Duration::from_millis(100), "within the 10 Hz period, got {d:?}");
    }

    #[test]
    fn inline_node_still_ticks_through_execution_enum() {
        // An InProcess node runs via Execution::Inline unchanged.
        let mut g = Graph::new();
        let c = g.add_node("_TestConst", None).unwrap();
        g.tick_at(std::time::Instant::now());
        assert!(g.latest_frame(c, "out").is_some(), "inline execution path intact");
    }

    #[test]
    fn execute_node_stamps_index_like_the_inline_path() {
        // A pure source (no matching triggering input) gets a fresh per-output counter:
        // index 0 on its first emit, advancing to 1 on the second — proving the extracted
        // execute_node/stamp_meta_parts preserve the inline stamping behavior.
        let mut g = Graph::new();
        let c = g.add_node("_TestConst", None).unwrap();
        g.tick_at(std::time::Instant::now());
        let first = g.latest_frame(c, "out").unwrap().meta().index();
        g.tick_at(std::time::Instant::now());
        let second = g.latest_frame(c, "out").unwrap().meta().index();
        assert_eq!((first, second), (Some(0), Some(1)), "fresh per-output counter advances");
    }

    #[test]
    fn latest_frame_persists_across_non_emitting_ticks() {
        // A gated source emits every OTHER tick. latest_frame must keep returning the
        // last emitted frame on the silent ticks — viewers of a sparse / fast-ticked
        // producer see its latest data, not a None gap (the Oscillator-at-high-rate case).
        let mut g = Graph::new();
        let s = g.add_node("_TestGated", None).unwrap();
        g.tick(); // n=0 -> emits
        assert!(g.latest_frame(s, "out").is_some(), "first emit present");
        g.tick(); // n=1 -> runs but emits nothing (output reset to None)
        assert!(g.latest_frame(s, "out").is_some(), "persists last emit across a silent tick");
    }

    // A deterministic stand-in for the pyo3 evaluator, so the engine's binding lifecycle
    // + scheduling + resolution are testable without a Python interpreter. It recognizes
    // `nd('name')` (first f32 of that node's single output), `globals.name` (that global's
    // numeric value), a bare number (a constant), and `ERR` (a compile failure).
    #[derive(Default)]
    struct MockEval {
        exprs: std::sync::Mutex<HashMap<u64, MockExpr>>,
        next: std::sync::atomic::AtomicU64,
    }
    #[derive(Clone)]
    enum MockExpr {
        Ref(String),
        Global(String),
        Const(f64),
    }
    impl goofi_node::ExprEvaluator for MockEval {
        fn compile(&self, source: &str) -> Result<goofi_node::Compiled, goofi_node::ExprError> {
            if source == "ERR" {
                return Err("mock compile error".into());
            }
            let (expr, refs) = if let Some(name) =
                source.strip_prefix("nd('").and_then(|s| s.strip_suffix("')"))
            {
                (MockExpr::Ref(name.to_string()), vec![name.to_string()])
            } else if let Some(name) = source.strip_prefix("globals.") {
                (MockExpr::Global(name.to_string()), vec![])
            } else {
                let v: f64 = source.parse().map_err(|_| goofi_node::ExprError("mock: not a number".into()))?;
                (MockExpr::Const(v), vec![])
            };
            let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            self.exprs.lock().unwrap().insert(id, expr);
            // The same scanner the real evaluator uses, so dirty-tracking is exercised faithfully.
            Ok(goofi_node::Compiled { id, refs, global_refs: goofi_node::global_ref_names(source) })
        }
        fn eval(&self, id: u64, ctx: &goofi_node::EvalCtx<'_>) -> Result<Param, goofi_node::ExprError> {
            let expr = self.exprs.lock().unwrap().get(&id).cloned().ok_or_else(|| goofi_node::ExprError("mock: no such id".into()))?;
            let v: f64 = match expr {
                MockExpr::Const(c) => c,
                MockExpr::Ref(node) => match ctx.refs.get(&(node.clone(), None)).and_then(|o| o.clone()) {
                    Some(data) => first_f32(&data) as f64,
                    None => return Err(goofi_node::ExprError(format!("mock: nd('{node}') missing"))),
                },
                MockExpr::Global(name) => match ctx.globals.f64(&name) {
                    Some(v) => v,
                    None => return Err(goofi_node::ExprError(format!("mock: globals.{name} missing"))),
                },
            };
            Ok(match ctx.target {
                Param::Int { vmin, vmax, .. } => Param::Int { value: v.round() as i64, vmin: *vmin, vmax: *vmax },
                _ => Param::Float { value: v, vmin: 0.0, vmax: 0.0 },
            })
        }
        fn release(&self, id: u64) {
            self.exprs.lock().unwrap().remove(&id);
        }
    }

    fn eval_graph() -> Graph {
        let mut g = Graph::new();
        g.set_evaluator(Arc::new(MockEval::default()));
        g
    }

    #[test]
    fn constant_expression_drives_a_param_before_process() {
        // Bind _TestConst.value to the literal expression "5"; process must read 5.
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "5", true, false).unwrap();
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 5.0);
    }

    #[test]
    fn an_expression_reaches_a_param_the_node_mirrors_to_a_field() {
        // `on_param_changed` is the documented single source of truth for param→field, and
        // `update_param` dispatches it. The expression path wrote `entry.params` and stopped, so a
        // hot param (Oscillator.sfreq is the shipped case) ignored its binding while the inspector
        // showed the bound value.
        let mut g = eval_graph();
        let n = g.add_node("_TestMirror", None).unwrap();
        g.set_expression(n, "mirror", "value", "9", true, false).unwrap();
        g.tick();
        let out = as_f32_vec(&g.latest_frame(n, "out").unwrap());
        assert_eq!(out[0], 9.0, "the evaluated value reached the node's field");

        // And a settled binding does not hammer the hook: the seed call plus exactly one for the
        // expression, then nothing while the value is unchanged.
        let calls = out[1];
        g.tick();
        g.tick();
        assert_eq!(
            as_f32_vec(&g.latest_frame(n, "out").unwrap())[1],
            calls,
            "an unchanged evaluated value re-dispatches nothing"
        );
    }

    #[test]
    fn binding_oscillator_sfreq_re_rates_the_shipped_hot_param() {
        // The regression on a real node: `Oscillator.sfreq` is the library's one mirrored param,
        // and it kept emitting its seeded 250 Hz while the inspector showed (and the .gfi saved)
        // the bound value.
        use std::time::Duration;
        let mut g = eval_graph();
        let osc = g.add_node("Oscillator", None).unwrap();
        g.set_expression(osc, "oscillator", "sfreq", "200", true, false).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // anchors pacing; a source emits nothing in its first zero-length interval
        g.tick_at(t0 + Duration::from_millis(100));
        let f = g.latest_frame(osc, "out").expect("frame");
        assert_eq!(f.meta().sfreq(), Some(200.0), "the bound rate reached the mirrored field");
        assert_eq!(as_f32_vec(&f).len(), 20, "and paced the block: 200 Hz over 100 ms");
    }

    #[test]
    fn nd_reference_resolves_same_tick_via_dag_lifting() {
        // src emits value 3; host.value = nd('src'). The ref edge schedules src before
        // host, so host reads THIS tick's value — 3 in one tick, not next tick.
        let mut g = eval_graph();
        let src = g.add_node("_TestConst", None).unwrap();
        g.rename_node(src, "src").unwrap();
        g.update_param(src, "constant", "value", Param::float(3.0, -1e9, 1e9)).unwrap();
        let host = g.add_node("_TestConst", None).unwrap();
        g.set_expression(host, "constant", "value", "nd('src')", true, false).unwrap();
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(host, "out").unwrap()), 3.0, "same-tick nd() resolution");
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
        // And it still resolves end-to-end through the new name.
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(host, "out").unwrap()), 3.0, "resolves via nd('signal')");
    }

    #[test]
    fn expression_values_report_live_evaluated_params() {
        // The live preview seam: after a tick, the evaluated value of each ENABLED binding
        // is reported (a plain literal param is not), and a disabled binding drops out.
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "7", true, false).unwrap();
        g.tick();
        let vals = g.expression_values(n);
        assert_eq!(vals.len(), 1, "only the expression-bound param is reported");
        let (group, name, p) = vals[0];
        assert_eq!((group, name), ("constant", "value"));
        assert!(matches!(p, Param::Float { value, .. } if (value - 7.0).abs() < 1e-9), "carries the evaluated value");
        // Disabling the binding removes it from the live set (its value is now the literal).
        g.set_expression(n, "constant", "value", "7", false, false).unwrap();
        assert!(g.expression_values(n).is_empty(), "disabled binding is not a live value");
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
        g.set_expression(host, "constant", "value", "nd('ghost')", true, false).unwrap();
        g.tick();
        assert!(g.last_error(host).is_some(), "missing ref surfaces on the node error channel");
        let info = g.param_expression(host, "constant", "value").expect("binding present");
        assert!(info.error.is_some(), "field error indicator set");
        // The literal value (default 0) is kept.
        assert_eq!(first_f32(&g.latest_frame(host, "out").unwrap()), 0.0);
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
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(host, "out").unwrap()), 0.0, "disabled binding is inert");
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
            goofi_node::param(g2.params(uid2).unwrap(), "constant", "value").unwrap().as_f64(),
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
    fn native_tick_latency_and_stability() {
        // Concrete latency/stability read for the NATIVE (in-process Rust) node path — the
        // counterpart to the subprocess-tier benchmark. Drive the reference fan-out shape
        // (Oscillator → 8 Buffers) unbounded (max_frequency 0 → every tick computes) and report
        // the full-graph per-tick latency distribution. Native nodes share the process (no IPC),
        // so this is the pure compute+propagate path.
        let mut g = Graph::new();
        let unbounded = |g: &mut Graph, uid| {
            g.update_param(uid, "common", "max_frequency", Param::float(0.0, 0.0, 1e9)).unwrap();
        };
        let osc = g.add_node("Oscillator", None).unwrap();
        unbounded(&mut g, osc);
        // A sample rate far above the tick rate, so the Oscillator has whole samples to emit on
        // EVERY tick. At its 250 Hz default an unbounded tick loop outruns the generator and the
        // node early-returns with nothing — the buffers would then never trigger and the
        // measurement would be dominated by empty ticks doing no work at all.
        g.update_param(osc, "oscillator", "sfreq", Param::float(1.0e6, 1.0, 1.0e9)).unwrap();
        for _ in 0..8 {
            let b = g.add_node("Buffer", None).unwrap();
            g.update_param(b, "buffer", "size", Param::int(256, 1, 1_000_000)).unwrap();
            unbounded(&mut g, b);
            g.add_link(osc, "out", b, "data").unwrap();
        }

        for _ in 0..100 {
            g.tick(); // warm up (buffers fill, buffers/paths hot)
        }
        // A buffer's emitted frame carries a source-origin `index`; it advances only on a tick
        // that actually propagated, so comparing it across the loop proves the ticks did work.
        let buf_index = |g: &Graph, u: Uid| {
            g.latest_frame(u, "out").and_then(|d| d.meta().index()).unwrap_or(0)
        };
        let a_buffer = *g.node_uids().iter().find(|&&u| g.type_name(u) == Some("Buffer")).unwrap();
        let index_before = buf_index(&g, a_buffer);

        let iters = 3000usize;
        let mut lat: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            g.tick();
            lat.push(t0.elapsed().as_secs_f64() * 1e6); // microseconds
        }
        let advanced = buf_index(&g, a_buffer).saturating_sub(index_before);
        // Every buffer produced a frame (stability — the graph propagated end-to-end each tick).
        assert!(g.node_uids().iter().all(|&u| g.latest_frame(u, "out").is_some()), "all nodes emit");
        // …and the timed ticks were PRODUCTIVE, so the distribution below measures the fan-out
        // doing real work rather than a graph that idled through most of the loop. (At the
        // Oscillator's 250 Hz default an unbounded loop outruns the generator and most ticks
        // emit nothing — the measurement then says nothing about the fan-out cost.)
        assert!(
            advanced as usize >= iters,
            "only {advanced} of {iters} timed ticks propagated — the benchmark is measuring idle ticks"
        );

        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = lat.iter().sum::<f64>() / iters as f64;
        let p = |q: f64| lat[((iters as f64 * q) as usize).min(iters - 1)];
        eprintln!(
            "native graph tick latency (Oscillator→8 Buffers, {iters} ticks): \
             min={:.1}us  p50={:.1}us  p99={:.1}us  max={:.1}us  mean={mean:.1}us",
            lat[0], p(0.50), p(0.99), lat[iters - 1]
        );
        // Gate on the MINIMUM. Cargo runs 150+ sibling tests across every core while this loop is
        // timed, and the machine is a desktop with whatever else the user is running — so every
        // sample carries preemption the code did not cause. The minimum over 3000 ticks is the
        // least-preempted one, i.e. the closest thing to the true cost, and it is the only
        // statistic that actually holds still: measured 262-334 us across idle, full-suite, and
        // suite+busy-browser runs, while the median swung 566 us → 4060 us over the same range.
        // (A median gate lived here and flaked exactly that way.) 1 ms is ~3x the observed floor —
        // tight enough to catch a real regression in the fan-out path, immune to machine load.
        assert!(lat[0] < 1000.0, "fastest tick {:.1}us exceeds the budget", lat[0]);
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
        // remove_node and clear (hence load_doc) must release the evaluator's handles.
        let mock = Arc::new(MockEval::default());
        let mut g = Graph::new();
        g.set_evaluator(mock.clone());
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "5", true, false).unwrap();
        assert_eq!(mock.exprs.lock().unwrap().len(), 1, "compiled once");
        g.remove_node(n).unwrap();
        assert_eq!(mock.exprs.lock().unwrap().len(), 0, "released on remove_node");
        let n2 = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n2, "constant", "value", "7", true, false).unwrap();
        assert_eq!(mock.exprs.lock().unwrap().len(), 1);
        g.clear();
        assert_eq!(mock.exprs.lock().unwrap().len(), 0, "released on clear");
    }

    #[test]
    fn binding_error_clears_on_recovery_even_for_a_never_running_node() {
        // _TestSink has a trigger input, autotrigger off, and (unwired) never runs — so
        // run_node never fires for it. The node-level error must still clear when its
        // expression recovers, because last_error() derives the binding error on read.
        let mut g = eval_graph();
        let sink = g.add_node("_TestSink", None).unwrap();
        g.set_expression(sink, "control", "value", "nd('src')", true, false).unwrap();
        g.tick();
        assert!(g.last_error(sink).is_some(), "missing ref errors while idle");
        let src = g.add_node("_TestConst", None).unwrap();
        g.rename_node(src, "src").unwrap();
        g.tick();
        assert!(g.last_error(sink).is_none(), "recovery clears the node error on a never-running node");
    }

    #[test]
    fn multiple_binding_errors_surface_deterministically() {
        // Two errored bindings on one node -> the smaller ParamKey (constant/length <
        // constant/value) wins, deterministically (not HashMap order).
        let mut g = eval_graph();
        let n = g.add_node("_TestConst", None).unwrap();
        g.set_expression(n, "constant", "value", "nd('gv')", true, false).unwrap();
        g.set_expression(n, "constant", "length", "nd('gl')", true, false).unwrap();
        g.tick();
        let err = g.last_error(n).expect("a binding error surfaces");
        assert!(err.contains("gl"), "deterministic min-ParamKey selection, got: {err}");
    }

    #[test]
    fn length_preserving_node_propagates_source_index() {
        // _TestConst(len 2) -> Echo (echoes -> len 2). The echo's output frame
        // count matches its single index-bearing input, so it PROPAGATES the
        // source's origin index rather than starting a fresh counter — an upstream
        // drop stays visible at the sink. Pre-tick the source unwired so its index
        // is a non-zero 3, distinguishable from a fresh-from-0 counter.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10)).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        for _ in 0..3 {
            g.tick(); // src advances to index 2; echo (unwired, triggered) never runs
        }
        g.add_link(src, "out", echo, "in").unwrap();
        g.tick(); // src -> index 3; echo runs, matches len -> propagates 3
        let f = g.latest_frame(echo, "out").expect("echo ran");
        assert_eq!(f.meta().index(), Some(3), "propagates the source's index, not fresh 0");
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
        let buf = g.add_node("Buffer", None).unwrap();
        g.add_link(src, "out", buf, "data").unwrap();
        let mut idx = Vec::new();
        for _ in 0..4 {
            g.tick();
            idx.push(g.latest_frame(buf, "out").unwrap().meta().index().unwrap());
        }
        assert_eq!(idx, vec![0, 1, 2, 3], "buffer index must be a monotonic fresh timeline");
    }

    #[test]
    fn length_changing_node_uses_fresh_index() {
        // _TestConst(len 2) -> Counter (emits len 1). The output frame count (1)
        // never matches the input (2), so no input is the same timeline: the counter
        // starts its OWN fresh index at 0, independent of the source's index (3).
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10)).unwrap();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        for _ in 0..3 {
            g.tick(); // src advances to index 2; counter (unwired) never runs
        }
        g.add_link(src, "out", cnt, "in").unwrap();
        g.tick(); // src -> index 3; counter runs, len mismatch -> fresh index 0
        let f = g.latest_frame(cnt, "out").expect("counter ran");
        assert_eq!(f.meta().index(), Some(0), "fresh counter, not the source's 3");
    }

    #[test]
    fn every_type_that_free_runs_says_so_in_its_own_declaration() {
        // The scheduler currently free-runs any node with no *triggering* input, whatever its
        // params say (`wants_run`'s `!has_trigger_inputs` term). That implicit rule is what the
        // async runtime removes, so a type that relies on it has to declare the pacing itself —
        // via `producer`, or by declaring `common.autotrigger` in its own params.
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
    fn common_max_frequency_caps_a_production_node() {
        use std::time::Duration;
        // Cap a real source (_TestConst, a free-running generator) at 10 Hz via
        // its `common` group; its emit index advances only on admitted ticks.
        let mut g = Graph::new();
        let c = g.add_node("_TestConst", None).unwrap();
        g.update_param(c, "common", "max_frequency", Param::float(10.0, 0.0, 60.0)).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // run -> index 0
        g.tick_at(t0 + Duration::from_millis(50)); // skip
        g.tick_at(t0 + Duration::from_millis(100)); // run -> index 1
        g.tick_at(t0 + Duration::from_millis(210)); // run -> index 2
        assert_eq!(g.latest_frame(c, "out").unwrap().meta().index(), Some(2), "capped to 3 emits");
    }

    #[test]
    fn run_policy_survives_gfi_roundtrip() {
        use std::time::Duration;
        // A saved max_frequency must re-derive into the loaded node's run gate.
        let mut g = Graph::new();
        let c = g.add_node("_TestConst", None).unwrap();
        g.update_param(c, "common", "max_frequency", Param::float(10.0, 0.0, 60.0)).unwrap();
        let yaml = g.serialize();

        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        let c2 = g2.node_uids()[0];
        assert_eq!(
            goofi_node::param(g2.params(c2).unwrap(), "common", "max_frequency").unwrap().as_f64(),
            Some(10.0),
            "max_frequency round-trips"
        );
        let t0 = Instant::now();
        g2.tick_at(t0);
        g2.tick_at(t0 + Duration::from_millis(50)); // skip -> gate active after load
        g2.tick_at(t0 + Duration::from_millis(100));
        assert_eq!(g2.latest_frame(c2, "out").unwrap().meta().index(), Some(1), "gate active post-load");
    }

    #[test]
    fn autotrigger_does_not_free_run_a_wired_trigger_node() {
        // A wired triggered node with common.autotrigger=true must still run ONLY
        // when a fresh frame arrives on its wired trigger — matching Python's
        // `autotrigger AND _has_no_triggering_inputs()`. Gated source emits every
        // other tick; over 6 ticks the counter must run exactly 3 times, not 6.
        let mut g = Graph::new();
        let src = g.add_node("_TestGated", None).unwrap();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        g.add_link(src, "out", cnt, "in").unwrap();
        g.update_param(cnt, "common", "autotrigger", Param::boolean(true)).unwrap();
        for _ in 0..6 {
            g.tick();
        }
        assert_eq!(
            first_f32(&g.latest_frame(cnt, "out").expect("counter ran")),
            3.0,
            "autotrigger must not fire a wired-trigger node on its idle ticks"
        );
    }

    #[test]
    fn autotrigger_free_runs_an_unwired_trigger_node() {
        // The faithful counterpart: a node that DECLARES a trigger input but has it
        // UNWIRED, with autotrigger=true, free-runs every tick (Python:
        // `_has_no_triggering_inputs()` is true when the slot has no source). This
        // guards the fix from over-correcting the wired case into this one.
        let mut g = Graph::new();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        g.update_param(cnt, "common", "autotrigger", Param::boolean(true)).unwrap();
        for _ in 0..3 {
            g.tick();
        }
        assert_eq!(
            first_f32(&g.latest_frame(cnt, "out").expect("free-ran")),
            3.0,
            "an unwired trigger node with autotrigger must free-run"
        );
    }

    #[test]
    fn ctx_now_is_seconds_since_first_tick() {
        use std::time::Duration;
        let mut g = Graph::new();
        let n = g.add_node("_TestNow", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // first tick anchors the reference -> now == 0
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 0.0);
        g.tick_at(t0 + Duration::from_millis(250)); // 0.25 s later
        assert!((first_f32(&g.latest_frame(n, "out").unwrap()) - 0.25).abs() < 1e-4);
    }

    #[test]
    fn a_load_restarts_the_node_clock() {
        use std::time::Duration;
        // A patch loaded five minutes into a session must behave like the same patch loaded at
        // boot. The load happens at a genuinely ADVANCED clock — at t0 the broken code and the
        // fixed one agree, so a fixture that loads at t≈0 proves nothing.
        let mut g = Graph::new();
        let n = g.add_node("_TestNow", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0);
        let t5m = t0 + Duration::from_secs(300);
        g.tick_at(t5m);
        assert_eq!(
            first_f32(&g.latest_frame(n, "out").unwrap()),
            300.0,
            "the session clock genuinely advanced before the load"
        );

        let doc = g.serialize();
        g.load_doc(&doc).unwrap();
        g.tick_at(t5m); // the loaded patch's first tick re-anchors the reference
        assert_eq!(
            first_f32(&g.latest_frame(n, "out").unwrap()),
            0.0,
            "a loaded patch starts its clock at zero, whenever it was loaded"
        );
        g.tick_at(t5m + Duration::from_millis(250));
        assert!(
            (first_f32(&g.latest_frame(n, "out").unwrap()) - 0.25).abs() < 1e-4,
            "and advances from there"
        );
    }

    #[test]
    fn rate_cap_gates_runs_by_wall_clock() {
        use std::time::Duration;
        // A 10 Hz (0.1s period) free-running source. Drive tick_at with a synthetic
        // clock and assert it runs only once the period has elapsed since last run.
        let mut g = Graph::new();
        let src = g.add_node("_TestCapped", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // never run -> runs (count 1)
        g.tick_at(t0 + Duration::from_millis(50)); // 0.05 < 0.1 -> skip
        g.tick_at(t0 + Duration::from_millis(100)); // 0.10 elapsed -> run (count 2)
        g.tick_at(t0 + Duration::from_millis(120)); // 0.02 since last -> skip
        g.tick_at(t0 + Duration::from_millis(210)); // 0.11 since last -> run (count 3)
        assert_eq!(
            first_f32(&g.latest_frame(src, "out").unwrap()),
            3.0,
            "10 Hz cap admitted exactly 3 of 5 ticks"
        );
    }

    fn ufreq(g: &Graph, uid: Uid, slot: &str) -> Option<f64> {
        g.latest_frame(uid, slot).unwrap().meta().ufreq()
    }

    #[test]
    fn ufreq_measures_steady_source_rate() {
        use std::time::Duration;
        // A pure source ticked every 10 ms emits at a steady 100 Hz. The first frame
        // has no interval to measure; from the second on, a steady period reads exact.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0);
        assert_eq!(ufreq(&g, src, "out"), None, "first emit: no interval yet");
        g.tick_at(t0 + Duration::from_millis(10));
        let uf = ufreq(&g, src, "out").expect("measured after 2nd emit");
        assert!((uf - 100.0).abs() < 1e-6, "10 ms period -> 100 Hz, got {uf}");
        g.tick_at(t0 + Duration::from_millis(20));
        let uf3 = ufreq(&g, src, "out").expect("still measured");
        assert!((uf3 - 100.0).abs() < 1e-6, "steady source stays exact, got {uf3}");
    }

    #[test]
    fn ufreq_reflects_the_rate_cap_not_the_tick_rate() {
        use std::time::Duration;
        // A 10 Hz-capped source ticked at 100 Hz emits every ~0.1 s. Its ufreq must
        // read the emit rate (~10 Hz), NOT the tick rate.
        let mut g = Graph::new();
        let src = g.add_node("_TestCapped", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // run (emit 1) -> no interval yet
        assert_eq!(ufreq(&g, src, "out"), None);
        g.tick_at(t0 + Duration::from_millis(50)); // skipped by the cap
        g.tick_at(t0 + Duration::from_millis(100)); // run (emit 2): dt = 0.1 s
        let uf = ufreq(&g, src, "out").expect("measured after 2nd emit");
        assert!((uf - 10.0).abs() < 1e-6, "capped emit rate -> 10 Hz, got {uf}");
    }

    #[test]
    fn ufreq_is_node_level_same_on_every_slot() {
        use std::time::Duration;
        // "fast" emits every 10 ms run; "slow" every other run. ufreq is measured PER
        // NODE (the node emits every run -> 100 Hz), so BOTH slots carry the node's
        // 100 Hz — not the slow slot's own 50 Hz cadence.
        let mut g = Graph::new();
        let src = g.add_node("_TestTwoRate", None).unwrap();
        let t0 = Instant::now();
        for i in 0..6 {
            g.tick_at(t0 + Duration::from_millis(10 * i));
        }
        let fast = ufreq(&g, src, "fast").expect("fast measured");
        let slow = ufreq(&g, src, "slow").expect("slow measured");
        assert!((fast - 100.0).abs() < 1e-6, "node rate on fast slot -> 100 Hz, got {fast}");
        assert!((slow - 100.0).abs() < 1e-6, "same node rate on slow slot -> 100 Hz, got {slow}");
    }

    #[test]
    fn ufreq_guards_nonadvancing_clock() {
        use std::time::Duration;
        // Two emits at the SAME instant (dt == 0) must never yield inf/NaN: before a
        // measurement exists it stays None; afterwards it keeps the prior estimate.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // emit 1
        g.tick_at(t0); // emit 2, dt == 0, no prior estimate
        assert_eq!(ufreq(&g, src, "out"), None, "dt==0 with no estimate stays None");
        g.tick_at(t0 + Duration::from_millis(10)); // emit 3: dt = 0.01 -> 100 Hz
        assert!((ufreq(&g, src, "out").unwrap() - 100.0).abs() < 1e-6);
        g.tick_at(t0 + Duration::from_millis(10)); // emit 4, dt == 0: keep prior estimate
        let uf = ufreq(&g, src, "out").unwrap();
        assert!(uf.is_finite(), "dt==0 must not produce inf/NaN, got {uf}");
        assert!((uf - 100.0).abs() < 1e-6, "keeps the prior 100 Hz estimate, got {uf}");
    }

    #[test]
    fn node_ufreq_exposes_the_measured_rate() {
        use std::time::Duration;
        // The control-plane accessor the bridge forwards to the node header.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0);
        assert_eq!(g.node_ufreq(src), None, "no rate before the 2nd emit");
        g.tick_at(t0 + Duration::from_millis(10));
        let uf = g.node_ufreq(src).expect("measured");
        assert!((uf - 100.0).abs() < 1e-6, "node_ufreq -> 100 Hz, got {uf}");
    }

    #[test]
    fn ufreq_survives_the_data_plane_wire() {
        use std::time::Duration;
        // End-to-end through the bridge's exact seam: an engine-stamped frame,
        // encoded as `goofi_codec::encode(latest_frame(..))` (see bridge/lib.rs),
        // carries ufreq across the wire so the browser inspector shows it.
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0);
        g.tick_at(t0 + Duration::from_millis(10)); // steady 100 Hz
        let frame = g.latest_frame(src, "out").unwrap();
        assert!((frame.meta().ufreq().unwrap() - 100.0).abs() < 1e-6);

        let wire = goofi_codec::encode(&frame);
        let back = goofi_codec::decode(&wire).expect("data-plane frame decodes");
        assert_eq!(back.meta().ufreq(), frame.meta().ufreq(), "ufreq round-trips the data plane");
        assert!((back.meta().ufreq().unwrap() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn default_policy_runs_every_tick_regardless_of_clock() {
        use std::time::Duration;
        // A default-policy source (unbounded) must run on every tick even when the
        // clock barely advances — proving the rate gate is inert without a cap
        // (backward compatibility with the pre-RunPolicy scheduler).
        let mut g = Graph::new();
        let src = g.add_node("_TestConst", None).unwrap();
        let t0 = Instant::now();
        for i in 0..5 {
            g.tick_at(t0 + Duration::from_nanos(i)); // clock essentially frozen
        }
        // 5 emits -> the generator's index advanced to 4 (ran every tick).
        assert_eq!(g.latest_frame(src, "out").unwrap().meta().index(), Some(4));
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
        let c = g.add_node("_TestRefLenChange", None).unwrap();
        g.add_link(rs, "out", c, "ref").unwrap();
        for _ in 0..3 {
            g.tick(); // rs -> index 2; c dormant (data unwired, triggered node)
        }
        g.add_link(ds, "out", c, "data").unwrap();
        g.tick(); // rs -> index 3 (len 1); ds -> index 0 (len 4); c emits len 1
        let f = g.latest_frame(c, "out").expect("consumer ran");
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

        g.tick(); // must NOT unwind past here (would poison the graph lock)

        std::panic::set_hook(prev);

        // The panic is captured as the node's error; the healthy node still ran.
        assert!(
            g.last_error(boom).unwrap_or("").contains("panic"),
            "panic must be captured as an error"
        );
        assert_eq!(first_f32(&g.latest_frame(ok, "out").unwrap()), 9.0);
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
        g.tick();
        std::panic::set_hook(prev);

        let first = g.error_age(boom).expect("the error is stamped the moment it appears");
        std::thread::sleep(Duration::from_millis(120));
        g.tick();
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
        let uid = g.add_node("_TestChangingError", None).unwrap();
        g.tick();
        assert_eq!(g.last_error(uid), Some("failure 1"));

        std::thread::sleep(Duration::from_millis(300));
        g.tick();
        assert_eq!(g.last_error(uid), Some("failure 2"), "the node is failing differently now");
        let age = g.error_age(uid).expect("still errored");
        assert!(
            age < Duration::from_millis(150),
            "a new message is a new error, not the old one still standing: {age:?}",
        );
    }
}
