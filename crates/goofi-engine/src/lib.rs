//! goofi-engine — the graph + a minimal single-threaded tick scheduler (M1).
//!
//! Grows into the work-stealing compute pool + reserved RT sub-pool + timer-wheel
//! autotrigger in M2. For now: instantiate catalog nodes, wire one-wire-per-input
//! links, and `tick()` all nodes once in topological order, moving each node's
//! outputs into its consumers' inputs (latest-wins) so a single pass propagates
//! through an acyclic graph. Each node's latest output frame is exposed for the
//! data plane.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use goofi_core::{Data, Param};
use goofi_node::{Inputs, NodeCtx, NodeManifest, Outputs, ParamGroups, ParamKey, RunPolicy};
use indexmap::IndexMap;
use rayon::prelude::*;

/// Sub-patch forest model + pure projector (see `subpatch.rs`). Phase 1: types +
/// `materialize`/`resolve_boundary`, not yet wired into the live graph.
pub mod subpatch;

/// A stable node identity. Encoded as a 12-hex string for the `.gfi` / frontend
/// (the same key those use), a `u64` internally.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Uid(pub u64);

impl Uid {
    pub fn to_hex(self) -> String {
        format!("{:012x}", self.0)
    }
    pub fn from_hex(s: &str) -> Option<Uid> {
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

/// Per-NODE measured emit-rate state (see [`stamp_meta`]). Tracks the wall-clock
/// (`ctx.now`) of the node's previous productive emit and the smoothed inter-emit
/// interval; `ufreq = 1/ema`. `last_emit == None` until the first emit, `ema == None`
/// until the second gives one interval to seed it.
struct UfreqMeter {
    last_emit: Option<f64>,
    ema: Option<f64>,
}

/// One wire feeding a `multi` input slot: its source `(uid, out-slot)` identity and
/// that wire's latest-wins frame (`None` until it first emits).
type WireCell = (Uid, &'static str, Option<Data>);

struct NodeEntry {
    manifest: &'static NodeManifest,
    node: Box<dyn goofi_node::Node>,
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
    last_error: Option<String>,
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
/// inverse of [`param_value_json`], and the SSOT for the engine load path (`set_param_from_json`)
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

/// A param bound to an expression (engine-side record; the node stays oblivious — the
/// engine writes the evaluated value into its params before it runs). See the
/// param-expressions design.
struct ExprBinding {
    source: String,
    enabled: bool,
    triggers_process: bool,
    /// Compiled handle owned by the evaluator (`None` if compile failed / no evaluator).
    id: Option<goofi_node::BindingId>,
    /// Statically-extracted `nd()` references (empty for a ref-less/time expression).
    refs: Vec<goofi_node::ExprRef>,
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
    /// Wall-clock reference, anchored at the first tick, so `NodeCtx::now` is
    /// seconds-since-start (deterministic under an injected clock).
    start: Option<Instant>,
    /// The injected param-expression evaluator (pyo3, from goofi-py). `None` → bindings
    /// are stored + round-trip but can't evaluate (graceful degrade to the literal).
    evaluator: Option<Arc<dyn goofi_node::ExprEvaluator>>,
    /// ── Sub-patch forest (the authoritative composition model; the flat `nodes`/`links`
    /// above is its projection). Empty ⇒ a plain flat graph, byte-for-byte today's behavior.
    defs: IndexMap<subpatch::DefId, subpatch::SubPatchDef>,
    /// Live instances (does NOT include the synthetic ROOT scope).
    instances: IndexMap<Uid, subpatch::Instance>,
    /// leaf/instance uid → its parent instance (`None` = ROOT scope). A uid absent from this
    /// map is ROOT-scoped (so an ordinary flat graph needs no entries).
    scope_of: HashMap<Uid, Option<Uid>>,
    /// leaf/instance uid → its template-local name within its scope (absent ⇒ use the node name).
    local_of: HashMap<Uid, subpatch::Local>,
    next_def: u64,
    /// Patch-scoped globals (system + user). System globals are seeded here; a `clear`/load
    /// re-asserts them. Read by param expressions + node setup/process; persisted to `.gfi`.
    globals: goofi_core::globals::GlobalStore,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
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
            start: None,
            evaluator: None,
            defs: IndexMap::new(),
            instances: IndexMap::new(),
            scope_of: HashMap::new(),
            local_of: HashMap::new(),
            next_def: 1,
            globals: goofi_core::globals::GlobalStore::new(),
        }
    }

    // ── Globals ─────────────────────────────────────────────────────────────────────────────
    // Patch-scoped named scalars. System globals (`default_ufreq`) are seeded + delete-protected;
    // user globals are add/edit/remove/rename. Read by expressions (`globals.<name>`) + node ctx.

    /// A read-only snapshot of the current globals for expression eval / node setup+process.
    pub fn globals_snapshot(&self) -> goofi_core::globals::GlobalsSnapshot {
        self.globals.snapshot()
    }

    /// Every global in order, tagged `(name, value, is_system)` — for the CRDT mirror + `.gfi`.
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
        for entry in self.nodes.values_mut() {
            for b in entry.bindings.values_mut() {
                if b.global_refs.iter().any(|g| g == name) {
                    b.last_eval = None;
                }
            }
        }
        Ok(())
    }

    /// Inject the param-expression evaluator (pyo3, from goofi-py). Wired by the CLI at
    /// startup; without it, expression bindings are stored but not evaluated.
    pub fn set_evaluator(&mut self, evaluator: Arc<dyn goofi_node::ExprEvaluator>) {
        self.evaluator = Some(evaluator);
    }

    /// Register a node type discovered at runtime. `manifest` must be `'static`
    /// (runtime types leak one manifest per type — bounded, catalog-lifetime); its
    /// `make` field is unused (instances come from `factory`).
    ///
    /// A name that collides with a built-in catalog type or an already-registered
    /// runtime type is refused (with a warning) rather than silently shadowed or
    /// overwritten — a built-in always wins `add_node`/`load_doc` resolution, and a
    /// blind overwrite would orphan the loser's leaked manifest and make its node
    /// unreachable. Returns whether the type was registered.
    pub fn register_dyn_type(
        &mut self,
        manifest: &'static NodeManifest,
        factory: NodeFactory,
    ) -> bool {
        let name = manifest.type_name;
        if goofi_node::find(name).is_some() {
            eprintln!("warning: runtime node type `{name}` collides with a built-in; ignoring it");
            return false;
        }
        if self.dyn_types.contains_key(name) {
            eprintln!("warning: runtime node type `{name}` already registered; ignoring the duplicate");
            return false;
        }
        self.dyn_types.insert(name, DynType { manifest, factory });
        true
    }

    /// Whether a type name resolves to either the compile-time catalog or a
    /// runtime-registered type.
    fn known_type(&self, type_name: &str) -> bool {
        goofi_node::find(type_name).is_some() || self.dyn_types.contains_key(type_name)
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
    /// A process / bootstrap error (`last_error`) wins; otherwise the errored expression
    /// binding with the smallest `ParamKey` — a deterministic pick, since `bindings` is a
    /// `HashMap` whose iteration order is randomized. Deriving on read (rather than caching
    /// into `last_error`) means a binding that recovers on a node that never runs again
    /// still clears, and the two channels can't drift apart.
    pub fn last_error(&self, uid: Uid) -> Option<&str> {
        let e = self.nodes.get(&uid)?;
        if let Some(err) = e.last_error.as_deref() {
            return Some(err);
        }
        e.bindings
            .iter()
            .filter_map(|(k, b)| b.error.as_deref().map(|s| (k, s)))
            .min_by(|a, b| a.0.cmp(b.0))
            .map(|(_, s)| s)
    }

    fn mint(&mut self) -> Uid {
        let u = Uid(self.next_uid);
        self.next_uid += 1;
        u
    }

    /// Construct (but do not insert) a node by type name — the shared front half of `add_node` /
    /// `add_node_at`. Resolves the compile-time catalog or a runtime-registered type and builds its
    /// params (defaulting to the type's defaults).
    fn build_node(
        &self,
        type_name: &str,
        params: Option<ParamGroups>,
    ) -> Result<(&'static NodeManifest, ParamGroups, Box<dyn goofi_node::Node>), String> {
        if let Some(m) = goofi_node::find(type_name) {
            let p = goofi_node::with_common(params.unwrap_or_else(|| m.default_params()));
            let n = (m.factory)();
            Ok((m, p, n))
        } else if let Some(dt) = self.dyn_types.get(type_name) {
            let p = goofi_node::with_common(params.unwrap_or_else(|| dt.manifest.default_params()));
            let n = (dt.factory)(&p);
            Ok((dt.manifest, p, n))
        } else {
            Err(format!("unknown node type `{type_name}`"))
        }
    }

    /// Instantiate a node by type name (compile-time catalog or a
    /// runtime-registered type). `params` defaults to the type's defaults.
    pub fn add_node(
        &mut self,
        type_name: &str,
        params: Option<ParamGroups>,
    ) -> Result<Uid, String> {
        let seed = params.is_none();
        let (manifest, params, node) = self.build_node(type_name, params)?;
        let uid = self.insert_node(manifest, node, params);
        if seed {
            self.seed_default_expressions(uid, manifest);
        }
        Ok(uid)
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

    /// Seed a live expression binding for each of the type's `default_expr` params — the fresh-add
    /// analogue of a literal default. Skipped entirely without an evaluator (the `spec` literal is the
    /// graceful fallback, never an errored "no evaluator" binding). Only fresh adds (`params == None`)
    /// call this; a restore/load supplies explicit params + its own captured expressions.
    fn seed_default_expressions(&mut self, uid: Uid, manifest: &'static NodeManifest) {
        if self.evaluator.is_none() {
            return;
        }
        for decl in manifest.params {
            if let Some(expr) = decl.default_expr {
                let _ = self.set_expression(uid, decl.group, decl.name, expr, true, false);
            }
        }
    }

    /// Build a `NodeEntry` from a manifest + a constructed node, run its `setup`,
    /// seed its I/O buffers, assign a fresh name + minted uid, and insert it. Shared by the
    /// catalog and runtime instantiation paths.
    fn insert_node(
        &mut self,
        manifest: &'static NodeManifest,
        node: Box<dyn goofi_node::Node>,
        params: ParamGroups,
    ) -> Uid {
        let uid = self.mint();
        let name = self.fresh_name(&manifest.type_name.to_lowercase());
        self.insert_node_at(uid, name, manifest, node, params);
        uid
    }

    /// Insert a constructed node at a SPECIFIC uid + display name — the reconcile path, which
    /// spawns sub-patch members at their deterministic uids. The uid must be free.
    fn insert_node_at(
        &mut self,
        uid: Uid,
        name: String,
        manifest: &'static NodeManifest,
        mut node: Box<dyn goofi_node::Node>,
        params: ParamGroups,
    ) {
        let mut ctx = NodeCtx::new();
        // `setup` latches the globals as of insert time (`process` reads them live each tick).
        ctx.globals = self.globals.snapshot();
        // Seed the node by replaying `on_param_changed` for each declared param
        // (not `common`, which is the scheduler's), then run derived one-time init.
        // The FIRST error from replay-or-setup becomes the node's bootstrap error;
        // the node is still inserted (no restart loop), matching the setup pipe.
        let mut last_error = None;
        for (group, entries) in &params {
            if group == "common" {
                continue;
            }
            for (name, value) in entries {
                if let Err(e) = node.on_param_changed(&ParamKey::new(group.as_str(), name.as_str()), value) {
                    last_error.get_or_insert(e.0);
                }
            }
        }
        if let Err(e) = node.setup(&mut ctx, &goofi_node::Params::new(&params)) {
            last_error.get_or_insert(e.0);
        }

        let inputs: IndexMap<&'static str, Option<Data>> =
            manifest.inputs.iter().filter(|s| !s.multi).map(|s| (s.name, None)).collect();
        let multi_inputs: IndexMap<&'static str, Vec<WireCell>> =
            manifest.inputs.iter().filter(|s| s.multi).map(|s| (s.name, Vec::new())).collect();
        let outputs = manifest.output_buffer();

        let has_trigger_inputs = manifest.inputs.iter().any(|i| i.trigger_process);
        let run_policy = RunPolicy::from_params(&params);
        self.nodes.insert(
            uid,
            NodeEntry {
                manifest,
                node,
                params,
                inputs,
                multi_inputs,
                outputs,
                last_outputs: IndexMap::new(),
                bindings: HashMap::new(),
                ctx,
                last_error,
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

    /// Whether a display name is taken by any live leaf node OR sub-patch instance. The two share
    /// one display-name namespace — both become member local keys when captured into a def — so
    /// uniqueness must span both, else a leaf renamed to an instance's `subpatch{N}` name collapses
    /// onto the same local key on grouping and silently drops a member.
    fn name_in_use(&self, name: &str) -> bool {
        self.nodes.values().any(|e| e.name == name) || self.instances.values().any(|i| i.name == name)
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

    pub fn name(&self, uid: Uid) -> Option<&str> {
        self.nodes.get(&uid).map(|e| e.name.as_str())
    }

    pub fn pos(&self, uid: Uid) -> Option<[f64; 2]> {
        self.nodes.get(&uid).map(|e| e.pos)
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
        // Shared sub-patch definitions store EXTERNAL nd() refs verbatim by display name, so a
        // renamed external producer must follow into the def too — else a later duplicate is
        // instantiated from the stale name (Python: _rewrite_record_nd over the definitions).
        // Templates aren't compiled, so rewrite the stored source string in place.
        for def in self.defs.values_mut() {
            for member in def.members.values_mut() {
                if let subpatch::MemberDecl::Leaf(leaf) = member {
                    for ex in &mut leaf.expressions {
                        if let Some(src) =
                            goofi_node::rewrite_nd_refs(&ex.source, |n| (n == old).then(|| new.to_string()))
                        {
                            ex.source = src;
                        }
                    }
                }
            }
        }
        touched
    }

    pub fn set_node_pos(&mut self, uid: Uid, pos: [f64; 2]) -> Result<(), String> {
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

    /// The parent instance of a node/instance (`None` = ROOT scope). Absent ⇒ ROOT, so a
    /// plain flat graph needs no entries.
    pub fn scope_of(&self, uid: Uid) -> Option<Uid> {
        self.scope_of.get(&uid).copied().flatten()
    }

    /// A node's template-local name within its scope (its display name by default).
    pub fn local_of(&self, uid: Uid) -> Option<&str> {
        self.local_of.get(&uid).map(|s| s.as_str())
    }

    /// Live instance uids (excludes the synthetic ROOT).
    pub fn instance_uids(&self) -> Vec<Uid> {
        self.instances.keys().copied().collect()
    }

    pub fn instance(&self, uid: Uid) -> Option<&subpatch::Instance> {
        self.instances.get(&uid)
    }

    pub fn def(&self, def_id: subpatch::DefId) -> Option<&subpatch::SubPatchDef> {
        self.defs.get(&def_id)
    }

    /// Chain-resolve an instance's boundary port to the single physical inner leaf `(uid, slot)`
    /// it exposes (walking nested instances); `None` if unwired. Used by the snapshot projection
    /// and the data plane (a viewer on `inst/bnd` subscribes to this leaf).
    pub fn resolve_boundary(&self, inst: Uid, bnd: &str) -> Option<(Uid, String)> {
        subpatch::resolve_boundary(&self.defs, &self.instances, inst, bnd)
    }

    /// How many live instances reference a def: 1 ⇒ unique (serializes inline), ≥2 ⇒ shared.
    pub fn def_refcount(&self, def_id: subpatch::DefId) -> usize {
        self.instances.values().filter(|i| i.def_id == def_id).count()
    }

    fn mint_def(&mut self) -> subpatch::DefId {
        let d = subpatch::DefId(self.next_def);
        self.next_def += 1;
        d
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

    /// Capture a live leaf as a def `LeafDecl` (type/params/expressions/pos), for grouping.
    fn capture_leaf_decl(&self, uid: Uid) -> Option<subpatch::LeafDecl> {
        let e = self.nodes.get(&uid)?;
        let mut expressions: Vec<subpatch::ExprDecl> = e
            .bindings
            .iter()
            .map(|(k, b)| subpatch::ExprDecl {
                group: k.group.clone(),
                name: k.name.clone(),
                source: b.source.clone(),
                enabled: b.enabled,
                triggers_process: b.triggers_process,
            })
            .collect();
        expressions.sort_by(|a, b| (&a.group, &a.name).cmp(&(&b.group, &b.name)));
        Some(subpatch::LeafDecl {
            type_name: e.manifest.type_name.to_string(),
            params: e.params.clone(),
            expressions,
            pos: e.pos,
        })
    }

    /// Re-tag a member's scope. `scope_of` is the single source of truth for parentage (an
    /// instance's parent is just `scope_of[inst]`), so this is the one place membership changes.
    /// `None` = ROOT scope.
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

    /// Display name of a leaf node OR an instance (its `name` field) — the local key a member
    /// takes when captured into a def. `self.name` alone returns `None` for an instance.
    fn display_name(&self, uid: Uid) -> String {
        self.name(uid)
            .map(str::to_string)
            .or_else(|| self.instances.get(&uid).map(|i| i.name.clone()))
            .unwrap_or_default()
    }

    /// The member of `member_set` that transitively contains `uid` — `uid` itself if it is a
    /// direct member, else the ancestor instance (walking up scopes) that is a member. `None`
    /// if `uid` lies outside every member. Lets link-capture treat a leaf buried in a nested
    /// member instance as "inside the group".
    fn containing_member(&self, uid: Uid, member_set: &std::collections::HashSet<Uid>) -> Option<Uid> {
        let mut cur = uid;
        loop {
            if member_set.contains(&cur) {
                return Some(cur);
            }
            cur = self.scope_of(cur)?;
        }
    }

    /// The boundary id on `inst`'s def (a member instance) whose chain-to-leaf resolution is
    /// exactly `(leaf, slot)` in direction `dir`. Used to name the interior endpoint of a link
    /// that crosses into a nested member: the link references the member's BOUNDARY, not the
    /// buried leaf. Handles arbitrary nesting depth (resolve_boundary recurses down).
    fn boundary_exposing(&self, inst: Uid, leaf: Uid, slot: &str, dir: subpatch::Dir) -> Option<subpatch::BndId> {
        let def = self.defs.get(&self.instances.get(&inst)?.def_id)?;
        def.interface
            .iter()
            .filter(|(_, b)| b.dir == dir)
            .find(|(bnd, _)| self.resolve_boundary(inst, bnd).is_some_and(|(u, s)| u == leaf && s == slot))
            .map(|(bnd, _)| bnd.clone())
    }

    /// The slot name for interior endpoint `(endpoint, slot)` as seen from a group whose direct
    /// member is `member`: the real slot when `member` IS the endpoint (a leaf), else the nested
    /// member's boundary id exposing it (falling back to the real slot for a corrupt graph with
    /// no such boundary). The `local` half is always `member`'s local — the caller supplies it.
    fn endpoint_slot(&self, member: Uid, endpoint: Uid, slot: &str, dir: subpatch::Dir) -> String {
        if member == endpoint {
            slot.to_string()
        } else {
            // Invariant: a link crossing a nested member's edge exists BECAUSE that member exposes a
            // boundary to the buried leaf (grouping created it), so this always resolves. The raw-slot
            // fallback would produce a LocalLink whose local names an instance but whose slot names a
            // leaf slot — which resolve_endpoint silently drops. Assert in debug; stay safe in release.
            self.boundary_exposing(member, endpoint, slot, dir).unwrap_or_else(|| {
                debug_assert!(
                    false,
                    "endpoint_slot: nested member {member:?} exposes no boundary for {endpoint:?}/{slot} (capture invariant violated)"
                );
                slot.to_string()
            })
        }
    }

    /// Group `members` (leaf nodes and/or existing instances, all in ONE scope) into a new
    /// sub-patch instance. Pure bookkeeping: captures a def from the live members, derives its
    /// interface from the cut links, and re-tags membership. Returns the new instance uid.
    /// The flat `nodes`/`links` and every member's uid are UNCHANGED.
    pub fn group_nodes(&mut self, members: &[Uid], pos: [f64; 2]) -> Result<Uid, String> {
        use subpatch::{Boundary, Dir, LocalLink, MemberDecl, NestedDecl, Pillar};
        if members.is_empty() {
            return Err("group_nodes: empty selection".into());
        }
        // 1. Validate BEFORE any mutation: each exists, and all share one scope.
        let mut scope: Option<Option<Uid>> = None;
        for &m in members {
            if !self.nodes.contains_key(&m) && !self.instances.contains_key(&m) {
                return Err(format!("group_nodes: no such node {m}"));
            }
            let s = self.scope_of(m);
            match scope {
                None => scope = Some(s),
                Some(prev) if prev != s => {
                    return Err("group_nodes: members span multiple scopes".into())
                }
                _ => {}
            }
        }
        let parent = scope.unwrap();
        let member_set: std::collections::HashSet<Uid> = members.iter().copied().collect();

        // 2. Capture each member as a MemberDecl under its display-name local (globally unique,
        //    hence unique within the def).
        let mut def_members: IndexMap<subpatch::Local, MemberDecl> = IndexMap::new();
        let mut inst_members: IndexMap<subpatch::Local, Uid> = IndexMap::new();
        let mut local_by_uid: HashMap<Uid, subpatch::Local> = HashMap::new();
        for &m in members {
            let local = self.display_name(m);
            let decl = if let Some(inst) = self.instances.get(&m) {
                MemberDecl::Nested(NestedDecl { def_id: inst.def_id, pos: inst.pos })
            } else {
                MemberDecl::Leaf(self.capture_leaf_decl(m).ok_or("group_nodes: member vanished")?)
            };
            def_members.insert(local.clone(), decl);
            inst_members.insert(local.clone(), m);
            local_by_uid.insert(m, local);
        }

        // 3. Classify each link by TRANSITIVE containment — an endpoint buried inside a nested
        //    member counts as inside the group. Both inside → an internal link; exactly one inside
        //    → a boundary (one per inner (node, slot)). An interior endpoint that sits inside a
        //    nested member is named by that member's BOUNDARY id (not the buried leaf), so the
        //    runtime resolves it chain-to-leaf. `containing_member` returns the DIRECT member the
        //    endpoint belongs to; `endpoint_slot` maps the slot to a boundary id when nested.
        let mut interface: IndexMap<subpatch::BndId, Boundary> = IndexMap::new();
        let mut internal: Vec<LocalLink> = Vec::new();
        let mut seen: std::collections::HashSet<(Uid, &'static str, bool)> = std::collections::HashSet::new();
        let (mut in_n, mut out_n) = (0usize, 0usize);
        for l in &self.links {
            let out_m = self.containing_member(l.node_out, &member_set);
            let in_m = self.containing_member(l.node_in, &member_set);
            match (out_m, in_m) {
                // A link with both endpoints buried inside ONE nested-instance member belongs to
                // that member's own def (captured there), not this enclosing def — skip it, else it
                // becomes an invalid self-loop (subpatchN.out → subpatchN.in) the runtime drops.
                (Some(om), Some(im)) if om == im && self.instances.contains_key(&om) => {}
                (Some(om), Some(im)) => internal.push(LocalLink {
                    out: local_by_uid[&om].clone(),
                    out_slot: self.endpoint_slot(om, l.node_out, l.slot_out, Dir::Out),
                    in_: local_by_uid[&im].clone(),
                    in_slot: self.endpoint_slot(im, l.node_in, l.slot_in, Dir::In),
                }),
                (Some(om), None) => {
                    if !seen.insert((l.node_out, l.slot_out, true)) {
                        continue;
                    }
                    let dtype = self.output_slot_type(l.node_out, l.slot_out).unwrap_or(goofi_core::SlotType::Array);
                    let name = format!("out{out_n}");
                    interface.insert(
                        name.clone(),
                        Boundary {
                            dir: Dir::Out,
                            pillar: Pillar::Signal,
                            dtype,
                            inner: Some((local_by_uid[&om].clone(), self.endpoint_slot(om, l.node_out, l.slot_out, Dir::Out))),
                            pos: [pos[0] + 220.0, pos[1] + 40.0 * out_n as f64],
                            name,
                        },
                    );
                    out_n += 1;
                }
                (None, Some(im)) => {
                    if !seen.insert((l.node_in, l.slot_in, false)) {
                        continue;
                    }
                    let dtype = self.input_slot_type(l.node_in, l.slot_in).unwrap_or(goofi_core::SlotType::Array);
                    let name = format!("in{in_n}");
                    interface.insert(
                        name.clone(),
                        Boundary {
                            dir: Dir::In,
                            pillar: Pillar::Signal,
                            dtype,
                            inner: Some((local_by_uid[&im].clone(), self.endpoint_slot(im, l.node_in, l.slot_in, Dir::In))),
                            pos: [pos[0] - 40.0, pos[1] + 40.0 * in_n as f64],
                            name,
                        },
                    );
                    in_n += 1;
                }
                (None, None) => {}
            }
        }

        // 4. Mint + register. Members stay live; only membership re-tags.
        let def_id = self.mint_def();
        let inst_uid = self.mint();
        let disp = self.mint_subpatch_name(inst_uid);
        self.defs.insert(
            def_id,
            subpatch::SubPatchDef { name: disp.clone(), members: def_members, links: internal, interface },
        );
        self.instances.insert(
            inst_uid,
            subpatch::Instance { uid: inst_uid, name: disp, def_id, pos, members: inst_members },
        );
        for &m in members {
            self.set_member_scope(m, Some(inst_uid));
            self.local_of.insert(m, local_by_uid[&m].clone());
        }
        self.scope_of.insert(inst_uid, parent);
        Ok(inst_uid)
    }

    /// Inline an instance back into its parent scope: re-tag each member to the parent scope,
    /// drop the instance, and GC its def if now unreferenced. External flat links already point
    /// at the members, so they survive verbatim. Returns the restored member uids.
    pub fn expand_instance(&mut self, inst: Uid) -> Result<Vec<Uid>, String> {
        let instance = self
            .instances
            .get(&inst)
            .ok_or_else(|| format!("expand_instance: no such instance {inst}"))?;
        let def_id = instance.def_id;
        let restored: Vec<Uid> = instance.members.values().copied().collect();
        let parent = self.scope_of(inst); // the grandparent scope members fall back to
        for &m in &restored {
            self.set_member_scope(m, parent);
            self.local_of.remove(&m); // back to display-name addressing in the parent scope
        }
        self.instances.shift_remove(&inst);
        self.scope_of.remove(&inst);
        self.local_of.remove(&inst);
        if self.def_refcount(def_id) == 0 {
            self.defs.shift_remove(&def_id);
        }
        Ok(restored)
    }

    /// Delete a whole sub-patch instance: remove every member (recursing into nested
    /// instances, tearing down leaves), drop the instance, and GC its def if now unreferenced.
    /// The frontend routes Delete-on-an-instance and the inverse of `duplicate_shared` here.
    pub fn remove_instance(&mut self, inst: Uid) -> Result<(), String> {
        let instance = self
            .instances
            .get(&inst)
            .ok_or_else(|| format!("remove_instance: no such instance {inst}"))?;
        let def_id = instance.def_id;
        let members: Vec<Uid> = instance.members.values().copied().collect();
        for m in members {
            if self.instances.contains_key(&m) {
                self.remove_instance(m)?; // nested instance subtree
            } else {
                let _ = self.remove_node(m); // leaf (tolerate an already-gone member)
            }
        }
        self.instances.shift_remove(&inst);
        self.scope_of.remove(&inst);
        self.local_of.remove(&inst);
        if self.def_refcount(def_id) == 0 {
            self.defs.shift_remove(&def_id);
        }
        Ok(())
    }

    // ── Boundary authoring (interface entries on a def; never live nodes) ─────────
    // A boundary is a naming indirection over an inner leaf slot. All edits mutate the
    // instance's DEF, so on a shared def they mirror to every sibling for free (the def is
    // the SSOT — resolve/describe read from it). External wires stay flat leaf→leaf links.

    /// The template-local of `member` within `inst` (its key in the instance's members map).
    fn member_local(&self, inst: Uid, member: Uid) -> Option<subpatch::Local> {
        self.instances
            .get(&inst)?
            .members
            .iter()
            .find(|(_, &u)| u == member)
            .map(|(l, _)| l.clone())
    }

    fn def_id_of(&self, inst: Uid) -> Result<subpatch::DefId, String> {
        self.instances
            .get(&inst)
            .map(|i| i.def_id)
            .ok_or_else(|| format!("no such instance {inst}"))
    }

    /// Add an UNWIRED boundary to an instance's def; returns its stable `BndId` (`in{n}`/
    /// `out{n}`). `dtype` is the caller's provisional type until the port is wired.
    pub fn add_boundary(
        &mut self,
        inst: Uid,
        dir: subpatch::Dir,
        dtype: goofi_core::SlotType,
        pos: [f64; 2],
    ) -> Result<subpatch::BndId, String> {
        let def_id = self.def_id_of(inst)?;
        let def = self.defs.get_mut(&def_id).ok_or("add_boundary: missing def")?;
        let prefix = match dir {
            subpatch::Dir::In => "in",
            subpatch::Dir::Out => "out",
        };
        let mut n = 0;
        while def.interface.contains_key(&format!("{prefix}{n}")) {
            n += 1;
        }
        let bnd = format!("{prefix}{n}");
        def.interface.insert(
            bnd.clone(),
            subpatch::Boundary { dir, pillar: subpatch::Pillar::Signal, dtype, inner: None, pos, name: bnd.clone() },
        );
        Ok(bnd)
    }

    /// Point a boundary at an inner member slot (one boundary per inner slot). `inner_node`
    /// must be a member of `inst`; the boundary's dtype is resolved from that slot.
    pub fn wire_boundary(&mut self, inst: Uid, bnd: &str, inner_node: Uid, inner_slot: &str) -> Result<(), String> {
        let local = self.member_local(inst, inner_node).ok_or("wire_boundary: inner is not a member of this instance")?;
        let def_id = self.def_id_of(inst)?;
        let dir = self
            .defs
            .get(&def_id)
            .and_then(|d| d.interface.get(bnd))
            .map(|b| b.dir)
            .ok_or("wire_boundary: no such boundary")?;
        let dtype = match dir {
            subpatch::Dir::In => self.input_slot_type(inner_node, inner_slot),
            subpatch::Dir::Out => self.output_slot_type(inner_node, inner_slot),
        }
        .ok_or("wire_boundary: no such inner slot")?;
        let target = (local, inner_slot.to_string());
        let def = self.defs.get_mut(&def_id).ok_or("wire_boundary: missing def")?;
        if def.interface.iter().any(|(id, b)| id != bnd && b.inner.as_ref() == Some(&target)) {
            return Err("wire_boundary: that inner slot is already exposed by another boundary".into());
        }
        let b = def.interface.get_mut(bnd).ok_or("wire_boundary: no such boundary")?;
        b.inner = Some(target);
        b.dtype = dtype;
        Ok(())
    }

    /// Drop a boundary. External flat links stay valid leaf→leaf links (they never referenced
    /// the boundary at runtime), so they are left in place.
    pub fn remove_boundary(&mut self, inst: Uid, bnd: &str) -> Result<(), String> {
        let def_id = self.def_id_of(inst)?;
        let def = self.defs.get_mut(&def_id).ok_or("remove_boundary: missing def")?;
        def.interface.shift_remove(bnd).ok_or("remove_boundary: no such boundary")?;
        Ok(())
    }

    /// Relabel a boundary's display name. The `bnd_id` is unchanged, so external wires survive.
    pub fn rename_boundary(&mut self, inst: Uid, bnd: &str, name: &str) -> Result<(), String> {
        let def_id = self.def_id_of(inst)?;
        let b = self
            .defs
            .get_mut(&def_id)
            .and_then(|d| d.interface.get_mut(bnd))
            .ok_or("rename_boundary: no such boundary")?;
        b.name = name.to_string();
        Ok(())
    }

    /// Move a boundary pill inside the entered view.
    pub fn set_boundary_pos(&mut self, inst: Uid, bnd: &str, pos: [f64; 2]) -> Result<(), String> {
        let def_id = self.def_id_of(inst)?;
        let b = self
            .defs
            .get_mut(&def_id)
            .and_then(|d| d.interface.get_mut(bnd))
            .ok_or("set_boundary_pos: no such boundary")?;
        b.pos = pos;
        Ok(())
    }

    // ── reconcile + sharing: spawn subtrees / re-project shared defs ──────────────
    // reconcile is the engine of SPAWNING (a fresh sibling, a loaded instance) and of shared
    // topology edits. Grouping/expand never call it (their members are already live).

    /// Instantiate one planned leaf at its deterministic uid, applying its params then its
    /// captured expressions.
    fn insert_planned_leaf(&mut self, pn: &subpatch::PlannedLeaf) -> Result<(), String> {
        // Shares `build_node`'s catalog-or-dyn resolution; `reconcile` pre-validates every type, so
        // the unknown-type branch is unreachable here.
        let (manifest, params, node) = self.build_node(&pn.type_name, Some(pn.params.clone()))?;
        let name = self.fresh_name(&manifest.type_name.to_lowercase());
        self.insert_node_at(pn.uid, name, manifest, node, params);
        for ex in &pn.expressions {
            let _ = self.set_expression(pn.uid, &ex.group, &ex.name, &ex.source, ex.enabled, ex.triggers_process);
        }
        Ok(())
    }

    /// Bring the live flat graph into agreement with `plan` (the forest projection). Validation
    /// runs before any mutation. Diffs by uid: a surviving member keeps its `NodeEntry`
    /// (buffers, ufreq, bindings, `/data` subs); only the delta spawns/drops. Links whose BOTH
    /// endpoints are members of a covered scope are managed to match the plan; external flat
    /// links (one endpoint outside every covered scope) are untouched.
    fn reconcile(&mut self, plan: subpatch::FlatPlan) -> Result<(), String> {
        for pn in &plan.nodes {
            if !self.known_type(&pn.type_name) {
                return Err(format!("reconcile: unknown node type `{}`", pn.type_name));
            }
        }
        let covered: std::collections::HashSet<Uid> = plan.nodes.iter().map(|n| n.scope).collect();
        let planned: std::collections::HashSet<Uid> = plan.nodes.iter().map(|n| n.uid).collect();
        let is_covered = |g: &Graph, u: Uid| {
            g.scope_of.get(&u).copied().flatten().is_some_and(|s| covered.contains(&s))
        };

        // 1. Remove live members of covered scopes the plan dropped.
        let stale: Vec<Uid> = self
            .nodes
            .keys()
            .copied()
            .filter(|&u| is_covered(self, u) && !planned.contains(&u))
            .collect();
        for u in stale {
            self.remove_node(u)?;
        }

        // 2. Insert planned members not yet live; (re)tag membership/local for all. Pos is
        //    applied ONLY to freshly-inserted members — a surviving member keeps its live pos
        //    (so an unrelated duplicate/re-project never snaps a user-moved member back to the
        //    def's group-time pos; shared pos-mirroring is the §4.5 edit path, not reconcile's).
        for pn in &plan.nodes {
            if !self.nodes.contains_key(&pn.uid) {
                self.insert_planned_leaf(pn)?;
                let _ = self.set_node_pos(pn.uid, pn.pos);
            }
            self.scope_of.insert(pn.uid, Some(pn.scope));
            self.local_of.insert(pn.uid, pn.local.clone());
        }

        // 3. Managed links → exactly the plan's links.
        let desired: std::collections::HashSet<(Uid, String, Uid, String)> = plan
            .links
            .iter()
            .map(|l| (l.out, l.out_slot.clone(), l.in_, l.in_slot.clone()))
            .collect();
        let current: Vec<(Uid, &'static str, Uid, &'static str)> = self
            .links
            .iter()
            .filter(|l| is_covered(self, l.node_out) && is_covered(self, l.node_in))
            .map(|l| (l.node_out, l.slot_out, l.node_in, l.slot_in))
            .collect();
        for (a, so, b, si) in current {
            if !desired.contains(&(a, so.to_string(), b, si.to_string())) {
                self.remove_link(a, so, b, si)?;
            }
        }
        for l in &plan.links {
            let _ = self.add_link(l.out, &l.out_slot, l.in_, &l.in_slot);
        }
        Ok(())
    }

    /// Allocate the deterministic member uids for a fresh instance, salt-rehashing on collision
    /// with any live uid (so determinism means *stability*, not literal hash equality).
    fn alloc_member_uids(&self, inst_uid: Uid, def_id: subpatch::DefId) -> IndexMap<subpatch::Local, Uid> {
        let mut members = IndexMap::new();
        if let Some(def) = self.defs.get(&def_id) {
            for local in def.members.keys() {
                let mut salt = 0u64;
                loop {
                    let seed = inst_uid.0 ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let uid = Uid(subpatch::fold_u64(seed, local));
                    let clash = self.nodes.contains_key(&uid)
                        || self.instances.contains_key(&uid)
                        || members.values().any(|&u| u == uid);
                    if !clash {
                        members.insert(local.clone(), uid);
                        break;
                    }
                    salt += 1;
                }
            }
        }
        members
    }

    /// Recursively register a fresh instance subtree for `def_id` at `inst_uid` under `parent`:
    /// allocate its member uids, register the Instance, then recurse into every NESTED member
    /// (its allocated uid becomes the nested instance's uid). Registers instances only — the leaf
    /// members are spawned + wired by the caller's `materialize` + `reconcile`. Needed so a shared
    /// def CONTAINING a sub-patch projects the sibling's whole subtree, not just its top leaves.
    fn spawn_instance_tree(&mut self, inst_uid: Uid, def_id: subpatch::DefId, parent: Option<Uid>, pos: [f64; 2]) {
        let members = self.alloc_member_uids(inst_uid, def_id);
        let nested: Vec<(Uid, subpatch::DefId, [f64; 2])> = self
            .defs
            .get(&def_id)
            .map(|def| {
                members
                    .iter()
                    .filter_map(|(local, &uid)| match def.members.get(local) {
                        Some(subpatch::MemberDecl::Nested(nd)) => Some((uid, nd.def_id, nd.pos)),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let disp = self.mint_subpatch_name(inst_uid);
        self.instances.insert(
            inst_uid,
            subpatch::Instance { uid: inst_uid, name: disp, def_id, pos, members },
        );
        self.scope_of.insert(inst_uid, parent);
        for (nuid, ndef, npos) in nested {
            self.spawn_instance_tree(nuid, ndef, Some(inst_uid), npos);
        }
    }

    /// Promote an instance's def to shared and spawn a strict-mirror sibling (deterministic
    /// member uids, `reconcile`d live + wired). The original's leaves are untouched. A nested
    /// sub-patch member spawns its own sibling subtree (`spawn_instance_tree`).
    pub fn duplicate_shared(&mut self, inst: Uid, pos: [f64; 2]) -> Result<Uid, String> {
        let (def_id, parent) = {
            let i = self.instances.get(&inst).ok_or_else(|| format!("duplicate_shared: no such instance {inst}"))?;
            (i.def_id, self.scope_of(inst))
        };
        let new_inst = self.mint();
        self.spawn_instance_tree(new_inst, def_id, parent, pos);
        let plan = subpatch::materialize(&self.defs, &self.instances);
        self.reconcile(plan)?;
        Ok(new_inst)
    }

    /// Every member that mirrors `member` under a SHARED def: the member itself plus its
    /// counterparts (same template-local) in every other instance of the same def. Just
    /// `[member]` when `member` is a ROOT node or a member of a UNIQUE def (no peers).
    pub fn shared_member_peers(&self, member: Uid) -> Vec<Uid> {
        let Some(inst_uid) = self.scope_of(member) else { return vec![member] };
        let (Some(inst), Some(local)) = (self.instances.get(&inst_uid), self.local_of(member)) else {
            return vec![member];
        };
        let def_id = inst.def_id;
        if self.def_refcount(def_id) < 2 {
            return vec![member];
        }
        let local = local.to_string();
        self.instances
            .values()
            .filter(|i| i.def_id == def_id)
            .filter_map(|i| i.members.get(&local).copied())
            .collect()
    }

    /// Re-capture a leaf member's live decl into its def (params/expressions/pos), so a later
    /// `duplicate_shared` and a save/load carry the edited value. No-op for a ROOT node or a
    /// nested-instance member.
    fn sync_def_member(&mut self, member: Uid) {
        let Some(inst_uid) = self.scope_of(member) else { return };
        if self.instances.contains_key(&member) {
            return; // a nested instance member syncs through its own edits
        }
        let (Some(def_id), Some(local)) = (
            self.instances.get(&inst_uid).map(|i| i.def_id),
            self.local_of(member).map(|s| s.to_string()),
        ) else {
            return;
        };
        let Some(leaf) = self.capture_leaf_decl(member) else { return };
        if let Some(def) = self.defs.get_mut(&def_id) {
            def.members.insert(local, subpatch::MemberDecl::Leaf(leaf));
        }
    }

    /// Apply a param edit, re-projecting to every shared sibling (§4.5): the def is the SSOT for
    /// a shared member, so an edit hits ALL its instances and syncs the def's stored decl.
    /// Returns the uids actually updated (for per-node `state_update` broadcast). A ROOT node or
    /// unique member updates only itself — identical to `update_param`.
    pub fn update_member_param(&mut self, uid: Uid, group: &str, name: &str, value: Param) -> Result<Vec<Uid>, String> {
        let peers = self.shared_member_peers(uid);
        for &peer in &peers {
            self.update_param(peer, group, name, value.clone())?;
        }
        self.sync_def_member(uid);
        Ok(peers)
    }

    /// Move a member, re-projecting to every shared sibling (§4.5 — a shared sub-patch's internal
    /// layout is shared) and syncing the def's stored pos. A ROOT node or the instance box itself
    /// (not a shared member) moves only itself — identical to `set_node_pos`.
    pub fn set_member_pos(&mut self, uid: Uid, pos: [f64; 2]) -> Result<Vec<Uid>, String> {
        // A sub-patch instance box lives in `instances`, not `nodes`, and carries its OWN
        // position independent of sibling instances (two shared instances sit at different
        // spots on the canvas). Move just this instance — delegating to set_node_pos would fail
        // with "no such node" and abort the whole drag RPC.
        if let Some(inst) = self.instances.get_mut(&uid) {
            inst.pos = pos;
            return Ok(vec![uid]);
        }
        let peers = self.shared_member_peers(uid);
        for &peer in &peers {
            self.set_node_pos(peer, pos)?;
        }
        self.sync_def_member(uid);
        Ok(peers)
    }

    /// Bind (or unbind) a member's param expression, re-projecting to every shared sibling
    /// (§4.5 — a shared sub-patch's authored logic is shared) and syncing the def's stored
    /// decl so a further duplicate inherits it. A ROOT node or unique member binds only
    /// itself — identical to `set_expression`. Returns the uids actually updated (for the
    /// per-node `state_update` broadcast). `nd()` refs resolve by global name as elsewhere,
    /// so the source is projected verbatim — consistent with `update_member_param`.
    pub fn set_member_expression(
        &mut self,
        uid: Uid,
        group: &str,
        name: &str,
        source: &str,
        enabled: bool,
        triggers_process: bool,
    ) -> Result<Vec<Uid>, String> {
        let peers = self.shared_member_peers(uid);
        for &peer in &peers {
            self.set_expression(peer, group, name, source, enabled, triggers_process)?;
        }
        self.sync_def_member(uid);
        Ok(peers)
    }

    /// Fork a shared instance's def to a fresh private copy (refcount 1) and repoint the
    /// instance. Pure bookkeeping — the live leaves already match the fork, so nothing respawns.
    /// Only a ROOT instance may be forked: a nested instance's def id is mirrored in its parent
    /// def's `NestedDecl`, so forking it in isolation would leave the parent projecting siblings
    /// against a stale def. Re-sharing that correctly needs a parent-def cascade — reject instead
    /// of silently corrupting (make the enclosing instance unique first).
    pub fn make_unique(&mut self, inst: Uid) -> Result<subpatch::DefId, String> {
        if self.scope_of(inst).is_some() {
            return Err("make_unique: cannot fork a nested instance — make its enclosing sub-patch unique first".into());
        }
        let old_def = self.instances.get(&inst).ok_or_else(|| format!("make_unique: no such instance {inst}"))?.def_id;
        let body = self.defs.get(&old_def).ok_or("make_unique: missing def")?.clone();
        let new_def = self.mint_def();
        self.defs.insert(new_def, body);
        self.instances.get_mut(&inst).unwrap().def_id = new_def;
        if self.def_refcount(old_def) == 0 {
            self.defs.shift_remove(&old_def);
        }
        Ok(new_def)
    }

    /// Inverse of `make_unique`: repoint a unique instance back onto a target def (bump its
    /// refcount, GC the abandoned private fork). Pure bookkeeping — live leaves already match.
    /// Root-only, symmetric with `make_unique`: repointing a nested instance would desync its
    /// parent def's `NestedDecl`.
    pub fn re_share_instance(&mut self, inst: Uid, def_id: subpatch::DefId) -> Result<Uid, String> {
        if self.scope_of(inst).is_some() {
            return Err("re_share_instance: cannot re-share a nested instance — operate on its enclosing sub-patch".into());
        }
        if !self.defs.contains_key(&def_id) {
            return Err(format!("re_share_instance: no such def {}", def_id.to_hex()));
        }
        let old_def = self.instances.get(&inst).ok_or_else(|| format!("re_share_instance: no such instance {inst}"))?.def_id;
        self.instances.get_mut(&inst).unwrap().def_id = def_id;
        if old_def != def_id && self.def_refcount(old_def) == 0 {
            self.defs.shift_remove(&old_def);
        }
        Ok(inst)
    }

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
        entry
            .node
            .on_param_changed(&ParamKey::new(group, name), &value)
            .map_err(|e| e.0)
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

    pub fn add_link(
        &mut self,
        node_out: Uid,
        slot_out: &str,
        node_in: Uid,
        slot_in: &str,
    ) -> Result<(), String> {
        let slot_out = self
            .resolve_output(node_out, slot_out)
            .ok_or_else(|| format!("no output slot `{slot_out}` on {node_out}"))?;
        let slot_in = self
            .resolve_input(node_in, slot_in)
            .ok_or_else(|| format!("no input slot `{slot_in}` on {node_in}"))?;

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
        self.defs.clear();
        self.instances.clear();
        self.scope_of.clear();
        self.local_of.clear();
        // Globals are patch content: a load starts from a fresh system-seeded store (load_doc then
        // repopulates user globals from the `.gfi`). `dyn_types` stays (catalog, not content).
        self.globals = goofi_core::globals::GlobalStore::new();
    }

    fn force_set_name(&mut self, uid: Uid, name: &str) {
        if let Some(e) = self.nodes.get_mut(&uid) {
            e.name = name.to_string();
        }
    }

    fn set_param_from_json(&mut self, uid: Uid, group: &str, name: &str, val: &serde_json::Value) {
        let existing = self
            .nodes
            .get(&uid)
            .and_then(|e| goofi_node::param(&e.params, group, name))
            .cloned();
        let Some(existing) = existing else {
            return;
        };
        // Load path: never fire a trigger on load (fire_triggers = false).
        let newp = param_from_json(&existing, val, false);
        let _ = self.update_param(uid, group, name, newp);
    }

    /// Serialize the graph to a `.gfi` v4 document (YAML text). v4 is the recursive,
    /// multi-pillar envelope: `version`/`pillar_default`/`definitions` at the top, with the
    /// nodes/links nested under `root` (sub-patch `definitions`/`instances` are empty until
    /// that subsystem lands). A signal-only patch is byte-equivalent to the old v3 flat form
    /// modulo the version bump + `root` nesting.
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

        // Sub-patch forest. Only the non-derivable structure is persisted: def NAME + INTERFACE
        // (boundaries) and per-instance {def, parent, pos, members}. A def's member bodies and
        // internal links are re-captured from the flat nodes/links on load (which stay in
        // `nodes`/`links` above), so there is no param duplication or staleness.
        let mut definitions = Map::new();
        for (def_id, def) in &self.defs {
            let mut iface = Map::new();
            for (bnd, b) in &def.interface {
                iface.insert(
                    bnd.clone(),
                    json!({
                        "dir": match b.dir { subpatch::Dir::In => "in", subpatch::Dir::Out => "out" },
                        "pillar": b.pillar.name(),
                        "dtype": b.dtype.name(),
                        "inner_local": b.inner.as_ref().map(|(l, _)| l.clone()),
                        "inner_slot": b.inner.as_ref().map(|(_, s)| s.clone()),
                        "pos": b.pos,
                        "name": b.name,
                    }),
                );
            }
            definitions.insert(def_id.to_hex(), json!({ "name": def.name, "interface": Value::Object(iface) }));
        }
        let mut inst_map = Map::new();
        for (uid, inst) in &self.instances {
            let mut members = Map::new();
            for (local, muid) in &inst.members {
                members.insert(local.clone(), json!(muid.to_hex()));
            }
            inst_map.insert(
                uid.to_hex(),
                json!({
                    "name": inst.name,
                    "def": inst.def_id.to_hex(),
                    "parent": self.scope_of(*uid).map(|p| p.to_hex()),
                    "pos": inst.pos,
                    "members": Value::Object(members),
                }),
            );
        }

        let root = json!({ "nodes": Value::Object(nodes), "links": links, "instances": Value::Object(inst_map) });
        // Globals (system + user) as `{name: {value, type}}`. On load, entries `set` existing system
        // globals and `add` user ones, then `reassert_system` back-fills; so a system global always
        // round-trips and an older patch simply picks up any new system default.
        let mut globals = serde_json::Map::new();
        for (name, value, _is_system) in self.globals.entries() {
            globals.insert(name.to_string(), global_to_json(value));
        }
        let doc = json!({
            "version": 5,
            "pillar_default": "signal",
            "globals": Value::Object(globals),
            "definitions": Value::Object(definitions),
            "root": root,
        });
        serde_yaml_ng::to_string(&doc).unwrap_or_default()
    }

    /// Replace the graph from a `.gfi` document (v3 or v4). Node types are validated
    /// before the current graph is torn down (a rejected load is a no-op). v3 keeps
    /// `nodes`/`links` flat at the top level; v4 nests them under `root` (with
    /// `definitions`/`instances` for sub-patches) — both up-convert to the same graph.
    pub fn load_doc(&mut self, text: &str) -> Result<(), String> {
        let doc: serde_json::Value = serde_yaml_ng::from_str(text).map_err(|e| e.to_string())?;
        let (nodes_v, links_v) = match doc.get("version").and_then(|v| v.as_i64()) {
            Some(3) => (doc.get("nodes"), doc.get("links")),
            // v4 and v5 share the `root` nesting; v5 adds the top-level `globals` block (loaded below).
            Some(4) | Some(5) => {
                let root = doc.get("root");
                (root.and_then(|r| r.get("nodes")), root.and_then(|r| r.get("links")))
            }
            _ => return Err("unsupported .gfi version (expected 3, 4, or 5)".into()),
        };
        let nodes = nodes_v.and_then(|v| v.as_object()).ok_or("missing `nodes`")?;
        for rec in nodes.values() {
            let ty = rec.get("type").and_then(|v| v.as_str()).ok_or("node missing `type`")?;
            if !self.known_type(ty) {
                return Err(format!("unknown node type `{ty}`"));
            }
        }

        self.clear();
        // Globals load BEFORE nodes so a node's `globals.*` param default-expression resolves at
        // instantiation. `clear()` already re-seeded the system globals; each entry sets an existing
        // (system) global or adds a user one. Malformed entries are skipped (best-effort load).
        if let Some(globals) = doc.get("globals").and_then(|v| v.as_object()) {
            for (name, entry) in globals {
                if let Some(value) = global_from_json(entry) {
                    let _ = self.globals.apply_change(name, Some(value));
                }
            }
        }
        let mut idmap: HashMap<String, Uid> = HashMap::new();
        for (old, rec) in nodes {
            let ty = rec["type"].as_str().unwrap();
            let uid = self.add_node(ty, None)?;
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
            if let Some(groups) = rec.get("params").and_then(|v| v.as_object()) {
                for (group, names) in groups {
                    if let Some(nm) = names.as_object() {
                        for (name, val) in nm {
                            self.set_param_from_json(uid, group, name, val);
                        }
                    }
                }
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
        // Reconstruct the sub-patch forest (v4). The members are already live flat nodes; here
        // we re-tag membership and rebuild each def's body from those live nodes + links (the
        // interface is the only serialized-verbatim part). idmap maps a persisted flat uid to
        // its live one; instances + defs are minted fresh (uids remap, structure preserved).
        let root_v = doc.get("root");
        let insts_v = root_v.and_then(|r| r.get("instances")).and_then(|v| v.as_object());
        let defs_v = doc.get("definitions").and_then(|v| v.as_object());
        self.reload_forest(insts_v, defs_v, &idmap);
        Ok(())
    }

    /// Rebuild `instances`/`defs`/`scope_of`/`local_of` from a loaded v4 document, after the
    /// flat nodes/links are live. Uids are remapped (instances/defs minted fresh); member uids
    /// resolve through `idmap` (a flat leaf) or a freshly-minted instance uid (a nested member).
    fn reload_forest(
        &mut self,
        insts_v: Option<&serde_json::Map<String, serde_json::Value>>,
        defs_v: Option<&serde_json::Map<String, serde_json::Value>>,
        idmap: &HashMap<String, Uid>,
    ) {
        use subpatch::{Boundary, Dir, LocalLink, MemberDecl, NestedDecl, Pillar};
        let (Some(insts), Some(defs)) = (insts_v, defs_v) else { return };

        // Mint fresh ids first, so nested member refs + parent refs resolve regardless of order.
        let mut instmap: HashMap<String, Uid> = HashMap::new();
        for old in insts.keys() {
            instmap.insert(old.clone(), self.mint());
        }
        let mut defmap: HashMap<String, subpatch::DefId> = HashMap::new();
        for old in defs.keys() {
            defmap.insert(old.clone(), self.mint_def());
        }
        let resolve_uid = |s: &str| idmap.get(s).copied().or_else(|| instmap.get(s).copied());

        // 1. Instance records + membership tags.
        for (old, rec) in insts {
            let uid = instmap[old];
            let Some(def_id) = rec.get("def").and_then(|v| v.as_str()).and_then(|d| defmap.get(d)).copied() else {
                continue;
            };
            let name = rec.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let parent = rec.get("parent").and_then(|v| v.as_str()).and_then(|s| instmap.get(s)).copied();
            let pos = rec
                .get("pos")
                .and_then(|v| v.as_array())
                .and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?]))
                .unwrap_or([0.0, 0.0]);
            let mut members: IndexMap<subpatch::Local, Uid> = IndexMap::new();
            if let Some(m) = rec.get("members").and_then(|v| v.as_object()) {
                for (local, mv) in m {
                    if let Some(ru) = mv.as_str().and_then(resolve_uid) {
                        members.insert(local.clone(), ru);
                    }
                }
            }
            for (local, &muid) in &members {
                self.scope_of.insert(muid, Some(uid));
                self.local_of.insert(muid, local.clone());
            }
            self.scope_of.insert(uid, parent);
            self.instances.insert(uid, subpatch::Instance { uid, name, def_id, pos, members });
        }

        // 2a. Deserialize every def's NAME + INTERFACE first (empty body) so that when 2b maps a
        //     nested-boundary link, every instance's def interface is already present for
        //     `boundary_exposing` regardless of def iteration order.
        for (old, rec) in defs {
            let def_id = defmap[old];
            let def_name = rec.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let mut interface: IndexMap<subpatch::BndId, Boundary> = IndexMap::new();
            if let Some(iface) = rec.get("interface").and_then(|v| v.as_object()) {
                for (bnd, b) in iface {
                    let dir = if b.get("dir").and_then(|v| v.as_str()) == Some("in") { Dir::In } else { Dir::Out };
                    let dtype = match b.get("dtype").and_then(|v| v.as_str()) {
                        Some("STRING") => goofi_core::SlotType::String,
                        Some("TABLE") => goofi_core::SlotType::Table,
                        _ => goofi_core::SlotType::Array,
                    };
                    let inner = match (
                        b.get("inner_local").and_then(|v| v.as_str()),
                        b.get("inner_slot").and_then(|v| v.as_str()),
                    ) {
                        (Some(l), Some(s)) => Some((l.to_string(), s.to_string())),
                        _ => None,
                    };
                    let pos = b
                        .get("pos")
                        .and_then(|v| v.as_array())
                        .and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?]))
                        .unwrap_or([0.0, 0.0]);
                    let name = b.get("name").and_then(|v| v.as_str()).unwrap_or(bnd).to_string();
                    interface.insert(bnd.clone(), Boundary { dir, pillar: Pillar::Signal, dtype, inner, pos, name });
                }
            }
            self.defs.insert(def_id, subpatch::SubPatchDef { name: def_name, members: IndexMap::new(), links: vec![], interface });
        }

        // 2b. Populate every def's member bodies (Leaf/Nested) from a referencing instance's live
        //     members, BEFORE any link capture. Link capture (2c) resolves nested-boundary
        //     endpoints via `resolve_boundary`, which reads a nested CHILD def's members to tell
        //     Leaf from Nested — so all members must exist first, else a def whose id exceeds its
        //     nested child's (e.g. a `make_unique` fork) would resolve against empty members and
        //     silently drop the interior link. Interfaces (2a) + members (2b) → 2c is order-free.
        for old in defs.keys() {
            let def_id = defmap[old];
            let Some(inst) = self.instances.values().find(|i| i.def_id == def_id).cloned() else { continue };
            let mut members: IndexMap<subpatch::Local, MemberDecl> = IndexMap::new();
            for (local, &muid) in &inst.members {
                let decl = if let Some(nested) = self.instances.get(&muid) {
                    MemberDecl::Nested(NestedDecl { def_id: nested.def_id, pos: nested.pos })
                } else if let Some(leaf) = self.capture_leaf_decl(muid) {
                    MemberDecl::Leaf(leaf)
                } else {
                    continue;
                };
                members.insert(local.clone(), decl);
            }
            if let Some(def) = self.defs.get_mut(&def_id) {
                def.members = members;
            }
        }

        // 2c. Capture each def's internal links from a referencing instance's live links. With all
        //     interfaces + members present, `endpoint_slot`/`boundary_exposing` resolve nested
        //     endpoints regardless of def order. A link with BOTH endpoints buried inside ONE
        //     nested member belongs to that member's own def — skip it (else an invalid self-loop).
        for old in defs.keys() {
            let def_id = defmap[old];
            let Some(inst) = self.instances.values().find(|i| i.def_id == def_id).cloned() else { continue };
            let member_set: std::collections::HashSet<Uid> = inst.members.values().copied().collect();
            let mut links: Vec<LocalLink> = Vec::new();
            for l in &self.links {
                let (Some(om), Some(im)) =
                    (self.containing_member(l.node_out, &member_set), self.containing_member(l.node_in, &member_set))
                else {
                    continue;
                };
                if om == im && self.instances.contains_key(&om) {
                    continue; // fully inside a nested member — captured in that member's own def
                }
                if let (Some(ol), Some(il)) = (self.local_of.get(&om).cloned(), self.local_of.get(&im).cloned()) {
                    links.push(LocalLink {
                        out: ol,
                        out_slot: self.endpoint_slot(om, l.node_out, l.slot_out, Dir::Out),
                        in_: il,
                        in_slot: self.endpoint_slot(im, l.node_in, l.slot_in, Dir::In),
                    });
                }
            }
            if let Some(def) = self.defs.get_mut(&def_id) {
                def.links = links;
            }
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
                    if let Some(prod) = self.uid_by_name(&r.node) {
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
                let mut names: Vec<&str> = b.refs.iter().map(|r| r.node.as_str()).collect();
                names.sort_unstable();
                names.dedup();
                for nm in names {
                    let mut put = |k: (String, Option<String>), data: Option<Data>| {
                        seen.insert(k.clone(), data.as_ref().and_then(|d| d.meta().index));
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
                    if let Some(g) = entry.params.get_mut(&key.group) {
                        g.insert(key.name.clone(), p);
                    }
                    let triggers = entry.bindings.get(&key).is_some_and(|b| b.triggers_process);
                    if let Some(b) = entry.bindings.get_mut(&key) {
                        b.last_seen = seen;
                        b.last_eval = Some(now);
                        b.error = None;
                    }
                    if key.group == "common" {
                        entry.run_policy = RunPolicy::from_params(&entry.params);
                    }
                    if triggers {
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
            // Same "wants to run" predicate the tick uses (minus a consumed trigger).
            let wants_run = e.trigger_pending
                || !e.has_trigger_inputs
                || (e.run_policy.autotrigger && !wired.contains(uid));
            if !wants_run {
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

            // Phase A — run every runnable node in this level in parallel. Each
            // closure touches only its own entry (disjoint `&mut`), so there is no
            // shared state and the result is independent of thread scheduling.
            let ran: Vec<Uid> = {
                let batch: Vec<(Uid, &mut NodeEntry)> = self
                    .nodes
                    .iter_mut()
                    .filter(|(uid, e)| {
                        if !set.contains(uid) {
                            return false;
                        }
                        // A pure source free-runs; a fresh trigger fires; autotrigger
                        // free-runs only a node with no *wired* trigger (Python parity).
                        let wants_run = e.trigger_pending
                            || !e.has_trigger_inputs
                            || (e.run_policy.autotrigger && !wired.contains(uid));
                        let since_last = e.last_run.map(|t| now.saturating_duration_since(t).as_secs_f64());
                        e.run_policy.should_run(since_last, wants_run)
                    })
                    .map(|(uid, e)| {
                        e.last_run = Some(now);
                        e.ctx.now = now_secs;
                        // Live globals for `process` (Arc bump); `setup` latched them at insert time.
                        e.ctx.globals = globals.clone();
                        (*uid, e)
                    })
                    .collect();
                let ran: Vec<Uid> = batch.iter().map(|(u, _)| *u).collect();
                batch.into_par_iter().for_each(|(_, entry)| run_node(entry));
                ran
            };

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
    }
}

/// Run a single node's `process` in place: clear its outputs, tick its context,
/// stamp each emitted frame's continuity index, and capture any error or panic on
/// its error channel. Panic isolation keeps one faulty node from unwinding through
/// the scheduler (and, in the bridge, poisoning the graph mutex). Called from the
/// parallel phase, so it touches only `entry` (index stamping included — the
/// counter and both I/O buffers all live in `entry`, so it stays disjoint).
fn run_node(entry: &mut NodeEntry) {
    entry.trigger_pending = false;
    entry.ctx.tick += 1;
    for v in entry.outputs.values_mut() {
        *v = None;
    }
    // Materialize each multi slot's present frames in connection order for the node
    // (Arc-bump clones). Empty for nodes with no multi slots — the common case pays
    // nothing beyond an empty map.
    let multis: IndexMap<&'static str, Vec<Data>> = entry
        .multi_inputs
        .iter()
        .map(|(k, cells)| (*k, cells.iter().filter_map(|(_, _, o)| o.clone()).collect()))
        .collect();
    let inp = Inputs::with_multi(&entry.inputs, &multis);
    let params = goofi_node::Params::new(&entry.params);
    let node = &mut entry.node;
    let ctx = &mut entry.ctx;
    let mut out = Outputs::new(&mut entry.outputs);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        node.process(&inp, &mut out, ctx, &params)
    }));
    // The process/bootstrap error channel. A binding error is NOT folded in here — it is
    // derived on read by `last_error()`, so a binding that recovers surfaces even on a node
    // that never runs process again (an idle node's run_node is not called).
    entry.last_error = match result {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e.0),
        Err(p) => Some(panic_message(p)),
    };
    stamp_meta(entry);
    // Persist each freshly-emitted (stamped) frame so `latest_frame` keeps returning it
    // on later ticks where this node emits nothing — viewers of a sparse producer never
    // blink to None. Disjoint field borrows.
    let (outputs, last) = (&entry.outputs, &mut entry.last_outputs);
    for (slot, out) in outputs.iter() {
        if let Some(d) = out {
            last.insert(*slot, d.clone());
        }
    }
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
fn stamp_meta(entry: &mut NodeEntry) {
    // Nothing emitted this tick → no meta to stamp, and the ufreq meter only advances
    // on a productive emit. Skip the whole index-timeline scan (the common case for a
    // rate-gated or idle node that ran but produced nothing).
    if entry.outputs.values().all(|o| o.is_none()) {
        return;
    }
    // Only triggering inputs carry the data timeline; control inputs are excluded.
    let triggering: std::collections::HashSet<&str> = entry
        .manifest
        .inputs
        .iter()
        .filter(|s| s.trigger_process)
        .map(|s| s.name)
        .collect();
    // Snapshot the index-bearing triggering inputs (index, frame_count) — no borrow held.
    let input_frames: Vec<(u64, usize)> = entry
        .inputs
        .iter()
        .filter(|(name, _)| triggering.contains(*name))
        .filter_map(|(_, o)| o.as_ref())
        .filter_map(|d| d.meta().index.map(|i| (i, frame_count(d))))
        .collect();
    // Node-level ufreq: EMA of the inter-emit interval, inverted. `None` until the
    // second emit; a non-advancing clock (`dt <= 0`) keeps the prior estimate.
    let now = entry.ctx.now;
    let node_ufreq = {
        let m = &mut entry.ufreq_meter;
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
    // Disjoint field borrows: rewrite outputs while advancing the index counters.
    let outputs = &mut entry.outputs;
    let counters = &mut entry.index_counters;
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
    use goofi_core::{DType, Meta, SlotType, Value};
    use goofi_node::{
        default_factory, Isolation, Node, NodeManifest, NodeResult, OutputDecl, ParamDecl,
        ParamSpec, Params, SlotDecl,
    };

    /// Empty param declaration, shared by the many test nodes with no own params.
    static NO_PARAMS: &[ParamDecl] = &[];

    #[test]
    fn graph_seeds_and_edits_globals() {
        use goofi_core::globals::GlobalValue;
        let mut g = Graph::new();
        // A fresh graph carries the system globals.
        assert_eq!(g.globals().get("default_ufreq"), Some(&GlobalValue::Float(30.0)));
        assert!(g.globals().is_system("default_ufreq"));
        // Edit a system global's value; add + remove a user global.
        g.apply_global_change("default_ufreq", Some(GlobalValue::Int(60))).unwrap(); // coerces to Float
        assert_eq!(g.globals_snapshot().f64("default_ufreq"), Some(60.0));
        g.apply_global_change("subject", Some(GlobalValue::Str("P07".into()))).unwrap();
        assert_eq!(g.globals_snapshot().str("subject"), Some("P07"));
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
    fn default_expr_falls_back_to_the_literal_without_an_evaluator() {
        // No evaluator wired ⇒ no binding is minted; the param keeps its spec-default literal (5.0),
        // never an errored "no evaluator" binding. Graceful degrade for headless / eval-less runs.
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
        let params = goofi_node::with_common(goofi_node::find("_TestDefaultExpr").unwrap().default_params());
        let n = g.add_node_at("_TestDefaultExpr", Some(params), Uid(0xD15C), "restored").unwrap();
        assert!(
            g.param_expression(n, "control", "rate").is_none(),
            "restore must not auto-bind — the doc is the source of truth"
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
    }];
    static SINK_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "control",
        name: "value",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        default_expr: None,
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
                let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
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
            let d = Data::from_array_bytes(DType::F32, vec![1], (self.runs as f32).to_le_bytes().to_vec(), Meta::empty())
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
            let d = Data::from_array_bytes(DType::F32, vec![1], sum.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static ADD_IN: &[SlotDecl] = &[
        SlotDecl { name: "a", kind: SlotType::Array, trigger_process: true, multi: false },
        SlotDecl { name: "b", kind: SlotType::Array, trigger_process: true, multi: false },
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
            let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
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
            factory: default_factory::<Panicky>,
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
            let d = Data::from_array_bytes(DType::F32, vec![1], (self.runs as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    // 10 Hz (-> 0.1s), autotriggering. `frequency_mode` is filled by `with_common`.
    static CAPPED_PARAMS: &[ParamDecl] = &[
        ParamDecl { group: "common", name: "autotrigger", spec: ParamSpec::Bool { default: true }, default_expr: None },
        ParamDecl { group: "common", name: "max_frequency", spec: ParamSpec::Float { default: 10.0, min: 0.0, max: 60.0 }, default_expr: None },
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
            let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static REF_IN: &[SlotDecl] = &[
        SlotDecl { name: "data", kind: SlotType::Array, trigger_process: true, multi: false },
        SlotDecl { name: "ref", kind: SlotType::Array, trigger_process: false, multi: false },
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
            factory: default_factory::<RefLenChange>,
        }
    }

    // A source that emits the engine-supplied wall clock (ctx.now) as its value,
    // to prove NodeCtx::now advances deterministically under an injected clock.
    #[derive(Default)]
    struct NowSource;
    impl Node for NowSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let d = Data::from_array_bytes(DType::F32, vec![1], (c.now as f32).to_le_bytes().to_vec(), Meta::empty())
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
            let d = Data::from_array_bytes(DType::F32, vec![1], v.to_le_bytes().to_vec(), Meta::empty())
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
            let d = Data::from_array_bytes(DType::F32, vec![1], v.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static DEFAULT_EXPR_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "control",
        name: "rate",
        spec: ParamSpec::Float { default: 5.0, min: 0.0, max: 1000.0 },
        default_expr: Some("globals.default_ufreq"),
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
            factory: default_factory::<DefaultExprSource>,
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
                Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
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
            let d = Data::from_array_bytes(DType::F32, vec![vals.len()], bytes, Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    static COLLECT_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "common",
        name: "autotrigger",
        spec: ParamSpec::Bool { default: true },
        default_expr: None,
    }];
    static COLLECT_IN: &[SlotDecl] = &[SlotDecl {
        name: "ins",
        kind: SlotType::Array,
        trigger_process: true,
        multi: true,
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
            factory: default_factory::<Collect>,
        }
    }

    // `_TestConst` (the constant-array source these tests use as a generic value
    // source) is the hidden test node in goofi-nodes — one shared definition across
    // the engine, bridge, and goofi-py suites.

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
        assert!(yaml.contains("version: 5"));

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
        let bad = "version: 3\nnodes:\n  \"00000000000a\":\n    type: NotAReal Node\n    pos: [0, 0]\nlinks: []\n";
        assert!(g.load_doc(bad).is_err());
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

    // A runtime source built by a captured closure (not a bare fn pointer) —
    // stands in for a pyo3 node whose factory captures a Python class handle.
    struct RtSource {
        base: f32,
    }
    impl Node for RtSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let d = Data::from_array_bytes(DType::F32, vec![1], self.base.to_le_bytes().to_vec(), Meta::empty())
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
        factory: rt_stub_factory,
    };

    #[test]
    fn register_dyn_type_refuses_collisions() {
        let mut g = Graph::new();
        // Collides with the built-in "Oscillator": refused, and add_node still
        // resolves the native node (the dyn factory would panic via rt_stub_make).
        assert!(!g.register_dyn_type(&COLLIDE_MANIFEST, Box::new(|_| unreachable!())));
        assert!(g.dyn_type_manifests().is_empty());
        let osc = g.add_node("Oscillator", None).unwrap();
        assert_eq!(g.manifest(osc).unwrap().category, "inputs"); // the native one

        // A fresh name registers once; a second registration of the same name is
        // refused rather than overwriting (which would orphan the first's manifest).
        assert!(g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 }))));
        assert!(!g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 2.0 }))));
        assert_eq!(g.dyn_type_manifests().len(), 1);
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
        assert_eq!(f.meta().index, Some(2), "3 emits -> indices 0,1,2 (latest 2)");
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
                (MockExpr::Ref(name.to_string()), vec![goofi_node::ExprRef { node: name.to_string(), slot: None }])
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
    fn serialize_emits_v5_root_nested_and_roundtrips() {
        // .gfi v5: version 5, a `pillar_default`, nodes/links nested under `root`, plus a `globals`
        // block (the recursive multi-pillar envelope). A signal-only patch round-trips.
        let mut g = Graph::new();
        let n = g.add_node("_TestConst", None).unwrap();
        g.update_param(n, "constant", "value", Param::float(7.0, -1.0e9, 1.0e9)).unwrap();
        let yaml = g.serialize();
        assert!(yaml.contains("version: 5"), "emits v5; got:\n{yaml}");
        assert!(yaml.contains("pillar_default: signal"), "carries the default pillar");
        assert!(yaml.contains("root:"), "nodes/links nested under root");
        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_uids().len(), 1, "node round-trips");
        let uid2 = g2.node_uids()[0];
        assert_eq!(
            goofi_node::param(g2.params(uid2).unwrap(), "constant", "value").unwrap().as_f64(),
            Some(7.0),
            "param round-trips through v5",
        );
    }

    #[test]
    fn globals_round_trip_through_gfi_v5() {
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
    fn v4_patch_loads_with_system_globals_seeded() {
        // A pre-globals v4 patch (no `globals` block) loads fine — the system defaults are seeded.
        let v4 = "version: 4\npillar_default: signal\ndefinitions: {}\nroot:\n  nodes:\n    n0: { type: _TestConst, name: c0, pos: [1.0, 2.0], params: {} }\n  links: []\n  instances: {}\n";
        let mut g = Graph::new();
        g.load_doc(v4).unwrap();
        assert_eq!(g.node_uids().len(), 1, "v4 nodes load");
        assert_eq!(
            g.globals().get("default_ufreq"),
            Some(&goofi_core::globals::GlobalValue::Float(30.0)),
            "system default seeded on a globals-less patch",
        );
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
        for _ in 0..8 {
            let b = g.add_node("Buffer", None).unwrap();
            g.update_param(b, "buffer", "size", Param::int(256, 1, 1_000_000)).unwrap();
            unbounded(&mut g, b);
            g.add_link(osc, "out", b, "data").unwrap();
        }

        for _ in 0..100 {
            g.tick(); // warm up (buffers fill, buffers/paths hot)
        }
        let iters = 3000usize;
        let mut lat: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            g.tick();
            lat.push(t0.elapsed().as_secs_f64() * 1e6); // microseconds
        }
        // Every buffer produced a frame (stability — the graph propagated end-to-end each tick).
        assert!(g.node_uids().iter().all(|&u| g.latest_frame(u, "out").is_some()), "all nodes emit");

        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = lat.iter().sum::<f64>() / iters as f64;
        let p = |q: f64| lat[((iters as f64 * q) as usize).min(iters - 1)];
        eprintln!(
            "native graph tick latency (Oscillator→8 Buffers, {iters} ticks): \
             min={:.1}us  p50={:.1}us  p99={:.1}us  max={:.1}us  mean={mean:.1}us",
            lat[0], p(0.50), p(0.99), lat[iters - 1]
        );
        // A full 9-node graph tick must stay a tiny fraction of a 60 Hz budget (16.6 ms) — a
        // generous ceiling that still catches a regression to millisecond-scale per-tick cost.
        assert!(p(0.99) < 2000.0, "p99 tick {:.1}us exceeds the budget", p(0.99));
    }

    #[test]
    fn group_nodes_is_bookkeeping_only() {
        // Group a 2-node chain: one instance appears with two members, the interface exposes
        // the downstream output, and the FLAT graph is byte-identical — same node uids, same
        // links, no respawn.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        let c = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap(); // wholly inside once [a,b] group
        g.add_link(b, "out", c, "in").unwrap(); // CUT: b inside, c outside → output boundary
        let nodes_before = g.node_uids();
        let links_before = g.links_view().len();

        let inst = g.group_nodes(&[a, b], [100.0, 100.0]).unwrap();

        // Flat runtime unchanged.
        assert_eq!(g.node_uids(), nodes_before, "no node minted/removed; uids identical");
        assert_eq!(g.links_view().len(), links_before, "both links untouched (boundary is a view)");
        // Membership re-tagged.
        assert_eq!(g.scope_of(a), Some(inst), "a is now a member of the instance");
        assert_eq!(g.scope_of(b), Some(inst));
        assert_eq!(g.scope_of(c), None, "c stays at ROOT (external)");
        assert_eq!(g.scope_of(inst), None, "the instance sits at ROOT");
        // Forest shape: one instance, one def refcount 1 (unique), two members.
        let instance = g.instance(inst).unwrap();
        assert_eq!(instance.members.len(), 2);
        assert_eq!(g.def_refcount(instance.def_id), 1, "unique def");
        let def = g.def(instance.def_id).unwrap();
        // The a→b link is internal (a def local link); only the b→c cut mints a boundary.
        assert_eq!(def.links.len(), 1, "a→b captured as an internal local link");
        assert_eq!(def.interface.len(), 1, "one cut link → one boundary");
        let (_bnd, boundary) = def.interface.iter().next().unwrap();
        assert_eq!(boundary.dir, subpatch::Dir::Out, "downstream output boundary");
        let (inner_local, inner_slot) = boundary.inner.as_ref().expect("wired boundary");
        assert_eq!(inner_local, g.name(b).unwrap(), "inner is b (by its local name)");
        assert_eq!(inner_slot, "out", "inner is b's out slot");
    }

    #[test]
    fn group_nodes_rejects_mixed_scope_without_mutating() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestConst", None).unwrap();
        let inner = g.group_nodes(&[a], [0.0, 0.0]).unwrap(); // a is now scoped to `inner`
        let defs_before = g.instance_uids().len();
        // a (in `inner`) + b (ROOT) span two scopes → rejected, nothing changes.
        let err = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap_err();
        assert!(err.contains("scope"), "mixed-scope error; got {err}");
        assert_eq!(g.instance_uids().len(), defs_before, "no instance created on failure");
        assert_eq!(g.scope_of(a), Some(inner), "a's membership untouched");
        assert_eq!(g.scope_of(b), None, "b stays at ROOT");
    }

    #[test]
    fn boundary_authoring_add_wire_rename_and_one_per_inner() {
        use subpatch::Dir;
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        // Group just `a`: a→b is a cut, so a.out is auto-exposed as out0. a.in is UNexposed
        // (no external link into it), so we author an input boundary onto it.
        let inst = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let def_id = g.instance(inst).unwrap().def_id;
        let before = g.def(def_id).unwrap().interface.len();

        let bnd = g.add_boundary(inst, Dir::In, goofi_core::SlotType::Array, [10.0, 10.0]).unwrap();
        assert_eq!(g.def(def_id).unwrap().interface.len(), before + 1, "boundary added");
        assert!(g.def(def_id).unwrap().interface[&bnd].inner.is_none(), "born unwired");

        g.wire_boundary(inst, &bnd, a, "in").unwrap();
        assert_eq!(
            g.resolve_boundary(inst, &bnd),
            Some((a, "in".to_string())),
            "wired boundary resolves to a.in",
        );

        // One boundary per inner slot: a.in is now exposed by `bnd`, so a second boundary
        // wiring to the same slot is rejected.
        let extra = g.add_boundary(inst, Dir::In, goofi_core::SlotType::Array, [0.0, 0.0]).unwrap();
        let err = g.wire_boundary(inst, &extra, a, "in").unwrap_err();
        assert!(err.contains("already exposed"), "one-boundary-per-inner enforced; got {err}");

        // rename keeps the bnd_id (external wires survive), only the label changes.
        g.rename_boundary(inst, &bnd, "signal").unwrap();
        assert_eq!(g.def(def_id).unwrap().interface[&bnd].name, "signal");
        assert!(g.def(def_id).unwrap().interface.contains_key(&bnd), "bnd_id unchanged after rename");

        // wiring a non-member is rejected.
        let outsider = g.add_node("_TestConst", None).unwrap();
        assert!(g.wire_boundary(inst, &bnd, outsider, "in").is_err(), "non-member rejected");
    }

    #[test]
    fn duplicate_shared_spawns_a_wired_sibling() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap(); // internal to the group
        let inst = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();
        let def_id = g.instance(inst).unwrap().def_id;
        assert_eq!(g.node_uids().len(), 2, "grouping spawned nothing");
        assert_eq!(g.def_refcount(def_id), 1, "unique before duplication");

        let sib = g.duplicate_shared(inst, [50.0, 50.0]).unwrap();
        assert_eq!(g.def_refcount(def_id), 2, "the def is now shared");
        assert_eq!(g.node_uids().len(), 4, "the sibling's two members were spawned");

        let orig: std::collections::HashSet<Uid> = g.instance(inst).unwrap().members.values().copied().collect();
        let sibs: std::collections::HashSet<Uid> = g.instance(sib).unwrap().members.values().copied().collect();
        assert!(orig.is_disjoint(&sibs), "sibling has its own distinct member uids");
        assert_eq!(orig, [a, b].into_iter().collect(), "the original's leaves are untouched");

        // The sibling's internal link (const'→echo') was projected live.
        let sib_uids: Vec<Uid> = g.instance(sib).unwrap().members.values().copied().collect();
        let linked = g.links_view().iter().any(|l| sib_uids.contains(&l.node_out) && sib_uids.contains(&l.node_in));
        assert!(linked, "the sibling's internal link is live");
        assert_eq!(g.links_view().len(), 2, "one internal link per instance, external untouched");
    }

    #[test]
    fn shared_param_edit_reprojects_to_every_sibling() {
        // A param edit on one shared member mirrors to the sibling (def is the SSOT), and the
        // def's stored decl is synced so a further duplicate inherits the edit.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let inst = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let sib = g.duplicate_shared(inst, [10.0, 10.0]).unwrap();
        let a2 = *g.instance(sib).unwrap().members.values().next().unwrap();

        // Peers = both instances' members of that local.
        let peers = g.shared_member_peers(a);
        assert_eq!(peers.len(), 2, "shared member has one sibling peer");

        // Edit the constant on the original member → both members update.
        let updated = g.update_member_param(a, "constant", "value", Param::float(9.0, -1e9, 1e9)).unwrap();
        assert_eq!(updated.len(), 2, "both siblings updated");
        let val = |g: &Graph, u| goofi_node::param(g.params(u).unwrap(), "constant", "value").unwrap().as_f64();
        assert_eq!(val(&g, a), Some(9.0), "edited member");
        assert_eq!(val(&g, a2), Some(9.0), "sibling mirrored");

        // The def carries the edit: a fresh duplicate inherits 9.0.
        let sib3 = g.duplicate_shared(inst, [20.0, 20.0]).unwrap();
        let a3 = *g.instance(sib3).unwrap().members.values().next().unwrap();
        assert_eq!(val(&g, a3), Some(9.0), "a new sibling inherits the edited param from the def");
    }

    #[test]
    fn shared_expression_edit_reprojects_to_every_sibling() {
        // Binding a param to an expression on one shared member mirrors the binding to every
        // sibling (the def is the SSOT), and the def's stored decl is synced so a further
        // duplicate inherits it — the §4.5 analogue of the literal-param re-projection.
        // Uses a disabled binding so the source round-trips without needing the pyo3 evaluator.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let inst = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let sib = g.duplicate_shared(inst, [10.0, 10.0]).unwrap();
        let a2 = *g.instance(sib).unwrap().members.values().next().unwrap();

        let updated = g
            .set_member_expression(a, "constant", "value", "sin(t)", false, false)
            .unwrap();
        assert_eq!(updated.len(), 2, "both siblings received the binding");
        let src = |g: &Graph, u| g.param_expression(u, "constant", "value").map(|e| e.source);
        assert_eq!(src(&g, a).as_deref(), Some("sin(t)"), "edited member bound");
        assert_eq!(src(&g, a2).as_deref(), Some("sin(t)"), "sibling mirrored the binding");

        // The def carries the binding: a fresh duplicate inherits the expression.
        let sib3 = g.duplicate_shared(inst, [20.0, 20.0]).unwrap();
        let a3 = *g.instance(sib3).unwrap().members.values().next().unwrap();
        assert_eq!(src(&g, a3).as_deref(), Some("sin(t)"), "a new sibling inherits the expression from the def");

        // Unbinding on one member likewise clears it on every sibling.
        let updated = g
            .set_member_expression(a, "constant", "value", "", false, false)
            .unwrap();
        assert_eq!(updated.len(), 3, "all three siblings cleared");
        assert_eq!(src(&g, a2), None, "sibling unbound");
        assert_eq!(src(&g, a3), None, "sibling unbound");
    }

    #[test]
    fn renaming_an_external_producer_follows_into_shared_defs() {
        // A shared sub-patch member references an external node via nd('ext'). Renaming that
        // external node must rewrite the reference in the DEFINITION too, not only in the
        // live bindings — else a later duplicate is instantiated from the stale name (Python:
        // _rewrite_record_nd over the definitions). Disabled binding => no evaluator needed.
        let mut g = Graph::new();
        let ext = g.add_node("_TestConst", None).unwrap();
        g.rename_node(ext, "ext").unwrap();
        let a = g.add_node("_TestConst", None).unwrap();
        let inst = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        g.duplicate_shared(inst, [10.0, 10.0]).unwrap(); // 2 instances -> the def is the SSOT
        g.set_member_expression(a, "constant", "value", "nd('ext')", false, false).unwrap();

        let touched = g.rename_node(ext, "signal").unwrap();
        let src = |g: &Graph, u| g.param_expression(u, "constant", "value").map(|e| e.source);
        assert_eq!(src(&g, a).as_deref(), Some("nd('signal')"), "live member binding rewritten");
        assert!(touched.contains(&a), "the live member is reported as a referrer");

        // The def followed the rename: a fresh duplicate inherits nd('signal'), not nd('ext').
        let sib3 = g.duplicate_shared(inst, [20.0, 20.0]).unwrap();
        let a3 = *g.instance(sib3).unwrap().members.values().next().unwrap();
        assert_eq!(
            src(&g, a3).as_deref(),
            Some("nd('signal')"),
            "a fresh duplicate inherits the rewritten ref from the def, not the stale name"
        );
    }

    #[test]
    fn shared_member_peers_are_only_the_node_itself_when_unique() {
        // A ROOT node and a unique-instance member have no peers (edits stay local).
        let mut g = Graph::new();
        let root = g.add_node("_TestConst", None).unwrap();
        assert_eq!(g.shared_member_peers(root), vec![root], "ROOT node: no peers");
        let inst = g.group_nodes(&[root], [0.0, 0.0]).unwrap();
        let _ = inst;
        assert_eq!(g.shared_member_peers(root), vec![root], "unique-def member: no peers");
    }

    #[test]
    fn remove_instance_tears_down_the_whole_subtree() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let inst = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();
        let def_id = g.instance(inst).unwrap().def_id;
        let sib = g.duplicate_shared(inst, [50.0, 50.0]).unwrap();
        assert_eq!(g.node_uids().len(), 4);

        // Removing the sibling removes ONLY its two members; the original survives.
        g.remove_instance(sib).unwrap();
        assert!(g.instance(sib).is_none(), "sibling instance gone");
        assert_eq!(g.node_uids().len(), 2, "only the sibling's members were removed");
        assert!(g.instance(inst).is_some(), "original instance untouched");
        assert_eq!(g.def_refcount(def_id), 1, "def back to unique (still referenced by the original)");

        // Removing the original tears down the rest + GCs the def.
        g.remove_instance(inst).unwrap();
        assert_eq!(g.node_uids().len(), 0, "all leaves torn down");
        assert!(g.def(def_id).is_none(), "def GC'd once unreferenced");
    }

    #[test]
    fn duplicate_shared_preserves_a_moved_original_members_pos() {
        // reconcile must not snap a surviving member back to the def's group-time pos.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let inst = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        g.set_node_pos(a, [123.0, 456.0]).unwrap(); // user moves the member after grouping
        g.duplicate_shared(inst, [10.0, 10.0]).unwrap();
        assert_eq!(g.pos(a), Some([123.0, 456.0]), "the original member keeps its moved position");
    }

    #[test]
    fn grouping_a_selection_containing_a_nested_instance_maps_interior_links_to_its_boundary() {
        // a → b, then group [a] so the a→b link is exposed as inner's out-boundary. Grouping
        // [inner, b] must recognize that a (inside the nested `inner`) is transitively within the
        // outer group: a→b is INTERNAL to outer, captured as a link from inner's boundary to b —
        // NOT mis-derived as an outer input boundary. The runtime model already supports a local
        // link whose endpoint slot is a nested instance's BndId (see subpatch::LocalLink); this is
        // the capture side catching up.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let inner = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        // inner exposes exactly one OUT boundary for a.out.
        let inner_def = g.instance(inner).unwrap().def_id;
        let bnd = g.def(inner_def).unwrap().interface.keys().next().unwrap().clone();

        let outer = g.group_nodes(&[inner, b], [100.0, 0.0]).unwrap();
        let def = g.def(g.instance(outer).unwrap().def_id).unwrap();

        assert!(def.interface.is_empty(), "nothing crosses outer's edge — a→b is fully internal");
        assert_eq!(def.links.len(), 1, "the inner→b connection is captured as one internal link");
        let link = &def.links[0];
        assert_eq!(link.out_slot, bnd, "the interior endpoint references inner's boundary as its slot");
        assert_eq!(link.in_slot, "in", "the leaf endpoint keeps its real slot");
        // The out local names the nested instance; the in local names the leaf b.
        let inner_local = g.local_of(inner).map(|s| s.to_string()).or_else(|| g.name(inner).map(str::to_string)).unwrap();
        let b_local = g.local_of(b).map(|s| s.to_string()).or_else(|| g.name(b).map(str::to_string)).unwrap();
        assert_eq!(link.out, inner_local, "internal link's producer is the nested instance");
        assert_eq!(link.in_, b_local, "internal link's consumer is the leaf");
    }

    #[test]
    fn duplicate_shared_projects_a_nested_sub_patchs_internal_link() {
        // Sharing an outer sub-patch that CONTAINS a nested sub-patch must spawn the sibling's own
        // nested instance AND wire its interior link (a→b, crossing the inner boundary) live —
        // proving materialize/reconcile resolve a local link whose slot is a nested BndId.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let inner = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let outer = g.group_nodes(&[inner, b], [100.0, 0.0]).unwrap();
        assert_eq!(g.node_uids().len(), 2, "grouping is bookkeeping-only");
        assert_eq!(g.links_view().len(), 1, "one live internal link a→b");

        let sib = g.duplicate_shared(outer, [200.0, 0.0]).unwrap();
        assert_eq!(g.node_uids().len(), 4, "the sibling's own a'/b' leaves were spawned");
        let sib_members: Vec<Uid> = g.instance(sib).unwrap().members.values().copied().collect();
        assert!(sib_members.iter().any(|m| g.instance(*m).is_some()), "sibling has its own nested instance");
        assert_eq!(g.links_view().len(), 2, "the sibling's interior a'→b' link is live");
    }

    #[test]
    fn nested_shared_sub_patch_survives_a_gfi_round_trip() {
        // Save/load a SHARED sub-patch that contains a nested sub-patch, then duplicate again on
        // the reloaded graph. The post-load duplicate only wires correctly if reload_forest
        // re-captured the outer def body with the nested-boundary internal link (not a spurious
        // input boundary) — the load-side analogue of the group_nodes capture fix.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let inner = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let outer = g.group_nodes(&[inner, b], [100.0, 0.0]).unwrap();
        g.duplicate_shared(outer, [200.0, 0.0]).unwrap();
        assert_eq!(g.node_uids().len(), 4);
        assert_eq!(g.links_view().len(), 2);
        let doc = g.serialize();

        let mut g2 = Graph::new();
        g2.load_doc(&doc).unwrap();
        assert_eq!(g2.node_uids().len(), 4, "all four leaves reloaded");
        assert_eq!(g2.links_view().len(), 2, "both interior links reloaded");

        // Duplicate a reloaded root instance — the third sibling's whole subtree must spawn+wire.
        let root_inst = g2
            .instance_uids()
            .into_iter()
            .find(|&u| g2.scope_of(u).is_none())
            .expect("a root sub-patch instance survived load");
        g2.duplicate_shared(root_inst, [300.0, 0.0]).unwrap();
        assert_eq!(g2.node_uids().len(), 6, "third sibling's a''/b'' spawned from the reloaded def");
        assert_eq!(g2.links_view().len(), 3, "third sibling's interior link wired from the reloaded def");
    }

    #[test]
    fn expanding_an_outer_sub_patch_re_parents_its_nested_instance_to_the_grandparent() {
        // Un-grouping an outer that contains a nested sub-patch must reset the nested instance's
        // scope to the grandparent (here ROOT).
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let inner = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let outer = g.group_nodes(&[inner, b], [100.0, 0.0]).unwrap();
        assert_eq!(g.scope_of(inner), Some(outer), "inner nested under outer");

        g.expand_instance(outer).unwrap();
        assert!(g.instance(outer).is_none(), "outer dissolved");
        assert!(g.instance(inner).is_some(), "inner survives as a now-root instance");
        assert_eq!(g.scope_of(inner), None, "inner re-tagged to ROOT scope");
    }

    #[test]
    fn grouping_a_nested_member_does_not_capture_its_private_internal_link() {
        // `inner` has a FULLY-internal link (a→mid, both buried inside inner). Grouping [inner]
        // into `outer` must NOT sweep that buried link into outer's def as an invalid self-loop —
        // it belongs to inner's own def. Regression for the transitive-containment over-capture.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let mid = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", mid, "in").unwrap();
        let inner = g.group_nodes(&[a, mid], [0.0, 0.0]).unwrap(); // a→mid fully internal to inner
        assert!(g.def(g.instance(inner).unwrap().def_id).unwrap().interface.is_empty(), "inner exposes nothing");

        let outer = g.group_nodes(&[inner], [100.0, 0.0]).unwrap();
        let outer_def = g.def(g.instance(outer).unwrap().def_id).unwrap();
        assert!(outer_def.links.is_empty(), "the buried a→mid link is NOT captured into outer (no self-loop)");
        assert!(outer_def.interface.is_empty(), "outer exposes nothing either");

        // Sharing outer still projects each sibling's own a→mid — no phantom/duplicated links.
        g.duplicate_shared(outer, [200.0, 0.0]).unwrap();
        assert_eq!(g.node_uids().len(), 4, "sibling spawned its own a'/mid'");
        assert_eq!(g.links_view().len(), 2, "each instance's a→mid is live; no spurious link");
    }

    #[test]
    fn rename_rejects_collision_with_a_sub_patch_instance_name() {
        // Leaves and instances share one display-name namespace; renaming a leaf onto an instance's
        // `subpatch{N}` name would collapse them to one member local key on grouping and drop one.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestConst", None).unwrap();
        let inst = g.group_nodes(&[b], [0.0, 0.0]).unwrap();
        let inst_name = g.instance(inst).unwrap().name.clone();
        assert!(g.rename_node(a, &inst_name).is_err(), "cannot rename a leaf onto an instance's name");
    }

    #[test]
    fn group_nodes_mints_a_unique_instance_name_avoiding_a_future_leaf_collision() {
        // rename_node guards a leaf against colliding with an EXISTING instance name, but the mint
        // side was unguarded: a leaf renamed to `subpatch{N}` where N is a not-yet-minted uid is
        // allowed, then the instance minted at Uid(N) would take the same display name — collapsing
        // the two into one member local key on a later group and orphaning the leaf. The mint must
        // pick a unique name.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap(); // uid 1
        let b = g.add_node("_TestConst", None).unwrap(); // uid 2 → the next mint() is uid 3
        // Rename a leaf to the name the next instance WOULD mint. Allowed today (nothing is named it).
        g.rename_node(a, "subpatch3").unwrap();
        let inst = g.group_nodes(&[b], [0.0, 0.0]).unwrap();
        // The instance name must NOT collide with the leaf's display name.
        assert_ne!(g.instance(inst).unwrap().name, g.name(a).unwrap(), "minted instance name must be unique");
        // And grouping the leaf + the instance keeps the leaf a REAL member (not orphaned).
        let outer = g.group_nodes(&[a, inst], [0.0, 0.0]).unwrap();
        assert_eq!(g.scope_of(a), Some(outer));
        assert!(
            g.instance(outer).unwrap().members.values().any(|&u| u == a),
            "the leaf stays a real member of the outer instance"
        );
    }

    #[test]
    fn make_unique_and_re_share_reject_a_nested_instance() {
        // Forking a nested instance in isolation would desync its parent def's NestedDecl, so both
        // make_unique and re_share reject it (root-only). The enclosing root instance still forks.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let inner = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let outer = g.group_nodes(&[inner, b], [10.0, 0.0]).unwrap();
        assert_eq!(g.scope_of(inner), Some(outer), "inner is nested");

        assert!(g.make_unique(inner).is_err(), "cannot fork a nested instance");
        let some_def = g.instance(outer).unwrap().def_id;
        assert!(g.re_share_instance(inner, some_def).is_err(), "cannot re-share a nested instance");
        assert!(g.make_unique(outer).is_ok(), "a ROOT instance still forks");
    }

    #[test]
    fn make_unique_forks_a_shared_def_and_re_share_inverts_it() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let inst = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let shared_def = g.instance(inst).unwrap().def_id;
        let sib = g.duplicate_shared(inst, [10.0, 10.0]).unwrap();
        assert_eq!(g.def_refcount(shared_def), 2);
        let node_count = g.node_uids().len();

        let new_def = g.make_unique(sib).unwrap();
        assert_ne!(new_def, shared_def, "a fresh private def");
        assert_eq!(g.def_refcount(shared_def), 1, "original instance still shares the old def");
        assert_eq!(g.def_refcount(new_def), 1, "the sibling is now on its private fork");
        assert_eq!(g.node_uids().len(), node_count, "no respawn — pure bookkeeping");

        // re_share_instance is the exact inverse: repoint back + GC the fork.
        g.re_share_instance(sib, shared_def).unwrap();
        assert_eq!(g.def_refcount(shared_def), 2, "re-shared onto the original def");
        assert!(g.def(new_def).is_none(), "the abandoned private fork is GC'd");
    }

    #[test]
    fn expand_instance_restores_membership_and_gcs_the_def() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let inst = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();
        let def_id = g.instance(inst).unwrap().def_id;

        let restored = g.expand_instance(inst).unwrap();
        assert_eq!(restored.len(), 2, "both members restored");
        assert!(restored.contains(&a) && restored.contains(&b), "same member uids");
        assert_eq!(g.scope_of(a), None, "a back at ROOT");
        assert_eq!(g.scope_of(b), None);
        assert!(g.instance(inst).is_none(), "instance dropped");
        assert_eq!(g.def_refcount(def_id), 0, "def unreferenced");
        assert!(g.def(def_id).is_none(), "def GC'd");
        // The flat a→b link survived the round-trip.
        assert_eq!(g.links_view().len(), 1, "external/internal links intact");
    }

    #[test]
    fn sub_patch_forest_survives_a_gfi_roundtrip() {
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        let inst = g.group_nodes(&[a, b], [10.0, 20.0]).unwrap();
        g.duplicate_shared(inst, [200.0, 0.0]).unwrap();
        let def0 = g.instance(inst).unwrap().def_id;
        assert_eq!(g.def_refcount(def0), 2, "shared before save");

        let yaml = g.serialize();
        assert!(yaml.contains("instances:"), "forest persisted");
        assert!(yaml.contains("definitions:"), "defs persisted");

        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_uids().len(), 4, "all four member leaves restored");
        assert_eq!(g2.instance_uids().len(), 2, "both instances restored");
        assert_eq!(g2.links_view().len(), 2, "each instance's internal link restored");

        // Both instances still share ONE def (refcount 2) — sharing survives the round-trip.
        let insts = g2.instance_uids();
        let (d0, d1) = (g2.instance(insts[0]).unwrap().def_id, g2.instance(insts[1]).unwrap().def_id);
        assert_eq!(d0, d1, "both instances reference the same reconstructed def");
        assert_eq!(g2.def_refcount(d0), 2, "shared refcount preserved");
        assert_eq!(g2.instance(insts[0]).unwrap().members.len(), 2, "members restored");

        // Proof the def BODY was reconstructed (not just its shell): it can spawn a new sibling
        // with its two members + internal link.
        let before = g2.node_uids().len();
        g2.duplicate_shared(insts[0], [0.0, 0.0]).unwrap();
        assert_eq!(g2.node_uids().len(), before + 2, "reconstructed def spawns a wired sibling");
    }

    #[test]
    fn v3_document_upconverts_and_loads() {
        // A legacy flat v3 document (nodes/links at the top level) still loads — the loader
        // up-converts it (wraps under `root`) so old patches keep working.
        let v3 = "version: 3\nnodes:\n  n0: { type: _TestConst, name: c0, pos: [1.0, 2.0], params: {} }\nlinks: []\n";
        let mut g = Graph::new();
        g.load_doc(v3).unwrap();
        assert_eq!(g.node_uids().len(), 1, "v3 node loaded");
        assert_eq!(g.name(g.node_uids()[0]), Some("c0"), "v3 name preserved");
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
        assert_eq!(f.meta().index, Some(3), "propagates the source's index, not fresh 0");
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
            idx.push(g.latest_frame(buf, "out").unwrap().meta().index.unwrap());
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
        assert_eq!(f.meta().index, Some(0), "fresh counter, not the source's 3");
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
        // Default is unbounded + not autotriggering (behavior-preserving).
        assert_eq!(common["max_frequency"].as_f64(), Some(0.0));
        assert_eq!(common["autotrigger"].as_bool(), Some(false));
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
        assert_eq!(g.latest_frame(c, "out").unwrap().meta().index, Some(2), "capped to 3 emits");
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
        assert_eq!(g2.latest_frame(c2, "out").unwrap().meta().index, Some(1), "gate active post-load");
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
        g.latest_frame(uid, slot).unwrap().meta().ufreq
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
        assert!((frame.meta().ufreq.unwrap() - 100.0).abs() < 1e-6);

        let wire = goofi_codec::encode(&frame);
        let back = goofi_codec::decode(&wire).expect("data-plane frame decodes");
        assert_eq!(back.meta().ufreq, frame.meta().ufreq, "ufreq round-trips the data plane");
        assert!((back.meta().ufreq.unwrap() - 100.0).abs() < 1e-6);
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
        assert_eq!(g.latest_frame(src, "out").unwrap().meta().index, Some(4));
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
        assert_eq!(f.meta().index, Some(0), "control input must not be the timeline");
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

    #[test]
    fn set_member_pos_moves_a_sub_patch_instance_box() {
        // Dragging a collapsed sub-patch instance box routes set_node_pos → set_member_pos with
        // the INSTANCE uid. An instance lives in `instances`, not `nodes`, so the old delegation
        // to set_node_pos returned Err("no such node") and the box never moved. It must update
        // the instance's own pos and report the instance uid as moved.
        let mut g = Graph::new();
        let a = g.add_node("_TestConst", None).unwrap();
        let b = g.add_node("_TestConst", None).unwrap();
        let inst = g.group_nodes(&[a, b], [10.0, 20.0]).unwrap();
        assert_eq!(g.instance(inst).unwrap().pos, [10.0, 20.0]);

        let moved = g.set_member_pos(inst, [77.0, 88.0]).expect("moving an instance box succeeds");
        assert_eq!(moved, vec![inst], "the instance uid is reported moved");
        assert_eq!(g.instance(inst).unwrap().pos, [77.0, 88.0], "the box position updated");
    }
}
