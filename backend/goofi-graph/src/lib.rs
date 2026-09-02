//! The graph, and the nodes that schedule themselves.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use goofi_core::Param;
use goofi_node::{
    BindingView, BoundVar, DrainWaker, Edge, Engine, EventId, ExprMode, GraphView, Isolation,
    IsolationCell, LibraryEntry, NodeManifest, NodeView, ParamDecl, ParamGroups, ParamKey,
    Request, Status, Touched,
};
use indexmap::IndexMap;

pub mod archive;

pub mod subpatch;
pub mod layout;

pub mod command;
pub use command::{open_batch, BatchScope, Command, CommandHistory, Outcome, SourceState};

pub mod expr_rewrite;

pub use goofi_node::Uid;

/// One literal for the writer, the reader and the refusal message, so a bump cannot leave the
/// message lying about what this build reads. It moves when a format change has to reject an
/// archive somebody actually holds — not once per change while the format is still moving.
const MANIFEST_VERSION: i64 = 1;

pub use goofi_core::globals::NAME_RULE;

/// What a node IS. The thin distinction the backend keeps and the frontend never sees: a leaf runs,
/// so it carries a thread and params; a facade and a port do not, so they carry neither.
enum Kind {
    /// Boxed: a leaf's runtime state dwarfs the other two, and one map holds all three.
    Leaf(Box<Leaf>),
    /// A sub-patch facade. Its members are whatever `scope_of` places inside it.
    Facade,
    /// A boundary port. It relays rather than produces, so its dtype is fixed by its type at birth.
    Port(subpatch::Port),
}

/// The state only a RUNNING node has: the RECORD the ops write, and the health its instance
/// reports.
struct Leaf {
    manifest: &'static NodeManifest,
    /// The type's cell, captured at birth — shared per type, so a runtime demotion of a Python
    /// type reads through here too.
    isolation: &'static IsolationCell,
    /// The id of the engine whose library resolved this node's type — its runtime authority.
    engine: &'static str,
    /// The param RECORD — the literals `serialize` writes. An evaluated value must never reach it.
    params: Arc<ParamGroups>,
    /// The graph resolves each binding's references and ships it; the NODE evaluates it.
    bindings: HashMap<ParamKey, ParamSource>,
    health: Health,
}

/// What the running instance reports about itself — a one-way projection with two writers by
/// construction: a BIRTH replaces the whole struct, and the status drain mutates it after.
struct Health {
    /// `Some` when INITIALIZATION failed — the param replay and `setup()` together, which are one
    /// unit. Not `last_error`, which a later process failure would overwrite.
    setup_error: Option<String>,
    last_error: Option<String>,
    /// The derived error and WHEN it first read that way — re-stamped only when the message
    /// changes, so the instant is its onset: a settling pipeline reads differently from a broken one.
    error_since: Option<(String, Instant)>,
    /// The stage the node last reported; `creating` until it reports anything. The `error` stage
    /// is DERIVED from the fault, never stored here.
    stage: &'static str,
    /// The same number the node stamps as `meta["ufreq"]`. `None` until it has emitted twice.
    ufreq: Option<f64>,
    /// What the node last reported evaluating its bindings to. Kept apart from `params` so a
    /// broken binding still has the authored literal to fall back to.
    evaluated: IndexMap<ParamKey, Param>,
    /// Every error THIS INSTANCE reported, by param. It dies with the instance, where
    /// `ParamSource::bind_error` survives because the source does.
    param_errors: IndexMap<ParamKey, String>,
    /// A refreshable `Str` param's re-enumerated options — the overlay a projection reads over
    /// the record's declared options. The answer to a ⟳ lands here, never in the record.
    options: IndexMap<ParamKey, Vec<String>>,
}

impl Health {
    /// A birth's whole health: fresh, so a reborn node cannot show its predecessor's numbers,
    /// carrying the one fact only the birth knows — whether its services came up.
    fn born(boot_error: Option<String>) -> Health {
        Health {
            setup_error: boot_error,
            last_error: None,
            error_since: None,
            stage: "creating",
            ufreq: None,
            evaluated: IndexMap::new(),
            param_errors: IndexMap::new(),
            options: IndexMap::new(),
        }
    }
}

/// One entry in the ONE node map: a leaf, a sub-patch facade or a boundary port. Everything an op
/// can address about a node — its name, where it sits, what it shows — is here for all three.
struct NodeEntry {
    kind: Kind,
    name: String,
    pos: [f64; 2],
    /// Opaque: persisted and round-tripped, never interpreted.
    viewers: serde_json::Value,
}

impl NodeEntry {
    fn leaf(&self) -> Option<&Leaf> {
        match &self.kind {
            Kind::Leaf(l) => Some(l),
            _ => None,
        }
    }
    fn leaf_mut(&mut self) -> Option<&mut Leaf> {
        match &mut self.kind {
            Kind::Leaf(l) => Some(l),
            _ => None,
        }
    }
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

/// A param's value as JSON, and the one definition of it — the inverse of [`param_from_json`].
pub fn param_value_json(p: &Param) -> serde_json::Value {
    use serde_json::json;
    match p {
        Param::Float { value, .. } => json!(value),
        Param::Int { value, .. } => json!(value),
        Param::Bool { value } => json!(value),
        Param::Str { value, .. } => json!(value),
    }
}

/// Coerce a JSON scalar into a `Param` of `existing`'s type, keeping its bounds.
pub fn param_from_json(existing: &Param, v: &serde_json::Value) -> Param {
    match existing {
        Param::Float { vmin, vmax, .. } => Param::Float { value: v.as_f64().unwrap_or(0.0), vmin: *vmin, vmax: *vmax },
        Param::Int { vmin, vmax, .. } => Param::Int {
            value: v.as_i64().or_else(|| v.as_f64().map(|f| f.round() as i64)).unwrap_or(0),
            vmin: *vmin,
            vmax: *vmax,
        },
        Param::Bool { .. } => Param::Bool { value: v.as_bool().unwrap_or(false) },
        Param::Str { options, refresh, .. } => Param::Str {
            value: v.as_str().unwrap_or("").to_string(),
            options: options.clone(),
            refresh: *refresh,
        },
    }
}

/// The one params bag — `node add`'s birth entries and `node param edit` both fold into
/// `{group: {param: value | {value, expression, reference, mode, triggers}}}` — as one [`Command::EditParam`]
/// per entry, typed and merged against the params the node holds NOW.
pub fn param_commands(
    g: &Graph,
    uid: Uid,
    params: &serde_json::Value,
) -> Result<Vec<Command>, String> {
    let groups = params.as_object().ok_or("params is {group: {param: …}}")?;
    let mut cmds = Vec::new();
    for (group, entries) in groups {
        let entries =
            entries.as_object().ok_or_else(|| format!("params.{group} is {{param: …}}"))?;
        for (name, spec) in entries {
            let existing = g
                .params(uid)
                .and_then(|p| goofi_node::param(&p, group, name).cloned())
                .ok_or_else(|| format!("no param {group}.{name}"))?;
            let cur = g.param_source(uid, group, name);
            let (value, source) = param_change(&existing, cur, spec)
                .map_err(|e| format!("params.{group}.{name}: {e}"))?;
            if value.is_none() && source.is_none() {
                return Err(format!("params.{group}.{name} sets neither a value nor a source"));
            }
            cmds.push(Command::EditParam {
                uid,
                group: group.clone(),
                name: name.clone(),
                value,
                source,
            });
        }
    }
    Ok(cmds)
}

/// A CLI `--value` arrives as its raw string; the DECLARED type says what it meant.
fn coerced_value(existing: &Param, v: &serde_json::Value) -> Param {
    let parsed = match (existing, v.as_str()) {
        (Param::Str { .. }, _) | (_, None) => None,
        (_, Some(s)) => serde_json::from_str::<serde_json::Value>(s).ok(),
    };
    param_from_json(existing, parsed.as_ref().unwrap_or(v))
}

/// One `params.<group>.<name>` entry: a bare literal, or `{value, expression, reference, mode,
/// triggers}`. No param type is an object, so the two forms cannot be confused. A text given names
/// its mode unless one is said; a mode or a trigger alone edits what is retained.
fn param_change(
    existing: &Param,
    cur: Option<SourceInfo>,
    spec: &serde_json::Value,
) -> Result<(Option<Param>, Option<SourceState>), String> {
    let Some(o) = spec.as_object() else {
        return Ok((Some(coerced_value(existing, spec)), None));
    };
    if let Some(k) = o
        .keys()
        .find(|k| !matches!(k.as_str(), "value" | "expression" | "reference" | "mode" | "triggers"))
    {
        return Err(format!("unknown field `{k}` — value, expression, reference, mode, triggers"));
    }
    let field = |k: &str| o.get(k).filter(|v| !v.is_null());
    let value = field("value").map(|v| coerced_value(existing, v));
    let text = |k: &str| {
        field(k)
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| format!("{k} is a string, not {v}")))
            .transpose()
    };
    let expression = text("expression")?;
    let reference = text("reference")?;
    let mode = field("mode")
        .map(|v| {
            v.as_str()
                .and_then(Mode::parse)
                .ok_or_else(|| format!("mode is `constant`, `expression` or `reference`, not {v}"))
        })
        .transpose()?;
    let triggers = field("triggers")
        .map(|v| v.as_bool().ok_or_else(|| format!("triggers is a bool, not {v}")))
        .transpose()?;
    if expression.is_none() && reference.is_none() && mode.is_none() && triggers.is_none() {
        return Ok((value, None));
    }
    let given = |t: &Option<String>| t.as_deref().is_some_and(|s| !s.is_empty());
    let mode = match (mode, given(&expression), given(&reference)) {
        (Some(m), _, _) => m,
        (None, true, true) => {
            return Err("an expression and a reference at once: say which with `mode`".to_string())
        }
        (None, true, false) => Mode::Expression,
        (None, false, true) => Mode::Reference,
        // Clearing the active text is what switches it off.
        (None, false, false) => match cur.as_ref().map(|c| c.mode).unwrap_or_default() {
            Mode::Expression if expression.as_deref() == Some("") => Mode::Constant,
            Mode::Reference if reference.as_deref() == Some("") => Mode::Constant,
            m => m,
        },
    };
    let state = SourceState {
        mode,
        expression: expression.or_else(|| cur.as_ref().map(|c| c.expression.clone())).unwrap_or_default(),
        reference: reference.or_else(|| cur.as_ref().map(|c| c.reference.clone())).unwrap_or_default(),
        triggers: triggers.unwrap_or_else(|| cur.as_ref().is_some_and(|c| c.triggers_process)),
    };
    match state.mode {
        Mode::Expression if state.expression.is_empty() => {
            return Err("mode `expression` with no expression to evaluate".to_string())
        }
        Mode::Reference if state.reference.is_empty() => {
            return Err("mode `reference` with no reference to follow".to_string())
        }
        _ => {}
    }
    Ok((value, Some(state)))
}

/// A global as `{value, type}` — the shape in the `.gfi` and the doc.
pub fn global_to_json(v: &goofi_core::globals::GlobalValue) -> serde_json::Value {
    serde_json::to_value(v).expect("a scalar enum serializes")
}

/// The inverse of [`global_to_json`]; `None` if malformed. Type-directed on purpose: a fraction
/// offered to an `int` global rounds instead of failing.
pub fn global_from_json(entry: &serde_json::Value) -> Option<goofi_core::globals::GlobalValue> {
    use goofi_core::globals::GlobalValue;
    serde_json::from_value(entry.clone()).ok().or_else(|| match entry.get("type")?.as_str()? {
        "int" => Some(GlobalValue::Int(entry.get("value")?.as_f64()?.round() as i64)),
        _ => None,
    })
}

/// The active source of a param's value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Constant,
    Expression,
    Reference,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Constant => "constant",
            Mode::Expression => "expression",
            Mode::Reference => "reference",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "constant" => Some(Mode::Constant),
            "expression" => Some(Mode::Expression),
            "reference" => Some(Mode::Reference),
            _ => None,
        }
    }
}

/// The one variable a reference rewrites to: the bare-variable source the runtime copies without
/// an evaluator.
const REF_VAR: &str = "ref";

/// A param's source record, graph-side: the AUTHORED expression and reference — both retained
/// whatever the mode, since a toggle is never destructive — and everything below `triggers_process`
/// is DERIVED from the active one, re-derived whenever it or a name the graph resolves changes.
struct ParamSource {
    mode: Mode,
    expression: String,
    reference: String,
    triggers_process: bool,
    /// Compiled from [`Self::rewritten`], never from [`Self::source`]: the evaluator is handed
    /// variables, not names. `None` when the compile failed, or there is no evaluator.
    id: Option<goofi_node::BindingId>,
    /// Derived: `source` with every reference replaced by a generated variable.
    rewritten: String,
    /// Derived: one entry per variable `rewritten` names, resolved against the graph.
    vars: Vec<BoundVar>,
    /// The rewrite's variable list BEFORE resolution — a variable that failed to resolve no longer
    /// says what it was looking for, and that is what a new node or global has to re-resolve.
    terms: Vec<expr_rewrite::VarRef>,
    /// Why the GRAPH could not bind this source. Written by `set_expression` and nowhere else — it
    /// describes the SOURCE, so it outlives any one instance.
    bind_error: Option<String>,
}

impl ParamSource {
    /// Whether the graph SHIPS this source. A constant mode and a source the graph could not bind
    /// both leave the param on its literal, and the node is TOLD so.
    fn live(&self) -> bool {
        self.mode != Mode::Constant && self.bind_error.is_none()
    }

    fn state(&self) -> SourceState {
        SourceState {
            mode: self.mode,
            expression: self.expression.clone(),
            reference: self.reference.clone(),
            triggers: self.triggers_process,
        }
    }
}

/// [`ParamSource`] projected for the bridge and the `.gfi`.
pub struct SourceInfo {
    pub mode: Mode,
    pub expression: String,
    pub reference: String,
    pub triggers_process: bool,
    pub error: Option<String>,
}

/// The authoritative graph + scheduler.
/// One end of a wire, resolved once: the slot's `&'static` name, its dtype, and whether it takes
/// many wires. A leaf reads it off its manifest; a port answers for itself.
#[derive(Clone, Copy)]
struct SlotFace {
    name: &'static str,
    kind: goofi_core::SlotType,
    multi: bool,
}

/// What a source address really produces. A port relays rather than runs, so its stream is
/// whatever is wired into it — and "nothing yet" is an answer, not a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    /// A real leaf output slot — the only thing the transport can subscribe to.
    At(Uid, &'static str),
    /// The port the walk stopped at, because nothing feeds it yet.
    Open(Uid),
}

pub struct Graph {
    nodes: IndexMap<Uid, NodeEntry>,
    links: Vec<Link>,
    next_uid: u64,
    /// The panel arrangement, held FLAT — the fifth doc root. Every mutation is an ordinary
    /// command, so the layout has exactly one projection, as nodes and links do.
    arrangement: layout::Layout,
    /// Why a stored arrangement was refused — so a fallback to the default is stated rather than
    /// silent. Cleared by every load.
    arrangement_warning: Option<String>,
    /// Where a client is LOOKING. Not a doc root — converging it would drag peers and dirty the
    /// patch on mere navigation — but persistence is the other axis, so it rides the `.gfi`.
    viewpoint: serde_json::Value,
    /// Types that exist on disk but cannot load here → why. Greyed in the palette, so a node
    /// needing an uninstalled dependency explains itself instead of silently not existing.
    unavailable: std::collections::BTreeMap<String, String>,
    /// The types that came from the open patch's workspace — the one thing about a type that only
    /// the scan can know. Re-derived wholesale by each scan.
    patch_types: std::collections::HashSet<String>,
    /// One clock across every node thread rather than one per birth: `NodeCtx::now` is
    /// seconds-since-patch-start.
    start: Instant,
    /// `None` ⇒ bindings are stored and round-trip but never evaluate; the literal stands.
    evaluator: Option<Arc<dyn goofi_node::ExprEvaluator>>,
    /// uid → parent scope (absent = ROOT). The ONE source of truth for parentage and membership.
    scope_of: HashMap<Uid, Option<Uid>>,
    /// Patch-scoped globals, system ones seeded and re-asserted by every `clear`/load.
    globals: goofi_core::globals::GlobalStore,
    /// The registered engines, signal first, reached only through the trait. Registered at the
    /// composition root, so an empty set is a bare MODEL — it serializes, and runs nothing.
    engines: Vec<Box<dyn Engine>>,
    /// Shared with every engine and the drain worker: a report notifies, the worker parks on it.
    waker: Arc<DrainWaker>,
    /// What service names are scoped by. Random, not the bridge's instance id: a service name has
    /// to be unique on the MACHINE, across this process's own graphs and every stale record.
    instance: String,
    /// Every uid's birth generation, bumped on EVERY birth and never reset — it keeps a reborn
    /// node's service names clear of its predecessor's, whose teardown does not block. Survives
    /// `clear()` and `load_doc`; never enters the archive.
    generations: HashMap<Uid, u64>,
    /// Params whose options a node has re-enumerated since anyone looked. Options are the one
    /// thing a node reports that the doc has no field for, so the worker must be TOLD to echo them.
    refreshed: Vec<(Uid, ParamKey)>,
    /// What the current batch changed and [`Self::settle`] has not yet delivered.
    touched: Vec<Touched>,
    /// Raised while a multi-step batch is mid-flight, so the drain-side settle cannot deliver its
    /// intermediates. On the GRAPH, not a thread-local: the drain is another thread.
    open_batches: u32,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Graph {
    /// A node's transport is owned by its own thread and releases its segments when it DROPS, so
    /// raising the halt flags and returning leaves every one allocated if the process exits first.
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// `Arc::make_mut` clones only while a reader holds the previous snapshot, so a caller that took
/// one keeps a consistent version and an unshared record is edited in place.
fn edit_params(leaf: &mut Leaf, edit: impl FnOnce(&mut ParamGroups)) {
    edit(Arc::make_mut(&mut leaf.params));
}

/// A global as the [`Param`] an expression variable carries. The bounds are a carrier's, not a
/// control's: the evaluator coerces the RESULT to the target param's own type and range.
fn global_as_param(value: &goofi_core::globals::GlobalValue) -> Param {
    use goofi_core::globals::GlobalValue as G;
    match value {
        G::Float(v) => Param::float(*v, f64::NEG_INFINITY, f64::INFINITY),
        G::Int(v) => Param::int(*v, i64::MIN, i64::MAX),
        G::Bool(v) => Param::boolean(*v),
        G::Str(v) => Param::str_free(v.clone()),
    }
}

/// A fresh service-name scope for one graph — the resolver input the graph owns and mints.
/// Random rather than a pid, which is reused, and a recycled scope would JOIN stale services.
fn mint_instance() -> String {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).expect("the OS random source");
    format!("{:016x}", u64::from_be_bytes(bytes))
}

/// The lowest free doorbell id in the expression range, or `None` when a node has spent all 64.
fn next_event_id(taken: &[EventId]) -> Option<EventId> {
    (65..=128).find(|id| !taken.contains(id))
}

impl Graph {
    pub fn new() -> Graph {
        let waker = Arc::new(DrainWaker::default());
        let start = Instant::now();
        Graph {
            engines: Vec::new(),
            waker,
            nodes: IndexMap::new(),
            links: Vec::new(),
            next_uid: 1,
            unavailable: std::collections::BTreeMap::new(),
            patch_types: std::collections::HashSet::new(),
            arrangement: layout::Layout::default(),
            arrangement_warning: None,
            viewpoint: serde_json::Value::Null,
            start,
            evaluator: None,
            scope_of: HashMap::new(),
            globals: goofi_core::globals::GlobalStore::new(),
            instance: mint_instance(),
            generations: HashMap::new(),
            refreshed: Vec::new(),
            touched: Vec::new(),
            open_batches: 0,
        }
    }

    /// Stop every node and WAIT for each to release its shared memory. The waiting is why this is
    /// not what `clear()` does: only a process about to EXIT has no "a moment later".
    pub fn shutdown(&mut self) {
        for e in self.engines_mut() {
            e.shutdown();
        }
        self.nodes.clear();
        self.touched.clear();
    }

    /// The authoritative globals store — `entries()` serves the CRDT mirror and the `.gfi`, and an
    /// expression binding resolves through `get`.
    pub fn globals(&self) -> &goofi_core::globals::GlobalStore {
        &self.globals
    }

    /// Apply one global change (`None` = remove; a system delete is refused; a NEW global lands at
    /// ordered position `at` — a delete/rename undo re-adds at the original slot). Every binding
    /// that READS this global is re-resolved and re-sent — a global's value is shipped inline.
    pub fn apply_global_change(
        &mut self,
        name: &str,
        value: Option<goofi_core::globals::GlobalValue>,
        at: Option<usize>,
    ) -> Result<(), String> {
        self.globals.apply_change(name, value, at)?;
        self.invalidate_bindings_reading(name);
        Ok(())
    }

    /// Re-resolve every expression that names a boundary PORT or a sub-patch FACADE. A leaf's
    /// stream is fixed by its manifest, but a port relays and a facade exposes its ports — so
    /// anything that moves a wire moves what `nd()` reads there. Free when the patch has neither.
    fn rebind_ports(&mut self) {
        let names: Vec<String> = self
            .nodes
            .iter()
            .filter(|(_, e)| !matches!(e.kind, Kind::Leaf(_)))
            .map(|(_, e)| e.name.clone())
            .collect();
        for name in names {
            self.rebind_naming(&name);
        }
    }

    /// Re-resolve and re-send every expression binding that reads global `name`, so its new value
    /// reaches the nodes reading it (only those bindings pay). Shared by the global mutators.
    fn invalidate_bindings_reading(&mut self, name: &str) {
        let reading = self.bindings_where(|b| {
            b.terms.iter().any(|t| matches!(t, expr_rewrite::VarRef::Global { key, .. } if key == name))
        });
        self.rebind(&reading);
    }

    /// Re-resolve and re-send every binding whose source SPELLS the display name `name` — in either
    /// position, since a port's name is read as a node's and as its facade's slot label. §5.3's
    /// "renamed, added, removed or restarted", stated once.
    fn rebind_naming(&mut self, name: &str) {
        let naming = self.bindings_where(|b| {
            b.terms.iter().any(|t| match t {
                expr_rewrite::VarRef::Node { name: n, slot, .. } => {
                    n == name || slot.as_deref() == Some(name)
                }
                expr_rewrite::VarRef::NodeParam { name: n, .. } => n == name,
                _ => false,
            })
        });
        self.rebind(&naming);
    }

    /// Re-resolve every binding that reads param `key` of `target` — through `nd('name').params`
    /// or the target's own `me.params` — so an authored edit reaches its readers. Evaluation
    /// results never come through here, so a chain of driven params cannot cascade.
    fn invalidate_bindings_reading_param(&mut self, target: Uid, key: &ParamKey) {
        let Some(target_name) = self.name(target).map(str::to_string) else { return };
        let reading: Vec<(Uid, ParamKey)> = self
            .leaves()
            .flat_map(|(uid, e)| e.bindings.iter().map(move |(k, b)| (uid, k.clone(), b)))
            .filter(|(consumer, _, b)| {
                b.terms.iter().any(|t| match t {
                    expr_rewrite::VarRef::NodeParam { name, group, param, .. } => {
                        *name == target_name && *group == key.group && *param == key.name
                    }
                    expr_rewrite::VarRef::MeParam { group, param, .. } => {
                        *consumer == target && *group == key.group && *param == key.name
                    }
                    _ => false,
                })
            })
            .map(|(uid, k, _)| (uid, k))
            .collect();
        self.rebind(&reading);
    }

    /// Every binding matching a predicate, as `(node, param)` — the addressing `rebind` takes.
    fn bindings_where(&self, want: impl Fn(&ParamSource) -> bool) -> Vec<(Uid, ParamKey)> {
        self.leaves()
            .flat_map(|(uid, e)| e.bindings.iter().map(move |(k, b)| (uid, k.clone(), b)))
            .filter(|(_, _, b)| want(b))
            .map(|(uid, key, _)| (uid, key))
            .collect()
    }

    /// Re-run `set_source` on each of these records from its AUTHORED texts — the one operation
    /// that re-derives the rewrite, the variables and the handle, and records delivery.
    fn rebind(&mut self, bindings: &[(Uid, ParamKey)]) {
        for (uid, key) in bindings {
            let Some(state) = self.source_state(*uid, key) else { continue };
            let _ = self.set_source(*uid, &key.group, &key.name, state);
        }
    }

    fn source_state(&self, uid: Uid, key: &ParamKey) -> Option<SourceState> {
        self.leaf(uid).and_then(|e| e.bindings.get(key)).map(ParamSource::state)
    }

    /// A param's source record as a command carries it, `None` when it holds none.
    pub fn source_state_of(&self, uid: Uid, group: &str, name: &str) -> Option<SourceState> {
        self.source_state(uid, &ParamKey::new(group, name))
    }

    /// Inject the param-expression evaluator (pyo3, from goofi-python). Wired by the CLI at
    /// startup; without it, expression bindings are stored but not evaluated.
    pub fn set_evaluator(&mut self, evaluator: Arc<dyn goofi_node::ExprEvaluator>) {
        self.evaluator = Some(evaluator.clone());
        for e in self.engines_mut() {
            e.set_evaluator(evaluator.clone());
        }
    }

    /// Register an engine, signal first by convention. Its library joins the merged view, and its
    /// nodes ride every generic path — the trait is the whole integration.
    pub fn register_engine(&mut self, engine: Box<dyn Engine>) {
        self.engines.push(engine);
    }

    /// What every engine and the drain worker share: a node report notifies it, the worker parks
    /// on it between paced duties — the alternative to a poll-to-discover.
    pub fn drain_waker(&self) -> Arc<DrainWaker> {
        self.waker.clone()
    }

    fn engines(&self) -> impl Iterator<Item = &dyn Engine> {
        self.engines.iter().map(|e| e.as_ref() as &dyn Engine)
    }

    fn engines_mut(&mut self) -> impl Iterator<Item = &mut dyn Engine> {
        self.engines.iter_mut().map(|e| e.as_mut() as &mut dyn Engine)
    }

    /// One registered engine, by id — how the composition root reaches a concrete door through
    /// [`Engine::as_any_mut`].
    pub fn engine_mut(&mut self, id: &str) -> Option<&mut dyn Engine> {
        self.engines.iter_mut().map(|e| e.as_mut() as &mut dyn Engine).find(|e| e.id() == id)
    }

    /// The one merged view the palette reads: every engine's library, in registration order.
    pub fn library_manifests(&self) -> Vec<&'static NodeManifest> {
        self.engines().flat_map(|e| e.library()).map(|l| l.manifest).collect()
    }

    /// The manifest `type_name` resolves to, from whichever engine's library advertises it.
    pub fn type_manifest(&self, type_name: &str) -> Option<&'static NodeManifest> {
        self.library_entry(type_name).map(|(_, l)| l.manifest)
    }

    /// The library entry `type_name` resolves to, and the id of the engine that advertised it —
    /// which IS the engine the type belongs to. Two libraries claiming one name resolve to the
    /// FIRST advertiser, signal first — a decided outcome, not an accident.
    fn library_entry(&self, type_name: &str) -> Option<(&'static str, LibraryEntry)> {
        self.engines().find_map(|e| {
            e.library()
                .into_iter()
                .find(|l| l.manifest.type_name == type_name)
                .map(|l| (e.id(), l))
        })
    }

    /// The universal param group the owning engine adds to every one of its nodes — the palette
    /// and the default-expression seeding read declarations through this one door.
    pub fn universal_decls_of(&self, type_name: &str) -> Vec<ParamDecl> {
        let Some((id, entry)) = self.library_entry(type_name) else { return Vec::new() };
        self.engines()
            .find(|e| e.id() == id)
            .map(|e| e.universal_decls(entry.manifest))
            .unwrap_or_default()
    }

    /// Forget the unavailable row for a type that now resolves — a registration's caller clears
    /// it, or the greyed row would give one name two palette rows.
    pub fn forget_unavailable(&mut self, type_name: &str) -> bool {
        self.unavailable.remove(type_name).is_some()
    }

    /// Whether a type name resolves to either the compile-time catalog or a
    /// runtime-registered type.
    fn known_type(&self, type_name: &str) -> bool {
        self.library_entry(type_name).is_some()
    }

    /// The ONE phrasing for a rejected type, shared by `build_node` and the load gate. An
    /// unavailable type names its missing dependency; anything else reads as the typo it is.
    fn reject_type(&self, type_name: &str) -> String {
        match self.unavailable.get(type_name) {
            Some(reason) => format!("node type `{type_name}` is unavailable: {reason}"),
            None => format!("unknown node type `{type_name}`"),
        }
    }

    /// `creating` / `setup` / `ready` / `error`. Only `creating` is the graph's own — a node whose
    /// thread has not reported in yet. `error` means there is NO instance running.
    pub fn node_stage(&self, uid: Uid) -> &'static str {
        // A facade and a port never run, so they reach no stage — `ready` is what "nothing is
        // starting up here" means for something that is simply present.
        let Some(entry) = self.leaf(uid) else {
            return match self.nodes.contains_key(&uid) {
                true => "ready",
                false => "error",
            };
        };
        // A `process()` raise is deliberately NOT folded in: the stage says whether the node has an
        // instance behind it, and what its last run did is the ERROR, which rides its own field.
        if entry.health.setup_error.is_some() {
            return "error";
        }
        entry.health.stage
    }

    /// The node's current measured update frequency (Hz), as it last reported it. `None` until it
    /// has been measured (≥2 emits).
    pub fn node_ufreq(&self, uid: Uid) -> Option<f64> {
        self.leaf(uid).and_then(|e| e.health.ufreq)
    }

    /// Which node INSTANCE this uid holds: bumped on every birth, so a report from the node born at
    /// a uid is distinguishable from its predecessor's last one.
    pub fn node_generation(&self, uid: Uid) -> u64 {
        self.generation(uid)
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

    /// Record a type that could not be loaded, and why. Refused while any engine's library still
    /// advertises the name — the scanner displaces a stale runtime type BEFORE recording it.
    pub fn register_unavailable(&mut self, type_name: String, reason: String) -> bool {
        if self.library_entry(&type_name).is_some() {
            return false;
        }
        self.unavailable.insert(type_name, reason);
        true
    }

    /// The unloadable types, `(type_name, reason)`, sorted by name.
    pub fn unavailable_types(&self) -> impl Iterator<Item = (&str, &str)> {
        self.unavailable.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Declare which runtime types came from the open patch's own workspace — the palette's
    /// provenance badge. Written WHOLESALE, because only the scan knows the answer.
    pub fn set_patch_types(&mut self, names: std::collections::HashSet<String>) {
        self.patch_types = names;
    }

    /// Whether `type_name` came from the open patch (see [`Graph::set_patch_types`]). Everything
    /// else — built-ins and the shipped node directory alike — reads as shipped.
    pub fn is_patch_type(&self, type_name: &str) -> bool {
        self.patch_types.contains(type_name)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The RUNNING node at `uid`, or `None` for a facade, a port, or nothing at all — the one seam
    /// every reader of a leaf-only field goes through.
    fn leaf(&self, uid: Uid) -> Option<&Leaf> {
        self.nodes.get(&uid)?.leaf()
    }

    fn leaf_mut(&mut self, uid: Uid) -> Option<&mut Leaf> {
        self.nodes.get_mut(&uid)?.leaf_mut()
    }

    /// Every running node, in insertion order.
    fn leaves(&self) -> impl Iterator<Item = (Uid, &Leaf)> {
        self.nodes.iter().filter_map(|(u, e)| e.leaf().map(|l| (*u, l)))
    }

    pub fn contains(&self, uid: Uid) -> bool {
        self.leaf(uid).is_some()
    }

    /// Is `uid` a live endpoint a wire may name — a leaf or a boundary port? A facade is not: an
    /// address naming one is folded onto its port before any link is stored.
    pub fn wirable(&self, uid: Uid) -> bool {
        self.contains(uid) || self.stub(uid).is_some()
    }

    /// Node uids in insertion order.
    pub fn node_uids(&self) -> Vec<Uid> {
        self.leaves().map(|(u, _)| u).collect()
    }

    /// Is there a node of ANY kind at `uid`?
    pub fn exists(&self, uid: Uid) -> bool {
        self.nodes.contains_key(&uid)
    }

    /// Every uid in the patch — leaves, facades and ports alike.
    pub fn all_uids(&self) -> Vec<Uid> {
        self.nodes.keys().copied().collect()
    }

    pub fn type_name(&self, uid: Uid) -> Option<&'static str> {
        self.leaf(uid).map(|e| e.manifest.type_name)
    }

    pub fn manifest(&self, uid: Uid) -> Option<&'static NodeManifest> {
        self.leaf(uid).map(|e| e.manifest)
    }

    /// The tier `uid`'s instance runs on — a leaf alone wears one.
    pub fn node_tier(&self, uid: Uid) -> Option<Isolation> {
        self.leaf(uid).map(|e| e.isolation.get())
    }

    /// A TYPE's tier, from whichever engine's library advertises it.
    pub fn type_tier(&self, type_name: &str) -> Option<Isolation> {
        self.library_entry(type_name).map(|(_, l)| l.isolation.get())
    }

    /// Derived fresh on read, so a binding that recovers on a node which never runs again still
    /// clears. Initialization failure wins, then a process error, then the smallest errored key.
    pub fn last_error(&self, uid: Uid) -> Option<&str> {
        if let Some(leaf) = self.leaf(uid) {
            return entry_error(leaf);
        }
        // A facade runs nothing, so its health is its members': the first errored descendant, at any
        // depth. Derived HERE, so a human's badge and an agent's read are one answer. The walked
        // set doubles as the cycle guard a hand-edited `.gfi` needs.
        let mut walk = vec![uid];
        let mut at = 0;
        while at < walk.len() {
            let u = walk[at];
            at += 1;
            if let Some(err) = self.leaf(u).and_then(entry_error) {
                return Some(err);
            }
            for m in self.scope_members(u) {
                if !walk.contains(&m) {
                    walk.push(m);
                }
            }
        }
        None
    }

    /// How long this node's CURRENT error has been standing, or `None` when it is healthy. The
    /// clock restarts when the message changes and at every rebirth, so it never outlives an instance.
    pub fn error_age(&self, uid: Uid) -> Option<Duration> {
        let (_, since) = self.leaf(uid)?.health.error_since.as_ref()?;
        Some(since.elapsed())
    }

    pub(crate) fn mint(&mut self) -> Uid {
        let u = Uid(self.next_uid);
        self.next_uid += 1;
        u
    }

    /// The uid a loaded record restores at — the one the archive named, unless it is unreadable or
    /// already `claimed`. Restoring rather than reminting is what makes a load restore IDENTITY.
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
    /// The record a fresh instance of `type_name` starts from — the owning engine's own
    /// normalization, resolved without constructing the node. Also what the palette renders.
    pub fn default_params_of(&self, type_name: &str, supplied: Option<ParamGroups>) -> Result<ParamGroups, String> {
        let (id, _) = self.library_entry(type_name).ok_or_else(|| self.reject_type(type_name))?;
        self.engines()
            .find(|e| e.id() == id)
            .expect("the entry named it")
            .normalize_params(type_name, supplied)
    }

    /// Instantiate a node by type name (compile-time catalog or a runtime-registered type).
    /// `params` defaults to the type's defaults.
    pub fn add_node(&mut self, type_name: &str, params: Option<ParamGroups>) -> Result<Uid, String> {
        self.create_node(type_name, None, "", params, None)
    }

    /// Create the node `type_name` names — a leaf, a sub-patch facade or a boundary port — at
    /// `uid` when one is given and a fresh mint otherwise. The last two run nothing, so they carry
    /// no manifest, no thread and no params; what a port carries instead is the `scope` it is a
    /// port OF, which is what makes it one. An empty or taken `name` is minted fresh.
    pub fn create_node(
        &mut self,
        type_name: &str,
        uid: Option<Uid>,
        name: &str,
        params: Option<ParamGroups>,
        scope: Option<Uid>,
    ) -> Result<Uid, String> {
        if let Some(u) = uid.filter(|u| self.nodes.contains_key(u)) {
            return Err(format!("node add: uid {} already in use", u.to_hex()));
        }
        if let Some(s) = scope.filter(|s| !self.is_facade(*s)) {
            return Err(format!("node add: no such scope {s}"));
        }
        let (kind, base) = match subpatch::boundary_type(type_name) {
            Some((dir, dtype)) => {
                if scope.is_none() {
                    return Err("node add: a boundary port needs a scope — it is a port OF a sub-patch".into());
                }
                (Kind::Port(subpatch::Port { dir, dtype }), dir.name().to_string())
            }
            None if type_name == subpatch::SCOPE_TYPE => (Kind::Facade, "subpatch".to_string()),
            None => {
                // A leaf is the only kind with a manifest, so it is the only one that can seed the
                // default expressions its type declares.
                let seed = params.is_none();
                let (engine, entry) =
                    self.library_entry(type_name).ok_or_else(|| self.reject_type(type_name))?;
                let params = self.default_params_of(type_name, params)?;
                let uid = self.claim(uid);
                let born = self.pick_name(name, &entry.manifest.type_name.to_lowercase(), None);
                self.insert_node_at(uid, born.clone(), engine, entry, params);
                let manifest = entry.manifest;
                if seed {
                    self.seed_default_expressions(uid, manifest);
                }
                self.set_member_scope(uid, scope);
                // A name that meant nothing a moment ago now names a producer (§5.3). This also
                // covers undo-of-delete, which is how a binding survives a delete and a restore.
                self.rebind_naming(&born);
                return Ok(uid);
            }
        };
        let uid = self.claim(uid);
        let born = self.pick_name(name, &base, None);
        self.nodes.insert(
            uid,
            NodeEntry { kind, name: born.clone(), pos: [0.0, 0.0], viewers: serde_json::json!({}) },
        );
        self.set_member_scope(uid, scope);
        self.rebind_naming(&born);
        Ok(uid)
    }

    /// The uid a create will use, with the mint counter kept past it so a restored uid is never
    /// handed out a second time.
    fn claim(&mut self, uid: Option<Uid>) -> Uid {
        let uid = uid.unwrap_or_else(|| self.mint());
        self.next_uid = self.next_uid.max(uid.0 + 1);
        uid
    }

    /// The display name a create lands on: the one asked for, or a fresh `base<N>` when it is
    /// empty, already worn, or not a legal name. A CREATE degrades where a rename refuses, because
    /// this is also the restore path — a hand-edited archive must cost one name, not the patch.
    fn pick_name(&self, want: &str, base: &str, except: Option<Uid>) -> String {
        match self.name_taken(want, except) || !goofi_core::globals::is_valid_name(want) {
            true => self.fresh_name(base),
            false => want.to_string(),
        }
    }

    fn seed_default_expressions(&mut self, uid: Uid, manifest: &'static NodeManifest) {
        if self.evaluator.is_none() {
            return;
        }
        // The manifest's own declarations win over the engine's universal group, as they do on
        // the value side.
        let declared = manifest.params.iter().map(|d| (d.group, d.name, d.expression));
        let universal = self
            .universal_decls_of(manifest.type_name)
            .into_iter()
            .filter(|d| !manifest.params.iter().any(|o| o.group == d.group && o.name == d.name))
            .map(|d| (d.group, d.name, d.expression))
            .collect::<Vec<_>>();
        for (group, name, expression) in declared.chain(universal) {
            if let Some(e) = expression {
                let enabled = matches!(e.mode, ExprMode::On);
                let state = SourceState {
                    mode: if enabled { Mode::Expression } else { Mode::Constant },
                    expression: e.source.to_string(),
                    reference: String::new(),
                    triggers: e.trigger,
                };
                let _ = self.set_source(uid, group, name, state);
            }
        }
    }

    /// Where a node gets its runtime instance. This IS the birth §3.1 counts, whichever door it
    /// came through — a fresh add, a restart, an undo of a delete, a load.
    fn insert_node_at(
        &mut self,
        uid: Uid,
        name: String,
        engine: &'static str,
        entry: LibraryEntry,
        params: ParamGroups,
    ) {
        let generation = self.bump_generation(uid);
        let type_name = entry.manifest.type_name;
        let boot_error = self
            .engine_mut(engine)
            .expect("the library entry named it")
            .insert(uid, type_name, generation, &params);
        self.nodes.insert(
            uid,
            NodeEntry {
                kind: Kind::Leaf(Box::new(Leaf {
                    manifest: entry.manifest,
                    isolation: entry.isolation,
                    engine,
                    params: Arc::new(params),
                    bindings: HashMap::new(),
                    health: Health::born(boot_error),
                })),
                name,
                pos: [0.0, 0.0],
                viewers: serde_json::json!({}),
            },
        );
    }

    /// Every display name in the patch with the uid wearing it — leaves, sub-patch facades and
    /// boundary ports share ONE namespace, because `nd('name')` addresses any of them.
    fn named(&self) -> impl Iterator<Item = (Uid, &str)> {
        self.nodes
            .iter()
            .map(|(u, e)| (*u, e.name.as_str()))
    }

    fn name_in_use(&self, name: &str) -> bool {
        self.named().any(|(_, n)| n == name)
    }

    /// Is `name` already worn by something other than `except` — `None` when nothing is exempt, as
    /// for a node not yet born? `AddNode` and `EditNode` both tolerate a collision as a no-op, so
    /// the user-facing error is raised at the RPC boundary.
    pub fn name_taken(&self, name: &str, except: Option<Uid>) -> bool {
        self.named().any(|(u, n)| Some(u) != except && n == name)
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

    /// Display name of anything a uid can name — one map, so one lookup for all three kinds.
    pub fn name(&self, uid: Uid) -> Option<&str> {
        self.nodes.get(&uid).map(|e| e.name.as_str())
    }

    /// Where anything a uid can name sits on the canvas.
    pub fn pos(&self, uid: Uid) -> Option<[f64; 2]> {
        self.nodes.get(&uid).map(|e| e.pos)
    }

    /// The boundary port a uid names, with the scope holding it. Ports are few and scopes fewer, so
    /// this scans rather than keeping a second index beside `scopes`.
    pub fn stub(&self, uid: Uid) -> Option<(Uid, subpatch::Port)> {
        match self.nodes.get(&uid)?.kind {
            Kind::Port(p) => Some((self.scope_of.get(&uid).copied().flatten()?, p)),
            _ => None,
        }
    }

    /// Is `uid` a sub-patch facade?
    pub fn is_facade(&self, uid: Uid) -> bool {
        matches!(self.nodes.get(&uid).map(|e| &e.kind), Some(Kind::Facade))
    }

    /// A scope's boundary ports, in the map's insertion order — which is the order they were
    /// authored in, and the order a facade lists its slots.
    pub fn ports_of(&self, scope: Uid) -> Vec<Uid> {
        self.nodes
            .iter()
            .filter(|(u, e)| matches!(e.kind, Kind::Port(_)) && self.scope_of(**u) == Some(scope))
            .map(|(u, _)| *u)
            .collect()
    }

    /// The physical `(node, slot)` a boundary port's data lives on, or `None` while nothing is
    /// behind it. One call into [`Graph::stream`], which is the walk a cable follows too.
    pub fn stub_stream(&self, port: Uid) -> Option<(Uid, &'static str)> {
        match self.stream(port, subpatch::BOUNDARY_SLOT)? {
            Stream::At(leaf, slot) => Some((leaf, slot)),
            Stream::Open(_) => None,
        }
    }

    /// A node's params as of now. An owned snapshot rather than a borrow: cloning the `Arc` is
    /// cheap, and a `&` would borrow the whole graph for as long as the caller held it.
    pub fn params(&self, uid: Uid) -> Option<Arc<ParamGroups>> {
        self.leaf(uid).map(|e| e.params.clone())
    }

    /// A param's re-enumerated options where the instance has answered a refresh — the overlay a
    /// projection reads over the record's declared options. Never persisted; dies with the instance.
    pub fn refreshed_options(&self, uid: Uid, group: &str, name: &str) -> Option<&[String]> {
        self.leaf(uid)?.health.options.get(&ParamKey::new(group, name)).map(Vec::as_slice)
    }

    /// Rename a node. Every `nd('old')` in the patch follows to `nd('new')`, and the referrer uids
    /// come back so the bridge can rebroadcast them. The rewrite happens only on success.
    pub fn rename_node(&mut self, uid: Uid, name: &str) -> Result<Vec<Uid>, String> {
        if !goofi_core::globals::is_valid_name(name) {
            return Err(format!("`{name}` is not a legal name: {NAME_RULE}"));
        }
        if self.name_in_use(name) {
            return Err(format!("display name `{name}` already in use"));
        }
        // A facade, a boundary port and a leaf all wear a name in the ONE namespace `nd()` reads,
        // and now in one map — so the rename is one write and the rewrite below is shared.
        let e = self.nodes.get_mut(&uid).ok_or_else(|| format!("no such node {uid}"))?;
        let old_name = std::mem::replace(&mut e.name, name.to_string());
        // `name_in_use` guarantees `name != old_name`, so the rename genuinely moved the
        // display name — propagate it into every expression that referenced it.
        let touched = self.rewrite_nd_refs_for_rename(uid, &old_name, name);
        // …and re-resolve the ones ALREADY written against the new name: such a binding has no
        // `nd('<old>')` for the rewrite to follow, and this rename is what makes it resolvable.
        self.rebind_naming(name);
        Ok(touched)
    }

    /// Rewrite `nd('old')` -> `nd('new')` across every param expression, re-binding each changed
    /// source. Returns the distinct referrer uids whose source changed.
    fn rewrite_nd_refs_for_rename(&mut self, uid: Uid, old: &str, new: &str) -> Vec<Uid> {
        // A port's name is ALSO a slot label — on the facade that holds it, and nowhere else — so
        // the one rename reaches an expression in both positions through the one rewrite.
        let facade = self.stub(uid).and_then(|(scope, _)| self.name(scope)).map(str::to_string);
        let rename = |n: &str, slot: Option<&str>| {
            (
                (n == old).then(|| new.to_string()),
                (slot == Some(old) && Some(n) == facade.as_deref()).then(|| new.to_string()),
            )
        };
        let mut edits: Vec<(Uid, ParamKey, SourceState)> = Vec::new();
        for (ruid, entry) in self.leaves() {
            for (key, b) in &entry.bindings {
                let expression = expr_rewrite::rename_refs(&b.expression, rename);
                let reference = expr_rewrite::rename_reference(&b.reference, rename);
                if expression.is_some() || reference.is_some() {
                    let mut state = b.state();
                    state.expression = expression.unwrap_or(state.expression);
                    state.reference = reference.unwrap_or(state.reference);
                    edits.push((ruid, key.clone(), state));
                }
            }
        }
        let mut referrers: Vec<Uid> = Vec::new();
        for (ruid, key, state) in edits {
            if self.set_source(ruid, &key.group, &key.name, state).is_ok() && !referrers.contains(&ruid) {
                referrers.push(ruid);
            }
        }
        // Expressions live only on the live flat nodes now (no def templates) — the loop above has
        // already followed the rename into every one.
        referrers
    }

    pub fn set_node_pos(&mut self, uid: Uid, pos: [f64; 2]) -> Result<(), String> {
        let e = self.nodes.get_mut(&uid).ok_or_else(|| format!("no such node {uid}"))?;
        e.pos = pos;
        Ok(())
    }

    /// Replace a node's opaque viewer view-state blob (persisted to `.gfi`, echoed in node
    /// info). The backend never interprets it — it is the editor's per-slot kind/settings.
    pub fn set_node_viewers(&mut self, uid: Uid, viewers: serde_json::Value) -> Result<(), String> {
        let e = self.nodes.get_mut(&uid).ok_or_else(|| format!("no such node {uid}"))?;
        e.viewers = viewers;
        Ok(())
    }

    /// The viewer view-state blob of anything a uid can name (empty object if never set).
    pub fn viewers(&self, uid: Uid) -> Option<&serde_json::Value> {
        self.nodes.get(&uid).map(|e| &e.viewers)
    }

    // Grouping never touches the flat runtime — the members stay the exact live nodes they were,
    // and only their membership re-tags.

    /// The parent scope of a node/scope (`None` = ROOT). Absent ⇒ ROOT, so a plain flat graph
    /// needs no entries.
    pub fn scope_of(&self, uid: Uid) -> Option<Uid> {
        self.scope_of.get(&uid).copied().flatten()
    }

    /// Everything `scope_of` places inside `scope`, ports included — they are members like any
    /// other node, which is the whole of what a scope holds.
    pub fn scope_members(&self, scope: Uid) -> Vec<Uid> {
        self.nodes.keys().copied().filter(|u| self.scope_of(*u) == Some(scope)).collect()
    }


    /// Chain-resolve a scope's stub port to the single physical inner leaf `(uid, slot)` it exposes,
    /// walking nested scopes; `None` if unwired.
    pub fn resolve_stub(&self, scope: Uid, stub: &str) -> subpatch::StubInner {
        let port = Uid::from_hex(stub).filter(|p| self.stub(*p).is_some_and(|(s, _)| s == scope))?;
        let mut at = port;
        // A hand-edited `.gfi` can persist a cyclic chain; walking it must stop, not recurse.
        let mut seen: Vec<Uid> = Vec::new();
        loop {
            if seen.contains(&at) {
                return None;
            }
            seen.push(at);
            let (node, slot) = self.port_inner(at)?;
            match self.stub(node).is_some() {
                true => at = node,
                false => return Some((node, slot.to_string())),
            }
        }
    }

    /// The output slots of anything a uid names: `(key, label, dtype)`. A leaf's declarations; a
    /// facade's OUT ports, keyed by the port uid so a rename cannot orphan a wire or a panel, and
    /// LABELLED by the port's own name so nothing has to keep a second map; and the single `value`
    /// a port wears — either direction, because a port RELAYS rather than produces, so what it
    /// stands in front of is readable from both sides of the wall.
    pub fn output_slots(&self, uid: Uid) -> Vec<(String, String, goofi_core::SlotType)> {
        if self.is_facade(uid) {
            return self
                .ports_of(uid)
                .into_iter()
                .filter_map(|id| self.stub(id).map(|(_, p)| (id, p)))
                .filter(|(_, p)| p.dir == subpatch::Dir::Out)
                .map(|(id, p)| (id.to_hex(), self.name(id).unwrap_or("").to_string(), p.dtype))
                .collect();
        }
        if let Some((_, st)) = self.stub(uid) {
            let slot = subpatch::BOUNDARY_SLOT.to_string();
            return vec![(slot.clone(), slot, st.dtype)];
        }
        self.nodes
            .get(&uid)
            .and_then(|e| e.leaf())
            .map(|e| {
                e.manifest.outputs.iter().map(|o| (o.name.to_string(), o.name.to_string(), o.kind)).collect()
            })
            .unwrap_or_default()
    }

    /// The input slots of anything a uid names, as [`Graph::output_slots`] gives the output ones:
    /// a leaf's declarations, a facade's IN ports, and a port's single `value` — which it wears on
    /// both sides, because it relays and so consumes and produces the one wire.
    pub fn input_slots(&self, uid: Uid) -> Vec<(String, String, goofi_core::SlotType)> {
        if self.is_facade(uid) {
            return self
                .ports_of(uid)
                .into_iter()
                .filter_map(|id| self.stub(id).map(|(_, p)| (id, p)))
                .filter(|(_, p)| p.dir == subpatch::Dir::In)
                .map(|(id, p)| (id.to_hex(), self.name(id).unwrap_or("").to_string(), p.dtype))
                .collect();
        }
        if let Some((_, st)) = self.stub(uid) {
            let slot = subpatch::BOUNDARY_SLOT.to_string();
            return vec![(slot.clone(), slot, st.dtype)];
        }
        self.leaf(uid)
            .map(|e| {
                e.manifest.inputs.iter().map(|s| (s.name.to_string(), s.name.to_string(), s.kind)).collect()
            })
            .unwrap_or_default()
    }

    /// The type name of anything a uid names — a leaf's, a facade's, or a port's boundary type.
    pub fn node_type(&self, uid: Uid) -> Option<&'static str> {
        if self.is_facade(uid) {
            return Some(subpatch::SCOPE_TYPE);
        }
        if let Some((_, st)) = self.stub(uid) {
            return Some(subpatch::boundary_type_name(st.dir, st.dtype));
        }
        self.leaf(uid).map(|e| e.manifest.type_name)
    }

    /// A facade address, folded one level onto the port it names. Everything else is itself. What
    /// is BEHIND that port is a separate question ([`Graph::stream`]), asked at plan time.
    pub fn normalise(&self, uid: Uid, slot: &str) -> (Uid, String) {
        match self.is_facade(uid) {
            true => match Uid::from_hex(slot) {
                Some(port) if self.stub(port).is_some() => (port, subpatch::BOUNDARY_SLOT.to_string()),
                _ => (uid, slot.to_string()),
            },
            false => (uid, slot.to_string()),
        }
    }

    /// THE resolution: what real leaf slot an address stands for. A port relays, so the walk takes
    /// one hop per boundary, however deep the nesting. `Open` is the port the walk stopped at
    /// because nothing feeds it yet — a legitimate answer, never an error. `None` only when the
    /// address names no output slot at all. One function, so `nd()` and a cable follow one wiring.
    pub fn stream(&self, uid: Uid, slot: &str) -> Option<Stream> {
        let (mut at, mut slot) = self.normalise(uid, slot);
        let mut seen: Vec<Uid> = Vec::new();
        loop {
            if self.leaf(at).is_some() {
                return Some(Stream::At(at, self.resolve_output(at, &slot)?));
            }
            // A hand-edited `.gfi` can persist a cyclic chain; walking it must stop, not recurse.
            if seen.contains(&at) {
                return Some(Stream::Open(at));
            }
            seen.push(at);
            self.stub(at)?;
            let Some((next, next_slot)) = self.stub_feed(at) else {
                return Some(Stream::Open(at));
            };
            (at, slot) = (next, next_slot);
        }
    }

    /// The wire on a port's INSIDE — the member it feeds (an IN port) or drains (an OUT port). An
    /// ordinary link, so this is a lookup rather than a field beside `links` to keep in step.
    pub fn port_inner(&self, port: Uid) -> Option<(Uid, &'static str)> {
        let (_, st) = self.stub(port)?;
        match st.dir {
            subpatch::Dir::In => {
                self.links.iter().find(|l| l.node_out == port).map(|l| (l.node_in, l.slot_in))
            }
            subpatch::Dir::Out => {
                self.links.iter().find(|l| l.node_in == port).map(|l| (l.node_out, l.slot_out))
            }
        }
    }

    /// Every real leaf INPUT an output address reaches, walking forward through any chain of ports.
    /// A port relays, so what a producer really feeds is whatever sits past it — and it may fan out.
    fn sinks(&self, node: Uid, slot: &'static str) -> Vec<(Uid, &'static str)> {
        let (mut out, mut seen) = (Vec::new(), Vec::new());
        let mut stack = vec![(node, slot)];
        while let Some((n, s)) = stack.pop() {
            for l in self.links.iter().filter(|l| l.node_out == n && l.slot_out == s) {
                match self.stub(l.node_in).is_some() {
                    // A hand-edited `.gfi` can persist a cyclic chain; walking it must stop.
                    true if !seen.contains(&l.node_in) => {
                        seen.push(l.node_in);
                        stack.push((l.node_in, l.slot_in));
                    }
                    true => {}
                    false => out.push((l.node_in, l.slot_in)),
                }
            }
        }
        out
    }

    /// One hop back from a port to whatever feeds it — the wire arriving AT it, whichever side of
    /// the wall that is. A port relays, so its direction decides which side is which but never what
    /// the answer is: the stream is simply what is wired in.
    fn stub_feed(&self, port: Uid) -> Option<(Uid, String)> {
        self.links.iter().find(|l| l.node_in == port).map(|l| (l.node_out, l.slot_out.to_string()))
    }

    /// The scope an end of a wire FACES. A leaf faces the scope it lives in; a port relays across a
    /// wall, so it faces its own scope on one side and the parent on the other. Two ends may be
    /// linked exactly when their faces agree — which is one rule for a cable inside a sub-patch, a
    /// cable outside it, and the refusal of an in-port wired the wrong way round.
    fn face(&self, uid: Uid, producer: bool) -> Option<Option<Uid>> {
        if let Some((scope, st)) = self.stub(uid) {
            let inward = st.dir == subpatch::Dir::In;
            return Some(match inward == producer {
                true => Some(scope),
                false => self.scope_of(scope),
            });
        }
        self.nodes.contains_key(&uid).then(|| self.scope_of(uid))
    }

    /// One end of a wire: the `&'static` name a link is keyed by, the dtype the cross-dtype check
    /// needs, and whether it takes many wires. Owned rather than a borrowed decl, because a PORT
    /// has no manifest to borrow one from — its dtype is its own, fixed by its type at birth.
    fn find_output(&self, uid: Uid, slot: &str) -> Option<SlotFace> {
        if let Some((_, st)) = self.stub(uid) {
            return (slot == subpatch::BOUNDARY_SLOT)
                .then_some(SlotFace { name: subpatch::BOUNDARY_SLOT, kind: st.dtype, multi: false });
        }
        self.leaf(uid)?
            .manifest
            .outputs
            .iter()
            .find(|o| o.name == slot)
            .map(|o| SlotFace { name: o.name, kind: o.kind, multi: false })
    }

    /// The consumer end, as [`Graph::find_output`] is the producer end. A port wears the one
    /// `value` slot on both sides: it relays, so it is a consumer and a producer of the same wire.
    fn find_input(&self, uid: Uid, slot: &str) -> Option<SlotFace> {
        if let Some((_, st)) = self.stub(uid) {
            return (slot == subpatch::BOUNDARY_SLOT)
                .then_some(SlotFace { name: subpatch::BOUNDARY_SLOT, kind: st.dtype, multi: false });
        }
        self.leaf(uid)?
            .manifest
            .inputs
            .iter()
            .find(|s| s.name == slot)
            .map(|s| SlotFace { name: s.name, kind: s.kind, multi: s.multi })
    }

    fn output_slot_type(&self, uid: Uid, slot: &str) -> Option<goofi_core::SlotType> {
        self.find_output(uid, slot).map(|o| o.kind)
    }

    fn input_slot_type(&self, uid: Uid, slot: &str) -> Option<goofi_core::SlotType> {
        self.find_input(uid, slot).map(|s| s.kind)
    }

    /// Move a node or scope into `scope` (`None` = ROOT), returning its prior membership — the one
    /// validated re-parent seam. Errors on an unknown uid or a `scope` that is not a live scope.
    pub fn reparent(&mut self, uid: Uid, scope: Option<Uid>) -> Result<Option<Uid>, String> {
        if !self.nodes.contains_key(&uid) {
            return Err(format!("reparent: no such node/scope {uid}"));
        }
        if let Some(s) = scope {
            if !self.is_facade(s) {
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

    /// The member of `member_set` that transitively contains `uid`, or `None` when it lies outside
    /// every member. Lets link classification treat a buried leaf as inside the group.
    fn containing_member(&self, uid: Uid, member_set: &std::collections::HashSet<Uid>) -> Option<Uid> {
        let mut cur = uid;
        loop {
            if member_set.contains(&cur) {
                return Some(cur);
            }
            cur = self.scope_of(cur)?;
        }
    }

    /// The stub on nested scope `scope` whose chain-to-leaf resolution is exactly `(leaf, slot)`
    /// in direction `dir` — the interior endpoint of a link crossing into a nested member.
    fn stub_exposing(&self, scope: Uid, leaf: Uid, slot: &str, dir: subpatch::Dir) -> Option<Uid> {
        self.ports_of(scope)
            .into_iter()
            .filter(|id| self.stub(*id).is_some_and(|(_, p)| p.dir == dir))
            .find(|id| self.resolve_stub(scope, &id.to_hex()).is_some_and(|(u, sl)| u == leaf && sl == slot))
    }

    /// The direct member of `scope` on the path from `leaf` up the `scope_of` tree, or `leaf` itself
    /// when it is a direct member. `None` if `leaf` is not inside `scope`.
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

    /// The inner-slot key a group boundary stub should reference for a crossing link: the real slot,
    /// a nested scope's existing stub, or a freshly MINTED chain of ports, each recorded in `minted`.
    fn expose_in_nested_member(
        &mut self,
        member: Uid,
        leaf: Uid,
        slot: &str,
        dir: subpatch::Dir,
        minted: &mut Vec<(Uid, Uid)>,
    ) -> String {
        if member == leaf {
            return slot.to_string();
        }
        // The thing crossing is ALREADY a port of this member — one leaf slot sits behind exactly
        // one chain of ports, so the outer port names this one rather than minting a rival.
        if self.stub(leaf).is_some_and(|(s, _)| s == member) {
            return leaf.to_hex();
        }
        if let Some(id) = self.stub_exposing(member, leaf, slot, dir) {
            return id.to_hex();
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
        let base = self.pos(member).unwrap_or([0.0, 0.0]);
        let pos = match dir {
            subpatch::Dir::Out => [base[0] + 220.0, base[1]],
            subpatch::Dir::In => [base[0] - 40.0, base[1]],
        };
        let ty = subpatch::boundary_type_name(dir, dtype);
        let Ok(id) = self.create_node(ty, None, "", None, Some(member)) else {
            return slot.to_string();
        };
        let _ = self.set_node_pos(id, pos);
        minted.push((member, id));
        let (node, slot) = inner;
        let _ = match dir {
            subpatch::Dir::In => self.add_link(id, subpatch::BOUNDARY_SLOT, node, &slot),
            subpatch::Dir::Out => self.add_link(node, &slot, id, subpatch::BOUNDARY_SLOT),
        };
        id.to_hex()
    }

    /// The single common parent scope of `members`, or an error if the set is empty or spans several
    /// scopes. Shared by `group_nodes` and `restore_scope`.
    fn common_parent(&self, members: &[Uid]) -> Result<Option<Uid>, String> {
        if members.is_empty() {
            return Err("group: empty selection".into());
        }
        let mut parent: Option<Option<Uid>> = None;
        for &m in members {
            if !self.nodes.contains_key(&m) {
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

    /// Like [`Self::group_nodes`], but records into `minted` every stub it has to MINT on a
    /// pre-existing nested member, so the `Group` inverse can un-mint them.
    pub fn group_nodes_capturing(
        &mut self,
        members: &[Uid],
        pos: [f64; 2],
        minted: &mut Vec<(Uid, Uid)>,
    ) -> Result<Uid, String> {
        use subpatch::Dir;
        // 1. Validate BEFORE any mutation: each exists, and all share one parent scope.
        let parent = self.common_parent(members)?;
        let member_set: std::collections::HashSet<Uid> = members.iter().copied().collect();
        let scope_uid = self.mint();
        // Registered BEFORE its ports are minted. A port's label comes from the patch's ONE display
        // namespace, so a batch that names every port out of the state it started in hands each of
        // them the same label — and `nd()` then cannot tell two ports apart.
        let disp = self.fresh_name("subpatch");
        self.nodes.insert(
            scope_uid,
            NodeEntry { kind: Kind::Facade, name: disp, pos, viewers: serde_json::json!({}) },
        );
        self.set_member_scope(scope_uid, parent);

        // 2. Classify each link by TRANSITIVE containment. Exactly one endpoint inside mints a
        //    port for the slot it crosses at, and the cable is SPLIT in two around it — the outer
        //    half ends at the port, the inner half carries it to the member. Both are ordinary
        //    links; several cables leaving one slot share one port.
        let mut ports: std::collections::HashMap<(Uid, &'static str, bool), Uid> =
            std::collections::HashMap::new();
        let mut wires: Vec<(Uid, String, Uid, String)> = Vec::new();
        let mut cut: Vec<Link> = Vec::new();
        let (mut in_n, mut out_n) = (0usize, 0usize);
        // Snapshot the links: `expose_in_nested_member` may MINT an intermediate stub and needs
        // `&mut self`, so the classification cannot hold a borrow on `self.links`.
        let links = self.links.clone();
        for l in &links {
            let out_m = self.containing_member(l.node_out, &member_set);
            let in_m = self.containing_member(l.node_in, &member_set);
            let (member, at_slot, outward) = match (out_m, in_m) {
                (Some(om), None) => (om, l.slot_out, true),
                (None, Some(im)) => (im, l.slot_in, false),
                _ => continue,
            };
            let end = if outward { l.node_out } else { l.node_in };
            let key = (end, at_slot, outward);
            let port = match ports.get(&key) {
                Some(id) => *id,
                None => {
                    let (dir, dtype, at) = match outward {
                        true => (
                            Dir::Out,
                            self.output_slot_type(end, at_slot).unwrap_or(goofi_core::SlotType::Array),
                            [pos[0] + 220.0, pos[1] + 40.0 * out_n as f64],
                        ),
                        false => (
                            Dir::In,
                            self.input_slot_type(end, at_slot).unwrap_or(goofi_core::SlotType::Array),
                            [pos[0] - 40.0, pos[1] + 40.0 * in_n as f64],
                        ),
                    };
                    let inner_slot = self.expose_in_nested_member(member, end, at_slot, dir, minted);
                    let ty = subpatch::boundary_type_name(dir, dtype);
                    let id = self.create_node(ty, None, "", None, Some(scope_uid))?;
                    let _ = self.set_node_pos(id, at);
                    match outward {
                        true => {
                            wires.push((member, inner_slot, id, subpatch::BOUNDARY_SLOT.to_string()));
                            out_n += 1;
                        }
                        false => {
                            wires.push((id, subpatch::BOUNDARY_SLOT.to_string(), member, inner_slot));
                            in_n += 1;
                        }
                    }
                    ports.insert(key, id);
                    id
                }
            };
            cut.push(l.clone());
            wires.push(match outward {
                true => (
                    port,
                    subpatch::BOUNDARY_SLOT.to_string(),
                    l.node_in,
                    l.slot_in.to_string(),
                ),
                false => (
                    l.node_out,
                    l.slot_out.to_string(),
                    port,
                    subpatch::BOUNDARY_SLOT.to_string(),
                ),
            });
        }
        // The whole cable goes, so the two halves replace it rather than racing its single-input
        // eviction — a port wired to a member's input would otherwise evict the very cable it carries.
        self.links.retain(|l| !cut.contains(l));

        // 3. Re-tag membership. Members stay live; only `scope_of` changes.
        for &m in members {
            self.set_member_scope(m, Some(scope_uid));
        }
        self.set_member_scope(scope_uid, parent);
        // 4. …and only NOW wire the minted ports: a cable's two ends must face the same scope, and
        //    until the re-tag above there was no scope for them to face.
        for (a, so, b, si) in wires {
            self.add_link(a, &so, b, &si)?;
        }
        Ok(scope_uid)
    }

    /// Recreate a scope EXACTLY — the inverse of `expand_instance`, moving the members back under
    /// `scope_id` with the captured stubs verbatim, so undo/redo is uid-stable.
    pub fn restore_scope(
        &mut self,
        scope_id: Uid,
        name: String,
        pos: [f64; 2],
        members: &[Uid],
        parent: Option<Uid>,
    ) -> Result<Uid, String> {
        if self.is_facade(scope_id) {
            return Err(format!("restore_scope: scope {scope_id} already live"));
        }
        // The parent is captured explicitly rather than derived from members, so an EMPTY scope
        // restores fine. Re-tag only members that actually exist.
        // An empty name means MINT one, the rule every create uses — which is what lets a COPY
        // land beside its original.
        self.nodes.insert(
            scope_id,
            NodeEntry {
                kind: Kind::Facade,
                name: self.pick_name(&name, "subpatch", Some(scope_id)),
                pos,
                viewers: serde_json::json!({}),
            },
        );
        for &m in members {
            if self.nodes.contains_key(&m) {
                self.set_member_scope(m, Some(scope_id));
            }
        }
        // A peer may have dissolved the captured parent since. Writing it verbatim would install a
        // dangling-parent orphan, so degrade to ROOT, as the `SetScope` child does.
        self.set_member_scope(scope_id, parent.filter(|p| self.is_facade(*p)));
        Ok(scope_id)
    }


    /// The display names of a scope's ports, captured before a removal that drops them — a `nd()`
    /// binding on a name nothing wears can never be re-resolved by a later edit, so the invalidation
    /// has to happen at the removal.
    fn stub_names(&self, scope: Uid) -> Vec<String> {
        self.ports_of(scope)
            .into_iter()
            .filter_map(|p| self.name(p).map(str::to_string))
            .collect()
    }

    /// Dissolve a scope, answering the cables its removal JOINED — each pair of halves that met at
    /// a port, now one link. Uid-stable.
    pub fn expand_instance(
        &mut self,
        scope: Uid,
    ) -> Result<Vec<(Uid, &'static str, Uid, &'static str)>, String> {
        if !self.is_facade(scope) {
            return Err(format!("expand_instance: no such scope {scope}"));
        }
        let dropped = self.stub_names(scope);
        let restored = self.scope_members(scope);
        let parent = self.scope_of(scope); // the grandparent scope members fall back to

        // Every port is about to go, and each carried half a cable on either side of it. Capture the
        // JOIN of those halves — the wall is what the two halves existed for, and the wall is going.
        let ports: Vec<Uid> = self.ports_of(scope);
        let mut splices: Vec<(Uid, &'static str, Uid, &'static str)> = Vec::new();
        for &port in &ports {
            let Some(feed) = self.links.iter().find(|l| l.node_in == port).map(|l| (l.node_out, l.slot_out))
            else {
                continue;
            };
            for l in self.links.iter().filter(|l| l.node_out == port) {
                splices.push((feed.0, feed.1, l.node_in, l.slot_in));
            }
        }
        self.links.retain(|l| !ports.contains(&l.node_in) && !ports.contains(&l.node_out));

        for &m in &restored {
            self.set_member_scope(m, parent);
        }
        for p in &ports {
            self.nodes.shift_remove(p);
            self.scope_of.remove(p);
        }
        self.nodes.shift_remove(&scope);
        self.scope_of.remove(&scope);
        // …and only now, with the members up one level and the wall gone, do the two halves of each
        // cable become one link that both ends can face.
        let mut joined = Vec::new();
        for (a, so, b, si) in splices {
            if self.add_link(a, so, b, si).is_ok() {
                joined.push((a, so, b, si));
            }
        }
        for name in dropped {
            self.rebind_naming(&name);
        }
        Ok(joined)
    }

    /// Delete a whole sub-patch scope: tear down every member, recursing into nested scopes, then
    /// drop the scope.
    pub fn remove_instance(&mut self, scope: Uid) -> Result<(), String> {
        if !self.is_facade(scope) {
            return Err(format!("remove_instance: no such scope {scope}"));
        }
        let dropped = self.stub_names(scope);
        for m in self.scope_members(scope) {
            if self.is_facade(m) {
                self.remove_instance(m)?; // nested scope subtree
            } else {
                let _ = self.remove_node(m); // leaf (tolerate an already-gone member)
            }
        }
        self.nodes.shift_remove(&scope);
        self.scope_of.remove(&scope);
        for name in dropped {
            self.rebind_naming(&name);
        }
        Ok(())
    }

    /// The port of the ENCLOSING scope that exposes `(scope, port)`, if one does. Read through
    /// `inner`, so it answers the same before and after the port itself is gone.
    /// Take every cable that touches `uid` off the graph, through the door `remove_link` is — so
    /// the consumers it fed are re-planned rather than left holding a feed that has gone.
    fn cut_cables(&mut self, uid: Uid) {
        for l in self.links_view() {
            if l.node_in == uid || l.node_out == uid {
                let _ = self.remove_link(l.node_out, l.slot_out, l.node_in, l.slot_in);
            }
        }
    }

    /// Take a boundary port off a scope, answering the port it removed. Every `nd()` naming it goes
    /// unresolvable here rather than at the next edit that happens to touch it.
    pub fn remove_stub(&mut self, scope: Uid, port: Uid) -> Option<(subpatch::Port, String, [f64; 2])> {
        self.stub(port).filter(|(s, _)| *s == scope)?;
        // Its wires go with it, both halves — through the ONE door that removes a link, and BEFORE
        // the port does, so what the port relayed to is re-planned rather than left subscribed.
        self.cut_cables(port);
        let e = self.nodes.shift_remove(&port)?;
        self.scope_of.remove(&port);
        let Kind::Port(p) = e.kind else { return None };
        let st = (p, e.name, e.pos);
        self.rebind_naming(&st.1);
        Some(st)
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
        if let (Some(ev), Some(leaf)) = (&self.evaluator, entry.leaf()) {
            for b in leaf.bindings.values() {
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
        // The planner holds its OWN handle on this node's channel, which is the graph's end of its
        // services. `forget` rather than `detach`: this uid is retired, so nothing queued applies.
        if let Some(engine) = removed.leaf().map(|l| l.engine) {
            if let Some(e) = self.engine_mut(engine) {
                e.remove(uid);
            }
        }
        // §5.3: every binding that referenced this node by name is now unresolvable and must be
        // told so — a variable naming a dead producer's service is one the node waits on forever.
        let name = removed.name.clone();
        self.rebind_naming(&name);
        // Drop any membership tag: a removed node has no scope. Leaving it dangling would make a
        // reused uid (a delete→undo that restores the scope) self-parent via `common_parent`.
        self.scope_of.remove(&uid);
        // Drop links touching the node, then re-plan every consumer slot one of them fed. Links
        // INTO it need none: its thread is halted and its services are going with it.
        let dropped: Vec<Link> = self
            .links
            .iter()
            .filter(|l| l.node_out == uid || l.node_in == uid)
            .cloned()
            .collect();
        self.links
            .retain(|l| l.node_out != uid && l.node_in != uid);
        for l in dropped.iter().filter(|l| l.node_in != uid) {
            self.touched.push(Touched::Slot(l.node_in, l.slot_in));
        }
        self.rebind_ports();
        Ok(())
    }

    /// Respawn a node's instance IN PLACE. Everything that identifies it in the patch survives, so
    /// remove+add is no substitute. A Python node re-runs the source CAPTURED AT DISCOVERY.
    pub fn restart_node(&mut self, uid: Uid) -> Result<(), String> {
        // A facade has no thread of its own, so restarting it is restarting what is inside it — to
        // any depth. A port has neither thread nor members, so it is a restart of nothing.
        if self.is_facade(uid) {
            for m in self.scope_members(uid) {
                self.restart_node(m)?;
            }
            return Ok(());
        }
        if self.stub(uid).is_some() {
            return Ok(());
        }
        let entry = self.leaf(uid).ok_or_else(|| format!("no such node {uid}"))?;
        let type_name = entry.manifest.type_name;
        let held = entry.params.clone();
        // Fold what the node HAS onto what its type declares NOW: only the saved VALUE carries
        // over — bounds, options and variant are the edited file's to state.
        let mut params = self.default_params_of(type_name, None)?;
        for (group, held) in &*held {
            let Some(g) = params.get_mut(group) else { continue };
            for (name, value) in held {
                if let Some(slot) = g.get_mut(name) {
                    *slot = param_from_json(slot, &param_value_json(value));
                }
            }
        }
        // Resolve BEFORE touching the entry: a type that no longer resolves leaves the old
        // instance running rather than half-killing the node.
        let (engine, lib) =
            self.library_entry(type_name).ok_or_else(|| self.reject_type(type_name))?;
        let params = self.default_params_of(type_name, Some(params))?;

        // A restart is a BIRTH at this uid: without the generation bump the reborn node re-opens
        // names the corpse's ports still hold, and remove halts the corpse before the new Ready.
        let old_engine = self.leaf(uid).map(|e| e.engine).expect("looked up above");
        let generation = self.bump_generation(uid);
        if let Some(e) = self.engine_mut(old_engine) {
            e.remove(uid);
        }
        let boot_error = self
            .engine_mut(engine)
            .expect("the library entry named it")
            .insert(uid, type_name, generation, &params);
        let entry = self.leaf_mut(uid).expect("looked up above");
        // The MANIFEST goes with the instance: keeping the old one leaves the graph describing a
        // node not running.
        entry.manifest = lib.manifest;
        entry.isolation = lib.isolation;
        entry.engine = engine;
        // A swap, not a new record: the graph's readers hold this very handle, so replacing it
        // would leave them reading the corpse's params.
        entry.params = Arc::new(params);
        // The whole health is REBORN (§4, §6.2): the corpse's reports describe an instance that
        // no longer exists, and a fresh struct has nothing of the corpse's to show.
        entry.health = Health::born(boot_error);
        // `bindings` are left untouched — their compiled handles may only be dropped through
        // `release_entry_bindings`, and `bind_error` describes a SOURCE this rebirth did not touch.

        // A wire onto a slot the reshape retired can never propagate and cannot be repaired — the
        // slot is gone from the palette. Keeping it draws a cable the runtime ignores.
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
        // holds a name that no longer resolves. Re-resolved, not patched: the slot may be gone too.
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
            .leaf_mut(uid)
            .ok_or_else(|| format!("no such node {uid}"))?;
        if entry.params.get(group).is_none() {
            return Err(format!("no such param group `{group}`"));
        }
        edit_params(entry, |p| {
            p.entry(group.to_string()).or_default().insert(name.to_string(), value.clone());
        });
        // A LITERAL on a driven param switches it to constant, which is what the node does with
        // this write's `SetParam`; what the record retained stays retained.
        let key = ParamKey::new(group, name);
        if let Some(mut state) = self.source_state(uid, &key).filter(|s| s.mode != Mode::Constant) {
            state.mode = Mode::Constant;
            let _ = self.set_source(uid, group, name, state);
        }
        // The record has moved and the delivery is recorded for settle; nothing else happens
        // here. `on_param_changed` runs on the node's own thread, so its failure arrives as a fault.
        self.notify_param(uid, &key);
        self.invalidate_bindings_reading_param(uid, &key);
        Ok(())
    }

    /// Ask the node to re-enumerate a refreshable `Str` param's options — the ⟳ button. It answers
    /// only that the request was DISPATCHED; the options arrive as a later `RefreshOptions` status.
    pub fn refresh_param(&mut self, uid: Uid, group: &str, name: &str) -> Result<(), String> {
        let entry = self.leaf(uid).ok_or_else(|| format!("no such node {uid}"))?;
        let live = entry.params.clone();
        let param = goofi_node::param(&live, group, name)
            .ok_or_else(|| format!("no such param `{group}.{name}`"))?;
        if !matches!(param, Param::Str { refresh: true, .. }) {
            return Err(format!("param `{group}.{name}` is not refreshable"));
        }
        let engine = self.leaf(uid).map(|e| e.engine).expect("checked above");
        let key = ParamKey::new(group, name);
        if let Some(e) = self.engine_mut(engine) {
            e.request(uid, Request::RefreshParam { key });
        }
        Ok(())
    }

    /// Set a param's source record: its mode, and the expression and reference it retains. A record
    /// with nothing retained and a constant mode is removed. A bind error is stored on the record,
    /// never a refusal — the source outlives any one instance.
    pub fn set_source(
        &mut self,
        uid: Uid,
        group: &str,
        name: &str,
        state: SourceState,
    ) -> Result<(), String> {
        if self.leaf(uid).is_none() {
            return Err(format!("no such node {uid}"));
        }
        let key = ParamKey::new(group, name);
        // Only an empty record is a true unbind, and `unbind` owns the release on that path — so it
        // goes FIRST. Releasing here too gave the evaluator two `release` calls for one handle.
        if state.is_empty() {
            self.unbind(uid, &key);
            self.notify_param(uid, &key);
            return Ok(());
        }
        // Release any prior compiled handle first — this path REPLACES it.
        if let Some(prev) = self.leaf(uid).and_then(|e| e.bindings.get(&key)) {
            if let (Some(ev), Some(id)) = (&self.evaluator, prev.id) {
                ev.release(id);
            }
        }
        // A record binds a real param: a dangling one is invisible in the descriptor and
        // unclearable from the UI.
        let Some(param) =
            goofi_node::param(&self.nodes[&uid].leaf().expect("a leaf").params, group, name).cloned()
        else {
            return Err(format!("no such param `{group}/{name}`"));
        };
        // Both retained texts are scanned whatever the mode, because `terms` is what a later
        // rename or globals edit re-resolves against. Only the active one gets variables and a handle.
        let scanned = (!state.expression.is_empty()).then(|| expr_rewrite::rewrite(&state.expression));
        let reference = (!state.reference.is_empty()).then(|| parse_reference(&state.reference));
        let mut terms: Vec<expr_rewrite::VarRef> =
            scanned.iter().flatten().flat_map(|(_, refs)| refs.clone()).collect();
        if let Some(Ok(r)) = &reference {
            terms.push(r.clone());
        }
        let missing = |vars: &[BoundVar]| {
            vars.iter().find_map(|v| match v {
                BoundVar::Missing { reason, .. } => Some(reason.clone()),
                _ => None,
            })
        };
        let (rewritten, vars, mut error) = match state.mode {
            Mode::Constant => (String::new(), Vec::new(), None),
            Mode::Expression => match scanned {
                Some(Ok((rewritten, refs))) => {
                    let vars = self.resolve_vars(uid, &key, &refs);
                    let error = missing(&vars);
                    (rewritten, vars, error)
                }
                Some(Err(e)) => (state.expression.clone(), Vec::new(), Some(e.0)),
                None => (String::new(), Vec::new(), Some("no expression to evaluate".to_string())),
            },
            Mode::Reference => match reference {
                Some(Ok(r)) => {
                    let vars = self.resolve_vars(uid, &key, std::slice::from_ref(&r));
                    let error = missing(&vars).or_else(|| self.reference_kind_error(&r, &param));
                    (REF_VAR.to_string(), vars, error)
                }
                Some(Err(e)) => (String::new(), Vec::new(), Some(e)),
                None => (String::new(), Vec::new(), Some("no reference to follow".to_string())),
            },
        };
        let id = match (&self.evaluator, state.mode, error.is_none()) {
            (Some(ev), Mode::Expression, true) => match ev.compile(&rewritten) {
                Ok(c) => Some(c.id),
                Err(e) => {
                    error = Some(e.0);
                    None
                }
            },
            (None, Mode::Expression, _) => {
                error = Some("no expression evaluator available".to_string());
                None
            }
            _ => None,
        };
        let record = ParamSource {
            mode: state.mode,
            expression: state.expression,
            reference: state.reference,
            triggers_process: state.triggers,
            id,
            rewritten,
            vars,
            terms,
            bind_error: error,
        };
        if let Some(e) = self.leaf_mut(uid) {
            e.bindings.insert(key.clone(), record);
        }
        self.notify_param(uid, &key);
        Ok(())
    }

    /// Why a resolved reference cannot feed this param: the producer's slot kind against the
    /// param's type. `None` when they agree, or when the producer was not found (already reported).
    fn reference_kind_error(&self, r: &expr_rewrite::VarRef, param: &Param) -> Option<String> {
        let expr_rewrite::VarRef::Node { name, slot: Some(slot), .. } = r else { return None };
        let uid = self.uid_by_name(name)?;
        let kind = self.output_slots(uid).into_iter().find(|(_, label, _)| label == slot)?.2;
        let (wants, ok) = match param {
            Param::Str { .. } => ("STRING", kind == goofi_core::SlotType::String),
            _ => ("ARRAY", kind == goofi_core::SlotType::Array),
        };
        (!ok).then(|| {
            format!("`{name}.{slot}` is a {} output; this param references a {wants} one", kind.name())
        })
    }

    /// Drop a binding and release its compiled handle — the shared tail of an empty `set_expression`
    /// and of a literal write over a bound param. It does NOT re-plan: its callers do, exactly once.
    fn unbind(&mut self, uid: Uid, key: &ParamKey) {
        let Some(binding) = self.leaf_mut(uid).and_then(|e| e.bindings.remove(key)) else {
            return;
        };
        if let (Some(ev), Some(id)) = (&self.evaluator, binding.id) {
            ev.release(id);
        }
    }

    /// Storing the record is only HALF of a param edit: a parked node is never rung by a bare
    /// pointer swap. The delivery is recorded here and runs at [`Self::settle`], once per batch.
    fn notify_param(&mut self, uid: Uid, key: &ParamKey) {
        self.touched.push(Touched::Param(uid, key.clone()));
    }

    /// Resolve a rewrite's variables against the graph: a producer output, a global's value, or the
    /// reason neither was found. Event ids come from §3.2's `65..=128` budget, lowest free first.
    fn resolve_vars(&self, consumer: Uid, key: &ParamKey, refs: &[expr_rewrite::VarRef]) -> Vec<BoundVar> {
        let mut taken: Vec<EventId> = self
            .leaf(consumer)
            .into_iter()
            .flat_map(|e| e.bindings.iter().filter(|(k, _)| *k != key))
            .flat_map(|(_, b)| &b.vars)
            .filter_map(|v| match v {
                BoundVar::Stream { event_id, .. } => Some(*event_id),
                _ => None,
            })
            .collect();
        // One shape for every stream reference, `me.out` included: a resolved wire plus an event id.
        let stream = |var: &str, taken: &mut Vec<EventId>, resolved: Result<(Uid, &'static str), String>| match resolved {
            Err(reason) => BoundVar::Missing { var: var.to_string(), reason },
            Ok((producer, slot)) => match next_event_id(taken) {
                None => BoundVar::Missing {
                    var: var.to_string(),
                    reason: "too many expression references on this node".to_string(),
                },
                Some(event_id) => {
                    taken.push(event_id);
                    BoundVar::Stream { var: var.to_string(), producer, slot, event_id }
                }
            },
        };
        let value = |var: &str, resolved: Result<Param, String>| match resolved {
            Ok(value) => BoundVar::Value { var: var.to_string(), value },
            Err(reason) => BoundVar::Missing { var: var.to_string(), reason },
        };
        refs.iter()
            .map(|r| match r {
                expr_rewrite::VarRef::Global { var, key } => match self.globals.get(key) {
                    Some(v) => BoundVar::Value { var: var.clone(), value: global_as_param(v) },
                    None => BoundVar::Missing {
                        var: var.clone(),
                        reason: format!("global `{key}` is not defined"),
                    },
                },
                expr_rewrite::VarRef::Node { var, name, slot } => {
                    stream(var, &mut taken, self.resolve_stream(name, slot.as_deref()))
                }
                expr_rewrite::VarRef::MeOut { var, slot } => {
                    stream(var, &mut taken, self.resolve_own_stream(consumer, slot.as_deref()))
                }
                expr_rewrite::VarRef::NodeParam { var, name, group, param } => {
                    value(var, self.uid_by_name(name)
                        .ok_or_else(|| format!("no node named `{name}`"))
                        .and_then(|uid| self.param_value_of(uid, name, group, param)))
                }
                expr_rewrite::VarRef::MeParam { var, group, param } => {
                    value(var, self.param_value_of(consumer, "me", group, param))
                }
            })
            .collect()
    }

    /// This node's own output, for `me.out.slot` and bare `me` — a leaf's manifest is the slot
    /// vocabulary.
    fn resolve_own_stream(&self, uid: Uid, slot: Option<&str>) -> Result<(Uid, &'static str), String> {
        let entry = self.leaf(uid).ok_or("`me` reads a running node")?;
        let outputs = entry.manifest.outputs;
        match slot {
            Some(want) => outputs
                .iter()
                .find(|o| o.name == want)
                .map(|o| (uid, o.name))
                .ok_or_else(|| format!("this node has no output `{want}`")),
            None if outputs.len() == 1 => Ok((uid, outputs[0].name)),
            None if outputs.is_empty() => Err("this node has no outputs".to_string()),
            None => Err("`me` is ambiguous: this node has multiple outputs; use `me.out.<slot>`"
                .to_string()),
        }
    }

    /// A param's EFFECTIVE value for an expression: the evaluated report when the param is
    /// driven, else the authored record. `who` names the node in the error as the source spelled it.
    fn param_value_of(&self, uid: Uid, who: &str, group: &str, name: &str) -> Result<Param, String> {
        let entry = self
            .leaf(uid)
            .ok_or_else(|| format!("`{who}` holds no params: a port relays and a facade fronts"))?;
        entry
            .health
            .evaluated
            .get(&ParamKey::new(group, name))
            .cloned()
            .or_else(|| goofi_node::param(&entry.params, group, name).cloned())
            .ok_or_else(|| format!("`{who}` has no param `{group}/{name}`"))
    }

    /// The producer output a `nd('name')` term names, or why it names none. A bare reference to a
    /// multi-output node is refused HERE — the graph is what knows how many outputs a node has.
    fn resolve_stream(&self, name: &str, slot: Option<&str>) -> Result<(Uid, &'static str), String> {
        let uid = self.uid_by_name(name).ok_or_else(|| format!("no node named `{name}`"))?;
        // The slot vocabulary is one question and the stream behind it another, and every node kind
        // answers both — a leaf, a facade whose outputs are its ports, and a port itself.
        let outputs = self.output_slots(uid);
        let key = match slot {
            // A reference is by NAME, because `nd()` reads the one display namespace: a facade's
            // slot is its port's name, never the uid the document keys it by.
            Some(want) => outputs
                .iter()
                .find(|(_, label, _)| label == want)
                .map(|(key, _, _)| key.clone())
                .ok_or_else(|| format!("node `{name}` has no output `{want}`"))?,
            None if outputs.len() == 1 => outputs[0].0.clone(),
            None if outputs.is_empty() => return Err(format!("node `{name}` has no outputs")),
            None => {
                return Err(format!(
                    "nd('{name}') is ambiguous: it has multiple outputs; use nd('{name}').out.<slot>"
                ))
            }
        };
        match self.stream(uid, &key) {
            Some(Stream::At(leaf, slot)) => Ok((leaf, slot)),
            Some(Stream::Open(port)) => Err(format!(
                "port `{}` has nothing wired to it yet",
                self.name(port).unwrap_or(name)
            )),
            None => Err(format!("node `{name}` has no output `{key}`")),
        }
    }

    /// The source record on a param, for the bridge descriptor + `.gfi` (or `None` if the param is
    /// a plain literal with nothing retained).
    pub fn param_source(&self, uid: Uid, group: &str, name: &str) -> Option<SourceInfo> {
        let entry = self.leaf(uid)?;
        let key = ParamKey::new(group, name);
        let b = entry.bindings.get(&key)?;
        Some(SourceInfo {
            mode: b.mode,
            expression: b.expression.clone(),
            reference: b.reference.clone(),
            triggers_process: b.triggers_process,
            // Derived rather than stored: the graph could not bind it, or the node could not
            // evaluate it, and a binding the graph refused is never shipped for the node to judge.
            error: b.bind_error.clone().or_else(|| entry.health.param_errors.get(&key).cloned()),
        })
    }

    /// Every source record on a node as `(group, name, state)` — what a delete's inverse must
    /// re-apply, since params alone carry only the literal value.
    pub fn param_sources(&self, uid: Uid) -> Vec<(String, String, SourceState)> {
        self.leaf(uid)
            .map(|e| {
                e.bindings.iter().map(|(k, b)| (k.group.clone(), k.name.clone(), b.state())).collect()
            })
            .unwrap_or_default()
    }

    /// What the params with a live mode currently evaluate to — the inspector's preview. A constant
    /// is excluded: its value is the literal, already on the descriptor.
    pub fn expression_values(&self, uid: Uid) -> Vec<(&str, &str, &Param)> {
        let Some(entry) = self.leaf(uid) else {
            return Vec::new();
        };
        entry
            .health
            .evaluated
            .iter()
            .filter(|(key, _)| entry.bindings.get(key).is_some_and(|b| b.mode != Mode::Constant))
            .map(|(key, p)| (key.group.as_str(), key.name.as_str(), p))
            .collect()
    }

    /// Resolve a node display name to its uid — `nd('name')` and the CLI's name spelling alike.
    pub fn uid_by_name(&self, name: &str) -> Option<Uid> {
        self.named().find(|(_, n)| *n == name).map(|(u, _)| u)
    }

    /// A node reference as any caller may spell it: a uid, or the unique display name. An
    /// existing uid wins a hex-looking name; a well-formed uid that names nothing stays a uid, so
    /// an idempotent remove keeps its meaning.
    pub fn resolve_ref(&self, raw: &str) -> Option<Uid> {
        match Uid::from_hex(raw) {
            Some(uid) if self.exists(uid) => Some(uid),
            hex => self.uid_by_name(raw).or(hex),
        }
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

    /// The wire currently feeding a SINGLE input `(node_in, slot)` — the one an `add_link` would
    /// evict, so the `AddLink` command's inverse can restore it. `None` for a multi input.
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

    /// Does this exact (resolved) wire already exist? Lets a command detect an idempotent AddLink,
    /// so its inverse is a no-op too instead of destroying the pre-existing wire.
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
        // A facade address IS its port, folded here so no stored link ever names a scope — one
        // normalisation, at the one door every link authoring path goes through.
        let (node_out, so) = self.normalise(node_out, slot_out);
        let (node_in, si) = self.normalise(node_in, slot_in);
        let (slot_out, slot_in) = (so.as_str(), si.as_str());
        // Each slot's face, taken once: it carries both the `&'static` name a link is keyed by and
        // the dtype the check below needs, so there is no second lookup that could fail on its own.
        let out = self
            .find_output(node_out, slot_out)
            .ok_or_else(|| format!("no output slot `{slot_out}` on {node_out}"))?;
        let inp = self
            .find_input(node_in, slot_in)
            .ok_or_else(|| format!("no input slot `{slot_in}` on {node_in}"))?;
        let (slot_out, slot_in) = (out.name, inp.name);
        // Both ends must face the same scope, or the cable crosses a wall without a port to carry
        // it — which is also what refuses an IN port wired from inside, its consumer side being out.
        if self.face(node_out, true) != self.face(node_in, false) {
            let label = |uid: Uid| self.name(uid).unwrap_or("?").to_string();
            return Err(format!(
                "cannot link {} to {}: they are not in the same sub-patch — wire it to a boundary port",
                label(node_out),
                label(node_in),
            ));
        }
        // A cross-dtype cable can never carry data — the consumer reads with the wrong accessor and
        // sits empty forever. Refused here, the one door every link authoring path goes through.
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
        // A multi slot keeps its wires in connection order, which IS `links`' own order; a single
        // input takes one, so a second wire EVICTS the first. The node hears one declarative set.
        if !self.is_multi_input(node_in, slot_in) {
            self.links
                .retain(|l| !(l.node_in == node_in && l.slot_in == slot_in));
        }
        self.links.push(new);
        self.touched.push(Touched::Slot(node_in, slot_in));
        self.rebind_ports();
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
            self.touched.push(Touched::Slot(node_in, slot_in));
        }
        self.rebind_ports();
        Ok(())
    }

    /// The resolver input every service name is scoped by. The graph MINTS it and carries it;
    /// deriving a name from it is `goofi-transport`'s, which the graph never links.
    pub fn instance(&self) -> &str {
        &self.instance
    }

    /// The patch clock origin engines compute `NodeCtx::now` from.
    pub fn patch_start(&self) -> Instant {
        self.start
    }
    /// The generation of the node about to be born at `uid`: 0 for a first birth, one more than
    /// the last for every rebirth.
    fn bump_generation(&mut self, uid: Uid) -> u64 {
        let next = self.generations.get(&uid).map_or(0, |g| g + 1);
        self.generations.insert(uid, next);
        next
    }

    fn generation(&self, uid: Uid) -> u64 {
        self.generations.get(&uid).copied().unwrap_or(0)
    }

    /// A multi-step batch is opening: hold every settle until [`Self::release_settle`], so the
    /// drain cannot deliver the batch's intermediates.
    pub fn hold_settle(&mut self) {
        self.open_batches += 1;
    }

    pub fn release_settle(&mut self) {
        self.open_batches = self.open_batches.saturating_sub(1);
    }

    /// Deliver what the batch changed: one decision per touched item, from settled state, each
    /// item once however often the batch touched it. Free when nothing was.
    pub fn settle(&mut self) {
        if self.open_batches > 0 {
            return;
        }
        let raw = std::mem::take(&mut self.touched);
        if raw.is_empty() && !self.engines().any(|e| e.dirty()) {
            return;
        }
        // Port consumers expand to the leaf inputs behind them, a node the batch also removed is
        // owed nothing, and each item is delivered once however often the batch touched it.
        let mut touched: Vec<Touched> = Vec::new();
        for t in raw {
            match t {
                Touched::Slot(uid, slot) => {
                    let sinks: Vec<(Uid, &'static str)> = if self.leaf(uid).is_some() {
                        vec![(uid, slot)]
                    } else {
                        self.sinks(uid, subpatch::BOUNDARY_SLOT)
                    };
                    for (n, sl) in sinks {
                        let t = Touched::Slot(n, sl);
                        if !touched.contains(&t) {
                            touched.push(t);
                        }
                    }
                }
                Touched::Param(uid, key) => {
                    if self.leaf(uid).is_none() {
                        continue;
                    }
                    let t = Touched::Param(uid, key);
                    if !touched.contains(&t) {
                        touched.push(t);
                    }
                }
            }
        }
        let edges = self.resolved_edges();
        let rings: HashMap<&'static str, bool> =
            self.engines().map(|e| (e.id(), e.doorbell_driven())).collect();
        let Graph { nodes, generations, instance, engines, .. } = self;
        let view = build_view(nodes, generations, instance, &edges, &rings);
        for e in engines.iter_mut() {
            e.settle(&view, &touched);
        }
    }

    /// Every leaf-to-leaf wire, ports resolved away, in link order — which IS a multi input's
    /// wire order. Computed once per settle, so no engine re-implements the relay walk.
    fn resolved_edges(&self) -> Vec<Edge> {
        self.links
            .iter()
            .filter(|l| self.leaf(l.node_in).is_some())
            .filter_map(|l| match self.stream(l.node_out, l.slot_out)? {
                Stream::At(u, s) => {
                    Some(Edge { producer: (u, s), consumer: (l.node_in, l.slot_in) })
                }
                Stream::Open(_) => None,
            })
            .collect()
    }

    /// The status-drain worker's engine-side half: take every engine's waiting reports and
    /// apply the health plane. Answers how many landed, so a caller can tell a quiet graph from
    /// one it stopped hearing.
    pub fn drain_status(&mut self) -> usize {
        let mut applied = 0;
        {
            let Graph { nodes, refreshed, engines, .. } = self;
            let mut apply =
                |uid: Uid, status: Status| apply_status_to(nodes, refreshed, uid, status);
            for e in engines.iter_mut() {
                applied += e.drain(&mut apply);
            }
        }
        // A direct driver has no bridge to settle for it, and the drain-side settle lands what
        // the drain marked pending.
        self.settle();
        applied
    }

    /// Apply one health report — the drain's door, public so a test can inject one.
    pub fn apply_status(&mut self, uid: Uid, status: Status) {
        apply_status_to(&mut self.nodes, &mut self.refreshed, uid, status);
    }

    /// The params whose options were re-enumerated since the last call — the worker's cue to echo
    /// them. A QUEUE, because options are the one part of a node the doc has no field for.
    pub fn take_refreshed(&mut self) -> Vec<(Uid, ParamKey)> {
        std::mem::take(&mut self.refreshed)
    }
    /// Remove all nodes and links.
    pub fn clear(&mut self) {
        // Release each node's compiled expression handles before dropping them (load_doc
        // goes through here, so a File→Open cycle can't leak the evaluator's registry).
        for e in self.nodes.values() {
            self.release_entry_bindings(e);
        }
        // N explicit removes, then the nodes wholesale — a removal derived from absence would be
        // the engine-observes-the-graph mirror the seam rejects.
        let removed: Vec<(Uid, &'static str)> = self.leaves().map(|(u, l)| (u, l.engine)).collect();
        for (uid, engine) in removed {
            if let Some(e) = self.engine_mut(engine) {
                e.remove(uid);
            }
        }
        self.nodes.clear();
        self.links.clear();
        self.scope_of.clear();
        // Whatever the batch touched addressed nodes this clear destroyed; the generations stay,
        // keeping whatever is born at those uids next clear of what just died.
        self.touched.clear();
        // An un-echoed refresh names a node the patch no longer holds — and a load restores uids,
        // so that number can come back and the echo be read as an answer nobody asked for.
        self.refreshed.clear();
        // Globals are patch CONTENT, so a load starts from a fresh seeded store; `dyn_types` is
        // catalog and stays.
        self.globals = goofi_core::globals::GlobalStore::new();
        // The node clock belongs to the PATCH: one loaded an hour in must compute what it would
        // have at boot. Safe only because every reader of this clock was dropped just above.
        self.start = Instant::now();
        let start = self.start;
        for e in self.engines_mut() {
            e.reset_clock(start);
        }
    }

    /// Take the name a RESTORE asks for. It goes through the same gate a create does, so an
    /// archive naming something illegal or already worn costs that NAME and not the patch.
    fn force_set_name(&mut self, uid: Uid, name: &str) {
        let Some(base) = self.node_type(uid).map(name_base) else { return };
        let name = self.pick_name(name, &base, Some(uid));
        if let Some(e) = self.nodes.get_mut(&uid) {
            e.name = name;
        }
    }

    /// The `patch.yaml` manifest inside the archive: `nodes`/`links` and a flat `scopes` block
    /// under `root`, `globals` at the top. A plain flat patch has an empty `scopes` block.
    /// Every uid `roots` reaches: the roots, whatever their scopes hold to any depth, and those
    /// scopes' ports. The subtree a copy, a delete and an export all mean by "these nodes".
    pub fn subtree_of(&self, roots: &[Uid]) -> Vec<Uid> {
        let mut out: Vec<Uid> = Vec::new();
        let mut stack: Vec<Uid> = roots.iter().rev().copied().collect();
        while let Some(u) = stack.pop() {
            if !self.nodes.contains_key(&u) || out.contains(&u) {
                continue;
            }
            out.push(u);
            if self.is_facade(u) {
                stack.extend(self.scope_members(u));
            }
        }
        out
    }

    /// The `{nodes, links}` fragment for `uids` — the exact shape the `.gfi` root carries, so what
    /// a clipboard holds and what a patch holds are ONE format. A link rides only when BOTH of its
    /// ends are in the fragment; a cable reaching out of it was never the fragment's.
    pub fn fragment(&self, uids: &[Uid]) -> serde_json::Value {
        use serde_json::{json, Map, Value};
        let want: std::collections::HashSet<Uid> = uids.iter().copied().collect();
        let mut nodes = Map::new();
        // ONE loop over ONE map: a leaf, a facade and a port are all node records, and membership
        // rides each record rather than a member list beside what `scope_of` owns.
        for (uid, e) in self.nodes.iter().filter(|(u, _)| want.contains(u)) {
            let mut rec = Map::new();
            rec.insert("type".into(), json!(self.node_type(*uid).unwrap_or("")));
            rec.insert("name".into(), json!(e.name));
            rec.insert("pos".into(), json!(e.pos));
            let mut params = Map::new();
            if let Some(leaf) = e.leaf() {
                for (group, names) in &*leaf.params.clone() {
                    let gmap: Map<String, Value> =
                        names.iter().map(|(n, p)| (n.clone(), param_value_json(p))).collect();
                    params.insert(group.clone(), Value::Object(gmap));
                }
                // Persist source records (sorted for a stable diff) — else a save/load silently
                // freezes every live-driven param to its last evaluated literal.
                if !leaf.bindings.is_empty() {
                    let mut binds: Vec<(&ParamKey, &ParamSource)> = leaf.bindings.iter().collect();
                    binds.sort_by(|a, b| a.0.cmp(b.0));
                    rec.insert(
                        "sources".into(),
                        Value::Array(
                            binds
                                .iter()
                                .map(|(k, b)| {
                                    json!({ "group": k.group, "name": k.name, "mode": b.mode.as_str(),
                                            "expression": b.expression, "reference": b.reference,
                                            "triggers": b.triggers_process })
                                })
                                .collect(),
                        ),
                    );
                }
            }
            rec.insert("params".into(), Value::Object(params));
            // An empty viewer blob stays out, so a fresh patch has no noise.
            if e.viewers.as_object().is_some_and(|m| !m.is_empty()) {
                rec.insert("viewers".into(), e.viewers.clone());
            }
            if let Some(p) = self.scope_of(*uid).filter(|p| want.contains(p)) {
                rec.insert("scope".into(), json!(p.to_hex()));
            }
            nodes.insert(uid.to_hex(), Value::Object(rec));
        }
        // A port's inner wire is a link like any other — the same one `add_link` writes — so a
        // fragment has one relation kind as well as one entity kind.
        let links: Vec<Value> = self
            .links
            .iter()
            .filter(|l| want.contains(&l.node_out) && want.contains(&l.node_in))
            .map(|l| json!([l.node_out.to_hex(), l.slot_out, l.node_in.to_hex(), l.slot_in]))
            .collect();
        json!({ "nodes": Value::Object(nodes), "links": links })
    }

    /// The params a record asks for, folded over the type's defaults. NON-seeding, because a
    /// restore must not re-synthesize a binding the user had unbound.
    fn record_params(&self, ty: &str, rec: &serde_json::Value) -> Result<ParamGroups, String> {
        let mut params = self.default_params_of(ty, None)?;
        let Some(groups) = rec.get("params").and_then(|v| v.as_object()) else { return Ok(params) };
        for (group, names) in groups {
            let (Some(nm), Some(g)) = (names.as_object(), params.get_mut(group)) else { continue };
            for (name, val) in nm {
                if let Some(existing) = g.get_mut(name) {
                    *existing = param_from_json(existing, val);
                }
            }
        }
        Ok(params)
    }

    /// Build `uid -> fresh uid` for every record of a fragment, so a link and a `scope` can be
    /// remapped before anything is created.
    fn remap_fragment(&mut self, nodes: &serde_json::Map<String, serde_json::Value>) -> HashMap<String, Uid> {
        nodes.keys().map(|old| (old.clone(), self.mint())).collect()
    }

    /// Add a `{nodes, links}` fragment under `scope`, shifted by `offset`, on FRESH uids. Answers
    /// the ONE command that does it — so a paste is one undo step — beside what each record's uid
    /// became, which is what a caller selects afterwards.
    pub fn import_fragment(
        &mut self,
        doc: &serde_json::Value,
        scope: Option<Uid>,
        offset: [f64; 2],
    ) -> Result<(command::Command, HashMap<String, String>), String> {
        use command::Command;
        let nodes = doc.get("nodes").and_then(|v| v.as_object()).ok_or("paste: missing `nodes`")?;
        if let Some(s) = scope.filter(|s| !self.is_facade(*s)) {
            return Err(format!("paste: no such scope {s}"));
        }
        for rec in nodes.values() {
            let ty = rec.get("type").and_then(|v| v.as_str()).ok_or("paste: a record has no `type`")?;
            if !structural(ty) && !self.known_type(ty) {
                return Err(self.reject_type(ty));
            }
        }
        let idmap = self.remap_fragment(nodes);
        // The names the copy will wear, picked BEFORE anything is built, because an expression in
        // the fragment spells a NAME: a source left naming the original binds the copy to it, and
        // outlives the original's deletion as a broken reference.
        let mut taken: std::collections::HashSet<String> =
            self.nodes.values().map(|e| e.name.clone()).collect();
        let mut renamed: HashMap<String, String> = HashMap::new();
        for rec in nodes.values() {
            let base = name_base(rec["type"].as_str().unwrap_or(""));
            let fresh = (0..)
                .map(|n| format!("{base}{n}"))
                .find(|c| !taken.contains(c))
                .expect("an unbounded counter finds a free name");
            taken.insert(fresh.clone());
            renamed.insert(rec.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(), fresh);
        }
        let by_old = |old: &str| renamed.get(old).cloned();
        let at = |p: [f64; 2]| [p[0] + offset[0], p[1] + offset[1]];
        // A facade before the members that name it, and a port after every facade: a port is a
        // port OF a scope, so it takes its scope at birth where everything else is placed below.
        let kind_order = |ty: &str| match (ty == subpatch::SCOPE_TYPE, subpatch::boundary_type(ty)) {
            (true, _) => 0,
            (_, Some(_)) => 2,
            _ => 1,
        };
        let mut order: Vec<&String> = nodes.keys().collect();
        order.sort_by_key(|old| kind_order(nodes[*old]["type"].as_str().unwrap_or("")));
        let mut cmds: Vec<Command> = Vec::new();
        for old in &order {
            let rec = &nodes[*old];
            let ty = rec["type"].as_str().unwrap_or("");
            let inner = rec.get("scope").and_then(|v| v.as_str()).and_then(|s| idmap.get(s)).copied();
            cmds.push(Command::AddNode {
                type_name: ty.to_string(),
                pos: at(read_pos(rec)),
                uid: Some(idmap[*old]),
                name: by_old(rec.get("name").and_then(|v| v.as_str()).unwrap_or("")),
                params: (!structural(ty)).then(|| self.record_params(ty, rec)).transpose()?,
                sources: record_sources(rec)
                    .into_iter()
                    .map(|(group, name, mut s)| {
                        // Both positions a display name is read in: a copied node's own, and — under
                        // a copied facade — a copied PORT's, which is what that facade calls a slot.
                        let remap = |named: &str, slot: Option<&str>| {
                            let to = by_old(named);
                            let label = to.as_ref().and(slot).and_then(by_old);
                            (to, label)
                        };
                        if let Some(src) = expr_rewrite::rename_refs(&s.expression, remap) {
                            s.expression = src;
                        }
                        if let Some(r) = expr_rewrite::rename_reference(&s.reference, remap) {
                            s.reference = r;
                        }
                        (group, name, s)
                    })
                    .collect(),
                viewers: rec.get("viewers").filter(|v| v.is_object()).map(|v| remap_slots(v, &idmap)),
                // A port cannot exist without a scope, so it takes the paste target when its own
                // facade is not in the fragment — the same fallback every other kind gets below.
                scope: subpatch::boundary_type(ty).and(inner.or(scope)),
            });
        }
        for old in &order {
            let rec = &nodes[*old];
            let inner = rec.get("scope").and_then(|v| v.as_str()).and_then(|s| idmap.get(s)).copied();
            // A record naming no scope INSIDE the fragment is a root of it, so it lands where the
            // paste was aimed; one naming a scope in here keeps the shape it was copied with.
            cmds.push(Command::SetScope { uid: idmap[*old], scope: inner.or(scope) });
        }
        for l in doc.get("links").and_then(|v| v.as_array()).into_iter().flatten() {
            let Some(a) = l.as_array().filter(|a| a.len() == 4) else { continue };
            let (Some(no), Some(ni)) = (
                a[0].as_str().and_then(|s| idmap.get(s)).copied(),
                a[2].as_str().and_then(|s| idmap.get(s)).copied(),
            ) else {
                continue;
            };
            cmds.push(Command::AddLink {
                node_out: no,
                slot_out: a[1].as_str().unwrap_or("").to_string(),
                node_in: ni,
                slot_in: a[3].as_str().unwrap_or("").to_string(),
            });
        }
        let rename = idmap.into_iter().map(|(old, new)| (old, new.to_hex())).collect();
        Ok((Command::Compound(cmds), rename))
    }

    pub fn serialize(&self) -> String {
        use serde_json::{json, Value};
        let root = self.fragment(&self.all_uids());
        // An ORDERED array, because the order is observable and a keyed map would alphabetize it
        // away. On load, `reassert_system` back-fills — so an older patch picks up a new default.
        let globals: Vec<Value> = self
            .globals
            .entries()
            // A locked global is the MACHINE's; writing it into a patch would carry one machine's
            // path onto another.
            .filter(|(_, _, _, locked)| !locked)
            .map(|(name, value, _is_system, _)| {
                let mut e = global_to_json(value); // {value, type}
                if let Value::Object(ref mut m) = e {
                    m.insert("name".to_string(), Value::String(name.to_string()));
                }
                e
            })
            .collect();
        let mut doc = json!({
            "version": MANIFEST_VERSION,
            "goofi": env!("CARGO_PKG_VERSION"),
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

    /// Replace the graph from a `.gfi` manifest. Node types are validated before the current
    /// graph is torn down (a rejected load is a no-op).
    pub fn load_doc(&mut self, text: &str) -> Result<(), String> {
        let doc: serde_json::Value = serde_yaml_ng::from_str(text).map_err(|e| e.to_string())?;
        let (nodes_v, links_v) = match doc.get("version").and_then(|v| v.as_i64()) {
            Some(MANIFEST_VERSION) => {
                let root = doc.get("root");
                (root.and_then(|r| r.get("nodes")), root.and_then(|r| r.get("links")))
            }
            _ => {
                // `goofi:` is read BEFORE the gate refuses, so the refusal can name the writer.
                let writer = doc
                    .get("goofi")
                    .and_then(|v| v.as_str())
                    .map(|w| format!(" — the file was written by goofi {w}"))
                    .unwrap_or_default();
                return Err(format!(
                    "unsupported .gfi version (this build reads version {MANIFEST_VERSION}){writer}"
                ));
            }
        };
        let nodes = nodes_v.and_then(|v| v.as_object()).ok_or("missing `nodes`")?;
        for rec in nodes.values() {
            let ty = rec.get("type").and_then(|v| v.as_str()).ok_or("node missing `type`")?;
            // A facade and a boundary port are the model's own types, not the palette's: they have
            // no module to be missing, so the availability gate is not theirs to pass.
            if structural(ty) {
                continue;
            }
            if !self.known_type(ty) {
                return Err(self.reject_type(ty));
            }
        }

        self.clear();
        // Globals load BEFORE nodes so a node's `globals.*` default-expression resolves at
        // instantiation, IN FILE ORDER. Malformed entries are skipped (best-effort load).
        if let Some(serde_json::Value::Array(arr)) = doc.get("globals") {
            for entry in arr {
                if let (Some(name), Some(value)) =
                    (entry.get("name").and_then(|v| v.as_str()), global_from_json(entry))
                {
                    let _ = self.globals.apply_change(name, Some(value), None);
                }
            }
        }
        // Every uid this load hands out, restored or minted — what keeps two records from landing
        // on one uid when a hand-written file spells the same number two ways.
        let mut claimed: HashSet<Uid> = HashSet::new();
        let mut idmap: HashMap<String, Uid> = HashMap::new();
        // Every uid FIRST, so a record's `scope` and a link's endpoints resolve whatever the
        // iteration order — one uid space, so one pass answers for all three entity kinds.
        for old in nodes.keys() {
            let uid = self.restore_uid(old, &claimed);
            claimed.insert(uid);
            idmap.insert(old.clone(), uid);
        }
        // Then the facades, before anything that can name one.
        for (old, rec) in nodes.iter().filter(|(_, r)| r["type"] == subpatch::SCOPE_TYPE) {
            self.nodes.insert(
                idmap[old],
                NodeEntry {
                    kind: Kind::Facade,
                    name: String::new(),
                    pos: read_pos(rec),
                    viewers: serde_json::json!({}),
                },
            );
            self.force_set_name(idmap[old], rec.get("name").and_then(|v| v.as_str()).unwrap_or(""));
            if let Some(v) = rec.get("viewers").filter(|v| v.is_object()) {
                let _ = self.set_node_viewers(idmap[old], v.clone());
            }
        }
        for (old, rec) in nodes.iter().filter(|(_, r)| !structural(r["type"].as_str().unwrap_or(""))) {
            let ty = rec["type"].as_str().unwrap();
            // Folded in BEFORE construction, since `insert_node` runs `setup()`.
            let params = self.record_params(ty, rec)?;
            let (engine, entry) = self.library_entry(ty).ok_or_else(|| self.reject_type(ty))?;
            let params = self.default_params_of(ty, Some(params))?;
            // The record's KEY is its uid — restored, not reminted (see `restore_uid`). The name is
            // the type's fresh one only until the record's own `name` lands, just below.
            let uid = idmap[old];
            let name = self.fresh_name(&entry.manifest.type_name.to_lowercase());
            self.insert_node_at(uid, name, engine, entry, params);
            if let Some(name) = rec.get("name").and_then(|v| v.as_str()) {
                self.force_set_name(uid, name);
            }
            let _ = self.set_node_pos(uid, read_pos(rec));
            if let Some(v) = rec.get("viewers").filter(|v| v.is_object()) {
                let _ = self.set_node_viewers(uid, v.clone());
            }
            for (group, name, state) in record_sources(rec) {
                let _ = self.set_source(uid, &group, &name, state);
            }
        }
        // Membership, from each record's own `scope`. It is set before the ports so a port's
        // scope is already a member of whatever holds IT.
        for (old, rec) in nodes {
            let parent = rec.get("scope").and_then(|v| v.as_str()).and_then(|s| idmap.get(s)).copied();
            if parent.is_some() {
                self.set_member_scope(idmap[old], parent);
            }
        }
        // The ports. Each is a member record whose type carries its direction and dtype, so nothing
        // about it is re-derived from the wire it will get below.
        for (old, rec) in nodes {
            let Some((dir, dtype)) = subpatch::boundary_type(rec["type"].as_str().unwrap_or("")) else {
                continue;
            };
            let uid = idmap[old];
            // A port with no scope is not one, and the membership pass above is what gave it its.
            if self.scope_of(uid).is_none() {
                continue;
            }
            self.nodes.insert(
                uid,
                NodeEntry {
                    kind: Kind::Port(subpatch::Port { dir, dtype }),
                    name: String::new(),
                    pos: read_pos(rec),
                    viewers: rec
                        .get("viewers")
                        .filter(|v| v.is_object())
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({})),
                },
            );
            self.force_set_name(uid, rec.get("name").and_then(|v| v.as_str()).unwrap_or(""));
        }
        if let Some(links) = links_v.and_then(|v| v.as_array()) {
            for l in links {
                let Some(a) = l.as_array().filter(|a| a.len() == 4) else { continue };
                let no = a[0].as_str().and_then(|s| idmap.get(s)).copied();
                let ni = a[2].as_str().and_then(|s| idmap.get(s)).copied();
                let (Some(no), Some(ni)) = (no, ni) else { continue };
                let (so, si) = (a[1].as_str().unwrap_or(""), a[3].as_str().unwrap_or(""));
                // A link with one end on a port IS that port's inner wire — the same dispatch
                // `add_link` makes, so the file and the op vocabulary say one thing.
                let _ = self.add_link(no, so, ni, si);
            }
        }
        // A load writes scopes and ports straight into the maps, so it never pays the
        // `rebind_naming` a live add does — and a binding parsed before them names nothing yet.
        self.rebind_ports();
        self.viewpoint = doc.get("viewpoint").cloned().unwrap_or(serde_json::Value::Null);
        // A corrupt arrangement costs the CHROME, never the patch. The reason is kept for the load
        // reply; an ABSENT arrangement is not a corrupt one and warns about nothing.
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
}

/// Is this a type the MODEL owns rather than the palette — a sub-patch facade or a boundary port?
fn structural(ty: &str) -> bool {
    ty == subpatch::SCOPE_TYPE || subpatch::boundary_type(ty).is_some()
}

/// The stem a minted display name counts from: a leaf's type, and the kind's own word for the two
/// that have no type to be named after.
fn name_base(type_name: &str) -> String {
    match subpatch::boundary_type(type_name) {
        Some((dir, _)) => dir.name().to_string(),
        None if type_name == subpatch::SCOPE_TYPE => "subpatch".to_string(),
        None => type_name.to_lowercase(),
    }
}

/// A viewer blob under the uids a paste minted. A facade keys its blob by PORT UID, so a copy that
/// kept the original's keys would point at slots it does not have; a leaf keys its by slot NAME,
/// which no remap names, so it rides through unchanged.
fn remap_slots(viewers: &serde_json::Value, idmap: &HashMap<String, Uid>) -> serde_json::Value {
    match viewers.as_object() {
        None => viewers.clone(),
        Some(m) => serde_json::Value::Object(
            m.iter()
                .map(|(k, v)| (idmap.get(k).map_or_else(|| k.clone(), |u| u.to_hex()), v.clone()))
                .collect(),
        ),
    }
}

/// The expression bindings a record carries, in the shape [`command::Command::AddNode`] re-applies.
fn record_sources(rec: &serde_json::Value) -> Vec<(String, String, SourceState)> {
    let text = |ex: &serde_json::Value, k: &str| ex.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    rec.get("sources")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|ex| {
            Some((
                ex.get("group")?.as_str()?.to_string(),
                ex.get("name")?.as_str()?.to_string(),
                SourceState {
                    mode: ex.get("mode").and_then(|v| v.as_str()).and_then(Mode::parse).unwrap_or_default(),
                    expression: text(ex, "expression"),
                    reference: text(ex, "reference"),
                    triggers: ex.get("triggers").and_then(|v| v.as_bool()).unwrap_or(false),
                },
            ))
        })
        .collect()
}

/// A reference's `node.slot` as the one variable its record resolves — the same term an
/// expression's `nd('node').out.slot` rewrites to.
fn parse_reference(reference: &str) -> Result<expr_rewrite::VarRef, String> {
    let Some((name, slot)) = reference.split_once('.') else {
        return Err(format!("a reference spells `node.slot`, not `{reference}`"));
    };
    if !goofi_core::globals::is_valid_name(name) || !goofi_core::globals::is_valid_name(slot) {
        return Err(format!("`{reference}` is not a legal reference: {NAME_RULE}"));
    }
    Ok(expr_rewrite::VarRef::Node {
        var: REF_VAR.to_string(),
        name: name.to_string(),
        slot: Some(slot.to_string()),
    })
}

fn read_pos(rec: &serde_json::Value) -> [f64; 2] {
    rec.get("pos")
        .and_then(|v| v.as_array())
        .and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?]))
        .unwrap_or([0.0, 0.0])
}

/// The health plane's one mutator: apply one report off any engine's drain. A free function so
/// the drain can hold the engines and the node map apart.
fn apply_status_to(
    nodes: &mut IndexMap<Uid, NodeEntry>,
    refreshed: &mut Vec<(Uid, ParamKey)>,
    uid: Uid,
    status: Status,
) {
    let Some(entry) = nodes.get_mut(&uid).and_then(NodeEntry::leaf_mut) else { return };
    match status {
        Status::Stage { stage } => entry.health.stage = stage.as_str(),
        Status::Ufreq { hz } => entry.health.ufreq = Some(hz),
        // The options are the node's answer to a refresh (§8.5). They land in the health
        // OVERLAY, never a reply or the record: the RPC that asked has already returned.
        Status::RefreshOptions { key, options } => {
            if let Some(options) = options {
                entry.health.options.insert(key.clone(), options);
            }
            // Queued whether or not there were any: this IS the answer to a ⟳, and the client
            // lifts its spinner off the echo. A node with no hook for the param answers `None`.
            refreshed.push((uid, key));
        }
        Status::Fault { fault } => match fault {
            // A clean run clears Setup/Process/Boot together and never touches a binding
            // error, which only that binding evaluating successfully clears (§6).
            None => {
                entry.health.setup_error = None;
                entry.health.last_error = None;
            }
            Some(goofi_node::NodeFault::Setup { msg, .. }) => entry.health.setup_error = Some(msg),
            Some(goofi_node::NodeFault::Process { msg, .. }) => entry.health.last_error = Some(msg),
        },
        // One record for what the instance reported, bound param or not. On the binding it would
        // outlive the instance, since a reborn node has nothing to announce clearing.
        Status::BindingErrors { errors } => {
            for (key, msg) in errors {
                match msg {
                    Some(msg) => {
                        entry.health.param_errors.insert(key, msg);
                    }
                    None => {
                        entry.health.param_errors.shift_remove(&key);
                    }
                }
            }
        }
        Status::ParamValues { evaluated } => {
            entry.health.evaluated = evaluated.into_iter().collect();
        }
    }
    // Stamp when the error first read the way it does now — re-stamped only when the message
    // changes, so the instant is its onset.
    let current = entry_error(entry).map(str::to_string);
    if entry.health.error_since.as_ref().map(|(m, _)| m.as_str()) != current.as_deref() {
        entry.health.error_since = current.map(|m| (m, Instant::now()));
    }
}

/// The settled view, borrowed from the one model — built at the settle point and nowhere else.
fn build_view<'a>(
    nodes: &'a IndexMap<Uid, NodeEntry>,
    generations: &HashMap<Uid, u64>,
    instance: &'a str,
    edges: &'a [Edge],
    rings: &HashMap<&'static str, bool>,
) -> GraphView<'a> {
    let nodes = nodes
        .iter()
        .filter_map(|(uid, e)| {
            let leaf = e.leaf()?;
            let bindings = leaf
                .bindings
                .iter()
                .map(|(key, b)| BindingView {
                    key,
                    kind: match b.mode {
                        Mode::Reference => goofi_node::SourceKind::Reference,
                        _ => goofi_node::SourceKind::Expression,
                    },
                    rewritten: &b.rewritten,
                    vars: &b.vars,
                    trigger: b.triggers_process,
                    id: b.id,
                    live: b.live(),
                })
                .collect();
            Some((
                *uid,
                NodeView {
                    engine: leaf.engine,
                    generation: generations.get(uid).copied().unwrap_or(0),
                    rings: rings.get(leaf.engine).copied().unwrap_or(true),
                    manifest: leaf.manifest,
                    params: leaf.params.as_ref(),
                    bindings,
                },
            ))
        })
        .collect();
    GraphView { instance, edges, nodes }
}

/// One node's current error, derived fresh from the places one can arise. A free function so the
/// status drain can read it while holding a `&mut NodeEntry`.
fn entry_error(e: &Leaf) -> Option<&str> {
    // Initialization failure outranks a process error, and is the only thing that CAN be true
    // beside one: if `setup` failed, `process` never ran.
    if let Some(err) = e.health.setup_error.as_deref() {
        return Some(err);
    }
    if let Some(err) = e.health.last_error.as_deref() {
        return Some(err);
    }
    // Both param-keyed error records, ordered by key together, so which record an error landed in
    // cannot decide whether the badge ever shows it.
    e.bindings
        .iter()
        .filter_map(|(k, b)| b.bind_error.as_deref().map(|s| (k, s)))
        .chain(e.health.param_errors.iter().map(|(k, m)| (k, m.as_str())))
        .min_by(|a, b| a.0.cmp(b.0))
        .map(|(_, s)| s)
}

