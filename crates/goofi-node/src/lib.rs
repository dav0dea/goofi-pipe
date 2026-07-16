//! goofi-node — the ONE node abstraction plus its runtime plumbing and the
//! native compile-time catalog.
//!
//! Every node — native Rust, in-process pyo3 (free-threaded), or subprocess —
//! implements [`Node`]. The scheduler never branches on backend. A node holds
//! its own current param values as fields (seeded by `make`, updated via
//! `on_param_changed`); `process` reads them directly, so the tick path never
//! does a param-map lookup. The engine owns trigger arbitration, rate limiting,
//! index stamping, and output gating *outside* the node.

use std::fmt;

use goofi_core::{Data, Param, SlotType};
use indexmap::IndexMap;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NodeError(pub String);

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for NodeError {}
impl From<String> for NodeError {
    fn from(s: String) -> Self {
        NodeError(s)
    }
}
impl From<&str> for NodeError {
    fn from(s: &str) -> Self {
        NodeError(s.to_string())
    }
}

pub type NodeResult = std::result::Result<(), NodeError>;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// Grouped params: `group -> (name -> Param)`, insertion-ordered.
pub type ParamGroups = IndexMap<String, IndexMap<String, Param>>;

/// A `(group, name)` address into a node's params.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamKey {
    pub group: String,
    pub name: String,
}

impl ParamKey {
    pub fn new(group: impl Into<String>, name: impl Into<String>) -> ParamKey {
        ParamKey {
            group: group.into(),
            name: name.into(),
        }
    }
}

/// Look up a param by `(group, name)`.
pub fn param<'a>(p: &'a ParamGroups, group: &str, name: &str) -> Option<&'a Param> {
    p.get(group)?.get(name)
}

/// A static, declarative parameter descriptor — the param analogue of [`SlotDecl`]/
/// [`OutputDecl`], holding only `&'static str` + primitives so a node can declare its
/// params as a `static PARAMS: &[ParamDecl]` (a literal `&[Param]` is impossible —
/// `Param::Str` owns heap `String`/`Vec`). The runtime [`ParamGroups`] is built from
/// these on demand by [`NodeManifest::default_params`].
pub struct ParamDecl {
    pub group: &'static str,
    pub name: &'static str,
    pub spec: ParamSpec,
}

/// The kind + defaults of a declared param.
pub enum ParamSpec {
    Float { default: f64, min: f64, max: f64 },
    Int { default: i64, min: i64, max: i64 },
    Bool { default: bool },
    Str { default: &'static str, options: &'static [&'static str], refresh: bool },
    Trigger,
}

impl ParamSpec {
    /// Materialize the runtime [`Param`] this descriptor declares.
    fn to_param(&self) -> Param {
        match *self {
            ParamSpec::Float { default, min, max } => Param::float(default, min, max),
            ParamSpec::Int { default, min, max } => Param::int(default, min, max),
            ParamSpec::Bool { default } => Param::boolean(default),
            ParamSpec::Str { default, options, refresh } => Param::Str {
                value: default.to_string(),
                options: (!options.is_empty())
                    .then(|| options.iter().map(|s| s.to_string()).collect()),
                refresh,
            },
            ParamSpec::Trigger => Param::Trigger { fired: false },
        }
    }
}

/// Build a grouped [`ParamGroups`] from a flat, group-tagged declaration list —
/// group order = first-seen, name order = declaration order (matching the old
/// imperative `default_params()` builders).
pub fn params_from_decls(decls: &[ParamDecl]) -> ParamGroups {
    let mut groups = ParamGroups::new();
    for d in decls {
        groups
            .entry(d.group.to_string())
            .or_default()
            .insert(d.name.to_string(), d.spec.to_param());
    }
    groups
}

/// A read-only, typed view of a node's current params, handed to `setup`/`process`
/// so a *cold* param (read occasionally, mirrored to no field, with no side effect)
/// can be read live — needing no field and no `on_param_changed` arm. The engine's
/// `NodeEntry.params` is the source of truth, so a live edit is visible on the next
/// tick. Read each param into a local once at the top of `process`; the per-*sample*
/// hot loop then reads the local, never the map.
pub struct Params<'a>(&'a ParamGroups);

impl<'a> Params<'a> {
    pub fn new(groups: &'a ParamGroups) -> Params<'a> {
        Params(groups)
    }
    pub fn f64(&self, group: &str, name: &str) -> Option<f64> {
        param(self.0, group, name).and_then(Param::as_f64)
    }
    pub fn i64(&self, group: &str, name: &str) -> Option<i64> {
        param(self.0, group, name).and_then(Param::as_i64)
    }
    pub fn bool(&self, group: &str, name: &str) -> Option<bool> {
        param(self.0, group, name).and_then(Param::as_bool)
    }
    pub fn str(&self, group: &str, name: &str) -> Option<&str> {
        param(self.0, group, name).and_then(Param::as_str)
    }
    /// The underlying groups, for the rare node that needs to iterate.
    pub fn groups(&self) -> &ParamGroups {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Tick I/O
// ---------------------------------------------------------------------------

/// The per-tick input view handed to a node. Single-source slots hold the latest
/// `Data` (`None` if unwired / no frame yet); `multi` slots hold an ordered list of
/// the latest frame from each connected wire (present-only, connection order —
/// materialized by the engine). Borrowed for the duration of a tick. The two maps
/// are keyed disjointly by slot name (a slot is single XOR multi).
pub struct Inputs<'a> {
    singles: &'a IndexMap<&'static str, Option<Data>>,
    multis: Option<&'a IndexMap<&'static str, Vec<Data>>>,
}

impl<'a> Inputs<'a> {
    /// A single-source-only input view (no `multi` slots).
    pub fn new(singles: &'a IndexMap<&'static str, Option<Data>>) -> Inputs<'a> {
        Inputs { singles, multis: None }
    }
    /// An input view with `multi` slots (the engine's materialized per-wire lists).
    pub fn with_multi(
        singles: &'a IndexMap<&'static str, Option<Data>>,
        multis: &'a IndexMap<&'static str, Vec<Data>>,
    ) -> Inputs<'a> {
        Inputs { singles, multis: Some(multis) }
    }
    /// The latest frame on a single slot. On a `multi` slot, the first present frame
    /// (a convenience — `multi` nodes read [`get_multi`](Self::get_multi)).
    pub fn get(&self, name: &str) -> Option<&Data> {
        if let Some(o) = self.singles.get(name) {
            if let Some(d) = o.as_ref() {
                return Some(d);
            }
        }
        self.multis.and_then(|m| m.get(name)).and_then(|v| v.first())
    }
    /// The ordered list of present frames on a `multi` slot (connection order). On a
    /// single slot, a total 0/1-element slice — so a node need not special-case arity.
    pub fn get_multi(&self, name: &str) -> &[Data] {
        if let Some(m) = self.multis {
            if let Some(v) = m.get(name) {
                return v.as_slice();
            }
        }
        match self.singles.get(name) {
            Some(Some(d)) => std::slice::from_ref(d),
            _ => &[],
        }
    }
}

/// A pre-sized output sink (seeded with the manifest's output slot names). A
/// node writes with `out.set(slot, data)`; slots left unset emit nothing.
pub struct Outputs<'a> {
    slots: &'a mut IndexMap<&'static str, Option<Data>>,
}

impl<'a> Outputs<'a> {
    pub fn new(slots: &'a mut IndexMap<&'static str, Option<Data>>) -> Outputs<'a> {
        Outputs { slots }
    }
    /// Set an output slot. Writing an unknown slot name is a no-op.
    pub fn set(&mut self, name: &str, data: Data) {
        if let Some(s) = self.slots.get_mut(name) {
            *s = Some(data);
        }
    }
}

/// Per-tick engine context handed to a node.
#[derive(Debug, Default)]
pub struct NodeCtx {
    /// Monotonic tick counter for this node.
    pub tick: u64,
    /// Wall-clock seconds since the graph's first tick (monotonic, `0.0` on the
    /// first tick). Wall-clock-paced generators (audio) read this to emit exactly
    /// the samples that elapsed, drift-free; most nodes ignore it.
    pub now: f64,
}

impl NodeCtx {
    pub fn new() -> NodeCtx {
        NodeCtx::default()
    }
}

// ---------------------------------------------------------------------------
// RunPolicy — the scheduler's projection of the `common` param group
// ---------------------------------------------------------------------------

/// How `common.max_frequency` is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FrequencyMode {
    /// `max_frequency` is a rate in Hz — the node runs at most that many times/sec.
    #[default]
    UpdatesPerSecond,
    /// `max_frequency` is a period in seconds — the node runs once per that many sec.
    SecondsPerUpdate,
}

/// When a node's `process` may run, lifted out of the params so the tick path
/// never does a map lookup. This is the single-process engine's adaptation of the
/// Python node loop's autotrigger gate + `_rate_limit_sleep`: because one shared
/// loop drives every node, a node cannot *sleep* to pace itself (that would stall
/// the others) — instead the scheduler *gates* each node's run on elapsed
/// wall-clock, so a node capped at N Hz simply runs on the ticks where its period
/// has elapsed and is skipped on the rest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunPolicy {
    /// Run every allowed tick even with no fresh input — a free-running producer.
    /// Only takes effect when the node has no *wired* triggering input; a node
    /// whose trigger input is connected runs on that input's rate regardless (the
    /// engine enforces this, since wiring isn't visible here). See [`Self::should_run`].
    pub autotrigger: bool,
    /// Max run rate. `<= 0` is unbounded: an input-triggered node then runs at its
    /// input's rate, a free-running one every tick (so it must set a finite cap to
    /// not saturate the loop).
    pub max_frequency: f64,
    pub frequency_mode: FrequencyMode,
}

impl Default for RunPolicy {
    fn default() -> RunPolicy {
        RunPolicy {
            autotrigger: false,
            max_frequency: 0.0,
            frequency_mode: FrequencyMode::UpdatesPerSecond,
        }
    }
}

impl RunPolicy {
    /// The minimum seconds between runs, or `None` when unbounded (`max_frequency
    /// <= 0`). Ported from the Python `_rate_limit_sleep` period computation.
    pub fn period(&self) -> Option<f64> {
        if self.max_frequency <= 0.0 {
            return None;
        }
        Some(match self.frequency_mode {
            FrequencyMode::UpdatesPerSecond => 1.0 / self.max_frequency,
            FrequencyMode::SecondsPerUpdate => self.max_frequency,
        })
    }

    /// Whether a node that already wants to run this tick is admitted by its rate
    /// cap. `wants_run` folds the whole trigger/source/autotrigger decision — the
    /// engine computes it from graph topology, because whether `autotrigger`
    /// free-runs a node depends on whether its trigger input is *wired* (Python's
    /// `autotrigger AND _has_no_triggering_inputs()`), which this type can't see.
    /// This method only applies the frequency cap on top: unbounded always passes,
    /// a never-run node (`since_last == None`) runs immediately, otherwise the
    /// period must have elapsed.
    pub fn should_run(&self, since_last: Option<f64>, wants_run: bool) -> bool {
        if !wants_run {
            return false;
        }
        match self.period() {
            None => true,
            Some(p) => since_last.is_none_or(|dt| dt >= p),
        }
    }

    /// Read the policy from a node's `common` param group, defaulting each field
    /// when the group or a key is absent (so a node without a `common` group is a
    /// triggered, unbounded node — the safe default).
    pub fn from_params(p: &ParamGroups) -> RunPolicy {
        let autotrigger = param(p, "common", "autotrigger")
            .and_then(Param::as_bool)
            .unwrap_or(false);
        let max_frequency = param(p, "common", "max_frequency")
            .and_then(Param::as_f64)
            .unwrap_or(0.0);
        let frequency_mode = match param(p, "common", "frequency_mode").and_then(Param::as_str) {
            Some("seconds-per-update") => FrequencyMode::SecondsPerUpdate,
            _ => FrequencyMode::UpdatesPerSecond,
        };
        RunPolicy {
            autotrigger,
            max_frequency,
            frequency_mode,
        }
    }
}

/// Guarantee a node's params carry the universal `common` scheduling group (the
/// engine's equivalent of Python's `DEFAULT_PARAMS["common"]`), so rate controls
/// exist on every node uniformly. Any keys a node already declared are kept;
/// missing ones are filled with behavior-preserving defaults (unbounded, not
/// autotriggering). `common` is placed first for a stable frontend ordering. Used
/// both when instantiating a node (the engine) and when projecting a node type to
/// the palette (the bridge), so type-level and instance-level params agree.
pub fn with_common(params: ParamGroups) -> ParamGroups {
    let mut common = params.get("common").cloned().unwrap_or_default();
    common
        .entry("autotrigger".to_string())
        .or_insert_with(|| Param::boolean(false));
    common
        .entry("max_frequency".to_string())
        .or_insert_with(|| Param::float(0.0, 0.0, 60.0));
    common
        .entry("frequency_mode".to_string())
        .or_insert_with(|| Param::Str {
            value: "updates-per-second".to_string(),
            options: Some(vec![
                "updates-per-second".to_string(),
                "seconds-per-update".to_string(),
            ]),
            refresh: false,
        });
    let mut merged = ParamGroups::new();
    merged.insert("common".to_string(), common);
    for (k, v) in params {
        if k != "common" {
            merged.insert(k, v);
        }
    }
    merged
}

// ---------------------------------------------------------------------------
// The node trait
// ---------------------------------------------------------------------------

pub trait Node: Send {
    /// Derived one-time init, after the node's params have been seeded (via the
    /// construction replay of `on_param_changed`). Reads live params from `p`; may
    /// fail terminally (surfaced on the node's error channel via the bootstrap pipe).
    fn setup(&mut self, _ctx: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
    /// The tick body: read latest inputs + live params, write outputs. Pure w.r.t.
    /// transport. Cold params are read from `p`; hot/stateful params are mirrored to
    /// fields via `on_param_changed`.
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        ctx: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult;
    /// Optional: react to a param edit — mirror a hot field or run a side effect
    /// (re-anchor pacing, reallocate a buffer). This same handler seeds mirrored
    /// fields at construction (the engine replays it per declared param), so it is
    /// the single source of truth for param→field. Cold params need no arm here.
    fn on_param_changed(&mut self, _key: &ParamKey, _v: &Param) -> NodeResult {
        Ok(())
    }
    /// Optional: re-enumerate a `Str` param's options for the UI's ⟳ button
    /// (device/stream pickers). Paired with `on_param_changed` by name.
    fn on_param_refreshed(&mut self, _key: &ParamKey) -> Option<Vec<String>> {
        None
    }
    // Teardown is `impl Drop for TheNode`, not a trait method — it runs automatically
    // when the engine drops the boxed node, and can't be forgotten.
}

/// The generic node factory the manifest stores — construct a default instance,
/// type-erased. The engine seeds its params afterward by replaying
/// `on_param_changed`, so no per-node constructor boilerplate is written.
pub fn default_factory<T: Node + Default + 'static>() -> Box<dyn Node> {
    Box::new(T::default())
}

// ---------------------------------------------------------------------------
// Param expressions — the injected evaluator seam (impl lives in goofi-py, so the
// engine core carries no pyo3 dependency). See the param-expressions design.
// ---------------------------------------------------------------------------

/// An opaque handle to a compiled expression, owned by the evaluator.
pub type BindingId = u64;

/// A param-expression failure: compile error, runtime exception, ambiguous bare
/// `nd()`, missing ref with no fallback, or a type-incompatible result. Surfaced as a
/// core node error (the same channel as a `process` error) plus a per-param indicator.
#[derive(Debug, Clone)]
pub struct ExprError(pub String);

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ExprError {}
impl From<String> for ExprError {
    fn from(s: String) -> Self {
        ExprError(s)
    }
}
impl From<&str> for ExprError {
    fn from(s: &str) -> Self {
        ExprError(s.to_string())
    }
}

/// A reference an expression makes to another node's output: `nd('node').slot`, or a
/// bare `nd('node')` (`slot` = `None` → the node's single output slot; a node with more
/// than one output slot makes a bare `nd()` an error).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExprRef {
    pub node: String,
    pub slot: Option<String>,
}

/// The result of compiling an expression: the evaluator's opaque handle plus the
/// statically-extracted references, so the engine knows the dependency set (for
/// scheduling + dirty-tracking) without executing the snippet.
pub struct Compiled {
    pub id: BindingId,
    pub refs: Vec<ExprRef>,
}

/// Per-evaluation context handed to [`ExprEvaluator::eval`].
pub struct EvalCtx<'a> {
    /// Resolved referenced `Data` keyed by `(node name, slot|None)`. A `None` value
    /// means the ref is missing / has not emitted (or is a feedback back-edge with no
    /// prior frame) — the expression sees `nd(...)` as absent and may default it.
    pub refs: &'a std::collections::HashMap<(String, Option<String>), Option<Data>>,
    /// Engine wall-clock seconds (`NodeCtx::now`) — for time-based (ref-less) expressions.
    pub t: f64,
    /// The param being driven, a type template the evaluator coerces its result to.
    pub target: &'a Param,
}

/// Evaluates param expressions. Implemented by `goofi-py` against the free-threaded
/// interpreter and injected into the engine, so the engine core carries no pyo3
/// dependency. The engine owns the binding lifecycle + scheduling and calls this only
/// to compile / evaluate / release.
pub trait ExprEvaluator: Send + Sync {
    /// Compile a snippet once; returns the opaque handle + extracted `nd()` refs.
    fn compile(&self, source: &str) -> Result<Compiled, ExprError>;
    /// Evaluate a compiled expression to a concrete [`Param`] value.
    fn eval(&self, id: BindingId, ctx: &EvalCtx<'_>) -> Result<Param, ExprError>;
    /// Release a compiled expression (on unbind / node removal).
    fn release(&self, id: BindingId);
}

// ---------------------------------------------------------------------------
// Manifest + native catalog (inventory)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isolation {
    InProcess,
    Subprocess,
}

pub struct SlotDecl {
    pub name: &'static str,
    pub kind: SlotType,
    /// Whether fresh data on this slot wakes `process()` (vs. a held reference input).
    pub trigger_process: bool,
    /// A `multi` (variadic) input slot accepts an arbitrary number of wires and
    /// delivers them to the node as an ordered `&[Data]` (via `inp.get_multi`),
    /// latest-wins per wire, in connection order. Fixed by the node author here —
    /// a slot is single or multi for the life of the node type, never toggled.
    pub multi: bool,
}

pub struct OutputDecl {
    pub name: &'static str,
    pub kind: SlotType,
}

/// Static, declarative node metadata, registered at compile time via `inventory`.
pub struct NodeManifest {
    pub type_name: &'static str,
    pub category: &'static str,
    pub doc: &'static str,
    pub inputs: &'static [SlotDecl],
    pub outputs: &'static [OutputDecl],
    /// Declared params — the param analogue of `inputs`/`outputs`. The runtime
    /// `ParamGroups` is built on demand by [`Self::default_params`].
    pub params: &'static [ParamDecl],
    pub isolation: Isolation,
    /// Build a default instance (type-erased). The engine seeds params afterward by
    /// replaying `on_param_changed`; for native nodes this is `default_factory::<T>`.
    pub factory: fn() -> Box<dyn Node>,
}

impl NodeManifest {
    /// A fresh output buffer seeded with this manifest's output slot names.
    pub fn output_buffer(&self) -> IndexMap<&'static str, Option<Data>> {
        self.outputs.iter().map(|o| (o.name, None)).collect()
    }
    /// The runtime default params, built from the static [`ParamDecl`] list. Callers
    /// layer [`with_common`] on top (as before). Replaces the old
    /// `default_params: fn() -> ParamGroups` field.
    pub fn default_params(&self) -> ParamGroups {
        params_from_decls(self.params)
    }
}

inventory::collect!(NodeManifest);

/// Iterate the native node catalog.
pub fn catalog() -> impl Iterator<Item = &'static NodeManifest> {
    inventory::iter::<NodeManifest>()
}

/// Find a native node manifest by type name.
pub fn find(type_name: &str) -> Option<&'static NodeManifest> {
    catalog().find(|m| m.type_name == type_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Data, DType, Meta, SlotType};

    #[test]
    fn inputs_and_outputs_tick_io() {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", None);
        let inp = Inputs::new(&inmap);
        assert!(inp.get("data").is_none());
        assert!(inp.get("missing").is_none());

        let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
            .unwrap();
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        {
            let mut out = Outputs::new(&mut outmap);
            out.set("out", d.clone());
            out.set("nonexistent", d); // writing an unknown slot is a no-op
        }
        assert!(outmap.get("out").unwrap().is_some());
    }

    #[test]
    fn get_multi_returns_present_frames_in_connection_order() {
        fn mk(v: f32) -> Data {
            Data::from_array_bytes(DType::F32, vec![1], v.to_le_bytes().to_vec(), Meta::empty()).unwrap()
        }
        fn val(d: &Data) -> f32 {
            match d.value() {
                goofi_core::Value::Array(s) => f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()),
                _ => panic!(),
            }
        }
        let singles: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        let mut multis: IndexMap<&'static str, Vec<Data>> = IndexMap::new();
        multis.insert("ins", vec![mk(1.0), mk(2.0), mk(3.0)]);
        let inp = Inputs::with_multi(&singles, &multis);
        let got = inp.get_multi("ins");
        assert_eq!(got.len(), 3);
        assert_eq!([val(&got[0]), val(&got[1]), val(&got[2])], [1.0, 2.0, 3.0], "order preserved");
        assert_eq!(inp.get("ins").map(val), Some(1.0), "get() on a multi slot -> first present");
        assert!(inp.get_multi("absent").is_empty());

        // get_multi on a single slot is total: 0/1-element slice.
        let mut singles2: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        singles2.insert("one", Some(mk(9.0)));
        singles2.insert("empty", None);
        let inp2 = Inputs::new(&singles2);
        assert_eq!(inp2.get_multi("one").len(), 1);
        assert_eq!(val(&inp2.get_multi("one")[0]), 9.0);
        assert!(inp2.get_multi("empty").is_empty());
        assert!(inp2.get_multi("missing").is_empty());
    }

    static DECL_PARAMS: &[ParamDecl] = &[
        ParamDecl { group: "g", name: "freq", spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 10.0 } },
        ParamDecl { group: "g", name: "n", spec: ParamSpec::Int { default: 4, min: 1, max: 9 } },
        ParamDecl { group: "g", name: "wave", spec: ParamSpec::Str { default: "sine", options: &["sine", "saw"], refresh: false } },
        ParamDecl { group: "z", name: "on", spec: ParamSpec::Bool { default: true } },
    ];

    #[test]
    fn params_from_decls_preserves_group_and_name_order_and_values() {
        let p = params_from_decls(DECL_PARAMS);
        // Group order = first-seen ("g" before "z"); name order = declaration order.
        assert_eq!(p.keys().collect::<Vec<_>>(), vec!["g", "z"]);
        assert_eq!(p["g"].keys().collect::<Vec<_>>(), vec!["freq", "n", "wave"]);
        assert_eq!(param(&p, "g", "freq").and_then(Param::as_f64), Some(1.0));
        assert_eq!(param(&p, "g", "n").and_then(Param::as_i64), Some(4));
        assert_eq!(param(&p, "z", "on").and_then(Param::as_bool), Some(true));
        // A Str with options materializes them; refresh flag carried through.
        match param(&p, "g", "wave").unwrap() {
            Param::Str { value, options, refresh } => {
                assert_eq!(value, "sine");
                assert_eq!(options.as_deref(), Some(&["sine".to_string(), "saw".to_string()][..]));
                assert!(!refresh);
            }
            _ => panic!("expected Str"),
        }
    }

    #[test]
    fn params_view_reads_typed_values() {
        let p = params_from_decls(DECL_PARAMS);
        let view = Params::new(&p);
        assert_eq!(view.f64("g", "freq"), Some(1.0));
        assert_eq!(view.i64("g", "n"), Some(4));
        assert_eq!(view.str("g", "wave"), Some("sine"));
        assert_eq!(view.bool("z", "on"), Some(true));
        assert_eq!(view.f64("g", "missing"), None);
    }

    #[test]
    fn manifest_default_params_builds_from_decls() {
        let m = find("_NodeTestNop").unwrap();
        assert!(m.default_params().is_empty(), "Nop declares no params");
    }

    #[test]
    fn param_lookup() {
        let mut g = IndexMap::new();
        g.insert("x".to_string(), Param::float(1.0, 0.0, 2.0));
        let mut groups: ParamGroups = IndexMap::new();
        groups.insert("grp".to_string(), g);
        assert_eq!(param(&groups, "grp", "x").and_then(Param::as_f64), Some(1.0));
        assert!(param(&groups, "grp", "missing").is_none());
        assert!(param(&groups, "nogroup", "x").is_none());
    }

    #[derive(Default)]
    struct Nop;
    impl Node for Nop {
        fn process(
            &mut self,
            _i: &Inputs<'_>,
            _o: &mut Outputs<'_>,
            _c: &mut NodeCtx,
            _p: &Params<'_>,
        ) -> NodeResult {
            Ok(())
        }
    }
    static NOP_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    static NOP_PARAMS: &[ParamDecl] = &[];
    inventory::submit! {
        NodeManifest {
            type_name: "_NodeTestNop",
            category: "test",
            doc: "",
            inputs: &[],
            outputs: NOP_OUT,
            params: NOP_PARAMS,
            isolation: Isolation::InProcess,
            factory: default_factory::<Nop>,
        }
    }

    #[test]
    fn run_policy_period_by_mode() {
        // Unbounded when max_frequency <= 0.
        assert_eq!(RunPolicy::default().period(), None);
        // updates-per-second: period = 1/f.
        let ups = RunPolicy { max_frequency: 4.0, ..Default::default() };
        assert_eq!(ups.period(), Some(0.25));
        // seconds-per-update: period = f.
        let spu = RunPolicy {
            max_frequency: 2.0,
            frequency_mode: FrequencyMode::SecondsPerUpdate,
            ..Default::default()
        };
        assert_eq!(spu.period(), Some(2.0));
    }

    #[test]
    fn run_policy_gates_on_wants_run() {
        // should_run is purely the rate gate over the engine's `wants_run`
        // decision: no desire to run -> never runs; a desire + unbounded -> runs.
        // (The autotrigger/trigger/source logic lives in the engine, not here.)
        let p = RunPolicy::default();
        assert!(!p.should_run(None, false), "doesn't want to run -> no");
        assert!(p.should_run(None, true), "wants to run + unbounded -> yes");
    }

    #[test]
    fn run_policy_rate_caps_frequency() {
        // Capped at 10 Hz (period 0.1s); `wants_run` is true throughout.
        let p = RunPolicy { max_frequency: 10.0, ..Default::default() };
        assert!(p.should_run(None, true), "never run yet -> runs immediately");
        assert!(!p.should_run(Some(0.05), true), "0.05s < 0.1s period -> skip");
        assert!(p.should_run(Some(0.10), true), "period elapsed -> run");
        assert!(p.should_run(Some(0.30), true), "well past period -> run");
        // Unbounded ignores elapsed time entirely.
        let unbounded = RunPolicy::default();
        assert!(unbounded.should_run(Some(0.0), true));
    }

    #[test]
    fn run_policy_from_params_reads_common_group() {
        let mut common = IndexMap::new();
        common.insert("autotrigger".to_string(), Param::boolean(true));
        common.insert("max_frequency".to_string(), Param::float(30.0, 0.0, 60.0));
        common.insert("frequency_mode".to_string(), Param::str_free("seconds-per-update"));
        let mut groups: ParamGroups = IndexMap::new();
        groups.insert("common".to_string(), common);
        let p = RunPolicy::from_params(&groups);
        assert_eq!(
            p,
            RunPolicy { autotrigger: true, max_frequency: 30.0, frequency_mode: FrequencyMode::SecondsPerUpdate }
        );
        // A node with no `common` group defaults to triggered + unbounded.
        assert_eq!(RunPolicy::from_params(&ParamGroups::new()), RunPolicy::default());
    }

    #[test]
    fn catalog_registration_and_output_buffer() {
        let m = find("_NodeTestNop").expect("registered via inventory");
        assert_eq!(m.outputs.len(), 1);
        assert_eq!(m.isolation, Isolation::InProcess);
        let buf = m.output_buffer();
        assert!(buf.contains_key("out"));
        assert!(catalog().any(|m| m.type_name == "_NodeTestNop"));
    }
}
