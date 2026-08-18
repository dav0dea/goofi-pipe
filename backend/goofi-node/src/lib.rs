//! goofi-node — the ONE node abstraction plus its runtime plumbing and the
//! native compile-time catalog.
//!
//! Every node — native Rust, in-process pyo3 (free-threaded), or subprocess —
//! implements [`Node`]. The scheduler never branches on backend. A node holds
//! its own current param values as fields (seeded by `make`, updated via
//! `on_param_changed`); `process` reads them directly, so a run never
//! does a param-map lookup. The engine owns trigger arbitration, rate limiting,
//! index stamping, and output gating *outside* the node.

use std::fmt;

use goofi_core::{Data, Param, SlotType};
use indexmap::IndexMap;

pub mod discover;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Defines a `pub struct $name(pub String)` error newtype with the byte-identical
/// Display / Error / `From<String>` / `From<&str>` impls every node-error string type wants.
macro_rules! string_error {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone)]
        pub struct $name(pub String);
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl std::error::Error for $name {}
        impl From<String> for $name {
            fn from(s: String) -> Self {
                $name(s)
            }
        }
        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                $name(s.to_string())
            }
        }
    };
}

string_error!(NodeError);

pub type NodeResult = std::result::Result<(), NodeError>;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// Grouped params: `group -> (name -> Param)`, insertion-ordered.
pub type ParamGroups = IndexMap<String, IndexMap<String, Param>>;

/// A `(group, name)` address into a node's params. Serializable because it addresses a param over
/// the wire too — every `SetParam`, ack and binding error the async runtime sends names one.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy)]
pub struct ParamDecl {
    pub group: &'static str,
    pub name: &'static str,
    pub spec: ParamSpec,
    /// An optional default *expression* (e.g. `"globals.default_ufreq"`): when a node is freshly
    /// instantiated, the engine seeds a binding on this param instead of a plain literal, so it
    /// tracks the referenced globals/refs. The `spec` default is the graceful fallback (used verbatim
    /// when no evaluator is wired). `None` ⇒ an ordinary literal-default param.
    pub expression: Option<ExprDecl>,
    /// Help text for the UI's tooltip. Static per-type metadata, so it lives here and never on
    /// the runtime [`Param`] — a doc string on the value would be copied into the CRDT doc, the
    /// `.gfi`, and every param clone.
    pub doc: Option<&'static str>,
}

/// A declared param expression: its source plus the two flags the fx editor exposes per binding.
/// One optional struct rather than three flat fields on [`ParamDecl`], so a mode or a trigger flag
/// with no source is unconstructible — and so the ~30 params that declare no expression say so in
/// one word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExprDecl {
    pub source: &'static str,
    /// Whether this expression starts live. `Off` is a *carried* expression — the source is there
    /// for the inspector's fx toggle to turn on, so a param can ship the expression its author
    /// expects the user to want without imposing it.
    pub mode: ExprMode,
    /// Whether re-evaluating it also wakes `process()`.
    pub trigger: bool,
}

/// Whether a declared [`ExprDecl`] is live or merely carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExprMode {
    /// Saved on the declaration, awaiting the inspector toggle — the `spec` literal is in force.
    Off,
    /// Live: the engine seeds an enabled binding, and the param tracks what the expression reads.
    On,
}

/// The kind + defaults of a declared param.
#[derive(Clone, Copy)]
pub enum ParamSpec {
    Float { default: f64, min: f64, max: f64 },
    Int { default: i64, min: i64, max: i64 },
    Bool { default: bool },
    Str { default: &'static str, options: &'static [&'static str], refresh: bool },
    Trigger,
}

impl ParamSpec {
    /// Materialize the runtime [`Param`] this descriptor declares.
    fn to_param(self) -> Param {
        match self {
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
/// run. Read each param into a local once at the top of `process`; the per-*sample*
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

/// The per-run input view handed to a node. Single-source slots hold the latest
/// `Data` (`None` if unwired / no frame yet); `multi` slots hold an ordered list of
/// the latest frame from each connected wire (present-only, connection order —
/// materialized by the engine). Borrowed for the duration of one run. The two maps
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

/// Per-run engine context handed to a node.
#[derive(Debug, Default, Clone)]
pub struct NodeCtx {
    /// Wall-clock seconds since the PATCH began (monotonic). One clock across every
    /// node thread rather than one per birth, so a node born later does not start at
    /// `0.0`. Wall-clock-paced generators (audio) read this to emit exactly the
    /// samples that elapsed, drift-free; most nodes ignore it.
    pub now: f64,
    /// The patch globals as of this run. `process` reads them live (a mid-run edit is seen on the
    /// next run); `setup` latches them once at setup time. Empty for a node run outside a graph.
    pub globals: goofi_core::globals::GlobalsSnapshot,
}

impl NodeCtx {
    pub fn new() -> NodeCtx {
        NodeCtx::default()
    }
}

// ---------------------------------------------------------------------------
// RunPolicy — the scheduler's projection of the `common` param group
// ---------------------------------------------------------------------------

/// The two ways a user can author `common.max_frequency`. These are pure *input*
/// conventions — [`RunPolicy`] normalizes both to updates-per-second, so the scheduler
/// only ever reasons in Hz (the sentinels live here so `with_common` and `from_params`
/// agree on the one spelling).
/// The two spellings `common.frequency_mode` admits. Public because they ARE the vocabulary —
/// they reach the catalog, the wire and the inspector.
pub const FREQ_MODE_UPDATES_PER_SECOND: &str = "updates-per-second";
pub const FREQ_MODE_SECONDS_PER_UPDATE: &str = "seconds-per-update";

/// When a node's `process` may run, lifted out of the params so a run never does a
/// map lookup.
///
/// Every node owns a thread and decides for itself when to run, so a capped node
/// PARKS for the remainder of its period rather than being skipped by a shared
/// scheduler — see `goofi_engine::runtime`'s `next_wake`. That is the difference
/// from the retired central loop, where a node could not sleep without stalling
/// every other node.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct RunPolicy {
    /// Run whenever the rate cap allows, with no fresh input — a free-running producer.
    ///
    /// Independent of the input slots: it is one of the three reasons a node wakes, and
    /// none of them consults topology. An earlier version of this doc said it "only takes
    /// effect when the node has no *wired* triggering input". That was never true after the
    /// cutover, and it is the opposite of what `goofi_engine::runtime` states.
    /// Defaults to `false` (triggered).
    pub autotrigger: bool,
    /// Max run rate in **updates-per-second** (Hz). `<= 0` is unbounded (the default): an
    /// input-triggered node then runs at its input's rate, a free-running one as fast as
    /// its thread allows (so it must set a finite cap to not saturate a core). A node authored
    /// in `seconds-per-update` mode is normalized to Hz by [`Self::from_params`], so
    /// this is always a rate — the mode is a pure input convenience.
    pub max_frequency: f64,
}

impl RunPolicy {
    /// The minimum seconds between runs, or `None` when unbounded (`max_frequency
    /// <= 0`). Ported from the Python `_rate_limit_sleep` period computation.
    pub fn period(&self) -> Option<f64> {
        (self.max_frequency > 0.0).then(|| 1.0 / self.max_frequency)
    }

    /// Read the policy from a node's `common` param group, defaulting each field
    /// when the group or a key is absent (so a node without a `common` group is a
    /// triggered, unbounded node — the safe default). A `seconds-per-update` period is
    /// normalized to a Hz rate here (`1/period`), so `max_frequency` is always a rate.
    pub fn from_params(p: &ParamGroups) -> RunPolicy {
        let autotrigger = param(p, "common", "autotrigger")
            .and_then(Param::as_bool)
            .unwrap_or(false);
        let raw = param(p, "common", "max_frequency")
            .and_then(Param::as_f64)
            .unwrap_or(0.0);
        let seconds_per_update = param(p, "common", "frequency_mode").and_then(Param::as_str)
            == Some(FREQ_MODE_SECONDS_PER_UPDATE);
        // A period P seconds is a rate of 1/P Hz; `raw <= 0` stays unbounded in either mode.
        let max_frequency = if seconds_per_update && raw > 0.0 { 1.0 / raw } else { raw };
        RunPolicy { autotrigger, max_frequency }
    }
}

/// One universal `common` param, expressed as a function of the manifest it is being added to.
/// The function IS the declaration — there is no base-plus-override pair — so a param is defined
/// in exactly one place and states its own condition there. Most read nothing from the manifest.
///
/// These run while the `common` group is being BUILT, so a declaration may read the manifest's
/// static shape (`producer`, slots) but must never read `m.params` for a `common` key: that is a
/// half-built world.
pub type CommonDecl = fn(&NodeManifest) -> ParamDecl;

/// Run on the node's own schedule instead of waiting for an input frame — which is exactly what
/// being a producer means, so the default IS `m.producer`. It is the spec *default* that moves and
/// not the materialized value, so the declaration a reader sees — or that Task 4's seeding walk
/// reads — is the one `with_common` materializes. One number, not two kept in step.
fn autotrigger(m: &NodeManifest) -> ParamDecl {
    ParamDecl {
        group: "common",
        name: "autotrigger",
        spec: ParamSpec::Bool { default: m.producer },
        expression: None,
        doc: Some(
            "Run on the node's own schedule, instead of waiting for an input frame. \
             Turn this on for sources; leave it off for transforms driven by their input.",
        ),
    }
}

/// The rate cap. Every node CARRIES the patch's producer rate as an expression, so any node can be
/// paced by `globals.default_ufreq` with one inspector toggle; on a producer it is already live,
/// because a source is what the patch rate is for.
///
/// `trigger: true` unconditionally, and it is INERT here — do not read it as load-bearing. Spec
/// §1.1: a `common.*` arrival never sets `trigger_pending`, because re-pacing is not a reason to
/// run. What re-paces a sleeping producer is the graph re-resolving this binding on a
/// `default_ufreq` edit and re-sending it (`Graph::invalidate_bindings_reading`), and the node's
/// `common` branch re-deriving `RunPolicy` from the arrival — `trigger` is nowhere in that path. It
/// is declared for interface completeness: the field is on every `ParamDecl`, the frontend renders
/// it, and a non-`common` declaration means it.
fn max_frequency(m: &NodeManifest) -> ParamDecl {
    ParamDecl {
        group: "common",
        name: "max_frequency",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 100.0 },
        expression: Some(ExprDecl {
            source: "globals.default_ufreq",
            mode: if m.producer { ExprMode::On } else { ExprMode::Off },
            trigger: true,
        }),
        doc: Some(
            "Rate cap for this node, read through `frequency_mode`. 0 means uncapped — the node \
             runs as often as the scheduler and its inputs allow.",
        ),
    }
}

/// How to read [`max_frequency`]: a rate, or a period.
fn frequency_mode(_: &NodeManifest) -> ParamDecl {
    ParamDecl {
        group: "common",
        name: "frequency_mode",
        spec: ParamSpec::Str {
            default: FREQ_MODE_UPDATES_PER_SECOND,
            options: &[FREQ_MODE_UPDATES_PER_SECOND, FREQ_MODE_SECONDS_PER_UPDATE],
            refresh: false,
        },
        expression: None,
        doc: Some(
            "How to read `max_frequency`: as a rate in Hz (updates per second), or as a period \
             in seconds between updates — convenient for very slow nodes.",
        ),
    }
}

/// The universal `common` scheduling group, declared once. A fourth param is added here and
/// nowhere else: the loop in [`with_common`] carries no name match, and
/// `Graph::seed_default_expressions` walks the same list for the expression half.
pub static COMMON_DECLS: &[CommonDecl] = &[autotrigger, max_frequency, frequency_mode];

/// The universal declarations as THIS node type sees them — the ONE place a manifest is allowed to
/// change what the `common` group means. The value half ([`with_common`]) and Task 4's expression
/// seeding both read this, so they cannot disagree about what a producer gets.
pub fn common_decls(m: &NodeManifest) -> impl Iterator<Item = ParamDecl> + '_ {
    COMMON_DECLS.iter().map(move |d| d(m))
}

/// Guarantee a node's params carry the universal `common` scheduling group (the
/// engine's equivalent of Python's `DEFAULT_PARAMS["common"]`), so rate controls
/// exist on every node uniformly. **Any key the node declared itself is kept untouched** — this
/// function never overwrites a manifest's own param definition; `or_insert_with` is the whole of
/// that rule. Missing ones are materialized from [`common_decls`], which is where the manifest
/// decides what a missing key defaults to. `common` is placed first for a stable frontend
/// ordering. Used both when instantiating a node (the engine) and when projecting a node type to
/// the palette (the bridge), so type-level and instance-level params agree.
pub fn with_common(params: ParamGroups, m: &NodeManifest) -> ParamGroups {
    let mut common = params.get("common").cloned().unwrap_or_default();
    for d in common_decls(m) {
        common.entry(d.name.to_string()).or_insert_with(|| d.spec.to_param());
    }
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
    /// Derived init, after the node's params have been seeded (via the replay of
    /// `on_param_changed`). Reads live params from `p`. It runs once if it succeeds; an `Err`
    /// leaves the node UNINITIALIZED — surfaced on its error channel, nothing else runs against
    /// it, and the next interaction retries the whole initialization on this same instance.
    /// (The detached tier is the exception: its worker latches the failure and `restart_node` is
    /// the retry door — see `goofi_engine::ensure_initialized`.)
    fn setup(&mut self, _ctx: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
    /// The run body: read latest inputs + live params, write outputs. Pure w.r.t.
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
    /// fields at initialization — the engine replays it per declared param, at construction and
    /// again on every retry of a failed `setup` — so it is the single source of truth for
    /// param→field. Cold params need no arm here.
    fn on_param_changed(&mut self, _key: &ParamKey, _v: &Param) -> NodeResult {
        Ok(())
    }
    /// Optional: re-enumerate a `Str` param's options for the UI's ⟳ button
    /// (device/stream pickers). Paired with `on_param_changed` by name. `p` is the node's LIVE
    /// params — a picker usually enumerates against its current settings (which host, which
    /// driver, which directory), and a node that never runs would otherwise see only the values
    /// it was constructed with.
    fn on_param_refreshed(&mut self, _key: &ParamKey, _p: &Params<'_>) -> Option<Vec<String>> {
        None
    }
    // Teardown is `impl Drop for TheNode`, not a trait method — it runs automatically
    // when the engine drops the boxed node, and can't be forgotten. It is also the ONLY release
    // mechanism, and it does NOT fire between initialization retries: a `setup` that fails partway
    // is called again on the same instance, so it must itself release what it acquired before
    // returning `Err`, or it leaks a handle per retry.
}

/// The generic node factory the manifest stores — construct a default instance,
/// type-erased. The engine seeds its params afterward by replaying
/// `on_param_changed`, so no per-node constructor boilerplate is written.
pub fn default_factory<T: Node + Default + 'static>() -> Box<dyn Node> {
    Box::new(T::default())
}

// ---------------------------------------------------------------------------
// Param expressions — the injected evaluator seam (impl lives in goofi-python, so the
// engine core carries no pyo3 dependency). See the param-expressions design.
// ---------------------------------------------------------------------------

/// An opaque handle to a compiled expression, owned by the evaluator.
pub type BindingId = u64;

string_error!(
    /// A param-expression failure: compile error, runtime exception, ambiguous bare
    /// `nd()`, missing ref with no fallback, or a type-incompatible result. Surfaced as a
    /// core node error (the same channel as a `process` error) plus a per-param indicator.
    ExprError
);

/// The result of compiling an expression: the evaluator's opaque handle. What the snippet
/// REFERENCES is not extracted here — the graph rewrote every `nd()` and `globals.*` term into a
/// generated variable before compiling (spec §5.3), so the compiled source names none of them and
/// the graph already holds the resolved variable map.
pub struct Compiled {
    pub id: BindingId,
}

/// One expression variable's value, as the graph resolved it (spec §5.3). A stream variable carries
/// a producer's frame; a `globals.*` variable carries a scalar.
#[derive(Clone, Debug)]
pub enum Local {
    Frame(Data),
    Value(Param),
}

/// Per-evaluation context handed to [`ExprEvaluator::eval`].
pub struct EvalCtx<'a> {
    /// The expression's variables, keyed by the GENERATED name the rewrite minted (`__v0`). A
    /// `None` value is a variable that has not arrived yet — the expression sees it as absent.
    ///
    /// This replaced `refs` (keyed by node name and slot) and `globals` together: the rewrite is
    /// what makes both unnecessary, and keeping either would leave two ways to reach a reference —
    /// one the graph resolved and one the evaluator resolved for itself.
    pub locals: &'a std::collections::HashMap<String, Option<Local>>,
    /// Engine wall-clock seconds (`NodeCtx::now`) — for time-based (variable-less) expressions.
    pub t: f64,
    /// The param being driven, a type template the evaluator coerces its result to.
    pub target: &'a Param,
}

/// Evaluates param expressions. Implemented by `goofi-python` against the free-threaded
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
// `nd()` reference scanning — the one source of truth for finding `nd('name')`
// references in an expression source. Both extraction (which producers an
// expression depends on) and rewriting (renaming a referenced node) run off this
// scan, so they can never disagree on what counts as a reference.
// ---------------------------------------------------------------------------

/// One `nd(..)` call [`scan_nd_calls`] found, with every span its two consumers need.
///
/// The two spans exist because the two consumers want different halves of one call: a RENAME
/// replaces the name literal and must leave every other byte alone, while the expression REWRITE
/// (`goofi_engine::expr_rewrite`) replaces the whole term. They share this scan rather than each
/// carrying their own, because the drift between two word-boundary rules is invisible — a call one
/// of them declines to see is a rename that silently stops following, or a reference that never
/// becomes a variable.
pub struct NdCall<'a> {
    /// Byte offset of the `n` in `nd` — where the TERM begins.
    pub start: usize,
    /// The name literal's content, between the quotes — what a rename replaces.
    pub name_start: usize,
    pub name_end: usize,
    /// One past the call's closing `)`, or `None` when the call does not close with one (an extra
    /// argument, an unterminated call). A rename still applies to those; a rewrite cannot span
    /// them, and leaves them verbatim so the failure is a visible eval error.
    pub end: Option<usize>,
    pub name: &'a str,
}

/// Scan `source` for `nd('name')` / `nd("name")` calls, in source order.
///
/// A plain lexical scan, not a full parse: `nd` must be a standalone token (word
/// boundary before it, so `round('x')`, `s.rfind('y')`, `grand('z')` do NOT match),
/// whitespace between `nd` and `(` is tolerated (`nd ('sig')` is a valid call), and the
/// first argument must be a single string literal (a non-literal `nd(x)` is skipped).
pub fn scan_nd_calls(source: &str) -> Vec<NdCall<'_>> {
    let b = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 2 <= b.len() {
        if &b[i..i + 2] != b"nd" {
            i += 1;
            continue;
        }
        // Word boundary before `nd` — reject `grand(`, `round(`, `.rfind(`, etc.
        let boundary = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        let mut j = i + 2;
        while j < b.len() && (b[j] as char).is_whitespace() {
            j += 1;
        }
        if boundary && j < b.len() && b[j] == b'(' {
            j += 1;
            while j < b.len() && (b[j] as char).is_whitespace() {
                j += 1;
            }
            if j < b.len() && (b[j] == b'\'' || b[j] == b'"') {
                let q = b[j];
                j += 1;
                let start = j;
                while j < b.len() && b[j] != q {
                    j += 1;
                }
                if j < b.len() {
                    // Where the whole call ends, when it ends cleanly: past the closing quote,
                    // past any whitespace, on a `)`.
                    let mut close = j + 1;
                    while close < b.len() && (b[close] as char).is_whitespace() {
                        close += 1;
                    }
                    let end = (b.get(close) == Some(&b')')).then_some(close + 1);
                    out.push(NdCall {
                        start: i,
                        name_start: start,
                        name_end: j,
                        end,
                        name: &source[start..j],
                    });
                    i = j + 1;
                    continue;
                }
            }
        }
        i += 2;
    }
    out
}

/// One `globals.<name>` read [`scan_globals`] found.
pub struct GlobalRead<'a> {
    /// The whole term, `globals.` included — what the expression rewrite replaces.
    pub start: usize,
    pub end: usize,
    pub name: &'a str,
}

/// Scan `source` for `globals.<name>` reads, in source order. The same word-boundary rule
/// [`scan_nd_calls`] applies, so `myglobals.x` is not a reference; the name must be a valid
/// identifier, so a bare `globals.` and a digit-led name are not either.
///
/// A byte scan, not a parse: a match inside a string literal is read as a reference. That costs a
/// variable nothing else names, never a missed one.
pub fn scan_globals(source: &str) -> Vec<GlobalRead<'_>> {
    const PREFIX: &str = "globals.";
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(pos) = source[i..].find(PREFIX) {
        let start = i + pos;
        i = start + PREFIX.len();
        if start > 0 && is_ident(bytes[start - 1]) {
            continue;
        }
        let name_start = start + PREFIX.len();
        let mut end = name_start;
        while end < bytes.len() && is_ident(bytes[end]) {
            end += 1;
        }
        if end > name_start && !bytes[name_start].is_ascii_digit() {
            out.push(GlobalRead { start, end, name: &source[name_start..end] });
            i = end;
        }
    }
    out
}

/// Rewrite every `nd('name')` literal for which `rename(name)` returns `Some(new)`,
/// preserving the literal's quote style and every other byte of the source. Returns
/// `Some(new_source)` if any literal changed, else `None` (nothing to do). Used when a
/// referenced node is renamed: `nd('old')` follows to `nd('new')` across the graph.
///
/// This edits the AUTHORED source, which is the SSOT — the rewritten form the node runs is derived
/// from it and re-derived after every rename.
pub fn rewrite_nd_refs(source: &str, rename: impl Fn(&str) -> Option<String>) -> Option<String> {
    // Collect (start, end, replacement) then splice right-to-left so earlier byte offsets
    // stay valid as the string is edited.
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    for call in scan_nd_calls(source) {
        if let Some(new) = rename(call.name) {
            edits.push((call.name_start, call.name_end, new));
        }
    }
    if edits.is_empty() {
        return None;
    }
    let mut out = source.to_string();
    for (start, end, repl) in edits.into_iter().rev() {
        out.replace_range(start..end, &repl);
    }
    Some(out)
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
    /// A **required** slot must hold data when the node runs. The engine checks the slot's
    /// last-store — presence, never wiring — before `process` is invoked, and reports an error
    /// instead of running. So a required slot is one a node may read unconditionally; a
    /// non-required slot may be absent and the node handles that itself.
    pub required: bool,
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
    /// This type is a SOURCE: it makes frames on its own schedule rather than in answer to an
    /// input. The only pacing an author declares — everything downstream inherits its cadence
    /// through triggers, so a consumer never states a rate. Today it does exactly one thing:
    /// default `common.autotrigger` on (see [`with_common`]), and turn [`COMMON_DECLS`]'s carried
    /// `globals.default_ufreq` expression live — a source is what the patch rate is for, so it is
    /// paced by it out of the box while a consumer merely carries the source for the fx toggle.
    pub producer: bool,
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
