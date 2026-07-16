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
#[derive(Clone, Debug, PartialEq, Eq)]
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

// ---------------------------------------------------------------------------
// Tick I/O
// ---------------------------------------------------------------------------

/// The latest `Data` on each subscribed input slot (`None` if unwired / no frame
/// yet). Borrowed for the duration of a tick.
pub struct Inputs<'a> {
    slots: &'a IndexMap<&'static str, Option<Data>>,
}

impl<'a> Inputs<'a> {
    pub fn new(slots: &'a IndexMap<&'static str, Option<Data>>) -> Inputs<'a> {
        Inputs { slots }
    }
    pub fn get(&self, name: &str) -> Option<&Data> {
        self.slots.get(name).and_then(|o| o.as_ref())
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

// ---------------------------------------------------------------------------
// The node trait
// ---------------------------------------------------------------------------

pub trait Node: Send {
    /// One-time init; may fail terminally.
    fn setup(&mut self, _ctx: &mut NodeCtx) -> NodeResult {
        Ok(())
    }
    /// The tick body: read latest inputs, write outputs. Pure w.r.t. transport.
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, ctx: &mut NodeCtx) -> NodeResult;
    /// Apply a param edit to the node's own state.
    fn on_param_changed(&mut self, _key: &ParamKey, _v: &Param) -> NodeResult {
        Ok(())
    }
    /// Re-enumerate a `StringParam`'s options (device pickers).
    fn refresh_options(&mut self, _key: &ParamKey) -> Option<Vec<String>> {
        None
    }
    fn terminate(&mut self) {}
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
    pub default_params: fn() -> ParamGroups,
    pub isolation: Isolation,
    pub make: fn(&ParamGroups) -> Box<dyn Node>,
}

impl NodeManifest {
    /// A fresh output buffer seeded with this manifest's output slot names.
    pub fn output_buffer(&self) -> IndexMap<&'static str, Option<Data>> {
        self.outputs.iter().map(|o| (o.name, None)).collect()
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
    fn param_lookup() {
        let mut g = IndexMap::new();
        g.insert("x".to_string(), Param::float(1.0, 0.0, 2.0));
        let mut groups: ParamGroups = IndexMap::new();
        groups.insert("grp".to_string(), g);
        assert_eq!(param(&groups, "grp", "x").and_then(Param::as_f64), Some(1.0));
        assert!(param(&groups, "grp", "missing").is_none());
        assert!(param(&groups, "nogroup", "x").is_none());
    }

    struct Nop;
    impl Node for Nop {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            Ok(())
        }
    }
    fn nop_params() -> ParamGroups {
        ParamGroups::new()
    }
    fn nop_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(Nop)
    }
    static NOP_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_NodeTestNop",
            category: "test",
            doc: "",
            inputs: &[],
            outputs: NOP_OUT,
            default_params: nop_params,
            isolation: Isolation::InProcess,
            make: nop_make,
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
