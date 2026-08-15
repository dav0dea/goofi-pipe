//! The per-node runtime (spec §1, §2, §6): the wake loop's body, the three run paths, and a
//! node's faults.
//!
//! A node decides for itself when to run. There are exactly three reasons, all decided
//! node-locally: a frame on a `trigger_process` input (path A), `common.autotrigger` (path B), and
//! a value arriving in a bound non-`common` param's mailbox (path C). Nothing here consults
//! topology — **autotrigger is independent of input slots**, so there is no `wired` term and no
//! connected-trigger-input counter, and a node that declares no trigger input and leaves
//! autotrigger off never runs, which is correct.
//!
//! This module is standalone: it owns one node against a [`Transport`], and nothing in [`Graph`]
//! drives it yet.
//!
//! [`Graph`]: crate::Graph

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use goofi_core::{Data, Param};
use goofi_node::{Inputs, Node, NodeCtx, NodeManifest, Outputs, ParamGroups, ParamKey, Params, RunPolicy};
use indexmap::IndexMap;

mod mailbox;
mod wire;

pub use mailbox::{Binding, Mailbox};
pub use wire::{Control, EventId, MemoryTransport, ParamValue, ServiceName, Status, Transport, Var, VarName};

/// The scheduling namespace. A `common.*` param decides *when* a node runs, so it is resolved
/// before the gates are read and never inside a run (§1.1).
const COMMON: &str = "common";

/// `SETUP_RETRY_INTERVAL` in the wall-clock milliseconds a [`NodeFault`] carries, so the interval
/// is stated once for the whole engine rather than once per clock.
const SETUP_RETRY_MS: f64 = crate::SETUP_RETRY_INTERVAL * 1000.0;

/// What is wrong with a node. `None` is healthy.
///
/// Four variants because `entry_error` folds four sources, and wall-clock `f64` rather than
/// [`Instant`] because a fault is reported over the wire.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeFault {
    Setup { msg: String, since: f64, last_attempt: f64 },
    Process { msg: String, since: f64 },
    Boot { msg: String, since: f64 },
    Expr { key: ParamKey, msg: String, since: f64 },
}

impl NodeFault {
    pub fn msg(&self) -> &str {
        match self {
            NodeFault::Setup { msg, .. }
            | NodeFault::Process { msg, .. }
            | NodeFault::Boot { msg, .. }
            | NodeFault::Expr { msg, .. } => msg,
        }
    }
}

/// One node, its scheduling state, and its faults.
pub struct NodeRuntime {
    manifest: &'static NodeManifest,
    node: Box<dyn Node>,
    transport: Arc<dyn Transport>,

    /// Path B: the node always wants to run and paces itself. Lifted out of [`RunPolicy`] because
    /// it is the *gate* the three paths share while the policy is the *pacing*; one
    /// `RunPolicy::from_params` writes both, so they cannot disagree.
    pub autotrigger: bool,
    /// Paths A and C: something asked this node to run and it has not run since.
    pub trigger_pending: bool,
    pub run_policy: RunPolicy,
    pub last_run: Option<Instant>,
    /// Set by ANY arrival that can affect a `common.*` param, whatever path it came in on (§1.1).
    pub common_dirty: bool,

    /// The node's FULL params — the literal record overlaid with evaluated bindings. What
    /// `process()` reads and what `RunPolicy::from_params` is given.
    pub effective: ParamGroups,
    /// The SPARSE bound subset, which exists only as the wire projection in
    /// [`Status::ParamValues`]. Handing THIS to `RunPolicy::from_params` would silently default
    /// every absent key — which is why the two maps are named apart (§2).
    pub evaluated: IndexMap<ParamKey, Param>,
    pub bindings: IndexMap<ParamKey, Binding>,

    /// Latest-wins input cells, one per declared single input slot.
    pub inputs: IndexMap<&'static str, Option<Data>>,
    pub ctx: NodeCtx,

    pub fault: Option<NodeFault>,
    /// Binding errors are a MAP, not a fault variant: several bindings can be errored at once and
    /// each renders on its own inspector field. [`NodeFault::Expr`] is the derived node-level
    /// roll-up ([`Self::node_fault`]), not the record.
    pub binding_errors: HashMap<ParamKey, String>,
    /// When the binding-error set last changed — the roll-up's `since`, which the map itself has
    /// no room for.
    binding_errors_since: f64,
    initialized: bool,
}

impl NodeRuntime {
    /// Build a node from its manifest and initialize it: seed its params, then `setup()`. A
    /// failing `setup` leaves the node UNINITIALIZED with a [`NodeFault::Setup`] standing, and
    /// nothing runs against it until a retry succeeds.
    pub fn new(manifest: &'static NodeManifest, transport: Arc<dyn Transport>) -> NodeRuntime {
        let effective = goofi_node::with_common(manifest.default_params(), manifest);
        let run_policy = RunPolicy::from_params(&effective);
        let mut runtime = NodeRuntime {
            manifest,
            node: (manifest.factory)(),
            transport,
            autotrigger: run_policy.autotrigger,
            trigger_pending: false,
            run_policy,
            last_run: None,
            common_dirty: false,
            effective,
            evaluated: IndexMap::new(),
            bindings: IndexMap::new(),
            inputs: manifest.inputs.iter().filter(|s| !s.multi).map(|s| (s.name, None)).collect(),
            ctx: NodeCtx::new(),
            fault: None,
            binding_errors: HashMap::new(),
            binding_errors_since: 0.0,
            initialized: false,
        };
        runtime.initialize();
        runtime
    }

    // -----------------------------------------------------------------------
    // The gates (§2)
    // -----------------------------------------------------------------------

    /// Whether this wake runs `process()`. An autotriggering node always wants to; any other node
    /// runs when something triggered it; both are held to the rate cap.
    pub fn should_process(&self) -> bool {
        (self.autotrigger || self.trigger_pending) && self.rate_cap_elapsed()
    }

    /// How long to park, or `None` to park indefinitely. A node holding a pending trigger the cap
    /// refuses re-arms on cap release rather than parking with work in hand.
    pub fn next_wake(&self) -> Option<Duration> {
        (self.autotrigger || self.trigger_pending).then(|| self.cap_release())
    }

    fn rate_cap_elapsed(&self) -> bool {
        match self.run_policy.period() {
            None => true,
            Some(p) => self.last_run.is_none_or(|t| t.elapsed().as_secs_f64() >= p),
        }
    }

    /// How long until the cap admits another run. Zero when it already does — an uncapped node
    /// wakes immediately, which is what free-running means.
    fn cap_release(&self) -> Duration {
        match (self.run_policy.period(), self.last_run) {
            (Some(p), Some(last)) => Duration::from_secs_f64(p).saturating_sub(last.elapsed()),
            _ => Duration::ZERO,
        }
    }

    // -----------------------------------------------------------------------
    // The wake loop (§2)
    // -----------------------------------------------------------------------

    /// One iteration of the wake loop, minus the park: §3.3 makes a notification a pure hint — the
    /// truth is in the control mailbox and the latest-wins cells — so the drain never consults the
    /// event ids, and the loop that parks on [`Transport::wait`] wraps this rather than replacing
    /// it. Paths A and C arrive through the `deliver_*` doors until a real transport subscribes
    /// for them.
    pub fn run_once(&mut self) {
        self.drain_control();

        // §1.1 — pacing is resolved BEFORE the gates are read, whichever path dirtied it.
        if self.common_dirty {
            self.eval_common_bindings();
            self.run_policy = RunPolicy::from_params(&self.effective);
            self.autotrigger = self.run_policy.autotrigger;
            self.common_dirty = false;
        }

        if self.should_process() {
            // §2.1 — the non-common bindings, in the same breath as the run that reads them.
            self.eval_bindings();
            self.run();
        }
    }

    fn drain_control(&mut self) {
        for msg in self.transport.drain_control() {
            match msg {
                Control::SetParam { key, value } => self.set_param(key, value),
            }
        }
    }

    /// Write a param. A `Literal` on a bound param unbinds it (§3.4); an `Expr` binds it and is
    /// evaluated once here — the authoring moment — because without that a binding error can
    /// neither appear nor clear on a node that never runs (§2.1).
    pub fn set_param(&mut self, key: ParamKey, value: ParamValue) {
        match value {
            ParamValue::Literal(p) => {
                self.bindings.shift_remove(&key);
                self.evaluated.shift_remove(&key);
                self.clear_binding_error(&key);
                self.set_effective(&key, p.clone());
                if key.group == COMMON {
                    self.common_dirty = true;
                } else {
                    self.on_param_changed(&key, &p);
                }
            }
            ParamValue::Expr { source, vars, trigger } => {
                self.bindings.insert(key.clone(), Binding::new(source, vars, trigger));
                if key.group == COMMON {
                    self.common_dirty = true;
                } else {
                    self.eval_bindings_where(|k| *k == key);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // The three arrival paths (§1)
    // -----------------------------------------------------------------------

    /// Path A — a frame on an input slot. A `trigger_process` slot wakes the node; a reference
    /// slot updates the cell and nothing more.
    pub fn deliver_input(&mut self, slot: &str, frame: Data) {
        let Some(decl) = self.manifest.inputs.iter().find(|s| s.name == slot) else { return };
        self.inputs.insert(decl.name, Some(frame));
        if decl.trigger_process {
            self.trigger_pending = true;
        }
    }

    /// A `Var::Value` arrival — the graph resolving a global and delivering it inline (§5.2), the
    /// shape a `Control::SetParam` carries.
    pub fn deliver_expr_arrival(&mut self, key: ParamKey, value: Param) {
        self.deliver(key, value);
    }

    /// Path C — a `Var::Stream` mailbox write: a producer's frame landing in a bound variable.
    pub fn deliver_stream_arrival(&mut self, key: ParamKey, value: Param) {
        self.deliver(key, value);
    }

    /// §1.1's rule, stated ONCE and by key NAMESPACE rather than by arrival path: any arrival that
    /// can affect a `common.*` param re-paces the node **without** setting `trigger_pending`, and
    /// only a non-`common` binding that declares `trigger` fires path C. `trigger` is therefore
    /// ignored on a `common.*` key — every node's `common.max_frequency` declares it, and it means
    /// nothing there.
    ///
    /// Both `deliver_*` doors route through here because saying the rule per drain function is
    /// what leaves the stream path uncovered: a `common.autotrigger` bound to `nd('gate')` arrives
    /// there, and with autotrigger still false the node parks forever holding the value that would
    /// have started it.
    fn deliver(&mut self, key: ParamKey, value: Param) {
        let Some(binding) = self.bindings.get_mut(&key) else { return };
        binding.deliver(value);
        let trigger = binding.trigger;
        if key.group == COMMON {
            self.common_dirty = true;
        } else if trigger {
            self.trigger_pending = true;
        }
    }

    // -----------------------------------------------------------------------
    // Bindings (§2.1)
    // -----------------------------------------------------------------------

    /// The `common.*` bindings, evaluated on ARRIVAL because they decide whether a run happens at
    /// all — the deliberate exception to §2.1's evaluate-before-the-run rule.
    fn eval_common_bindings(&mut self) {
        self.eval_bindings_where(|k| k.group == COMMON);
    }

    /// Every other binding, evaluated immediately before the run that reads it.
    fn eval_bindings(&mut self) {
        self.eval_bindings_where(|k| k.group != COMMON);
    }

    fn eval_bindings_where(&mut self, want: impl Fn(&ParamKey) -> bool) {
        let keys: Vec<ParamKey> = self.bindings.keys().filter(|k| want(k)).cloned().collect();
        let mut errors: Vec<(ParamKey, Option<String>)> = Vec::new();
        let mut values_changed = false;
        for key in keys {
            match self.bindings[&key].evaluate() {
                Err(msg) => {
                    if self.binding_errors.get(&key) != Some(&msg) {
                        self.binding_errors.insert(key.clone(), msg.clone());
                        errors.push((key, Some(msg)));
                    }
                }
                // Nothing has arrived yet: the param's literal stands, and that is not an error.
                Ok(None) => {
                    if self.clear_binding_error(&key) {
                        errors.push((key, None));
                    }
                }
                Ok(Some(value)) => {
                    if self.clear_binding_error(&key) {
                        errors.push((key.clone(), None));
                    }
                    let previous = self.evaluated.insert(key.clone(), value.clone());
                    if previous.as_ref() == Some(&value) {
                        continue;
                    }
                    values_changed = true;
                    self.set_effective(&key, value.clone());
                    // The hook is the single source of truth for param→field, and the only way an
                    // evaluated value reaches a node's mirrored field. A `common.*` param has no
                    // field to mirror — it is the scheduler's, not the node's.
                    if key.group != COMMON {
                        self.on_param_changed(&key, &value);
                    }
                }
            }
        }
        if !errors.is_empty() {
            self.binding_errors_since = now_ms();
            self.transport.report(Status::BindingErrors { errors });
        }
        if values_changed {
            let evaluated = self.evaluated.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            self.transport.report(Status::ParamValues { evaluated });
        }
    }

    /// Run the param hook, recording a rejection or a panic as that binding's error — a node is
    /// third-party code and its hooks run where a panic would otherwise escape the loop.
    fn on_param_changed(&mut self, key: &ParamKey, value: &Param) {
        let result = crate::guard_lifecycle(|| self.node.on_param_changed(key, value))
            .unwrap_or_else(crate::fold_panic);
        if let Err(e) = result {
            self.binding_errors.insert(key.clone(), e.0.clone());
            self.binding_errors_since = now_ms();
            self.transport.report(Status::BindingErrors { errors: vec![(key.clone(), Some(e.0))] });
        }
    }

    fn clear_binding_error(&mut self, key: &ParamKey) -> bool {
        self.binding_errors.remove(key).is_some()
    }

    fn set_effective(&mut self, key: &ParamKey, value: Param) {
        self.effective.entry(key.group.clone()).or_default().insert(key.name.clone(), value);
    }

    // -----------------------------------------------------------------------
    // Running, and faults (§6)
    // -----------------------------------------------------------------------

    fn run(&mut self) {
        // The frame that asked for this run has been seen; holding it would fire the node twice
        // the moment it recovers. Consumed whether or not the run itself gets as far as `process`.
        self.trigger_pending = false;
        self.last_run = Some(Instant::now());
        if !self.ensure_initialized() {
            return;
        }
        let mut outputs = self.manifest.output_buffer();
        let result = {
            let inputs = Inputs::new(&self.inputs);
            let params = Params::new(&self.effective);
            let mut out = Outputs::new(&mut outputs);
            crate::guard_lifecycle(|| self.node.process(&inputs, &mut out, &mut self.ctx, &params))
                .unwrap_or_else(crate::fold_panic)
        };
        match result {
            Ok(()) => {
                // Clearing on success is safe because `process` is unreachable while a setup error
                // stands, so a clean run PROVES setup succeeded. It does NOT clear a binding
                // error, which only that binding evaluating successfully clears.
                self.set_fault(None);
                for (slot, frame) in outputs.iter() {
                    if let Some(frame) = frame {
                        self.transport.publish(slot, frame);
                    }
                }
            }
            Err(e) => self.set_fault(Some(NodeFault::Process { msg: e.0, since: now_ms() })),
        }
    }

    /// The initialization gate: a node whose `setup()` failed is UNINITIALIZED, so nothing runs
    /// against it until a retry succeeds. Paced by [`SETUP_RETRY_MS`] because a wake is not a user
    /// asking — it is one of however many the pacer admits.
    fn ensure_initialized(&mut self) -> bool {
        if self.initialized {
            return true;
        }
        if let Some(NodeFault::Setup { last_attempt, .. }) = &self.fault {
            if now_ms() - last_attempt < SETUP_RETRY_MS {
                return false;
            }
        }
        self.initialize();
        self.initialized
    }

    /// The param replay and `setup()` together, which are one unit — a retry re-runs both, on the
    /// same instance.
    fn initialize(&mut self) {
        let attempt = now_ms();
        match crate::seed_node(&mut *self.node, &self.effective, &mut self.ctx) {
            None => {
                self.initialized = true;
                self.set_fault(None);
            }
            Some(msg) => {
                self.set_fault(Some(NodeFault::Setup { msg, since: now_ms(), last_attempt: attempt }))
            }
        }
    }

    /// Install a fault, keeping `since` when nothing changed: the node stamps its own `since` when
    /// its fault CHANGES, and reports only transitions — so a process error recurring every run
    /// is one console line, not one per run.
    fn set_fault(&mut self, next: Option<NodeFault>) {
        let unchanged = match (&self.fault, &next) {
            (Some(current), Some(next)) => {
                std::mem::discriminant(current) == std::mem::discriminant(next)
                    && current.msg() == next.msg()
            }
            (None, None) => true,
            _ => false,
        };
        if unchanged {
            return;
        }
        self.fault = next;
        self.transport.report(Status::Fault { fault: self.fault.clone() });
    }

    /// The node-level roll-up the editor's badge draws: the standing fault, or — when there is
    /// none — the lowest-keyed binding error, which is `entry_error`'s precedence.
    pub fn node_fault(&self) -> Option<NodeFault> {
        if let Some(fault) = &self.fault {
            return Some(fault.clone());
        }
        self.binding_errors.iter().min_by(|a, b| a.0.cmp(b.0)).map(|(key, msg)| NodeFault::Expr {
            key: key.clone(),
            msg: msg.clone(),
            since: self.binding_errors_since,
        })
    }
}

/// Wall-clock milliseconds since the epoch — what a [`NodeFault`] timestamp is, because it travels
/// over the wire and [`Instant`] does not serialize.
fn now_ms() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Meta, Param, SlotType};
    use goofi_node::{
        default_factory, Isolation, NodeResult, OutputDecl, ParamDecl, SlotDecl,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn autotrigger_is_independent_of_input_slots() {
        // spec §1: if autotrigger is true the node ALWAYS wants to run and just rate-limits itself.
        // Whether it declares a trigger input, and whether that input is wired, does not enter into
        // it. There is no `wired` term and no connected_trigger_inputs counter.
        let mut r = fixture_with_trigger_input();
        r.autotrigger = true;
        r.trigger_pending = false;
        r.last_run = None;
        assert!(r.should_process(), "autotrigger runs with no arrival and an unwired input");
    }

    #[test]
    fn a_node_with_no_trigger_inputs_and_no_autotrigger_never_runs() {
        // spec §1: "and that is correct". The old !has_trigger_inputs free-run term is gone.
        let mut r = fixture_no_inputs();
        r.autotrigger = false;
        r.trigger_pending = false;
        assert!(!r.should_process());
        assert_eq!(r.next_wake(), None, "and it parks rather than spinning");
    }

    #[test]
    fn a_capped_node_holding_a_trigger_rearms_on_cap_release() {
        // The failure this prevents: next_wake() returning None while trigger_pending is set parks
        // the node forever with work in hand.
        let mut r = fixture_no_inputs();
        r.autotrigger = false;
        r.trigger_pending = true;
        r.run_policy.max_frequency = 10.0;
        r.last_run = Some(now_minus_ms(10)); // 90ms still to wait
        assert!(!r.should_process(), "the cap refuses it");
        let w = r.next_wake().expect("but it MUST re-arm");
        assert!(w > Duration::ZERO && w <= Duration::from_millis(90));
    }

    #[test]
    fn a_common_arrival_repaces_without_running() {
        // spec §1.1: re-pacing is not a reason to run. A producer runs anyway on its own schedule;
        // a consumer must not fire because a global changed.
        let mut r = fixture_no_inputs();
        r.autotrigger = false;
        r.deliver_expr_arrival(ParamKey::new("common", "max_frequency"), Param::float(60.0, 0.0, 100.0));
        assert!(!r.trigger_pending, "common.* never sets trigger_pending");
        r.run_once();
        assert_eq!(r.run_policy.max_frequency, 60.0, "but the policy IS re-derived");
    }

    #[test]
    fn a_common_toggle_arriving_on_a_stream_var_still_repaces() {
        // The bug this pins: stating the rule by drain FUNCTION instead of key NAMESPACE leaves the
        // Var::Stream path uncovered — a common.autotrigger bound to nd('gate') arrives in
        // drain_expr, and with autotrigger still false next_wake() is None, so the node parks
        // forever holding the value that would have started it.
        let mut r = fixture_no_inputs();
        r.autotrigger = false;
        assert_eq!(r.next_wake(), None, "parked");
        r.deliver_stream_arrival(ParamKey::new("common", "autotrigger"), Param::boolean(true));
        r.run_once();
        assert!(r.autotrigger, "the toggle landed");
        assert!(r.next_wake().is_some(), "and the node is reachable again");
    }

    #[test]
    fn a_clean_run_clears_setup_but_not_expr() {
        // spec §6: process() is unreachable while a setup error stands, so a clean run PROVES setup
        // succeeded. Expr is different — only a successful re-evaluation of that binding clears it.
        let mut r = fixture_no_inputs();
        r.fault = Some(NodeFault::Setup { msg: "boom".into(), since: 0.0, last_attempt: 0.0 });
        r.run_once();
        assert!(r.fault.is_none());

        r.binding_errors.insert(ParamKey::new("osc", "freq"), "bad ref".into());
        r.run_once();
        assert!(!r.binding_errors.is_empty(), "a clean process does not fix a broken expression");
    }

    #[test]
    fn several_bindings_can_be_errored_at_once() {
        // spec §6: binding errors are a MAP, not a variant — each renders on its own inspector field.
        let mut r = fixture_no_inputs();
        r.binding_errors.insert(ParamKey::new("osc", "freq"), "a".into());
        r.binding_errors.insert(ParamKey::new("osc", "amp"), "b".into());
        assert_eq!(r.binding_errors.len(), 2);
        assert!(matches!(r.node_fault(), Some(NodeFault::Expr { .. })), "rolled up for the node badge");
    }

    #[test]
    fn a_stream_arrival_repaces_a_consumer_without_ever_running_it() {
        // The other half of the two `common` tests above, on a node whose autotrigger is false in
        // its params and not merely in a poked field: the arrival must land (a value the fixture
        // did not already hold) and must not run the node.
        let (mut r, t) = consumer_fixture();
        r.deliver_stream_arrival(ParamKey::new("common", "max_frequency"), Param::float(25.0, 0.0, 100.0));
        r.run_once();
        assert_eq!(r.run_policy.max_frequency, 25.0, "the delivered value is what re-paced it");
        assert!(!r.autotrigger, "and a consumer is still a consumer");
        assert!(t.published().is_empty(), "a global changing never fires a consumer");
    }

    #[test]
    fn a_frame_on_a_trigger_slot_wakes_the_node_and_a_reference_slot_does_not() {
        // Path A. `trigger_process` is the whole of it — a reference input is read when the node
        // runs for some other reason, never a reason of its own.
        let mut r = fixture_with_trigger_input();
        r.deliver_input("ref", frame());
        assert!(!r.trigger_pending, "a reference input is not a trigger");
        r.deliver_input("in", frame());
        assert!(r.trigger_pending);
        assert!(r.should_process(), "even with autotrigger off");
    }

    #[test]
    fn a_bound_param_triggers_on_arrival_and_never_on_re_evaluation() {
        // spec §2.1: an ARRIVAL is what triggers; evaluation is what runs. The old engine set the
        // flag on every evaluation, which pinned it on permanently for an always-due binding — so
        // the distinction has to be structural, not a rate gate or a changed-comparison.
        let (mut r, _t) = consumer_fixture();
        r.set_param(ParamKey::new("osc", "freq"), stream_expr(true));
        r.deliver_stream_arrival(ParamKey::new("osc", "freq"), Param::float(3.0, 0.0, 10.0));
        assert!(r.trigger_pending, "the arrival triggered it");
        r.run_once();
        assert!(!r.trigger_pending, "the run consumed it");
        r.run_once();
        assert!(!r.trigger_pending, "re-evaluating the same value is not a new arrival");
    }

    #[test]
    fn a_failing_setup_gates_process_and_stands() {
        // A node whose `setup()` failed is uninitialized: not a run, not an output, and the fault
        // it reported is the one that stays until a retry succeeds.
        let t = Arc::new(MemoryTransport::default());
        let mut r = NodeRuntime::new(&BAD_SETUP, t.clone());
        assert!(matches!(r.fault, Some(NodeFault::Setup { .. })), "the failure is the node's fault");
        r.run_once();
        assert!(matches!(r.fault, Some(NodeFault::Setup { .. })));
        assert!(t.published().is_empty(), "process is unreachable while a setup error stands");
    }

    #[test]
    fn a_recurring_process_error_is_reported_once() {
        // §6.2: the node reports TRANSITIONS, so the status worker needs no diffing — and the
        // console does not repaint the same line at the node's run rate.
        let t = Arc::new(MemoryTransport::default());
        let mut r = NodeRuntime::new(&BAD_PROCESS, t.clone());
        r.run_once();
        r.run_once();
        let faults: Vec<_> = t.reported().into_iter().filter(|s| matches!(s, Status::Fault { .. })).collect();
        assert_eq!(faults.len(), 1, "two failing runs, one transition");
        assert!(matches!(&r.fault, Some(NodeFault::Process { msg, .. }) if msg == "no"));
    }

    #[test]
    fn a_binding_error_reaches_the_graph_and_clears_on_recovery() {
        // The map is per-binding, and both edges are reported: an error that cannot clear is a node
        // stuck showing a failure it has recovered from.
        let (mut r, t) = consumer_fixture();
        let key = ParamKey::new("osc", "freq");
        r.set_param(
            key.clone(),
            ParamValue::Expr {
                source: "__v0".to_string(),
                vars: vec![Var::Missing { name: "__v0".to_string(), reason: "no node named `ghost`".to_string() }],
                trigger: true,
            },
        );
        assert_eq!(r.binding_errors.get(&key).map(String::as_str), Some("no node named `ghost`"));
        assert!(matches!(r.node_fault(), Some(NodeFault::Expr { .. })));

        r.deliver_stream_arrival(key.clone(), Param::float(3.0, 0.0, 10.0));
        r.run_once();
        assert!(r.binding_errors.is_empty(), "the value arrived, so the reference resolved");
        let errors: Vec<_> = t.reported().into_iter().filter(|s| matches!(s, Status::BindingErrors { .. })).collect();
        assert_eq!(errors.len(), 2, "both edges reported");
    }

    // -----------------------------------------------------------------------
    // Fixtures
    // -----------------------------------------------------------------------

    /// A node with no input slots. It is a PRODUCER, so it runs on its own schedule and the fault
    /// tests can observe a run.
    ///
    /// Both `common` gates are BOUND — an arrival needs a mailbox to land in — and both declare
    /// `trigger: true`, which is what the universal declaration carries and what makes "`trigger`
    /// is inert on `common.*`" observable rather than assumed. The autotrigger binding starts
    /// holding `false`: a fixture whose mailbox already agrees with the value under test cannot
    /// tell a landed arrival from an ignored one. And the runtime starts SETTLED, because a dirty
    /// flag left over from wiring would mask the very arrival these tests are about.
    fn fixture() -> (NodeRuntime, Arc<MemoryTransport>) {
        let transport = Arc::new(MemoryTransport::default());
        let mut r = NodeRuntime::new(&PRODUCER, transport.clone());
        r.set_param(ParamKey::new("common", "max_frequency"), stream_expr(true));
        r.set_param(ParamKey::new("common", "autotrigger"), stream_expr(true));
        r.deliver_stream_arrival(ParamKey::new("common", "autotrigger"), Param::boolean(false));
        r.common_dirty = false;
        (r, transport)
    }

    fn fixture_no_inputs() -> NodeRuntime {
        fixture().0
    }

    /// The same, on a node that is not a producer: its `common.autotrigger` is false in its params
    /// and not merely in a poked field.
    fn consumer_fixture() -> (NodeRuntime, Arc<MemoryTransport>) {
        let transport = Arc::new(MemoryTransport::default());
        let mut r = NodeRuntime::new(&CONSUMER, transport.clone());
        r.set_param(ParamKey::new("common", "max_frequency"), stream_expr(true));
        r.common_dirty = false;
        (r, transport)
    }

    /// A node declaring one trigger input and one reference input, neither wired to anything.
    fn fixture_with_trigger_input() -> NodeRuntime {
        NodeRuntime::new(&TRIGGERED, Arc::new(MemoryTransport::default()))
    }

    /// The single-variable identity binding §5.3's rewrite produces for `globals.default_ufreq`.
    fn stream_expr(trigger: bool) -> ParamValue {
        ParamValue::Expr {
            source: "__v0".to_string(),
            vars: vec![Var::Stream { name: "__v0".to_string(), service: "svc".to_string(), event_id: 65 }],
            trigger,
        }
    }

    fn now_minus_ms(ms: u64) -> Instant {
        Instant::now() - Duration::from_millis(ms)
    }

    fn frame() -> Data {
        Data::string("x", Meta::empty())
    }

    // -----------------------------------------------------------------------
    // Test nodes
    // -----------------------------------------------------------------------

    #[derive(Default)]
    struct Emit;
    impl Node for Emit {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            out.set("out", frame());
            Ok(())
        }
    }

    #[derive(Default)]
    struct BadSetup;
    impl Node for BadSetup {
        fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Err("no device".into())
        }
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            out.set("out", frame());
            Ok(())
        }
    }

    #[derive(Default)]
    struct BadProcess;
    impl Node for BadProcess {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Err("no".into())
        }
    }

    static OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::String }];
    static NO_PARAMS: &[ParamDecl] = &[];
    static SLOTS: &[SlotDecl] = &[
        SlotDecl { name: "in", kind: SlotType::String, trigger_process: true, multi: false, required: false },
        SlotDecl { name: "ref", kind: SlotType::String, trigger_process: false, multi: false, required: false },
    ];

    static PRODUCER: NodeManifest = manifest("_RuntimeProducer", &[], true, default_factory::<Emit>);
    static CONSUMER: NodeManifest = manifest("_RuntimeConsumer", &[], false, default_factory::<Emit>);
    static TRIGGERED: NodeManifest = manifest("_RuntimeTriggered", SLOTS, false, default_factory::<Emit>);
    static BAD_SETUP: NodeManifest = manifest("_RuntimeBadSetup", &[], true, default_factory::<BadSetup>);
    static BAD_PROCESS: NodeManifest = manifest("_RuntimeBadProcess", &[], true, default_factory::<BadProcess>);

    const fn manifest(
        type_name: &'static str,
        inputs: &'static [SlotDecl],
        producer: bool,
        factory: fn() -> Box<dyn Node>,
    ) -> NodeManifest {
        NodeManifest {
            type_name,
            category: "test",
            doc: "",
            inputs,
            outputs: OUT,
            params: NO_PARAMS,
            isolation: Isolation::InProcess,
            producer,
            factory,
        }
    }
}
