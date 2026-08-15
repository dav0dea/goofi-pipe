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

    /// Path B lives in `run_policy.autotrigger`, beside the cap that paces it — one value derived
    /// by one `RunPolicy::from_params`, so there is nothing to keep in step.
    ///
    /// Paths A and C: something asked this node to run and it has not run since.
    pub trigger_pending: bool,
    pub run_policy: RunPolicy,
    pub last_run: Option<Instant>,
    /// Set by ANY arrival that can affect a `common.*` param, whatever path it came in on (§1.1).
    pub common_dirty: bool,

    /// The param RECORD: literals only, which is what the `.gfi` persists and what a broken or
    /// not-yet-arrived binding falls back to (§2.1). Kept apart from `effective` because an
    /// evaluated value would otherwise erase the number the user authored.
    pub literals: ParamGroups,
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
            trigger_pending: false,
            run_policy,
            last_run: None,
            common_dirty: false,
            literals: effective.clone(),
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
        (self.run_policy.autotrigger || self.trigger_pending) && self.rate_cap_elapsed()
    }

    /// How long to park, or `None` to park indefinitely. A node holding a pending trigger the cap
    /// refuses re-arms on cap release rather than parking with work in hand.
    pub fn next_wake(&self) -> Option<Duration> {
        (self.run_policy.autotrigger || self.trigger_pending).then(|| self.cap_release())
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
        // §5.2: a re-send carrying a resolved value IS an arrival — that is how a globals edit
        // reaches a bound param — while binding a bare `nd()` reference only subscribes.
        let triggering = match &value {
            ParamValue::Literal(_) => false,
            ParamValue::Expr { vars, trigger, .. } => {
                *trigger && vars.iter().any(|v| matches!(v, Var::Value { .. }))
            }
        };
        // The record moves FIRST, so the initialization retry below replays the NEW value rather
        // than the one that broke `setup()`.
        let literal = match value {
            ParamValue::Literal(p) => {
                self.bindings.shift_remove(&key);
                self.evaluated.shift_remove(&key);
                let cleared = self.record_binding_error(&key, None);
                self.report_binding_errors(cleared.into_iter().collect());
                self.set_literal(&key, p.clone());
                Some(p)
            }
            ParamValue::Expr { source, vars, trigger } => {
                let binding = Binding::new(source, vars, trigger);
                match self.bindings.get_mut(&key) {
                    Some(existing) => existing.rebind(binding),
                    None => {
                        self.bindings.insert(key.clone(), binding);
                    }
                }
                // A `common.*` binding is evaluated by the pacing pass instead, before the gates.
                if key.group != COMMON {
                    self.eval_bindings_where(|k| *k == key);
                }
                None
            }
        };
        self.arrived(&key, triggering);
        // §5.1 + D3: a param write is an INTERACTION, and an interaction retries the initialization
        // first — a node whose `setup()` failed on a bad param has no other way back when it never
        // runs, since `run()` is the only other caller of the gate. Unthrottled, unlike a wake:
        // this is a user asking, not one of however many the pacer admits.
        let was_initialized = self.initialized;
        let healed = self.ensure_initialized() && !was_initialized;
        // `initialize` replays the whole record through `on_param_changed`, so a retry that
        // succeeded has already delivered this edit — notifying again would double-apply it. And
        // an UNINITIALIZED node hears nothing at all (D3).
        if let Some(p) = literal {
            if self.initialized && !healed && key.group != COMMON {
                self.on_param_changed(&key, &p);
            }
        }
    }

    // -----------------------------------------------------------------------
    // The three arrival paths (§1)
    // -----------------------------------------------------------------------

    /// Path A — a frame on an input slot. A `trigger_process` slot wakes the node; a reference
    /// slot updates the cell and nothing more.
    ///
    /// Single slots only: a `multi` slot keeps one cell per WIRE, ordered by that wire's position
    /// in the slot's service list, so it arrives with the transport that subscribes per wire.
    pub fn deliver_input(&mut self, slot: &str, frame: Data) {
        let Some(decl) = self.manifest.inputs.iter().find(|s| s.name == slot && !s.multi) else {
            return;
        };
        self.inputs.insert(decl.name, Some(frame));
        if decl.trigger_process {
            self.trigger_pending = true;
        }
    }

    /// Path C on the DATA plane — a producer's frame landing in a bound variable's mailbox. Its
    /// sibling is [`Self::set_param`] on the control plane, which carries the same three §1.1
    /// arrivals a `Control` message can (a literal, a binding, a global the graph resolved inline);
    /// §5.3 lands both in the same mailbox, so the two planes differ in how a value gets here and
    /// in nothing else.
    pub fn deliver_arrival(&mut self, key: ParamKey, value: Param) {
        let Some(binding) = self.bindings.get_mut(&key) else {
            // A value for a param this node has no binding on: a re-send that raced an unbind. The
            // graph is the only sender and it stops sending on unbind, so dropping it converges.
            return;
        };
        binding.deliver(value);
        let trigger = binding.trigger;
        self.arrived(&key, trigger);
    }

    /// What an arrival does to the schedule — §1.1's rule, stated ONCE and by key NAMESPACE rather
    /// than per arrival path. Both planes call it, which is the point: a `common.autotrigger` bound
    /// to `nd('gate')` arrives on the data plane while the same key bound to a global arrives on
    /// the control plane, and covering only one leaves the node parked forever holding the value
    /// that would have started it.
    fn arrived(&mut self, key: &ParamKey, trigger: bool) {
        if key.group == COMMON {
            // Re-pacing is not a reason to run: a producer runs anyway on its own schedule, and a
            // consumer must not fire because a global changed. `trigger` is therefore IGNORED
            // here — every node's `common.max_frequency` declares it, and it means nothing on this
            // namespace.
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
            // A binding with nothing in its mailbox yet is not an error — the literal simply
            // stands. A binding that cannot be evaluated at all is, and it falls back to the same
            // literal for this run.
            let evaluated = match self.bindings[&key].evaluate() {
                Ok(value) => {
                    errors.extend(self.record_binding_error(&key, None));
                    value
                }
                Err(msg) => {
                    errors.extend(self.record_binding_error(&key, Some(msg)));
                    None
                }
            };
            values_changed |= match &evaluated {
                Some(value) => self.evaluated.insert(key.clone(), value.clone()).as_ref() != Some(value),
                None => self.evaluated.shift_remove(&key).is_some(),
            };
            let Some(next) = evaluated.or_else(|| self.literal(&key)) else { continue };
            if goofi_node::param(&self.effective, &key.group, &key.name) == Some(&next) {
                continue;
            }
            self.set_effective(&key, next.clone());
            // The hook is the single source of truth for param→field, and the only way an
            // evaluated value reaches a node's mirrored field. A `common.*` param has no field to
            // mirror — it is the scheduler's, not the node's — and an UNINITIALIZED node hears
            // nothing at all (D3); the value is in the record, so the retry's replay delivers it
            // when `setup` finally succeeds.
            if key.group != COMMON && self.initialized {
                self.on_param_changed(&key, &next);
            }
        }
        self.report_binding_errors(errors);
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
            let recorded = self.record_binding_error(key, Some(e.0));
            self.report_binding_errors(recorded.into_iter().collect());
        }
    }

    /// Record or clear a binding's error, answering with the wire entry when it CHANGED. The map
    /// is the record and `Status::BindingErrors` is a delta, so an unchanged error is silent and a
    /// cleared one is announced — an error that cannot clear leaves a node showing a failure it
    /// has recovered from.
    fn record_binding_error(
        &mut self,
        key: &ParamKey,
        msg: Option<String>,
    ) -> Option<(ParamKey, Option<String>)> {
        match msg {
            Some(msg) if self.binding_errors.get(key) == Some(&msg) => None,
            Some(msg) => {
                self.binding_errors.insert(key.clone(), msg.clone());
                Some((key.clone(), Some(msg)))
            }
            None => self.binding_errors.remove(key).map(|_| (key.clone(), None)),
        }
    }

    fn report_binding_errors(&mut self, errors: Vec<(ParamKey, Option<String>)>) {
        if errors.is_empty() {
            return;
        }
        self.binding_errors_since = now_ms();
        self.transport.report(Status::BindingErrors { errors });
    }

    fn literal(&self, key: &ParamKey) -> Option<Param> {
        goofi_node::param(&self.literals, &key.group, &key.name).cloned()
    }

    /// Write the param RECORD, and `effective` with it — for an unbound param they are one number.
    fn set_literal(&mut self, key: &ParamKey, value: Param) {
        self.literals.entry(key.group.clone()).or_default().insert(key.name.clone(), value.clone());
        self.set_effective(key, value);
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
        if !self.ensure_initialized_paced() {
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

    /// The initialization gate (D3): a node whose `setup()` failed is UNINITIALIZED, so nothing
    /// runs against it — not `process`, not a param callback — and any interaction retries the
    /// initialization first. Answers whether the node may be run against.
    fn ensure_initialized(&mut self) -> bool {
        if self.initialized {
            return true;
        }
        self.initialize();
        self.initialized
    }

    /// The same gate on a WAKE, which is not a user asking but one of however many the pacer
    /// admits — so the retry is paced by [`SETUP_RETRY_MS`], and every attempt restarts the window.
    fn ensure_initialized_paced(&mut self) -> bool {
        if self.initialized {
            return true;
        }
        if let Some(NodeFault::Setup { last_attempt, .. }) = &self.fault {
            if now_ms() - last_attempt < SETUP_RETRY_MS {
                return false;
            }
        }
        self.ensure_initialized()
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
    ///
    /// An unchanged fault still moves the parts of the RECORD that are not the transition.
    /// `last_attempt` is one: it paces the next retry, so freezing it at the first failure turns
    /// the backoff off entirely and the node re-attempts at its wake rate. `since` is precisely
    /// the part that must not move.
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
            if let (
                Some(NodeFault::Setup { last_attempt, .. }),
                Some(NodeFault::Setup { last_attempt: attempted, .. }),
            ) = (&mut self.fault, &next)
            {
                *last_attempt = *attempted;
            }
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
        default_factory, Isolation, NodeResult, OutputDecl, ParamDecl, ParamSpec, SlotDecl,
    };
    use std::cell::{Cell, RefCell};
    use std::time::{Duration, Instant};

    #[test]
    fn autotrigger_is_independent_of_input_slots() {
        // spec §1: if autotrigger is true the node ALWAYS wants to run and just rate-limits itself.
        // Whether it declares a trigger input, and whether that input is wired, does not enter into
        // it. There is no `wired` term and no connected_trigger_inputs counter.
        let mut r = fixture_with_trigger_input();
        r.run_policy.autotrigger = true;
        r.trigger_pending = false;
        r.last_run = None;
        assert!(r.should_process(), "autotrigger runs with no arrival and an unwired input");
    }

    #[test]
    fn a_node_with_no_trigger_inputs_and_no_autotrigger_never_runs() {
        // spec §1: "and that is correct". The old !has_trigger_inputs free-run term is gone.
        let mut r = fixture_no_inputs();
        r.run_policy.autotrigger = false;
        r.trigger_pending = false;
        assert!(!r.should_process());
        assert_eq!(r.next_wake(), None, "and it parks rather than spinning");
    }

    #[test]
    fn a_capped_node_holding_a_trigger_rearms_on_cap_release() {
        // The failure this prevents: next_wake() returning None while trigger_pending is set parks
        // the node forever with work in hand.
        let mut r = fixture_no_inputs();
        r.run_policy.autotrigger = false;
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
        let (mut r, t) = fixture();
        r.run_policy.autotrigger = false;
        r.set_param(ParamKey::new("common", "max_frequency"), value_expr(Param::float(60.0, 0.0, 100.0), true));
        assert!(!r.trigger_pending, "common.* never sets trigger_pending");
        r.run_once();
        assert_eq!(r.run_policy.max_frequency, 60.0, "but the policy IS re-derived");
        assert!(published(&t).is_empty(), "and the node did not run");
    }

    #[test]
    fn a_common_toggle_arriving_on_a_stream_var_still_repaces() {
        // The bug this pins: stating the rule by drain FUNCTION instead of key NAMESPACE leaves the
        // Var::Stream path uncovered — a common.autotrigger bound to nd('gate') arrives in
        // drain_expr, and with autotrigger still false next_wake() is None, so the node parks
        // forever holding the value that would have started it.
        let mut r = fixture_no_inputs();
        r.run_policy.autotrigger = false;
        assert_eq!(r.next_wake(), None, "parked");
        r.deliver_arrival(ParamKey::new("common", "autotrigger"), Param::boolean(true));
        r.run_once();
        assert!(r.run_policy.autotrigger, "the toggle landed");
        assert!(r.next_wake().is_some(), "and the node is reachable again");
    }

    #[test]
    fn a_clean_run_clears_setup_but_not_expr() {
        // spec §6: process() is unreachable while a setup error stands, so a clean run PROVES setup
        // succeeded. Expr is different — only a successful re-evaluation of that binding clears it.
        //
        // The fault is DRIVEN by a failing `setup`, never constructed: an initialized node carrying
        // a Setup fault is a state production cannot reach, and underneath one `process` runs.
        let t = Arc::new(MemoryTransport::default());
        let mut r = NodeRuntime::new(&FLAKY_SETUP, t.clone());
        assert!(matches!(r.fault, Some(NodeFault::Setup { .. })), "the first attempt failed");
        r.run_once();
        assert!(published(&t).is_empty(), "and nothing ran while it stood");

        expire_setup_backoff(&mut r);
        r.run_once();
        assert!(r.fault.is_none());
        assert_eq!(published(&t), ["out: ok"], "the retry succeeded and the run went through");

        let key = ParamKey::new("cfg", "scale");
        r.set_param(key.clone(), missing_expr("no node named `lfo`"));
        r.run_once();
        assert_eq!(published(&t).len(), 2, "the run happened");
        assert!(r.binding_errors.contains_key(&key), "a clean process does not fix a broken expression");
    }

    #[test]
    fn a_recovering_run_clears_a_process_fault() {
        // The other half of "a clean run clears Setup/Process/Boot": a node that failed once and
        // then worked must stop drawing errored, and both edges reach the console.
        let t = Arc::new(MemoryTransport::default());
        let mut r = NodeRuntime::new(&FLAKY_PROCESS, t.clone());
        r.run_once();
        assert!(matches!(r.fault, Some(NodeFault::Process { .. })));
        assert!(published(&t).is_empty(), "a failing run emits nothing");

        r.run_once();
        assert!(r.fault.is_none(), "the next clean run clears it");
        assert_eq!(published(&t), ["out: run 2"]);
        assert_eq!(fault_reports(&t), [Some("boom".to_string()), None], "both transitions reported");
    }

    #[test]
    fn several_bindings_can_be_errored_at_once() {
        // spec §6: binding errors are a MAP, not a variant — each renders on its own inspector
        // field. Driven through the binding path, because a map filled by hand cannot show that
        // the code keeps more than one; and the roll-up is read by key, because `matches!` on the
        // variant alone would accept any of them.
        let (mut r, _t) = consumer_fixture();
        r.set_param(ParamKey::new("osc", "freq"), missing_expr("no node named `lfo`"));
        r.set_param(ParamKey::new("osc", "amp"), missing_expr("no node named `env`"));
        assert_eq!(r.binding_errors.len(), 2);
        match r.node_fault() {
            Some(NodeFault::Expr { key, msg, .. }) => {
                assert_eq!(key, ParamKey::new("osc", "amp"), "the lowest key, as entry_error orders them");
                assert_eq!(msg, "no node named `env`");
            }
            other => panic!("expected a rolled-up Expr fault for the badge, got {other:?}"),
        }
    }

    #[test]
    fn a_stream_arrival_repaces_a_consumer_without_ever_running_it() {
        // The other half of the two `common` tests above, on a node whose autotrigger is false in
        // its params and not merely in a poked field: the arrival must land (a value the fixture
        // did not already hold) and must not run the node.
        let (mut r, t) = consumer_fixture();
        r.deliver_arrival(ParamKey::new("common", "max_frequency"), Param::float(25.0, 0.0, 100.0));
        r.run_once();
        assert_eq!(r.run_policy.max_frequency, 25.0, "the delivered value is what re-paced it");
        assert!(!r.run_policy.autotrigger, "and a consumer is still a consumer");
        assert!(t.published().is_empty(), "a global changing never fires a consumer");
    }

    #[test]
    fn a_frame_on_a_trigger_slot_wakes_the_node_and_a_reference_slot_does_not() {
        // Path A. `trigger_process` is the whole of the WAKING; both cells are read when the run
        // happens, which is what makes a reference input worth holding at all. The node echoes
        // what it saw, so a `deliver_input` that never wrote a cell cannot pass this.
        let (mut r, t) = triggered_fixture();
        r.deliver_input("ref", text_frame("R"));
        assert!(!r.trigger_pending, "a reference input is not a trigger");
        assert!(!r.should_process());

        r.deliver_input("in", text_frame("A"));
        assert!(r.trigger_pending);
        assert!(r.should_process(), "even with autotrigger off");
        r.run_once();
        assert_eq!(published(&t), ["out: A|R"], "both cells reached process()");
    }

    #[test]
    fn a_run_publishes_the_frame_it_just_produced() {
        // The oracle every "did it run?" assertion leans on: emptiness alone cannot tell a node
        // that did not run from a publish path that drops what it is given.
        let (mut r, t) = fixture();
        r.run_once();
        r.run_once();
        assert_eq!(published(&t), ["out: run 1", "out: run 2"], "each run's own frame, in order");
    }

    #[test]
    fn a_multi_slot_frame_is_refused_rather_than_half_served() {
        // A multi slot keeps one cell per WIRE, ordered by that wire's position in the slot's
        // service list. Taking the single-source cell for it would read back as the whole slot.
        let mut r = NodeRuntime::new(&MULTI_IN, Arc::new(MemoryTransport::default()));
        r.deliver_input("many", text_frame("A"));
        assert!(!r.inputs.contains_key("many"), "no single-source cell was minted for it");
        assert!(!r.trigger_pending, "and it did not wake the node");
    }

    #[test]
    fn a_bound_param_triggers_on_arrival_and_never_on_re_evaluation() {
        // spec §2.1: an ARRIVAL is what triggers; evaluation is what runs. The old engine set the
        // flag on every evaluation, which pinned it on permanently for an always-due binding — so
        // the distinction has to be structural, not a rate gate or a changed-comparison.
        let (mut r, _t) = consumer_fixture();
        r.set_param(ParamKey::new("osc", "freq"), stream_expr(true));
        r.deliver_arrival(ParamKey::new("osc", "freq"), Param::float(3.0, 0.0, 10.0));
        assert!(r.trigger_pending, "the arrival triggered it");
        r.run_once();
        assert!(!r.trigger_pending, "the run consumed it");
        r.run_once();
        assert!(!r.trigger_pending, "re-evaluating the same value is not a new arrival");
    }

    #[test]
    fn a_binding_with_trigger_off_lands_its_value_without_waking_the_node() {
        // §5.2: on a non-`common` key the binding's `trigger` flag is what decides path C. It also
        // separates ARRIVAL from EVALUATION — the old engine set the flag whenever a binding
        // evaluated, which for an always-due expression pinned it on forever.
        let (mut r, t) = consumer_fixture();
        let key = ParamKey::new("cfg", "scale");
        r.set_param(key.clone(), value_expr(Param::float(3.0, 0.0, 1000.0), false));
        assert_eq!(effective_f64(&r, &key), Some(3.0), "the value landed");
        assert!(!r.trigger_pending, "and nothing asked the node to run");
        r.run_once();
        assert!(published(&t).is_empty());

        r.set_param(key, value_expr(Param::float(4.0, 0.0, 1000.0), true));
        assert!(r.trigger_pending, "the same arrival with `trigger` on IS path C");
    }

    #[test]
    fn the_evaluated_values_are_projected_to_the_graph() {
        // `Status::ParamValues` is the source for today's `param_values` event, and it carries the
        // SPARSE bound subset — never the full param record.
        let (mut r, t) = consumer_fixture();
        let key = ParamKey::new("cfg", "scale");
        r.set_param(key.clone(), value_expr(Param::float(3.0, 0.0, 1000.0), false));
        assert_eq!(param_value_reports(&t), [vec![(key, Param::float(3.0, 0.0, 1000.0))]]);
    }

    #[test]
    fn a_wake_drains_the_control_mailbox() {
        // The graph writes control, the node reads it on its next wake, and nothing else connects
        // the two — a message the drain never takes is a param edit the node never hears.
        let (mut r, t) = consumer_fixture();
        let key = ParamKey::new("cfg", "scale");
        t.send(Control::SetParam {
            key: key.clone(),
            value: ParamValue::Literal(Param::float(7.0, 0.0, 1000.0)),
        });
        assert_eq!(effective_f64(&r, &key), Some(1.0), "not before the wake");
        r.run_once();
        assert_eq!(effective_f64(&r, &key), Some(7.0));
    }

    #[test]
    fn a_period_authored_in_seconds_is_read_as_a_rate() {
        // `frequency_mode` is a pure input convention: the scheduler only ever reasons in Hz, so
        // both spellings normalize before they reach the cap. Setting a literal on a bound
        // `common` key also unbinds it, which is the same `common` write path.
        let (mut r, _t) = consumer_fixture();
        r.set_param(
            ParamKey::new("common", "frequency_mode"),
            ParamValue::Literal(Param::str_free("seconds-per-update")),
        );
        r.set_param(ParamKey::new("common", "max_frequency"), ParamValue::Literal(Param::float(0.5, 0.0, 100.0)));
        r.run_once();
        assert_eq!(r.run_policy.max_frequency, 2.0, "one update every half second is 2 Hz");
        assert!(r.bindings.get(&ParamKey::new("common", "max_frequency")).is_none(), "and it unbound");
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
    fn unbinding_a_broken_param_clears_its_error_on_the_wire_too() {
        // §3.4: a literal on a bound param unbinds it, and `Status::BindingErrors` is a DELTA — so
        // the clear has to be SENT. The sibling path (a recovering evaluation) sends it; an error
        // that only one of the two can clear leaves an inspector field stuck red.
        let (mut r, t) = consumer_fixture();
        let key = ParamKey::new("cfg", "scale");
        r.set_param(key.clone(), missing_expr("no node named `lfo`"));
        assert_eq!(
            binding_error_reports(&t),
            [vec![(key.clone(), Some("no node named `lfo`".to_string()))]],
            "the error was announced",
        );

        r.set_param(key.clone(), ParamValue::Literal(Param::float(5.0, 0.0, 1000.0)));
        assert!(r.binding_errors.is_empty(), "the record cleared");
        assert_eq!(
            binding_error_reports(&t),
            [
                vec![(key.clone(), Some("no node named `lfo`".to_string()))],
                vec![(key, None)],
            ],
            "and so was the clear",
        );
    }

    #[test]
    fn a_broken_binding_falls_back_to_the_param_it_was_authored_with() {
        // spec §2.1: a failed binding falls back to its LITERAL for that run, which needs the
        // literal to still exist. Overwriting the record with each evaluated value leaves a param
        // holding the last number a since-deleted reference gave it — an oscillator authored at
        // 5 Hz runs at 500 forever because `nd('lfo')` said so once.
        let (mut r, _t) = consumer_fixture();
        let key = ParamKey::new("cfg", "scale");
        r.set_param(key.clone(), ParamValue::Literal(Param::float(5.0, 0.0, 1000.0)));
        r.set_param(key.clone(), value_expr(Param::float(500.0, 0.0, 1000.0), false));
        assert_eq!(effective_f64(&r, &key), Some(500.0), "the binding is in force");

        // `lfo` is deleted: the graph re-sends the binding with its reference unresolved.
        r.set_param(key.clone(), missing_expr("no node named `lfo`"));
        assert_eq!(effective_f64(&r, &key), Some(5.0), "back to the authored value, not the stale 500");
        assert_eq!(r.binding_errors.get(&key).map(String::as_str), Some("no node named `lfo`"));
        assert!(!r.evaluated.contains_key(&key), "and it has no evaluated value to project");
    }

    #[test]
    fn correcting_the_param_that_broke_setup_heals_a_node_that_never_runs() {
        // spec §5.1: a param write runs the D3 initialization retry FIRST. Without it a consumer
        // with no triggers and autotrigger off is broken permanently — `run()` is the only other
        // caller of the gate and it never runs — where `update_param` heals such a node today.
        let mut r = NodeRuntime::new(&NEEDS_PARAM, Arc::new(MemoryTransport::default()));
        assert!(matches!(r.fault, Some(NodeFault::Setup { .. })), "setup refused the default");
        assert_eq!(hook_log(), ["cfg.ok", "cfg.scale"], "birth replayed the record");

        r.set_param(ParamKey::new("cfg", "ok"), ParamValue::Literal(Param::boolean(true)));
        assert!(r.fault.is_none(), "the correction re-initialized the node");
        assert_eq!(
            hook_log(),
            ["cfg.ok", "cfg.scale", "cfg.ok", "cfg.scale"],
            "the retry's replay applied the edit — notifying again would double-apply it",
        );
    }

    #[test]
    fn an_uninitialized_node_hears_no_param_hook() {
        // D3: nothing runs against a node whose `setup()` failed — not `process`, not a param
        // callback. The value still lands, so it is there for the replay when the retry succeeds.
        let mut r = NodeRuntime::new(&NEEDS_PARAM, Arc::new(MemoryTransport::default()));
        r.set_param(ParamKey::new("cfg", "scale"), value_expr(Param::float(2.0, 0.0, 4.0), false));
        assert_eq!(
            hook_log(),
            ["cfg.ok", "cfg.scale", "cfg.ok", "cfg.scale"],
            "the retry replayed the record, and nothing else was dispatched",
        );
        assert_eq!(goofi_node::param(&r.effective, "cfg", "scale").and_then(Param::as_f64), Some(2.0));
        assert!(r.fault.is_some(), "and it is still uninitialized");
    }

    #[test]
    fn a_failed_setup_retries_on_a_backoff_that_restarts_from_each_attempt() {
        // The backoff is what keeps a node whose device is missing from re-opening it at its wake
        // rate — a producer at `default_ufreq` would attempt ~30×/s, and `setup` acquires. It is
        // paced from the LAST attempt, so an unchanged fault must still move its `last_attempt`
        // even though it is deliberately not re-broadcast.
        let mut r = NodeRuntime::new(&RETRY_PROBE, Arc::new(MemoryTransport::default()));
        assert_eq!(setup_attempts(), 1, "birth is the first attempt");
        for _ in 0..50 {
            r.run_once();
        }
        assert_eq!(setup_attempts(), 1, "50 wakes inside the window are one");

        expire_setup_backoff(&mut r);
        r.run_once();
        assert_eq!(setup_attempts(), 2, "the window's end admits exactly one");
        for _ in 0..50 {
            r.run_once();
        }
        assert_eq!(setup_attempts(), 2, "and it restarts from THAT attempt");
    }

    #[test]
    fn a_fault_is_reported_on_transition_and_a_new_message_is_a_new_one() {
        // §6.2: the node reports TRANSITIONS, so the status worker needs no diffing — and the
        // console does not repaint the same line at the node's run rate. The comparison is
        // discriminant AND message, because a node failing differently is saying something new.
        let t = Arc::new(MemoryTransport::default());
        let mut r = NodeRuntime::new(&BAD_PROCESS, t.clone());
        r.run_once();
        r.run_once();
        assert_eq!(fault_reports(&t), [Some("no".to_string())], "two failing runs, one transition");

        r.run_once();
        assert_eq!(
            fault_reports(&t),
            [Some("no".to_string()), Some("still no".to_string())],
            "a different complaint is a different fault",
        );
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

        r.deliver_arrival(key.clone(), Param::float(3.0, 0.0, 10.0));
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
    /// holding `false` so a re-derivation genuinely stops this producer, which is what lets
    /// `a_common_arrival_repaces_without_running` mean its own name. And the runtime starts
    /// SETTLED, because a dirty flag left over from wiring would mask the very arrival these
    /// tests are about.
    ///
    /// What this fixture CANNOT show, measured by dropping every arrival on the floor: the
    /// autotrigger toggle test delivers `true`, which is also this producer's literal, so it pins
    /// the dirty-marking and not the value. The value landing is pinned on the consumer, where the
    /// two differ — [`a_stream_arrival_repaces_a_consumer_without_ever_running_it`].
    fn fixture() -> (NodeRuntime, Arc<MemoryTransport>) {
        let transport = Arc::new(MemoryTransport::default());
        let mut r = NodeRuntime::new(&PRODUCER, transport.clone());
        r.set_param(ParamKey::new("common", "max_frequency"), stream_expr(true));
        r.set_param(ParamKey::new("common", "autotrigger"), stream_expr(true));
        r.deliver_arrival(ParamKey::new("common", "autotrigger"), Param::boolean(false));
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
    fn triggered_fixture() -> (NodeRuntime, Arc<MemoryTransport>) {
        let transport = Arc::new(MemoryTransport::default());
        (NodeRuntime::new(&TRIGGERED, transport.clone()), transport)
    }

    fn fixture_with_trigger_input() -> NodeRuntime {
        triggered_fixture().0
    }

    /// The single-variable identity binding §5.3's rewrite produces for `globals.default_ufreq`,
    /// awaiting a producer.
    fn stream_expr(trigger: bool) -> ParamValue {
        ParamValue::Expr {
            source: "__v0".to_string(),
            vars: vec![Var::Stream { name: "__v0".to_string(), service: "svc".to_string(), event_id: 65 }],
            trigger,
        }
    }

    /// A binding the graph could not resolve — a deleted reference, a removed global.
    fn missing_expr(reason: &str) -> ParamValue {
        ParamValue::Expr {
            source: "__v0".to_string(),
            vars: vec![Var::Missing { name: "__v0".to_string(), reason: reason.to_string() }],
            trigger: false,
        }
    }

    /// The same binding with its variable already resolved — a `globals.*` read the graph delivered
    /// inline, which is how a globals edit reaches a node (§5.2).
    fn value_expr(value: Param, trigger: bool) -> ParamValue {
        ParamValue::Expr {
            source: "__v0".to_string(),
            vars: vec![Var::Value { name: "__v0".to_string(), value }],
            trigger,
        }
    }

    fn binding_error_reports(t: &MemoryTransport) -> Vec<Vec<(ParamKey, Option<String>)>> {
        t.reported()
            .into_iter()
            .filter_map(|s| match s {
                Status::BindingErrors { errors } => Some(errors),
                _ => None,
            })
            .collect()
    }

    fn effective_f64(r: &NodeRuntime, key: &ParamKey) -> Option<f64> {
        goofi_node::param(&r.effective, &key.group, &key.name).and_then(Param::as_f64)
    }

    fn now_minus_ms(ms: u64) -> Instant {
        Instant::now() - Duration::from_millis(ms)
    }

    /// Move the recorded attempt out of the backoff window, rather than sleeping a second for it.
    fn expire_setup_backoff(r: &mut NodeRuntime) {
        let Some(NodeFault::Setup { last_attempt, .. }) = &mut r.fault else {
            panic!("not a setup fault");
        };
        *last_attempt -= SETUP_RETRY_MS + 1.0;
    }

    fn text_frame(s: &str) -> Data {
        Data::string(s, Meta::empty())
    }

    fn text(d: &Data) -> String {
        match d.value() {
            goofi_core::Value::Str(s) => s.to_string(),
            _ => panic!("expected a string frame"),
        }
    }

    /// Every frame the node emitted, as `slot: content` — the oracle that tells a run that
    /// happened from one that did not.
    fn published(t: &MemoryTransport) -> Vec<String> {
        t.published().iter().map(|(slot, frame)| format!("{slot}: {}", text(frame))).collect()
    }

    fn fault_reports(t: &MemoryTransport) -> Vec<Option<String>> {
        t.reported()
            .into_iter()
            .filter_map(|s| match s {
                Status::Fault { fault } => Some(fault.map(|f| f.msg().to_string())),
                _ => None,
            })
            .collect()
    }

    fn param_value_reports(t: &MemoryTransport) -> Vec<Vec<(ParamKey, Param)>> {
        t.reported()
            .into_iter()
            .filter_map(|s| match s {
                Status::ParamValues { evaluated } => Some(evaluated),
                _ => None,
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Test nodes
    // -----------------------------------------------------------------------

    /// Emits a frame naming the run that produced it, so a test can tell "the node did not run"
    /// from "publishing is broken" — an oracle that only ever asserts emptiness cannot.
    #[derive(Default)]
    struct Emit {
        runs: usize,
    }
    impl Node for Emit {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.runs += 1;
            out.set("out", text_frame(&format!("run {}", self.runs)));
            Ok(())
        }
    }

    /// Reads BOTH its input slots and emits what it saw, so path A's cells are observable.
    #[derive(Default)]
    struct Echo;
    impl Node for Echo {
        fn process(&mut self, i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let read = |name: &str| i.get(name).map(text).unwrap_or_else(|| "-".to_string());
            out.set("out", text_frame(&format!("{}|{}", read("in"), read("ref"))));
            Ok(())
        }
    }

    /// Fails its first `setup`, then succeeds — the only way to reach a standing Setup fault the
    /// way production does, on a node that is genuinely uninitialized underneath it.
    #[derive(Default)]
    struct FlakySetup {
        attempts: usize,
    }
    impl Node for FlakySetup {
        fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.attempts += 1;
            (self.attempts > 1).then_some(()).ok_or_else(|| "no device".into())
        }
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            out.set("out", text_frame("ok"));
            Ok(())
        }
    }

    /// Fails its first `process`, then succeeds.
    #[derive(Default)]
    struct FlakyProcess {
        runs: usize,
    }
    impl Node for FlakyProcess {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.runs += 1;
            if self.runs == 1 {
                return Err("boom".into());
            }
            out.set("out", text_frame(&format!("run {}", self.runs)));
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
            out.set("out", text_frame("ok"));
            Ok(())
        }
    }

    thread_local! {
        /// Every `on_param_changed` key a node on this thread heard, in order. Thread-local rather
        /// than static because the harness gives each test its own thread, so two tests sharing a
        /// node type cannot see each other's calls.
        static HOOK_LOG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
        /// `setup()` attempts on this thread's node — the retry backoff's observable.
        static SETUP_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
    }

    fn hook_log() -> Vec<String> {
        HOOK_LOG.with(|log| log.borrow().clone())
    }

    fn setup_attempts() -> usize {
        SETUP_ATTEMPTS.with(Cell::get)
    }

    #[derive(Default)]
    struct RetryProbe;
    impl Node for RetryProbe {
        fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            SETUP_ATTEMPTS.with(|n| n.set(n.get() + 1));
            Err("no device".into())
        }
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Ok(())
        }
    }

    /// A node whose `setup()` refuses until a param is corrected — the D3 retry door's whole point.
    #[derive(Default)]
    struct NeedsGoodParam {
        ok: bool,
    }
    impl Node for NeedsGoodParam {
        fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.ok.then_some(()).ok_or_else(|| "cfg.ok is false".into())
        }
        fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
            HOOK_LOG.with(|log| log.borrow_mut().push(format!("{}.{}", key.group, key.name)));
            if key.name == "ok" {
                self.ok = v.as_bool().unwrap_or(false);
            }
            Ok(())
        }
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Ok(())
        }
    }

    /// Fails every run, and changes its complaint on the third — a different message is a
    /// different fault.
    #[derive(Default)]
    struct BadProcess {
        runs: usize,
    }
    impl Node for BadProcess {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            self.runs += 1;
            Err(if self.runs > 2 { "still no" } else { "no" }.into())
        }
    }

    static OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::String }];
    static NO_PARAMS: &[ParamDecl] = &[];
    static SLOTS: &[SlotDecl] = &[
        SlotDecl { name: "in", kind: SlotType::String, trigger_process: true, multi: false, required: false },
        SlotDecl { name: "ref", kind: SlotType::String, trigger_process: false, multi: false, required: false },
    ];

    static PRODUCER: NodeManifest = manifest("_RuntimeProducer", &[], true, default_factory::<Emit>);
    static CONSUMER: NodeManifest =
        NodeManifest { params: SCALE_PARAMS, ..manifest("_RuntimeConsumer", &[], false, default_factory::<Emit>) };
    static SCALE_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "cfg",
        name: "scale",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 1000.0 },
        expression: None,
        doc: None,
    }];
    static TRIGGERED: NodeManifest = manifest("_RuntimeTriggered", SLOTS, false, default_factory::<Echo>);
    static MULTI_IN: NodeManifest = manifest("_RuntimeMultiIn", MULTI_SLOT, false, default_factory::<Echo>);
    static FLAKY_SETUP: NodeManifest = NodeManifest {
        params: SCALE_PARAMS,
        ..manifest("_RuntimeFlakySetup", &[], true, default_factory::<FlakySetup>)
    };
    static FLAKY_PROCESS: NodeManifest =
        manifest("_RuntimeFlakyProcess", &[], true, default_factory::<FlakyProcess>);
    static MULTI_SLOT: &[SlotDecl] = &[SlotDecl {
        name: "many",
        kind: SlotType::String,
        trigger_process: true,
        multi: true,
        required: false,
    }];
    static BAD_SETUP: NodeManifest = manifest("_RuntimeBadSetup", &[], true, default_factory::<BadSetup>);
    static BAD_PROCESS: NodeManifest = manifest("_RuntimeBadProcess", &[], true, default_factory::<BadProcess>);
    static RETRY_PROBE: NodeManifest = manifest("_RuntimeRetryProbe", &[], true, default_factory::<RetryProbe>);
    static NEEDS_PARAM: NodeManifest = NodeManifest {
        params: CFG_PARAMS,
        ..manifest("_RuntimeNeedsParam", &[], false, default_factory::<NeedsGoodParam>)
    };
    static CFG_PARAMS: &[ParamDecl] = &[
        ParamDecl { group: "cfg", name: "ok", spec: ParamSpec::Bool { default: false }, expression: None, doc: None },
        ParamDecl {
            group: "cfg",
            name: "scale",
            spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 4.0 },
            expression: None,
            doc: None,
        },
    ];

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
