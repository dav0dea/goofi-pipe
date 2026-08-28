//! The per-node runtime: the wake loop's body, the three run paths, and a node's faults.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use goofi_core::{Data, Param};
use goofi_node::{
    ExprEvaluator, Inputs, Node, NodeCtx, NodeManifest, Outputs, ParamGroups, ParamKey, Params,
    RunPolicy,
};
use indexmap::IndexMap;

mod mailbox;
pub(crate) mod plan;
mod transport;
mod wire;

pub use mailbox::{Binding, Mailbox};
pub use transport::{
    door_service, iox_node, nodes_dir, open_output_subscriber, output_service,
    reclaim_stale_resources, service_base, sweep_once, ByteSubscriber, Doorbell, IoxNode,
    IoxTransport, NodeChannel,
};
/// Only the graph mints a scope, and it is in this crate.
pub(crate) use transport::service_instance;
pub use wire::{
    Control, ControlSink, Envelope, EventId, NodeStage, ParamValue, ServiceName, Status, Transport,
    Var, VarName,
};

/// The scheduling namespace. A `common.*` param decides *when* a node runs, so it is resolved
/// before the gates are read and never inside a run (§1.1).
const COMMON: &str = "common";

/// `SETUP_RETRY_INTERVAL` in the wall-clock milliseconds a [`NodeFault`] carries.
const SETUP_RETRY_MS: f64 = crate::SETUP_RETRY_INTERVAL * 1000.0;

/// How often a node reports its measured update rate. A rate is a MEASUREMENT rather than a
/// transition, and an uncapped producer would take one per emit.
const UFREQ_REPORT_MS: u128 = 250;

/// How close to its deadline a node stops listening for rings and sleeps to it instead. A timed
/// wait's timeout is rounded up by the OS — to a jiffie on Linux, 15.6 ms on Windows — and `sleep`
/// is not, so the last stretch uses it. Not a tuning knob, and deliberately not adaptive.
const LISTEN_FLOOR: Duration = Duration::from_millis(25);

/// The smoothing factor of the `ufreq` EMA: how much the newest interval moves the estimate.
const UFREQ_EMA_ALPHA: f64 = 0.2;

/// The `ufreq` meter's state: when this node last emitted, and the smoothed interval between emits.
#[derive(Default)]
struct UfreqMeter {
    last_emit: Option<f64>,
    ema: Option<f64>,
}

/// The one thing that stops a node's manager-side thread. A flag rather than a control message: a
/// node removed before it answered [`Status::Ready`] has no sink to receive one.
#[derive(Default)]
pub struct Halt {
    stop: AtomicBool,
    /// Set once the [`NodeRuntime`] — and with it every iceoryx2 port — has been DROPPED, which is
    /// what releases the shared memory. The only thing a teardown can usefully wait for.
    released: AtomicBool,
}

impl Halt {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
    fn release(&self) {
        self.released.store(true, Ordering::Release);
    }
    pub fn released(&self) -> bool {
        self.released.load(Ordering::Acquire)
    }
}

/// The pseudo input slot a bound param's producer wires ride: a binding subscribes exactly as an
/// input slot does, through the one door. Namespaced with a character no declared slot may carry.
pub fn expr_wire_slot(key: &ParamKey) -> String {
    format!("expr:{}:{}", key.group, key.name)
}

/// The [`ParamKey`] an [`expr_wire_slot`] name refers back to, or `None` for a declared slot.
fn expr_wire_key(slot: &str) -> Option<ParamKey> {
    let rest = slot.strip_prefix("expr:")?;
    let (group, name) = rest.split_once(':')?;
    Some(ParamKey::new(group, name))
}

/// What is wrong with a node. Wall-clock `f64` rather than [`Instant`], because a fault is
/// reported over the wire.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NodeFault {
    Setup { msg: String, since: f64, last_attempt: f64 },
    Process { msg: String, since: f64 },
}

impl NodeFault {
    pub fn msg(&self) -> &str {
        match self {
            NodeFault::Setup { msg, .. } | NodeFault::Process { msg, .. } => msg,
        }
    }
}

/// One node, its scheduling state, and its faults.
pub struct NodeRuntime {
    manifest: &'static NodeManifest,
    node: Box<dyn Node>,
    transport: Arc<dyn Transport>,
    /// Shared with the graph: the node EVALUATES, the graph COMPILES, so the authoring RPC can
    /// answer with a real compile error. `None` ⇒ every bound param falls back to its literal.
    evaluator: Option<Arc<dyn ExprEvaluator>>,
    /// The graph's clock origin, so `NodeCtx::now` is seconds-since-start on every node's thread
    /// rather than seconds-since-this-node's-birth.
    started: Instant,

    /// Something asked this node to run and it has not run since. Autotrigger is not here — it
    /// lives in `run_policy`, beside the cap that paces it.
    pub(crate) trigger_pending: bool,
    pub(crate) run_policy: RunPolicy,
    pub(crate) last_run: Option<Instant>,
    /// Set by ANY arrival that can affect a `common.*` param, whatever path it came in on (§1.1).
    pub(crate) common_dirty: bool,

    /// The param RECORD: literals only, which is what the `.gfi` persists and what a broken or
    /// not-yet-arrived binding falls back to (§2.1). Apart from `effective`, or an evaluated value
    /// would erase the number the user authored.
    pub(crate) literals: ParamGroups,
    /// The node's FULL params — the literal record overlaid with evaluated bindings. What
    /// `process()` reads and what `RunPolicy::from_params` is given.
    pub(crate) effective: ParamGroups,
    /// The SPARSE bound subset, which exists only as the wire projection in
    /// [`Status::ParamValues`]. Handing THIS to `RunPolicy::from_params` would default every
    /// absent key.
    pub(crate) evaluated: IndexMap<ParamKey, Param>,
    pub(crate) bindings: IndexMap<ParamKey, Binding>,

    /// Latest-wins input cells, one per declared single input slot.
    pub(crate) inputs: IndexMap<&'static str, Option<Data>>,
    /// Per-WIRE latest-wins cells for each `multi` input slot, in the order the last `InSlot` set
    /// named — which IS `Inputs::get_multi`'s connection order.
    pub(crate) multi_wires: IndexMap<&'static str, Vec<(ServiceName, Option<Data>)>>,
    pub(crate) ctx: NodeCtx,
    /// Per-output-slot emit counter for `meta["index"]` — engine-owned, the node never sees it.
    index_counters: HashMap<&'static str, u64>,
    /// Per-NODE measured update rate for `meta["ufreq"]`: one meter, stamped onto every slot this
    /// node emits, because ufreq describes the node rather than a slot.
    ufreq_meter: UfreqMeter,
    /// When the rate was last REPORTED, which is not when it was last measured.
    last_ufreq_report: Option<Instant>,
    stage: NodeStage,

    pub(crate) fault: Option<NodeFault>,
    /// A MAP, not a fault variant: several bindings can be errored at once, each on its own field.
    pub(crate) binding_errors: HashMap<ParamKey, String>,
    initialized: bool,
}

/// Everything a node's thread needs that is the GRAPH's rather than the node's.
pub struct NodeEnv {
    pub evaluator: Option<Arc<dyn ExprEvaluator>>,
    pub started: Instant,
}

impl NodeEnv {
    /// The environment of a node that belongs to no graph — what a test driving a [`NodeRuntime`]
    /// directly gets.
    pub fn detached() -> NodeEnv {
        NodeEnv { evaluator: None, started: Instant::now() }
    }
}

impl NodeRuntime {
    /// Seed the node's params, then `setup()`. A failing `setup` leaves it UNINITIALIZED with a
    /// [`NodeFault::Setup`] standing, and nothing runs against it until a retry succeeds.
    pub fn new(
        manifest: &'static NodeManifest,
        node: Box<dyn Node>,
        params: ParamGroups,
        transport: Arc<dyn Transport>,
        env: NodeEnv,
    ) -> NodeRuntime {
        let effective = goofi_node::with_common(params, manifest);
        let run_policy = RunPolicy::from_params(&effective);
        let mut runtime = NodeRuntime {
            manifest,
            node,
            transport,
            evaluator: env.evaluator,
            started: env.started,
            trigger_pending: false,
            run_policy,
            last_run: None,
            common_dirty: false,
            literals: effective.clone(),
            effective,
            evaluated: IndexMap::new(),
            bindings: IndexMap::new(),
            inputs: manifest.inputs.iter().filter(|s| !s.multi).map(|s| (s.name, None)).collect(),
            multi_wires: manifest.inputs.iter().filter(|s| s.multi).map(|s| (s.name, Vec::new())).collect(),
            ctx: NodeCtx::new(),
            index_counters: HashMap::new(),
            ufreq_meter: UfreqMeter::default(),
            last_ufreq_report: None,
            stage: NodeStage::Setup,
            fault: None,
            binding_errors: HashMap::new(),
            initialized: false,
        };
        // §4's birth barrier: the graph addresses nothing until this lands. Sent before `setup()`
        // runs, so a `setup` that fails at birth still has somewhere to report to.
        runtime.transport.report(Status::Ready);
        runtime.transport.report(Status::Stage { stage: NodeStage::Setup });
        runtime.initialize();
        runtime.publish_stage();
        runtime
    }

    /// Announce the node's lifecycle stage when it CHANGED. `error` is the graph's derivation from
    /// the fault and is not a stage a node can claim.
    fn publish_stage(&mut self) {
        let next = if self.initialized { NodeStage::Ready } else { NodeStage::Setup };
        if next != self.stage {
            self.stage = next;
            self.transport.report(Status::Stage { stage: next });
        }
    }

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

    /// When the cap next admits a run: one period after the last one. `None` means now. Read fresh
    /// from `last_run` every time, so a cap edited mid-park takes effect on the spot.
    fn due(&self) -> Option<Instant> {
        Some(self.last_run? + Duration::from_secs_f64(self.run_policy.period()?))
    }

    fn rate_cap_elapsed(&self) -> bool {
        self.due().is_none_or(|d| Instant::now() >= d)
    }

    /// How long until the cap admits another run. Zero when it already does.
    fn cap_release(&self) -> Duration {
        self.due().map_or(Duration::ZERO, |d| d.saturating_duration_since(Instant::now()))
    }

    /// One iteration of the wake loop, minus the park. §3.3 makes a notification a pure hint, so
    /// the drain never consults the event ids.
    pub fn run_once(&mut self) {
        // Frames FIRST: a control message may re-wire the slot a frame just arrived on, and
        // applying the wiring before draining would throw away the frame the old wire delivered.
        for (slot, wire, frame) in self.transport.drain_inputs() {
            self.deliver_input(&slot, wire, frame);
        }
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

    /// Apply every waiting control message and ACK each one — a message applied but not acked
    /// stalls the wire change that sent it. Slot messages go to the transport.
    fn drain_control(&mut self) {
        let transport = self.transport.clone();
        for Envelope { seq, control } in transport.drain_control() {
            let ok = match control {
                Control::InSlot { slot, services } => {
                    let wired = transport.wire_in(&slot, &services);
                    // The node's own cells follow the set it was told to hold: a slot with no wire
                    // left holds no frame, and a `multi` slot's cells keep their producers' order.
                    self.reslot(&slot, &services);
                    wired
                }
                Control::OutSlot { slot, targets } => transport.wire_out(&slot, &targets),
                Control::SetParam { key, value } => {
                    self.set_param(key, value);
                    Ok(())
                }
                Control::RefreshParam { key } => {
                    self.refresh_param(key);
                    Ok(())
                }
            };
            transport.report(Status::Ack { seq, ok });
        }
    }

    /// Re-enumerate a refreshable `Str` param's options and report them (§8.5). The hook is
    /// third-party code, so a panic in it has to become a report rather than kill this thread.
    fn refresh_param(&mut self, key: ParamKey) {
        // D3: an interaction retries the initialization first, so a picker whose node failed
        // `setup()` rescans as soon as that node comes up.
        let options = if self.ensure_initialized() {
            let params = Params::new(&self.effective);
            crate::guard_lifecycle(|| self.node.on_param_refreshed(&key, &params)).unwrap_or(None)
        } else {
            None
        };
        self.publish_stage();
        if let Some(options) = &options {
            // The record moves too, so the next `serialize` and inspector read agree with what was
            // just reported.
            if let Some(Param::Str { options: slot, .. }) =
                self.literals.get_mut(&key.group).and_then(|g| g.get_mut(&key.name))
            {
                *slot = Some(options.clone());
            }
            if let Some(Param::Str { options: slot, .. }) =
                self.effective.get_mut(&key.group).and_then(|g| g.get_mut(&key.name))
            {
                *slot = Some(options.clone());
            }
        }
        self.transport.report(Status::RefreshOptions { key, options });
    }

    /// Apply a slot's new wire set to the node's OWN cells: a surviving wire keeps its frame, and a
    /// wire that left takes its frame with it.
    fn reslot(&mut self, slot: &str, services: &[ServiceName]) {
        if let Some(cells) = self.multi_wires.get_mut(slot) {
            let mut previous = std::mem::take(cells);
            *cells = services
                .iter()
                .map(|service| {
                    let held = previous
                        .iter_mut()
                        .find(|(name, _)| name == service)
                        .and_then(|(_, frame)| frame.take());
                    (service.clone(), held)
                })
                .collect();
            return;
        }
        // A single slot has at most one wire, so an empty set is a disconnection: the cell it fed
        // must clear, or a node keeps running on the frame of a producer it is no longer wired to.
        if services.is_empty() {
            if let Some(cell) = self.inputs.get_mut(slot) {
                *cell = None;
            }
        }
    }

    /// Write a param. A `Literal` on a bound param unbinds it (§3.4); an `Expr` binds it and is
    /// evaluated once here, or a binding error could never appear on a node that never runs.
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
                // An unbind drops the binding's subscriptions with it — the producer is told to
                // stop ringing by the graph's phase 1, and this is the consumer's own half.
                if self.bindings.shift_remove(&key).is_some() {
                    let _ = self.transport.wire_in(&expr_wire_slot(&key), &[]);
                }
                if self.evaluated.shift_remove(&key).is_some() {
                    self.report_param_values();
                }
                let cleared = self.record_binding_error(&key, None);
                self.report_binding_errors(cleared.into_iter().collect());
                self.set_literal(&key, p.clone());
                Some(p)
            }
            ParamValue::Expr { source, vars, trigger, id } => {
                let binding = Binding::new(source, vars, trigger, id);
                // §5.3: an expression reference IS a link, so its producers are subscribed through
                // the one subscribe door, on a pseudo-slot named after the param.
                let services: Vec<ServiceName> =
                    binding.streams.iter().map(|(_, service)| service.clone()).collect();
                let _ = self.transport.wire_in(&expr_wire_slot(&key), &services);
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
        // A param write is an INTERACTION, and an interaction retries the initialization first.
        // Unthrottled, unlike a wake — this is a user asking.
        let was_initialized = self.initialized;
        let healed = self.ensure_initialized() && !was_initialized;
        // `initialize` replays the whole record through `on_param_changed`, so a retry that
        // succeeded has already delivered this edit. An UNINITIALIZED node hears nothing (D3).
        if let Some(p) = literal {
            if self.initialized && !healed && key.group != COMMON {
                self.on_param_changed(&key, &p);
            }
        }
    }

    /// A frame off one of this node's wires. ONE door for a declared slot and a bound param alike:
    /// an [`expr_wire_slot`] name lands in a binding's mailbox, a declared one in an input cell.
    pub fn deliver_input(&mut self, slot: &str, wire: usize, frame: Data) {
        if let Some(key) = expr_wire_key(slot) {
            let Some(binding) = self.bindings.get_mut(&key) else {
                // A frame for a param this node has no binding on: a producer that emitted between
                // the unbind and the graph's phase 1. Dropping it converges.
                return;
            };
            binding.deliver(wire, frame);
            let trigger = binding.trigger;
            self.arrived(&key, trigger);
            return;
        }
        let Some(decl) = self.manifest.inputs.iter().find(|s| s.name == slot) else {
            // A slot this node does not declare: a frame from a wire the graph has since re-planned.
            return;
        };
        if decl.multi {
            match self.multi_wires.get_mut(decl.name).and_then(|cells| cells.get_mut(wire)) {
                Some(cell) => cell.1 = Some(frame),
                // A wire index the last `InSlot` set does not name. Dropped rather than appended:
                // appending would put the frame where `Inputs::get_multi` reads another producer.
                None => return,
            }
        } else {
            self.inputs.insert(decl.name, Some(frame));
        }
        if decl.trigger_process {
            self.trigger_pending = true;
        }
    }

    /// What an arrival does to the schedule, stated ONCE and by key NAMESPACE rather than per path.
    fn arrived(&mut self, key: &ParamKey, trigger: bool) {
        if key.group == COMMON {
            // Re-pacing is not a reason to run, so `trigger` is IGNORED on this namespace.
            self.common_dirty = true;
        } else if trigger {
            self.trigger_pending = true;
        }
    }

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
            // An empty mailbox is not an error — the literal stands. The target is the LITERAL
            // rather than the last evaluated value, which would let a binding drift its own type.
            let target = self.literal(&key).unwrap_or_else(|| Param::float(0.0, f64::NEG_INFINITY, f64::INFINITY));
            let now = self.ctx.now;
            let evaluator = self.evaluator.clone();
            let evaluated = match self.bindings[&key].evaluate(evaluator.as_deref(), now, &target) {
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
            // The hook is the only way an evaluated value reaches a node's mirrored field. A
            // `common.*` param has none, and an UNINITIALIZED node gets it from the retry's replay.
            if key.group != COMMON && self.initialized {
                self.on_param_changed(&key, &next);
            }
        }
        self.report_binding_errors(errors);
        if values_changed {
            self.report_param_values();
        }
    }

    /// The whole sparse map, never a delta — the graph replaces its copy with this, so a value it
    /// is no longer told about is one it would otherwise preview for ever.
    fn report_param_values(&mut self) {
        let evaluated = self.evaluated.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        self.transport.report(Status::ParamValues { evaluated });
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

    /// Record or clear a binding's error, answering only when it CHANGED. The status is a delta,
    /// which is safe only because the graph files it against the INSTANCE.
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

    fn run(&mut self) {
        // The frame that asked for this run has been seen; holding it would fire the node twice
        // the moment it recovers. Consumed whether or not the run itself gets as far as `process`.
        self.trigger_pending = false;
        self.last_run = Some(Instant::now());
        if !self.ensure_initialized_paced() {
            self.publish_stage();
            return;
        }
        self.publish_stage();
        // A required slot must HOLD data when the node runs — presence, never wiring, so a slot
        // wired to a producer that has emitted nothing reads the same as an unwired one.
        if let Some(slot) = self.missing_required() {
            self.set_fault(Some(NodeFault::Process {
                msg: format!("required input slot `{slot}` has no data"),
                since: now_ms(),
            }));
            return;
        }
        self.ctx.now = self.started.elapsed().as_secs_f64();
        let multis = self.materialize_multis();
        let mut outputs = self.manifest.output_buffer();
        let result = {
            let inputs = Inputs::with_multi(&self.inputs, &multis);
            let params = Params::new(&self.effective);
            let mut out = Outputs::new(&mut outputs);
            crate::guard_lifecycle(|| self.node.process(&inputs, &mut out, &mut self.ctx, &params))
                .unwrap_or_else(crate::fold_panic)
        };
        match result {
            Ok(()) => {
                // Clearing on success is safe because `process` is unreachable while a setup error
                // stands. It does NOT clear a binding error, which only a good evaluation clears.
                self.set_fault(None);
                // The engine's own meta goes on before anything leaves the node — there is no
                // second stamping site.
                let ufreq = stamp_meta(
                    self.manifest,
                    &self.inputs,
                    &mut outputs,
                    self.ctx.now,
                    &mut self.index_counters,
                    &mut self.ufreq_meter,
                );
                for (slot, frame) in outputs.iter() {
                    if let Some(frame) = frame {
                        self.transport.publish(slot, frame);
                    }
                }
                self.report_ufreq(ufreq);
            }
            Err(e) => self.set_fault(Some(NodeFault::Process { msg: e.0, since: now_ms() })),
        }
    }

    /// A `required` input slot holding no frame, or `None`. Read over the WIRE CELLS: a `multi`
    /// slot's frames live in [`Self::multi_wires`], not in `inputs`.
    fn missing_required(&self) -> Option<&'static str> {
        self.manifest.inputs.iter().filter(|s| s.required).find_map(|slot| {
            let absent = if slot.multi {
                self.multi_wires.get(slot.name).is_none_or(|c| !c.iter().any(|(_, f)| f.is_some()))
            } else {
                self.inputs.get(slot.name).and_then(Option::as_ref).is_none()
            };
            absent.then_some(slot.name)
        })
    }

    /// The present frames on each `multi` slot, in wire order — absent wires dropped, so a node
    /// sees only the frames that actually arrived.
    fn materialize_multis(&self) -> IndexMap<&'static str, Vec<Data>> {
        self.multi_wires
            .iter()
            .map(|(slot, cells)| (*slot, cells.iter().filter_map(|(_, f)| f.clone()).collect()))
            .collect()
    }

    /// Report the measured rate, at most every [`UFREQ_REPORT_MS`]. `None` is the first emit, which
    /// has no interval yet and therefore nothing to say.
    fn report_ufreq(&mut self, hz: Option<f64>) {
        let Some(hz) = hz else { return };
        let now = Instant::now();
        if self.last_ufreq_report.is_some_and(|t| (now - t).as_millis() < UFREQ_REPORT_MS) {
            return;
        }
        self.last_ufreq_report = Some(now);
        self.transport.report(Status::Ufreq { hz });
    }

    /// The initialization gate (D3): a node whose `setup()` failed is UNINITIALIZED, so nothing
    /// runs against it, and any interaction retries the initialization first.
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

    /// The param replay and `setup()` together, which are one unit — a retry re-runs both.
    fn initialize(&mut self) {
        let attempt = now_ms();
        match crate::seed_node(&mut *self.node, &self.effective, &mut self.ctx) {
            None => {
                self.initialized = true;
                self.set_fault(None);
            }
            // One clock read: the attempt IS when the failure happened, and `set_fault` keeps the
            // `since` of a fault that has not changed anyway.
            Some(msg) => {
                self.set_fault(Some(NodeFault::Setup { msg, since: attempt, last_attempt: attempt }))
            }
        }
    }

    /// Install a fault, keeping `since` when nothing changed — the node reports only TRANSITIONS.
    /// An unchanged fault still moves `last_attempt`, or the backoff turns off entirely.
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

    /// Run the node until it is halted — the body of its manager-side thread (§2). One loop for
    /// every execution kind, which differ only in what `process()` does.
    pub fn run_forever(mut self, halt: Arc<Halt>) {
        while !halt.stopped() {
            self.run_once();
            match self.next_wake() {
                // Nothing due: block until something rings. Indefinitely, because every reason to
                // wake IS a ring — the [`Halt`] flag included, whose door `signal_stop` rings.
                None => {
                    self.transport.wait(None);
                }
                // Due now — an uncapped free-runner. Parking on a zero timeout would add a syscall
                // per run and change nothing.
                Some(d) if d.is_zero() => continue,
                // Due, but not yet: listen while the deadline is far enough away that the wait's
                // rounding has room, so a ring is still answered at once.
                Some(d) if d > LISTEN_FLOOR => {
                    self.transport.wait(Some(d - LISTEN_FLOOR));
                }
                // The last stretch. `sleep` is the one timed primitive no platform rounds; the
                // zero-timeout wait keeps the listener drained while this park is deaf to it.
                Some(d) => {
                    self.transport.wait(Some(Duration::ZERO));
                    std::thread::sleep(d);
                }
            }
        }
    }
}

/// How a node's instance is BUILT, deferred so the building happens on the node's own thread. A
/// Python node's construction executes its module, and `Graph::add_node` holds the graph mutex.
pub type NodeBuild = Box<dyn FnOnce(&ParamGroups) -> Box<dyn Node> + Send>;

pub fn spawn(
    manifest: &'static NodeManifest,
    build: NodeBuild,
    params: ParamGroups,
    transport: Arc<dyn Transport>,
    env: NodeEnv,
    halt: Arc<Halt>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name(format!("goofi-{}", manifest.type_name))
        .spawn(move || {
            // A node removed inside its own build window never runs `setup()` — which may open a
            // device — and releases at once rather than after the import it no longer needs.
            if !halt.stopped() {
                // `run_forever` takes the runtime by value, so returning from it drops this node's
                // whole iceoryx2 end — which is why the flag is raised HERE and not in the loop.
                let node = build(&params);
                NodeRuntime::new(manifest, node, params, transport, env).run_forever(halt.clone());
            }
            halt.release();
        })
}

/// A `Data`'s total element count — the timeline discriminator, rather than a static per-slot
/// flag: a length-preserving transform's output matches its input's count, and nothing else does.
fn frame_count(d: &Data) -> usize {
    match d.value() {
        goofi_core::Value::Array(s) => s.shape().iter().product(),
        goofi_core::Value::Str(s) => s.chars().count(),
        goofi_core::Value::Table(m) => m.len(),
    }
}

/// Stamp the engine-owned meta on every frame just emitted, and answer the node's measured rate.
/// `index` follows the one matching TRIGGERING input, else a fresh per-slot counter; `ufreq` is the
/// node's own EMA, never inherited from upstream.
fn stamp_meta(
    manifest: &'static NodeManifest,
    inputs: &IndexMap<&'static str, Option<Data>>,
    outputs: &mut IndexMap<&'static str, Option<Data>>,
    now: f64,
    counters: &mut HashMap<&'static str, u64>,
    meter: &mut UfreqMeter,
) -> Option<f64> {
    // Nothing emitted → no meta to stamp, and the meter only advances on a productive emit.
    if outputs.values().all(|o| o.is_none()) {
        return None;
    }
    // Only triggering inputs carry the data timeline; control inputs are excluded.
    let triggering: std::collections::HashSet<&str> =
        manifest.inputs.iter().filter(|s| s.trigger_process).map(|s| s.name).collect();
    let input_frames: Vec<(u64, usize)> = inputs
        .iter()
        .filter(|(name, _)| triggering.contains(*name))
        .filter_map(|(_, o)| o.as_ref())
        .filter_map(|d| d.meta().index().map(|i| (i, frame_count(d))))
        .collect();
    // EMA of the inter-emit interval, inverted. `None` until the second emit; a non-advancing
    // clock (`dt <= 0`) keeps the prior estimate.
    let node_ufreq = match meter.last_emit {
        None => {
            meter.last_emit = Some(now);
            None
        }
        Some(prev) => {
            let dt = now - prev;
            meter.last_emit = Some(now);
            if dt > 0.0 {
                let ema = meter.ema.map_or(dt, |p| UFREQ_EMA_ALPHA * dt + (1.0 - UFREQ_EMA_ALPHA) * p);
                meter.ema = Some(ema);
                Some(1.0 / ema)
            } else {
                meter.ema.map(|e| 1.0 / e)
            }
        }
    };
    for (slot, slot_opt) in outputs.iter_mut() {
        let Some(d) = slot_opt else { continue };
        let of = frame_count(d);
        let mut matches = input_frames.iter().filter(|(_, f)| *f == of).map(|(i, _)| *i);
        let counter = counters.entry(*slot).or_insert(0);
        let index = match (matches.next(), matches.next()) {
            (Some(i), None) => i,
            _ => *counter,
        };
        // Keep the fresh counter past whatever was emitted: a slot that MATCHES on one frame and
        // then goes fresh would otherwise restart at 0 and regress the index at stream start.
        *counter = index + 1;
        *d = d.with_stamps(index, node_ufreq);
    }
    node_ufreq
}

/// Wall-clock milliseconds since the epoch — what a [`NodeFault`] timestamp is, because it travels
/// over the wire and [`Instant`] does not serialize.
fn now_ms() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

