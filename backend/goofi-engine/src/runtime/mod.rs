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
//! A [`NodeRuntime`] owns one node against a [`Transport`], and [`spawn`] gives it the thread it
//! runs on. [`Graph`] builds one per node at birth, plans its wiring ([`plan`]) and reads back what
//! it reports ([`Status`]); it never runs a node itself. There is no tick.
//!
//! ## The ledger the cutover closed
//!
//! Every item the four preceding tasks deferred here is closed, and the closures are stated rather
//! than left to be re-derived:
//!
//! 1. **Meta stamping** — [`stamp_meta`] runs in [`NodeRuntime::run`], on the frames the node just
//!    emitted, before they are published. `index` and `ufreq` stay engine-owned.
//! 2. **`last_outputs`** — dropped on purpose with `Graph::latest_frame` (§7). A viewer subscribes
//!    to `/data` like any other consumer, so there is nothing left that reads a node for a frame.
//! 3. **The required-input gate** — [`NodeRuntime::missing_required`], checked over the WIRE CELLS
//!    rather than over `inputs`: a `multi` slot's frames never enter `inputs`, so a gate reading
//!    that map alone passes every node with a required multi slot.
//! 4. **`ensure_initialized`** — one gate, here. The graph-side copy is gone with the tick.
//! 5. **Frames reach a node from its own wires** — the wake loop drains
//!    [`Transport::drain_inputs`], which is the only door a frame comes in through, for declared
//!    slots and for an expression variable's pseudo-slot alike.
//! 6. **The birth barrier** — a node publishes [`Status::Ready`] once its own services exist, and
//!    the graph attaches its control sink on that report and not before. A `Control` sent earlier
//!    would be published to a subscriber that does not exist yet, and pub/sub has no history. The
//!    barrier is a WINDOW, though, and `add_node` answers inside it: what was said during it is
//!    re-PLANNED on attach for anything with graph state to re-derive it from (item 9), and HELD
//!    for anything without — a `RefreshParam` is a request, not a state, so `WirePlanner::send`
//!    queues it rather than dropping it.
//! 7. **Expressions are evaluated** — in the node, immediately before the run that reads them
//!    (§2.1), from the mailboxes its variables name.
//! 8. **[`Binding::evaluate`] holds the evaluator** — a rewritten source richer than one variable
//!    goes to [`goofi_node::ExprEvaluator`] with §5.3's locals channel.
//! 9. **`replan_slot`'s callers** — birth, removal, restart and load join the link and binding
//!    changes, all of them through `Graph::attach_control_sink` / `Graph::slots_touching`. What
//!    that walk names is every subscription touching the node — its input slots, the bindings on
//!    either end of it, and **every param channel the graph has ever spoken on for it**. The last
//!    is not a nicety: a plain literal param is neither a link nor a binding, so without it the
//!    ordinary `add_node(); update_param()` pair fell into the birth window and was lost.
//! 10. **`WirePlanner::is_idle`** — deleted. Every live node has a channel.
//! 11. **`spawn_stats`** — the worker stayed; its POLLING went. It drains at 1 ms and broadcasts at
//!     2 Hz, because the drain is the runtime's clock and the broadcast is the UI's. Every one
//!     of the four events it sourced (`error`, `node_stage`, `node_stats`, `param_values`) is now a
//!     node's own report: [`Status::Fault`] / [`Status::BindingErrors`], [`Status::Stage`],
//!     [`Status::Ufreq`] and [`Status::ParamValues`], applied by `Graph::apply_status`. Every one
//!     is a TRANSITION the node stamps itself, which is why nothing here diffs — the exception is
//!     `Ufreq`, a measurement rather than a transition, paced at [`UFREQ_REPORT_MS`]. The worker
//!     that drains them, and its rule against dirtying the patch, belong to the bridge.
//!
//! Two things were DROPPED on purpose rather than deferred, and are listed so neither reads as an
//! oversight:
//!
//! - **`scheduling_edges` no longer lifts `nd()` references into the topo DAG.** That ordering
//!   existed so a referenced producer ran earlier in the same tick and the expression saw
//!   this-tick's value. §2.1 moved evaluation into the node, so the lifting ordered nothing and the
//!   guarantee it bought no longer exists to buy. The whole tick went with it.
//! - **The pyo3 evaluator's `nd()` proxy and `globals` namespace.** The rules they enforced did not
//!   all retire with them, and the ones that MOVED are pinned at their new home rather than left to
//!   the reader: a bare `nd()` on a multi-output producer, an unknown `.slot`, and a `globals.` name
//!   the patch does not define are now refused by `Graph::resolve_stream` / `resolve_vars` at bind
//!   time (pinned in `goofi-engine`), and "a variable that has not arrived raises" is pinned in
//!   `goofi-python`. What genuinely retired: the proxy's ~28 operator dunders (a numpy array
//!   supports them natively), its key-absent-vs-value-`None` distinction (there is one `Option` now),
//!   `_Globals`' `__getattribute__` shim (a global's NAME never reaches Python any more), and
//!   `Compiled`'s `refs`/`global_refs` extraction (the compiled source names neither).
//!
//! [`Graph`]: crate::Graph

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use goofi_core::globals::GlobalsSnapshot;
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
    door_service, iox_node, open_output_subscriber, output_service, reclaim_stale_resources,
    service_base, ByteSubscriber, Doorbell, IoxNode, IoxTransport, NodeChannel,
};
/// Only [`Graph`] mints a scope, and it is in this crate.
pub(crate) use transport::service_instance;
pub use wire::{
    Control, ControlSink, Envelope, EventId, NodeStage, ParamValue, ServiceName, Status, Transport,
    Var, VarName,
};

/// The scheduling namespace. A `common.*` param decides *when* a node runs, so it is resolved
/// before the gates are read and never inside a run (§1.1).
const COMMON: &str = "common";

/// `SETUP_RETRY_INTERVAL` in the wall-clock milliseconds a [`NodeFault`] carries, so the interval
/// is stated once for the whole engine rather than once per clock.
const SETUP_RETRY_MS: f64 = crate::SETUP_RETRY_INTERVAL * 1000.0;

/// How often a node reports its measured update rate. Every other [`Status`] is a transition and is
/// sent when it happens; a rate is a MEASUREMENT and an uncapped producer takes one per emit, so
/// reporting each would flood the status service with what the bridge coalesces to 10 Hz anyway.
const UFREQ_REPORT_MS: u128 = 250;

/// How long a parked node waits before looking again when nothing is due. A wake is a hint (§3.3),
/// so a missed doorbell costs at most this — and it is what makes a node notice its [`Halt`].
const PARK_CEILING: Duration = Duration::from_millis(50);

/// The smoothing factor of the `ufreq` EMA: how much the newest inter-emit interval moves the
/// estimate. Low enough that a single slow frame does not swing the readout, high enough that a
/// real rate change is visible within a second.
const UFREQ_EMA_ALPHA: f64 = 0.2;

/// The `ufreq` meter's state: when this node last emitted, and the smoothed interval between emits.
#[derive(Default)]
struct UfreqMeter {
    last_emit: Option<f64>,
    ema: Option<f64>,
}

/// The one thing that stops a node's manager-side thread. See [`wire`]'s note on why this is a flag
/// and not a `Control::Terminate`: a node removed before it answered [`Status::Ready`] has no sink,
/// and a whole-graph `clear` has no sequence to order.
#[derive(Default)]
pub struct Halt {
    stop: AtomicBool,
    /// Set by the node's own thread once its [`NodeRuntime`] — and with it every iceoryx2 port and
    /// the node behind them — has been DROPPED. That drop is what releases the node's shared
    /// memory, and it is the only thing a teardown can usefully wait for: the halt flag says the
    /// thread was asked, this says it is done.
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

/// The pseudo input slot a bound param's producer wires ride. A binding subscribes to a producer
/// exactly as an input slot does (§5.3), so it goes through the one subscribe door rather than a
/// second one — and the name is namespaced with a character no declared slot may carry, so it can
/// never collide with one.
pub fn expr_wire_slot(key: &ParamKey) -> String {
    format!("expr:{}:{}", key.group, key.name)
}

/// The [`ParamKey`] an [`expr_wire_slot`] name refers back to, or `None` for a declared slot.
fn expr_wire_key(slot: &str) -> Option<ParamKey> {
    let rest = slot.strip_prefix("expr:")?;
    let (group, name) = rest.split_once(':')?;
    Some(ParamKey::new(group, name))
}

/// What is wrong with a node. `None` is healthy.
///
/// Four variants because `entry_error` folds four sources, and wall-clock `f64` rather than
/// [`Instant`] because a fault is reported over the wire.
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
    /// The evaluator that compiled this node's bindings, shared with the graph that compiled them
    /// (§2.1 evaluates in the node; `set_expression` compiles in the graph so the authoring RPC can
    /// answer with a real compile error). `None` on a graph with no evaluator injected, where every
    /// bound param falls back to its literal.
    evaluator: Option<Arc<dyn ExprEvaluator>>,
    /// The patch globals as the node reads them (§5.2): a lock-free handle onto the graph's record,
    /// re-read before each run so `process` sees an edit on its next run rather than never.
    globals: Arc<ArcSwap<GlobalsSnapshot>>,
    /// The graph's clock origin, so `NodeCtx::now` is seconds-since-start on every node's thread
    /// rather than seconds-since-this-node's-birth.
    started: Instant,

    /// Path B lives in `run_policy.autotrigger`, beside the cap that paces it — one value derived
    /// by one `RunPolicy::from_params`, so there is nothing to keep in step.
    ///
    /// Paths A and C: something asked this node to run and it has not run since.
    pub(crate) trigger_pending: bool,
    pub(crate) run_policy: RunPolicy,
    pub(crate) last_run: Option<Instant>,
    /// Set by ANY arrival that can affect a `common.*` param, whatever path it came in on (§1.1).
    pub(crate) common_dirty: bool,

    /// The param RECORD: literals only, which is what the `.gfi` persists and what a broken or
    /// not-yet-arrived binding falls back to (§2.1). Kept apart from `effective` because an
    /// evaluated value would otherwise erase the number the user authored.
    pub(crate) literals: ParamGroups,
    /// The node's FULL params — the literal record overlaid with evaluated bindings. What
    /// `process()` reads and what `RunPolicy::from_params` is given.
    pub(crate) effective: ParamGroups,
    /// The SPARSE bound subset, which exists only as the wire projection in
    /// [`Status::ParamValues`]. Handing THIS to `RunPolicy::from_params` would silently default
    /// every absent key — which is why the two maps are named apart (§2).
    pub(crate) evaluated: IndexMap<ParamKey, Param>,
    pub(crate) bindings: IndexMap<ParamKey, Binding>,

    /// Latest-wins input cells, one per declared single input slot.
    pub(crate) inputs: IndexMap<&'static str, Option<Data>>,
    /// Per-WIRE latest-wins cells for each declared `multi` input slot, keyed by the producer's
    /// service name and held in the order the last `InSlot` set named — which IS
    /// `Inputs::get_multi`'s connection order (§3.5). Kept apart from [`Self::inputs`] because the
    /// two partition the manifest's slots, and because the required-input gate has to see both:
    /// a gate reading only `inputs` passes every node with a required multi slot.
    pub(crate) multi_wires: IndexMap<&'static str, Vec<(ServiceName, Option<Data>)>>,
    pub(crate) ctx: NodeCtx,
    /// Per-output-slot emit counter for `meta["index"]` — engine-owned, the node never sees it.
    index_counters: HashMap<&'static str, u64>,
    /// Per-NODE measured update rate for `meta["ufreq"]`: one meter, stamped onto every slot this
    /// node emits, because ufreq describes the node rather than a slot.
    ufreq_meter: UfreqMeter,
    /// When the rate was last REPORTED, which is not when it was last measured — see
    /// [`UFREQ_REPORT_MS`].
    last_ufreq_report: Option<Instant>,
    stage: NodeStage,

    pub(crate) fault: Option<NodeFault>,
    /// Binding errors are a MAP, not a fault variant: several bindings can be errored at once and
    /// each renders on its own inspector field. [`NodeFault::Expr`] is the derived node-level
    /// roll-up, not the record: the GRAPH folds this map into the one node-level badge, ordering
    /// by key (`entry_error`).
    pub(crate) binding_errors: HashMap<ParamKey, String>,
    initialized: bool,
}

/// Everything a node's thread needs that is the GRAPH's rather than the node's: the shared records
/// it reads through, the evaluator that compiled its bindings, and the clock origin. One struct
/// because every one of them is handed over at birth and none of them is ever replaced — a restart
/// builds a new runtime rather than mutating this.
pub struct NodeEnv {
    pub evaluator: Option<Arc<dyn ExprEvaluator>>,
    pub globals: Arc<ArcSwap<GlobalsSnapshot>>,
    pub started: Instant,
}

impl NodeEnv {
    /// The environment of a node that belongs to no graph — what a test driving a [`NodeRuntime`]
    /// directly gets, and what `NodeCtx::default` already meant by empty globals.
    pub fn detached() -> NodeEnv {
        NodeEnv {
            evaluator: None,
            globals: Arc::new(ArcSwap::from_pointee(GlobalsSnapshot::default())),
            started: Instant::now(),
        }
    }
}

impl NodeRuntime {
    /// Build a node from its manifest and initialize it: seed its params, then `setup()`. A
    /// failing `setup` leaves the node UNINITIALIZED with a [`NodeFault::Setup`] standing, and
    /// nothing runs against it until a retry succeeds.
    ///
    /// `params` is the record the graph holds for this node — a fresh add's type defaults, or a
    /// load's saved values — so the node's own literals and the `.gfi` are the same numbers from
    /// its first `setup()` rather than from its first param message.
    pub fn new(
        manifest: &'static NodeManifest,
        node: Box<dyn Node>,
        params: ParamGroups,
        transport: Arc<dyn Transport>,
        env: NodeEnv,
    ) -> NodeRuntime {
        let effective = goofi_node::with_common(params, manifest);
        let run_policy = RunPolicy::from_params(&effective);
        let mut ctx = NodeCtx::new();
        // `setup` latches the globals as of birth; `process` re-reads them before every run.
        ctx.globals = (**env.globals.load()).clone();
        let mut runtime = NodeRuntime {
            manifest,
            node,
            transport,
            evaluator: env.evaluator,
            globals: env.globals,
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
            ctx,
            index_counters: HashMap::new(),
            ufreq_meter: UfreqMeter::default(),
            last_ufreq_report: None,
            stage: NodeStage::Setup,
            fault: None,
            binding_errors: HashMap::new(),
            initialized: false,
        };
        // §4's birth barrier: the graph addresses nothing until this lands, because the control
        // SUBSCRIBER now exists and a message published before it did would simply be lost. Sent
        // before `setup()` runs, so a `setup` that fails at birth still has somewhere to report to.
        runtime.transport.report(Status::Ready);
        runtime.transport.report(Status::Stage { stage: NodeStage::Setup });
        runtime.initialize();
        runtime.publish_stage();
        runtime
    }

    /// Announce the node's lifecycle stage when it CHANGED. `error` is the graph's derivation from
    /// the fault and is deliberately not a stage a node can claim, so an initialized node is
    /// `ready` whatever its last `process` did.
    fn publish_stage(&mut self) {
        let next = if self.initialized { NodeStage::Ready } else { NodeStage::Setup };
        if next != self.stage {
            self.stage = next;
            self.transport.report(Status::Stage { stage: next });
        }
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
    /// event ids, and [`Self::run_forever`] wraps this rather than replacing it.
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

    /// Apply every waiting control message and ack each one. The ack is what tells the graph this
    /// node is in sync with it (§3.4) — and the wire sequence's phases are ordered by nothing else,
    /// so a message applied but not acked stalls the change that sent it.
    ///
    /// The slot messages are handed to the transport rather than acted on here: which subscribers a
    /// wire needs is the transport's whole subject, and the runtime's is when to run.
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
    /// third-party code and it runs here rather than under the graph lock, which is the whole point
    /// of the move — but a panic in it still has to become a report rather than kill this thread.
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
            // The record moves too, so the next `serialize` and the next inspector read agree with
            // what was just reported rather than with the type's declaration.
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

    /// Apply a slot's new wire set to the node's OWN cells. The transport keeps the subscribers;
    /// these are the frames already in hand, and what happens to them is the runtime's call: a
    /// surviving wire keeps its frame, a wire that left takes its frame with it.
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
                // An unbind drops the binding's subscriptions with it — the producer is told to
                // stop ringing by the graph's phase 1, and this is the consumer's own half.
                if self.bindings.shift_remove(&key).is_some() {
                    let _ = self.transport.wire_in(&expr_wire_slot(&key), &[]);
                }
                self.evaluated.shift_remove(&key);
                let cleared = self.record_binding_error(&key, None);
                self.report_binding_errors(cleared.into_iter().collect());
                self.set_literal(&key, p.clone());
                Some(p)
            }
            ParamValue::Expr { source, vars, trigger, id } => {
                let binding = Binding::new(source, vars, trigger, id);
                // §5.3: an expression reference IS a link, so its producers are subscribed through
                // the one subscribe door every wire goes through — on a pseudo-slot named after the
                // param, which is what lets a frame arriving there address a variable.
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

    /// Paths A and C's data half — a frame off one of this node's wires, addressed by the slot it
    /// arrived on and that wire's position in the slot's service set.
    ///
    /// One door for both, because a bound param subscribes through `wire_in` exactly as an input
    /// slot does (§5.3): an [`expr_wire_slot`] name lands in a binding's mailbox, a declared slot
    /// name in an input cell. A `trigger_process` slot wakes the node; a reference slot updates the
    /// cell and nothing more.
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
                // A wire index the last `InSlot` set does not name — a frame that crossed the
                // sequence that removed it. Dropped rather than appended: appending would put the
                // frame at a position `Inputs::get_multi` reads as a different producer's.
                None => return,
            }
        } else {
            self.inputs.insert(decl.name, Some(frame));
        }
        if decl.trigger_process {
            self.trigger_pending = true;
        }
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
            // §2.1: the target param is the type template the evaluator coerces its result to,
            // and it is the LITERAL — the number the user authored — never the last evaluated
            // value, which would let a binding drift its own type from one run to the next.
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
        // Live globals and the graph's clock for `process`; `setup` latched the globals at birth.
        self.ctx.now = self.started.elapsed().as_secs_f64();
        self.ctx.globals = (**self.globals.load()).clone();
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
                // stands, so a clean run PROVES setup succeeded. It does NOT clear a binding
                // error, which only that binding evaluating successfully clears.
                self.set_fault(None);
                // The engine's own meta goes on before anything leaves the node, so a consumer and
                // a viewer read the same `index`/`ufreq` — there is no second stamping site.
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

    /// The name of a `required` input slot holding no frame, or `None` when every one of them is
    /// fed. Read over the WIRE CELLS rather than over `inputs`: a `multi` slot's frames live in
    /// [`Self::multi_wires`], so a gate reading `inputs` alone silently passes every node with a
    /// required multi slot.
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
            // One clock read: the attempt IS when the failure happened, and `set_fault` keeps the
            // `since` of a fault that has not changed anyway.
            Some(msg) => {
                self.set_fault(Some(NodeFault::Setup { msg, since: attempt, last_attempt: attempt }))
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
    /// Run the node until it is halted — the body of its manager-side thread (§2). One loop for
    /// every execution kind: a native Rust node, an in-process Python node and a subprocess node's
    /// proxy differ only in what `process()` does.
    pub fn run_forever(mut self, halt: Arc<Halt>) {
        while !halt.stopped() {
            self.run_once();
            match self.next_wake() {
                // Nothing due: park until something rings, or until the ceiling makes the node
                // look at its halt flag again.
                None => {
                    self.transport.wait(Some(PARK_CEILING));
                }
                // Due now — an uncapped free-runner. Looping straight back is what §8.9 means by
                // "an uncapped producer saturates a core"; parking on a zero timeout would add a
                // syscall per run and change nothing.
                Some(d) if d.is_zero() => continue,
                Some(d) => {
                    self.transport.wait(Some(d.min(PARK_CEILING)));
                }
            }
        }
    }
}

/// Give a node its own thread (§5: every node has a manager-side one).
///
/// The [`NodeRuntime`] is built INSIDE the thread, which is what takes `setup()` off the graph
/// lock: a `setup` that opens a device or dials a socket is exactly the one that blocks, and the
/// caller has already answered its RPC with the node as born. The transport is built by the caller
/// instead, because creating it is the one step whose failure has nowhere to be reported to.
pub fn spawn(
    manifest: &'static NodeManifest,
    node: Box<dyn Node>,
    params: ParamGroups,
    transport: Arc<dyn Transport>,
    env: NodeEnv,
    halt: Arc<Halt>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name(format!("goofi-{}", manifest.type_name))
        .spawn(move || {
            // `run_forever` takes the runtime by value, so returning from it drops this node's
            // whole iceoryx2 end. Only after that is the node's shared memory actually gone, which
            // is why the flag is raised HERE and not inside the loop.
            NodeRuntime::new(manifest, node, params, transport, env).run_forever(halt.clone());
            halt.release();
        })
}

/// The number of frames a `Data` spans — its total element count (numpy `.size` for an array, `len`
/// for a string/table). This, not a static per-slot flag, is the timeline discriminator: a
/// length-preserving transform's output matches its input's frame count; a generator or
/// length-changing transform does not.
fn frame_count(d: &Data) -> usize {
    match d.value() {
        goofi_core::Value::Array(s) => s.shape().iter().product(),
        goofi_core::Value::Str(s) => s.chars().count(),
        goofi_core::Value::Table(m) => m.len(),
    }
}

/// Stamp the engine-owned meta — `index` and `ufreq` — on every frame this node just emitted (the
/// node never touches either), and answer with the node's measured rate.
///
/// **index**: for each output, propagate the index of the SINGLE index-bearing TRIGGERING input
/// whose frame count equals the output's — that input is the same data timeline, so an upstream
/// drop stays visible downstream. A non-triggering (control/reference) input — an oscillator's
/// scalar frequency, say — is never a timeline candidate even if its length happens to match. With
/// zero, or more than one, matching inputs (a generator, a length-changing transform, or an
/// ambiguous fan-in) the slot starts a fresh per-output counter that advances one per emit.
///
/// **ufreq**: the NODE's measured update rate (Hz) — an EMA of the inter-emit interval keyed on
/// `ctx.now`, `None` until a second emit gives one interval. Measured PER NODE, advanced once per
/// productive run, and the same value stamped onto every emitted slot: ufreq describes how often
/// the node updates, not a per-slot cadence. Authoritative — overwritten every emit, never
/// inherited from upstream meta.
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
        // Keep the fresh counter monotonically past whatever we emitted. Without this, a slot that
        // MATCHES on one frame (an accumulator's first output length equals its input length) then
        // goes fresh would restart the counter at 0 — duplicating or regressing the index at stream
        // start (the Oscillator→Buffer reference patch).
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

