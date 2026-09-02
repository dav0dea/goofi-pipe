//! The signal engine behind the seam: hosts, the wire planner and the library of the async
//! runtime. Its drain only COLLECTS — acks and readies mark planner state — and the settle that
//! follows every drain is the one place messages are composed, always against the settled view.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use goofi_node::{BoundVar, DrainWaker, Engine, EventId, GraphView, IsolationCell, LibraryEntry, NodeManifest, ParamGroups, ParamKey, Request, Status, Touched, Uid};

use crate::runtime::{
    self,
    plan::{Phase, Slot, SlotKey, WirePlanner},
};

/// A CEILING, not a join: a wedged node must not be able to wedge the exit. What one that misses
/// it leaves behind is what [`runtime::reclaim_stale_resources`] takes on the next startup.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(2);


/// One node's manager-side thread, and the graph's end of its wires. A node is *known* when
/// `insert` answers and *addressable* only once it reports Ready.
struct NodeHost {
    /// A flag rather than a `Control::Terminate`, because a node removed before it was
    /// addressable has no sink to receive one.
    halt: Arc<runtime::Halt>,
    /// `None` when the services could not be created: the node then exists carrying its boot
    /// error and nothing else.
    channel: Option<Arc<runtime::NodeChannel>>,
}

impl NodeHost {
    /// Never joined here: the thread may be inside a long `process()`, and every caller holds the
    /// graph mutex.
    fn signal_stop(&self) {
        self.halt.stop();
        if let Some(channel) = &self.channel {
            channel.wake();
        }
    }
}

impl Drop for NodeHost {
    fn drop(&mut self) {
        self.signal_stop();
    }
}

/// A [`goofi_signal_sdk::NodeFactory`] shared with the node's own thread, which is where the build happens.
type SharedFactory = Arc<dyn Fn(&ParamGroups) -> Box<dyn goofi_signal_sdk::Node> + Send + Sync>;

/// A registered type: its manifest, its tier cell and how an instance is built.
struct DynType {
    manifest: &'static NodeManifest,
    isolation: &'static IsolationCell,
    factory: SharedFactory,
}

pub struct SignalEngine {
    /// What service names are scoped by — handed down from the graph, whose resolver inputs it is.
    instance: String,
    evaluator: Option<Arc<dyn goofi_node::ExprEvaluator>>,
    started: Instant,
    waker: Arc<DrainWaker>,
    wire: WirePlanner,
    hosts: HashMap<Uid, NodeHost>,
    dyn_types: HashMap<&'static str, DynType>,
    /// The interpreters a `.py` file is probed and run with; none until the host provides them.
    pub(crate) python: Option<crate::scan::Python>,
    /// What the last probe of each file decided, at the stamp it decided it.
    pub(crate) probed: HashMap<std::path::PathBuf, (goofi_node::Stamp, crate::scan::Probed)>,
    /// Every built artifact loaded so far, by path: a library is opened once and never closed.
    pub(crate) rust_loaded: HashMap<std::path::PathBuf, Arc<goofi_signal_sdk::host::Loaded>>,
    /// Readies the drain collected; the settle that follows re-plans each from an empty base.
    pending_ready: Vec<Uid>,
    /// Sequences whose phase an ack completed; the settle that follows advances each.
    pending_advance: Vec<SlotKey>,
}

impl SignalEngine {
    pub fn new(instance: String, started: Instant, waker: Arc<DrainWaker>) -> SignalEngine {
        // What a crashed run left, reclaimed at engine construction rather than by whoever opens
        // the first port — which used to be the user's first add.
        goofi_transport::sweep_once();
        SignalEngine {
            instance,
            evaluator: None,
            started,
            waker,
            wire: WirePlanner::default(),
            hosts: HashMap::new(),
            dyn_types: HashMap::new(),
            python: None,
            probed: HashMap::new(),
            rust_loaded: HashMap::new(),
            pending_ready: Vec::new(),
            pending_advance: Vec::new(),
        }
    }

    /// Register a type; `manifest` leaks, once per type. A name another type held is REPLACED,
    /// because a rescan re-registers what it finds — answered as `true`.
    pub fn register_dyn_type(
        &mut self,
        manifest: &'static NodeManifest,
        factory: goofi_signal_sdk::NodeFactory,
        isolation: &'static IsolationCell,
    ) -> bool {
        let name = manifest.type_name;
        self.dyn_types.insert(name, DynType { manifest, isolation, factory: Arc::from(factory) }).is_some()
    }

    pub fn remove_dyn_type(&mut self, type_name: &str) -> bool {
        self.dyn_types.remove(type_name).is_some()
    }

    pub fn find_entry(&self, type_name: &str) -> Option<LibraryEntry> {
        self.dyn_types
            .get(type_name)
            .map(|dt| LibraryEntry { manifest: dt.manifest, isolation: dt.isolation })
    }

    /// Every consumer subscription of this engine's whose wiring names `uid`: the planner's own
    /// record of channels spoken on, the settled edges, and the bindings on either end.
    fn keys_touching(&self, view: &GraphView<'_>, uid: Uid) -> Vec<SlotKey> {
        let mut keys = self.wire.keys_for(uid);
        for e in view.edges.iter().filter(|e| e.producer.0 == uid || e.consumer.0 == uid) {
            let key = (e.consumer.0, Slot::In(e.consumer.1));
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        for (consumer, node) in &view.nodes {
            for b in &node.bindings {
                let touches = *consumer == uid
                    || b.vars.iter().filter_map(BoundVar::wire).any(|(p, _)| p == uid);
                let key = (*consumer, Slot::Bind(b.key.clone()));
                if touches && !keys.contains(&key) {
                    keys.push(key);
                }
            }
        }
        keys
    }

    fn replan(&mut self, view: &GraphView<'_>, key: SlotKey) {
        // Engines FILTER the whole-graph view: this engine plans only subscriptions its own
        // nodes hold; a foreign consumer's engine drains its boundary at its own clock.
        if view.nodes.get(&key.0).is_none_or(|n| n.engine != self.id()) {
            return;
        }
        let desired = desired_wires(view, &key);
        let previous = self.wire.planned(&key);
        // An In set that did not move carries nothing — a batch that ends where it started says
        // nothing. A Bind sequence still runs: its phase 2 IS the param delivery, wires or not.
        if matches!(key.1, Slot::In(_)) && desired == previous {
            return;
        }
        let removed = previous.iter().copied().filter(|w| !desired.contains(w)).collect();
        let added = desired.iter().copied().filter(|w| !previous.contains(w)).collect();
        // A begin cancels the key's previous sequence — and any ack collected for it, or a stale
        // deferred advance would step the NEW sequence past a phase nobody acked.
        self.pending_advance.retain(|k| k != &key);
        self.wire.begin(key.clone(), desired, removed, added);
        self.advance(view, key);
    }

    /// Walk the phases until one has something to send. A phase with no recipients is skipped
    /// rather than sent empty, or the sequence would park on an ack that never comes.
    fn advance(&mut self, view: &GraphView<'_>, key: SlotKey) {
        while let Some(phase) = self.wire.step(&key) {
            let messages = self.compose_wire(view, &key, phase);
            if self.wire.dispatch(&key, messages) {
                return;
            }
        }
    }

    /// One phase's messages, composed from the settled view as it stands NOW; `Apply` carries the
    /// sequence's own desired set, which must not shift under it.
    fn compose_wire(
        &self,
        view: &GraphView<'_>,
        key: &SlotKey,
        phase: Phase,
    ) -> Vec<(Uid, runtime::Control)> {
        match phase {
            // Phase 2 is the SUBSCRIBE, whichever kind of consumer this is — an input slot's full
            // service set, or a binding's whole re-resolved expression. Both are declarative.
            Phase::Apply => match &key.1 {
                Slot::In(slot) => {
                    let services = self
                        .wire
                        .desired(key)
                        .iter()
                        .filter_map(|(uid, slot)| output_service_in(view, *uid, slot))
                        .collect();
                    vec![(key.0, runtime::Control::InSlot { slot: slot.to_string(), services })]
                }
                Slot::Bind(k) => self.compose_set_param(view, key.0, k).into_iter().collect(),
            },
            Phase::Shrink | Phase::Grow => self
                .wire
                .recipients(key, phase)
                .into_iter()
                .map(|(uid, slot)| {
                    let targets = self.out_targets(view, uid, slot);
                    (uid, runtime::Control::OutSlot { slot: slot.to_string(), targets })
                })
                .collect(),
        }
    }

    /// The `SetParam` a binding's phase 2 carries: the rewritten source while the binding stands,
    /// and the param's LITERAL once it does not, which is what says the binding is gone.
    fn compose_set_param(
        &self,
        view: &GraphView<'_>,
        uid: Uid,
        key: &ParamKey,
    ) -> Option<(Uid, runtime::Control)> {
        let node = view.nodes.get(&uid)?;
        let value = match node.bindings.iter().find(|b| b.key == key).filter(|b| b.live) {
            Some(b) => runtime::ParamValue::Expr {
                source: b.rewritten.to_string(),
                vars: b.vars.iter().map(|v| wire_var(view, v)).collect(),
                trigger: b.trigger,
                // The graph compiled it, the node evaluates it (§2.1) — one handle, so the two
                // ends can never be evaluating different source.
                id: b.id,
            },
            None => runtime::ParamValue::Literal(
                goofi_node::param(node.params, &key.group, &key.name)?.clone(),
            ),
        };
        Some((uid, runtime::Control::SetParam { key: key.clone(), value }))
    }

    /// Every doorbell one output slot rings, with the event id that says why the far node woke —
    /// the union of its wire consumers and its expression subscribers (§5.3). A consumer whose
    /// engine drains at its own tick is never rung.
    fn out_targets(
        &self,
        view: &GraphView<'_>,
        producer: Uid,
        slot: &'static str,
    ) -> Vec<(runtime::ServiceName, EventId)> {
        // The ordering guarantee is per TARGET, not per sequence: a consumer whose own sequence
        // has not applied this wire is not a subscriber yet, which is what the phases prevent.
        let wired = view
            .edges
            .iter()
            .filter(|e| e.producer == (producer, slot))
            .filter(|e| {
                !self.wire.unapplied(&(e.consumer.0, Slot::In(e.consumer.1)), (producer, slot))
            })
            .filter_map(|e| {
                let node = view.nodes.get(&e.consumer.0)?;
                if !node.rings {
                    return None;
                }
                let at = node.manifest.inputs.iter().position(|s| s.name == e.consumer.1)?;
                let id = (at < 64).then_some(at as EventId + 1)?;
                Some((door_of(view, e.consumer.0)?, id))
            })
            .collect::<Vec<_>>()
            .into_iter();
        let bound = view.nodes.iter().flat_map(|(consumer, node)| {
            node.bindings.iter().filter(|b| b.live).flat_map(move |b| {
                b.vars
                    .iter()
                    .filter(move |v| v.wire() == Some((producer, slot)))
                    .filter(move |_| {
                        !self
                            .wire
                            .unapplied(&(*consumer, Slot::Bind(b.key.clone())), (producer, slot))
                    })
                    .filter_map(move |v| match v {
                        BoundVar::Stream { event_id, .. } if node.rings => {
                            Some((door_of(view, *consumer)?, *event_id))
                        }
                        _ => None,
                    })
            })
        });
        wired.chain(bound).collect()
    }
}

/// A consumer subscription's desired producers, read off the settled view.
fn desired_wires(view: &GraphView<'_>, key: &SlotKey) -> Vec<(Uid, &'static str)> {
    match &key.1 {
        Slot::In(slot) => {
            // A slot past the event-id budget takes no wires, exactly as it takes no rings.
            let in_budget = view
                .nodes
                .get(&key.0)
                .and_then(|n| n.manifest.inputs.iter().position(|s| s.name == *slot))
                .is_some_and(|at| at < 64);
            if !in_budget {
                return Vec::new();
            }
            view.wires_into(key.0, slot).collect()
        }
        Slot::Bind(k) => view
            .nodes
            .get(&key.0)
            .and_then(|n| n.bindings.iter().find(|b| b.key == k))
            .filter(|b| b.live)
            .map(|b| b.vars.iter().filter_map(BoundVar::wire).collect())
            .unwrap_or_default(),
    }
}

/// One output slot's data service name, from the view's birth facts.
fn output_service_in(view: &GraphView<'_>, uid: Uid, slot: &str) -> Option<runtime::ServiceName> {
    let node = view.nodes.get(&uid)?;
    Some(goofi_transport::output_service(
        &goofi_transport::service_base(view.instance, uid, node.generation),
        slot,
    ))
}

fn door_of(view: &GraphView<'_>, uid: Uid) -> Option<runtime::ServiceName> {
    let node = view.nodes.get(&uid)?;
    Some(goofi_transport::door_service(&goofi_transport::service_base(view.instance, uid, node.generation)))
}

/// A resolved variable as the NODE sees it: a service name rather than a uid, because a node
/// addresses a producer by service and cannot resolve anything for itself (§5.3).
fn wire_var(view: &GraphView<'_>, var: &BoundVar) -> runtime::Var {
    match var {
        BoundVar::Stream { var, producer, slot, event_id } => runtime::Var::Stream {
            name: var.clone(),
            service: output_service_in(view, *producer, slot).unwrap_or_default(),
            event_id: *event_id,
        },
        BoundVar::Value { var, value } => {
            runtime::Var::Value { name: var.clone(), value: value.clone() }
        }
        BoundVar::Missing { var, reason } => {
            runtime::Var::Missing { name: var.clone(), reason: reason.clone() }
        }
    }
}

impl Engine for SignalEngine {
    fn id(&self) -> &'static str {
        "signal"
    }

    fn doorbell_driven(&self) -> bool {
        true
    }

    fn dirty(&self) -> bool {
        !self.pending_ready.is_empty() || !self.pending_advance.is_empty()
    }

    fn scan(&mut self, dir: &std::path::Path) -> Vec<goofi_node::ScannedType> {
        crate::scan::scan(self, dir)
    }

    fn remove_type(&mut self, type_name: &str) -> bool {
        self.remove_dyn_type(type_name)
    }

    fn rust_sdk(&self) -> Option<&'static str> {
        Some(goofi_build::SIGNAL.name)
    }

    fn library(&self) -> Vec<LibraryEntry> {
        self.dyn_types
            .values()
            .map(|dt| LibraryEntry { manifest: dt.manifest, isolation: dt.isolation })
            .collect()
    }

    fn normalize_params(
        &self,
        type_name: &str,
        supplied: Option<ParamGroups>,
    ) -> Result<ParamGroups, String> {
        let entry = self
            .find_entry(type_name)
            .ok_or_else(|| format!("no node type `{type_name}` in the signal library"))?;
        let base = supplied.unwrap_or_else(|| entry.manifest.default_params());
        Ok(crate::with_common(base, entry.manifest))
    }

    fn insert(
        &mut self,
        uid: Uid,
        type_name: &str,
        generation: u64,
        params: &ParamGroups,
    ) -> Option<String> {
        let build: runtime::NodeBuild = match self.dyn_types.get(type_name) {
            Some(dt) => {
                let f = dt.factory.clone();
                Box::new(move |p| f(p))
            }
            None => return Some(format!("no node type `{type_name}` in the signal library")),
        };
        let manifest = self.find_entry(type_name).expect("resolved above").manifest;
        let halt = Arc::new(runtime::Halt::default());
        let base = goofi_transport::service_base(&self.instance, uid, generation);
        let started = runtime::IoxTransport::create(&self.instance, uid, generation, manifest)
            .and_then(|transport| Ok((transport, runtime::NodeChannel::open(&base)?)))
            .and_then(|(transport, channel)| {
                let env = runtime::NodeEnv {
                    evaluator: self.evaluator.clone(),
                    started: self.started,
                };
                let transport = Arc::new(runtime::WakingTransport {
                    inner: Arc::new(transport),
                    waker: self.waker.clone(),
                });
                // The join handle is dropped on purpose: holding one would tempt a caller into
                // joining under the graph mutex while the node is inside a long `process()`.
                runtime::spawn(manifest, build, params.clone(), transport, env, halt.clone())
                    .map(|_| channel)
                    .map_err(|e| format!("could not start the node's thread: {e}"))
            });
        let (host, boot_error) = match started {
            Ok(channel) => (NodeHost { halt, channel: Some(Arc::new(channel)) }, None),
            Err(e) => (NodeHost { halt, channel: None }, Some(e)),
        };
        self.hosts.insert(uid, host);
        boot_error
    }

    fn remove(&mut self, uid: Uid) {
        // Dropping the host halts the corpse's thread without waiting; the wire state and any
        // held request go with it, so a successor at this uid starts clean.
        self.hosts.remove(&uid);
        self.wire.forget(uid);
        self.pending_ready.retain(|u| *u != uid);
        self.pending_advance.retain(|(u, _)| *u != uid);
    }

    fn settle(&mut self, view: &GraphView<'_>, touched: &[Touched]) {
        // Readies first: an attach re-plans from an EMPTY base, and what it begins must not be
        // clobbered by this batch's own touches.
        for uid in std::mem::take(&mut self.pending_ready) {
            for key in self.keys_touching(view, uid) {
                self.wire.forget_planned(&key);
                self.replan(view, key);
            }
        }
        for t in touched {
            match t {
                Touched::Slot(uid, slot) => self.replan(view, (*uid, Slot::In(slot))),
                Touched::Param(uid, key) => self.replan(view, (*uid, Slot::Bind(key.clone()))),
            }
        }
        for key in std::mem::take(&mut self.pending_advance) {
            self.advance(view, key);
        }
    }

    fn drain(&mut self, apply: &mut dyn FnMut(Uid, Status)) -> usize {
        let channels: Vec<(Uid, Arc<runtime::NodeChannel>)> = self
            .hosts
            .iter()
            .filter_map(|(uid, h)| h.channel.clone().map(|c| (*uid, c)))
            .collect();
        let mut applied = 0;
        for (uid, channel) in channels {
            for status in channel.drain_status() {
                applied += 1;
                match status {
                    // An ack is the planner's, and it must still land after the node it came
                    // from is gone — or a sequence parks forever on an unanswered message.
                    runtime::WireStatus::Ack { seq, ok } => {
                        if let Some(key) = self.wire.ack(seq, ok) {
                            self.pending_advance.push(key);
                        }
                    }
                    // The birth barrier lifting: the node is addressable, and the settle that
                    // follows this drain re-plans everything it touches.
                    runtime::WireStatus::Ready => {
                        if let Some(c) = self.hosts.get(&uid).and_then(|h| h.channel.clone()) {
                            self.wire.attach(uid, c);
                        }
                        self.pending_ready.push(uid);
                    }
                    runtime::WireStatus::Health(status) => apply(uid, status),
                }
            }
        }
        applied
    }

    fn request(&mut self, uid: Uid, request: Request) {
        match request {
            Request::RefreshParam { key } => {
                self.wire.send(uid, runtime::Control::RefreshParam { key });
            }
        }
    }

    /// Every node born after computes from the new origin.
    fn reset_clock(&mut self, origin: Instant) {
        self.started = origin;
    }

    fn set_evaluator(&mut self, evaluator: Arc<dyn goofi_node::ExprEvaluator>) {
        self.evaluator = Some(evaluator);
    }

    /// The `common` scheduling group: signal semantics, added to every signal node.
    fn universal_decls(&self, manifest: &'static NodeManifest) -> Vec<goofi_node::ParamDecl> {
        crate::common_decls(manifest).collect()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Stop every node and WAIT for each to release its shared memory — a ceiling, because only a
    /// process about to EXIT has no "a moment later".
    fn shutdown(&mut self) {
        for host in self.hosts.values() {
            host.signal_stop();
        }
        let deadline = Instant::now() + SHUTDOWN_WAIT;
        while self.hosts.values().any(|h| !h.halt.released()) {
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.hosts.clear();
        self.wire.reset_channels();
        self.pending_ready.clear();
        self.pending_advance.clear();
    }
}
