//! The control half of a scheduled engine: one thread per node, parked on the node's own door.
//! It is the one writer of the node's param atomics — a constant and an evaluated binding land
//! through the same hand — the crossing every Array input enters through, and the tap every
//! reader of an output drinks from.
//!
//! What an arrival BECOMES and what a tap publishes is the engine's, behind [`Half`]. Everything
//! above that — the thread, the door, the desired state, the bindings, the reports — is here.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use goofi_core::{Data, Param};
use goofi_node::{
    BindingId, DrainWaker, EventId, ExprEvaluator, Expression, NodeManifest, ParamDecl, ParamGroups, ParamKey,
    Status, Uid, Var,
};
use goofi_transport::{
    data_service, door_service, event_service, iox_node, open_output_subscriber, output_service, publisher,
    take_where, ByteService, BytePublisher, ByteSubscriber, Doorbell, Halt, IoxNode, Listener, INITIAL_SLICE,
};
use indexmap::IndexMap;

/// How often the paced duties run: [`Half::tick`], and a binding with no stream variable
/// re-evaluated, at this pace whatever rings in between.
const TICK: Duration = Duration::from_millis(10);

/// What the engine wants a node's control half to hold — the WHOLE of it, sent when it changes.
#[derive(Clone, PartialEq)]
pub struct Desired {
    /// The record value per param: what an unbound param reads, and the type a binding coerces to.
    pub consts: Vec<Param>,
    pub subs: Vec<Sub>,
    /// Per output: the doors it rings, by name, once something is published on it.
    pub targets: Vec<Vec<(String, EventId)>>,
}

#[derive(Clone, PartialEq)]
pub enum Sub {
    /// An Array input: the producer service, and the inbox its frames enter.
    Slot { inbox: usize, service: String },
    /// A binding this half evaluates: everything the engine's own plan does not carry.
    Bind { param: usize, key: ParamKey, source: String, id: Option<BindingId>, vars: Vec<(String, Var)> },
}

/// What every control half of one engine shares. An engine's own additions live beside it, in a
/// struct of its own that only its [`Half`] sees.
pub struct Shared {
    pub evaluator: Mutex<Option<Arc<dyn ExprEvaluator>>>,
    pub reports: Mutex<Vec<(Uid, Status)>>,
    pub waker: Arc<DrainWaker>,
    /// A half saw something only a settle can act on — a shape moved, a file opened.
    pub replan: AtomicBool,
}

impl Shared {
    pub fn new(waker: Arc<DrainWaker>) -> Shared {
        Shared { evaluator: Mutex::new(None), reports: Mutex::new(Vec::new()), waker, replan: AtomicBool::new(false) }
    }

    fn report(&self, uid: Uid, status: Status) {
        self.reports.lock().unwrap().push((uid, status));
        self.waker.notify();
    }

    /// Ask the engine for a settle, and wake the drain that runs one.
    pub fn ask_settle(&self) {
        self.replan.store(true, Ordering::Release);
        self.waker.notify();
    }

    /// Hand the engine's own pending statuses and every half's reports to `apply`, and answer how
    /// many. ONE lock over the reports, so nothing lands between a read and a clear.
    pub fn drain(&self, pending: &mut Vec<(Uid, Status)>, apply: &mut dyn FnMut(Uid, Status)) -> usize {
        let mut all = std::mem::take(pending);
        all.append(&mut self.reports.lock().expect("the reports"));
        let n = all.len();
        for (uid, status) in all {
            apply(uid, status);
        }
        n
    }
}

/// What a scheduled engine's nodes are faulted with, and the deltas a settle owes the graph. The
/// engine states the WHOLE current set each settle; this answers only what moved.
#[derive(Default)]
pub struct Faults(IndexMap<Uid, String>);

impl Faults {
    pub fn settle(&mut self, now: impl IntoIterator<Item = (Uid, String)>, since: f64) -> Vec<(Uid, Status)> {
        let now: IndexMap<Uid, String> = now.into_iter().collect();
        let mut out: Vec<(Uid, Status)> =
            self.0.keys().filter(|u| !now.contains_key(*u)).map(|u| (*u, Status::Fault { fault: None })).collect();
        for (uid, msg) in &now {
            if self.0.get(uid) != Some(msg) {
                let fault = Some(goofi_node::NodeFault::Process { msg: msg.clone(), since });
                out.push((*uid, Status::Fault { fault }));
            }
        }
        self.0 = now;
        out
    }

    pub fn forget(&mut self, uid: Uid) {
        self.0.shift_remove(&uid);
    }
}

/// This tick's settled state, as [`Half::tick`] reads it.
pub struct Cx<'a> {
    /// The record value per declared param.
    pub consts: &'a [Param],
    /// The live value per declared param — a constant, or what a binding last evaluated to.
    pub params: &'a [AtomicU64],
    /// The pulse params raised since the last tick, by index.
    pub pulses: &'a [usize],
    /// Per output: whether anyone subscribes to its data service right now.
    pub readers: &'a [bool],
}

/// What one tick of a [`Half`] changed. The errors are the WHOLE current set for the keys the
/// half owns; the core files only what moved.
#[derive(Default)]
pub struct Ticked {
    pub errors: Vec<(ParamKey, Option<String>)>,
    /// Only a settle can finish what this tick found.
    pub replan: bool,
}

/// The engine's half of a node's control thread: what an arrival becomes, and what goes out.
pub trait Half {
    /// A frame arrived on Array input `inbox`; `true` asks for a settle.
    fn arrive(&mut self, inbox: usize, frame: &Data) -> bool;
    /// The wire into Array input `inbox` is gone.
    fn unwired(&mut self, _inbox: usize) {}
    /// The paced duties: publish what each output holds, and say what changed.
    fn tick(&mut self, cx: &Cx<'_>, publish: &mut dyn FnMut(usize, &[u8])) -> Ticked;
    /// The options behind this type's ONE refreshable `Str` param — the graph refuses a refresh
    /// on any other, so which param asked is not a question a half has to answer.
    fn refresh(&mut self) -> Option<Vec<String>> {
        None
    }
}

/// What the engine leaves for a control half: its whole desired state, the refreshes asked, and
/// the pulses fired.
#[derive(Default)]
struct Mail {
    desired: Option<Desired>,
    refresh: Vec<ParamKey>,
    pulse: Vec<ParamKey>,
}

/// The engine's end of one control half.
pub struct Handle {
    mail: Arc<Mutex<Mail>>,
    pub halt: Arc<Halt>,
    bell: Doorbell,
    /// What was last sent, so a settle that changes nothing says nothing.
    last: Mutex<Option<Desired>>,
}

impl Handle {
    fn send(&self, desired: Desired) {
        *self.last.lock().expect("the last desired") = Some(desired.clone());
        self.mail.lock().unwrap().desired = Some(desired);
        let _ = self.bell.ring(0);
    }

    /// Send only what is new, and say whether it did.
    pub fn send_if_changed(&self, desired: Desired) -> bool {
        let fresh = self.last.lock().expect("the last desired").as_ref() != Some(&desired);
        if fresh {
            self.send(desired);
        }
        fresh
    }

    pub fn refresh(&self, key: ParamKey) {
        self.mail.lock().unwrap().refresh.push(key);
        let _ = self.bell.ring(0);
    }

    pub fn pulse(&self, key: ParamKey) {
        self.mail.lock().unwrap().pulse.push(key);
        let _ = self.bell.ring(0);
    }

    pub fn stop(&self) {
        self.halt.stop();
        let _ = self.bell.ring(0);
    }
}

pub struct Spawn {
    /// The engine this node belongs to; the control thread wears it.
    pub engine: &'static str,
    pub uid: Uid,
    pub base: String,
    pub manifest: &'static NodeManifest,
    pub params: Arc<[AtomicU64]>,
    pub started: Instant,
}

/// Create the node's services on the caller's thread, where a failure can still be reported, and
/// park the control half on them. `make` builds the engine's half ON that thread, so a half may
/// hold what does not cross one — an audio stream, a MIDI connection.
pub fn spawn<H: Half + 'static>(
    spawn: Spawn,
    shared: Arc<Shared>,
    bells: &IoxNode,
    make: impl FnOnce() -> H + Send + 'static,
) -> Result<Handle, String> {
    let node = iox_node()?;
    let door = event_service(&node, &door_service(&spawn.base))?;
    let listener = door.listener_builder().create().map_err(|e| format!("listener: {e}"))?;
    let bell = Doorbell::open(bells, &door_service(&spawn.base))?;
    let mut outs = Vec::with_capacity(spawn.manifest.outputs.len());
    for out in spawn.manifest.outputs {
        let service = data_service(&node, &output_service(&spawn.base, out.name))?;
        let publisher = publisher(&service, out.name, INITIAL_SLICE)?;
        outs.push(Out { service, publisher, bells: Vec::new() });
    }
    let mail = Arc::new(Mutex::new(Mail::default()));
    let halt = Arc::new(Halt::default());
    let (thread_mail, thread_halt) = (mail.clone(), halt.clone());
    std::thread::Builder::new()
        .name(format!("goofi-{}-{}", spawn.engine, spawn.manifest.type_name))
        .spawn(move || {
            // The half is BUILT in here too: a factory that panics must still release the halt,
            // or the exit waits its whole ceiling on a node that never started.
            let inner = thread_halt.clone();
            let run = AssertUnwindSafe(move || {
                let control = Control {
                    uid: spawn.uid,
                    manifest: spawn.manifest,
                    started: spawn.started,
                    params: spawn.params,
                    consts: Vec::new(),
                    outs,
                    slots: Vec::new(),
                    binds: Vec::new(),
                    evaluated: IndexMap::new(),
                    errors: IndexMap::new(),
                    pulsed: Vec::new(),
                    pulses: Vec::new(),
                    shared,
                    mail: thread_mail,
                    last_tick: Instant::now(),
                    listener,
                    half: make(),
                    node,
                };
                control.run(&inner);
            });
            let _ = std::panic::catch_unwind(run);
            thread_halt.release();
        })
        .map_err(|e| format!("could not start the node's control thread: {e}"))?;
    Ok(Handle { mail, halt, bell, last: Mutex::new(None) })
}

/// One output's door out: who drinks from it, and who to wake once something is on it.
struct Out {
    service: ByteService,
    publisher: BytePublisher,
    bells: Vec<(String, Doorbell, EventId)>,
}

struct SlotSub {
    inbox: usize,
    service: String,
    subscriber: ByteSubscriber,
}

struct Bind {
    param: usize,
    key: ParamKey,
    expr: Expression,
    /// Per stream variable: its name, the service, and this half's subscriber on it.
    streams: Vec<(String, String, ByteSubscriber)>,
}

struct Control<H: Half> {
    uid: Uid,
    manifest: &'static NodeManifest,
    started: Instant,
    params: Arc<[AtomicU64]>,
    consts: Vec<Param>,
    outs: Vec<Out>,
    slots: Vec<SlotSub>,
    binds: Vec<Bind>,
    evaluated: IndexMap<ParamKey, Param>,
    errors: IndexMap<ParamKey, String>,
    /// The params a pulse raised, each lowered once a control tick has passed since its raise.
    pulsed: Vec<(usize, Instant)>,
    /// Every raise since the last tick, so a duty on the tick's cadence cannot miss the edge.
    pulses: Vec<usize>,
    shared: Arc<Shared>,
    mail: Arc<Mutex<Mail>>,
    last_tick: Instant,
    listener: Listener,
    half: H,
    /// Last: every port above is built from it, and fields drop in declaration order.
    node: IoxNode,
}

impl<H: Half> Control<H> {
    fn run(mut self, halt: &Halt) {
        while !halt.stopped() {
            let _ = self.listener.timed_wait_all(|_| {}, TICK);
            if halt.stopped() {
                break;
            }
            let params = &self.params;
            self.pulsed.retain(|(i, raised)| {
                let held = raised.elapsed() < TICK;
                if !held {
                    params[*i].store(0.0f64.to_bits(), Ordering::Relaxed);
                }
                held
            });
            let mail = std::mem::take(&mut *self.mail.lock().unwrap());
            if let Some(d) = mail.desired {
                self.apply(d);
            }
            for key in mail.refresh {
                let options = self.half.refresh();
                self.shared.report(self.uid, Status::RefreshOptions { key, options });
            }
            for key in &mail.pulse {
                if let Some(i) = self.index_of(key) {
                    self.raise(i);
                }
            }
            self.receive();
            if self.last_tick.elapsed() >= TICK {
                self.last_tick = Instant::now();
                self.tick();
            }
        }
    }

    fn apply(&mut self, d: Desired) {
        self.consts = d.consts;
        let (slots, binds): (Vec<Sub>, Vec<Sub>) = d.subs.into_iter().partition(|s| matches!(s, Sub::Slot { .. }));
        self.apply_slots(slots);
        self.apply_binds(binds);
        self.apply_bells(d.targets);
        for (i, c) in self.consts.iter().enumerate() {
            let bound = self.binds.iter().any(|b| b.param == i);
            let raised = self.pulsed.iter().any(|(p, _)| *p == i);
            if !bound && !raised {
                self.params[i].store(scalar(c).to_bits(), Ordering::Relaxed);
            }
        }
        let mut pass = Pass::default();
        for i in 0..self.binds.len() {
            self.evaluate(i, &mut pass);
        }
        self.report(pass);
    }

    fn apply_slots(&mut self, subs: Vec<Sub>) {
        let mut old = std::mem::take(&mut self.slots);
        for sub in subs {
            let Sub::Slot { inbox, service } = sub else { continue };
            let kept = take_where(&mut old, |s| s.service == service).map(|s| s.subscriber);
            let Some(subscriber) = kept.or_else(|| open_output_subscriber(&self.node, &service).ok()) else { continue };
            self.slots.push(SlotSub { inbox, service, subscriber });
        }
        for dropped in old {
            self.half.unwired(dropped.inbox);
        }
    }

    fn apply_binds(&mut self, subs: Vec<Sub>) {
        let mut old = std::mem::take(&mut self.binds);
        for sub in subs {
            let Sub::Bind { param, key, source, id, vars } = sub else { continue };
            let mut previous = take_where(&mut old, |b| b.key == key);
            let mut streams = Vec::new();
            let mut kept_names = Vec::new();
            let mut resolved = Vec::with_capacity(vars.len());
            for (var, v) in vars {
                let v = match v {
                    Var::Stream(service) => {
                        let kept = previous
                            .as_mut()
                            .and_then(|p| take_where(&mut p.streams, |(n, s, _)| *n == var && *s == service));
                        if kept.is_some() {
                            kept_names.push(var.clone());
                        }
                        match kept.map(|(_, _, s)| Ok(s)).unwrap_or_else(|| open_output_subscriber(&self.node, &service)) {
                            Ok(subscriber) => {
                                streams.push((var.clone(), service.clone(), subscriber));
                                Var::Stream(service)
                            }
                            Err(e) => Var::Missing(e),
                        }
                    }
                    other => other,
                };
                resolved.push((var, v));
            }
            let mut expr = Expression::new(source, id, resolved);
            if let Some(p) = &previous {
                expr.carry(&p.expr, |name| kept_names.iter().any(|n| n == name));
            }
            self.binds.push(Bind { param, key, expr, streams });
        }
        let mut pass = Pass::default();
        for dropped in old {
            pass.values |= self.evaluated.shift_remove(&dropped.key).is_some();
            if self.errors.shift_remove(&dropped.key).is_some() {
                pass.errors.push((dropped.key, None));
            }
        }
        self.report(pass);
    }

    fn apply_bells(&mut self, targets: Vec<Vec<(String, EventId)>>) {
        for (out, targets) in self.outs.iter_mut().zip(targets) {
            let mut old = std::mem::take(&mut out.bells);
            for (door, id) in targets {
                let kept = take_where(&mut old, |(d, _, _)| *d == door).map(|(_, bell, _)| bell);
                let Some(bell) = kept.or_else(|| Doorbell::open(&self.node, &door).ok()) else { continue };
                out.bells.push((door, bell, id));
            }
        }
    }

    fn index_of(&self, key: &ParamKey) -> Option<usize> {
        self.manifest.params.iter().position(|d| d.group == key.group && d.name == key.name)
    }

    /// Record or clear a param's error, keeping only what CHANGED: the graph files the delta
    /// against the instance.
    fn record_error(&mut self, key: ParamKey, error: Option<String>, pass: &mut Pass) {
        let changed = match &error {
            Some(e) => self.errors.insert(key.clone(), e.clone()).as_ref() != Some(e),
            None => self.errors.shift_remove(&key).is_some(),
        };
        if changed {
            pass.errors.push((key, error));
        }
    }

    /// What a pass of evaluations changed, said ONCE — a batch yields at most one decision, and
    /// evaluating a node's every binding is one batch.
    fn report(&self, pass: Pass) {
        if pass.values {
            let evaluated = self.evaluated.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            self.shared.report(self.uid, Status::ParamValues { evaluated });
        }
        if !pass.errors.is_empty() {
            self.shared.report(self.uid, Status::BindingErrors { errors: pass.errors });
        }
    }

    /// Every frame that arrived, in order into the half, latest-wins into a mailbox — and every
    /// binding a frame reached is evaluated once.
    fn receive(&mut self) {
        let mut moved = false;
        for s in &self.slots {
            while let Ok(Some(sample)) = s.subscriber.receive() {
                if let Ok(frame) = goofi_codec::decode(sample.payload()) {
                    moved |= self.half.arrive(s.inbox, &frame);
                }
            }
        }
        if moved {
            self.shared.ask_settle();
        }
        let mut touched = Vec::new();
        for (i, b) in self.binds.iter_mut().enumerate() {
            for (var, _, subscriber) in &b.streams {
                let mut newest = None;
                while let Ok(Some(sample)) = subscriber.receive() {
                    newest = goofi_codec::decode(sample.payload()).ok();
                }
                if let Some(frame) = newest {
                    b.expr.deliver(var, frame);
                    touched.push(i);
                }
            }
        }
        touched.dedup();
        let mut pass = Pass::default();
        for i in touched {
            self.evaluate(i, &mut pass);
        }
        self.report(pass);
    }

    /// The paced duties: a binding with no stream re-evaluates, and the half says what goes out.
    fn tick(&mut self) {
        let pulses = std::mem::take(&mut self.pulses);
        let mut pass = Pass::default();
        for i in 0..self.binds.len() {
            if self.binds[i].streams.is_empty() {
                self.evaluate(i, &mut pass);
            }
        }
        let readers: Vec<bool> = self.outs.iter().map(|o| goofi_transport::subscribers(&o.service) > 0).collect();
        let cx = Cx {
            consts: &self.consts,
            params: &self.params,
            pulses: &pulses,
            readers: &readers,
        };
        let outs = &self.outs;
        let ticked = self.half.tick(&cx, &mut |i, bytes| {
            let out = &outs[i];
            goofi_transport::publish(&out.publisher, bytes, out.bells.iter().map(|(_, bell, id)| (bell, *id)));
        });
        for (key, error) in ticked.errors {
            self.record_error(key, error, &mut pass);
        }
        if ticked.replan {
            self.shared.ask_settle();
        }
        self.report(pass);
    }

    /// Raise a pulse param for one control tick.
    fn raise(&mut self, i: usize) {
        self.params[i].store(1.0f64.to_bits(), Ordering::Relaxed);
        self.pulsed.push((i, Instant::now()));
        self.pulses.push(i);
    }

    /// One binding's value into its atomic — the literal when nothing has arrived or it cannot be
    /// evaluated — and the report of what changed.
    fn evaluate(&mut self, i: usize, pass: &mut Pass) {
        let b = &self.binds[i];
        let param = b.param;
        let target = &self.consts[param];
        let evaluator = self.shared.evaluator.lock().unwrap().clone();
        let t = self.started.elapsed().as_secs_f64();
        let (value, error) = match b.expr.evaluate(evaluator.as_deref(), t, target) {
            Ok(Some(v)) if !scalar(&v).is_finite() => (None, Some(format!("evaluated to {}", scalar(&v)))),
            Ok(v) => (v, None),
            Err(e) => (None, Some(e)),
        };
        let key = b.key.clone();
        // A source on a pulse is a gate: the RISE is the request, and an unevaluated pass keeps
        // the edge memory.
        if matches!(target, Param::Pulse) {
            if let Some(level) = value {
                let was_high = self.evaluated.insert(key.clone(), level.clone()).and_then(|p| p.as_bool()).unwrap_or(false);
                if !was_high && level.as_bool() == Some(true) {
                    self.raise(param);
                }
            }
            self.record_error(key, error, pass);
            return;
        }
        self.params[param].store(scalar(value.as_ref().unwrap_or(target)).to_bits(), Ordering::Relaxed);
        pass.values |= match value {
            Some(v) => self.evaluated.insert(key.clone(), v.clone()).as_ref() != Some(&v),
            None => self.evaluated.shift_remove(&key).is_some(),
        };
        self.record_error(key, error, pass);
    }
}

/// What one pass of binding evaluations changed. The values ride as the WHOLE sparse map, never a
/// delta — the graph replaces its copy with it, so a value it stops being told is one it would
/// otherwise preview for ever.
#[derive(Default)]
struct Pass {
    values: bool,
    errors: Vec<(ParamKey, Option<String>)>,
}

/// A param's scalar as an engine reads it: a number as itself, a bool as 0/1, an option as its
/// index, free text as 0, and a pulse — which holds no value — as 0.
pub fn scalar(p: &Param) -> f64 {
    p.as_f64().unwrap_or_else(|| match p {
        Param::Str { value, options: Some(options), .. } => {
            options.iter().position(|o| o == value).map_or(0.0, |i| i as f64)
        }
        _ => 0.0,
    })
}

/// The record's value for one declared param, the declared default where the record has none.
pub fn param_of(params: &ParamGroups, d: &ParamDecl) -> Param {
    goofi_node::param(params, d.group, d.name).cloned().unwrap_or_else(|| d.spec.to_param())
}

pub fn scalar_of(params: &ParamGroups, d: &ParamDecl) -> f64 {
    scalar(&param_of(params, d))
}

/// A `Str` param's text; every other kind — a number, a bool, a valueless pulse — has none, and
/// so has a param the first desired state has not delivered yet: a control half ticks from the
/// moment its thread starts, and its consts arrive after that, not before.
pub fn text(consts: &[Param], param: usize) -> String {
    match consts.get(param) {
        Some(Param::Str { value, .. }) => value.clone(),
        _ => String::new(),
    }
}

pub fn flag(consts: &[Param], param: usize) -> bool {
    consts.get(param).and_then(|p| p.as_bool()).unwrap_or(false)
}
