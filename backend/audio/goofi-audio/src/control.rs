//! An audio node's control half: one thread per node, parked on the node's own door. It is the
//! one writer of the node's param atomics — a constant and an evaluated binding land through the
//! same hand — the crossing every Array input enters through, and the tap every reader of an
//! output drinks from.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use goofi_audio_sdk::{BLOCK, MAX_CHANNELS};
use goofi_core::{Data, Meta, Param};
use goofi_node::{BindingId, DrainWaker, EventId, ExprEvaluator, Expression, ParamKey, Status, Uid, Var};
use goofi_transport::{
    data_service, door_service, event_service, iox_node, open_output_subscriber, output_service, publisher,
    take_where, ByteService, BytePublisher, ByteSubscriber, Doorbell, Halt, IoxNode, Listener, INITIAL_SLICE,
};
use indexmap::IndexMap;

use crate::{plan, RATE};

/// How often the paced duties run: a tapped output is published, and a binding with no stream
/// variable is re-evaluated, at this pace whatever rings in between.
pub const TICK: Duration = Duration::from_millis(10);
/// A tap holds this many blocks of the widest output; what does not fit is dropped, newest first.
pub const TAP_RING: usize = (1 + MAX_CHANNELS as usize * BLOCK) * 16;
/// An inbox holds one second of the widest frame at the rate; a frame that does not fit is
/// dropped whole.
pub const INBOX_RING: usize = RATE as usize * MAX_CHANNELS as usize;

/// What the engine wants a node's control half to hold — the WHOLE of it, sent when it changes.
#[derive(Clone, Debug, PartialEq)]
pub struct Desired {
    /// The record value per param: what an unbound param reads, and the type a binding coerces to.
    pub consts: Vec<Param>,
    pub subs: Vec<Sub>,
    /// Per output: the doors it rings, by name, once a tapped block is out.
    pub targets: Vec<Vec<(String, EventId)>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Sub {
    /// An Array input: the producer service, and the inbox its samples enter.
    Slot { inbox: usize, service: String },
    /// A binding this half evaluates: everything but a same-engine audio reference.
    Bind { param: usize, key: ParamKey, source: String, id: Option<BindingId>, vars: Vec<(String, Var)> },
}

/// What every control half of one engine shares.
pub struct Shared {
    pub evaluator: Mutex<Option<Arc<dyn ExprEvaluator>>>,
    pub reports: Mutex<Vec<(Uid, Status)>>,
    pub waker: Arc<DrainWaker>,
    /// An Array input saw a new channel count: only a settle can re-plan for it.
    pub replan: AtomicBool,
}

impl Shared {
    fn report(&self, uid: Uid, status: Status) {
        self.reports.lock().unwrap().push((uid, status));
        self.waker.notify();
    }
}

/// The engine's end of one control half.
pub struct Handle {
    desired: Arc<Mutex<Option<Desired>>>,
    pub halt: Arc<Halt>,
    /// The channel count each Array input last saw — what the plan sizes its inbox by.
    pub chans: Vec<Arc<AtomicU16>>,
    bell: Doorbell,
}

impl Handle {
    pub fn send(&self, desired: Desired) {
        *self.desired.lock().unwrap() = Some(desired);
        let _ = self.bell.ring(0);
    }

    pub fn stop(&self) {
        self.halt.stop();
        let _ = self.bell.ring(0);
    }
}

pub struct Spawn {
    pub uid: Uid,
    pub base: String,
    pub manifest: &'static goofi_node::NodeManifest,
    pub params: Arc<[AtomicU64]>,
    pub inboxes: Vec<rtrb::Producer<f32>>,
    pub taps: Vec<rtrb::Consumer<f32>>,
    pub started: Instant,
}

/// Create the node's services on the caller's thread, where a failure can still be reported, and
/// park the control half on them.
pub fn spawn(spawn: Spawn, shared: Arc<Shared>, bells: &IoxNode) -> Result<Handle, String> {
    let node = iox_node()?;
    let door = event_service(&node, &door_service(&spawn.base))?;
    let listener = door.listener_builder().create().map_err(|e| format!("listener: {e}"))?;
    let bell = Doorbell::open(bells, &door_service(&spawn.base))?;
    let mut outs = Vec::with_capacity(spawn.manifest.outputs.len());
    for (out, ring) in spawn.manifest.outputs.iter().zip(spawn.taps) {
        let service = data_service(&node, &output_service(&spawn.base, out.name))?;
        let publisher = publisher(&service, out.name, INITIAL_SLICE)?;
        outs.push(Out { ring, service, publisher, bells: Vec::new() });
    }
    let inboxes: Vec<Inbox> = spawn.inboxes.into_iter().map(Inbox::new).collect();
    let chans = inboxes.iter().map(|i| i.chans.clone()).collect();
    let desired = Arc::new(Mutex::new(None));
    let halt = Arc::new(Halt::default());
    let control = Control {
        uid: spawn.uid,
        started: spawn.started,
        params: spawn.params,
        consts: Vec::new(),
        inboxes,
        outs,
        slots: Vec::new(),
        binds: Vec::new(),
        evaluated: IndexMap::new(),
        errors: IndexMap::new(),
        shared,
        desired: desired.clone(),
        last_tick: Instant::now(),
        listener,
        node,
    };
    let thread_halt = halt.clone();
    std::thread::Builder::new()
        .name(format!("goofi-audio-{}", spawn.manifest.type_name))
        .spawn(move || {
            // A panic here is a bug, and it must still release the node's ports so the exit is real.
            let _ = std::panic::catch_unwind(AssertUnwindSafe(|| control.run(&thread_halt)));
            thread_halt.release();
        })
        .map_err(|e| format!("could not start the node's control thread: {e}"))?;
    Ok(Handle { desired, halt, chans, bell })
}

struct Inbox {
    ring: rtrb::Producer<f32>,
    chans: Arc<AtomicU16>,
    /// The fractional input position the next output sample reads, carried across frames.
    pos: f64,
}

impl Inbox {
    fn new(ring: rtrb::Producer<f32>) -> Inbox {
        Inbox { ring, chans: Arc::new(AtomicU16::new(1)), pos: 0.0 }
    }

    /// Resample one `[T]` or `[C, T]` frame linearly from its `sfreq` to the rate and enter it
    /// whole, as one chunk headed by its channel count and length. A frame with no `sfreq` enters
    /// one sample per sample, so a control value is held until the next. Answers whether the
    /// channel count moved.
    fn enter(&mut self, frame: &Data) -> Option<bool> {
        let goofi_core::Value::Array(a) = frame.value() else { return None };
        let (c, t) = match *a.shape() {
            [t] => (1, t),
            [c, t] => (c, t),
            _ => return None,
        };
        if c == 0 || t == 0 || c > MAX_CHANNELS as usize {
            return None;
        }
        let x: Vec<f32> = a.as_bytes().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("four bytes"))).collect();
        let step = frame.meta().sfreq().filter(|sf| *sf > 0.0).map_or(1.0, |sf| sf / RATE);
        let moved = self.chans.swap(c as u16, Ordering::Relaxed) != c as u16;
        if moved {
            self.pos = 0.0;
        }
        let pos = self.pos;
        let n = ((t as f64 - pos) / step).ceil().max(0.0) as usize;
        let Some(need) = n.checked_mul(c).and_then(|s| s.checked_add(2)) else { return Some(moved) };
        if let Ok(chunk) = self.ring.write_chunk_uninit(need) {
            let at = |ch: usize, i: usize| x[ch * t + i.min(t - 1)];
            let samples = (0..n).flat_map(|k| {
                let p = pos + k as f64 * step;
                let i = p.floor();
                let f = (p - i) as f32;
                let i = i as usize;
                (0..c).map(move |ch| at(ch, i) + (at(ch, i + 1) - at(ch, i)) * f)
            });
            chunk.fill_from_iter([c as f32, n as f32].into_iter().chain(samples));
        }
        self.pos = pos + n as f64 * step - t as f64;
        Some(moved)
    }
}

struct Out {
    ring: rtrb::Consumer<f32>,
    service: ByteService,
    publisher: BytePublisher,
    bells: Vec<(String, Doorbell, EventId)>,
}

impl Out {
    /// Everything the audio thread tapped since the last tick, as one planar `[C, T]` frame —
    /// up to a block whose channel count differs, which the next tick starts from.
    fn drain(&mut self) -> Option<(usize, Vec<f32>)> {
        let mut chans = 0;
        let mut planar: Vec<Vec<f32>> = Vec::new();
        while let Ok(head) = self.ring.read_chunk(1) {
            let c = head.as_slices().0.first().copied().unwrap_or(0.0) as usize;
            if c == 0 || (chans != 0 && c != chans) {
                if c == 0 {
                    head.commit_all();
                }
                break;
            }
            head.commit_all();
            let Ok(block) = self.ring.read_chunk(c * BLOCK) else { break };
            if chans == 0 {
                chans = c;
                planar = vec![Vec::new(); c];
            }
            let (a, b) = block.as_slices();
            let samples: Vec<f32> = a.iter().chain(b).copied().collect();
            for (ch, lane) in planar.iter_mut().enumerate() {
                lane.extend_from_slice(&samples[ch * BLOCK..(ch + 1) * BLOCK]);
            }
            block.commit_all();
        }
        (chans != 0).then(|| (chans, planar.concat()))
    }

    fn wanted(&self) -> bool {
        goofi_transport::subscribers(&self.service) > 0
    }

    fn publish(&self, frame: &Data) {
        let bytes = goofi_codec::encode(frame);
        goofi_transport::publish(&self.publisher, &bytes, self.bells.iter().map(|(_, bell, id)| (bell, *id)));
    }
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

struct Control {
    uid: Uid,
    started: Instant,
    params: Arc<[AtomicU64]>,
    consts: Vec<Param>,
    inboxes: Vec<Inbox>,
    outs: Vec<Out>,
    slots: Vec<SlotSub>,
    binds: Vec<Bind>,
    evaluated: IndexMap<ParamKey, Param>,
    errors: IndexMap<ParamKey, String>,
    shared: Arc<Shared>,
    desired: Arc<Mutex<Option<Desired>>>,
    last_tick: Instant,
    listener: Listener,
    /// Last: every port above is built from it, and fields drop in declaration order.
    node: IoxNode,
}

impl Control {
    fn run(mut self, halt: &Halt) {
        while !halt.stopped() {
            let _ = self.listener.timed_wait_all(|_| {}, TICK);
            if halt.stopped() {
                break;
            }
            let desired = self.desired.lock().unwrap().take();
            if let Some(d) = desired {
                self.apply(d);
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
            if !self.binds.iter().any(|b| b.param == i) {
                self.params[i].store(plan::scalar(c).to_bits(), Ordering::Relaxed);
            }
        }
        for i in 0..self.binds.len() {
            self.evaluate(i);
        }
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
            self.inboxes[dropped.inbox].pos = 0.0;
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
        let mut values_changed = false;
        let mut cleared = Vec::new();
        for dropped in old {
            values_changed |= self.evaluated.shift_remove(&dropped.key).is_some();
            if self.errors.shift_remove(&dropped.key).is_some() {
                cleared.push((dropped.key, None));
            }
        }
        if values_changed {
            self.report_values();
        }
        if !cleared.is_empty() {
            self.shared.report(self.uid, Status::BindingErrors { errors: cleared });
        }
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

    /// Every frame that arrived, in order into an inbox, latest-wins into a mailbox — and every
    /// binding a frame reached is evaluated once.
    fn receive(&mut self) {
        let mut moved = false;
        for s in &self.slots {
            while let Ok(Some(sample)) = s.subscriber.receive() {
                if let Ok(frame) = goofi_codec::decode(sample.payload()) {
                    moved |= self.inboxes[s.inbox].enter(&frame).unwrap_or(false);
                }
            }
        }
        if moved {
            self.shared.replan.store(true, Ordering::Release);
            self.shared.waker.notify();
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
        for i in touched {
            self.evaluate(i);
        }
    }

    /// The paced duties: a binding with no stream re-evaluates, and every tapped output goes out.
    fn tick(&mut self) {
        for i in 0..self.binds.len() {
            if self.binds[i].streams.is_empty() {
                self.evaluate(i);
            }
        }
        for out in &mut self.outs {
            let Some((c, planar)) = out.drain() else { continue };
            if !out.wanted() {
                continue;
            }
            let t = planar.len() / c;
            let bytes: Vec<u8> = planar.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Ok(frame) = Data::array_f32(vec![c, t], bytes, Meta::new().with_sfreq(Some(RATE))) {
                out.publish(&frame);
            }
        }
    }

    /// One binding's value into its atomic — the literal when nothing has arrived or it cannot be
    /// evaluated — and the report of what changed.
    fn evaluate(&mut self, i: usize) {
        let b = &self.binds[i];
        let target = &self.consts[b.param];
        let evaluator = self.shared.evaluator.lock().unwrap().clone();
        let t = self.started.elapsed().as_secs_f64();
        let (value, error) = match b.expr.evaluate(evaluator.as_deref(), t, target) {
            Ok(v) => (v, None),
            Err(e) => (None, Some(e)),
        };
        self.params[b.param].store(plan::scalar(value.as_ref().unwrap_or(target)).to_bits(), Ordering::Relaxed);
        let key = b.key.clone();
        let values_changed = match value {
            Some(v) => self.evaluated.insert(key.clone(), v.clone()).as_ref() != Some(&v),
            None => self.evaluated.shift_remove(&key).is_some(),
        };
        if values_changed {
            self.report_values();
        }
        let error_changed = match &error {
            Some(e) => self.errors.insert(key.clone(), e.clone()).as_ref() != Some(e),
            None => self.errors.shift_remove(&key).is_some(),
        };
        if error_changed {
            self.shared.report(self.uid, Status::BindingErrors { errors: vec![(key, error)] });
        }
    }

    /// The whole sparse map, never a delta — the graph replaces its copy with this.
    fn report_values(&self) {
        let evaluated = self.evaluated.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        self.shared.report(self.uid, Status::ParamValues { evaluated });
    }
}
