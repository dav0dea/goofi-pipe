//! An audio node's control half: one thread per node, parked on the node's own door. It is the
//! one writer of the node's param atomics — a constant and an evaluated binding land through the
//! same hand — the crossing every Array input enters through, and the tap every reader of an
//! output drinks from.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use goofi_audio_sdk::{BLOCK, MAX_CHANNELS};
use goofi_core::{Data, Meta, Param};
use goofi_node::{BindingId, DrainWaker, EventId, ExprEvaluator, Expression, NodeManifest, ParamKey, Status, Uid, Var};
use goofi_transport::{
    data_service, door_service, event_service, iox_node, open_output_subscriber, output_service, publisher,
    take_where, ByteService, BytePublisher, ByteSubscriber, Doorbell, Halt, IoxNode, Listener, INITIAL_SLICE,
};
use indexmap::IndexMap;

use crate::nodes::midi_in::{Note, NO_PORT};
use crate::nodes::{audio_in, audio_out, midi_in};
use crate::{plan, Clock, DEFAULT_DEVICE, NO_DEVICE, RATE};

/// How often the paced duties run: a tapped output is published, and a binding with no stream
/// variable is re-evaluated, at this pace whatever rings in between.
pub const TICK: Duration = Duration::from_millis(10);
/// A tap holds this many blocks of the widest output; what does not fit is dropped, newest first.
pub const TAP_RING: usize = (1 + MAX_CHANNELS as usize * BLOCK) * 16;
/// An inbox holds one second of the widest frame at the rate; a frame that does not fit is
/// dropped whole.
pub const INBOX_RING: usize = RATE as usize * MAX_CHANNELS as usize;
/// Notes a port may hold between two blocks.
pub const NOTE_RING: usize = 1024;

/// A ring's producer as an OS callback holds it: successive streams on one node share it, and a
/// callback that finds it taken drops that buffer rather than wait.
pub type Feed<T> = Arc<Mutex<rtrb::Producer<T>>>;

/// The control half's ends of a device's or a port's rings — none for a node that owns no OS
/// handle.
#[derive(Default)]
pub struct Ports {
    pub audio_in: Option<(Feed<f32>, Arc<AtomicU16>)>,
    pub midi_in: Option<Feed<Note>>,
}

/// What a control half opens on its own thread and never lets cross it: a stream is not `Send`
/// on every host. A device is opened at the clock's rate, so the name AND the rate gate a reopen.
#[derive(Default)]
struct Io {
    stream: Option<cpal::Stream>,
    midi: Option<midir::MidiInputConnection<()>>,
    device: Option<(String, f64)>,
    port: Option<String>,
    /// Raised by the input stream's error callback; the name is then tried once more.
    dead: Arc<AtomicBool>,
}

/// The device `name` names among `all`, `default` being the host's; `kind` words a refusal.
pub(crate) fn device(
    kind: &str,
    name: &str,
    default: Option<cpal::Device>,
    all: Result<impl Iterator<Item = cpal::Device>, impl std::fmt::Display>,
) -> Result<cpal::Device, String> {
    if name == DEFAULT_DEVICE {
        return default.ok_or_else(|| format!("no default {kind} device"));
    }
    all.map_err(|e| format!("{kind} devices: {e}"))?
        .find(|d| name_of(d).as_deref() == Some(name))
        .ok_or_else(|| format!("no {kind} device `{name}`"))
}

fn name_of(d: &cpal::Device) -> Option<String> {
    d.description().ok().map(|d| d.name().to_string())
}

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
    /// The clock's rate, `f64` bits: what a crossing resamples to and a tap is stamped with.
    pub rate: AtomicU64,
    /// What drives the blocks: a live stream is opened only where the device does.
    pub clock: Clock,
    /// What a plugin's own editor wrote — node, plugin param id, normalized value — for the
    /// worker to put through the param op.
    pub edits: Mutex<Vec<(Uid, u32, f64)>>,
}

/// What the engine leaves for a control half: its whole desired state, the refreshes asked, and
/// the pulses fired.
#[derive(Default)]
pub struct Mail {
    pub desired: Option<Desired>,
    pub refresh: Vec<ParamKey>,
    pub pulse: Vec<ParamKey>,
}

impl Shared {
    fn report(&self, uid: Uid, status: Status) {
        self.reports.lock().unwrap().push((uid, status));
        self.waker.notify();
    }
}

/// The engine's end of one control half.
pub struct Handle {
    mail: Arc<Mutex<Mail>>,
    pub halt: Arc<Halt>,
    /// The channel count each Array input last saw — what the plan sizes its inbox by.
    pub chans: Vec<Arc<AtomicU16>>,
    bell: Doorbell,
}

impl Handle {
    pub fn send(&self, desired: Desired) {
        self.mail.lock().unwrap().desired = Some(desired);
        let _ = self.bell.ring(0);
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
    pub uid: Uid,
    pub base: String,
    pub manifest: &'static NodeManifest,
    pub params: Arc<[AtomicU64]>,
    pub inboxes: Vec<rtrb::Producer<f32>>,
    pub taps: Vec<rtrb::Consumer<f32>>,
    pub ports: Ports,
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
    let mail = Arc::new(Mutex::new(Mail::default()));
    let halt = Arc::new(Halt::default());
    let control = Control {
        uid: spawn.uid,
        manifest: spawn.manifest,
        started: spawn.started,
        params: spawn.params,
        consts: Vec::new(),
        inboxes,
        outs,
        slots: Vec::new(),
        binds: Vec::new(),
        ports: spawn.ports,
        evaluated: IndexMap::new(),
        errors: IndexMap::new(),
        pulsed: Vec::new(),
        shared,
        mail: mail.clone(),
        last_tick: Instant::now(),
        listener,
        node,
    };
    let thread_halt = halt.clone();
    std::thread::Builder::new()
        .name(format!("goofi-audio-{}", spawn.manifest.type_name))
        .spawn(move || {
            let mut io = Io::default();
            // A panic here is a bug, and it must still release the node's ports so the exit is real.
            let _ = std::panic::catch_unwind(AssertUnwindSafe(|| control.run(&thread_halt, &mut io)));
            thread_halt.release();
        })
        .map_err(|e| format!("could not start the node's control thread: {e}"))?;
    Ok(Handle { mail, halt, chans, bell })
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
    fn enter(&mut self, frame: &Data, rate: f64) -> Option<bool> {
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
        let step = frame.meta().sfreq().filter(|sf| *sf > 0.0).map_or(1.0, |sf| sf / rate);
        let moved = self.chans.swap(c as u16, Ordering::Relaxed) != c as u16;
        if moved {
            self.pos = 0.0;
        }
        let pos = self.pos;
        let n = ((t as f64 - pos) / step).ceil().max(0.0) as usize;
        let Some(need) = n.checked_mul(c).and_then(|s| s.checked_add(2)) else { return Some(moved) };
        if let Ok(chunk) = self.ring.write_chunk_uninit(need) {
            let at = |ch: usize, i: usize| {
                let v = x[ch * t + i.min(t - 1)];
                if v.is_finite() { v } else { 0.0 }
            };
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
    manifest: &'static NodeManifest,
    started: Instant,
    params: Arc<[AtomicU64]>,
    consts: Vec<Param>,
    inboxes: Vec<Inbox>,
    outs: Vec<Out>,
    slots: Vec<SlotSub>,
    binds: Vec<Bind>,
    ports: Ports,
    evaluated: IndexMap<ParamKey, Param>,
    errors: IndexMap<ParamKey, String>,
    /// The params a pulse raised, each lowered once a control tick has passed since its raise.
    pulsed: Vec<(usize, Instant)>,
    shared: Arc<Shared>,
    mail: Arc<Mutex<Mail>>,
    last_tick: Instant,
    listener: Listener,
    /// Last: every port above is built from it, and fields drop in declaration order.
    node: IoxNode,
}

impl Control {
    fn run(mut self, halt: &Halt, io: &mut Io) {
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
            self.open_io(io);
            for key in mail.refresh {
                let options = self.enumerate();
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
                self.params[i].store(plan::scalar(c).to_bits(), Ordering::Relaxed);
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

    /// The one refreshable list a type has — the graph refuses a refresh on any other param —
    /// enumerated here rather than under the graph lock: the devices behind the host default, or
    /// the MIDI ports behind `none`.
    fn enumerate(&self) -> Option<Vec<String>> {
        let named = |devices: Option<Vec<cpal::Device>>| {
            let mut names = vec![DEFAULT_DEVICE.to_string()];
            names.extend(devices.into_iter().flatten().filter_map(|d| name_of(&d)).filter(|n| n != DEFAULT_DEVICE));
            names
        };
        let host = cpal::default_host();
        match self.manifest.type_name {
            audio_out::TYPE => Some(named(host.output_devices().ok().map(|d| d.collect()))),
            audio_in::TYPE => Some(named(host.input_devices().ok().map(|d| d.collect()))),
            midi_in::TYPE => {
                let mut names = vec![NO_PORT.to_string()];
                if let Ok(input) = midir::MidiInput::new("goofi") {
                    names.extend(input.ports().iter().filter_map(|p| input.port_name(p).ok()));
                }
                Some(names)
            }
            _ => None,
        }
    }

    /// A device or a port a param names is opened here, on this thread, when the name moves — or
    /// the clock's rate, or the stream died; a name that failed stands as an error on that param
    /// until it moves.
    fn open_io(&mut self, io: &mut Io) {
        let mut pass = Pass::default();
        if io.dead.swap(false, Ordering::Acquire) {
            io.stream = None;
            io.device = None;
        }
        if let Some((producer, chans)) = self.ports.audio_in.clone() {
            let wanted = (self.text(audio_in::P::DEVICE), self.shared.rate());
            if io.device.as_ref() != Some(&wanted) {
                io.stream = None;
                let (stream, error) = match open_input(&wanted.0, wanted.1, producer, io.dead.clone(), self.shared.clock) {
                    Ok(Some((stream, c))) => {
                        chans.store(c, Ordering::Relaxed);
                        (Some(stream), None)
                    }
                    Ok(None) => (None, Some(NO_DEVICE.to_string())),
                    Err(e) => (None, Some(e)),
                };
                self.shared.replan.store(true, Ordering::Release);
                self.shared.waker.notify();
                io.stream = stream;
                io.device = Some(wanted);
                self.record_error(self.key_of(audio_in::P::DEVICE), error, &mut pass);
            }
        }
        if let Some(producer) = self.ports.midi_in.clone() {
            let wanted = self.text(midi_in::P::PORT);
            if io.port.as_deref() != Some(wanted.as_str()) {
                io.midi = None;
                let error = if wanted == NO_PORT {
                    None
                } else {
                    match open_port(&wanted, producer) {
                        Ok(connection) => {
                            io.midi = Some(connection);
                            None
                        }
                        Err(e) => Some(e),
                    }
                };
                io.port = Some(wanted);
                self.record_error(self.key_of(midi_in::P::PORT), error, &mut pass);
            }
        }
        self.report(pass);
    }

    /// A `Str` param's text; every other kind — a number, a bool, a valueless pulse — has none.
    fn text(&self, param: usize) -> String {
        match &self.consts[param] {
            Param::Str { value, .. } => value.clone(),
            _ => String::new(),
        }
    }

    fn key_of(&self, param: usize) -> ParamKey {
        let d = &self.manifest.params[param];
        ParamKey::new(d.group, d.name)
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

    /// Every frame that arrived, in order into an inbox, latest-wins into a mailbox — and every
    /// binding a frame reached is evaluated once.
    fn receive(&mut self) {
        let mut moved = false;
        let rate = self.shared.rate();
        for s in &self.slots {
            while let Ok(Some(sample)) = s.subscriber.receive() {
                if let Ok(frame) = goofi_codec::decode(sample.payload()) {
                    moved |= self.inboxes[s.inbox].enter(&frame, rate).unwrap_or(false);
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
        let mut pass = Pass::default();
        for i in touched {
            self.evaluate(i, &mut pass);
        }
        self.report(pass);
    }

    /// The paced duties: a binding with no stream re-evaluates, and every tapped output goes out.
    fn tick(&mut self) {
        let mut pass = Pass::default();
        for i in 0..self.binds.len() {
            if self.binds[i].streams.is_empty() {
                self.evaluate(i, &mut pass);
            }
        }
        self.report(pass);
        for out in &mut self.outs {
            let Some((c, planar)) = out.drain() else { continue };
            if goofi_transport::subscribers(&out.service) == 0 {
                continue;
            }
            let t = planar.len() / c;
            let bytes: Vec<u8> = planar.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Ok(frame) = Data::array_f32(vec![c, t], bytes, Meta::new().with_sfreq(Some(self.shared.rate()))) {
                let bytes = goofi_codec::encode(&frame);
                goofi_transport::publish(&out.publisher, &bytes, out.bells.iter().map(|(_, bell, id)| (bell, *id)));
            }
        }
    }

    /// Raise a pulse param for one control tick.
    fn raise(&mut self, i: usize) {
        self.params[i].store(1.0f64.to_bits(), Ordering::Relaxed);
        self.pulsed.push((i, Instant::now()));
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
            Ok(Some(v)) if !plan::scalar(&v).is_finite() => (None, Some(format!("evaluated to {}", plan::scalar(&v)))),
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
        self.params[param].store(plan::scalar(value.as_ref().unwrap_or(target)).to_bits(), Ordering::Relaxed);
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

/// The device's input stream, opened AT the clock's rate — a device that cannot is the error —
/// its callback entering interleaved frames into the node's inbox as the Array crossing does.
/// The name is resolved whatever the clock, so an absent one is still named; only the device
/// clock opens what it resolved to.
fn open_input(
    name: &str,
    rate: f64,
    producer: Feed<f32>,
    dead: Arc<AtomicBool>,
    clock: Clock,
) -> Result<Option<(cpal::Stream, u16)>, String> {
    let host = cpal::default_host();
    let device = device("input", name, host.default_input_device(), host.input_devices())?;
    if !clock.owns_devices() {
        return Ok(None);
    }
    let mut config = device.default_input_config().map_err(|e| format!("`{name}`: {e}"))?.config();
    config.sample_rate = rate as u32;
    let channels = config.channels;
    let stream = device
        .build_input_stream::<f32, _, _>(
            config,
            move |data, _| {
                let Ok(mut inbox) = producer.try_lock() else { return };
                let frames = data.len() / channels as usize;
                if let Ok(chunk) = inbox.write_chunk_uninit(2 + data.len()) {
                    chunk.fill_from_iter([f32::from(channels), frames as f32].into_iter().chain(data.iter().copied()));
                }
            },
            move |e| {
                if matches!(e.kind(), cpal::ErrorKind::DeviceNotAvailable) {
                    dead.store(true, Ordering::Release);
                }
            },
            None,
        )
        .map_err(|e| format!("`{name}`: {e}"))?;
    stream.play().map_err(|e| format!("`{name}`: {e}"))?;
    Ok(Some((stream, channels)))
}

/// A MIDI port, its callback handing every note to the node's ring.
fn open_port(name: &str, producer: Feed<Note>) -> Result<midir::MidiInputConnection<()>, String> {
    let mut input = midir::MidiInput::new("goofi").map_err(|e| format!("midi: {e}"))?;
    input.ignore(midir::Ignore::All);
    let port = input
        .ports()
        .into_iter()
        .find(|p| input.port_name(p).is_ok_and(|n| n == name))
        .ok_or_else(|| format!("no MIDI port `{name}`"))?;
    input
        .connect(
            &port,
            "goofi-in",
            move |_, bytes, _| {
                if let (Some(note), Ok(mut notes)) = (Note::parse(bytes), producer.try_lock()) {
                    let _ = notes.push(note);
                }
            },
            (),
        )
        .map_err(|e| format!("`{name}`: {e}"))
}
