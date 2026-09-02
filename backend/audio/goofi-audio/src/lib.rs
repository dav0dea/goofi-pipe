//! The audio engine behind the `Engine` seam: synchronous, in-process, one 64-frame block per
//! callback. The engine (this file) owns the library, the slab indices, the plan and the clock;
//! the audio half (`runtime`) owns the instances and the arena, and hears from here by message; a
//! node's control half (`control`) is a thread of its own, parked on the node's door.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use goofi_audio_sdk::{AudioNode, BLOCK};
use goofi_core::{Param, SlotType};
use goofi_node::{
    DrainWaker, Engine, GraphView, LibraryEntry, NodeFault, NodeManifest, NodeStage, NodeView, ParamGroups,
    Request, Ringer, Status, Touched, Uid, Via, NATIVE,
};

mod control;
pub mod nodes;
mod plan;
mod runtime;

use control::{Desired, Handle, Shared, Sub};
use nodes::audio_out;
use plan::Plan;
use runtime::{Inbox, Msg, Retired, Runtime, Slot, MAX_PORTS};

/// The rate until a device names one.
pub const RATE: f64 = 48_000.0;

/// A CEILING, not a join: a wedged control half must not be able to wedge the exit.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(2);
/// A ceiling on a device open, which runs under the graph lock: a sound server that does not
/// answer must not wedge every op.
const OPEN_WAIT: Duration = Duration::from_secs(2);

/// What drives the blocks: the harness's `drive(frames)`, or the device the `AudioOut` nodes name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clock {
    External,
    Device,
}

/// The timing door: what the clock is doing, for `session status`.
pub struct AudioStatus {
    pub clock: &'static str,
    pub device: Option<String>,
    pub rate: f64,
    pub channels: u16,
    pub callbacks: u64,
    pub xruns: u64,
    pub render_max_us: u64,
}

#[derive(Default)]
pub(crate) struct Stats {
    callbacks: AtomicU64,
    xruns: AtomicU64,
    render_max_us: AtomicU64,
    /// Raised by the stream's error callback; the drain closes the clock and tries the name once more.
    dead: AtomicBool,
}

/// The running output stream, owned by a thread of its own: `cpal::Stream` is not `Send` on
/// every host. It plays once the runtime is cut to it; dropping this stops it and waits for that.
struct DeviceClock {
    name: String,
    channels: u16,
    go: Option<mpsc::Sender<()>>,
    done: mpsc::Receiver<()>,
}

impl DeviceClock {
    fn open(name: &str, runtime: Arc<Mutex<Runtime>>, stats: Arc<Stats>, waker: Arc<DrainWaker>) -> Result<(DeviceClock, f64), String> {
        let (opened, on_open) = mpsc::channel::<Result<(f64, u16), String>>();
        let (go, on_go) = mpsc::channel::<()>();
        let (done, on_done) = mpsc::channel::<()>();
        let device = name.to_string();
        std::thread::Builder::new()
            .name("goofi-audio-clock".into())
            .spawn(move || {
                let stream = match open_output(&device, runtime, stats.clone(), waker.clone()) {
                    Ok((stream, rate, channels)) => {
                        let _ = opened.send(Ok((rate, channels)));
                        stream
                    }
                    Err(e) => {
                        let _ = opened.send(Err(e));
                        return;
                    }
                };
                if on_go.recv().is_ok() {
                    if let Err(e) = stream.play() {
                        eprintln!("audio: {e}");
                        stats.dead.store(true, Ordering::Release);
                        waker.notify();
                    }
                    let _ = on_go.recv();
                }
                drop(stream);
                let _ = done.send(());
            })
            .map_err(|e| format!("could not start the clock thread: {e}"))?;
        let (rate, channels) = on_open
            .recv_timeout(OPEN_WAIT)
            .map_err(|_| format!("`{name}` did not open within {} s", OPEN_WAIT.as_secs()))??;
        Ok((DeviceClock { name: name.to_string(), channels, go: Some(go), done: on_done }, rate))
    }

    /// Start the callbacks — only once the runtime is cut to this stream's rate and width.
    fn play(&self) {
        if let Some(go) = &self.go {
            let _ = go.send(());
        }
    }
}

impl Drop for DeviceClock {
    fn drop(&mut self) {
        self.go = None;
        let _ = self.done.recv_timeout(SHUTDOWN_WAIT);
    }
}

/// The host default is what a `default` name means.
pub const DEFAULT_DEVICE: &str = "default";

fn open_output(name: &str, runtime: Arc<Mutex<Runtime>>, stats: Arc<Stats>, waker: Arc<DrainWaker>) -> Result<(cpal::Stream, f64, u16), String> {
    let host = cpal::default_host();
    let device = control::device("output", name, host.default_output_device(), host.output_devices())?;
    let supported = device.default_output_config().map_err(|e| format!("`{name}`: {e}"))?;
    let mut config = supported.config();
    if let cpal::SupportedBufferSize::Range { min, max } = supported.buffer_size() {
        if (*min..=*max).contains(&(BLOCK as u32)) {
            config.buffer_size = cpal::BufferSize::Fixed(BLOCK as u32);
        }
    }
    let rate = f64::from(config.sample_rate);
    let channels = config.channels;
    let died = stats.clone();
    let stream = device
        .build_output_stream::<f32, _, _>(
            config,
            move |data, _| {
                stats.callbacks.fetch_add(1, Ordering::Relaxed);
                let started = Instant::now();
                match runtime.try_lock() {
                    Ok(mut rt) => rt.render_into(data),
                    Err(_) => {
                        data.fill(0.0);
                        stats.xruns.fetch_add(1, Ordering::Relaxed);
                    }
                }
                stats.render_max_us.fetch_max(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            },
            // An underrun the backend reports recovers on its own and is an xrun; only a device
            // that is gone is death.
            move |e| {
                if matches!(e.kind(), cpal::ErrorKind::DeviceNotAvailable) {
                    died.dead.store(true, Ordering::Release);
                    waker.notify();
                } else {
                    died.xruns.fetch_add(1, Ordering::Relaxed);
                }
            },
            None,
        )
        .map_err(|e| format!("`{name}`: {e}"))?;
    Ok((stream, rate, channels))
}

/// The rings a device or a port fills, minted per instance: the DSP half's ends in the birth,
/// the control half's in the ports. A node that owns no OS handle gets neither.
fn rings_for(type_name: &str, chans: Arc<AtomicU16>) -> (nodes::Birth, control::Ports) {
    let mut birth = nodes::Birth { chans: chans.clone(), ..Default::default() };
    let mut ports = control::Ports::default();
    match type_name {
        nodes::audio_in::TYPE => {
            let (producer, consumer) = rtrb::RingBuffer::new(control::INBOX_RING);
            birth.inbox = Some(consumer);
            ports.audio_in = Some((Arc::new(Mutex::new(producer)), chans));
        }
        nodes::midi_in::TYPE => {
            let (producer, consumer) = rtrb::RingBuffer::new(control::NOTE_RING);
            birth.notes = Some(consumer);
            ports.midi_in = Some(Arc::new(Mutex::new(producer)));
        }
        _ => {}
    }
    (birth, ports)
}

pub(crate) struct Instance {
    pub(crate) idx: usize,
    /// Which occupant of `idx` this is: a plan compiled for an earlier one must not drive it.
    pub(crate) serial: u64,
    pub(crate) manifest: &'static NodeManifest,
    /// Answers `channels` on the control thread; the box that processes never leaves the audio
    /// thread.
    pub(crate) twin: Box<dyn AudioNode>,
    pub(crate) control: Handle,
    /// What the control half was last told; a settle that changes nothing says nothing.
    last: Option<Desired>,
}

pub struct AudioEngine {
    instance: String,
    started: Instant,
    clock: Clock,
    device: Option<DeviceClock>,
    /// The name last tried and what it answered: a name that failed is not tried again until it
    /// moves, because the open runs under the graph lock.
    tried: Option<(String, Option<String>)>,
    stats: Arc<Stats>,
    shared: Arc<Shared>,
    classes: Vec<(&'static NodeManifest, nodes::Make)>,
    runtime: Arc<Mutex<Runtime>>,
    inbox: rtrb::Producer<Msg>,
    outbox: rtrb::Consumer<Retired>,
    free: Vec<usize>,
    slab_len: usize,
    next_serial: u64,
    live: HashMap<Uid, Instance>,
    faulted: HashMap<Uid, String>,
    pending: Vec<(Uid, Status)>,
    dirty: bool,
    last: Plan,
    /// Every bell onto a control half is built from it — last, because fields drop in order.
    bells: goofi_transport::IoxNode,
}

const SLAB: usize = 64;
const QUEUE: usize = 4096;

impl AudioEngine {
    pub fn new(instance: String, started: Instant, waker: Arc<DrainWaker>, clock: Clock) -> AudioEngine {
        let classes = nodes::SHIPPED
            .iter()
            .map(|(type_name, m, make)| {
                let manifest: &'static NodeManifest = Box::leak(Box::new(NodeManifest {
                    type_name,
                    category: m.category,
                    doc: m.doc,
                    inputs: m.inputs,
                    outputs: m.outputs,
                    params: m.params,
                    producer: false,
                }));
                (manifest, *make)
            })
            .collect();
        let (inbox, to_audio) = rtrb::RingBuffer::new(QUEUE);
        let (from_audio, outbox) = rtrb::RingBuffer::new(QUEUE);
        AudioEngine {
            instance,
            started,
            clock,
            device: None,
            tried: None,
            stats: Arc::new(Stats::default()),
            shared: Arc::new(Shared {
                evaluator: Mutex::new(None),
                reports: Mutex::new(Vec::new()),
                waker,
                replan: Default::default(),
                rate: AtomicU64::new(RATE.to_bits()),
            }),
            classes,
            runtime: Arc::new(Mutex::new(Runtime::new(SLAB, to_audio, from_audio))),
            inbox,
            outbox,
            free: (0..SLAB).rev().collect(),
            slab_len: SLAB,
            next_serial: 0,
            live: HashMap::new(),
            faulted: HashMap::new(),
            pending: Vec::new(),
            dirty: false,
            last: Plan::default(),
            bells: goofi_transport::iox_node().expect("an iceoryx2 node for the audio engine's bells"),
        }
    }

    /// The external clock: render whole blocks until `frames` are ready, and hand them over
    /// interleaved — exactly what a device callback would receive.
    pub fn drive(&mut self, frames: usize) -> (Vec<f32>, u16) {
        let mut rt = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        let channels = rt.channels();
        let mut out = vec![0.0; frames * channels as usize];
        rt.render_into(&mut out);
        (out, channels)
    }

    /// Read without the runtime lock: taking it under the graph lock would cost a callback its block.
    pub fn status(&self) -> AudioStatus {
        AudioStatus {
            clock: match self.clock {
                Clock::External => "external",
                Clock::Device => "device",
            },
            device: self.device.as_ref().map(|d| d.name.clone()),
            rate: self.shared.rate(),
            channels: self.device.as_ref().map_or(self.last.output.1, |d| d.channels),
            callbacks: self.stats.callbacks.load(Ordering::Relaxed),
            xruns: self.stats.xruns.load(Ordering::Relaxed),
            render_max_us: self.stats.render_max_us.load(Ordering::Relaxed),
        }
    }

    /// What the audio thread handed back is dropped here, where dropping may take time.
    fn discard_retired(&mut self) {
        while let Ok(retired) = self.outbox.pop() {
            match retired {
                Retired::Slot(slot) => drop(slot),
                Retired::Plan(plan, arena) => drop((plan, arena)),
                Retired::Slab(slab) => drop(slab),
            }
        }
    }

    /// A message always lands. A full ring means the audio thread has not run for a long time, so
    /// taking its lock to apply the backlog costs no block.
    fn send(&mut self, mut msg: Msg) {
        while let Err(rtrb::PushError::Full(back)) = self.inbox.push(msg) {
            self.runtime.lock().unwrap_or_else(|e| e.into_inner()).apply_pending();
            msg = back;
        }
    }

    fn slot_index(&mut self) -> usize {
        if let Some(idx) = self.free.pop() {
            return idx;
        }
        let bigger = self.slab_len * 2;
        self.send(Msg::Grow((0..bigger).map(|_| None).collect()));
        self.free.extend((self.slab_len + 1..bigger).rev());
        let idx = self.slab_len;
        self.slab_len = bigger;
        idx
    }

    /// Everything one node's control half holds, read off the settled view: the constants, the
    /// Array inputs it drains, the bindings it evaluates, and the doors each output rings.
    fn desired_of(&self, view: &GraphView<'_>, uid: Uid, nv: &NodeView<'_>) -> Desired {
        let manifest = self.live[&uid].manifest;
        let consts = manifest.params.iter().map(|d| plan::param_of(nv.params, d)).collect();
        let mut subs = Vec::new();
        for (i, s) in manifest.inputs.iter().enumerate() {
            let Some(inbox) = plan::inbox_of(manifest, i) else { continue };
            let wired = view.wires_into(uid, s.name).next();
            if let Some(service) = wired.and_then(|(p, slot)| goofi_transport::output_of(view, p, slot)) {
                subs.push(Sub::Slot { inbox, service });
            }
        }
        for (param, d) in manifest.params.iter().enumerate() {
            let bound = nv.bindings.iter().find(|b| b.live && b.key.group == d.group && b.key.name == d.name);
            let Some(b) = bound.filter(|b| !plan::is_edge(b, &self.live)) else { continue };
            let vars = b.vars.iter().map(|v| goofi_transport::var_of(view, v)).collect();
            subs.push(Sub::Bind { param, key: b.key.clone(), source: b.rewritten.to_string(), id: b.id, vars });
        }
        let targets = manifest
            .outputs
            .iter()
            .map(|o| {
                view.ringers(uid, o.name)
                    .into_iter()
                    .filter(|r| !self.rides_the_plan(r))
                    .filter_map(|r| Some((goofi_transport::door_of(view, r.consumer)?, r.event_id)))
                    .collect()
            })
            .collect();
        Desired { consts, subs, targets }
    }

    /// Whether a ring would wake a same-engine consumer for what is a plan edge: an audio-typed
    /// input's wire, or a bare audio reference.
    fn rides_the_plan(&self, r: &Ringer<'_>) -> bool {
        let Some(consumer) = self.live.get(&r.consumer) else { return false };
        match r.via {
            Via::Slot(s) => consumer.manifest.inputs.iter().any(|i| i.name == s && i.kind == SlotType::Audio),
            Via::Binding(b) => plan::is_edge(b, &self.live),
        }
    }

    /// Every `AudioOut` with the device it names, by uid — the first is the clock's.
    fn audio_outs(&self, view: &GraphView<'_>) -> Vec<(Uid, String)> {
        let mut outs: Vec<(Uid, String)> = self
            .live
            .iter()
            .filter(|(_, inst)| inst.manifest.type_name == audio_out::TYPE)
            .filter_map(|(uid, inst)| {
                let nv = view.nodes.get(uid)?;
                let Param::Str { value, .. } = plan::param_of(nv.params, &inst.manifest.params[audio_out::P::DEVICE]) else { return None };
                Some((*uid, value))
            })
            .collect();
        outs.sort_by_key(|(uid, _)| uid.0);
        outs
    }

    /// Open, close or switch the output stream to `wanted`, and the error a device that will not
    /// open answers with. A name is tried ONCE: the previous clock is reopened and stands, and the
    /// error stands with it until the name moves. The old stream stops before the new one opens,
    /// because two names of one exclusive device cannot be open at once: silence during a switch.
    fn follow(&mut self, wanted: Option<&str>) -> Option<String> {
        if self.device.as_ref().map(|d| d.name.as_str()) == wanted {
            return None;
        }
        let Some(name) = wanted else {
            self.close();
            self.tried = None;
            return None;
        };
        if let Some((tried, error)) = &self.tried {
            if tried == name {
                return error.clone();
            }
        }
        let previous = self.close();
        let error = self.open(name).err();
        if error.is_some() {
            if let Some(previous) = previous {
                let _ = self.open(&previous);
            }
        }
        self.tried = Some((name.to_string(), error.clone()));
        error
    }

    /// Stop the clock and wait for it; the name it had.
    fn close(&mut self) -> Option<String> {
        let clock = self.device.take()?;
        self.runtime.lock().unwrap_or_else(|e| e.into_inner()).set_device(None);
        Some(clock.name.clone())
    }

    fn open(&mut self, name: &str) -> Result<(), String> {
        let (clock, rate) = DeviceClock::open(name, self.runtime.clone(), self.stats.clone(), self.shared.waker.clone())?;
        self.retune(rate, clock.channels);
        clock.play();
        self.device = Some(clock);
        Ok(())
    }

    /// The device's rate and width, under the runtime lock: every instance — the ones still on
    /// the ring included — is re-prepared when the rate moved, and the FIFO is re-cut to the width.
    fn retune(&mut self, rate: f64, channels: u16) {
        let mut rt = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        rt.apply_pending();
        if rate != self.shared.rate() {
            for slot in rt.slab.iter_mut().flatten() {
                slot.node.prepare(rate);
            }
            self.shared.rate.store(rate.to_bits(), Ordering::Relaxed);
        }
        rt.set_device(Some(channels));
    }
}

impl Engine for AudioEngine {
    fn id(&self) -> &'static str {
        "audio"
    }

    fn doorbell_driven(&self) -> bool {
        true
    }

    fn dirty(&self) -> bool {
        self.dirty || self.shared.replan.load(Ordering::Acquire)
    }

    fn library(&self) -> Vec<LibraryEntry> {
        self.classes.iter().map(|(manifest, _)| LibraryEntry { manifest, isolation: &NATIVE }).collect()
    }

    fn rust_sdk(&self) -> Option<&'static str> {
        Some("goofi-audio-sdk")
    }

    fn normalize_params(&self, type_name: &str, supplied: Option<ParamGroups>) -> Result<ParamGroups, String> {
        let (manifest, _) = self
            .classes
            .iter()
            .find(|(m, _)| m.type_name == type_name)
            .ok_or_else(|| format!("no node type `{type_name}` in the audio library"))?;
        let mut params = manifest.default_params();
        for (group, entries) in supplied.into_iter().flatten() {
            params.entry(group).or_default().extend(entries);
        }
        Ok(params)
    }

    fn insert(&mut self, uid: Uid, type_name: &str, generation: u64, params: &ParamGroups) -> Option<String> {
        let Some((manifest, make)) = self.classes.iter().find(|(m, _)| m.type_name == type_name).copied() else {
            return Some(format!("no audio node type `{type_name}`"));
        };
        let widest = manifest.params.len().max(manifest.inputs.len()).max(manifest.outputs.len());
        if widest > MAX_PORTS {
            return Some(format!("`{type_name}` declares more than {MAX_PORTS} ports"));
        }
        let chans = Arc::new(AtomicU16::new(1));
        let (birth, ports) = rings_for(type_name, chans.clone());
        let mut node = make(birth);
        node.prepare(self.shared.rate());
        let atomics: Arc<[AtomicU64]> =
            manifest.params.iter().map(|d| AtomicU64::new(plan::scalar_of(params, d).to_bits())).collect();
        let (inbox_in, inbox_out): (Vec<_>, Vec<_>) = manifest
            .inputs
            .iter()
            .filter(|s| s.kind != SlotType::Audio)
            .map(|_| rtrb::RingBuffer::<f32>::new(control::INBOX_RING))
            .unzip();
        let (tap_in, tap_out): (Vec<_>, Vec<_>) =
            manifest.outputs.iter().map(|_| rtrb::RingBuffer::<f32>::new(control::TAP_RING)).unzip();
        let spawn = control::Spawn {
            uid,
            base: goofi_transport::service_base(&self.instance, uid, generation),
            manifest,
            params: atomics.clone(),
            inboxes: inbox_in,
            taps: tap_out,
            ports,
            started: self.started,
        };
        let control = match control::spawn(spawn, self.shared.clone(), &self.bells) {
            Ok(handle) => handle,
            Err(e) => return Some(e),
        };
        let idx = self.slot_index();
        let serial = self.next_serial;
        self.next_serial += 1;
        let slot = Slot { serial, node, params: atomics, inboxes: inbox_out.into_iter().map(Inbox::new).collect(), taps: tap_in };
        self.send(Msg::Insert { idx, slot });
        let twin = make(nodes::Birth { chans, ..Default::default() });
        self.live.insert(uid, Instance { idx, serial, manifest, twin, control, last: None });
        self.pending.push((uid, Status::Stage { stage: NodeStage::Ready }));
        self.dirty = true;
        self.shared.waker.notify();
        None
    }

    fn remove(&mut self, uid: Uid) {
        if let Some(inst) = self.live.remove(&uid) {
            inst.control.stop();
            self.send(Msg::Remove(inst.idx));
            self.free.push(inst.idx);
            self.faulted.remove(&uid);
            self.pending.retain(|(u, _)| *u != uid);
            self.dirty = true;
        }
    }

    fn settle(&mut self, view: &GraphView<'_>, _touched: &[Touched]) {
        self.dirty = false;
        self.shared.replan.swap(false, Ordering::Acquire);
        for uid in self.live.keys().copied().collect::<Vec<_>>() {
            let Some(nv) = view.nodes.get(&uid) else { continue };
            let desired = self.desired_of(view, uid, nv);
            let inst = self.live.get_mut(&uid).expect("live");
            if inst.last.as_ref() != Some(&desired) {
                inst.control.send(desired.clone());
                inst.last = Some(desired);
            }
        }
        let outs = self.audio_outs(view);
        let clock = outs.first().map(|(_, device)| device.clone());
        let agrees = |device: &String| Some(device) == clock.as_ref();
        let mut faults: Vec<(Uid, String)> = outs
            .iter()
            .filter(|(_, device)| !agrees(device))
            .map(|(uid, _)| (*uid, format!("the clock is on `{}`", clock.as_deref().unwrap_or_default())))
            .collect();
        if self.clock == Clock::Device {
            if let Some(why) = self.follow(clock.as_deref()) {
                faults.extend(outs.iter().filter(|(_, device)| agrees(device)).map(|(uid, _)| (*uid, why.clone())));
            }
        }
        let silent: Vec<Uid> = faults.iter().map(|(u, _)| *u).collect();
        let (plan, looped) = plan::compile(view, &self.live, &silent);
        faults.extend(looped);
        let since = self.started.elapsed().as_secs_f64();
        let now: HashMap<Uid, String> = faults.into_iter().collect();
        for uid in self.faulted.keys().filter(|u| !now.contains_key(u)) {
            self.pending.push((*uid, Status::Fault { fault: None }));
        }
        for (uid, msg) in &now {
            if self.faulted.get(uid) != Some(msg) {
                self.pending.push((*uid, Status::Fault { fault: Some(NodeFault::Process { msg: msg.clone(), since }) }));
            }
        }
        self.faulted = now;
        if plan != self.last {
            let arena = vec![0.0; plan.arena_len];
            self.send(Msg::Plan { plan: plan.clone(), arena });
            self.last = plan;
        }
        if !self.pending.is_empty() {
            self.shared.waker.notify();
        }
    }

    fn drain(&mut self, apply: &mut dyn FnMut(Uid, Status)) -> usize {
        self.discard_retired();
        // A stream that died is closed here, and its name tried once more at the settle this asks for.
        if self.stats.dead.swap(false, Ordering::Acquire) {
            self.close();
            self.tried = None;
            self.dirty = true;
        }
        let mut pending = std::mem::take(&mut self.pending);
        pending.append(&mut self.shared.reports.lock().unwrap());
        let n = pending.len();
        for (uid, status) in pending {
            apply(uid, status);
        }
        n
    }

    /// A refresh runs on the node's own thread, never under the graph lock.
    fn request(&mut self, uid: Uid, request: Request) {
        let Request::RefreshParam { key } = request;
        if let Some(inst) = self.live.get(&uid) {
            inst.control.refresh(key);
        }
    }

    /// Every control half born after computes `t` from the new origin.
    fn reset_clock(&mut self, origin: Instant) {
        self.started = origin;
    }

    fn set_evaluator(&mut self, evaluator: Arc<dyn goofi_node::ExprEvaluator>) {
        *self.shared.evaluator.lock().unwrap() = Some(evaluator);
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Stop the clock and every control half, and WAIT for each to release its shared memory — a
    /// ceiling, because only a process about to EXIT has no "a moment later".
    fn shutdown(&mut self) {
        self.close();
        let halts: Vec<Arc<goofi_transport::Halt>> = self.live.values().map(|i| i.control.halt.clone()).collect();
        for uid in self.live.keys().copied().collect::<Vec<_>>() {
            self.remove(uid);
        }
        let deadline = Instant::now() + SHUTDOWN_WAIT;
        while halts.iter().any(|h| !h.released()) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Ok(mut rt) = self.runtime.lock() {
            rt.render_block();
        }
        self.discard_retired();
    }
}

impl Shared {
    pub(crate) fn rate(&self) -> f64 {
        f64::from_bits(self.rate.load(Ordering::Relaxed))
    }
}
