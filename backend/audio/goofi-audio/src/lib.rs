//! The audio engine behind the `Engine` seam: synchronous, in-process, one 64-frame block per
//! callback. The engine (this file) owns the library, the slab indices, the plan and the clock;
//! the audio half (`runtime`) owns the instances and the arena, and hears from here by message; a
//! node's control half (`control`) is a thread of its own, parked on the node's door.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use goofi_audio_sdk::host::Loaded;
use goofi_audio_sdk::{AudioNode, BLOCK, MAX_PORTS};
use goofi_core::{Param, SlotType};
use goofi_node::{
    DrainWaker, Engine, GraphView, LibraryEntry, NodeFault, NodeManifest, NodeStage, NodeView, ParamGroups,
    Request, Ringer, Status, Touched, Uid, Via, NATIVE,
};

mod control;
pub mod nodes;
mod plan;
mod runtime;
mod scan;
pub mod vst3;

use control::{Desired, Handle, Shared, Sub};
use nodes::{audio_out, Class};
use plan::Plan;
use runtime::{Fault, Inbox, Msg, Retired, Runtime, Slot, OVERRUNS};

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
    classes: HashMap<&'static str, Class>,
    /// Every built artifact loaded so far, by path: a library is opened once and never closed.
    rust_loaded: HashMap<PathBuf, Arc<Loaded>>,
    /// The child a bundle is scanned in, and the platform's plugin folders: the composition root's.
    vst3: Option<(PathBuf, Vec<PathBuf>)>,
    runtime: Arc<Mutex<Runtime>>,
    inbox: rtrb::Producer<Msg>,
    outbox: rtrb::Consumer<Retired>,
    free: Vec<usize>,
    slab_len: usize,
    next_serial: u64,
    live: HashMap<Uid, Instance>,
    workspace: Option<PathBuf>,
    /// A workspace that turned over, swept of dead blobs at the first settled state after it.
    sweep: bool,
    /// Nodes the audio thread put out of the plan — a panic, or the watchdog — until a restart.
    disabled: HashMap<Uid, String>,
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
        let classes: HashMap<&'static str, Class> = nodes::BUILT_IN
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
                (*type_name, Class { manifest, make: Arc::new(*make), plugin: None })
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
            rust_loaded: HashMap::new(),
            vst3: None,
            runtime: Arc::new(Mutex::new(Runtime::new(SLAB, to_audio, from_audio))),
            inbox,
            outbox,
            free: (0..SLAB).rev().collect(),
            slab_len: SLAB,
            next_serial: 0,
            live: HashMap::new(),
            workspace: None,
            sweep: false,
            disabled: HashMap::new(),
            faulted: HashMap::new(),
            pending: Vec::new(),
            dirty: false,
            last: Plan::default(),
            bells: goofi_transport::iox_node().expect("an iceoryx2 node for the audio engine's bells"),
        }
    }

    /// Where a `.vst3` bundle is scanned — a `goofi` answering `vst3-scan` — and which folders
    /// are scanned on the engine's own account, after every root.
    pub fn set_vst3(&mut self, scanner: PathBuf, dirs: Vec<PathBuf>) {
        self.vst3 = Some((scanner, dirs));
    }

    /// The external clock: render whole blocks until `frames` are ready, and hand them over
    /// interleaved — exactly what a device callback would receive.
    pub fn drive(&mut self, frames: usize) -> (Vec<f32>, u16) {
        let mut rt = self.runtime();
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

    /// Where `uid`'s opaque state is kept between two births; named by type, so a chosen uid
    /// never hands one type the bytes another left.
    fn state_path(&self, uid: Uid, type_name: &str) -> Option<PathBuf> {
        Some(self.state_dir()?.join(uid.to_hex()).join(type_name))
    }

    fn state_dir(&self) -> Option<PathBuf> {
        Some(self.workspace.as_ref()?.join(".goofi").join("state"))
    }

    /// Every blob of a uid the patch does not hold — a delete whose undo a load ended — so a
    /// number minted again is never born on a dead node's bytes.
    fn sweep_state(&self) {
        let Some(entries) = self.state_dir().and_then(|d| std::fs::read_dir(d).ok()) else { return };
        for entry in entries.flatten() {
            let held = entry.file_name().to_str().and_then(Uid::from_hex).is_some_and(|u| self.live.contains_key(&u));
            if !held {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }

    /// Keep what `save` answered, or nothing: a node with no state leaves no file behind.
    fn write_state(&self, uid: Uid, type_name: &str, bytes: Vec<u8>) {
        let Some(path) = self.state_path(uid, type_name) else { return };
        let written = if bytes.is_empty() {
            std::fs::remove_file(&path).or_else(|e| if e.kind() == std::io::ErrorKind::NotFound { Ok(()) } else { Err(e) })
        } else {
            std::fs::create_dir_all(path.parent().expect("a state file has a directory")).and_then(|()| std::fs::write(&path, bytes))
        };
        if let Err(e) = written {
            eprintln!("audio: could not keep {}: {e}", path.display());
        }
    }

    /// What the audio thread handed back is dropped here, where dropping may take time — and a
    /// node it put out of the plan is recorded, for the settle this asks for to name and re-plan.
    fn discard_retired(&mut self) {
        while let Ok(retired) = self.outbox.pop() {
            match retired {
                Retired::Slot(slot) => self.write_state(slot.uid, slot.type_name, slot.node.save()),
                Retired::Plan(plan, arena) => drop((plan, arena)),
                Retired::Slab(slab) => drop(slab),
                Retired::Faulted { uid, serial, fault } => {
                    if self.live.get(&uid).is_some_and(|i| i.serial == serial) {
                        let msg = match fault {
                            Fault::Panic(msg) => msg,
                            Fault::Overrun => format!("process overran the block {OVERRUNS} times in a row"),
                        };
                        self.disabled.insert(uid, msg);
                        self.dirty = true;
                    }
                }
            }
        }
    }

    /// A message always lands. A full ring means the audio thread has not run for a long time, so
    /// taking its lock to apply the backlog costs no block.
    /// The audio half's lock, held through a panic there: the slab is still the slab.
    fn runtime(&self) -> std::sync::MutexGuard<'_, Runtime> {
        self.runtime.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn send(&mut self, mut msg: Msg) {
        while let Err(rtrb::PushError::Full(back)) = self.inbox.push(msg) {
            self.runtime().apply_pending();
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
            .filter(|(uid, inst)| inst.manifest.type_name == audio_out::TYPE && !self.disabled.contains_key(uid))
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
        self.runtime().set_device(None);
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
        let mut rt = self.runtime();
        rt.apply_pending();
        if rate != self.shared.rate() {
            for slot in rt.slab.iter_mut().flatten() {
                slot.node.prepare(rate);
            }
            self.shared.rate.store(rate.to_bits(), Ordering::Relaxed);
            rt.block = Duration::from_secs_f64(BLOCK as f64 / rate);
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
        self.classes.values().map(|c| LibraryEntry { manifest: c.manifest, isolation: &NATIVE }).collect()
    }

    fn scan(&mut self, dir: &Path) -> Vec<goofi_node::ScannedType> {
        scan::scan(self, dir)
    }

    fn scan_own(&mut self) -> Vec<goofi_node::ScannedType> {
        let dirs = self.vst3.as_ref().map(|(_, dirs)| dirs.clone()).unwrap_or_default();
        dirs.iter().filter(|d| d.is_dir()).flat_map(|d| vst3::scan_dir(self, d)).collect()
    }

    fn remove_type(&mut self, type_name: &str) -> bool {
        !nodes::built_in(type_name) && self.classes.remove(type_name).is_some()
    }

    fn rust_sdk(&self) -> Option<&'static str> {
        Some(goofi_build::AUDIO.name)
    }

    fn set_workspace(&mut self, dir: &Path) {
        self.workspace = Some(dir.to_path_buf());
        self.sweep = true;
    }

    /// Every live node's state, under the runtime lock.
    fn persist(&mut self) {
        let states: Vec<(Uid, &'static str, Vec<u8>)> = {
            let mut rt = self.runtime();
            rt.apply_pending();
            rt.slab.iter().flatten().map(|s| (s.uid, s.type_name, s.node.save())).collect()
        };
        for (uid, type_name, bytes) in states {
            self.write_state(uid, type_name, bytes);
        }
    }

    fn normalize_params(&self, manifest: &'static NodeManifest, supplied: Option<ParamGroups>) -> ParamGroups {
        let mut params = manifest.default_params();
        for (group, entries) in supplied.into_iter().flatten() {
            params.entry(group).or_default().extend(entries);
        }
        params
    }

    fn insert(&mut self, uid: Uid, type_name: &str, generation: u64, params: &ParamGroups) -> Option<String> {
        let Some(Class { manifest, make, .. }) = self.classes.get(type_name).cloned() else {
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
        if let Some(bytes) = self.state_path(uid, type_name).and_then(|p| std::fs::read(p).ok()) {
            node.load(&bytes);
        }
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
        let slot = Slot {
            uid,
            type_name: manifest.type_name,
            serial,
            node,
            params: atomics,
            inboxes: inbox_out.into_iter().map(Inbox::new).collect(),
            taps: tap_in,
            dead: false,
            overruns: 0,
        };
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
            // The box comes back NOW, its state kept, so a restart's birth finds it: one callback
            // may find the runtime taken — a click at an authoring event, accepted.
            self.runtime().apply_pending();
            self.discard_retired();
            self.free.push(inst.idx);
            self.disabled.remove(&uid);
            self.faulted.remove(&uid);
            self.pending.retain(|(u, _)| *u != uid);
            self.dirty = true;
        }
    }

    fn settle(&mut self, view: &GraphView<'_>, _touched: &[Touched]) {
        self.dirty = false;
        if std::mem::take(&mut self.sweep) {
            self.sweep_state();
        }
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
        faults.extend(self.disabled.iter().map(|(u, m)| (*u, m.clone())));
        let (plan, looped) = plan::compile(view, &self.live, &silent, &self.disabled);
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
