//! An audio node's control half: one thread per node, parked on the node's own door. It is the
//! one writer of the node's param atomics — a constant and an evaluated binding land through the
//! same hand — the crossing every Array input enters through, and the tap every reader of an
//! output drinks from.

use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use goofi_audio_sdk::{high, BLOCK, MAX_CHANNELS};
use goofi_core::{Data, Meta, Param};
use goofi_node::{BindingId, DrainWaker, EventId, ExprEvaluator, Expression, NodeManifest, ParamKey, Status, Uid, Var};
use goofi_transport::{
    data_service, door_service, event_service, iox_node, open_output_subscriber, output_service, publisher,
    take_where, ByteService, BytePublisher, ByteSubscriber, Doorbell, Halt, IoxNode, Listener, INITIAL_SLICE,
};
use indexmap::IndexMap;

use crate::nodes::midi_in::{Note, NO_PORT};
use crate::nodes::{audio_in, audio_out, audio_playback, midi_in};
use crate::{plan, wav, Clock, DEFAULT_DEVICE, NO_DEVICE, RATE};

/// How often the paced duties run: a tapped output is published, and a binding with no stream
/// variable is re-evaluated, at this pace whatever rings in between.
pub const TICK: Duration = Duration::from_millis(10);
/// A tap holds this many blocks of the widest output; what does not fit is dropped, newest first.
pub const TAP_RING: usize = (1 + MAX_CHANNELS as usize * BLOCK) * 16;
/// An inbox holds one second of the widest frame at the rate; a frame that does not fit is
/// dropped whole.
pub const INBOX_RING: usize = RATE as usize * MAX_CHANNELS as usize;
/// A take's ring holds one second of the widest block, as an inbox holds one second of a frame.
pub const REC_RING: usize = (1 + MAX_CHANNELS as usize * BLOCK) * (RATE as usize / BLOCK);
/// Notes a port may hold between two blocks.
pub const NOTE_RING: usize = 1024;
/// How often a take patches its size fields, so a goofi that dies leaves a file that still plays.
const SYNC: Duration = Duration::from_secs(1);
/// How much of a file one read takes, in frames of the file's own rate.
const READ_CHUNK: usize = 2048;

/// A ring's producer as an OS callback holds it: successive streams on one node share it, and a
/// callback that finds it taken drops that buffer rather than wait.
pub type Feed<T> = Arc<Mutex<rtrb::Producer<T>>>;

/// The control half's ends of a device's or a port's rings — none for a node that owns no OS
/// handle.
#[derive(Default)]
pub struct Ports {
    pub audio_in: Option<(Feed<f32>, Arc<AtomicU16>)>,
    pub midi_in: Option<Feed<Note>>,
    /// The control half's end of an `AudioOut`'s take ring.
    pub rec: Option<rtrb::Consumer<f32>>,
    /// The ring an `AudioPlayback` fills from its file, and the width the file answered.
    pub play: Option<(rtrb::Producer<f32>, Arc<AtomicU16>)>,
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
    let mut ports = spawn.ports;
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
        rec: ports.rec.take().map(Rec::new),
        play: ports.play.take().map(Play::new),
        ports,
        evaluated: IndexMap::new(),
        errors: IndexMap::new(),
        pulsed: Vec::new(),
        pulses: Vec::new(),
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

/// Everything the audio thread pushed into a ring since the last read, as one planar `[C, T]`
/// frame — up to a block whose channel count differs, which the next read starts from. The
/// framing a tap and a take share, read in one place.
fn drain_blocks(ring: &mut rtrb::Consumer<f32>) -> Option<(usize, Vec<f32>)> {
    let mut chans = 0;
    let mut planar: Vec<Vec<f32>> = Vec::new();
    while let Ok(head) = ring.read_chunk(1) {
        let c = head.as_slices().0.first().copied().unwrap_or(0.0) as usize;
        if c == 0 || (chans != 0 && c != chans) {
            if c == 0 {
                head.commit_all();
            }
            break;
        }
        head.commit_all();
        let Ok(block) = ring.read_chunk(c * BLOCK) else { break };
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

/// An `AudioOut`'s take: the ring its DSP half fills, and the part being written. A break — the
/// width moved, the rate moved, or RIFF filled — closes the part and opens the next.
struct Rec {
    ring: rtrb::Consumer<f32>,
    take: Option<wav::Writer>,
    /// The name every part is numbered from, while a take is asked for.
    stem: Option<PathBuf>,
    part: u32,
    synced: Instant,
    /// The name and `unique` last seen; a move of either lets a take that failed be tried again.
    named: Option<(String, bool)>,
    error: Option<String>,
}

impl Rec {
    fn new(ring: rtrb::Consumer<f32>) -> Rec {
        Rec { ring, take: None, stem: None, part: 0, synced: Instant::now(), named: None, error: None }
    }

    fn drain(&mut self, named: &(String, bool), rate: f64) {
        while let Some((c, planar)) = drain_blocks(&mut self.ring) {
            if self.error.is_some() {
                continue;
            }
            if self.stem.is_none() {
                self.stem = Some(take_stem(&named.0, named.1));
                self.part = 0;
            }
            if let Err(e) = self.write(c, &planar, rate) {
                self.error = Some(e);
                self.stem = None;
                self.take = None;
            }
        }
    }

    fn write(&mut self, c: usize, planar: &[f32], rate: f64) -> Result<(), String> {
        let frames = planar.len() / c;
        let rate = rate.round().max(1.0) as u32;
        if self.take.as_ref().is_some_and(|w| w.channels != c as u16 || w.rate != rate) {
            self.take = None;
        }
        // Twice at most: a part opened for this chunk is empty, so it always takes it.
        for _ in 0..2 {
            if self.take.is_none() {
                self.part += 1;
                let stem = self.stem.clone().ok_or("no take is open")?;
                self.take = Some(wav::Writer::create(&part_path(&stem, self.part), rate, c as u16)?);
                self.synced = Instant::now();
            }
            let take = self.take.as_mut().expect("the part just opened");
            if take.write(planar, frames)? {
                if self.synced.elapsed() >= SYNC {
                    self.synced = Instant::now();
                    take.sync()?;
                }
                return Ok(());
            }
            self.take = None;
        }
        Ok(())
    }
}

/// An `AudioPlayback`'s file: the reader, the crossing that resamples it into the DSP half's
/// ring, and what the params last asked for.
struct Play {
    inbox: Inbox,
    file: Option<wav::Reader>,
    named: Option<String>,
    position: f64,
    ended: bool,
    error: Option<String>,
}

impl Play {
    fn new((ring, chans): (rtrb::Producer<f32>, Arc<AtomicU16>)) -> Play {
        let inbox = Inbox { ring, chans, pos: 0.0 };
        Play { inbox, file: None, named: None, position: 0.0, ended: false, error: None }
    }
}

/// One planar chunk of a file through the crossing every Array input enters by, so a file at its
/// own rate arrives at the engine's.
fn enter_planar(inbox: &mut Inbox, channels: u16, frames: usize, planar: &[f32], from: f64, rate: f64) -> bool {
    let bytes: Vec<u8> = planar.iter().flat_map(|v| v.to_le_bytes()).collect();
    match Data::array_f32(vec![channels as usize, frames], bytes, Meta::new().with_sfreq(Some(from))) {
        Ok(frame) => inbox.enter(&frame, rate).unwrap_or(false),
        Err(_) => false,
    }
}

/// Where a name is looked for: the recordings folder for a bare one, an absolute path as it is,
/// and `.wav` joined on where it is not already there — the spelling a take is written under.
fn source_path(name: &str) -> PathBuf {
    let name = name.trim();
    let name = if name.to_ascii_lowercase().ends_with(".wav") { name.to_string() } else { format!("{name}.wav") };
    if Path::new(&name).is_absolute() {
        PathBuf::from(name)
    } else {
        crate::recordings().join(name)
    }
}

/// Where a take lands: a bare name under the recordings folder, an absolute path as it is, and
/// the time joined on when the name must not be reused.
fn take_stem(file: &str, unique: bool) -> PathBuf {
    let name = file.trim();
    let name = name.strip_suffix(".wav").unwrap_or(name);
    let name = if name.is_empty() { "take" } else { name };
    let stem = if Path::new(name).is_absolute() { PathBuf::from(name) } else { crate::recordings().join(name) };
    if !unique {
        return stem;
    }
    let mut named = stem.into_os_string();
    named.push(format!("-{}", stamp()));
    PathBuf::from(named)
}

fn part_path(stem: &Path, part: u32) -> PathBuf {
    let mut name = stem.to_path_buf().into_os_string();
    if part > 1 {
        name.push(format!("-{part}"));
    }
    name.push(".wav");
    PathBuf::from(name)
}

/// UTC as `YYYYMMDD-HHMMSS`, so takes sort in the order they were played.
fn stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (days, rest) = ((secs / 86_400) as i64, secs % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}{month:02}{day:02}-{:02}{:02}{:02}", rest / 3600, (rest / 60) % 60, rest % 60)
}

impl Out {
    /// Everything the audio thread tapped since the last tick.
    fn drain(&mut self) -> Option<(usize, Vec<f32>)> {
        drain_blocks(&mut self.ring)
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
    /// An `AudioOut`'s take, and an `AudioPlayback`'s file; every other node has neither.
    rec: Option<Rec>,
    play: Option<Play>,
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

    /// A `Str` param's text; every other kind — a number, a bool, a valueless pulse — has none,
    /// and so has a param the first `apply` has not delivered yet: a control half ticks from the
    /// moment its thread starts, and its consts arrive with the first desired state, not before.
    fn text(&self, param: usize) -> String {
        match self.consts.get(param) {
            Some(Param::Str { value, .. }) => value.clone(),
            _ => String::new(),
        }
    }

    fn flag(&self, param: usize) -> bool {
        self.consts.get(param).and_then(|p| p.as_bool()).unwrap_or(false)
    }

    /// The take, driven by what the ring HOLDS: the DSP half pushes only while `record.on` is
    /// high, so the blocks are the request themselves. A take shorter than a tick still lands,
    /// and none loses the head a sampled level would cut. The name is read where the take opens,
    /// and one that will not open stands as an error until the take ends or the name moves.
    /// `record.on` is read live, so a bound gate records too.
    fn record(&mut self) -> Option<String> {
        let rate = self.shared.rate();
        let on = high(f64::from_bits(self.params[audio_out::P::ON].load(Ordering::Relaxed)) as f32);
        let named = (self.text(audio_out::P::FILE), self.flag(audio_out::P::UNIQUE));
        let rec = self.rec.as_mut()?;
        if rec.named.as_ref() != Some(&named) {
            rec.named = Some(named.clone());
            rec.error = None;
        }
        rec.drain(&named, rate);
        if !on {
            rec.take = None;
            rec.stem = None;
            rec.part = 0;
            rec.error = None;
        }
        rec.error.clone()
    }

    /// The file, driven from settled state: a name that moved is opened, a `position` that moved
    /// or a `reset` skips, and the ring is kept a second ahead so the DSP half never runs dry. A
    /// name that will not open stands as an error on it until it moves.
    fn playback(&mut self, pulses: &[usize]) -> Option<String> {
        let rate = self.shared.rate();
        let named = self.text(audio_playback::P::FILE);
        let position = f64::from_bits(self.params[audio_playback::P::POSITION].load(Ordering::Relaxed));
        let reset = pulses.contains(&audio_playback::P::RESET);
        let looping = self.flag(audio_playback::P::LOOPING);
        let play = self.play.as_mut()?;
        let mut moved = false;
        if play.named.as_deref() != Some(named.as_str()) {
            play.named = Some(named.clone());
            play.file = None;
            play.error = None;
            play.ended = false;
            play.position = position;
            play.inbox.pos = 0.0;
            let name = named.trim();
            if !name.is_empty() {
                match wav::Reader::open(&source_path(name)) {
                    Ok(f) if f.frames == 0 => play.error = Some(format!("{} holds no samples", f.path.display())),
                    Ok(f) => {
                        moved = play.inbox.chans.swap(f.channels, Ordering::Relaxed) != f.channels;
                        play.file = Some(f);
                    }
                    Err(e) => play.error = Some(e),
                }
            }
        }
        let mut dead = None;
        if let Some(file) = play.file.as_mut() {
            if reset || position != play.position {
                play.position = position;
                let at = if reset { 0 } else { (position * file.frames as f64) as u64 };
                dead = file.seek(at).err();
                play.ended = false;
                play.inbox.pos = 0.0;
            }
            // An eighth of a second ahead: enough over a tick, and what a skip waits out.
            let want = (rate as usize / 8).saturating_mul(file.channels as usize).min(INBOX_RING / 2);
            while dead.is_none() && INBOX_RING - play.inbox.ring.slots() < want {
                let (got, planar) = match file.read(READ_CHUNK) {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        dead = Some(e);
                        break;
                    }
                };
                if got == 0 {
                    if looping {
                        dead = file.seek(0).err();
                        continue;
                    }
                    if !play.ended {
                        play.ended = true;
                        // One quiet chunk, or `fill` holds the last sample it had as an offset.
                        let quiet = vec![0.0; file.channels as usize * BLOCK];
                        moved |= enter_planar(&mut play.inbox, file.channels, BLOCK, &quiet, file.rate as f64, rate);
                    }
                    break;
                }
                moved |= enter_planar(&mut play.inbox, file.channels, got, &planar, file.rate as f64, rate);
            }
        }
        if let Some(e) = dead {
            play.error = Some(e);
            play.file = None;
        }
        let error = play.error.clone();
        if moved {
            self.shared.replan.store(true, Ordering::Release);
            self.shared.waker.notify();
        }
        error
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
        let pulses = std::mem::take(&mut self.pulses);
        let mut pass = Pass::default();
        for i in 0..self.binds.len() {
            if self.binds[i].streams.is_empty() {
                self.evaluate(i, &mut pass);
            }
        }
        if self.rec.is_some() {
            let error = self.record();
            self.record_error(self.key_of(audio_out::P::ON), error, &mut pass);
        }
        if self.play.is_some() {
            let error = self.playback(&pulses);
            self.record_error(self.key_of(audio_playback::P::FILE), error, &mut pass);
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
