//! One plugin instance behind [`AudioNode`]: instantiated at `prepare` on the control thread, its
//! params and note events queued per block, its buses staged from the arena.

use std::path::PathBuf;
use std::sync::Arc;

use goofi_audio_sdk::{AudioNode, Block, BLOCK, GATE_HIGH, MAX_CHANNELS};
use goofi_node::Stamp;
use vst3::Steinberg::Vst::*;
use vst3::Steinberg::*;
use vst3::{ComPtr, ComWrapper};

use super::host::{Changes, Events, Host, Stream};
use super::module;
use super::ok;

/// The tempo the host reports. A CONSTANT for now, and a lie only in the sense that goofi has no
/// transport of its own to tell the truth from — but a plausible tempo is what a synced plugin
/// needs to run at all, where zero is what stops it.
const TEMPO: f64 = 120.0;

/// How a goofi param's scalar becomes the plugin's normalized value.
pub enum Kind {
    Float,
    Stepped(f64),
}

/// What the scan derived for one class: enough to stage a block and to instantiate.
pub struct Derived {
    pub binary: PathBuf,
    pub stamp: Stamp,
    pub cid: TUID,
    pub inputs: Vec<u16>,
    pub outputs: Vec<u16>,
    pub params: Vec<(ParamID, Kind)>,
}

pub struct Plugin {
    class: Arc<Derived>,
    live: Option<Live>,
    /// What instantiation refused; every `process` raises it, because the runtime marks a node
    /// dead only once the fault is on its way and expects the next block to fault again.
    failed: Option<String>,
    /// What `load` was handed. Kept, so a failed instantiation cannot answer `save` with nothing
    /// and have the engine delete the preset behind it.
    blob: Vec<u8>,
}

impl Plugin {
    pub fn new(class: Arc<Derived>) -> Plugin {
        Plugin { class, live: None, failed: None, blob: Vec::new() }
    }
}

impl AudioNode for Plugin {
    fn channels(&self, _ins: &[u16], _params: &[f64], outs: usize) -> Vec<u16> {
        (0..outs).map(|i| self.class.outputs.get(i).copied().unwrap_or(1).clamp(1, MAX_CHANNELS)).collect()
    }

    fn prepare(&mut self, rate: f64) {
        let result = match self.live.as_mut() {
            Some(live) => live.retune(rate),
            None => Live::open(&self.class, rate).map(|live| {
                live.load(&self.blob);
                self.live = Some(live)
            }),
        };
        self.failed = result.err();
    }

    fn process(&mut self, b: &mut Block<'_>) {
        if let Some(text) = &self.failed {
            std::panic::resume_unwind(Box::new(text.clone()));
        }
        match self.live.as_mut() {
            Some(live) => live.block(&self.class, b),
            None => {
                for out in b.outs.iter_mut() {
                    for c in 0..out.channels() as usize {
                        out.chan_mut(c).fill(0.0);
                    }
                }
            }
        }
    }

    fn save(&self) -> Vec<u8> {
        let Some(live) = &self.live else { return self.blob.clone() };
        let stream = ComWrapper::new(Stream::default());
        let ptr = stream.to_com_ptr::<IBStream>().expect("a stream is an IBStream");
        if unsafe { live.component.getState(ptr.as_ptr()) } != kResultOk {
            return self.blob.clone();
        }
        stream.bytes.take()
    }

    fn load(&mut self, bytes: &[u8]) {
        self.blob = bytes.to_vec();
        if let Some(live) = self.live.as_mut() {
            live.load(bytes);
        }
    }
}

struct Live {
    _host: ComPtr<FUnknown>,
    component: ComPtr<IComponent>,
    processor: ComPtr<IAudioProcessor>,
    /// The plugin's OTHER half, connected to the component for as long as this instance lives. Not
    /// here to be read — nothing asks it anything — but because a plugin whose halves were never
    /// introduced can sit muted, exactly as it reports no parameters when the scanner skips this.
    controller: Option<ComPtr<IEditController>>,
    wired: Option<Wire>,
    changes: ComWrapper<Changes>,
    changes_ptr: ComPtr<IParameterChanges>,
    events: ComWrapper<Events>,
    events_ptr: ComPtr<IEventList>,
    context: Box<ProcessContext>,
    ins: Buses,
    outs: Buses,
    /// The normalized value last handed over per plugin param; NaN sends it at the next block.
    sent: Vec<f64>,
    held: [Option<i16>; MAX_CHANNELS as usize],
    /// Whether `setupProcessing` has run: what a re-prepare must undo and a first one must not.
    prepared: bool,
}

// The one deviation from "no unsafe impl": the VST3 contract lets `process` run on a thread of
// the host's choosing once `setProcessing` was called, and these pointers cross exactly once.
unsafe impl Send for Live {}

impl Live {
    fn open(class: &Derived, rate: f64) -> Result<Live, String> {
        let factory = module::factory(&class.binary)?;
        let component: ComPtr<IComponent> = factory.create(&class.cid)?;
        let host = ComWrapper::new(Host).to_com_ptr::<FUnknown>().expect("a host is an FUnknown");
        unsafe { ok(component.initialize(host.as_ptr()), "initialize")? };
        let Some(processor) = component.cast::<IAudioProcessor>() else {
            unsafe { component.terminate() };
            return Err("the component is no IAudioProcessor".into());
        };
        let (controller, wired) = unsafe { pair(&factory, &component, &host) };
        let (ins, outs) = unsafe { arrange(&component, &processor, class) };
        let changes = ComWrapper::new(Changes::new(class.params.iter().map(|(id, _)| *id)));
        let changes_ptr = changes.to_com_ptr().expect("changes are an IParameterChanges");
        let events = ComWrapper::new(Events::with_capacity(BLOCK * MAX_CHANNELS as usize));
        let events_ptr = events.to_com_ptr().expect("events are an IEventList");
        let mut live = Live {
            _host: host,
            component,
            processor,
            controller,
            wired,
            changes,
            changes_ptr,
            events,
            events_ptr,
            context: Box::new(unsafe { std::mem::zeroed() }),
            ins,
            outs,
            sent: vec![f64::NAN; class.params.len()],
            held: [None; MAX_CHANNELS as usize],
            prepared: false,
        };
        live.retune(rate)?;
        Ok(live)
    }

    fn retune(&mut self, rate: f64) -> Result<(), String> {
        let mut setup = ProcessSetup {
            processMode: ProcessModes_::kRealtime as int32,
            symbolicSampleSize: SymbolicSampleSizes_::kSample32 as int32,
            maxSamplesPerBlock: BLOCK as int32,
            sampleRate: rate,
        };
        unsafe {
            if self.prepared {
                self.processor.setProcessing(0);
                self.component.setActive(0);
            }
            self.prepared = false;
            ok(self.processor.setupProcessing(&mut setup), "setupProcessing")?;
            activate_buses(&self.component, self.ins.buses.len(), self.outs.buses.len());
            ok(self.component.setActive(1), "setActive")?;
            // OPTIONAL in the SDK: a plugin that does not distinguish processing from active
            // answers kNotImplemented, which is an answer rather than a refusal.
            let processing = self.processor.setProcessing(1);
            if processing != kNotImplemented {
                ok(processing, "setProcessing")?;
            }
        }
        self.prepared = true;
        self.context.sampleRate = rate;
        // A playing, advancing transport: a zeroed context tells a plugin the host is stopped at
        // 0 BPM, and a tempo-synced engine then correctly produces nothing.
        self.context.state = (ProcessContext_::StatesAndFlags_::kPlaying
            | ProcessContext_::StatesAndFlags_::kTempoValid
            | ProcessContext_::StatesAndFlags_::kTimeSigValid
            | ProcessContext_::StatesAndFlags_::kProjectTimeMusicValid
            | ProcessContext_::StatesAndFlags_::kContTimeValid) as uint32;
        self.context.tempo = TEMPO;
        self.context.timeSigNumerator = 4;
        self.context.timeSigDenominator = 4;
        // The reactivation dropped the plugin's voices, so a gate still HIGH must note again.
        self.sent.fill(f64::NAN);
        self.held = [None; MAX_CHANNELS as usize];
        Ok(())
    }

    fn load(&self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let stream = ComWrapper::new(Stream::of(bytes));
        let ptr = stream.to_com_ptr::<IBStream>().expect("a stream is an IBStream");
        unsafe { self.component.setState(ptr.as_ptr()) };
    }

    fn block(&mut self, class: &Derived, b: &mut Block<'_>) {
        // The voice params, if any, are the ones the manifest carries beyond the plugin's own.
        let voice = b.params.len() - class.params.len();
        self.changes.clear();
        for (i, (_, kind)) in class.params.iter().enumerate() {
            let raw = b.params[voice + i].chan(0)[0] as f64;
            let value = match kind {
                Kind::Float => raw,
                Kind::Stepped(steps) => raw / steps,
            }
            .clamp(0.0, 1.0);
            if self.sent[i].to_bits() != value.to_bits() {
                self.sent[i] = value;
                self.changes.set(i, value);
            }
        }
        self.events.clear();
        if voice == 3 {
            let (gate, pitch, velocity) = (&b.params[0], &b.params[1], &b.params[2]);
            for c in 0..(gate.channels() as usize).min(MAX_CHANNELS as usize) {
                for (s, &g) in gate.chan(c).iter().enumerate() {
                    match (g >= GATE_HIGH, self.held[c]) {
                        (true, None) => {
                            let note = (60.0 + 12.0 * pitch.chan(c)[s]).round().clamp(0.0, 127.0) as i16;
                            self.held[c] = Some(note);
                            self.events.push(note_on(c, s, note, velocity.chan(c)[s]));
                        }
                        (false, Some(note)) => {
                            self.held[c] = None;
                            self.events.push(note_off(c, s, note));
                        }
                        _ => {}
                    }
                }
            }
        }
        for (bus, port) in self.ins.buses.iter_mut().zip(b.ins.iter()) {
            bus.silenceFlags = if port.wired() { 0 } else { u64::MAX };
        }
        for (i, port) in b.ins.iter().enumerate().take(self.ins.pointers.len()) {
            for (c, dst) in self.ins.pointers[i].iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(port.chan(c).as_ptr(), *dst, BLOCK) };
            }
        }
        for lanes in &self.outs.pointers {
            for dst in lanes {
                unsafe { std::ptr::write_bytes(*dst, 0, BLOCK) };
            }
        }
        let mut data = ProcessData {
            processMode: ProcessModes_::kRealtime as int32,
            symbolicSampleSize: SymbolicSampleSizes_::kSample32 as int32,
            numSamples: BLOCK as int32,
            numInputs: self.ins.buses.len() as int32,
            numOutputs: self.outs.buses.len() as int32,
            inputs: self.ins.buses.as_mut_ptr(),
            outputs: self.outs.buses.as_mut_ptr(),
            inputParameterChanges: self.changes_ptr.as_ptr(),
            outputParameterChanges: std::ptr::null_mut(),
            inputEvents: self.events_ptr.as_ptr(),
            outputEvents: std::ptr::null_mut(),
            processContext: &mut *self.context,
        };
        unsafe { self.processor.process(&mut data) };
        // Advanced AFTER the block it described, so the plugin's clock runs at the rate its own
        // audio does. A transport that is valid but frozen is a host paused on the first sample.
        self.context.projectTimeSamples += BLOCK as i64;
        self.context.continousTimeSamples += BLOCK as i64;
        self.context.projectTimeMusic += BLOCK as f64 / self.context.sampleRate * (TEMPO / 60.0);
        for (i, out) in b.outs.iter_mut().enumerate().take(self.outs.pointers.len()) {
            let lanes = &self.outs.pointers[i];
            for c in 0..out.channels() as usize {
                let src = lanes[c.min(lanes.len() - 1)];
                unsafe { std::ptr::copy_nonoverlapping(src, out.chan_mut(c).as_mut_ptr(), BLOCK) };
            }
        }
    }
}

/// The buses activated at the plugin's own default arrangements, and the staging sized from what
/// the LIVE instance then reports — never the scan's cached counts, which a plugin whose default
/// layout lives outside its binary can disagree with.
/// A component and its controller, each holding the other's connection point.
pub(super) type Wire = (ComPtr<IConnectionPoint>, ComPtr<IConnectionPoint>);

/// What pairing yields: the other half, and the connection to undo before tearing it down.
type Pair = (Option<ComPtr<IEditController>>, Option<Wire>);

/// The component's other half, introduced to it. The SAME sequence the scanner performs, and for
/// the same reason: the two are peers, and a plugin that cannot talk to its own controller behaves
/// as though it has nothing to say. Every step is optional — a single-object plugin needs none of
/// it — so a failure anywhere leaves the instance exactly as it was rather than refusing it.
unsafe fn pair(
    factory: &module::Factory,
    component: &ComPtr<IComponent>,
    context: &ComPtr<FUnknown>,
) -> Pair {
    if component.cast::<IEditController>().is_some() {
        return (None, None);
    }
    let mut ccid: TUID = [0; 16];
    if component.getControllerClassId(&mut ccid) != kResultOk {
        return (None, None);
    }
    let Ok(controller) = factory.create::<IEditController>(&ccid) else { return (None, None) };
    if controller.initialize(context.as_ptr()) != kResultOk {
        return (None, None);
    }
    let wired = introduce(component, &controller);
    (Some(controller), wired)
}

/// Connect a component and its controller, then seed the controller with the component's state —
/// the one sequence both the scan and the runtime need. The returned wire is undone by [`sunder`].
pub(super) unsafe fn introduce(component: &ComPtr<IComponent>, controller: &ComPtr<IEditController>) -> Option<Wire> {
    let wired = match (component.cast::<IConnectionPoint>(), controller.cast::<IConnectionPoint>()) {
        (Some(cp), Some(ccp)) if cp.connect(ccp.as_ptr()) == kResultOk && ccp.connect(cp.as_ptr()) == kResultOk => {
            Some((cp, ccp))
        }
        _ => None,
    };
    let state = ComWrapper::new(Stream::default());
    if let Some(s) = state.to_com_ptr::<IBStream>() {
        if component.getState(s.as_ptr()) == kResultOk {
            s.seek(0, IBStream_::IStreamSeekMode_::kIBSeekSet as int32, std::ptr::null_mut());
            controller.setComponentState(s.as_ptr());
        }
    }
    wired
}

/// Undo an [`introduce`], both directions, before either half is terminated.
pub(super) unsafe fn sunder(wire: &Wire) {
    let (cp, ccp) = wire;
    ccp.disconnect(cp.as_ptr());
    cp.disconnect(ccp.as_ptr());
}

unsafe fn arrange(component: &ComPtr<IComponent>, processor: &ComPtr<IAudioProcessor>, class: &Derived) -> (Buses, Buses) {
    let audio = MediaTypes_::kAudio as MediaType;
    let (input, output) = (BusDirections_::kInput as BusDirection, BusDirections_::kOutput as BusDirection);
    let arrangements = |dir: BusDirection, n: usize| -> Vec<SpeakerArrangement> {
        (0..n as int32)
            .map(|i| {
                let mut arrangement = 0;
                processor.getBusArrangement(dir, i, &mut arrangement);
                arrangement
            })
            .collect()
    };
    let (mut ins, mut outs) = (arrangements(input, class.inputs.len()), arrangements(output, class.outputs.len()));
    processor.setBusArrangements(ins.as_mut_ptr(), ins.len() as int32, outs.as_mut_ptr(), outs.len() as int32);
    let widths = |dir: BusDirection, n: usize| -> Vec<u16> {
        (0..n as int32)
            .map(|i| {
                let mut info: BusInfo = std::mem::zeroed();
                component.getBusInfo(audio, dir, i, &mut info);
                info.channelCount.clamp(1, MAX_CHANNELS as i32) as u16
            })
            .collect()
    };
    (Buses::new(&widths(input, ins.len())), Buses::new(&widths(output, outs.len())))
}

/// Activate the audio and event buses — AFTER `setupProcessing` and BEFORE `setActive`, the order
/// Steinberg's own host uses. Activating before the processing setup left some plugins (IK's
/// T-RackS among them) rendering silence, asked to route buses before the block shape was set.
unsafe fn activate_buses(component: &ComPtr<IComponent>, n_in: usize, n_out: usize) {
    let audio = MediaTypes_::kAudio as MediaType;
    let input = BusDirections_::kInput as BusDirection;
    for i in 0..n_in as int32 {
        component.activateBus(audio, input, i, 1);
    }
    for i in 0..n_out as int32 {
        component.activateBus(audio, BusDirections_::kOutput as BusDirection, i, 1);
    }
    let events = MediaTypes_::kEvent as MediaType;
    for i in 0..component.getBusCount(events, input) {
        component.activateBus(events, input, i, 1);
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        unsafe {
            if self.prepared {
                self.processor.setProcessing(0);
                self.component.setActive(0);
            }
            // Undone BEFORE either half is terminated: a component left pointing at a torn-down
            // controller is a use-after-free the plugin performs on itself.
            if let Some(wire) = &self.wired {
                sunder(wire);
            }
            if let Some(c) = &self.controller {
                c.terminate();
            }
            self.component.terminate();
        }
    }
}

/// Staged bus buffers: one block per channel per bus, and the pointer tables a plugin reads.
/// Every access after construction goes through the pointers, so the plugin's writes and ours
/// never race a Rust borrow.
struct Buses {
    _samples: Vec<Vec<[f32; BLOCK]>>,
    pointers: Vec<Vec<*mut f32>>,
    buses: Vec<AudioBusBuffers>,
}

impl Buses {
    fn new(widths: &[u16]) -> Buses {
        let mut samples: Vec<Vec<[f32; BLOCK]>> = widths.iter().map(|&w| vec![[0.0; BLOCK]; w.max(1) as usize]).collect();
        let mut pointers: Vec<Vec<*mut f32>> =
            samples.iter_mut().map(|bus| bus.iter_mut().map(|lane| lane.as_mut_ptr()).collect()).collect();
        let buses = pointers
            .iter_mut()
            .map(|lanes| AudioBusBuffers {
                numChannels: lanes.len() as int32,
                silenceFlags: 0,
                __field0: AudioBusBuffers__type0 { channelBuffers32: lanes.as_mut_ptr() },
            })
            .collect();
        Buses { _samples: samples, pointers, buses }
    }
}

fn note_on(channel: usize, sample: usize, pitch: i16, velocity: f32) -> Event {
    Event {
        busIndex: 0,
        sampleOffset: sample as int32,
        ppqPosition: 0.0,
        flags: 0,
        r#type: Event_::EventTypes_::kNoteOnEvent as u16,
        __field0: Event__type0 { noteOn: NoteOnEvent { channel: channel as i16, pitch, tuning: 0.0, velocity, length: 0, noteId: -1 } },
    }
}

fn note_off(channel: usize, sample: usize, pitch: i16) -> Event {
    Event {
        busIndex: 0,
        sampleOffset: sample as int32,
        ppqPosition: 0.0,
        flags: 0,
        r#type: Event_::EventTypes_::kNoteOffEvent as u16,
        __field0: Event__type0 { noteOff: NoteOffEvent { channel: channel as i16, pitch, velocity: 0.0, noteId: -1, tuning: 0.0 } },
    }
}
