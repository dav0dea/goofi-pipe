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
use super::module::Module;
use super::ok;

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
    /// An event input, so the first three params are the voice: gate, pitch, velocity.
    pub voice: bool,
    pub params: Vec<(ParamID, Kind)>,
}

pub struct Plugin {
    class: Arc<Derived>,
    live: Option<Live>,
    /// What instantiation refused; the next `process` raises it as the node's own panic.
    failed: Option<String>,
}

impl Plugin {
    pub fn new(class: Arc<Derived>) -> Plugin {
        Plugin { class, live: None, failed: None }
    }
}

impl AudioNode for Plugin {
    fn channels(&self, _ins: &[u16], _params: &[f64], outs: usize) -> Vec<u16> {
        (0..outs).map(|i| self.class.outputs.get(i).copied().unwrap_or(1).clamp(1, MAX_CHANNELS)).collect()
    }

    fn prepare(&mut self, rate: f64) {
        let result = if let Some(live) = self.live.as_mut() {
            live.retune(rate)
        } else {
            Live::open(&self.class, rate).map(|live| self.live = Some(live))
        };
        if let Err(e) = result {
            self.failed = Some(e);
        }
    }

    fn process(&mut self, b: &mut Block<'_>) {
        if let Some(text) = self.failed.take() {
            std::panic::resume_unwind(Box::new(text));
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
        let Some(live) = &self.live else { return Vec::new() };
        let stream = ComWrapper::new(Stream::default());
        let ptr = stream.to_com_ptr::<IBStream>().expect("a stream is an IBStream");
        if unsafe { live.component.getState(ptr.as_ptr()) } != kResultOk {
            return Vec::new();
        }
        stream.bytes.take()
    }

    fn load(&mut self, bytes: &[u8]) {
        let Some(live) = self.live.as_mut() else { return };
        let stream = ComWrapper::new(Stream::of(bytes));
        let ptr = stream.to_com_ptr::<IBStream>().expect("a stream is an IBStream");
        unsafe { live.component.setState(ptr.as_ptr()) };
        live.sent.fill(f64::NAN);
    }
}

struct Live {
    _context: ComPtr<FUnknown>,
    component: ComPtr<IComponent>,
    processor: ComPtr<IAudioProcessor>,
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
}

// The one deviation from "no unsafe impl": the VST3 contract lets `process` run on a thread of
// the host's choosing once `setProcessing` was called, and these pointers cross exactly once.
unsafe impl Send for Live {}

impl Live {
    fn open(class: &Derived, rate: f64) -> Result<Live, String> {
        let factory = Module::open(&class.binary)?.factory()?;
        let component: ComPtr<IComponent> = factory.create(&class.cid)?;
        let context = ComWrapper::new(Host).to_com_ptr::<FUnknown>().expect("a host is an FUnknown");
        unsafe { ok(component.initialize(context.as_ptr()), "initialize")? };
        let processor: ComPtr<IAudioProcessor> = component.cast().ok_or("the component is no IAudioProcessor")?;
        unsafe {
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
            for i in 0..ins.len() as int32 {
                component.activateBus(audio, input, i, 1);
            }
            for i in 0..outs.len() as int32 {
                component.activateBus(audio, output, i, 1);
            }
            if class.voice {
                component.activateBus(MediaTypes_::kEvent as MediaType, input, 0, 1);
            }
        }
        let changes = ComWrapper::new(Changes::new(class.params.iter().map(|(id, _)| *id)));
        let changes_ptr = changes.to_com_ptr().expect("changes are an IParameterChanges");
        let events = ComWrapper::new(Events::with_capacity(BLOCK * MAX_CHANNELS as usize));
        let events_ptr = events.to_com_ptr().expect("events are an IEventList");
        let mut live = Live {
            _context: context,
            component,
            processor,
            changes,
            changes_ptr,
            events,
            events_ptr,
            context: Box::new(unsafe { std::mem::zeroed() }),
            ins: Buses::new(&class.inputs),
            outs: Buses::new(&class.outputs),
            sent: vec![f64::NAN; class.params.len()],
            held: [None; MAX_CHANNELS as usize],
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
            self.processor.setProcessing(0);
            self.component.setActive(0);
            ok(self.processor.setupProcessing(&mut setup), "setupProcessing")?;
            ok(self.component.setActive(1), "setActive")?;
            ok(self.processor.setProcessing(1), "setProcessing")?;
        }
        self.context.sampleRate = rate;
        self.sent.fill(f64::NAN);
        Ok(())
    }

    fn block(&mut self, class: &Derived, b: &mut Block<'_>) {
        let voice = class.voice as usize * 3;
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
        if class.voice {
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
        for (i, port) in b.ins.iter().enumerate() {
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
        for (i, out) in b.outs.iter_mut().enumerate() {
            let lanes = &self.outs.pointers[i];
            for c in 0..out.channels() as usize {
                let src = lanes[c.min(lanes.len() - 1)];
                unsafe { std::ptr::copy_nonoverlapping(src, out.chan_mut(c).as_mut_ptr(), BLOCK) };
            }
        }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        unsafe {
            self.processor.setProcessing(0);
            self.component.setActive(0);
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
