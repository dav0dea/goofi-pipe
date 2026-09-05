//! `GoofiFixture` by `goofi`: one stereo input, one event input, one stereo output, one
//! continuous `Gain` parameter kept in its state. Its output is the input times the gain, plus
//! a sine at every held note's pitch, at half the note's velocity. Its editor draws nothing and
//! turns the gain to a quarter once it is attached — the whole loop a knob in a real one makes.
#![allow(non_snake_case)]

use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_void, CString};
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use vst3::{uid, Class, ComPtr, ComRef, ComWrapper, Steinberg::Vst::*, Steinberg::*};
#[cfg(target_os = "linux")]
use vst3::Steinberg::Linux::IRunLoopTrait;

const NAME: &str = "GoofiFixture";
/// A second audio class behind the same processor, differing only in the subcategories that
/// name it an instrument — which is what the palette reads to tag a plugin.
const SYNTH_NAME: &str = "GoofiSynth";
const SYNTH_CID: TUID = uid(0x2B3C4D5E, 0x6F708192, 0xA3B4C5D6, 0xE7F80919);
const VOICES: usize = 16;

fn copy_cstring(src: &str, dst: &mut [c_char]) {
    let c_string = CString::new(src).unwrap_or_default();
    let bytes = c_string.as_bytes_with_nul();
    for (src, dst) in bytes.iter().zip(dst.iter_mut()) {
        *dst = *src as c_char;
    }
    if let Some(last) = dst.last_mut() {
        *last = 0;
    }
}

fn copy_wstring(src: &str, dst: &mut [TChar]) {
    let mut len = 0;
    for (src, dst) in src.encode_utf16().zip(dst.iter_mut()) {
        *dst = src;
        len += 1;
    }
    dst[len.min(dst.len() - 1)] = 0;
}

#[derive(Clone, Copy, Default)]
struct Voice {
    note: Option<i16>,
    velocity: f32,
    phase: f32,
}

struct Processor {
    gain: AtomicU64,
    rate: AtomicU64,
    voices: RefCell<[Voice; VOICES]>,
    /// State beyond the params: latched the first time `Shape` reaches its last step, and carried
    /// ONLY by the state blob. It halves the note tone, so a load that lost the blob is heard.
    pushed: AtomicBool,
}

impl Class for Processor {
    type Interfaces = (IComponent, IAudioProcessor, IConnectionPoint);
}

// The processor accepts the connection; it is the controller that gates on it.
impl IConnectionPointTrait for Processor {
    unsafe fn connect(&self, _other: *mut IConnectionPoint) -> tresult {
        kResultOk
    }
    unsafe fn disconnect(&self, _other: *mut IConnectionPoint) -> tresult {
        kResultOk
    }
    unsafe fn notify(&self, _message: *mut IMessage) -> tresult {
        kResultOk
    }
}

impl Processor {
    const CID: TUID = uid(0x6F0F1A2B, 0x3C4D5E6F, 0x70819293, 0xA4B5C6D7);

    fn new() -> Processor {
        Processor {
            gain: AtomicU64::new(1.0f64.to_bits()),
            rate: AtomicU64::new(48_000.0f64.to_bits()),
            voices: RefCell::new([Voice::default(); VOICES]),
            pushed: AtomicBool::new(false),
        }
    }
}

impl IPluginBaseTrait for Processor {
    unsafe fn initialize(&self, _context: *mut FUnknown) -> tresult {
        kResultOk
    }

    unsafe fn terminate(&self) -> tresult {
        kResultOk
    }
}

impl IComponentTrait for Processor {
    unsafe fn getControllerClassId(&self, class_id: *mut TUID) -> tresult {
        *class_id = Controller::CID;
        kResultOk
    }

    unsafe fn setIoMode(&self, _mode: IoMode) -> tresult {
        kResultOk
    }

    unsafe fn getBusCount(&self, media_type: MediaType, dir: BusDirection) -> i32 {
        match (media_type as MediaTypes, dir as BusDirections) {
            (MediaTypes_::kAudio, _) => 1,
            (MediaTypes_::kEvent, BusDirections_::kInput) => 1,
            _ => 0,
        }
    }

    unsafe fn getBusInfo(&self, media_type: MediaType, dir: BusDirection, index: i32, bus: *mut BusInfo) -> tresult {
        if index != 0 || self.getBusCount(media_type, dir) == 0 {
            return kInvalidArgument;
        }
        let bus = &mut *bus;
        bus.mediaType = media_type;
        bus.direction = dir;
        let audio = media_type as MediaTypes == MediaTypes_::kAudio;
        bus.channelCount = if audio { 2 } else { VOICES as i32 };
        let name = match (audio, dir as BusDirections) {
            (true, BusDirections_::kInput) => "Input",
            (true, _) => "Output",
            _ => "Events",
        };
        copy_wstring(name, &mut bus.name);
        bus.busType = BusTypes_::kMain as BusType;
        bus.flags = BusInfo_::BusFlags_::kDefaultActive as u32;
        kResultOk
    }

    unsafe fn getRoutingInfo(&self, _in_info: *mut RoutingInfo, _out_info: *mut RoutingInfo) -> tresult {
        kNotImplemented
    }

    unsafe fn activateBus(&self, _media_type: MediaType, _dir: BusDirection, _index: i32, _state: TBool) -> tresult {
        kResultOk
    }

    unsafe fn setActive(&self, _state: TBool) -> tresult {
        kResultOk
    }

    unsafe fn setState(&self, state: *mut IBStream) -> tresult {
        let Some(state) = ComRef::from_raw(state) else { return kInvalidArgument };
        let mut bytes = [0u8; 9];
        let mut read = 0;
        if state.read(bytes.as_mut_ptr() as *mut c_void, 9, &mut read) == kResultOk && read == 9 {
            let gain: [u8; 8] = bytes[..8].try_into().expect("eight of nine");
            self.gain.store(f64::from_le_bytes(gain).to_bits(), Ordering::Relaxed);
            self.pushed.store(bytes[8] != 0, Ordering::Relaxed);
            return kResultOk;
        }
        kResultFalse
    }

    unsafe fn getState(&self, state: *mut IBStream) -> tresult {
        let Some(state) = ComRef::from_raw(state) else { return kInvalidArgument };
        let mut bytes = [0u8; 9];
        bytes[..8].copy_from_slice(&f64::from_bits(self.gain.load(Ordering::Relaxed)).to_le_bytes());
        bytes[8] = self.pushed.load(Ordering::Relaxed) as u8;
        let mut written = 0;
        state.write(bytes.as_ptr() as *mut c_void, 9, &mut written);
        kResultOk
    }
}

impl IAudioProcessorTrait for Processor {
    unsafe fn setBusArrangements(&self, inputs: *mut SpeakerArrangement, num_ins: i32, outputs: *mut SpeakerArrangement, num_outs: i32) -> tresult {
        if num_ins != 1 || num_outs != 1 || *inputs != SpeakerArr::kStereo || *outputs != SpeakerArr::kStereo {
            return kResultFalse;
        }
        kResultTrue
    }

    unsafe fn getBusArrangement(&self, _dir: BusDirection, index: i32, arr: *mut SpeakerArrangement) -> tresult {
        if index != 0 {
            return kInvalidArgument;
        }
        *arr = SpeakerArr::kStereo;
        kResultOk
    }

    unsafe fn canProcessSampleSize(&self, symbolic_sample_size: i32) -> tresult {
        match symbolic_sample_size as SymbolicSampleSizes {
            SymbolicSampleSizes_::kSample32 => kResultOk,
            SymbolicSampleSizes_::kSample64 => kNotImplemented,
            _ => kInvalidArgument,
        }
    }

    unsafe fn getLatencySamples(&self) -> u32 {
        0
    }

    unsafe fn setupProcessing(&self, setup: *mut ProcessSetup) -> tresult {
        self.rate.store((*setup).sampleRate.to_bits(), Ordering::Relaxed);
        kResultOk
    }

    unsafe fn setProcessing(&self, _state: TBool) -> tresult {
        kResultOk
    }

    unsafe fn process(&self, data: *mut ProcessData) -> tresult {
        let data = &*data;
        if let Some(changes) = ComRef::from_raw(data.inputParameterChanges) {
            for i in 0..changes.getParameterCount() {
                let Some(queue) = ComRef::from_raw(changes.getParameterData(i)) else { continue };
                let points = queue.getPointCount();
                let (mut offset, mut value) = (0, 0.0);
                if points > 0 && queue.getPoint(points - 1, &mut offset, &mut value) == kResultOk {
                    match queue.getParameterId() {
                        0 => self.gain.store(value.to_bits(), Ordering::Relaxed),
                        1 if value >= 0.99 => self.pushed.store(true, Ordering::Relaxed),
                        _ => {}
                    }
                }
            }
        }
        let mut voices = self.voices.borrow_mut();
        if let Some(events) = ComRef::from_raw(data.inputEvents) {
            for i in 0..events.getEventCount() {
                let mut event: Event = std::mem::zeroed();
                if events.getEvent(i, &mut event) != kResultOk {
                    continue;
                }
                match event.r#type as Event_::EventTypes {
                    Event_::EventTypes_::kNoteOnEvent => {
                        let on = event.__field0.noteOn;
                        let voice = &mut voices[on.channel.clamp(0, VOICES as i16 - 1) as usize];
                        *voice = Voice { note: Some(on.pitch), velocity: on.velocity, phase: 0.0 };
                    }
                    Event_::EventTypes_::kNoteOffEvent => {
                        let off = event.__field0.noteOff;
                        let voice = &mut voices[off.channel.clamp(0, VOICES as i16 - 1) as usize];
                        if voice.note == Some(off.pitch) {
                            voice.note = None;
                        }
                    }
                    _ => {}
                }
            }
        }
        let gain = f64::from_bits(self.gain.load(Ordering::Relaxed)) as f32;
        let rate = f64::from_bits(self.rate.load(Ordering::Relaxed)) as f32;
        let loud = if self.pushed.load(Ordering::Relaxed) { 0.25 } else { 0.5 };
        let n = data.numSamples as usize;
        if data.numOutputs != 1 {
            return kResultOk;
        }
        let out_bus = &*data.outputs;
        if out_bus.numChannels != 2 {
            return kResultOk;
        }
        let out_lanes = slice::from_raw_parts(out_bus.__field0.channelBuffers32, 2);
        let (out_l, out_r) = (slice::from_raw_parts_mut(out_lanes[0], n), slice::from_raw_parts_mut(out_lanes[1], n));
        let input = (data.numInputs == 1 && (*data.inputs).numChannels == 2).then(|| {
            let lanes = slice::from_raw_parts((*data.inputs).__field0.channelBuffers32, 2);
            (slice::from_raw_parts(lanes[0], n), slice::from_raw_parts(lanes[1], n))
        });
        for i in 0..n {
            let (l, r) = input.map_or((0.0, 0.0), |(l, r)| (l[i], r[i]));
            let mut tone = 0.0f32;
            for voice in voices.iter_mut() {
                if let Some(note) = voice.note {
                    let hz = 440.0 * 2f32.powf((note as f32 - 69.0) / 12.0);
                    tone += loud * voice.velocity * (voice.phase * std::f32::consts::TAU).sin();
                    voice.phase = (voice.phase + hz / rate) % 1.0;
                }
            }
            out_l[i] = gain * l + tone;
            out_r[i] = gain * r + tone;
        }
        kResultOk
    }

    unsafe fn getTailSamples(&self) -> u32 {
        0
    }
}

struct Controller {
    gain: Cell<f64>,
    /// The host, kept from `initialize` so the connection can ask it for a message.
    host: RefCell<Option<ComPtr<IHostApplication>>>,
    /// What a view edits through, kept from `setComponentHandler`.
    handler: RefCell<Option<ComPtr<IComponentHandler>>>,
    /// True once connected AND a host message was allocated — until then this publishes nothing,
    /// which is the exact bug the goofi host repairs.
    ready: Cell<bool>,
}

impl Class for Controller {
    type Interfaces = (IEditController, IConnectionPoint);
}

impl IConnectionPointTrait for Controller {
    /// Introduced to the processor: allocate a message from the host, and only then publish params.
    /// A host that connects but cannot allocate a message leaves this controller mute.
    unsafe fn connect(&self, _other: *mut IConnectionPoint) -> tresult {
        let Some(host) = self.host.borrow().clone() else { return kResultOk };
        let mut obj: *mut c_void = std::ptr::null_mut();
        let rc = host.createInstance(&IMessage_iid as *const _ as *mut _, &IMessage_iid as *const _ as *mut _, &mut obj);
        if rc == kResultOk && !obj.is_null() {
            if let Some(msg) = ComPtr::<FUnknown>::from_raw(obj as *mut FUnknown) {
                drop(msg);
            }
            self.ready.set(true);
        }
        kResultOk
    }
    unsafe fn disconnect(&self, _other: *mut IConnectionPoint) -> tresult {
        self.ready.set(false);
        kResultOk
    }
    unsafe fn notify(&self, _message: *mut IMessage) -> tresult {
        kResultOk
    }
}

impl Controller {
    const CID: TUID = uid(0x1E2F3A4B, 0x5C6D7E8F, 0x90A1B2C3, 0xD4E5F607);
}

impl IPluginBaseTrait for Controller {
    unsafe fn initialize(&self, context: *mut FUnknown) -> tresult {
        if let Some(ctx) = ComRef::from_raw(context) {
            *self.host.borrow_mut() = ctx.cast::<IHostApplication>();
        }
        kResultOk
    }

    unsafe fn terminate(&self) -> tresult {
        kResultOk
    }
}

impl IEditControllerTrait for Controller {
    unsafe fn setComponentState(&self, state: *mut IBStream) -> tresult {
        let Some(state) = ComRef::from_raw(state) else { return kInvalidArgument };
        let mut bytes = [0u8; 8];
        let mut read = 0;
        if state.read(bytes.as_mut_ptr() as *mut c_void, 8, &mut read) == kResultOk && read == 8 {
            self.gain.set(f64::from_le_bytes(bytes));
        }
        kResultOk
    }

    unsafe fn setState(&self, _state: *mut IBStream) -> tresult {
        kResultOk
    }

    unsafe fn getState(&self, _state: *mut IBStream) -> tresult {
        kResultOk
    }

    unsafe fn getParameterCount(&self) -> i32 {
        // Nothing until the host has introduced the two halves — the failure this pins.
        if self.ready.get() { 4 } else { 0 }
    }

    /// One of each shape the manifest derivation has a rule for: continuous, stepped within the
    /// `Str` ceiling, stepped past it, and one the host must omit.
    unsafe fn getParameterInfo(&self, param_index: i32, info: *mut ParameterInfo) -> tresult {
        let automate = ParameterInfo_::ParameterFlags_::kCanAutomate as i32;
        let (title, units, steps, default, flags) = match param_index {
            0 => ("Gain", "x", 0, 1.0, automate),
            1 => ("Shape", "", 2, 0.0, automate),
            2 => ("Steps", "", 200, 0.5, automate),
            3 => ("Meter", "", 0, 0.0, automate | ParameterInfo_::ParameterFlags_::kIsReadOnly as i32),
            _ => return kInvalidArgument,
        };
        let info = &mut *info;
        info.id = param_index as u32;
        copy_wstring(title, &mut info.title);
        copy_wstring(title, &mut info.shortTitle);
        copy_wstring(units, &mut info.units);
        info.stepCount = steps;
        info.defaultNormalizedValue = default;
        info.unitId = 0;
        info.flags = flags;
        kResultOk
    }

    unsafe fn getParamStringByValue(&self, id: u32, value_normalized: f64, string: *mut String128) -> tresult {
        let shown = match id {
            0 | 2 | 3 => format!("{value_normalized:.2}"),
            1 => ["soft", "mid", "hard"][(value_normalized * 2.0).round() as usize].to_string(),
            _ => return kInvalidArgument,
        };
        copy_wstring(&shown, &mut *string);
        kResultOk
    }

    unsafe fn getParamValueByString(&self, _id: u32, _string: *mut TChar, _value_normalized: *mut f64) -> tresult {
        kNotImplemented
    }

    unsafe fn normalizedParamToPlain(&self, _id: u32, value_normalized: f64) -> f64 {
        value_normalized
    }

    unsafe fn plainParamToNormalized(&self, _id: u32, plain_value: f64) -> f64 {
        plain_value
    }

    unsafe fn getParamNormalized(&self, _id: u32) -> f64 {
        self.gain.get()
    }

    unsafe fn setParamNormalized(&self, _id: u32, value: f64) -> tresult {
        self.gain.set(value);
        kResultOk
    }

    unsafe fn setComponentHandler(&self, handler: *mut IComponentHandler) -> tresult {
        *self.handler.borrow_mut() = ComRef::from_raw(handler).and_then(|h| h.cast::<IComponentHandler>());
        kResultOk
    }

    unsafe fn createView(&self, name: *const c_char) -> *mut IPlugView {
        let editor = !name.is_null() && std::ffi::CStr::from_ptr(name).to_str() == Ok("editor");
        let Some(handler) = self.handler.borrow().clone().filter(|_| editor) else { return std::ptr::null_mut() };
        let view = View {
            #[cfg(target_os = "linux")]
            turn: ComWrapper::new(Turn { handler: handler.clone(), fired: Cell::new(false) })
                .to_com_ptr::<Linux::ITimerHandler>()
                .expect("a turn is a timer handler"),
            #[cfg(target_os = "linux")]
            run_loop: RefCell::new(None),
            handler,
            frame: RefCell::new(None),
        };
        ComWrapper::new(view).to_com_ptr::<IPlugView>().expect("a view is a plug view").into_raw()
    }
}

/// The one gesture: the gain knob, turned to a quarter.
unsafe fn turn(handler: &ComPtr<IComponentHandler>) {
    handler.beginEdit(0);
    handler.performEdit(0, 0.25);
    handler.endEdit(0);
}

/// The editor: attached, it turns the knob — through the host's run loop on Linux, where a real
/// editor's own events and timers ride, and straight away where the platform pumps them itself.
struct View {
    handler: ComPtr<IComponentHandler>,
    frame: RefCell<Option<ComPtr<IPlugFrame>>>,
    #[cfg(target_os = "linux")]
    turn: ComPtr<Linux::ITimerHandler>,
    #[cfg(target_os = "linux")]
    run_loop: RefCell<Option<ComPtr<Linux::IRunLoop>>>,
}

impl Class for View {
    type Interfaces = (IPlugView,);
}

const PLATFORM: &str = if cfg!(windows) { "HWND" } else if cfg!(target_os = "macos") { "NSView" } else { "X11EmbedWindowID" };

impl IPlugViewTrait for View {
    unsafe fn isPlatformTypeSupported(&self, type_: FIDString) -> tresult {
        match !type_.is_null() && std::ffi::CStr::from_ptr(type_).to_str() == Ok(PLATFORM) {
            true => kResultTrue,
            false => kResultFalse,
        }
    }

    unsafe fn attached(&self, _parent: *mut c_void, _type: FIDString) -> tresult {
        #[cfg(target_os = "linux")]
        {
            let Some(run_loop) = self.frame.borrow().as_ref().and_then(|f| f.cast::<Linux::IRunLoop>()) else { return kResultFalse };
            run_loop.registerTimer(self.turn.as_ptr(), 10);
            *self.run_loop.borrow_mut() = Some(run_loop);
        }
        #[cfg(not(target_os = "linux"))]
        turn(&self.handler);
        kResultOk
    }

    unsafe fn removed(&self) -> tresult {
        #[cfg(target_os = "linux")]
        if let Some(run_loop) = self.run_loop.borrow_mut().take() {
            run_loop.unregisterTimer(self.turn.as_ptr());
        }
        kResultOk
    }

    unsafe fn onWheel(&self, _distance: f32) -> tresult {
        kResultFalse
    }

    unsafe fn onKeyDown(&self, _key: char16, _key_code: int16, _modifiers: int16) -> tresult {
        kResultFalse
    }

    unsafe fn onKeyUp(&self, _key: char16, _key_code: int16, _modifiers: int16) -> tresult {
        kResultFalse
    }

    unsafe fn getSize(&self, size: *mut ViewRect) -> tresult {
        *size = ViewRect { left: 0, top: 0, right: 320, bottom: 200 };
        kResultOk
    }

    unsafe fn onSize(&self, _new_size: *mut ViewRect) -> tresult {
        kResultOk
    }

    unsafe fn onFocus(&self, _state: TBool) -> tresult {
        kResultOk
    }

    unsafe fn setFrame(&self, frame: *mut IPlugFrame) -> tresult {
        *self.frame.borrow_mut() = ComRef::from_raw(frame).and_then(|f| f.cast::<IPlugFrame>());
        kResultOk
    }

    unsafe fn canResize(&self) -> tresult {
        kResultFalse
    }

    unsafe fn checkSizeConstraint(&self, _rect: *mut ViewRect) -> tresult {
        kResultTrue
    }
}

/// The timer the view registers with the host's run loop: it turns the knob once and then only ticks.
#[cfg(target_os = "linux")]
struct Turn {
    handler: ComPtr<IComponentHandler>,
    fired: Cell<bool>,
}

#[cfg(target_os = "linux")]
impl Class for Turn {
    type Interfaces = (Linux::ITimerHandler,);
}

#[cfg(target_os = "linux")]
impl Linux::ITimerHandlerTrait for Turn {
    unsafe fn onTimer(&self) {
        if !self.fired.replace(true) {
            turn(&self.handler);
        }
    }
}

struct Factory;

impl Class for Factory {
    type Interfaces = (IPluginFactory, IPluginFactory2);
}

/// One row per class the factory offers: cid, class category, name, VST3 subcategories.
/// Both `getClassInfo` and `getClassInfo2` read it, so the two answers cannot disagree.
const CLASSES: [(TUID, &str, &str, &str); 3] = [
    (Processor::CID, "Audio Module Class", NAME, "Fx"),
    (Controller::CID, "Component Controller Class", NAME, ""),
    (SYNTH_CID, "Audio Module Class", SYNTH_NAME, "Instrument|Synth"),
];

impl IPluginFactoryTrait for Factory {
    unsafe fn getFactoryInfo(&self, info: *mut PFactoryInfo) -> tresult {
        let info = &mut *info;
        copy_cstring("goofi", &mut info.vendor);
        copy_cstring("https://github.com/PhilippThoelke/goofi-pipe", &mut info.url);
        info.flags = PFactoryInfo_::FactoryFlags_::kUnicode as int32;
        kResultOk
    }

    unsafe fn countClasses(&self) -> i32 {
        CLASSES.len() as i32
    }

    unsafe fn getClassInfo(&self, index: i32, info: *mut PClassInfo) -> tresult {
        let Some(&(cid, category, name, _)) = CLASSES.get(index as usize) else {
            return kInvalidArgument;
        };
        let info = &mut *info;
        info.cid = cid;
        info.cardinality = PClassInfo_::ClassCardinality_::kManyInstances as int32;
        copy_cstring(category, &mut info.category);
        copy_cstring(name, &mut info.name);
        kResultOk
    }

    unsafe fn createInstance(&self, cid: FIDString, iid: FIDString, obj: *mut *mut c_void) -> tresult {
        let instance = match *(cid as *const TUID) {
            Processor::CID | SYNTH_CID => ComWrapper::new(Processor::new()).to_com_ptr::<FUnknown>(),
            Controller::CID => ComWrapper::new(Controller { gain: Cell::new(1.0), host: RefCell::new(None), handler: RefCell::new(None), ready: Cell::new(false) }).to_com_ptr::<FUnknown>(),
            _ => None,
        };
        match instance {
            Some(instance) => {
                let ptr = instance.as_ptr();
                ((*(*ptr).vtbl).queryInterface)(ptr, iid as *mut TUID, obj)
            }
            None => kInvalidArgument,
        }
    }
}

impl IPluginFactory2Trait for Factory {
    unsafe fn getClassInfo2(&self, index: i32, info: *mut PClassInfo2) -> tresult {
        let Some(&(cid, category, name, sub_categories)) = CLASSES.get(index as usize) else {
            return kInvalidArgument;
        };
        let info = &mut *info;
        info.cid = cid;
        info.cardinality = PClassInfo_::ClassCardinality_::kManyInstances as int32;
        info.classFlags = 0;
        copy_cstring(category, &mut info.category);
        copy_cstring(name, &mut info.name);
        copy_cstring(sub_categories, &mut info.subCategories);
        copy_cstring("goofi", &mut info.vendor);
        copy_cstring("1.0.0", &mut info.version);
        copy_cstring("VST 3.7.0", &mut info.sdkVersion);
        kResultOk
    }
}

#[cfg(target_os = "windows")]
#[no_mangle]
extern "system" fn InitDll() -> bool {
    true
}

#[cfg(target_os = "macos")]
#[no_mangle]
extern "system" fn bundleEntry(_bundle_ref: *mut c_void) -> bool {
    true
}

#[cfg(target_os = "linux")]
#[no_mangle]
extern "system" fn ModuleEntry(_library_handle: *mut c_void) -> bool {
    true
}

#[no_mangle]
extern "system" fn GetPluginFactory() -> *mut IPluginFactory {
    #[cfg(feature = "crash")]
    std::process::abort();
    #[cfg(not(feature = "crash"))]
    ComWrapper::new(Factory).to_com_ptr::<IPluginFactory>().unwrap().into_raw()
}
