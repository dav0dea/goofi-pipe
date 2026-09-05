//! The host's own COM objects: what a plugin is handed, and reads back from. None allocates once
//! built, so a block on the audio thread touches them freely.

use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_void};

use vst3::Steinberg::Vst::*;
use vst3::Steinberg::*;
use vst3::{Class, ComPtr, ComWrapper};

pub fn utf16(s: &[TChar]) -> String {
    let n = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    String::from_utf16_lossy(&s[..n])
}

pub fn cstr(s: &[c_char]) -> String {
    let bytes: Vec<u8> = s.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The application a plugin is initialized against: a name, and the two objects a plugin's halves
/// need in order to speak to each other.
pub struct Host;

impl Class for Host {
    type Interfaces = (IHostApplication,);
}

impl IHostApplicationTrait for Host {
    unsafe fn getName(&self, name: *mut String128) -> tresult {
        for (d, c) in (*name).iter_mut().zip("goofi\0".encode_utf16()) {
            *d = c;
        }
        kResultOk
    }

    /// The host's allocator, and the ONLY one: a connected component and controller talk by asking
    /// here for a message, so a host that answers nothing hands back a null the plugin then
    /// dereferences. Refusing was survivable only while nothing was ever connected — the moment the
    /// two halves were introduced, every iZotope plugin here segfaulted the scanner on the first
    /// `allocateMessage`. JUCE happens to null-check; the VST3 SDK does not require it to.
    unsafe fn createInstance(&self, cid: *mut TUID, _iid: *mut TUID, obj: *mut *mut c_void) -> tresult {
        // Keyed on the CLASS asked for, not the interface: a caller wanting an IMessage names the
        // message class and then queries it for whatever it holds. Compared rather than matched,
        // because these ids are lower-case constants and a pattern arm on one trips
        // `non_upper_case_globals` — which this repo builds as an error.
        let want = *cid;
        let made = if want == IMessage_iid {
            ComWrapper::new(Message::default()).to_com_ptr::<FUnknown>()
        } else if want == IAttributeList_iid {
            ComWrapper::new(Attributes::default()).to_com_ptr::<FUnknown>()
        } else {
            None
        };
        match made {
            Some(p) => {
                // The pointer LEAVES here owning a reference: the plugin releases it when done, and
                // dropping our ComPtr would otherwise free the object before it is ever used.
                *obj = p.into_raw() as *mut c_void;
                kResultOk
            }
            None => {
                *obj = std::ptr::null_mut();
                kNotImplemented
            }
        }
    }
}

/// One attribute's value. A plugin's halves exchange these by name and expect back exactly the kind
/// they put in, so the kind is stored with the value rather than coerced on the way out.
enum Attr {
    Int(int64),
    Float(f64),
    String(Vec<TChar>),
    Binary(Vec<u8>),
}

/// The bag an [`Message`] carries. Keyed by the plugin's own attribute ids, which are plain C
/// strings it invents; goofi never reads one, it only has to hand back what was put in.
#[derive(Default)]
pub struct Attributes {
    map: RefCell<Vec<(String, Attr)>>,
}

impl Attributes {
    fn key(id: IAttrID) -> String {
        if id.is_null() {
            return String::new();
        }
        // SAFETY: an IAttrID is a NUL-terminated C string by the SDK's definition.
        unsafe { std::ffi::CStr::from_ptr(id) }.to_string_lossy().into_owned()
    }

    fn put(&self, id: IAttrID, value: Attr) -> tresult {
        let key = Self::key(id);
        let mut map = self.map.borrow_mut();
        match map.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => map.push((key, value)),
        }
        kResultOk
    }
}

impl Class for Attributes {
    type Interfaces = (IAttributeList,);
}

impl IAttributeListTrait for Attributes {
    unsafe fn setInt(&self, id: IAttrID, value: int64) -> tresult {
        self.put(id, Attr::Int(value))
    }

    unsafe fn getInt(&self, id: IAttrID, value: *mut int64) -> tresult {
        let map = self.map.borrow();
        match map.iter().find(|(k, _)| *k == Self::key(id)) {
            Some((_, Attr::Int(v))) => {
                *value = *v;
                kResultOk
            }
            _ => kResultFalse,
        }
    }

    unsafe fn setFloat(&self, id: IAttrID, value: f64) -> tresult {
        self.put(id, Attr::Float(value))
    }

    unsafe fn getFloat(&self, id: IAttrID, value: *mut f64) -> tresult {
        let map = self.map.borrow();
        match map.iter().find(|(k, _)| *k == Self::key(id)) {
            Some((_, Attr::Float(v))) => {
                *value = *v;
                kResultOk
            }
            _ => kResultFalse,
        }
    }

    unsafe fn setString(&self, id: IAttrID, string: *const TChar) -> tresult {
        let mut out = Vec::new();
        if !string.is_null() {
            let mut p = string;
            while *p != 0 {
                out.push(*p);
                p = p.add(1);
            }
        }
        out.push(0);
        self.put(id, Attr::String(out))
    }

    /// `sizeInBytes` is BYTES of UTF-16, as the SDK spells it — the terminator is always written, so
    /// a buffer too small to hold the value still comes back a valid empty string.
    unsafe fn getString(&self, id: IAttrID, string: *mut TChar, sizeInBytes: uint32) -> tresult {
        let cap = (sizeInBytes as usize) / std::mem::size_of::<TChar>();
        if cap == 0 || string.is_null() {
            return kResultFalse;
        }
        let map = self.map.borrow();
        match map.iter().find(|(k, _)| *k == Self::key(id)) {
            Some((_, Attr::String(v))) => {
                let n = v.len().min(cap);
                std::ptr::copy_nonoverlapping(v.as_ptr(), string, n);
                *string.add(n - 1) = 0;
                kResultOk
            }
            _ => kResultFalse,
        }
    }

    unsafe fn setBinary(&self, id: IAttrID, data: *const c_void, sizeInBytes: uint32) -> tresult {
        let n = sizeInBytes as usize;
        let mut out = vec![0u8; n];
        if n > 0 && !data.is_null() {
            std::ptr::copy_nonoverlapping(data as *const u8, out.as_mut_ptr(), n);
        }
        self.put(id, Attr::Binary(out))
    }

    /// The pointer handed back is INTO the stored blob, which lives as long as this list does — the
    /// SDK's contract, and why the value is kept rather than copied out.
    unsafe fn getBinary(&self, id: IAttrID, data: *mut *const c_void, sizeInBytes: *mut uint32) -> tresult {
        let map = self.map.borrow();
        match map.iter().find(|(k, _)| *k == Self::key(id)) {
            Some((_, Attr::Binary(v))) => {
                *data = v.as_ptr() as *const c_void;
                *sizeInBytes = v.len() as uint32;
                kResultOk
            }
            _ => kResultFalse,
        }
    }
}

/// A message between a plugin's two halves: an id it chooses, and the attributes it fills in. goofi
/// only allocates these — the plugin is both sender and receiver.
#[derive(Default)]
pub struct Message {
    id: RefCell<std::ffi::CString>,
    attributes: RefCell<Option<ComPtr<IAttributeList>>>,
}

impl Class for Message {
    type Interfaces = (IMessage,);
}

impl IMessageTrait for Message {
    unsafe fn getMessageID(&self) -> FIDString {
        self.id.borrow().as_ptr()
    }

    unsafe fn setMessageID(&self, id: FIDString) {
        *self.id.borrow_mut() = match id.is_null() {
            true => std::ffi::CString::default(),
            false => std::ffi::CStr::from_ptr(id).to_owned(),
        };
    }

    /// Built on first ask and kept, because the caller expects the SAME list every time — writing
    /// through one borrow and reading through another is exactly how a message is used.
    unsafe fn getAttributes(&self) -> *mut IAttributeList {
        let mut held = self.attributes.borrow_mut();
        if held.is_none() {
            *held = ComWrapper::new(Attributes::default()).to_com_ptr::<IAttributeList>();
        }
        held.as_ref().map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut())
    }
}

/// A state blob as a stream: what `getState` writes and `setState` reads.
#[derive(Default)]
pub struct Stream {
    pub bytes: RefCell<Vec<u8>>,
    pos: Cell<usize>,
}

impl Stream {
    pub fn of(bytes: &[u8]) -> Stream {
        Stream { bytes: RefCell::new(bytes.to_vec()), pos: Cell::new(0) }
    }
}

impl Class for Stream {
    type Interfaces = (IBStream,);
}

impl IBStreamTrait for Stream {
    unsafe fn read(&self, buffer: *mut c_void, numBytes: int32, numBytesRead: *mut int32) -> tresult {
        let bytes = self.bytes.borrow();
        let pos = self.pos.get().min(bytes.len());
        let n = (numBytes.max(0) as usize).min(bytes.len() - pos);
        std::ptr::copy_nonoverlapping(bytes.as_ptr().add(pos), buffer as *mut u8, n);
        self.pos.set(pos + n);
        if !numBytesRead.is_null() {
            *numBytesRead = n as int32;
        }
        kResultOk
    }

    unsafe fn write(&self, buffer: *mut c_void, numBytes: int32, numBytesWritten: *mut int32) -> tresult {
        let mut bytes = self.bytes.borrow_mut();
        let pos = self.pos.get();
        let n = numBytes.max(0) as usize;
        if bytes.len() < pos + n {
            bytes.resize(pos + n, 0);
        }
        std::ptr::copy_nonoverlapping(buffer as *const u8, bytes.as_mut_ptr().add(pos), n);
        self.pos.set(pos + n);
        if !numBytesWritten.is_null() {
            *numBytesWritten = n as int32;
        }
        kResultOk
    }

    unsafe fn seek(&self, pos: int64, mode: int32, result: *mut int64) -> tresult {
        let len = self.bytes.borrow().len() as i64;
        let base = match mode as IBStream_::IStreamSeekMode {
            IBStream_::IStreamSeekMode_::kIBSeekSet => 0,
            IBStream_::IStreamSeekMode_::kIBSeekCur => self.pos.get() as i64,
            IBStream_::IStreamSeekMode_::kIBSeekEnd => len,
            _ => return kInvalidArgument,
        };
        let at = (base + pos).max(0);
        self.pos.set(at as usize);
        if !result.is_null() {
            *result = at;
        }
        kResultOk
    }

    unsafe fn tell(&self, pos: *mut int64) -> tresult {
        *pos = self.pos.get() as int64;
        kResultOk
    }
}

/// One parameter's change for this block: at most one point, at the first sample.
pub struct Queue {
    id: ParamID,
    value: Cell<Option<ParamValue>>,
}

impl Class for Queue {
    type Interfaces = (IParamValueQueue,);
}

impl IParamValueQueueTrait for Queue {
    unsafe fn getParameterId(&self) -> ParamID {
        self.id
    }

    unsafe fn getPointCount(&self) -> int32 {
        self.value.get().is_some() as int32
    }

    unsafe fn getPoint(&self, index: int32, sampleOffset: *mut int32, value: *mut ParamValue) -> tresult {
        match (index, self.value.get()) {
            (0, Some(v)) => {
                *sampleOffset = 0;
                *value = v;
                kResultOk
            }
            _ => kResultFalse,
        }
    }

    unsafe fn addPoint(&self, _sampleOffset: int32, _value: ParamValue, _index: *mut int32) -> tresult {
        kNotImplemented
    }
}

/// The block's parameter changes: the queues that moved, in order.
pub struct Changes {
    queues: Vec<(ComWrapper<Queue>, ComPtr<IParamValueQueue>)>,
    moved: RefCell<Vec<usize>>,
}

impl Changes {
    pub fn new(ids: impl Iterator<Item = ParamID>) -> Changes {
        let queues: Vec<_> = ids
            .map(|id| {
                let queue = ComWrapper::new(Queue { id, value: Cell::new(None) });
                let ptr = queue.to_com_ptr().expect("a queue is an IParamValueQueue");
                (queue, ptr)
            })
            .collect();
        let moved = RefCell::new(Vec::with_capacity(queues.len()));
        Changes { queues, moved }
    }

    pub fn clear(&self) {
        for (queue, _) in &self.queues {
            queue.value.set(None);
        }
        self.moved.borrow_mut().clear();
    }

    pub fn set(&self, i: usize, value: ParamValue) {
        self.queues[i].0.value.set(Some(value));
        self.moved.borrow_mut().push(i);
    }
}

impl Class for Changes {
    type Interfaces = (IParameterChanges,);
}

impl IParameterChangesTrait for Changes {
    unsafe fn getParameterCount(&self) -> int32 {
        self.moved.borrow().len() as int32
    }

    unsafe fn getParameterData(&self, index: int32) -> *mut IParamValueQueue {
        self.moved.borrow().get(index as usize).map_or(std::ptr::null_mut(), |&i| self.queues[i].1.as_ptr())
    }

    unsafe fn addParameterData(&self, _id: *const ParamID, _index: *mut int32) -> *mut IParamValueQueue {
        std::ptr::null_mut()
    }
}

/// The block's note events, in sample order.
pub struct Events {
    events: RefCell<Vec<Event>>,
}

impl Events {
    pub fn with_capacity(n: usize) -> Events {
        Events { events: RefCell::new(Vec::with_capacity(n)) }
    }

    pub fn clear(&self) {
        self.events.borrow_mut().clear();
    }

    pub fn push(&self, event: Event) {
        let mut events = self.events.borrow_mut();
        if events.len() < events.capacity() {
            events.push(event);
        }
    }
}

impl Class for Events {
    type Interfaces = (IEventList,);
}

impl IEventListTrait for Events {
    unsafe fn getEventCount(&self) -> int32 {
        self.events.borrow().len() as int32
    }

    unsafe fn getEvent(&self, index: int32, e: *mut Event) -> tresult {
        match self.events.borrow().get(index as usize) {
            Some(event) => {
                *e = *event;
                kResultOk
            }
            None => kResultFalse,
        }
    }

    unsafe fn addEvent(&self, _e: *mut Event) -> tresult {
        kNotImplemented
    }
}
