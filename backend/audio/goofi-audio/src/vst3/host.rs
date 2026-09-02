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

/// The application a plugin is initialized against: a name, and no factory of its own.
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

    unsafe fn createInstance(&self, _cid: *mut TUID, _iid: *mut TUID, obj: *mut *mut c_void) -> tresult {
        *obj = std::ptr::null_mut();
        kNotImplemented
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
