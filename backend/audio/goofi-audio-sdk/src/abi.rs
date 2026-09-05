//! The C boundary a built audio node crosses: one vtable of `extern "C"` entries over §3's block
//! as plain descriptors — the arena's own regions, no bytes and no codec — the shim that puts an
//! author's [`AudioNode`] behind it, and the two macros a node file and its generated crate spell.
//! Only code and plain data cross; never a Rust type, and never a panic.

use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{AudioNode, Block, Manifest, Port, PortMut, BLOCK, MAX_PORTS};

/// One input or param as the boundary spells it: `channels` planar blocks at `data`.
#[repr(C)]
pub struct PortDesc {
    pub data: *const f32,
    pub channels: u16,
    pub wired: bool,
}

impl PortDesc {
    pub fn of(p: &Port<'_>) -> PortDesc {
        PortDesc { data: p.data.as_ptr(), channels: p.channels, wired: p.wired }
    }
}

/// One output as the boundary spells it.
#[repr(C)]
pub struct OutDesc {
    pub data: *mut f32,
    pub channels: u16,
}

impl OutDesc {
    pub fn of(p: &mut PortMut<'_>) -> OutDesc {
        OutDesc { data: p.data.as_mut_ptr(), channels: p.channels }
    }
}

/// One block as it crosses: three arrays of descriptors, in declaration order.
#[repr(C)]
pub struct BlockDesc {
    pub ins: *const PortDesc,
    pub n_ins: usize,
    pub outs: *const OutDesc,
    pub n_outs: usize,
    pub params: *const PortDesc,
    pub n_params: usize,
}

pub use goofi_node::abi::{collect, version, Bytes, Write};

/// Every entry answers whether the node came through it without a panic; on `false` the panic's
/// own words are in the sink.
#[repr(C)]
pub struct VTable {
    pub create: unsafe extern "C" fn() -> *mut c_void,
    pub destroy: unsafe extern "C" fn(node: *mut c_void),
    pub channels: unsafe extern "C" fn(
        node: *mut c_void,
        ins: *const u16,
        n_ins: usize,
        params: *const f64,
        n_params: usize,
        outs: *mut u16,
        n_outs: usize,
        sink: *mut c_void,
        write: Write,
    ) -> bool,
    pub prepare: unsafe extern "C" fn(node: *mut c_void, rate: f64, sink: *mut c_void, write: Write) -> bool,
    pub process: unsafe extern "C" fn(node: *mut c_void, block: *const BlockDesc, sink: *mut c_void, write: Write) -> bool,
    pub feedback: unsafe extern "C" fn(node: *mut c_void, answer: *mut bool, sink: *mut c_void, write: Write) -> bool,
    /// The bytes go to the sink on `true`.
    pub save: unsafe extern "C" fn(node: *mut c_void, sink: *mut c_void, write: Write) -> bool,
    pub load: unsafe extern "C" fn(node: *mut c_void, ptr: *const u8, len: usize, sink: *mut c_void, write: Write) -> bool,
}

/// The `goofi_describe` answer: the manifest as the probe schema.
pub fn describe_c(m: &Manifest) -> *const c_char {
    goofi_node::abi::describe_once(|| goofi_node::describe(m.tags, m.doc, m.inputs, m.outputs, m.params, false))
}

/// Box a fresh node for the host; a constructor that panics answers null, which the host faults
/// at the first block rather than unwinding across the boundary.
pub fn instance(create: impl FnOnce() -> Box<dyn AudioNode>) -> *mut c_void {
    match catch_unwind(AssertUnwindSafe(create)) {
        Ok(node) => Box::into_raw(Box::new(node)) as *mut c_void,
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `node` came from [`instance`] and is not used after this.
pub unsafe extern "C" fn destroy(node: *mut c_void) {
    if !node.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(node as *mut Box<dyn AudioNode>))));
    }
}

unsafe fn with(node: *mut c_void, sink: *mut c_void, write: Write, f: impl FnOnce(&mut dyn AudioNode)) -> bool {
    let node = &mut **(node as *mut Box<dyn AudioNode>);
    match catch_unwind(AssertUnwindSafe(|| f(node))) {
        Ok(()) => true,
        Err(p) => {
            let text = goofi_node::panic_text(p);
            write(sink, Bytes::of(text.as_bytes()));
            false
        }
    }
}

/// # Safety
/// `node` came from [`instance`]; every pointer addresses its stated count.
pub unsafe extern "C" fn channels(
    node: *mut c_void,
    ins: *const u16,
    n_ins: usize,
    params: *const f64,
    n_params: usize,
    outs: *mut u16,
    n_outs: usize,
    sink: *mut c_void,
    write: Write,
) -> bool {
    let ins = slice(ins, n_ins);
    let params = slice(params, n_params);
    with(node, sink, write, |n| {
        let wanted = n.channels(ins, params, n_outs);
        for (i, w) in wanted.iter().enumerate().take(n_outs) {
            *outs.add(i) = *w;
        }
    })
}

/// # Safety
/// As [`channels`].
pub unsafe extern "C" fn prepare(node: *mut c_void, rate: f64, sink: *mut c_void, write: Write) -> bool {
    with(node, sink, write, |n| n.prepare(rate))
}

/// # Safety
/// As [`channels`]; every descriptor addresses `channels * BLOCK` floats that outlive the call.
pub unsafe extern "C" fn process(node: *mut c_void, block: *const BlockDesc, sink: *mut c_void, write: Write) -> bool {
    let d = &*block;
    let port = |p: &PortDesc| Port::new(slice(p.data, p.channels as usize * BLOCK), p.channels, p.wired);
    let ins: [Port<'_>; MAX_PORTS] = std::array::from_fn(|i| match i < d.n_ins {
        true => port(&*d.ins.add(i)),
        false => Port::new(&[], 0, false),
    });
    let params: [Port<'_>; MAX_PORTS] = std::array::from_fn(|i| match i < d.n_params {
        true => port(&*d.params.add(i)),
        false => Port::new(&[], 0, false),
    });
    let mut outs: [PortMut<'_>; MAX_PORTS] = std::array::from_fn(|i| match i < d.n_outs {
        true => {
            let o = &*d.outs.add(i);
            PortMut::new(slice_mut(o.data, o.channels as usize * BLOCK), o.channels)
        }
        false => PortMut::new(&mut [], 0),
    });
    let mut block = Block { ins: &ins[..d.n_ins], outs: &mut outs[..d.n_outs], params: &params[..d.n_params] };
    with(node, sink, write, |n| n.process(&mut block))
}

/// # Safety
/// As [`channels`]; `answer` is writable.
pub unsafe extern "C" fn feedback(node: *mut c_void, answer: *mut bool, sink: *mut c_void, write: Write) -> bool {
    with(node, sink, write, |n| *answer = n.feedback())
}

/// # Safety
/// As [`channels`].
pub unsafe extern "C" fn save(node: *mut c_void, sink: *mut c_void, write: Write) -> bool {
    with(node, sink, write, |n| {
        let bytes = n.save();
        write(sink, Bytes::of(&bytes));
    })
}

/// # Safety
/// As [`channels`]; `ptr` addresses `len` bytes.
pub unsafe extern "C" fn load(node: *mut c_void, ptr: *const u8, len: usize, sink: *mut c_void, write: Write) -> bool {
    let bytes = slice(ptr, len);
    with(node, sink, write, |n| n.load(bytes))
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    match len {
        0 => &[],
        n => std::slice::from_raw_parts(ptr, n),
    }
}

unsafe fn slice_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    match len {
        0 => &mut [],
        n => std::slice::from_raw_parts_mut(ptr, n),
    }
}

/// What a node file spells once: the type that implements [`AudioNode`] and the manifest it
/// declares.
#[macro_export]
macro_rules! export {
    ($node:ty, $manifest:expr) => {
        #[doc(hidden)]
        pub fn __goofi_create() -> Box<dyn $crate::AudioNode> {
            Box::new(<$node as Default>::default())
        }
        #[doc(hidden)]
        pub static __GOOFI_MANIFEST: &$crate::Manifest = &$manifest;
    };
}

/// What the generated crate spells around the node module: the three symbols the loader reads.
#[macro_export]
macro_rules! cdylib {
    ($node:ident) => {
        #[no_mangle]
        pub extern "C" fn goofi_version() -> *const ::std::ffi::c_char {
            $crate::abi::version()
        }
        #[no_mangle]
        pub extern "C" fn goofi_describe() -> *const ::std::ffi::c_char {
            $crate::abi::describe_c($node::__GOOFI_MANIFEST)
        }
        unsafe extern "C" fn __goofi_create_raw() -> *mut ::std::ffi::c_void {
            $crate::abi::instance($node::__goofi_create)
        }
        static __GOOFI_VTABLE: $crate::abi::VTable = $crate::abi::VTable {
            create: __goofi_create_raw,
            destroy: $crate::abi::destroy,
            channels: $crate::abi::channels,
            prepare: $crate::abi::prepare,
            process: $crate::abi::process,
            feedback: $crate::abi::feedback,
            save: $crate::abi::save,
            load: $crate::abi::load,
        };
        #[no_mangle]
        pub extern "C" fn goofi_audio_node() -> *const $crate::abi::VTable {
            &__GOOFI_VTABLE
        }
    };
}
