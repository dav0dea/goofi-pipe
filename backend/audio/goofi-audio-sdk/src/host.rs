//! The host half of the boundary: a built node's vtable behind the same [`AudioNode`] the engine
//! runs everything else as. A panic the shim caught comes back as a Rust panic on this side, so
//! the runtime's one catch around `process` sees a loaded node and a built-in one alike.

use std::ffi::c_void;

use goofi_node::NodeManifest;

use crate::abi::{BlockDesc, OutDesc, PortDesc, VTable};
use crate::{AudioNode, Block, MAX_PORTS};

/// One loaded node type: its vtable and the manifest the host leaked from `goofi_describe`.
pub struct Loaded {
    vtable: &'static VTable,
    manifest: &'static NodeManifest,
}

impl Loaded {
    /// # Safety
    /// `library` was built by [`crate::cdylib!`] at this SDK's version — its `goofi_version`
    /// was read and matched before this is called.
    pub unsafe fn open(library: &'static libloading::Library, manifest: &'static NodeManifest) -> Result<Loaded, String> {
        let entry: libloading::Symbol<unsafe extern "C" fn() -> *const VTable> =
            library.get(b"goofi_audio_node\0").map_err(|e| format!("no `goofi_audio_node` symbol: {e}"))?;
        let vtable = entry();
        if vtable.is_null() {
            return Err("`goofi_audio_node` answered null".into());
        }
        Ok(Loaded { vtable: &*vtable, manifest })
    }

    pub fn manifest(&self) -> &'static NodeManifest {
        self.manifest
    }

    pub fn instantiate(&self) -> Box<dyn AudioNode> {
        let node = unsafe { (self.vtable.create)() };
        Box::new(Handle { node, vtable: self.vtable })
    }
}

struct Handle {
    node: *mut c_void,
    vtable: &'static VTable,
}

// The instance is used from one thread at a time, as every node is: the control thread until
// it is inserted, the audio thread after.
unsafe impl Send for Handle {}

unsafe extern "C" fn collect(sink: *mut c_void, ptr: *const u8, len: usize) {
    let bytes = &mut *(sink as *mut Vec<u8>);
    if len > 0 {
        bytes.extend_from_slice(std::slice::from_raw_parts(ptr, len));
    }
}

const NONE: PortDesc = PortDesc { data: std::ptr::null(), channels: 0, wired: false };

impl AudioNode for Handle {
    fn channels(&self, ins: &[u16], params: &[f64], outs: usize) -> Vec<u16> {
        let mut out = vec![1u16; outs];
        if !self.node.is_null() {
            unsafe {
                (self.vtable.channels)(self.node, ins.as_ptr(), ins.len(), params.as_ptr(), params.len(), out.as_mut_ptr(), outs)
            };
        }
        out
    }

    fn prepare(&mut self, rate: f64) {
        if !self.node.is_null() {
            unsafe { (self.vtable.prepare)(self.node, rate) };
        }
    }

    fn process(&mut self, b: &mut Block<'_>) {
        if self.node.is_null() {
            std::panic::resume_unwind(Box::new("the constructor panicked".to_string()));
        }
        let ins: [PortDesc; MAX_PORTS] = std::array::from_fn(|i| b.ins.get(i).map_or(NONE, PortDesc::of));
        let params: [PortDesc; MAX_PORTS] = std::array::from_fn(|i| b.params.get(i).map_or(NONE, PortDesc::of));
        let outs: [OutDesc; MAX_PORTS] = std::array::from_fn(|i| match b.outs.get_mut(i) {
            Some(o) => OutDesc::of(o),
            None => OutDesc { data: std::ptr::null_mut(), channels: 0 },
        });
        let desc = BlockDesc {
            ins: ins.as_ptr(),
            n_ins: b.ins.len(),
            outs: outs.as_ptr(),
            n_outs: b.outs.len(),
            params: params.as_ptr(),
            n_params: b.params.len(),
        };
        let mut text: Vec<u8> = Vec::new();
        let ok = unsafe { (self.vtable.process)(self.node, &desc, &mut text as *mut Vec<u8> as *mut c_void, collect) };
        if !ok {
            std::panic::resume_unwind(Box::new(String::from_utf8_lossy(&text).into_owned()));
        }
    }

    fn feedback(&self) -> bool {
        !self.node.is_null() && unsafe { (self.vtable.feedback)(self.node) }
    }

    fn save(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        if !self.node.is_null() {
            unsafe { (self.vtable.save)(self.node, &mut bytes as *mut Vec<u8> as *mut c_void, collect) };
        }
        bytes
    }

    fn load(&mut self, bytes: &[u8]) {
        if !self.node.is_null() {
            unsafe { (self.vtable.load)(self.node, bytes.as_ptr(), bytes.len()) };
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { (self.vtable.destroy)(self.node) }
    }
}
