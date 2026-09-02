//! The half of a node's cdylib boundary that is the same whatever engine runs it: the version
//! symbol every loader checks first, the describe symbol every scan parses, and the byte slice a
//! reply or a panic's own words cross on.

use std::ffi::{c_char, c_void, CString};
use std::sync::OnceLock;

/// The `goofi_version` answer: the version the loader refuses a mismatch against. Every crate
/// inherits the workspace's, so this is the one number the whole boundary is built at.
pub fn version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// The `goofi_describe` answer, once per library: the probe schema every out-of-crate node
/// answers, whatever language or engine it belongs to.
pub fn describe_once(json: impl FnOnce() -> String) -> *const c_char {
    static DESCRIBED: OnceLock<CString> = OnceLock::new();
    DESCRIBED.get_or_init(|| CString::new(json()).expect("no NUL in a manifest")).as_ptr()
}

/// A borrowed byte slice as the boundary spells it.
#[repr(C)]
pub struct Bytes {
    pub ptr: *const u8,
    pub len: usize,
}

impl Bytes {
    pub fn of(s: &[u8]) -> Bytes {
        Bytes { ptr: s.as_ptr(), len: s.len() }
    }

    /// # Safety
    /// `ptr` addresses `len` readable bytes that outlive the slice.
    pub unsafe fn as_slice<'a>(&self) -> &'a [u8] {
        match self.len {
            0 => &[],
            n => std::slice::from_raw_parts(self.ptr, n),
        }
    }
}

/// The host's collector: the node writes, the host owns the bytes.
pub type Write = unsafe extern "C" fn(sink: *mut c_void, bytes: Bytes);

/// The host's end of [`Write`]. `sink` is the `Vec<u8>` the caller is filling.
///
/// # Safety
/// `sink` is a live `&mut Vec<u8>` and `bytes` names readable memory.
pub unsafe extern "C" fn collect(sink: *mut c_void, bytes: Bytes) {
    (*(sink as *mut Vec<u8>)).extend_from_slice(bytes.as_slice());
}
