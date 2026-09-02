//! The C boundary a built signal node crosses: one vtable of `extern "C"` entries over codec
//! bytes, the shim that puts an author's [`Node`] behind it, and the two macros a node file and
//! its generated crate spell. Only code and plain data cross; never a Rust type.

use std::ffi::{c_char, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

use goofi_core::{probe, Data};
use goofi_node::{ParamKey, ParamSpec, Params};
use indexmap::IndexMap;

use crate::{Inputs, Manifest, Node, NodeCtx, Outputs};

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

/// What a node may ask its runtime, as plain data.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ctx {
    pub now: f64,
}

/// The host's collector for a reply: the node writes, the host owns the bytes.
pub type Write = unsafe extern "C" fn(sink: *mut c_void, bytes: Bytes);

/// Every entry has one shape: a request in, a reply out through the host's sink.
pub type Call = unsafe extern "C" fn(node: *mut c_void, ctx: Ctx, request: Bytes, sink: *mut c_void, write: Write);

#[repr(C)]
pub struct VTable {
    pub create: unsafe extern "C" fn() -> *mut c_void,
    pub destroy: unsafe extern "C" fn(node: *mut c_void),
    pub setup: Call,
    pub process: Call,
    pub on_param_changed: Call,
    pub on_param_refreshed: Call,
}

/// The `goofi_version` answer: this SDK's version, which is goofi's.
pub fn version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// The `goofi_describe` answer: the manifest as the probe schema, once per library.
pub fn describe_c(manifest: &Manifest) -> *const c_char {
    static DESCRIBED: OnceLock<CString> = OnceLock::new();
    DESCRIBED.get_or_init(|| CString::new(describe(manifest)).expect("no NUL in a manifest")).as_ptr()
}

/// A manifest as the probe schema a Python node answers — one schema for every out-of-crate node.
pub fn describe(m: &Manifest) -> String {
    let intro = probe::Introspection {
        gil_safe: true,
        doc: m.doc.to_string(),
        category: Some(m.category.to_string()),
        producer: m.producer,
        inputs: m
            .inputs
            .iter()
            .map(|s| probe::Slot {
                name: s.name.to_string(),
                kind: s.kind.name().to_string(),
                trigger: s.trigger_process,
                multi: s.multi,
                required: s.required,
            })
            .collect(),
        outputs: m
            .outputs
            .iter()
            .map(|o| probe::OutSlot { name: o.name.to_string(), kind: o.kind.name().to_string() })
            .collect(),
        params: m
            .params
            .iter()
            .map(|p| probe::Param {
                group: p.group.to_string(),
                name: p.name.to_string(),
                doc: p.doc.map(str::to_string),
                expression: p.expression.map(|e| e.source.to_string()),
                spec: match p.spec {
                    ParamSpec::Int { default, min, max } => probe::ParamSpec::Int { default, min, max },
                    ParamSpec::Float { default, min, max } => probe::ParamSpec::Float { default, min, max },
                    ParamSpec::Bool { default } => probe::ParamSpec::Bool { default },
                    ParamSpec::Str { default, options, refresh } => probe::ParamSpec::Str {
                        default: default.to_string(),
                        options: options.iter().map(|s| s.to_string()).collect(),
                        refresh,
                    },
                },
            })
            .collect(),
    };
    serde_json::to_string(&intro).expect("a manifest serializes")
}

/// The instance behind a `*mut c_void`: the author's node and what the shim keeps around it.
pub struct Instance {
    node: Box<dyn Node>,
    manifest: &'static Manifest,
    ctx: NodeCtx,
}

/// Box a fresh node for the host; a constructor that panics answers null, which the host
/// reports as the node's setup error rather than unwinding across the boundary.
pub fn instance(create: impl FnOnce() -> Box<dyn Node>, manifest: &'static Manifest) -> *mut c_void {
    match catch_unwind(AssertUnwindSafe(create)) {
        Ok(node) => Box::into_raw(Box::new(Instance { node, manifest, ctx: NodeCtx::new() })) as *mut c_void,
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `node` came from [`instance`] and is not used after this.
pub unsafe extern "C" fn destroy(node: *mut c_void) {
    if !node.is_null() {
        drop(Box::from_raw(node as *mut Instance));
    }
}

/// Every entry the same way: decode, call, encode — a refusal or a panic is an error reply.
unsafe fn call(
    node: *mut c_void,
    ctx: Option<Ctx>,
    request: Bytes,
    sink: *mut c_void,
    write: Write,
    f: impl FnOnce(&mut Instance, &[u8]) -> Result<Vec<u8>, String>,
) {
    let inst = &mut *(node as *mut Instance);
    if let Some(ctx) = ctx {
        inst.ctx.now = ctx.now;
    }
    let request = request.as_slice();
    let reply = match catch_unwind(AssertUnwindSafe(|| f(inst, request))) {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(e)) => goofi_codec::encode_error_response(&e),
        Err(p) => goofi_codec::encode_error_response(&panic_message(p)),
    };
    write(sink, Bytes::of(&reply));
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = p.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic in node".to_string()
    }
}

fn process_request(req: &[u8]) -> Result<(goofi_codec::ParamMap, Vec<(String, Data)>), String> {
    match goofi_codec::decode_request(req)? {
        goofi_codec::Request::Process { params, slots } => Ok((params, slots)),
        goofi_codec::Request::Refresh { .. } => Err("a refresh where a run was expected".into()),
    }
}

/// # Safety
/// `node` came from [`instance`]; `request` and `sink` are the host's for the call.
pub unsafe extern "C" fn setup(node: *mut c_void, ctx: Ctx, request: Bytes, sink: *mut c_void, write: Write) {
    call(node, Some(ctx), request, sink, write, |inst, req| {
        let (params, _) = process_request(req)?;
        inst.node.setup(&mut inst.ctx, &Params::new(&params)).map_err(|e| e.0)?;
        Ok(goofi_codec::encode_response(&[]))
    })
}

/// # Safety
/// As [`setup`].
pub unsafe extern "C" fn process(node: *mut c_void, ctx: Ctx, request: Bytes, sink: *mut c_void, write: Write) {
    call(node, Some(ctx), request, sink, write, |inst, req| {
        let (params, slots) = process_request(req)?;
        let mut singles: IndexMap<&'static str, Option<Data>> =
            inst.manifest.inputs.iter().filter(|s| !s.multi).map(|s| (s.name, None)).collect();
        let mut multis: IndexMap<&'static str, Vec<Data>> =
            inst.manifest.inputs.iter().filter(|s| s.multi).map(|s| (s.name, Vec::new())).collect();
        for (name, data) in slots {
            if let Some(frames) = multis.get_mut(name.as_str()) {
                frames.push(data);
            } else if let Some(slot) = singles.get_mut(name.as_str()) {
                *slot = Some(data);
            }
        }
        let mut outputs: IndexMap<&'static str, Option<Data>> =
            inst.manifest.outputs.iter().map(|o| (o.name, None)).collect();
        let inputs = Inputs::with_multi(&singles, &multis);
        let mut out = Outputs::new(&mut outputs);
        inst.node.process(&inputs, &mut out, &mut inst.ctx, &Params::new(&params)).map_err(|e| e.0)?;
        let emitted: Vec<(&str, &Data)> =
            outputs.iter().filter_map(|(name, d)| d.as_ref().map(|d| (*name, d))).collect();
        Ok(goofi_codec::encode_response(&emitted))
    })
}

/// # Safety
/// As [`setup`]; `request` is `(group, name, Param)` as msgpack.
pub unsafe extern "C" fn on_param_changed(node: *mut c_void, ctx: Ctx, request: Bytes, sink: *mut c_void, write: Write) {
    let _ = ctx;
    call(node, None, request, sink, write, |inst, req| {
        let (group, name, value): (String, String, goofi_core::Param) =
            rmp_serde::from_slice(req).map_err(|e| e.to_string())?;
        inst.node.on_param_changed(&ParamKey::new(group, name), &value).map_err(|e| e.0)?;
        Ok(goofi_codec::encode_response(&[]))
    })
}

/// # Safety
/// As [`setup`]; `request` is a codec refresh request.
pub unsafe extern "C" fn on_param_refreshed(node: *mut c_void, ctx: Ctx, request: Bytes, sink: *mut c_void, write: Write) {
    let _ = ctx;
    call(node, None, request, sink, write, |inst, req| {
        let goofi_codec::Request::Refresh { params, group, name } = goofi_codec::decode_request(req)? else {
            return Err("a run where a refresh was expected".into());
        };
        let options = inst.node.on_param_refreshed(&ParamKey::new(group, name), &Params::new(&params));
        Ok(goofi_codec::encode_options_response(&options))
    })
}

/// What a node file spells once: the type that implements [`Node`] and the manifest it declares.
#[macro_export]
macro_rules! export {
    ($node:ty, $manifest:expr) => {
        #[doc(hidden)]
        pub fn __goofi_create() -> Box<dyn $crate::Node> {
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
            $crate::abi::instance($node::__goofi_create, $node::__GOOFI_MANIFEST)
        }
        static __GOOFI_VTABLE: $crate::abi::VTable = $crate::abi::VTable {
            create: __goofi_create_raw,
            destroy: $crate::abi::destroy,
            setup: $crate::abi::setup,
            process: $crate::abi::process,
            on_param_changed: $crate::abi::on_param_changed,
            on_param_refreshed: $crate::abi::on_param_refreshed,
        };
        #[no_mangle]
        pub extern "C" fn goofi_signal_node() -> *const $crate::abi::VTable {
            &__GOOFI_VTABLE
        }
    };
}
