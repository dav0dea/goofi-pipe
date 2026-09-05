//! The host half of the boundary: a built node's vtable behind the same [`Node`] the engine runs
//! everything else as, marshalled exactly as the subprocess tier is.

use std::ffi::c_void;

use goofi_codec::Response;
use goofi_core::Data;
use goofi_node::{NodeManifest, ParamKey, Params};

use crate::abi::{collect, Bytes, Call, Ctx, VTable};
use crate::{Inputs, Node, NodeCtx, NodeError, NodeResult, Outputs};

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
            library.get(b"goofi_signal_node\0").map_err(|e| format!("no `goofi_signal_node` symbol: {e}"))?;
        let vtable = entry();
        if vtable.is_null() {
            return Err("`goofi_signal_node` answered null".into());
        }
        Ok(Loaded { vtable: &*vtable, manifest })
    }

    pub fn manifest(&self) -> &'static NodeManifest {
        self.manifest
    }

    pub fn instantiate(&self) -> Box<dyn Node> {
        let node = unsafe { (self.vtable.create)() };
        Box::new(Handle { node, vtable: self.vtable, manifest: self.manifest })
    }
}

struct Handle {
    node: *mut c_void,
    vtable: &'static VTable,
    manifest: &'static NodeManifest,
}

// The instance is used from the one thread that runs it, as every node is.
unsafe impl Send for Handle {}

impl Handle {
    fn call(&mut self, entry: Call, now: f64, request: &[u8]) -> Result<Response, String> {
        if self.node.is_null() {
            return Err("the node's constructor panicked".into());
        }
        let mut reply: Vec<u8> = Vec::new();
        unsafe { entry(self.node, Ctx { now }, Bytes::of(request), &mut reply as *mut Vec<u8> as *mut c_void, collect) };
        goofi_codec::decode_response(&reply)
    }

    fn done(answer: Result<Response, String>) -> NodeResult {
        match answer {
            Ok(Response::Slots(_)) => Ok(()),
            Ok(Response::NodeError(msg)) => Err(NodeError(msg)),
            Ok(Response::Options(_)) => Err(NodeError("the node answered options where none were asked".into())),
            Err(e) => Err(NodeError(e)),
        }
    }
}

impl Node for Handle {
    fn setup(&mut self, ctx: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        Self::done(self.call(self.vtable.setup, ctx.now, &goofi_codec::encode_request(p.groups(), &[])))
    }

    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, ctx: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        // Every present frame crosses, a `multi` slot's under its one name repeated, each with
        // its source; a single slot's has none.
        let mut present: Vec<(&str, &str, &Data)> = Vec::new();
        for slot in self.manifest.inputs {
            if slot.multi {
                present.extend(inp.get_multi(slot.name).iter().map(|(source, d)| (slot.name, source.as_str(), d)));
            } else if let Some(d) = inp.get(slot.name) {
                present.push((slot.name, "", d));
            }
        }
        match self.call(self.vtable.process, ctx.now, &goofi_codec::encode_request(p.groups(), &present)) {
            Ok(Response::Slots(outs)) => {
                for (slot, data) in outs {
                    out.set(&slot, data);
                }
                Ok(())
            }
            other => Self::done(other),
        }
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &goofi_core::Param) -> NodeResult {
        let request = rmp_serde::to_vec(&(&key.group, &key.name, v)).map_err(|e| NodeError(e.to_string()))?;
        Self::done(self.call(self.vtable.on_param_changed, 0.0, &request))
    }

    fn on_param_refreshed(&mut self, key: &ParamKey, p: &Params<'_>) -> Option<Vec<String>> {
        let request = goofi_codec::encode_refresh_request(p.groups(), &key.group, &key.name);
        match self.call(self.vtable.on_param_refreshed, 0.0, &request) {
            Ok(Response::Options(options)) => options,
            _ => None,
        }
    }

    fn on_pulse(&mut self, key: &ParamKey, p: &Params<'_>) -> NodeResult {
        let request = goofi_codec::encode_pulse_request(p.groups(), &key.group, &key.name);
        Self::done(self.call(self.vtable.on_pulse, 0.0, &request))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { (self.vtable.destroy)(self.node) }
    }
}
