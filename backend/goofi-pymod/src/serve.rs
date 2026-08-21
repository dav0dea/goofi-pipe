//! `goofi.serve()` — the subprocess child loop (wheel only; `extension-module`): read the node
//! source and service names from the environment, run the node over the same [`crate::exec`]
//! seam the in-process tier uses, and speak `goofi_codec` frames over iceoryx2.

use std::collections::HashSet;
use std::time::Duration;

use goofi_codec::{decode_request, encode_error_response, encode_response};
use goofi_core::{Data as CoreData, SrcDtype};
use iceoryx2::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::loader::{find_node_class, module_from_source};

/// iceoryx2 byte-slice pool ceiling — matches the parent publisher's default.
const MAX_PAYLOAD: usize = 64 * 1024;

/// The subprocess entry point (`import goofi; goofi.serve()`); returns only on a fatal error.
#[pyfunction]
pub fn serve(py: Python<'_>) -> PyResult<()> {
    // FIRST, before the user module is even compiled, so a child orphaned during a slow import
    // still stops instead of reaching the poll loop.
    goofi_codec::liveness::watch_parent(&env(goofi_codec::liveness::ENV_VAR)?)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parent-liveness watcher: {e}")))?;

    // Route stdout -> stderr BEFORE compiling the user module, so a node's prints cannot reach
    // the parent's stdout.
    let os = py.import("os")?;
    os.call_method1("dup2", (2, 1))?;
    let sys = py.import("sys")?;
    sys.setattr("stdout", sys.getattr("stderr")?)?;

    let source = env("GOOFI_NODE_SRC")?;
    let req_name = env("GOOFI_IOX_REQ")?;
    let resp_name = env("GOOFI_IOX_RESP")?;

    let module = module_from_source(py, "goofi_node_main", &source)?;
    let instance = find_node_class(py, &module)?.call0()?;
    let out_slots = slot_names(&instance, "OUTPUTS")?;
    let in_slots = slot_names(&instance, "INPUTS")?;
    let out_refs: Vec<&str> = out_slots.iter().map(|s| s.as_str()).collect();
    let in_refs: Vec<&str> = in_slots.iter().map(|s| s.as_str()).collect();

    run_loop(py, &instance, &in_refs, &out_refs, &req_name, &resp_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// The slot names one declaration constant holds, in declaration order.
fn slot_names(instance: &Bound<'_, PyAny>, constant: &str) -> PyResult<Vec<String>> {
    instance.getattr(constant)?.cast::<PyDict>()?.iter().map(|(k, _)| k.extract()).collect()
}

/// Read a required env var, or a clean Python error naming it.
fn env(key: &str) -> PyResult<String> {
    std::env::var(key).map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(format!("{key} unset")))
}

/// Open the iceoryx2 ports (the mirror of the parent's) and run the request→process→response loop.
fn run_loop(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    in_slots: &[&str],
    out_slots: &[&str],
    req_name: &str,
    resp_name: &str,
) -> Result<(), String> {
    let node = NodeBuilder::new().create::<ipc::Service>().map_err(|e| format!("iox node: {e}"))?;
    // Must stay the same service config as the parent's `build_ports`.
    let mk = |name: &str| {
        node.service_builder(&name.try_into().map_err(|e| format!("bad service `{name}`: {e:?}"))?)
            .publish_subscribe::<[u8]>()
            .enable_safe_overflow(true)
            .max_publishers(1)
            .max_subscribers(16)
            .open_or_create()
            .map_err(|e| format!("service `{name}`: {e}"))
    };
    let req_sub =
        mk(req_name)?.subscriber_builder().create().map_err(|e| format!("req subscriber: {e}"))?;
    let resp_pub = mk(resp_name)?
        .publisher_builder()
        .initial_max_slice_len(MAX_PAYLOAD)
        .allocation_strategy(AllocationStrategy::PowerOfTwo)
        .create()
        .map_err(|e| format!("resp publisher: {e}"))?;

    let mut warned: HashSet<SrcDtype> = HashSet::new();
    let mut did_setup = false;
    let mut last_seq: Option<u32> = None;

    loop {
        // Latest-wins, mirroring the parent + iceoryx2 semantics.
        let mut latest = None;
        loop {
            match req_sub.receive() {
                Ok(Some(s)) => latest = Some(s),
                Ok(None) => break,
                Err(e) => return Err(format!("iox receive: {e}")),
            }
        }
        let Some(sample) = latest else {
            // DETACHED: holding the GIL over the idle poll would starve a node's own Python
            // threads, and a receiver thread started in `setup()` is this tier's canonical shape.
            py.detach(|| std::thread::sleep(Duration::from_micros(500)));
            continue;
        };
        let payload = sample.payload();
        if payload.len() < 4 {
            continue; // not a framed request
        }
        let seq = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        if last_seq == Some(seq) {
            continue; // a re-publish of an already-answered request — its response is in the buffer
        }
        let resp = handle(py, instance, in_slots, out_slots, &mut warned, &mut did_setup, &payload[4..])
            .map_err(|e| format!("node process: {e}"))?;

        // [u32 seq][response frame]
        let mut msg = Vec::with_capacity(4 + resp.len());
        msg.extend_from_slice(&seq.to_le_bytes());
        msg.extend_from_slice(&resp);
        resp_pub
            .loan_slice_uninit(msg.len())
            .map_err(|e| format!("iox loan: {e}"))?
            .write_from_slice(msg.as_slice())
            .send()
            .map_err(|e| format!("iox send: {e}"))?;
        last_seq = Some(seq);
    }
}

/// Decode one request → run the node → encode the response. A MALFORMED request is fatal; a node
/// raise is a per-tick error response, which the parent surfaces without respawning the child.
fn handle(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    in_slots: &[&str],
    out_slots: &[&str],
    warned: &mut HashSet<SrcDtype>,
    did_setup: &mut bool,
    body: &[u8],
) -> PyResult<Vec<u8>> {
    let (params, arrived) = decode_request(body).map_err(pyo3::exceptions::PyValueError::new_err)?;
    // The wire carries only the slots that hold a frame; widen it back to every declared slot,
    // `None` where nothing arrived.
    let inputs: Vec<(&str, Option<&CoreData>)> = in_slots
        .iter()
        .map(|name| (*name, arrived.iter().find(|(n, _)| n == name).map(|(_, d)| d)))
        .collect();
    match run_node(py, instance, &params, &inputs, out_slots, warned, did_setup) {
        Ok(outs) => {
            let slots: Vec<(&str, &CoreData)> = outs.iter().map(|(n, d)| (n.as_str(), d)).collect();
            Ok(encode_response(&slots))
        }
        Err(e) => Ok(encode_error_response(&e.to_string())),
    }
}

/// Run `setup()` until it SUCCEEDS, then `process()`; a setup that raised is retried on the next
/// request, and `process()` never runs after a failed one.
fn run_node(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    params: &crate::exec::Groups,
    inputs: &[(&str, Option<&CoreData>)],
    out_slots: &[&str],
    warned: &mut HashSet<SrcDtype>,
    did_setup: &mut bool,
) -> PyResult<Vec<(String, CoreData)>> {
    if !*did_setup {
        crate::exec::run_setup(py, instance, params)?;
        *did_setup = true;
    }
    crate::exec::run_process(py, instance, params, inputs, out_slots, warned)
}
