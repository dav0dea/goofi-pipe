//! `goofi.serve()` — the subprocess child loop (wheel only; `extension-module`).
//!
//! The child bootstrap is uniform: `import goofi; goofi.serve()`. `serve` reads its node
//! source + iceoryx2 service names from the environment, runs the node over the SAME
//! [`crate::exec`] seam the in-process tier uses, and speaks the shared `goofi_codec`
//! request/response frames over the **Rust** iceoryx2 transport.
//!
//! This REPLACES the old ~106-line inline Python `WORKER_SRC` + its Python `iceoryx2`
//! binding. Because the child is now Rust and shares `goofi_codec`, meta (channels/sfreq/
//! index/dtype) crosses with full fidelity, and cast-to-f32 + `warn_cast_once` live only in
//! the one shared `run_process` — fixing the WORKER_SRC silent-cast gap by construction.
//!
//! The transport-level `seq` (re-publish dedup) is the OUTER 4-byte prefix, matching the
//! parent's `one_roundtrip`; the codec payload knows nothing about it.
//!
//! The child also holds the read end of the parent's liveness pipe
//! ([`goofi_codec::liveness`]): a watcher thread ends this process the moment the parent's
//! write end closes, so a Ctrl-C'd or crashed manager cannot leave it spinning forever.

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

/// The subprocess entry point (`import goofi; goofi.serve()`). Blocks forever running the
/// node's tick loop; returns only on a fatal error — which surfaces as a Python exception,
/// so the parent sees the child exit and reaps + respawns it.
#[pyfunction]
pub fn serve(py: Python<'_>) -> PyResult<()> {
    // The parent-liveness watcher goes up FIRST — before the user module is even compiled —
    // so a child orphaned during a slow import still stops instead of reaching the poll loop.
    goofi_codec::liveness::watch_parent(&env(goofi_codec::liveness::ENV_VAR)?)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parent-liveness watcher: {e}")))?;

    // Route stdout -> stderr BEFORE compiling the user module, so any node/C-extension
    // write to stdout is harmless (the frame plane is SHM; there is no stdout protocol).
    let os = py.import("os")?;
    os.call_method1("dup2", (2, 1))?;
    let sys = py.import("sys")?;
    sys.setattr("stdout", sys.getattr("stderr")?)?;

    let source = env("GOOFI_NODE_SRC")?;
    let req_name = env("GOOFI_IOX_REQ")?;
    let resp_name = env("GOOFI_IOX_RESP")?;

    let module = module_from_source(py, "goofi_node_main", &source)?;
    let instance = find_node_class(py, &module)?.call0()?;
    let cfg = instance.call_method0("config_output_slots")?;
    let out_slots: Vec<String> =
        cfg.cast::<PyDict>()?.iter().map(|(k, _)| k.extract()).collect::<PyResult<_>>()?;
    let out_refs: Vec<&str> = out_slots.iter().map(|s| s.as_str()).collect();

    run_loop(py, &instance, &out_refs, &req_name, &resp_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Read a required env var, or a clean Python error naming it.
fn env(key: &str) -> PyResult<String> {
    std::env::var(key).map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(format!("{key} unset")))
}

/// Open the iceoryx2 ports (child = subscriber on `<id>_req`, publisher on `<id>_resp` —
/// the mirror of the parent) and run the request→process→response loop until a fatal error.
fn run_loop(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    out_slots: &[&str],
    req_name: &str,
    resp_name: &str,
) -> Result<(), String> {
    let node = NodeBuilder::new().create::<ipc::Service>().map_err(|e| format!("iox node: {e}"))?;
    // Same byte-slice service config as the parent's `build_ports` (0.9.3 ABI compatible).
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
        // Drain to the latest request (latest-wins, mirroring the parent + iceoryx2 semantics).
        let mut latest = None;
        loop {
            match req_sub.receive() {
                Ok(Some(s)) => latest = Some(s),
                Ok(None) => break,
                Err(e) => return Err(format!("iox receive: {e}")),
            }
        }
        let Some(sample) = latest else {
            std::thread::sleep(Duration::from_micros(500)); // idle poll; a request wakes it ~0.5ms
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
        let resp = handle(py, instance, out_slots, &mut warned, &mut did_setup, &payload[4..])
            .map_err(|e| format!("node process: {e}"))?;

        // [u32 seq][response frame] — the seq lets the parent dedup a re-publish.
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

/// Decode one request → run the node → encode the response. A MALFORMED request is a fatal
/// protocol error (the `?` kills the child). A node RAISE in `setup()`/`process()` is a
/// per-tick error reported back as an error response — the parent surfaces it like the
/// in-process `Ok(Err)` WITHOUT respawning the child (preserving node state + the real
/// exception text, using the SAME `e.to_string()` the in-process tier uses, so the two tiers'
/// error channels agree). The cast-to-f32 warning is deduped in `run_process`.
fn handle(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    out_slots: &[&str],
    warned: &mut HashSet<SrcDtype>,
    did_setup: &mut bool,
    body: &[u8],
) -> PyResult<Vec<u8>> {
    let (params, inputs) = decode_request(body).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let present: Vec<(&str, &CoreData)> = inputs.iter().map(|(n, d)| (n.as_str(), d)).collect();
    match run_node(py, instance, &params, &present, out_slots, warned, did_setup) {
        Ok(outs) => {
            let slots: Vec<(&str, &CoreData)> = outs.iter().map(|(n, d)| (n.as_str(), d)).collect();
            Ok(encode_response(&slots))
        }
        Err(e) => Ok(encode_error_response(&e.to_string())),
    }
}

/// Run `setup()` (once, on the first request) then `process()`; a raise in either is the node
/// error. `did_setup` is set BEFORE running setup so a setup raise is reported per-tick, not
/// retried forever (matching the in-process tier's run-once semantics).
fn run_node(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    params: &crate::exec::Groups,
    present: &[(&str, &CoreData)],
    out_slots: &[&str],
    warned: &mut HashSet<SrcDtype>,
    did_setup: &mut bool,
) -> PyResult<Vec<(String, CoreData)>> {
    if !*did_setup {
        *did_setup = true;
        crate::exec::run_setup(py, instance, params)?;
    }
    crate::exec::run_process(py, instance, params, present, out_slots, warned)
}
