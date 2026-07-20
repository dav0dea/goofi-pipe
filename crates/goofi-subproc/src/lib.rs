//! goofi-subproc — the subprocess node tier (Pathway C).
//!
//! A [`RemoteNode`] runs a Python node in an isolated **GIL** interpreter, one
//! process per node. It exists for two reasons: (1) deps that aren't
//! free-threading-safe can't run in the in-process pyo3 host; (2) per the latency
//! finding, a *separate interpreter* has its own object ownership, so parallel
//! heavy-Python compute avoids free-threaded CPython's biased-refcount penalty.
//!
//! Each tick is one request/response over **iceoryx2 shared memory** (the spec's
//! subprocess-boundary transport — the same zero-copy plane the Python backend uses).
//! The input `Data` is written as a compact body ([`encode_body`] — a typed `sfreq` prefix,
//! then the full meta as an OPAQUE `goofi_codec` blob the child echoes unchanged so channels
//! survive, then the shared [`goofi_codec::encode_array_body`] layout), prefixed with a
//! 4-byte request sequence, and published to the child's per-node `<id>_req` byte-slice
//! service; the child runs `process(x)` and publishes `[seq][body]` back on `<id>_resp`,
//! which we [`decode_body`]. The sequence disambiguates responses so a re-publish (needed
//! while the child's subscriber is still connecting) never returns a stale frame.
//!
//! Rust parent uses the iceoryx2 **Rust crate**; the Python child uses the iceoryx2 Python
//! binding directly (same 0.9.3 ABI — `[u8]` ⇄ `Slice[c_uint8]`, validated by a cross-language
//! round-trip). The child interpreter must therefore have `iceoryx2` + `numpy`.
//!
//! The node implements the same [`Node`] trait as native and in-process Python
//! nodes, so the scheduler never branches on backend and the engine hosts it
//! through the ordinary `register_dyn_type` seam.

use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use iceoryx2::prelude::*;

use goofi_core::Data;
use goofi_node::{Inputs, Node, NodeCtx, NodeResult, Outputs};

/// Per-process counter giving each spawned subprocess a unique iceoryx2 service-name base
/// (`goofi_sub_<pid>_<n>`), so concurrent nodes — and a respawn after a reset — never collide.
static SUBPROC_SEQ: AtomicU64 = AtomicU64::new(0);

/// iceoryx2 byte-slice pool ceiling per publisher (matches the Python transport's default).
const MAX_PAYLOAD: usize = 64 * 1024;

/// Default cap on how long a tick waits for a subprocess response before treating
/// the child as hung. Generous enough to cover cold start (spawn + numpy import);
/// a hung child is killed and surfaces as a node error rather than stranding the
/// scheduler (and, in the bridge, the graph mutex) indefinitely.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// The subprocess transport payload: `[sfreq: f64 LE, NaN=none][meta_len: u32 LE][meta msgpack]` then
/// the shared [`goofi_codec::encode_array_body`] layout. The meta blob is [`goofi_codec::pack_meta`] —
/// OPAQUE to the child (it echoes it unchanged, so channels/index survive) while Rust round-trips it
/// via [`goofi_codec::parse_meta`]. `sfreq` is ALSO carried as a typed prefix purely so the child can
/// read it (a PSD-class node needs it) without a msgpack parser; it overlays the parsed meta on the
/// way back, so it survives even when a shape-changing node drops the opaque meta. The array body is
/// viewed in place by the child (see `WORKER_SRC`).
fn encode_body(d: &Data) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&d.meta().sfreq.unwrap_or(f64::NAN).to_le_bytes());
    let meta = goofi_codec::pack_meta(d);
    out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    out.extend_from_slice(&meta);
    if let goofi_core::Value::Array(store) = d.value() {
        goofi_codec::encode_array_body(store, &mut out);
    }
    out
}

fn decode_body(buf: &[u8]) -> std::result::Result<Data, String> {
    if buf.len() < 12 {
        return Err(format!("short transport body ({} bytes)", buf.len()));
    }
    let sfreq = f64::from_le_bytes(buf[0..8].try_into().unwrap());
    let meta_len = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let meta_end = 12 + meta_len;
    let meta_bytes = buf.get(12..meta_end).ok_or("transport meta truncated")?;
    let mut meta = goofi_codec::parse_meta(meta_bytes)?;
    // The typed sfreq survives even when a shape-changing node emptied the opaque meta.
    if !sfreq.is_nan() {
        meta.sfreq = Some(sfreq);
    }
    // The child casts foreign dtypes to f32 at ingest; the source dtype is dropped here
    // (the dedup cast-warning is wired at RemoteNode in the following step).
    goofi_codec::decode_array_body(&buf[meta_end..], meta).map(|(d, _)| d)
}

/// The Python worker: a self-contained body codec (a typed `sfreq` prefix the child reads,
/// the meta as an opaque blob it echoes unchanged, then the array layout; NO GOOF magic —
/// parent+child are co-versioned) that runs the user's `process(x)` over **iceoryx2 shared
/// memory**. `sfreq` is exposed to the node via a module-level `goofi_meta` dict (a PSD-class
/// node reads it) and the input array is a read-only VIEW into the sample. Needs `iceoryx2` +
/// `numpy`. Service names arrive via `GOOFI_IOX_REQ`/`GOOFI_IOX_RESP`; the 4-byte request
/// sequence is echoed so the parent can dedup re-publishes.
const WORKER_SRC: &str = r#"
import sys, os, struct, ctypes, time
import numpy as np
import iceoryx2 as iox2

MAX_PAYLOAD = 64 * 1024
# Transport body: [f64 sfreq (NaN=none)][u32 meta_len][meta blob][u8 ndim][u8 dtype_len]
# [dtype][ndim x u32 shape][raw bytes]. The meta blob is OPAQUE here (a goofi_codec-encoded map) —
# the child echoes it unchanged so channels/index survive; only `sfreq` is read (typed prefix).
def decode_body(body):
    # `body` is a READ-ONLY memoryview over the shared-memory sample. The array is a numpy
    # VIEW straight into it (no copy); the sample is held for the duration of process(). A
    # node must not stash `x` across ticks — the buffer is released after the response is sent.
    sfreq = struct.unpack('<d', body[0:8])[0]
    ml = struct.unpack('<I', body[8:12])[0]
    meta = bytes(body[12:12 + ml])  # opaque; tiny (channel labels), echoed as-is
    off = 12 + ml
    ndim = body[off]; dl = body[off + 1]; off += 2
    dtype = bytes(body[off:off + dl]).decode(); off += dl
    shape = [struct.unpack('<I', body[off + 4 * i:off + 4 * i + 4])[0] for i in range(ndim)]
    off += 4 * ndim
    arr = np.frombuffer(body[off:], dtype=np.dtype(dtype)).reshape(shape)
    return arr, meta, (None if sfreq != sfreq else sfreq)

def encode_body(arr, meta, sfreq):
    arr = np.ascontiguousarray(arr)
    dtype = arr.dtype.str.encode()
    out = struct.pack('<d', float('nan') if sfreq is None else sfreq)
    out += struct.pack('<I', len(meta)) + bytes(meta)
    out += bytes([arr.ndim, len(dtype)]) + dtype
    for d in arr.shape:
        out += struct.pack('<I', d)
    out += arr.tobytes()
    return out

# iceoryx2 byte-slice service open (mirrors the Python backend's transport.py config so the
# Rust `[u8]` publisher/subscriber on the same service are ABI-compatible).
_node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

def _byte_service(name):
    return (_node.service_builder(iox2.ServiceName.new(name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .enable_safe_overflow(True)
            .max_publishers(1)
            .max_subscribers(16)
            .open_or_create())

_req_sub = _byte_service(os.environ['GOOFI_IOX_REQ']).subscriber_builder().create()
_resp_pub = (_byte_service(os.environ['GOOFI_IOX_RESP']).publisher_builder()
             .initial_max_slice_len(MAX_PAYLOAD)
             .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
             .create())

def _view(s):
    p = s.payload()
    n = p.number_of_elements
    # A read-only memoryview over the sample's shared memory — numpy frombuffer views it in
    # place. `.cast('B')` normalizes the ctypes format ('<B') to plain unsigned bytes, which
    # `bytes()` / `np.frombuffer` accept (a '<B' memoryview raises "unsupported format").
    return memoryview((ctypes.c_uint8 * n).from_address(p.data_ptr)).toreadonly().cast('B')

def _latest():
    latest = None
    while True:
        s = _req_sub.receive()
        if s is None:
            break
        latest = s
    return latest

def _send(payload):
    loan = _resp_pub.loan_slice_uninit(len(payload))
    ctypes.memmove(loan.payload_ptr, bytes(payload), len(payload))
    loan.assume_init().send()

# Point fd 1 (and Python-level stdout) at stderr BEFORE importing the user module, so any
# node/C-extension write to stdout is harmless (there is no stdout protocol anymore).
os.dup2(2, 1)
sys.stdout = sys.stderr

ns = {}
exec(os.environ['GOOFI_USER_SRC'], ns)
process = ns['process']

last_seq = None
while True:
    s = None  # drop the prior input sample before draining (bounds borrowed samples)
    s = _latest()
    if s is None:
        time.sleep(0.0005)  # idle poll; a request wakes it within ~0.5 ms
        continue
    mv = _view(s)
    seq = struct.unpack('<I', mv[0:4])[0]
    if seq == last_seq:
        continue  # a re-publish of an already-answered request — its response is in the buffer
    arr, meta_bytes, sfreq = decode_body(mv[4:])
    ns['goofi_meta'] = {'sfreq': sfreq}  # sfreq exposed to the node (read only if it needs it)
    # Preserve the output shape (do NOT ravel — that would flatten [C,T] channel data). res is
    # materialized + copied into the response before `s` is released next iteration, so a node
    # that returns the input view is still safe.
    res = np.ascontiguousarray(np.asarray(process(arr), dtype=arr.dtype))
    # Echo the opaque meta (channels/index) only when the shape is preserved — a length-changing
    # node's carried channels/index are stale. sfreq rides the typed prefix regardless.
    out_meta = meta_bytes if res.shape == arr.shape else b''
    _send(struct.pack('<I', seq) + encode_body(res, out_meta, sfreq))
    last_seq = seq
"#;

/// The iceoryx2 node + request publisher + response subscriber. Held directly on [`Running`]
/// because `ipc_threadsafe::Service` makes the ports `Send` — the previous design pushed them onto a
/// dedicated io thread solely to confine the `!Send` single-threaded (`ipc::Service`) ports. The node
/// must outlive the ports, so it rides along.
struct Ports {
    _node: iceoryx2::node::Node<ipc_threadsafe::Service>,
    req_pub: BytePublisher,
    resp_sub: ByteSubscriber,
}

/// The spawned child plus the iceoryx2 ports it talks over. `roundtrip` publishes a frame and polls
/// for the matching-sequence response inline (bounded by `timeout`), so a dead/hung child never
/// strands the caller — the port-owning io thread is gone.
struct Running {
    child: Child,
    ports: Ports,
    seq: u32,
}

/// Build the iceoryx2 node + `<id>_req` publisher + `<id>_resp` subscriber.
fn build_ports(req_name: &str, resp_name: &str) -> std::result::Result<Ports, String> {
    let node = NodeBuilder::new()
        .create::<ipc_threadsafe::Service>()
        .map_err(|e| format!("iox node: {e}"))?;
    let mk_pubsub = |name: &str| {
        node.service_builder(&name.try_into().map_err(|e| format!("bad service name `{name}`: {e:?}"))?)
            .publish_subscribe::<[u8]>()
            .enable_safe_overflow(true)
            .max_publishers(1)
            .max_subscribers(16)
            .open_or_create()
            .map_err(|e| format!("service `{name}`: {e}"))
    };
    let req_pub = mk_pubsub(req_name)?
        .publisher_builder()
        .initial_max_slice_len(MAX_PAYLOAD)
        .allocation_strategy(AllocationStrategy::PowerOfTwo)
        .create()
        .map_err(|e| format!("req publisher: {e}"))?;
    let resp_sub = mk_pubsub(resp_name)?
        .subscriber_builder()
        .create()
        .map_err(|e| format!("resp subscriber: {e}"))?;
    Ok(Ports { _node: node, req_pub, resp_sub })
}

impl Running {
    fn spawn(python: &str, source: &str) -> std::result::Result<Running, String> {
        let id = format!("goofi_sub_{}_{}", std::process::id(), SUBPROC_SEQ.fetch_add(1, Ordering::Relaxed));
        let req_name = format!("{id}_req");
        let resp_name = format!("{id}_resp");
        // No stdin/stdout protocol anymore — the child talks over iceoryx2. stdout/stderr are
        // inherited so node prints/tracebacks surface (the worker routes fd 1 to stderr).
        let mut child = Command::new(python)
            .arg("-c")
            .arg(WORKER_SRC)
            .env("GOOFI_USER_SRC", source)
            .env("GOOFI_IOX_REQ", &req_name)
            .env("GOOFI_IOX_RESP", &resp_name)
            // A subprocess node runs its OWN interpreter; a host `PYTHONPATH` (e.g. the pyo3/FT
            // tier's, injected by `.cargo/config.toml`) must not leak in and shadow the child's
            // numpy with an incompatible build. The child uses its interpreter's site-packages.
            .env_remove("PYTHONPATH")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn `{python}`: {e}"))?;
        match build_ports(&req_name, &resp_name) {
            Ok(ports) => Ok(Running { child, ports, seq: 0 }),
            // A port-setup failure would strand the child with no peer — reap it before erroring.
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(e)
            }
        }
    }

    /// Publish one frame and wait (bounded by `timeout`) for the matching-sequence response. On any
    /// error the caller ([`RemoteNode::process`]) reaps the child so the next tick respawns a fresh one.
    fn roundtrip(&mut self, frame: &[u8], timeout: Duration) -> std::result::Result<Vec<u8>, String> {
        self.seq = self.seq.wrapping_add(1);
        one_roundtrip(&self.ports.req_pub, &self.ports.resp_sub, self.seq, frame, timeout)
            .map_err(|e| format!("subprocess io: {e}"))
    }

    /// Kill + reap the child. The ports drop with `self`.
    fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

type BytePublisher = iceoryx2::port::publisher::Publisher<ipc_threadsafe::Service, [u8], ()>;
type ByteSubscriber = iceoryx2::port::subscriber::Subscriber<ipc_threadsafe::Service, [u8], ()>;

/// One request/response: publish `[seq][frame]` and poll `<resp>` for the sample whose leading
/// sequence matches. Re-publishes each idle millisecond (so the child gets it even if its
/// subscriber was still connecting when we first published) and drops any stale/mismatched
/// sample. Bounded by `timeout`.
fn one_roundtrip(
    req_pub: &BytePublisher,
    resp_sub: &ByteSubscriber,
    seq: u32,
    frame: &[u8],
    timeout: Duration,
) -> std::io::Result<Vec<u8>> {
    // Discard any stale response left from a prior tick before we start.
    while matches!(resp_sub.receive(), Ok(Some(_))) {}

    let mut msg = Vec::with_capacity(4 + frame.len());
    msg.extend_from_slice(&seq.to_le_bytes());
    msg.extend_from_slice(frame);

    let deadline = Instant::now() + timeout;
    loop {
        match req_pub.loan_slice_uninit(msg.len()) {
            Ok(sample) => {
                let _ = sample.write_from_slice(msg.as_slice()).send();
            }
            Err(e) => return Err(std::io::Error::other(format!("iox publish: {e}"))),
        }
        loop {
            match resp_sub.receive() {
                Ok(Some(sample)) => {
                    let payload = sample.payload();
                    if payload.len() >= 4
                        && u32::from_le_bytes(payload[0..4].try_into().unwrap()) == seq
                    {
                        return Ok(payload[4..].to_vec());
                    }
                    // A mismatched (stale) sample — keep draining this batch.
                }
                Ok(None) => break, // drained; re-publish + wait
                Err(e) => return Err(std::io::Error::other(format!("iox receive: {e}"))),
            }
        }
        if Instant::now() >= deadline {
            return Err(std::io::Error::other("subprocess did not respond in time"));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A Python node running in an isolated GIL subprocess. Construction is cheap and
/// infallible ([`RemoteNode::new`]); the subprocess is spawned lazily on the first
/// `process` (so a discovery factory never panics, and a spawn failure surfaces on
/// the node's error channel instead of crashing the graph).
pub struct RemoteNode {
    python: String,
    source: String,
    timeout: Duration,
    proc: Option<Running>,
}

impl RemoteNode {
    /// A remote node backed by `python` running `source` (defining
    /// `process(x) -> array-like`). No process spawns until the first tick.
    pub fn new(python: impl Into<String>, source: impl Into<String>) -> RemoteNode {
        RemoteNode {
            python: python.into(),
            source: source.into(),
            timeout: DEFAULT_TIMEOUT,
            proc: None,
        }
    }

    /// Override the per-tick response timeout (builder; mainly for tests/config).
    pub fn with_timeout(mut self, timeout: Duration) -> RemoteNode {
        self.timeout = timeout;
        self
    }

    /// Eagerly spawn (convenience for direct use / tests). Returns the spawn error.
    pub fn spawn(python: &str, source: &str) -> std::io::Result<RemoteNode> {
        let mut node = RemoteNode::new(python, source);
        node.ensure().map_err(std::io::Error::other)?;
        Ok(node)
    }

    fn ensure(&mut self) -> std::result::Result<&mut Running, String> {
        if self.proc.is_none() {
            self.proc = Some(Running::spawn(&self.python, &self.source)?);
        }
        Ok(self.proc.as_mut().unwrap())
    }

    /// Kill + reap the current child (if any) so the next tick respawns a fresh one.
    fn reset(&mut self) {
        if let Some(mut p) = self.proc.take() {
            p.shutdown();
        }
    }
}

impl Node for RemoteNode {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &goofi_node::Params<'_>) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let frame = encode_body(d);
        let timeout = self.timeout;
        // A dead/hung child (io error or timeout) is reaped so the NEXT tick spawns
        // a fresh subprocess, instead of leaving a zombie and erroring forever.
        let resp = match self.ensure().and_then(|r| r.roundtrip(&frame, timeout)) {
            Ok(r) => r,
            Err(e) => {
                self.reset();
                return Err(e.into());
            }
        };
        let data = decode_body(&resp)?;
        out.set("out", data);
        Ok(())
    }
}

impl Drop for RemoteNode {
    fn drop(&mut self) {
        self.reset();
    }
}

// ---------------------------------------------------------------------------
// GIL gate — the introspection authority that decides whether a Python node is
// safe to host in-process (free-threaded) or must be quarantined to a subprocess.
// ---------------------------------------------------------------------------

/// Probe whether running `source` (a node's module-level code — chiefly its
/// imports) leaves the interpreter's GIL DISABLED.
///
/// The probe runs in an ISOLATED subprocess on purpose: importing a C-extension
/// that lacks `Py_MOD_GIL_NOT_USED` silently re-enables the GIL *process-wide*
/// (the free-threaded footgun), so the check must never touch the host
/// interpreter that other in-process nodes share.
///
/// Returns `true` when `python` is free-threaded and the source's imports left
/// the GIL disabled — i.e. the node is safe to run in-process. Returns `false`
/// when the GIL is (or became) enabled, meaning the node must be routed to the
/// subprocess tier. A non-free-threaded interpreter (no `sys._is_gil_enabled`)
/// naturally reports enabled → `false`.
pub fn gil_safe(python: &str, source: &str) -> std::io::Result<bool> {
    const PROBE: &str = r#"
import sys, os
try:
    exec(compile(os.environ.get('GOOFI_PROBE_SRC', ''), '<node>', 'exec'), {})
except Exception:
    pass  # an import *error* is a separate concern; here we only judge GIL state
enabled = sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else True
sys.stdout.write('0' if not enabled else '1')
sys.stdout.flush()
"#;
    let out = Command::new(python)
        .arg("-c")
        .arg(PROBE)
        .env("GOOFI_PROBE_SRC", source)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()?;
    // Exactly "0" means the GIL stayed disabled → safe. Anything else (enabled,
    // empty output, a crash) is treated as unsafe → quarantine to subprocess.
    Ok(out.status.success() && out.stdout.trim_ascii() == b"0")
}

// ---------------------------------------------------------------------------
// Discovery — turn a directory of `process(x)` files into subprocess node types
// the engine hosts via `register_dyn_type` (mirrors `goofi_py::discover`, but the
// factory spawns a RemoteNode instead of an in-process PyNode).
// ---------------------------------------------------------------------------

use goofi_node::discover::{camel, leak_process_manifest, NodeFactory};
use goofi_node::{Isolation, NodeManifest};

/// A discovered subprocess node type, ready to `register_dyn_type` into a Graph.
pub struct SubprocNodeType {
    pub manifest: &'static NodeManifest,
    /// Builds a node instance (spawns lazily on first tick).
    pub factory: NodeFactory,
}

/// Build a subprocess-backed node type from a single file (running on `python`),
/// or `None` if it is not a node file: non-`.py`, `_`-prefixed, unreadable, or it
/// lacks `process`. Used per-file by [`discover`] and by the CLI's GIL-gate
/// auto-router (which routes a node here when [`gil_safe`] judged it unsafe).
pub fn discover_one(path: &std::path::Path, python: &str) -> Option<SubprocNodeType> {
    if path.extension().and_then(|e| e.to_str()) != Some("py") {
        return None;
    }
    let stem = path.file_stem().and_then(|s| s.to_str())?;
    if stem.starts_with('_') {
        return None;
    }
    let source = std::fs::read_to_string(path).ok()?;
    // Cheap guard: must plausibly define `process` (a missing one would only fail
    // once the subprocess is spawned and asked to run).
    if !source.contains("def process") {
        return None;
    }

    let manifest = leak_process_manifest(
        camel(stem),
        format!("Subprocess Python node from {}", path.display()),
        "subprocess",
        Isolation::Subprocess,
    );
    let python = python.to_string();
    let factory: NodeFactory =
        Box::new(move |_p| Box::new(RemoteNode::new(&python, &source)) as Box<dyn Node>);
    Some(SubprocNodeType { manifest, factory })
}

/// Scan `dir` for `*.py` node files (skipping `_`-prefixed) that define
/// `process`, returning subprocess-backed types that run on `python`.
pub fn discover(dir: &std::path::Path, python: &str) -> std::io::Result<Vec<SubprocNodeType>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    Ok(entries.iter().filter_map(|e| discover_one(&e.path(), python)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Data, Meta, Value};
    use goofi_node::ParamGroups;
    use indexmap::IndexMap;

    #[test]
    fn transport_roundtrips_array_sfreq_index_and_channels() {
        let bytes: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let mut meta = Meta { sfreq: Some(250.0), index: Some(7), ..Default::default() };
        // Channels ride in the opaque meta blob (the regression this guards: the typed-only
        // transport dropped them, losing EEG channel labels across the boundary).
        meta.channels = goofi_core::Axes::new().with(
            0,
            goofi_core::Axis::coords(vec![
                goofi_core::Coord::Str("Fz".into()),
                goofi_core::Coord::Str("Cz".into()),
            ]),
        );
        let d = Data::array_f32(vec![2, 3], bytes, meta).unwrap();
        let back = decode_body(&encode_body(&d)).unwrap();
        assert_eq!(back.meta().sfreq, Some(250.0));
        assert_eq!(back.meta().index, Some(7));
        let ch = back.meta().channels.get(0).and_then(|a| a.coords.clone()).expect("dim0 channels survive");
        assert_eq!(ch.len(), 2);
        match back.value() {
            Value::Array(s) => {
                assert_eq!(s.shape(), &[2, 3]);
                assert_eq!(f32::from_le_bytes(s.as_bytes()[4..8].try_into().unwrap()), 1.0);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn transport_encodes_absent_sfreq_and_index_as_sentinels() {
        let d = Data::array_f32(vec![1], 3.5f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
        let back = decode_body(&encode_body(&d)).unwrap();
        assert_eq!(back.meta().sfreq, None, "NaN sentinel decodes to None");
        assert_eq!(back.meta().index, None, "u64::MAX sentinel decodes to None");
        match back.value() {
            Value::Array(s) => {
                assert_eq!(f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()), 3.5)
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn worker_src_views_the_input_without_a_copy_or_msgpack() {
        // The child reads the request sample in place (a read-only numpy view over SHM), with
        // no msgpack and no defensive input copy — the Part B zero-copy-oriented win.
        assert!(!WORKER_SRC.contains("msgpack"), "no msgpack in the child");
        assert!(WORKER_SRC.contains("toreadonly"), "the input is a read-only SHM view");
        assert!(!WORKER_SRC.contains(".copy()"), "no defensive input copy on the hot path");
        assert!(!WORKER_SRC.contains("_sample_bytes"), "the input is not copied into a Python bytes");
    }

    #[test]
    fn psd_runs_over_the_transport_reading_sfreq() {
        let py = require_python();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/psd.py");
        // Discovery: CamelCase "Psd" on the subprocess tier.
        let ty = discover_one(&path, &py).expect("psd.py discovers as a subprocess node");
        assert_eq!(ty.manifest.type_name, "Psd");
        assert_eq!(ty.manifest.isolation, Isolation::Subprocess);

        // A 1x64 unit sine at exactly 8 cycles (bin 8, independent of sfreq); sfreq=1000 in
        // meta only scales the PSD magnitude, so a small peak proves the child read sfreq.
        let n = 64usize;
        let sfreq = 1000.0f64;
        let samples: Vec<u8> = (0..n)
            .flat_map(|i| ((2.0 * std::f64::consts::PI * 8.0 * i as f64 / n as f64).sin() as f32).to_le_bytes())
            .collect();
        let meta = Meta { sfreq: Some(sfreq), ..Default::default() };
        let d = Data::array_f32(vec![1, n], samples, meta).unwrap();

        let src = std::fs::read_to_string(&path).unwrap();
        let mut node = RemoteNode::new(&py, &src);
        let out = run(&mut node, d);

        match out.value() {
            Value::Array(s) => {
                assert_eq!(s.shape(), &[1, n / 2 + 1], "channels preserved; rfft bins");
                let psd: Vec<f32> =
                    s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                let peak = (0..psd.len()).max_by(|a, b| psd[*a].total_cmp(&psd[*b])).unwrap();
                assert_eq!(peak, 8, "spectral peak at the input frequency bin");
                // sfreq=1000 normalization -> a small peak; the fallback sfreq=1 would be ~1000x larger.
                assert!(psd[peak] < 1.0, "peak {} implies sfreq=1000 reached the child, not the fallback", psd[peak]);
            }
            _ => panic!("expected array"),
        }
        // sfreq also rides back through the typed transport meta.
        assert_eq!(out.meta().sfreq, Some(sfreq), "sfreq round-trips through the typed transport");
    }

    /// A `RemoteNode` holds its iceoryx2 ports directly (via `ipc_threadsafe::Service`), so it must
    /// stay `Send` for the scheduler. This compile-time guard fails loudly if the ports ever revert to
    /// the `!Send` `ipc::Service` — the whole reason the port-owning io thread could be removed.
    #[test]
    fn remote_node_stays_send() {
        fn _assert_send<T: Send>() {}
        _assert_send::<RemoteNode>();
    }

    /// A python with BOTH numpy and iceoryx2 (the subprocess transport), or None. Prefers
    /// `$GOOFI_SUBPROC_TEST_PYTHON`, then the repo's iceoryx2 `.venv`, then a PATH python. The
    /// probe strips `PYTHONPATH` exactly like the real child spawn ([`Running::spawn`]), so a
    /// host/pyo3 `PYTHONPATH` can't produce a false negative (it once masked real bugs by making
    /// the `.venv` python import an incompatible numpy → every tier test silently SKIPPED).
    fn usable_python() -> Option<String> {
        let mut cands: Vec<String> = Vec::new();
        if let Ok(p) = std::env::var("GOOFI_SUBPROC_TEST_PYTHON") {
            cands.push(p);
        }
        cands.push(format!("{}/../../.venv/bin/python", env!("CARGO_MANIFEST_DIR")));
        cands.push("python3".to_string());
        cands.push("python".to_string());
        for cand in cands {
            if let Ok(out) = Command::new(&cand)
                .arg("-c")
                .arg("import numpy, iceoryx2")
                .env_remove("PYTHONPATH")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
            {
                if out.success() {
                    return Some(cand);
                }
            }
        }
        None
    }

    /// Like [`usable_python`] but PANICS with an actionable message when none is found — the
    /// subprocess tier tests HARD-REQUIRE a python (numpy + iceoryx2) rather than silently
    /// skipping, so a missing/misconfigured interpreter fails loudly instead of hiding bugs.
    fn require_python() -> String {
        usable_python().unwrap_or_else(|| {
            panic!(
                "no python with numpy + iceoryx2 found (checked $GOOFI_SUBPROC_TEST_PYTHON, \
                 ./.venv/bin/python, python3, python). Provision ./.venv or set \
                 GOOFI_SUBPROC_TEST_PYTHON. The subprocess-tier tests require one."
            )
        })
    }

    fn run(node: &mut RemoteNode, d: Data) -> Data {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        let params = ParamGroups::new();
        {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx, &goofi_node::Params::new(&params)).expect("remote process");
        }
        outmap.get("out").unwrap().clone().expect("output frame")
    }

    fn try_run(node: &mut RemoteNode, d: Data) -> Result<Data, String> {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        let params = ParamGroups::new();
        let r = {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx, &goofi_node::Params::new(&params))
        };
        r.map_err(|e| e.0)?;
        Ok(outmap.get("out").unwrap().clone().expect("output frame"))
    }

    fn floats(d: &Data) -> Vec<f32> {
        match d.value() {
            Value::Array(s) => s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => panic!("not array"),
        }
    }

    #[test]
    fn subprocess_roundtrip_latency_and_stability() {
        // Concrete latency/stability read for the iceoryx2 subprocess tier: after cold start,
        // run many round-trips of a realistic EEG-sized frame and report the latency
        // distribution. Loose upper bound so it isn't CI-timing-flaky; every tick must succeed
        // (stability) and steady-state latency must stay well under the tick budget.
        let py = require_python();
        let mut node = RemoteNode::spawn(&py, "def process(x):\n    return x * 1.0\n").unwrap();
        // A 32-channel × 256-sample float32 frame (~32 KB) — a typical EEG buffer.
        let (c, t) = (32usize, 256usize);
        let buf: Vec<u8> = (0..c * t).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let make = || Data::array_f32(vec![c, t], buf.clone(), Meta::empty()).unwrap();

        // Warm up: the first tick pays cold start (spawn + numpy/iceoryx2 import + service open).
        let cold = std::time::Instant::now();
        let _ = run(&mut node, make());
        let cold_ms = cold.elapsed().as_secs_f64() * 1e3;

        let iters = 300usize;
        let mut lat: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            let out = run(&mut node, make());
            lat.push(t0.elapsed().as_secs_f64() * 1e3);
            assert_eq!(floats(&out).len(), c * t, "every tick round-trips intact (stability)");
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = lat.iter().sum::<f64>() / iters as f64;
        let p = |q: f64| lat[((iters as f64 * q) as usize).min(iters - 1)];
        eprintln!(
            "subproc iceoryx2 latency (32x256 f32, {iters} ticks): cold={cold_ms:.1}ms  \
             min={:.3}ms  p50={:.3}ms  p99={:.3}ms  max={:.3}ms  mean={mean:.3}ms",
            lat[0], p(0.50), p(0.99), lat[iters - 1]
        );
        // Steady-state p99 must be a small fraction of a 60 Hz tick (16.6 ms) — a generous
        // ceiling that still catches a regression to blocking/second-scale latency.
        assert!(p(0.99) < 10.0, "p99 round-trip {:.3}ms exceeds the budget", p(0.99));
    }

    #[test]
    fn channels_preserved_on_2d_length_preserving_node() {
        // Audit #1: the canonical EEG case. A [2,3] array with dim0 channel labels
        // through a length-preserving node must come back [2,3] with channels
        // intact — the old `.ravel()` flattened it to [6] and the decoder rejected
        // the frame (channels len 2 != shape 6).
        let py = require_python();
        let mut node = RemoteNode::spawn(&py, "def process(x):\n    return x * 2.0\n").unwrap();

        let mut meta = Meta::empty();
        meta.channels = goofi_core::Axes::new().with(
            0,
            goofi_core::Axis::coords(vec![
                goofi_core::Coord::Str("Fz".into()),
                goofi_core::Coord::Str("Cz".into()),
            ]),
        );
        let buf: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let d = Data::array_f32(vec![2, 3], buf, meta).unwrap();

        let out = try_run(&mut node, d).expect("2-D channel frame must round-trip");
        match out.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[2, 3], "shape must be preserved, not raveled"),
            _ => panic!("expected array"),
        }
        assert_eq!(floats(&out), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let ch = out.meta().channels.get(0).and_then(|a| a.coords.clone()).expect("dim0 channels preserved");
        assert_eq!(ch.len(), 2);
    }

    #[test]
    fn node_stdout_does_not_corrupt_the_protocol_frame() {
        // The protocol rides fd 1. A node that writes to stdout (here a flushed
        // print, but equally a C-extension's printf — the very kind this tier
        // hosts) must NOT inject bytes into the length-prefixed frame stream.
        let py = require_python();
        let mut node = RemoteNode::spawn(
            &py,
            "def process(x):\n    import sys\n    print('debug from the node', flush=True)\n    sys.stdout.flush()\n    return x * 2.0\n",
        )
        .unwrap();
        let d = Data::array_f32(vec![3], (0..3).flat_map(|i| (i as f32).to_le_bytes()).collect(), Meta::empty()).unwrap();
        let out = try_run(&mut node, d).expect("a node that prints must still round-trip");
        assert_eq!(floats(&out), vec![0.0, 2.0, 4.0]);
        // A second tick proves the stream stayed in sync (not just the first frame).
        let d2 = Data::array_f32(vec![2], [5.0f32, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect(), Meta::empty()).unwrap();
        assert_eq!(floats(&try_run(&mut node, d2).expect("second tick in sync")), vec![10.0, 12.0]);
    }

    #[test]
    fn crashed_child_is_reaped_and_respawns() {
        // Audit #2: a tick whose worker raises must Err, then a later tick must
        // succeed (a fresh subprocess is spawned) rather than being wedged forever.
        let py = require_python();
        let mut node = RemoteNode::spawn(
            &py,
            "def process(x):\n    if x[0] < 0:\n        raise ValueError('boom')\n    return x * 2.0\n",
        )
        .unwrap();

        let bad = Data::array_f32(vec![1], (-1.0f32).to_le_bytes().to_vec(), Meta::empty()).unwrap();
        assert!(try_run(&mut node, bad).is_err(), "worker raise must surface as an error");

        let good = Data::array_f32(vec![1], 3.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
        let out = try_run(&mut node, good).expect("must respawn and succeed on the next tick");
        assert_eq!(floats(&out), vec![6.0]);
    }

    #[test]
    fn hung_subprocess_times_out_instead_of_hanging() {
        // Audit #4: a worker that never responds must not block forever; with a
        // short timeout the tick returns an error promptly.
        let py = require_python();
        let mut node = RemoteNode::new(
            &py,
            "import time\ndef process(x):\n    while True:\n        time.sleep(1)\n",
        )
        .with_timeout(Duration::from_millis(600));

        let d = Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
        let t = std::time::Instant::now();
        let r = try_run(&mut node, d);
        assert!(r.is_err(), "a hung subprocess must error, not hang");
        assert!(t.elapsed() < Duration::from_secs(5), "must return near the timeout, not block");
    }

    #[test]
    fn large_frame_into_a_non_draining_child_times_out() {
        // Audit R2-#1: the worker blocks at module import (never enters its read
        // loop) AND the frame exceeds the OS pipe buffer (~64 KiB), so the WRITE
        // would block forever without a write-side timeout. Must error, not hang.
        let py = require_python();
        let mut node = RemoteNode::new(
            &py,
            "import time\ntime.sleep(30)\ndef process(x):\n    return x\n",
        )
        .with_timeout(Duration::from_millis(600));

        let n = 40_000usize; // 160 KB >> the 64 KiB initial slice — a big frame to a stuck child
        let buf: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let d = Data::array_f32(vec![n], buf, Meta::empty()).unwrap();

        let t = std::time::Instant::now();
        let r = try_run(&mut node, d);
        assert!(r.is_err(), "a child stuck before the loop must error, not hang");
        assert!(t.elapsed() < Duration::from_secs(5), "must return near the timeout");
    }

    #[test]
    fn large_frame_round_trips_over_shared_memory() {
        // A frame far larger than the 64 KiB initial slice must round-trip — iceoryx2 grows the
        // publisher's segment (PowerOfTwo), and the 4-byte sequence framing survives a big body.
        let py = require_python();
        let mut node = RemoteNode::spawn(&py, "def process(x):\n    return x * 2.0\n").unwrap();
        let n = 100_000usize; // 400 KB body
        let buf: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let d = Data::array_f32(vec![n], buf, Meta::empty()).unwrap();
        let out = run(&mut node, d);
        let vals = floats(&out);
        assert_eq!(vals.len(), n, "shape preserved across the SHM round-trip");
        assert_eq!(vals[1], 2.0, "1 * 2");
        assert_eq!(vals[10], 20.0, "10 * 2");
        assert_eq!(vals[n - 1], (n - 1) as f32 * 2.0, "last element doubled");
    }

    #[test]
    fn remote_node_runs_a_python_node_in_a_subprocess() {
        let py = require_python();
        let mut node =
            RemoteNode::spawn(&py, "def process(x):\n    return x * 2.0 + 1.0\n").unwrap();

        // Input carries sfreq/index; the worker passes meta through opaquely.
        let mut meta = Meta::empty();
        meta.sfreq = Some(128.0);
        meta.index = Some(7);
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = Data::array_f32(vec![3], buf, meta).unwrap();

        // Two ticks on the same long-lived subprocess (proves it stays alive).
        let out1 = run(&mut node, d.clone());
        assert_eq!(floats(&out1), vec![3.0, 5.0, 7.0]);
        let out2 = run(&mut node, d);
        assert_eq!(floats(&out2), vec![3.0, 5.0, 7.0]);

        // Meta (sfreq/index) survived the opaque round-trip; dtype stayed f32.
        assert_eq!(out1.meta().sfreq, Some(128.0));
        assert_eq!(out1.meta().index, Some(7));
        match out1.value() {
            Value::Array(s) => {
                assert_eq!(s.shape(), &[3]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn gil_safe_distinguishes_free_threaded_from_gil_interpreters() {
        // A normal (GIL) interpreter must be judged UNSAFE for in-process hosting —
        // it either reports the GIL enabled or lacks sys._is_gil_enabled entirely.
        // (python3.12 has numpy here but no _is_gil_enabled → enabled → unsafe.)
        if let Some(py) = usable_python() {
            assert!(
                !gil_safe(&py, "import numpy").unwrap(),
                "a GIL interpreter must be judged unsafe for in-process hosting"
            );
        }
        // A free-threaded interpreter importing an FT-safe dep (numpy 2.x on 3.14t)
        // keeps the GIL disabled → SAFE. Provided via env when available.
        if let Ok(ft) = std::env::var("GOOFI_FT_PYTHON") {
            assert!(
                gil_safe(&ft, "import numpy").unwrap(),
                "free-threaded interpreter + FT-safe deps must be judged safe"
            );
            // A bare source (no imports) is trivially safe on a free-threaded build.
            assert!(gil_safe(&ft, "x = 1\n").unwrap());
        } else {
            eprintln!("NOTE: set GOOFI_FT_PYTHON to also exercise the free-threaded branch");
        }
    }

    // `camel` is now the one shared goofi_node::discover fn both backends re-export, so the
    // in-process/subprocess type-name parity is guaranteed by construction (unit-tested in goofi-node).

    #[test]
    fn discover_yields_subprocess_types_that_run() {
        let py = require_python();
        let dir = std::env::temp_dir().join(format!("goofi_subdisc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("negate.py"), "def process(x):\n    return -x\n").unwrap();
        std::fs::write(dir.join("_hidden.py"), "def process(x):\n    return x\n").unwrap();
        std::fs::write(dir.join("nope.py"), "x = 1\n").unwrap();

        let types = discover(&dir, &py).unwrap();
        let names: Vec<&str> = types.iter().map(|t| t.manifest.type_name).collect();
        assert_eq!(names, vec!["Negate"]);
        assert_eq!(types[0].manifest.category, "subprocess");
        assert_eq!(types[0].manifest.isolation, Isolation::Subprocess);

        // The factory builds a working node.
        let mut node = (types[0].factory)(&ParamGroups::new());
        let buf: Vec<u8> = [1.0f32, -2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = Data::array_f32(vec![2], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        let params = ParamGroups::new();
        {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx, &goofi_node::Params::new(&params)).unwrap();
        }
        let got = floats(outmap.get("out").unwrap().as_ref().unwrap());
        assert_eq!(got, vec![-1.0, 2.0]);
        drop(node); // teardown now runs via Drop (was `terminate`)

        let _ = std::fs::remove_dir_all(&dir);
    }
}
