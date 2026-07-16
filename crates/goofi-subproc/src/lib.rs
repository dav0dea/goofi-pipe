//! goofi-subproc — the subprocess node tier (Pathway C).
//!
//! A [`RemoteNode`] runs a Python node in an isolated **GIL** interpreter, one
//! process per node. It exists for two reasons: (1) deps that aren't
//! free-threading-safe can't run in the in-process pyo3 host; (2) per the latency
//! finding, a *separate interpreter* has its own object ownership, so parallel
//! heavy-Python compute avoids free-threaded CPython's biased-refcount penalty.
//!
//! Each tick is one request/response: the input `Data` is GOOF-encoded
//! ([`goofi_codec::encode`]), length-prefixed, and written to the child's stdin;
//! the child runs the user `process(x)` and writes back a GOOF frame we decode
//! ([`goofi_codec::decode`]). Transport is pipes in this first cut; iceoryx2 SHM
//! is the intended production transport (the protocol is transport-agnostic).
//!
//! The node implements the same [`Node`] trait as native and in-process Python
//! nodes, so the scheduler never branches on backend and the engine hosts it
//! through the ordinary `register_dyn_type` seam.

use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::Duration;

use goofi_node::{Inputs, Node, NodeCtx, NodeResult, Outputs};

/// Default cap on how long a tick waits for a subprocess response before treating
/// the child as hung. Generous enough to cover cold start (spawn + numpy import);
/// a hung child is killed and surfaces as a node error rather than stranding the
/// scheduler (and, in the bridge, the graph mutex) indefinitely.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// The Python worker: a self-contained GOOF-array codec (meta passed through
/// opaquely — the Rust decoder re-derives shape/dtype from the body) that runs
/// the user's `process(x)` supplied via the `GOOFI_USER_SRC` env var. Needs only
/// numpy (no msgpack — meta bytes are never parsed here).
const WORKER_SRC: &str = r#"
import sys, os, struct
import numpy as np

MAGIC = b'GOOF'; VER = 2

def read_exact(f, n):
    b = b''
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            return None
        b += c
    return b

def decode_array(frame):
    assert frame[:4] == MAGIC and frame[4] == VER and frame[5] == 0, "not a GOOF array frame"
    ml = struct.unpack('<I', frame[6:10])[0]
    bl = struct.unpack('<I', frame[10:14])[0]
    meta = frame[14:14 + ml]
    body = frame[14 + ml:14 + ml + bl]
    ndim = body[0]; dl = body[1]; off = 2
    dtype = body[off:off + dl].decode(); off += dl
    shape = [struct.unpack('<I', body[off + 4 * i:off + 4 * i + 4])[0] for i in range(ndim)]
    off += 4 * ndim
    arr = np.frombuffer(body[off:], dtype=np.dtype(dtype)).reshape(shape).copy()
    return arr, meta

def encode_array(arr, meta):
    arr = np.ascontiguousarray(arr)
    dtype = arr.dtype.str.encode()
    body = bytes([arr.ndim, len(dtype)]) + dtype
    for d in arr.shape:
        body += struct.pack('<I', d)
    body += arr.tobytes()
    hdr = MAGIC + bytes([VER, 0]) + struct.pack('<I', len(meta)) + struct.pack('<I', len(body))
    return hdr + meta + body

ns = {}
exec(os.environ['GOOFI_USER_SRC'], ns)
process = ns['process']

inp = sys.stdin.buffer
outp = sys.stdout.buffer
while True:
    hdr = read_exact(inp, 4)
    if hdr is None:
        break
    n = struct.unpack('<I', hdr)[0]
    frame = read_exact(inp, n)
    arr, meta = decode_array(frame)
    # Preserve the output shape (do NOT ravel — that would flatten [C,T] channel
    # data to [C*T] and, with the carried channels meta, fail the decoder's
    # channel-length check on every tick).
    res = np.ascontiguousarray(np.asarray(process(arr), dtype=arr.dtype))
    # The carried meta describes the INPUT (shape + channel coords). If the node
    # changed the shape, that meta is stale (its channels would mismatch the new
    # shape and the decoder would reject the frame), so drop it; when the shape is
    # unchanged the meta (sfreq/index/channels) stays valid and rides through.
    out_meta = meta if res.shape == arr.shape else b''
    out = encode_array(res, out_meta)
    outp.write(struct.pack('<I', len(out)))
    outp.write(out)
    outp.flush()
"#;

/// The spawned child plus a single **io thread** that owns both pipes and does
/// the blocking write+read for each request. `roundtrip` hands it a frame over a
/// channel and waits for the response with `recv_timeout`, so BOTH a stuck write
/// (a large frame into a child that isn't draining stdin) and a stuck read are
/// bounded — a timeout kills the child, which errors whichever syscall the io
/// thread is blocked in, and it exits.
struct Running {
    child: Child,
    tx_req: Option<std::sync::mpsc::Sender<Vec<u8>>>,
    rx_resp: Receiver<std::io::Result<Vec<u8>>>,
    io_thread: Option<JoinHandle<()>>,
}

impl Running {
    fn spawn(python: &str, source: &str) -> std::result::Result<Running, String> {
        let mut child = Command::new(python)
            .arg("-c")
            .arg(WORKER_SRC)
            .env("GOOFI_USER_SRC", source)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn `{python}`: {e}"))?;
        let mut stdin = child.stdin.take().expect("piped stdin");
        let mut stdout = child.stdout.take().expect("piped stdout");
        let (tx_req, rx_req) = std::sync::mpsc::channel::<Vec<u8>>();
        let (tx_resp, rx_resp) = std::sync::mpsc::channel::<std::io::Result<Vec<u8>>>();
        // One request -> write it -> read one response, in lockstep. Exits when
        // the request channel closes (shutdown drops tx_req) or a pipe syscall
        // errors (the child died / was killed) — so it never strands.
        let io_thread = std::thread::spawn(move || {
            while let Ok(frame) = rx_req.recv() {
                let w = stdin
                    .write_all(&(frame.len() as u32).to_le_bytes())
                    .and_then(|_| stdin.write_all(&frame))
                    .and_then(|_| stdin.flush());
                if let Err(e) = w {
                    let _ = tx_resp.send(Err(e));
                    break;
                }
                let mut lenb = [0u8; 4];
                if let Err(e) = stdout.read_exact(&mut lenb) {
                    let _ = tx_resp.send(Err(e));
                    break;
                }
                let n = u32::from_le_bytes(lenb) as usize;
                let mut buf = vec![0u8; n];
                match stdout.read_exact(&mut buf) {
                    Ok(()) => {
                        if tx_resp.send(Ok(buf)).is_err() {
                            break; // receiver gone
                        }
                    }
                    Err(e) => {
                        let _ = tx_resp.send(Err(e));
                        break;
                    }
                }
            }
        });
        Ok(Running {
            child,
            tx_req: Some(tx_req),
            rx_resp,
            io_thread: Some(io_thread),
        })
    }

    /// Hand one frame to the io thread; wait up to `timeout` for the response.
    /// A timeout kills the child (unblocking a stuck write OR read) and errors.
    fn roundtrip(&mut self, frame: &[u8], timeout: Duration) -> std::result::Result<Vec<u8>, String> {
        match self.tx_req.as_ref() {
            Some(tx) if tx.send(frame.to_vec()).is_ok() => {}
            _ => return Err("subprocess io thread ended".into()),
        }
        match self.rx_resp.recv_timeout(timeout) {
            Ok(Ok(buf)) => Ok(buf),
            Ok(Err(e)) => Err(format!("subprocess io: {e}")),
            Err(RecvTimeoutError::Timeout) => {
                let _ = self.child.kill();
                Err(format!("subprocess did not respond within {timeout:?}"))
            }
            Err(RecvTimeoutError::Disconnected) => Err("subprocess io thread ended".into()),
        }
    }

    /// Kill + reap the child, close the request channel (so an idle io thread
    /// wakes), and join the io thread.
    fn shutdown(&mut self) {
        let _ = self.child.kill();
        self.tx_req = None; // wakes an io thread blocked in rx_req.recv()
        let _ = self.child.wait();
        if let Some(h) = self.io_thread.take() {
            let _ = h.join();
        }
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
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let frame = goofi_codec::encode(d);
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
        let data = goofi_codec::decode(&resp)?;
        out.set("out", data);
        Ok(())
    }

    fn terminate(&mut self) {
        self.reset();
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

use goofi_node::{Isolation, NodeManifest, OutputDecl, ParamGroups, SlotDecl};

static PY_IN: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: goofi_core::SlotType::Array,
    trigger_process: true,
}];
static PY_OUT: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: goofi_core::SlotType::Array,
}];
fn sp_params() -> ParamGroups {
    ParamGroups::new()
}
fn sp_stub_make(_: &ParamGroups) -> Box<dyn Node> {
    unreachable!("a discovered subprocess node is built by its factory")
}

/// Builds a node instance (spawns lazily on first tick).
pub type SubprocFactory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// A discovered subprocess node type, ready to `register_dyn_type` into a Graph.
pub struct SubprocNodeType {
    pub manifest: &'static NodeManifest,
    pub factory: SubprocFactory,
}

/// `snake_case` file stem -> `CamelCase` type name (matches `goofi_py::discover`
/// so the same file yields the same type name whichever backend hosts it).
fn camel(stem: &str) -> String {
    stem.split('_')
        .filter(|s| !s.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
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

    let type_name: &'static str = Box::leak(camel(stem).into_boxed_str());
    let doc: &'static str =
        Box::leak(format!("Subprocess Python node from {}", path.display()).into_boxed_str());
    let manifest: &'static NodeManifest = Box::leak(Box::new(NodeManifest {
        type_name,
        category: "subprocess",
        doc,
        inputs: PY_IN,
        outputs: PY_OUT,
        default_params: sp_params,
        isolation: Isolation::Subprocess,
        make: sp_stub_make,
    }));
    let python = python.to_string();
    let factory: SubprocFactory =
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
    use goofi_core::{DType, Data, Meta, Value};
    use indexmap::IndexMap;

    /// A python3 with numpy, or None (the subprocess tier test is skipped then).
    fn usable_python() -> Option<String> {
        for cand in ["python3", "python"] {
            if let Ok(out) = Command::new(cand)
                .arg("-c")
                .arg("import numpy")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
            {
                if out.success() {
                    return Some(cand.to_string());
                }
            }
        }
        None
    }

    fn run(node: &mut RemoteNode, d: Data) -> Data {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx).expect("remote process");
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
        let r = {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx)
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
    fn channels_preserved_on_2d_length_preserving_node() {
        // Audit #1: the canonical EEG case. A [2,3] array with dim0 channel labels
        // through a length-preserving node must come back [2,3] with channels
        // intact — the old `.ravel()` flattened it to [6] and the decoder rejected
        // the frame (channels len 2 != shape 6).
        let Some(py) = usable_python() else {
            eprintln!("SKIP: no python3 with numpy");
            return;
        };
        let mut node = RemoteNode::spawn(&py, "def process(x):\n    return x * 2.0\n").unwrap();

        let mut meta = Meta::empty();
        meta.channels.0.insert(
            0,
            std::sync::Arc::new(vec![
                goofi_core::Coord::Str("Fz".into()),
                goofi_core::Coord::Str("Cz".into()),
            ]),
        );
        let buf: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let d = Data::from_array_bytes(DType::F32, vec![2, 3], buf, meta).unwrap();

        let out = try_run(&mut node, d).expect("2-D channel frame must round-trip");
        match out.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[2, 3], "shape must be preserved, not raveled"),
            _ => panic!("expected array"),
        }
        assert_eq!(floats(&out), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let ch = out.meta().channels.0.get(&0).expect("dim0 channels preserved");
        assert_eq!(ch.len(), 2);
    }

    #[test]
    fn crashed_child_is_reaped_and_respawns() {
        // Audit #2: a tick whose worker raises must Err, then a later tick must
        // succeed (a fresh subprocess is spawned) rather than being wedged forever.
        let Some(py) = usable_python() else {
            eprintln!("SKIP: no python3 with numpy");
            return;
        };
        let mut node = RemoteNode::spawn(
            &py,
            "def process(x):\n    if x[0] < 0:\n        raise ValueError('boom')\n    return x * 2.0\n",
        )
        .unwrap();

        let bad = Data::from_array_bytes(DType::F32, vec![1], (-1.0f32).to_le_bytes().to_vec(), Meta::empty()).unwrap();
        assert!(try_run(&mut node, bad).is_err(), "worker raise must surface as an error");

        let good = Data::from_array_bytes(DType::F32, vec![1], 3.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
        let out = try_run(&mut node, good).expect("must respawn and succeed on the next tick");
        assert_eq!(floats(&out), vec![6.0]);
    }

    #[test]
    fn hung_subprocess_times_out_instead_of_hanging() {
        // Audit #4: a worker that never responds must not block forever; with a
        // short timeout the tick returns an error promptly.
        let Some(py) = usable_python() else {
            eprintln!("SKIP: no python3 with numpy");
            return;
        };
        let mut node = RemoteNode::new(
            &py,
            "import time\ndef process(x):\n    while True:\n        time.sleep(1)\n",
        )
        .with_timeout(Duration::from_millis(600));

        let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
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
        let Some(py) = usable_python() else {
            eprintln!("SKIP: no python3 with numpy");
            return;
        };
        let mut node = RemoteNode::new(
            &py,
            "import time\ntime.sleep(30)\ndef process(x):\n    return x\n",
        )
        .with_timeout(Duration::from_millis(600));

        let n = 40_000usize; // 160 KB >> 64 KiB pipe buffer
        let buf: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let d = Data::from_array_bytes(DType::F32, vec![n], buf, Meta::empty()).unwrap();

        let t = std::time::Instant::now();
        let r = try_run(&mut node, d);
        assert!(r.is_err(), "a stuck write must error, not hang");
        assert!(t.elapsed() < Duration::from_secs(5), "must return near the timeout");
    }

    #[test]
    fn remote_node_runs_a_python_node_in_a_subprocess() {
        let Some(py) = usable_python() else {
            eprintln!("SKIP: no python3 with numpy available");
            return;
        };
        let mut node =
            RemoteNode::spawn(&py, "def process(x):\n    return x * 2.0 + 1.0\n").unwrap();

        // Input carries sfreq/index; the worker passes meta through opaquely.
        let mut meta = Meta::empty();
        meta.sfreq = Some(128.0);
        meta.index = Some(7);
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = Data::from_array_bytes(DType::F32, vec![3], buf, meta).unwrap();

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
                assert_eq!(s.dtype(), DType::F32);
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

    #[test]
    fn camel_matches_the_inprocess_naming() {
        assert_eq!(camel("triple"), "Triple");
        assert_eq!(camel("my_band_filter"), "MyBandFilter");
    }

    #[test]
    fn discover_yields_subprocess_types_that_run() {
        let Some(py) = usable_python() else {
            eprintln!("SKIP: no python3 with numpy available");
            return;
        };
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
        let d = Data::from_array_bytes(DType::F32, vec![2], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let got = floats(outmap.get("out").unwrap().as_ref().unwrap());
        assert_eq!(got, vec![-1.0, 2.0]);
        node.terminate();

        let _ = std::fs::remove_dir_all(&dir);
    }
}
