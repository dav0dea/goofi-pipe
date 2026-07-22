//! goofi-subproc — the subprocess node tier (Pathway C).
//!
//! A [`RemoteNode`] runs a Python node in an isolated **GIL** interpreter, one
//! process per node. It exists for two reasons: (1) deps that aren't
//! free-threading-safe can't run in the in-process pyo3 host; (2) per the latency
//! finding, a *separate interpreter* has its own object ownership, so parallel
//! heavy-Python compute avoids free-threaded CPython's biased-refcount penalty.
//!
//! The child is the SAME `goofi.Node` class contract as the in-process tier, run by
//! `goofi.serve()` from the abi3 wheel — a **Rust** iceoryx2 loop reusing the shared
//! `goofi_pymod::exec` marshalling. The parent holds the other end of the transport.
//!
//! Each tick is one request/response over **iceoryx2 shared memory**. The parent encodes
//! `[u32 seq][request]` where the request is the shared [`goofi_codec::encode_request`]
//! frame (the live params + the present input slots, each a self-describing GOOF frame),
//! and publishes to the child's per-node `<id>_req` byte-slice service; the child runs the
//! node and publishes `[u32 seq][response]` back on `<id>_resp`, which the parent decodes
//! with [`goofi_codec::decode_response`]. The `seq` disambiguates responses so a re-publish
//! (needed while the child's subscriber is still connecting) never returns a stale frame.
//!
//! Both ends use the iceoryx2 **Rust crate** (the child from the wheel) over the 0.9.3 ABI —
//! the Python `iceoryx2` binding is gone. The child interpreter needs `goofi` (the wheel) +
//! `numpy`. Because the child shares `goofi_codec`, meta (channels/sfreq/index) crosses with
//! full fidelity and cast-to-f32 + warn live only in the child's shared `run_process`.
//!
//! The node implements the same [`Node`] trait as native and in-process Python
//! nodes, so the scheduler never branches on backend and the engine hosts it
//! through the ordinary `register_dyn_type` seam.

use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use iceoryx2::prelude::*;

use goofi_core::Data;
use goofi_node::{Inputs, Node, NodeCtx, NodeResult, Outputs, Params};

/// Per-process counter giving each spawned subprocess a unique iceoryx2 service-name base
/// (`goofi_sub_<pid>_<n>`), so concurrent nodes — and a respawn after a reset — never collide.
static SUBPROC_SEQ: AtomicU64 = AtomicU64::new(0);

/// iceoryx2 byte-slice pool ceiling per publisher (matches the child's `serve` config).
const MAX_PAYLOAD: usize = 64 * 1024;

/// Default cap on how long a tick waits for a subprocess response before treating
/// the child as hung. Generous enough to cover cold start (spawn + goofi/numpy import +
/// module compile + setup); a hung child is killed and surfaces as a node error rather
/// than stranding the scheduler (and, in the bridge, the graph mutex) indefinitely.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

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
    /// Spawn `python -c "import goofi; goofi.serve()"` with the node source + iceoryx2 service
    /// names in the environment, then build the parent's ports.
    fn spawn(python: &str, source: &str) -> std::result::Result<Running, String> {
        let id = format!("goofi_sub_{}_{}", std::process::id(), SUBPROC_SEQ.fetch_add(1, Ordering::Relaxed));
        let req_name = format!("{id}_req");
        let resp_name = format!("{id}_resp");
        // The child talks over iceoryx2; stdout/stderr are inherited so node prints/tracebacks
        // surface (the child routes its fd 1 to stderr, so a node print can't corrupt the SHM plane).
        let mut child = Command::new(python)
            .arg("-c")
            .arg("import goofi; goofi.serve()")
            .env("GOOFI_NODE_SRC", source)
            .env("GOOFI_IOX_REQ", &req_name)
            .env("GOOFI_IOX_RESP", &resp_name)
            // A subprocess node runs its OWN interpreter; a host `PYTHONPATH` (e.g. the pyo3/FT
            // tier's, injected by `.cargo/config.toml`) must not leak in and shadow the child's
            // numpy/goofi with an incompatible build. The child uses its interpreter's site-packages.
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
    /// This node's declared input / output slot names (from its manifest) — the keys the
    /// engine's `Inputs`/`Outputs` use, so `process` knows which slots to gather/emit.
    in_slots: Vec<&'static str>,
    out_slots: Vec<&'static str>,
    timeout: Duration,
    proc: Option<Running>,
}

impl RemoteNode {
    /// A remote node backed by `python` running `source` (a `goofi.Node` subclass), with the
    /// engine-facing input/output slot names from its manifest. No process spawns until the first tick.
    pub fn new(
        python: impl Into<String>,
        source: impl Into<String>,
        in_slots: Vec<&'static str>,
        out_slots: Vec<&'static str>,
    ) -> RemoteNode {
        RemoteNode {
            python: python.into(),
            source: source.into(),
            in_slots,
            out_slots,
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
    pub fn spawn(
        python: &str,
        source: &str,
        in_slots: Vec<&'static str>,
        out_slots: Vec<&'static str>,
    ) -> std::io::Result<RemoteNode> {
        let mut node = RemoteNode::new(python, source, in_slots, out_slots);
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
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        // Gather the present input slots; if the node has inputs but none arrived, there is
        // nothing to tick (the engine gates on the triggering slot).
        let present: Vec<(&str, &Data)> =
            self.in_slots.iter().filter_map(|name| inp.get(name).map(|d| (*name, d))).collect();
        if present.is_empty() && !self.in_slots.is_empty() {
            return Ok(());
        }

        let frame = goofi_codec::encode_request(p.groups(), &present);
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
        for (slot, data) in goofi_codec::decode_response(&resp)? {
            out.set(&slot, data);
        }
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
// Discovery — turn a directory of `goofi.Node` files into subprocess node types the
// engine hosts via `register_dyn_type`. Uses the SAME probe-based discoverer as the
// in-process tier (`goofi_node::discover`), so a file yields the same rich manifest
// (multi-slot + params) whichever backend hosts it; the factory spawns a RemoteNode.
// ---------------------------------------------------------------------------

use std::path::Path;

use goofi_node::discover::{Discovered, NodeFactory};
use goofi_node::{Isolation, NodeManifest};

/// A discovered subprocess node type, ready to `register_dyn_type` into a Graph.
pub struct SubprocNodeType {
    pub manifest: &'static NodeManifest,
    /// Builds a node instance (spawns lazily on first tick).
    pub factory: NodeFactory,
}

/// Turn a probe-[`Discovered`] (rich manifest + source path) into a subprocess [`SubprocNodeType`]:
/// the factory reads the file's source + builds a [`RemoteNode`] bound to the manifest's slot names.
fn subproc_type_from_discovered(python: &str, d: Discovered) -> SubprocNodeType {
    let manifest = d.manifest;
    let in_slots: Vec<&'static str> = manifest.inputs.iter().map(|s| s.name).collect();
    let out_slots: Vec<&'static str> = manifest.outputs.iter().map(|o| o.name).collect();
    let source = std::fs::read_to_string(&d.source).unwrap_or_default();
    let python = python.to_string();
    let factory: NodeFactory = Box::new(move |_p| {
        Box::new(RemoteNode::new(&python, &source, in_slots.clone(), out_slots.clone())) as Box<dyn Node>
    });
    SubprocNodeType { manifest, factory }
}

/// Build a subprocess-backed node type from a single file by running the `goofi.introspect`
/// probe on `python` (a GIL interpreter with `goofi` importable): the probe's rich manifest
/// gives multi-slot + params; the factory spawns a [`RemoteNode`]. `None` if it is not a node
/// file or the probe fails (missing dep / no `Node` subclass) — greyed out, never a catalog crash.
pub fn discover_one(path: &Path, python: &str) -> Option<SubprocNodeType> {
    let d = goofi_node::discover::discover_one(path, python, "subprocess", Isolation::Subprocess)?;
    Some(subproc_type_from_discovered(python, d))
}

/// Scan `dir` for node files, probing each on `python`; skips non-`.py`, `_`-prefixed, and
/// probe failures. Type names are the `CamelCase` file stem (shared with the in-process tier).
pub fn discover(dir: &Path, python: &str) -> std::io::Result<Vec<SubprocNodeType>> {
    let discovered = goofi_node::discover::discover(dir, python, "subprocess", Isolation::Subprocess)?;
    Ok(discovered.into_iter().map(|d| subproc_type_from_discovered(python, d)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Data, Meta, Param, Value};
    use goofi_node::ParamGroups;
    use indexmap::IndexMap;

    // Node sources are authored to the `goofi.Node` class contract (the one authoring shape;
    // the bare `def process(x)` function is gone). Raw strings keep Python indentation verbatim.

    /// Doubles its `data` input into `out`.
    const DOUBLE: &str = r#"
import goofi
class Double(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": data.data * 2.0}
"#;

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn arr(shape: Vec<usize>, v: &[f32], meta: Meta) -> Data {
        Data::array_f32(shape, f32s(v), meta).unwrap()
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

    /// Tick a node once: feed the named inputs + params, allocate the named output slots,
    /// return the output slot map (or the node error).
    fn tick(
        node: &mut RemoteNode,
        inputs: Vec<(&'static str, Data)>,
        out_names: &[&'static str],
        params: &ParamGroups,
    ) -> Result<IndexMap<&'static str, Option<Data>>, String> {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        for (n, d) in inputs {
            inmap.insert(n, Some(d));
        }
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        for n in out_names {
            outmap.insert(n, None);
        }
        let mut ctx = NodeCtx::new();
        let r = {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx, &Params::new(params))
        };
        r.map_err(|e| e.0)?;
        Ok(outmap)
    }

    /// The common `data -> out` single-slot, no-param tick; returns the `out` frame.
    fn run(node: &mut RemoteNode, d: Data) -> Data {
        try_run(node, d).expect("remote process")
    }
    fn try_run(node: &mut RemoteNode, d: Data) -> Result<Data, String> {
        let m = tick(node, vec![("data", d)], &["out"], &ParamGroups::new())?;
        Ok(m.get("out").unwrap().clone().expect("output frame"))
    }

    /// A python with BOTH goofi (the abi3 wheel) and numpy (the subprocess child needs both),
    /// or None. Prefers `$GOOFI_SUBPROC_TEST_PYTHON`, then the repo's `.venv`, then a PATH python.
    /// The probe strips `PYTHONPATH` exactly like the real child spawn ([`Running::spawn`]), so a
    /// host/pyo3 `PYTHONPATH` can't produce a false negative (it once masked real bugs by making
    /// the venv python import an incompatible numpy → every tier test silently SKIPPED).
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
                .arg("import goofi, numpy")
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
    /// subprocess tier tests HARD-REQUIRE a python (goofi + numpy) rather than silently
    /// skipping, so a missing/misconfigured interpreter fails loudly instead of hiding bugs.
    fn require_python() -> String {
        usable_python().unwrap_or_else(|| {
            panic!(
                "no python with goofi + numpy found (checked $GOOFI_SUBPROC_TEST_PYTHON, \
                 ./.venv/bin/python, python3, python). Install the goofi abi3 wheel into the \
                 interpreter: `maturin build -i <python> -o target/wheels && uv pip install \
                 --python <python> target/wheels/goofi-*.whl`. The subprocess-tier tests require one."
            )
        })
    }

    /// A `RemoteNode` holds its iceoryx2 ports directly (via `ipc_threadsafe::Service`), so it must
    /// stay `Send` for the scheduler. This compile-time guard fails loudly if the ports ever revert to
    /// the `!Send` `ipc::Service` — the whole reason the port-owning io thread could be removed.
    #[test]
    fn remote_node_stays_send() {
        fn _assert_send<T: Send>() {}
        _assert_send::<RemoteNode>();
    }

    #[test]
    fn remote_node_runs_a_class_node_in_a_subprocess() {
        let py = require_python();
        let src = r#"
import goofi
class Affine(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": data.data * 2.0 + 1.0}
"#;
        let mut node = RemoteNode::spawn(&py, src, vec!["data"], vec!["out"]).unwrap();

        // Input carries sfreq/index; a length-preserving node carries the input meta back.
        let mut meta = Meta::empty();
        meta.set_sfreq(Some(128.0));
        meta.set_index(Some(7));
        let d = arr(vec![3], &[1.0, 2.0, 3.0], meta);

        // Two ticks on the same long-lived subprocess (proves it stays alive).
        let out1 = run(&mut node, d.clone());
        assert_eq!(floats(&out1), vec![3.0, 5.0, 7.0]);
        let out2 = run(&mut node, d);
        assert_eq!(floats(&out2), vec![3.0, 5.0, 7.0]);

        assert_eq!(out1.meta().sfreq(), Some(128.0));
        assert_eq!(out1.meta().index(), Some(7));
        match out1.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[3]),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn a_param_reaches_the_subprocess_node() {
        // Mirrors the in-process host test: `setup` seeds `self._base`; `process` reads a live
        // param `gain.factor`. Proves setup ran (its value appears) AND a param crossed the wire.
        let py = require_python();
        let src = r#"
import goofi
class Scale(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    @staticmethod
    def config_params():
        return {"gain": {"factor": goofi.IntParam(1, 0, 100)}}
    def setup(self):
        self._base = 100.0
    def process(self, data):
        return {"out": data.data * self.params.gain.factor + self._base}
"#;
        let mut node = RemoteNode::spawn(&py, src, vec!["data"], vec!["out"]).unwrap();

        let mut params = ParamGroups::new();
        let mut gain = IndexMap::new();
        gain.insert("factor".to_string(), Param::int(3, 0, 100));
        params.insert("gain".to_string(), gain);

        let out = tick(&mut node, vec![("data", arr(vec![2], &[1.0, 2.0], Meta::empty()))], &["out"], &params)
            .expect("param tick");
        // 1*3 + 100 = 103 ; 2*3 + 100 = 106.
        assert_eq!(floats(out.get("out").unwrap().as_ref().unwrap()), vec![103.0, 106.0]);
    }

    #[test]
    fn psd_runs_over_the_transport_reading_sfreq() {
        let py = require_python();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/psd.py");
        // Discovery: CamelCase "Psd" on the subprocess tier, with its declared welch.nperseg param.
        let ty = discover_one(&path, &py).expect("psd.py discovers as a subprocess node");
        assert_eq!(ty.manifest.type_name, "Psd");
        assert_eq!(ty.manifest.isolation, Isolation::Subprocess);
        assert_eq!(ty.manifest.outputs[0].name, "psd");
        assert!(
            ty.manifest.params.iter().any(|p| p.group == "welch" && p.name == "nperseg"),
            "the welch.nperseg param is discovered into the manifest"
        );

        // A 1x64 unit sine at exactly 8 cycles (bin 8, independent of sfreq); sfreq=1000 in
        // meta only scales the PSD magnitude, so a small peak proves the child read sfreq.
        let n = 64usize;
        let sfreq = 1000.0f64;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / n as f64).sin() as f32)
            .collect();
        let d = arr(vec![1, n], &samples, Meta::new().with_sfreq(Some(sfreq)));

        let src = std::fs::read_to_string(&path).unwrap();
        let mut node = RemoteNode::new(&py, &src, vec!["data"], vec!["psd"]);
        let m = tick(&mut node, vec![("data", d)], &["psd"], &ParamGroups::new()).expect("psd tick");
        let out = m.get("psd").unwrap().as_ref().unwrap();

        match out.value() {
            Value::Array(s) => {
                assert_eq!(s.shape(), &[1, n / 2 + 1], "channels preserved; rfft bins");
                let psd = floats(out);
                let peak = (0..psd.len()).max_by(|a, b| psd[*a].total_cmp(&psd[*b])).unwrap();
                assert_eq!(peak, 8, "spectral peak at the input frequency bin");
                // sfreq=1000 normalization -> a small peak; the fallback sfreq=1 would be ~1000x larger.
                assert!(psd[peak] < 1.0, "peak {} implies sfreq=1000 reached the child, not the fallback", psd[peak]);
            }
            _ => panic!("expected array"),
        }
        // The node explicitly emits the frequency axis (a (array, meta) tuple return); that
        // node-authored meta crosses the transport intact — proving explicit output meta works.
        match out.meta().get("freqs") {
            Some(goofi_core::MetaValue::List(v)) => assert_eq!(v.len(), n / 2 + 1, "one freq per rfft bin"),
            other => panic!("expected a freqs list in the output meta, got {other:?}"),
        }
    }

    #[test]
    fn channels_preserved_on_2d_length_preserving_node() {
        // The canonical EEG case: a [2,3] array with dim0 channel labels through a length-preserving
        // node must come back [2,3] with channels intact (the shared full-meta codec carries them).
        let py = require_python();
        let mut node = RemoteNode::spawn(&py, DOUBLE, vec!["data"], vec!["out"]).unwrap();

        let mut meta = Meta::empty();
        meta.set_channels(goofi_core::Axes::new().with(
            0,
            goofi_core::Axis::coords(vec![
                goofi_core::Coord::Str("Fz".into()),
                goofi_core::Coord::Str("Cz".into()),
            ]),
        ));
        let d = arr(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], meta);

        let out = try_run(&mut node, d).expect("2-D channel frame must round-trip");
        match out.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[2, 3], "shape preserved"),
            _ => panic!("expected array"),
        }
        assert_eq!(floats(&out), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let ch = out.meta().channels().get(0).and_then(|a| a.coords.clone()).expect("dim0 channels preserved");
        assert_eq!(ch.len(), 2);
    }

    #[test]
    fn subprocess_roundtrip_latency_and_stability() {
        // Concrete latency/stability read for the iceoryx2 subprocess tier: after cold start,
        // run many round-trips of a realistic EEG-sized frame and report the latency distribution.
        let py = require_python();
        let ident = r#"
import goofi
class Ident(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": data.data * 1.0}
"#;
        let mut node = RemoteNode::spawn(&py, ident, vec!["data"], vec!["out"]).unwrap();
        // A 32-channel × 256-sample float32 frame (~32 KB) — a typical EEG buffer.
        let (c, t) = (32usize, 256usize);
        let vals: Vec<f32> = (0..c * t).map(|i| i as f32).collect();
        let make = || arr(vec![c, t], &vals, Meta::empty());

        // Warm up: the first tick pays cold start (spawn + goofi/numpy import + module compile).
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
    fn node_stdout_does_not_corrupt_the_transport() {
        // The child routes fd 1 to stderr before importing the node. A node that writes to stdout
        // (here a flushed print, but equally a C-extension's printf) must NOT inject bytes into the
        // SHM frame plane.
        let py = require_python();
        let src = r#"
import goofi
class Chatty(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        import sys
        print("debug from the node", flush=True)
        sys.stdout.flush()
        return {"out": data.data * 2.0}
"#;
        let mut node = RemoteNode::spawn(&py, src, vec!["data"], vec!["out"]).unwrap();
        let out = try_run(&mut node, arr(vec![3], &[0.0, 1.0, 2.0], Meta::empty())).expect("a printing node still round-trips");
        assert_eq!(floats(&out), vec![0.0, 2.0, 4.0]);
        // A second tick proves the stream stayed in sync (not just the first frame).
        let out2 = try_run(&mut node, arr(vec![2], &[5.0, 6.0], Meta::empty())).expect("second tick in sync");
        assert_eq!(floats(&out2), vec![10.0, 12.0]);
    }

    #[test]
    fn crashed_child_is_reaped_and_respawns() {
        // A tick whose node raises must Err, then a later tick must succeed (a fresh subprocess is
        // spawned) rather than being wedged forever.
        let py = require_python();
        let src = r#"
import goofi
class Boom(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        if data.data[0] < 0:
            raise ValueError("boom")
        return {"out": data.data * 2.0}
"#;
        let mut node = RemoteNode::spawn(&py, src, vec!["data"], vec!["out"]).unwrap();

        assert!(try_run(&mut node, arr(vec![1], &[-1.0], Meta::empty())).is_err(), "node raise must surface as an error");
        let out = try_run(&mut node, arr(vec![1], &[3.0], Meta::empty())).expect("must respawn and succeed on the next tick");
        assert_eq!(floats(&out), vec![6.0]);
    }

    #[test]
    fn hung_subprocess_times_out_instead_of_hanging() {
        // A node that never responds must not block forever; with a short timeout the tick errors.
        let py = require_python();
        let src = r#"
import time
import goofi
class Hang(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        while True:
            time.sleep(1)
"#;
        let mut node = RemoteNode::new(&py, src, vec!["data"], vec!["out"]).with_timeout(Duration::from_millis(600));
        let t = std::time::Instant::now();
        let r = try_run(&mut node, arr(vec![1], &[1.0], Meta::empty()));
        assert!(r.is_err(), "a hung subprocess must error, not hang");
        assert!(t.elapsed() < Duration::from_secs(5), "must return near the timeout, not block");
    }

    #[test]
    fn large_frame_into_a_stuck_child_times_out() {
        // The child blocks at module import (never enters its serve loop) AND the frame is large;
        // the publish is non-blocking SHM, so the tick must time out (not hang) with no response.
        let py = require_python();
        let src = r#"
import time
time.sleep(30)
import goofi
class Slow(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": data.data}
"#;
        let mut node = RemoteNode::new(&py, src, vec!["data"], vec!["out"]).with_timeout(Duration::from_millis(600));
        let n = 40_000usize; // 160 KB >> the 64 KiB initial slice — a big frame to a stuck child
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = std::time::Instant::now();
        let r = try_run(&mut node, arr(vec![n], &vals, Meta::empty()));
        assert!(r.is_err(), "a child stuck before the loop must error, not hang");
        assert!(t.elapsed() < Duration::from_secs(5), "must return near the timeout");
    }

    #[test]
    fn large_frame_round_trips_over_shared_memory() {
        // A frame far larger than the 64 KiB initial slice must round-trip — iceoryx2 grows the
        // publisher's segment (PowerOfTwo), and the 4-byte sequence framing survives a big body.
        let py = require_python();
        let mut node = RemoteNode::spawn(&py, DOUBLE, vec!["data"], vec!["out"]).unwrap();
        let n = 100_000usize; // 400 KB body
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let out = run(&mut node, arr(vec![n], &vals, Meta::empty()));
        let got = floats(&out);
        assert_eq!(got.len(), n, "shape preserved across the SHM round-trip");
        assert_eq!(got[1], 2.0, "1 * 2");
        assert_eq!(got[10], 20.0, "10 * 2");
        assert_eq!(got[n - 1], (n - 1) as f32 * 2.0, "last element doubled");
    }

    #[test]
    fn gil_safe_distinguishes_free_threaded_from_gil_interpreters() {
        // A normal (GIL) interpreter must be judged UNSAFE for in-process hosting — it either
        // reports the GIL enabled or lacks sys._is_gil_enabled entirely.
        if let Some(py) = usable_python() {
            assert!(
                !gil_safe(&py, "import numpy").unwrap(),
                "a GIL interpreter must be judged unsafe for in-process hosting"
            );
        }
        // A free-threaded interpreter importing an FT-safe dep keeps the GIL disabled → SAFE.
        if let Ok(ft) = std::env::var("GOOFI_FT_PYTHON") {
            assert!(gil_safe(&ft, "import numpy").unwrap(), "free-threaded + FT-safe deps must be judged safe");
            assert!(gil_safe(&ft, "x = 1\n").unwrap(), "a bare source is trivially safe on a free-threaded build");
        } else {
            eprintln!("NOTE: set GOOFI_FT_PYTHON to also exercise the free-threaded branch");
        }
    }

    #[test]
    fn discover_yields_subprocess_types_that_run() {
        let py = require_python();
        let dir = std::env::temp_dir().join(format!("goofi_subdisc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let negate = r#"
import goofi
class Negate(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": -data.data}
"#;
        std::fs::write(dir.join("negate.py"), negate).unwrap();
        std::fs::write(dir.join("_hidden.py"), negate).unwrap(); // underscore → skipped
        std::fs::write(dir.join("nope.py"), "x = 1\n").unwrap(); // no Node subclass → probe skips

        let types = discover(&dir, &py).unwrap();
        let names: Vec<&str> = types.iter().map(|t| t.manifest.type_name).collect();
        assert_eq!(names, vec!["Negate"]);
        assert_eq!(types[0].manifest.category, "subprocess");
        assert_eq!(types[0].manifest.isolation, Isolation::Subprocess);

        // The factory builds a working node.
        let mut node = (types[0].factory)(&ParamGroups::new());
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(arr(vec![2], &[1.0, -2.0], Meta::empty())));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        let params = ParamGroups::new();
        {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx, &Params::new(&params)).unwrap();
        }
        assert_eq!(floats(outmap.get("out").unwrap().as_ref().unwrap()), vec![-1.0, 2.0]);
        drop(node);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
