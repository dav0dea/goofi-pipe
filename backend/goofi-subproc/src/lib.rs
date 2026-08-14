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
//! `goofi_pymod::exec` marshalling. The parent holds the other end of the transport — and the
//! write end of a **parent-liveness pipe** ([`goofi_codec::liveness`]), which the child watches
//! for EOF, so a Ctrl-C'd or crashed manager can never orphan a forever-spinning child.
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
use goofi_node::{Inputs, Node, NodeCtx, NodeError, NodeResult, Outputs, Params};

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
    /// The parent's write end of the liveness pipe. Never written to: holding it open IS the
    /// signal. It closes when this struct drops — on a node removal, on a reset, and (the
    /// point of the exercise) when the OS tears this process down on a Ctrl-C or a crash,
    /// where no `Drop` ever runs. The child reads EOF and exits itself.
    parent_alive: Option<std::io::PipeWriter>,
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
        let mut cmd = Command::new(python);
        cmd.arg("-c")
            .arg("import goofi; goofi.serve()")
            .env("GOOFI_NODE_SRC", source)
            .env("GOOFI_IOX_REQ", &req_name)
            .env("GOOFI_IOX_RESP", &resp_name)
            // A subprocess node runs its OWN interpreter; a host `PYTHONPATH` (e.g. the pyo3/FT
            // tier's, injected by `.cargo/config.toml`) must not leak in and shadow the child's
            // numpy/goofi with an incompatible build. The child uses its interpreter's site-packages.
            .env_remove("PYTHONPATH")
            .env_remove("PYTHONHOME")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        // Arm the liveness pipe BEFORE the spawn it guards: the child inherits the read end and
        // stops itself the moment our write end closes, so a Ctrl-C or a crash here can't leave
        // it orphaned, spinning its poll loop forever.
        let armed = goofi_codec::liveness::arm(&mut cmd).map_err(|e| format!("liveness pipe: {e}"))?;
        let mut child = cmd.spawn().map_err(|e| format!("spawn `{python}`: {e}"))?;
        let parent_alive = Some(armed.into_writer());
        match build_ports(&req_name, &resp_name) {
            Ok(ports) => Ok(Running { child, ports, seq: 0, parent_alive }),
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
        one_roundtrip(&self.ports.req_pub, &self.ports.resp_sub, &mut self.child, self.seq, frame, timeout)
            .map_err(|e| format!("subprocess io: {e}"))
    }

    /// Kill + reap the child. The ports drop with `self`.
    fn shutdown(&mut self) {
        // Close the liveness pipe first: that stop reaches the child even when a signal or a
        // dead handle would defeat `kill`, and it is the same door a crashed parent uses.
        // The kill+wait then ends and reaps it immediately rather than waiting on the poll.
        drop(self.parent_alive.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

type BytePublisher = iceoryx2::port::publisher::Publisher<ipc_threadsafe::Service, [u8], ()>;
type ByteSubscriber = iceoryx2::port::subscriber::Subscriber<ipc_threadsafe::Service, [u8], ()>;

/// One request/response: publish `[seq][frame]` and poll `<resp>` for the sample whose leading
/// sequence matches. Re-publishes each idle millisecond (so the child gets it even if its
/// subscriber was still connecting when we first published) and drops any stale/mismatched
/// sample. Bounded by `timeout` — and by the child's own life: a process that has already died
/// is not a hang, so it is reported as the exit it was rather than after the full deadline.
fn one_roundtrip(
    req_pub: &BytePublisher,
    resp_sub: &ByteSubscriber,
    child: &mut Child,
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
        // Checked AFTER draining the response port, so a child that answered and then exited still
        // has its answer returned. An import crash, a C-extension segfault or an OOM kill leaves
        // nobody to answer, and waiting out the deadline would both park a worker for the full
        // production timeout and then name the wrong cause.
        if let Ok(Some(status)) = child.try_wait() {
            return Err(std::io::Error::other(format!("subprocess exited: {status}")));
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
    /// This node's declared INPUT slot names (from its manifest) — the keys `process` gathers
    /// from `Inputs`. Outputs are set by the child-returned slot names (the child is authoritative
    /// for output naming, via its `config_output_slots`), so the parent keeps no output list.
    in_slots: Vec<&'static str>,
    timeout: Duration,
    proc: Option<Running>,
}

impl RemoteNode {
    /// A remote node backed by `python` running `source` (a `goofi.Node` subclass), with the
    /// engine-facing INPUT slot names from its manifest. No process spawns until the first tick.
    pub fn new(python: impl Into<String>, source: impl Into<String>, in_slots: Vec<&'static str>) -> RemoteNode {
        RemoteNode {
            python: python.into(),
            source: source.into(),
            in_slots,
            timeout: DEFAULT_TIMEOUT,
            proc: None,
        }
    }

    /// Override the per-tick response timeout (builder; mainly for tests/config).
    pub fn with_timeout(mut self, timeout: Duration) -> RemoteNode {
        self.timeout = timeout;
        self
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
        // Only the PRESENT slots cross the wire — an absent one is the absence of an entry, and
        // the child rebuilds the full declared kwarg set from its own `config_input_slots()`.
        // A node with inputs but none arrived still ticks: what a missing non-required input
        // means is the node's own call (it receives `None`), and a required one never gets here.
        let present: Vec<(&str, &Data)> =
            self.in_slots.iter().filter_map(|name| inp.get(name).map(|d| (*name, d))).collect();

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
        // A node RAISE comes back as a NodeError WITHOUT killing the child: surface it like the
        // in-process tier's `Ok(Err)` and leave the child alive — node state preserved, error
        // reported instantly (no 10s respawn-timeout loop, no lost exception text).
        match goofi_codec::decode_response(&resp).map_err(NodeError)? {
            goofi_codec::Response::Slots(outs) => {
                for (slot, data) in outs {
                    out.set(&slot, data);
                }
                Ok(())
            }
            goofi_codec::Response::NodeError(msg) => Err(NodeError(msg)),
        }
    }
}

impl Drop for RemoteNode {
    fn drop(&mut self) {
        self.reset();
    }
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

pub use goofi_node::discover::Discovery;

/// Probe one file for this tier, reporting all three outcomes. The CLI uses this (rather than
/// [`discover_one`]) because it needs the failure REASON to list the node as unavailable, and one
/// probe spawn must answer both questions.
pub fn probe(path: &Path, python: &str) -> Discovery {
    goofi_node::discover::discover_one(path, python, "subprocess", Isolation::Subprocess)
}

/// Turn a probe-[`Discovered`] (rich manifest + source path) into a subprocess [`SubprocNodeType`]:
/// the factory reads the file's source + builds a [`RemoteNode`] bound to the manifest's slot names.
/// Public so a caller that already ran [`probe`] can build the type without a second spawn.
pub fn node_type_from(python: &str, d: Discovered) -> SubprocNodeType {
    subproc_type_from_discovered(python, d)
}

fn subproc_type_from_discovered(python: &str, d: Discovered) -> SubprocNodeType {
    let manifest = d.manifest;
    let in_slots: Vec<&'static str> = manifest.inputs.iter().map(|s| s.name).collect();
    let source = std::fs::read_to_string(&d.source).unwrap_or_default();
    let python = python.to_string();
    let factory: NodeFactory = Box::new(move |_p| {
        Box::new(RemoteNode::new(&python, &source, in_slots.clone())) as Box<dyn Node>
    });
    SubprocNodeType { manifest, factory }
}

/// Build a subprocess-backed node type from a single file by running the `goofi.introspect`
/// probe on `python` (a GIL interpreter with `goofi` importable): the probe's rich manifest
/// gives multi-slot + params; the factory spawns a [`RemoteNode`]. `None` if it is not a node
/// file or the probe fails (missing dep / no `Node` subclass) — greyed out, never a catalog crash.
pub fn discover_one(path: &Path, python: &str) -> Option<SubprocNodeType> {
    let goofi_node::discover::Discovery::Found(d) =
        goofi_node::discover::discover_one(path, python, "subprocess", Isolation::Subprocess)
    else {
        return None;
    };
    Some(subproc_type_from_discovered(python, d))
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
    /// or None. Prefers `$GOOFI_SUBPROC_TEST_PYTHON`, then the repo's `.gfivenv`, then a PATH python.
    /// The probe strips `PYTHONPATH` exactly like the real child spawn ([`Running::spawn`]), so a
    /// host/pyo3 `PYTHONPATH` can't produce a false negative (it once masked real bugs by making
    /// the venv python import an incompatible numpy → every tier test silently SKIPPED).
    fn usable_python() -> Option<String> {
        let mut cands: Vec<String> = Vec::new();
        if let Ok(p) = std::env::var("GOOFI_SUBPROC_TEST_PYTHON") {
            cands.push(p);
        }
        // Both venv layouts — `bin/` on unix, `Scripts/` on Windows — because the fallbacks below
        // are worse than a miss on Windows: `python3` there is an App Execution Alias that answers
        // every probe with a Microsoft Store advert instead of failing.
        cands.push(format!("{}/../../.gfivenv/bin/python", env!("CARGO_MANIFEST_DIR")));
        cands.push(format!("{}/../../.gfivenv/Scripts/python.exe", env!("CARGO_MANIFEST_DIR")));
        cands.push("python3".to_string());
        cands.push("python".to_string());
        for cand in cands {
            if let Ok(out) = Command::new(&cand)
                .arg("-c")
                .arg("import goofi, numpy")
                .env_remove("PYTHONPATH")
                .env_remove("PYTHONHOME")
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

    /// Serializes the subprocess-tier tests. Cargo runs a crate's tests on parallel threads and
    /// every one of these spawns a Python interpreter, so without this the latency measurement
    /// below is taken while a dozen siblings are booting numpy on the same cores — it measured
    /// the test harness, not the transport, and failed its budget at ~7 ms median.
    static TIER: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The interpreter to spawn children with, plus the tier lock — held until the value drops,
    /// i.e. for the rest of the test. Derefs to `&str`, so call sites read as a plain path.
    struct Tier {
        py: String,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl std::ops::Deref for Tier {
        type Target = str;
        fn deref(&self) -> &str {
            &self.py
        }
    }

    /// Like [`usable_python`] but PANICS with an actionable message when none is found — the
    /// subprocess tier tests HARD-REQUIRE a python (goofi + numpy) rather than silently
    /// skipping, so a missing/misconfigured interpreter fails loudly instead of hiding bugs.
    fn require_python() -> Tier {
        // A panicking test poisons the mutex; recover rather than cascade its failure onto
        // every sibling, which would bury the one real error.
        let _lock = TIER.lock().unwrap_or_else(|e| e.into_inner());
        let py = usable_python().unwrap_or_else(|| {
            panic!(
                "no python with goofi + numpy found (checked $GOOFI_SUBPROC_TEST_PYTHON, \
                 ./.gfivenv/bin/python, python3, python). Run `cargo run -p goofi-init`, which \
                 creates the venvs and installs the goofi wheel into them. The subprocess-tier \
                 tests require one."
            )
        });
        Tier { py, _lock }
    }

    /// Counts in a background Python thread started from `setup()` — the shape of every device
    /// input node (an OSC/LSL/serial receiver `serve_forever`-ing off the tick). `process` just
    /// reports the count, so the parent can see whether the thread ran between two ticks.
    const TICKER: &str = r#"
import threading, time
import numpy as np
import goofi
class Ticker(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def setup(self):
        self.count = 0
        def spin():
            while True:
                self.count += 1
                time.sleep(0.001)
        threading.Thread(target=spin, daemon=True).start()
    def process(self, data):
        return {"out": np.array([float(self.count)], dtype="float32")}
"#;

    #[test]
    fn a_nodes_own_python_thread_runs_while_the_child_is_idle() {
        // The subprocess tier exists to host GIL-bound libraries, and the canonical shape of a
        // device input is a receiver thread started in `setup()`. The child's serve loop is pure
        // Rust between requests, so if it holds the GIL across its idle sleep that thread is
        // starved for exactly as long as the node is not being ticked — which for an unwired or
        // slowly-paced node is forever.
        let py = require_python();
        let mut node = RemoteNode::new(&*py, TICKER, vec!["data"]);
        let d = arr(vec![1], &[0.0], Meta::empty());

        let first = floats(&run(&mut node, d.clone()))[0]; // cold start + setup + one tick
        std::thread::sleep(Duration::from_millis(300)); // an idle gap: no request in flight
        let second = floats(&run(&mut node, d))[0];

        // 300 ms at a 1 ms cadence is ~300 increments; a starved thread manages a couple at most,
        // stolen from the eval loop while `process` itself is running.
        assert!(
            second - first > 50.0,
            "the node's thread must run while the child idles: {first} -> {second}"
        );
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
        let mut node = RemoteNode::new(&*py, src, vec!["data"]);

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
        let mut node = RemoteNode::new(&*py, src, vec!["data"]);

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
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../nodes/psd.py");
        // Discovery: CamelCase "Psd" on the subprocess tier, with its declared welch.nperseg param.
        let ty = discover_one(&path, &py).expect("psd.py discovers as a subprocess node");
        assert_eq!(ty.manifest.type_name, "Psd");
        assert_eq!(ty.manifest.isolation, Isolation::Subprocess);
        assert_eq!(ty.manifest.outputs[0].name, "psd");
        let nperseg = ty
            .manifest
            .params
            .iter()
            .find(|p| p.group == "welch" && p.name == "nperseg")
            .expect("the welch.nperseg param is discovered into the manifest");
        // End-to-end for `doc=`: authored in the .py, emitted by the installed wheel's probe,
        // parsed here into the manifest the UI renders its tooltip from.
        assert!(
            nperseg.doc.is_some_and(|d| d.contains("Window length")),
            "the param's doc= crossed the probe; got {:?}",
            nperseg.doc
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
        let mut node = RemoteNode::new(&*py, &src, vec!["data"]);
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
        let mut node = RemoteNode::new(&*py, DOUBLE, vec!["data"]);

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
        let mut node = RemoteNode::new(&*py, ident, vec!["data"]);
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
        let mut node = RemoteNode::new(&*py, src, vec!["data"]);
        let out = try_run(&mut node, arr(vec![3], &[0.0, 1.0, 2.0], Meta::empty())).expect("a printing node still round-trips");
        assert_eq!(floats(&out), vec![0.0, 2.0, 4.0]);
        // A second tick proves the stream stayed in sync (not just the first frame).
        let out2 = try_run(&mut node, arr(vec![2], &[5.0, 6.0], Meta::empty())).expect("second tick in sync");
        assert_eq!(floats(&out2), vec![10.0, 12.0]);
    }

    #[test]
    fn node_error_surfaces_fast_without_killing_the_child() {
        // A per-tick node RAISE must surface as an error FAST (not via a 10s respawn-timeout) and
        // must NOT kill the child — the SAME subprocess handles the next tick with state intact,
        // matching the in-process tier's Ok(Err). The Python exception text rides back.
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
    def setup(self):
        self._ticks = 0
    def process(self, data):
        self._ticks += 1
        if data.data[0] < 0:
            raise ValueError("boom")
        return {"out": data.data + self._ticks}
"#;
        let mut node = RemoteNode::new(&*py, src, vec!["data"]);

        // A good tick: setup ran (ticks 0 -> 1), so 10 + 1 = 11.
        assert_eq!(floats(&run(&mut node, arr(vec![1], &[10.0], Meta::empty()))), vec![11.0]);

        // A raise: fast error carrying the exception text; the child stays alive.
        let t = std::time::Instant::now();
        let err = try_run(&mut node, arr(vec![1], &[-1.0], Meta::empty())).expect_err("a raise surfaces as an error");
        assert!(t.elapsed() < Duration::from_secs(2), "the error must surface fast, not via a respawn timeout");
        assert!(err.contains("boom"), "the Python exception text rides back: {err}");

        // The SAME child continues with state intact: ticks went 1 -> 2 (raised, still counted)
        // -> 3 here, so 10 + 3 = 13 proves NO respawn (a respawn would reset _ticks + re-run setup).
        assert_eq!(
            floats(&run(&mut node, arr(vec![1], &[10.0], Meta::empty()))),
            vec![13.0],
            "child survived the raise with its state (no respawn)"
        );
    }

    #[test]
    fn a_setup_that_raises_is_retried_on_the_next_tick() {
        // D3 across the tiers: the inline tier retries the whole initialization on any interaction,
        // so the child must too — it used to mark `did_setup` BEFORE running setup, deliberately,
        // "matching the in-process tier's run-once semantics". That parity argument now points the
        // other way. A device that was not ready at the first tick therefore comes back on the
        // next one, without a restart.
        let py = require_python();
        let src = r#"
import goofi
class LateBoot(goofi.Node):
    setups = 0
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def setup(self):
        LateBoot.setups += 1
        if LateBoot.setups < 2:
            raise RuntimeError("device is not open")
    def process(self, data):
        return {"out": data.data + LateBoot.setups}
"#;
        let mut node = RemoteNode::new(&*py, src, vec!["data"]);

        let err = try_run(&mut node, arr(vec![1], &[10.0], Meta::empty()))
            .expect_err("the first tick's setup raised");
        assert!(err.contains("device is not open"), "the Python exception text rides back: {err}");

        // The SAME child, one tick later: setup runs a second time and succeeds, so `process` runs
        // — 10 + 2 setups. A latched `did_setup` would either skip setup (and raise on the missing
        // attribute) or report the same failure forever.
        assert_eq!(
            floats(&run(&mut node, arr(vec![1], &[10.0], Meta::empty()))),
            vec![12.0],
            "the interaction retried the initialization and the node came up"
        );
    }

    #[test]
    fn a_dead_child_is_reaped_and_respawns() {
        // If the child PROCESS actually dies (not a catchable raise — here os._exit), the tick
        // times out, the child is reaped, and a later tick respawns a fresh one.
        let py = require_python();
        let src = r#"
import os
import goofi
class Killer(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        if data.data[0] < 0:
            os._exit(1)
        return {"out": data.data * 2.0}
"#;
        let mut node = RemoteNode::new(&*py, src, vec!["data"]).with_timeout(Duration::from_millis(800));
        assert!(try_run(&mut node, arr(vec![1], &[-1.0], Meta::empty())).is_err(), "a dead child surfaces as an error");
        let out = try_run(&mut node, arr(vec![1], &[3.0], Meta::empty())).expect("must respawn on the next tick");
        assert_eq!(floats(&out), vec![6.0]);
    }

    #[test]
    fn an_exited_child_is_noticed_at_once_instead_of_waiting_out_the_timeout() {
        // A child that has ALREADY died — import crash, C-extension segfault, OOM kill — is not a
        // hang: there is nobody left to answer. Watching only the response port made every such
        // death cost the full production timeout (10 s of a parked worker) and then report
        // "did not respond in time", which names the wrong cause. Run at the DEFAULT timeout so
        // the assertion is about noticing the death, not about a short test timeout.
        let py = require_python();
        let src = r#"
import os
import goofi
class Killer(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        os._exit(3)
"#;
        let mut node = RemoteNode::new(&*py, src, vec!["data"]);
        assert_eq!(node.timeout, DEFAULT_TIMEOUT, "the production timeout, not a test-shortened one");
        let t = std::time::Instant::now();
        let err = try_run(&mut node, arr(vec![1], &[1.0], Meta::empty())).expect_err("a dead child is an error");
        assert!(err.contains("subprocess exited"), "the error names the exit, not a timeout: {err}");
        assert!(
            t.elapsed() < Duration::from_secs(5),
            "noticed on the child's death, not after the {DEFAULT_TIMEOUT:?} deadline (took {:?})",
            t.elapsed()
        );
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
        let mut node = RemoteNode::new(&*py, src, vec!["data"]).with_timeout(Duration::from_millis(600));
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
        let mut node = RemoteNode::new(&*py, src, vec!["data"]).with_timeout(Duration::from_millis(600));
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
        let mut node = RemoteNode::new(&*py, DOUBLE, vec!["data"]);
        let n = 100_000usize; // 400 KB body
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let out = run(&mut node, arr(vec![n], &vals, Meta::empty()));
        let got = floats(&out);
        assert_eq!(got.len(), n, "shape preserved across the SHM round-trip");
        assert_eq!(got[1], 2.0, "1 * 2");
        assert_eq!(got[10], 20.0, "10 * 2");
        assert_eq!(got[n - 1], (n - 1) as f32 * 2.0, "last element doubled");
    }

    /// Bounded poll for a child to exit — never an unbounded wait that would hang the suite.
    fn child_exits_within(child: &mut Child, bound: Duration) -> bool {
        let deadline = Instant::now() + bound;
        loop {
            if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn the_child_stops_when_the_parents_liveness_pipe_closes() {
        // The mechanism itself, through the REAL spawn path: close only the parent's write end
        // — no kill, no signal — and the child must reach EOF and exit on its own.
        let py = require_python();
        let mut node = RemoteNode::new(&*py, DOUBLE, vec!["data"]);
        // A real tick first, so the child is fully up and inside its serve loop.
        assert_eq!(floats(&run(&mut node, arr(vec![1], &[2.0], Meta::empty()))), vec![4.0]);

        let running = node.proc.as_mut().expect("the tick spawned a child");
        drop(running.parent_alive.take()); // the parent "dies"

        let exited = child_exits_within(&mut running.child, Duration::from_secs(5));
        if !exited {
            // Never leave a spinning orphan behind, even when this test fails.
            let _ = running.child.kill();
            let _ = running.child.wait();
        }
        assert!(exited, "the child must exit on the liveness pipe's EOF; it was still alive after 5s");
    }

    /// Set on the helper process of [`a_hard_killed_parent_still_stops_the_child`]; its value
    /// is the interpreter to spawn the grandchild with. Gated with the two tests that read it —
    /// they are `/proc`-and-signals work — or it is a constant declared everywhere and used on
    /// one platform, which every other platform reports as dead code.
    #[cfg(target_os = "linux")]
    const HELPER_ENV: &str = "GOOFI_LIVENESS_HELPER_PYTHON";

    /// The intermediate parent for the hard-kill test, re-entered as a separate process (this
    /// test binary, one `#[test]` filtered in). With [`HELPER_ENV`] set it spawns a real child,
    /// announces its pid and then blocks forever holding the liveness pipe, so the outer test
    /// can `kill -9` a process that never gets to run a handler or a `Drop`. Unset — every
    /// ordinary suite run — it does nothing.
    #[cfg(target_os = "linux")]
    #[test]
    fn liveness_helper_process() {
        let Ok(py) = std::env::var(HELPER_ENV) else { return };
        let running = Running::spawn(&py, DOUBLE).expect("helper: spawn a child");
        println!("HELPER_CHILD_PID={}", running.child.id());
        std::io::Write::flush(&mut std::io::stdout()).expect("helper: flush the pid");
        loop {
            // Hold `running` — and with it the write end — until we are killed.
            std::thread::sleep(Duration::from_secs(60));
        }
    }

    /// Alive = present in /proc and not already a reaped-pending zombie.
    #[cfg(target_os = "linux")]
    fn pid_alive(pid: u32) -> bool {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else { return false };
        // `pid (comm) STATE …`, and comm may itself contain spaces or parens — scan past the last ')'.
        stat.rsplit_once(')')
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .is_some_and(|state| state != "Z")
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_hard_killed_parent_still_stops_the_child() {
        // The fidelity case a ctrl_c handler cannot cover: SIGKILL the intermediate parent, which
        // gets no chance to clean up. The OS closes its write end anyway, so the child still EOFs.
        let py = require_python();
        let mut helper = Command::new(std::env::current_exe().expect("test binary path"))
            .args(["--exact", "tests::liveness_helper_process", "--nocapture", "--test-threads=1"])
            .env(HELPER_ENV, &*py)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn the intermediate parent");

        // Read the announced pid with a bound; a helper that never gets there must not hang us.
        // libtest writes `test <name> ... ` without a newline, so the marker lands mid-line.
        let out = helper.stdout.take().expect("helper stdout");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(out).lines().map_while(Result::ok) {
                if let Some((_, pid)) = line.rsplit_once("HELPER_CHILD_PID=") {
                    let _ = tx.send(pid.trim().to_string());
                    return;
                }
            }
        });
        let announced = rx.recv_timeout(Duration::from_secs(30));
        if announced.is_err() {
            // Never strand the helper (and its child) when this test fails.
            let _ = helper.kill();
            let _ = helper.wait();
        }
        let child_pid: u32 =
            announced.expect("the helper must announce its child's pid").parse().expect("a numeric pid");

        // SIGKILL: no handler, no unwind, no Drop — only the OS closing the write end.
        let killed = Command::new("kill").args(["-9", &helper.id().to_string()]).status();
        assert!(killed.is_ok_and(|s| s.success()), "kill -9 the intermediate parent");
        let _ = helper.wait();

        let deadline = Instant::now() + Duration::from_secs(15);
        while pid_alive(child_pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let orphaned = pid_alive(child_pid);
        if orphaned {
            // Never leave a spinning orphan behind, even when this test fails.
            let _ = Command::new("kill").args(["-9", &child_pid.to_string()]).status();
        }
        assert!(
            !orphaned,
            "pid {child_pid} outlived its hard-killed parent by 15s — the liveness pipe did not fire"
        );
    }

    /// The per-file probe IS the discovery path: the CLI's router walks a directory itself and
    /// asks this once per file, so what a scan yields is exactly what these three answers say.
    /// Asserted over a directory rather than a lone file because the two files that must NOT
    /// become nodes are the ones a real scan puts in its way.
    #[test]
    fn discover_one_yields_subprocess_types_that_run_and_passes_over_the_rest() {
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

        assert!(discover_one(&dir.join("_hidden.py"), &py).is_none(), "`_`-prefixed is not a node");
        assert!(discover_one(&dir.join("nope.py"), &py).is_none(), "no Node subclass → no type");
        let ty = discover_one(&dir.join("negate.py"), &py).expect("a real node file");
        assert_eq!(ty.manifest.type_name, "Negate");
        assert_eq!(ty.manifest.category, "subprocess");
        assert_eq!(ty.manifest.isolation, Isolation::Subprocess);

        // The factory builds a working node.
        let mut node = (ty.factory)(&ParamGroups::new());
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
