//! The subprocess Python tier: one GIL interpreter per node, running the SAME `goofi.Node`
//! contract as the in-process tier through the SAME `goofi_pymod::exec` marshalling — so the two
//! cannot drift, and the scheduler never branches on which is hosting.
//!
//! It exists for deps that are not free-threading-safe, and because a separate interpreter has its
//! own object ownership, so parallel heavy-Python compute avoids the biased-refcount penalty.
//!
//! One run is one request/response over iceoryx2 shared memory, `[u32 seq][frame]` each way. The
//! `seq` is what makes a re-publish — needed while the child's subscriber is still connecting —
//! unable to return a stale frame. The parent also holds the write end of a LIVENESS PIPE the
//! child watches for EOF, so a crashed manager can never orphan a forever-spinning child.

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
/// How long a request waits on a child that has stopped answering. Public because it is the
/// tier's documented deadline, not an internal tuning knob.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

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
    /// for output naming, via its `OUTPUTS`), so the parent keeps no output list.
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
        // the child rebuilds the full declared kwarg set from its own `INPUTS`.
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

// ── Discovery: the SAME probe the in-process tier uses, so a file yields one manifest either way ─

use std::path::Path;

use goofi_node::discover::{Discovered, NodeFactory};
use goofi_node::{Isolation, NodeManifest};

/// A discovered subprocess node type, ready to `register_dyn_type` into a Graph.
pub struct SubprocNodeType {
    pub manifest: &'static NodeManifest,
    /// Builds a node instance (spawns lazily on first tick).
    pub factory: NodeFactory,
}

use crate::Discovery;

/// Probe one file for this tier, reporting all three outcomes — the ONE discovery door. The CLI
/// needs the failure REASON to list a node as unavailable, and one probe spawn must answer both
/// questions, so a caller pairs this with [`node_type_from`] rather than asking twice.
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

// The rest of this tier's suite lives in `goofi-tests`. What stayed is the liveness mechanism: it
// reaches `Running` and the child handle directly, and re-enters this very binary as an
// intermediate parent so a SIGKILL can be delivered to a process that gets no chance to clean up.
// Neither is reachable from outside, and publishing them so a test could name them would be worse.
#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Data, Meta, Value};
    use goofi_node::ParamGroups;
    use indexmap::IndexMap;

    // Node sources are authored to the `goofi.Node` class contract (the one authoring shape;
    // the bare `def process(x)` function is gone). Raw strings keep Python indentation verbatim.

    /// Doubles its `data` input into `out`.
    const DOUBLE: &str = r#"
import goofi
class Double(goofi.Node):
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
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
        // libtest names a test by its module path MINUS the crate root, so deriving the filter
        // rather than spelling it keeps this working when the module moves. It has moved: this
        // file was its own crate, where the name was `tests::…`; hard-coded, the filter silently
        // matched NOTHING the moment it became `subproc::tests::…` — no helper, no announced pid,
        // and a failure that reads as "the child was orphaned" rather than "nobody ran".
        let module = module_path!().split_once("::").map_or(module_path!(), |(_, rest)| rest);
        let helper_test = format!("{module}::liveness_helper_process");
        let mut helper = Command::new(std::env::current_exe().expect("test binary path"))
            .args(["--exact", &helper_test, "--nocapture", "--test-threads=1"])
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

}
