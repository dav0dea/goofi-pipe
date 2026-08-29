//! The subprocess Python tier: one GIL interpreter per node, one run per `[u32 seq][frame]`
//! request/response over iceoryx2 shared memory.

use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use iceoryx2::prelude::*;

use goofi_core::Data;
use goofi_node::{Inputs, Node, NodeCtx, NodeError, NodeResult, Outputs, ParamKey, Params};

/// Unique iceoryx2 service-name base per spawned subprocess, so concurrent nodes never collide.
static SUBPROC_SEQ: AtomicU64 = AtomicU64::new(0);

/// iceoryx2 byte-slice pool ceiling per publisher (matches the child's `serve` config).
const MAX_PAYLOAD: usize = 64 * 1024;

/// How long a request waits on a child that has stopped answering.
pub const TICK_TIMEOUT: Duration = Duration::from_secs(10);

/// The deadline for the FIRST request after a spawn, which also pays interpreter boot, the node
/// module's imports and `setup()` — seconds, not milliseconds, for a heavy import like numba.
pub const COLD_START_TIMEOUT: Duration = Duration::from_secs(60);

/// The iceoryx2 node + its ports. The node must outlive the ports it created.
struct Ports {
    _node: iceoryx2::node::Node<ipc_threadsafe::Service>,
    req_pub: BytePublisher,
    resp_sub: ByteSubscriber,
}

/// The spawned child plus the iceoryx2 ports it talks over.
struct Running {
    child: Child,
    ports: Ports,
    seq: u32,
    /// Never written to: holding this end open IS the signal, and its close — including one the
    /// OS does on a crash, where no `Drop` runs — is the EOF the child exits on.
    parent_alive: Option<std::io::PipeWriter>,
}

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
        let mut cmd = Command::new(python);
        cmd.arg("-c")
            .arg("import goofi; goofi.serve()")
            .env("GOOFI_NODE_SRC", source)
            .env("GOOFI_IOX_REQ", &req_name)
            .env("GOOFI_IOX_RESP", &resp_name)
            // The host's PYTHONPATH (the pyo3/FT tier's) must not shadow the child's own numpy/goofi.
            .env_remove("PYTHONPATH")
            .env_remove("PYTHONHOME")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        // Armed BEFORE the spawn it guards, so a Ctrl-C or a crash here cannot orphan the child.
        let armed = goofi_codec::liveness::arm(&mut cmd).map_err(|e| format!("liveness pipe: {e}"))?;
        let mut child = cmd.spawn().map_err(|e| format!("spawn `{python}`: {e}"))?;
        let parent_alive = Some(armed.into_writer());
        match build_ports(&req_name, &resp_name) {
            Ok(ports) => Ok(Running { child, ports, seq: 0, parent_alive }),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(e)
            }
        }
    }

    fn roundtrip(&mut self, frame: &[u8], timeout: Duration) -> std::result::Result<Vec<u8>, String> {
        self.seq = self.seq.wrapping_add(1);
        one_roundtrip(&self.ports.req_pub, &self.ports.resp_sub, &mut self.child, self.seq, frame, timeout)
            .map_err(|e| format!("subprocess io: {e}"))
    }

    fn shutdown(&mut self) {
        // Closed first: this stop reaches the child even where a signal or a dead handle defeats `kill`.
        drop(self.parent_alive.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

type BytePublisher = iceoryx2::port::publisher::Publisher<ipc_threadsafe::Service, [u8], ()>;
type ByteSubscriber = iceoryx2::port::subscriber::Subscriber<ipc_threadsafe::Service, [u8], ()>;

/// One request/response: publish `[seq][frame]` and poll for the reply with the matching sequence.
/// Re-published each idle millisecond, because the child's subscriber may still be connecting.
fn one_roundtrip(
    req_pub: &BytePublisher,
    resp_sub: &ByteSubscriber,
    child: &mut Child,
    seq: u32,
    frame: &[u8],
    timeout: Duration,
) -> std::io::Result<Vec<u8>> {
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
                }
                Ok(None) => break, // drained; re-publish + wait
                Err(e) => return Err(std::io::Error::other(format!("iox receive: {e}"))),
            }
        }
        // Checked AFTER draining, so a child that answered and then exited still gets its answer returned.
        if let Ok(Some(status)) = child.try_wait() {
            return Err(std::io::Error::other(format!("subprocess exited: {status}")));
        }
        if Instant::now() >= deadline {
            return Err(std::io::Error::other("subprocess did not respond in time"));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A Python node in an isolated GIL subprocess, spawned lazily on its first `process`.
pub struct RemoteNode {
    python: String,
    source: String,
    /// Declared INPUT slot names only: the child is authoritative for output naming.
    in_slots: Vec<&'static str>,
    proc: Option<Running>,
}

impl RemoteNode {
    pub fn new(python: impl Into<String>, source: impl Into<String>, in_slots: Vec<&'static str>) -> RemoteNode {
        RemoteNode {
            python: python.into(),
            source: source.into(),
            in_slots,
            proc: None,
        }
    }

    fn ensure(&mut self) -> std::result::Result<&mut Running, String> {
        if self.proc.is_none() {
            self.proc = Some(Running::spawn(&self.python, &self.source)?);
        }
        Ok(self.proc.as_mut().unwrap())
    }

    fn reset(&mut self) {
        if let Some(mut p) = self.proc.take() {
            p.shutdown();
        }
    }

    /// One request to the child, spawning it first if need be; an IO failure drops the child so
    /// the next request starts a fresh one.
    fn ask(&mut self, frame: &[u8]) -> Result<goofi_codec::Response, String> {
        let timeout = if self.proc.is_none() { COLD_START_TIMEOUT } else { TICK_TIMEOUT };
        let resp = self.ensure().and_then(|r| r.roundtrip(frame, timeout)).inspect_err(|_| self.reset())?;
        goofi_codec::decode_response(&resp)
    }
}

impl Node for RemoteNode {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        // Only the PRESENT slots cross the wire; the child rebuilds the declared kwarg set from `INPUTS`.
        let present: Vec<(&str, &Data)> =
            self.in_slots.iter().filter_map(|name| inp.get(name).map(|d| (*name, d))).collect();
        // A node RAISE does not kill the child: its state is preserved and the error is instant.
        match self.ask(&goofi_codec::encode_request(p.groups(), &present)).map_err(NodeError)? {
            goofi_codec::Response::Slots(outs) => {
                for (slot, data) in outs {
                    out.set(&slot, data);
                }
                Ok(())
            }
            goofi_codec::Response::NodeError(msg) => Err(NodeError(msg)),
            goofi_codec::Response::Options(_) => Err(NodeError("the child answered a tick with options".into())),
        }
    }

    fn on_param_refreshed(&mut self, key: &ParamKey, p: &Params<'_>) -> Option<Vec<String>> {
        match self.ask(&goofi_codec::encode_refresh_request(p.groups(), &key.group, &key.name)) {
            Ok(goofi_codec::Response::Options(options)) => options,
            _ => None,
        }
    }
}

impl Drop for RemoteNode {
    fn drop(&mut self) {
        self.reset();
    }
}

use std::path::Path;

use goofi_node::discover::{Discovered, NodeFactory};
use goofi_node::{Isolation, NodeManifest};

/// A discovered subprocess node type, ready to `register_dyn_type` into a Graph.
pub struct SubprocNodeType {
    pub manifest: &'static NodeManifest,
    pub isolation: &'static goofi_node::IsolationCell,
    pub factory: NodeFactory,
}

use crate::Discovery;

/// Probe one file for this tier, reporting all three outcomes.
pub fn probe(path: &Path, python: &str) -> Discovery {
    goofi_node::discover::discover_one(path, python, "subprocess", Isolation::Subprocess)
}

/// Turn a probe-[`Discovered`] into a [`SubprocNodeType`], without a second spawn.
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
    SubprocNodeType { manifest, isolation: d.isolation, factory }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Data, Meta, Value};
    use goofi_node::ParamGroups;
    use indexmap::IndexMap;

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

    /// Tick a node once, returning the output slot map or the node error.
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

    /// A python with both goofi and numpy, or None. It strips `PYTHONPATH` exactly like the real
    /// child spawn, so a host `PYTHONPATH` cannot produce a false negative.
    fn usable_python() -> Option<String> {
        let mut cands: Vec<String> = Vec::new();
        if let Ok(p) = std::env::var("GOOFI_SUBPROC_TEST_PYTHON") {
            cands.push(p);
        }
        // Both venv layouts, because on Windows `python3` is a Store-advert alias that never fails.
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

    /// Serializes the subprocess-tier tests; each of them spawns a Python interpreter.
    static TIER: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The interpreter to spawn children with, plus the tier lock held for the rest of the test.
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

    /// Like [`usable_python`] but panics with an actionable message: these tests never skip.
    fn require_python() -> Tier {
        // A panicking test poisons the mutex; recover so its failure does not bury every sibling.
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
        let py = require_python();
        let mut node = RemoteNode::new(&*py, DOUBLE, vec!["data"]);
        // A real tick first, so the child is fully up and inside its serve loop.
        assert_eq!(floats(&run(&mut node, arr(vec![1], &[2.0], Meta::empty()))), vec![4.0]);

        let running = node.proc.as_mut().expect("the tick spawned a child");
        drop(running.parent_alive.take()); // the parent "dies"

        let exited = child_exits_within(&mut running.child, Duration::from_secs(5));
        if !exited {
            let _ = running.child.kill();
            let _ = running.child.wait();
        }
        assert!(exited, "the child must exit on the liveness pipe's EOF; it was still alive after 5s");
    }

    /// The interpreter the helper process spawns its grandchild with.
    #[cfg(target_os = "linux")]
    const HELPER_ENV: &str = "GOOFI_LIVENESS_HELPER_PYTHON";

    /// The intermediate parent for the hard-kill test, re-entered as a separate process: with
    /// [`HELPER_ENV`] set it spawns a child, announces its pid and blocks; unset it does nothing.
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
        let py = require_python();
        // libtest names a test by its module path MINUS the crate root; derived, so a move cannot
        // leave the filter matching nothing.
        let module = module_path!().split_once("::").map_or(module_path!(), |(_, rest)| rest);
        let helper_test = format!("{module}::liveness_helper_process");
        let mut helper = Command::new(std::env::current_exe().expect("test binary path"))
            .args(["--exact", &helper_test, "--nocapture", "--test-threads=1"])
            .env(HELPER_ENV, &*py)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn the intermediate parent");

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
            let _ = helper.kill();
            let _ = helper.wait();
        }
        let child_pid: u32 =
            announced.expect("the helper must announce its child's pid").parse().expect("a numeric pid");

        let killed = Command::new("kill").args(["-9", &helper.id().to_string()]).status();
        assert!(killed.is_ok_and(|s| s.success()), "kill -9 the intermediate parent");
        let _ = helper.wait();

        let deadline = Instant::now() + Duration::from_secs(15);
        while pid_alive(child_pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let orphaned = pid_alive(child_pid);
        if orphaned {
            let _ = Command::new("kill").args(["-9", &child_pid.to_string()]).status();
        }
        assert!(
            !orphaned,
            "pid {child_pid} outlived its hard-killed parent by 15s — the liveness pipe did not fire"
        );
    }

}
