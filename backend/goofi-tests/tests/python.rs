//! Python nodes, driven the way a user drives them: a `.py` file in the patch's workspace, probed,
//! registered, added to the graph and run — on whichever tier its imports allow.
//!
//! Both tiers run the SAME `goofi.Node` contract through one marshalling seam, so the suite's
//! centre is the pair of assertions that they cannot drift. The subprocess half is unconditional
//! (that tier only ever SPAWNS an interpreter); the in-process half needs `--features embed`,
//! which LINKS libpython:
//!
//!   cargo test -p goofi-tests --features embed --test python
//!
//! Three properties keep test binaries of their own, each because it needs a FRESH interpreter:
//! `python_gil_tripwire`, `python_module_hygiene`, `python_init_order`.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use goofi_core::Data;
use goofi_tests::{hex, j, Goofi, Uid};

fn f32s(d: &Data) -> Vec<f32> {
    let goofi_core::Value::Array(a) = d.value() else { panic!("not an array: {d:?}") };
    a.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Serializes this binary's tier tests. Cargo runs a crate's tests on parallel threads and every
/// one of these spawns an interpreter, so without this a latency or liveness assertion is taken
/// while a dozen siblings boot numpy on the same cores.
static TIER: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The interpreter to spawn children with, plus the tier lock — held for the rest of the test.
struct Tier {
    py: String,
    _lock: std::sync::MutexGuard<'static, ()>,
}

/// A python with BOTH goofi (the abi3 wheel) and numpy, or a PANIC naming the fix. These tests
/// HARD-REQUIRE one rather than skipping: a silent skip once hid real bugs for days.
///
/// The probe strips `PYTHONPATH` exactly as the real child spawn does, so a host/pyo3 `PYTHONPATH`
/// cannot produce a false negative — it once made the venv python import an incompatible numpy.
fn require_python() -> Tier {
    // A panicking test poisons the mutex; recover rather than cascade its failure onto every
    // sibling, which would bury the one real error.
    let _lock = TIER.lock().unwrap_or_else(|e| e.into_inner());
    let mut cands: Vec<String> = std::env::var("GOOFI_SUBPROC_TEST_PYTHON").into_iter().collect();
    // Both venv layouts — `bin/` on unix, `Scripts/` on Windows — because the fallbacks are worse
    // than a miss there: `python3` on Windows is an App Execution Alias that answers every probe
    // with a Microsoft Store advert instead of failing.
    cands.push(format!("{}/../../.gfivenv/bin/python", env!("CARGO_MANIFEST_DIR")));
    cands.push(format!("{}/../../.gfivenv/Scripts/python.exe", env!("CARGO_MANIFEST_DIR")));
    cands.push("python3".into());
    cands.push("python".into());
    for py in cands {
        let ok = Command::new(&py).arg("-c").arg("import goofi, numpy")
            .env_remove("PYTHONPATH").env_remove("PYTHONHOME")
            .stdout(Stdio::null()).stderr(Stdio::null())
            .status().is_ok_and(|s| s.success());
        if ok {
            return Tier { py, _lock };
        }
    }
    panic!("no python with goofi + numpy found (checked $GOOFI_SUBPROC_TEST_PYTHON, \
            ./.gfivenv/bin/python, python3, python). Run `cargo run -p goofi-init`, which creates \
            the venvs and installs the goofi wheel into them.");
}

/// Write `source` into the patch's own node directory, probe it the way the CLI's scan does, and
/// register what comes back. Answers the type name the palette now offers.
///
/// This is the real seam: `probe` + `node_type_from` are the pair the node scan routes with, so a
/// node reaching the graph through them reaches it the way a user's file does.
fn install(g: &Goofi, py: &str, file: &str, source: &str) -> String {
    let dir = g.state.mount().join("nodes");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(file);
    std::fs::write(&path, source).unwrap();
    match goofi_python::subproc::probe(&path, py) {
        goofi_python::Discovery::Found(d) => {
            let t = goofi_python::subproc::node_type_from(py, d);
            let name = t.manifest.type_name.to_string();
            g.register_dyn(t.manifest, t.factory);
            name
        }
        goofi_python::Discovery::Unavailable { reason, .. } =>
            panic!("{file} probed as unavailable: {reason}"),
        goofi_python::Discovery::Skip => panic!("{file} was not taken for a node file at all"),
    }
}

/// Set a consumer to free-run, so it produces without anything wired upstream — what a user does
/// by hand to a node that reads a device rather than an input slot.
fn free_run(g: &Goofi, uid: Uid, hz: f64) {
    g.call("update_param", j!({ "node": hex(uid), "group": "common", "name": "autotrigger", "value": true }));
    g.call("update_param", j!({ "node": hex(uid), "group": "common", "name": "max_frequency", "value": hz }));
}

const AFFINE: &str = r#"
import goofi
class Affine(goofi.Node):
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PARAMS = {"gain": {"factor": goofi.IntParam(1, 0, 100)}}
    def setup(self):
        self._base = 10
    def process(self, data):
        return {"out": data.data * self.params.gain.factor + self._base}
"#;

#[test]
fn a_python_file_in_the_workspace_becomes_a_node_that_runs_and_takes_its_params() {
    // The whole path, in the order a user lives it: write the file, let goofi probe it, add it,
    // wire it, watch it run — and edit a param under the running node.
    let py = require_python();
    let g = Goofi::new();
    let ty = install(&g, &py.py, "affine.py", AFFINE);
    assert_eq!(ty, "Affine", "the type is named after the file stem");

    // The palette row is projected from the probe's manifest, so it already knows the node's
    // shape — slots and params a caller would otherwise have to instantiate to learn.
    let row = g.call("list_nodes", j!({}))["types"].as_array().unwrap().iter()
        .find(|t| t["type"] == "Affine").expect("Affine is in the palette").clone();
    assert_eq!(row["input_slots"]["data"], "ARRAY", "{row}");
    assert_eq!(row["output_slots"]["out"], "ARRAY", "{row}");
    assert_eq!(row["params"]["gain"]["factor"]["value"], 1, "{row}");

    let src = g.add("_TestConst");
    g.call("update_param", j!({ "node": hex(src), "group": "constant", "name": "value", "value": 1.0 }));
    let node = g.add("Affine");
    let probe = g.probe(node, "out");
    g.call("update_param", j!({ "node": hex(node), "group": "gain", "name": "factor", "value": 3 }));
    g.link(src, "out", node, "data");

    // A constant 1 in, so the frame is 1*3 + 10 = 13 — the 10 proves `setup` ran in the child, the
    // 3 that a live param crossed the wire.
    g.until("the affine node's frame", |_| probe.latest().filter(|d| f32s(d)[0] == 13.0));

    // A param edit reaches the running child, with no restart.
    g.call("update_param", j!({ "node": hex(node), "group": "gain", "name": "factor", "value": 0 }));
    g.until("the re-parameterized node", |_| {
        probe.latest().filter(|d| f32s(d)[0] == 10.0).map(|_| ())
    });
    assert!(g.error(node).is_none(), "a healthy python node carries no error");
}

#[test]
fn a_python_node_that_raises_reports_it_and_the_child_carries_on() {
    // A raise inside `process` is a per-run error, not a crash: the SAME child answers the next
    // run with its state intact. Read through the node's own error, which is where a user sees it.
    let py = require_python();
    let g = Goofi::new();
    install(&g, &py.py, "boom.py", r#"
import goofi
import numpy as np
class Boom(goofi.Node):
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    def setup(self):
        self._runs = 0
    def process(self, data):
        self._runs += 1
        if data.data[0] < 0:
            raise ValueError("the sensor read negative")
        return {"out": np.array([float(self._runs)], dtype="float32")}
"#);
    let src = g.add("_TestConst");
    let node = g.add("Boom");
    let probe = g.probe(node, "out");
    g.call("update_param", j!({ "node": hex(src), "group": "constant", "name": "value", "value": -1.0 }));
    g.link(src, "out", node, "data");

    let why = g.until("the raise to surface", |g| g.error(node));
    assert!(why.contains("the sensor read negative"), "the Python exception text rides back: {why}");

    // The same child answers the next run, with its state intact — a respawn would reset `_runs`
    // and re-run `setup`, so a count above 1 is the proof that the child never died.
    g.call("update_param", j!({ "node": hex(src), "group": "constant", "name": "value", "value": 1.0 }));
    let d = g.until("the recovered node", |_| probe.latest().filter(|d| f32s(d)[0] > 1.0));
    assert!(f32s(&d)[0] > 1.0, "the child survived the raise with its state: {:?}", f32s(&d));
    g.until("the error to clear", |g| g.error(node).is_none().then_some(()));
}

#[test]
fn a_python_node_writing_to_stdout_does_not_corrupt_the_transport() {
    // The child routes fd 1 to stderr before importing the node, so a node that prints — here a
    // flushed `print`, but equally a C extension's `printf` — cannot inject bytes into the shared
    // memory frame plane. Two runs, because the first frame alone would not show a stream that
    // went out of sync behind it.
    let py = require_python();
    let g = Goofi::new();
    install(&g, &py.py, "chatty.py", r#"
import sys
import goofi
class Chatty(goofi.Node):
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    def process(self, data):
        print("debug from the node", flush=True)
        sys.stdout.flush()
        return {"out": data.data * 2.0}
"#);
    let src = g.add("_TestCounter");
    let node = g.add("Chatty");
    let probe = g.probe(node, "out");
    g.link(src, "out", node, "data");

    let first = f32s(&g.until("a frame from the printing node", |_| probe.latest()))[0];
    let second = g.until("a later frame, still in sync", |_| {
        probe.latest().filter(|d| f32s(d)[0] > first)
    });
    assert_eq!(f32s(&second)[0] % 2.0, 0.0, "every frame is the doubled counter: {:?}", f32s(&second));
    assert!(g.error(node).is_none(), "the transport stayed in sync: {:?}", g.error(node));
}

#[test]
fn a_node_whose_setup_raises_retries_the_whole_initialization_on_the_next_wake() {
    // A device that was not ready at the first run comes back on the next one, without a restart —
    // the child used to mark `did_setup` BEFORE running setup, so a node that failed to open once
    // reported the same failure for ever.
    let py = require_python();
    let g = Goofi::new();
    install(&g, &py.py, "late_boot.py", r#"
import goofi
import numpy as np
class LateBoot(goofi.Node):
    setups = 0
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    def setup(self):
        LateBoot.setups += 1
        if LateBoot.setups < 2:
            raise RuntimeError("the device is not open")
    def process(self, data):
        return {"out": np.array([float(LateBoot.setups)], dtype="float32")}
"#);
    let src = g.add("_TestCounter");
    let node = g.add("LateBoot");
    let probe = g.probe(node, "out");
    g.link(src, "out", node, "data");

    // The frame VALUE is the oracle: `process` returns the setup count, so a 2 says the first setup
    // raised and the WHOLE initialization ran again on the same instance. The standing error is
    // deliberately not asserted here — the retry clears it about 2 ms later (measured), and no rate
    // cap widens that: the retry is the node's own next wake, not the next input frame. That a
    // failed setup reports why and blocks `process` until it succeeds is `running.rs`'s
    // `each_way_a_node_can_fail_is_reported_…`, where the failure stands instead of passing.
    let d = g.until("the retried initialization", |_| probe.latest());
    assert_eq!(f32s(&d)[0], 2.0, "setup ran a second time and the node came up");
}

#[test]
fn a_node_missing_a_dependency_is_listed_greyed_rather_than_vanishing() {
    // A node file that cannot load must EXPLAIN itself: silently not appearing reads as "goofi
    // ignored my file" rather than "install this dependency".
    let py = require_python();
    let g = Goofi::new();
    let dir = g.state.mount().join("nodes");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("needs_scipy.py");
    std::fs::write(&path, "import goofi\nimport definitely_not_installed\n").unwrap();

    match goofi_python::subproc::probe(&path, &py.py) {
        goofi_python::Discovery::Unavailable { type_name, reason } => {
            assert_eq!(type_name, "NeedsScipy");
            assert!(reason.contains("definitely_not_installed"), "the reason names the module: {reason}");
            g.state.graph.lock().unwrap().register_unavailable(type_name, reason);
        }
        goofi_python::Discovery::Found(_) =>
            panic!("a node with a missing import must not probe as loadable"),
        goofi_python::Discovery::Skip => panic!("the file was not taken for a node file at all"),
    }
    let row = g.call("list_nodes", j!({}))["types"].as_array().unwrap().iter()
        .find(|t| t["type"] == "NeedsScipy").expect("the greyed row is in the palette").clone();
    assert_eq!(row["available"], false, "{row}");
    assert!(row["doc"].as_str().unwrap().contains("definitely_not_installed"), "{row}");
    g.refuse("add_node", j!({ "type": "NeedsScipy" }));
}

#[test]
fn a_nodes_own_python_thread_runs_while_the_child_is_idle() {
    // The subprocess tier exists to host GIL-bound libraries, and the canonical shape of a device
    // input is a receiver thread started in `setup()`. The child's serve loop is pure Rust between
    // requests, so if it held the GIL across its idle wait, that thread would be starved for
    // exactly as long as the node is not being run — which for an unwired node is for ever.
    let py = require_python();
    let g = Goofi::new();
    install(&g, &py.py, "ticker.py", r#"
import threading, time
import numpy as np
import goofi
class Ticker(goofi.Node):
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    def setup(self):
        self.count = 0
        def spin():
            while True:
                self.count += 1
                time.sleep(0.001)
        threading.Thread(target=spin, daemon=True).start()
    def process(self, data):
        return {"out": np.array([float(self.count)], dtype="float32")}
"#);
    let node = g.add("Ticker");
    let probe = g.probe(node, "out");
    // Slowly, so most of the window below is genuine idle: the child's serve loop is between
    // requests, which is exactly when a held GIL would starve the node's own thread.
    free_run(&g, node, 2.0);
    let first = f32s(&g.until("the ticker's first frame", |_| probe.latest()))[0];
    // Measure over a fixed window rather than "the next frame": what separates a RUNNING thread
    // from a starved one is whether it advances at all, and a threshold calibrated on an idle
    // machine reads a merely CONTENDED one as starved (31 increments in one 500 ms window,
    // measured under `--features embed`, against a bar of 50).
    let opened = Instant::now();
    let later = f32s(&g.until("a frame a second and a half later", |_| {
        (opened.elapsed() >= Duration::from_millis(1500)).then(|| probe.latest()).flatten()
    }))[0];
    // A starved thread only advances while `process` itself holds the eval loop — three runs at
    // 2 Hz, so a couple of increments. Ten is five times that and a small fraction of the ~90 a
    // running thread manages even under contention.
    assert!(later - first > 10.0, "the node's thread must run while the child idles: {first} → {later}");
}

// ---------------------------------------------------------------------------
// The in-process tier, and the two tiers held against each other
// ---------------------------------------------------------------------------

#[cfg(feature = "embed")]
mod inproc {
    use std::time::Duration;

    use super::*;
    use goofi_core::{Meta, Param, Value};
    use goofi_node::{Inputs, Node, NodeCtx, Outputs, ParamGroups, Params};
    use goofi_python::inproc::PyNode;
    use goofi_python::subproc::RemoteNode;
    use indexmap::IndexMap;

    /// Run one node once with the named input and params, reading back `out`. Deliberately below
    /// the graph: what is being compared is the MARSHALLING, and two engine runs would differ in
    /// timing long before they differed in a byte.
    fn once(node: &mut dyn Node, input: Option<Data>, params: &ParamGroups)
        -> (goofi_node::NodeResult, Option<Data>) {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", input);
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        let res = {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx, &Params::new(params))
        };
        (res, outmap.get("out").unwrap().clone())
    }

    fn frame() -> Data {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        Data::array_f32(vec![3], bytes, Meta::new().with_sfreq(Some(250.0))).unwrap()
    }

    /// The interpreter the subprocess half of a parity test spawns: the FT one by default, since
    /// it has goofi and numpy too.
    fn subproc_python() -> String {
        let ft = goofi_python::inproc::interpreter_path()
            .expect("no FT interpreter (PYO3_PYTHON) — run `cargo run -p goofi-init`");
        std::env::var("GOOFI_SUBPROC_TEST_PYTHON").unwrap_or(ft)
    }

    /// One authored node exercised by both tiers. It reads a param and a `setup` value, and returns
    /// a NON-f32 (int32) array — so the shared cast-to-f32 and the carry-input-meta path both run
    /// identically on each tier by construction.
    const PARITY: &str = r#"
import goofi
import numpy as np
class Parity(goofi.Node):
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PARAMS = {"gain": {"factor": goofi.IntParam(1, 0, 100)}}
    def setup(self):
        self._base = 10
    def process(self, data):
        return {"out": (data.data * self.params.gain.factor + self._base).astype(np.int32)}
"#;

    /// The same node, able to TELL the two absent cases apart. A source that merely tolerated a
    /// missing input could not distinguish "called with None" from "never called at all", which is
    /// precisely the difference the tiers have to agree on.
    const ABSENT: &str = r#"
import goofi
import numpy as np
class Absent(goofi.Node):
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    def process(self, data):
        if data is None:
            return {"out": np.array([-1.0], dtype=np.float32)}
        return {"out": data.data * 2.0}
"#;

    #[test]
    fn one_source_run_on_both_tiers_produces_the_same_frame() {
        let py = subproc_python();
        let mut params = ParamGroups::new();
        params.insert("gain".into(), IndexMap::from([("factor".to_string(), Param::int(3, 0, 100))]));

        // In-process: seed params + run setup, as the engine does. Subprocess: the child runs setup
        // lazily on its first request — two different mechanisms, one required answer.
        let mut here = PyNode::from_source(PARITY, vec!["data"], vec!["out"]).expect("PyNode");
        here.setup(&mut NodeCtx::new(), &Params::new(&params)).expect("in-process setup");
        let (_, a) = once(&mut here, Some(frame()), &params);
        let mut there = RemoteNode::new(&py, PARITY, vec!["data"]);
        let (_, b) = once(&mut there, Some(frame()), &params);

        let (a, b) = (a.expect("in-process frame"), b.expect("subprocess frame"));
        // 1*3+10, 2*3+10, 3*3+10 — the shared cast, the live param and setup's base, all at once.
        assert_eq!(f32s(&a), f32s(&b), "the two tiers must produce identical values");
        assert_eq!(f32s(&a), vec![13.0, 16.0, 19.0], "shared cast + param + setup base");
        assert_eq!((a.meta().sfreq(), b.meta().sfreq()), (Some(250.0), Some(250.0)),
                   "identical carried meta");
        match (a.value(), b.value()) {
            (Value::Array(sa), Value::Array(sb)) => assert_eq!(sa.shape(), sb.shape()),
            _ => panic!("both tiers must return arrays"),
        }
    }

    #[test]
    fn both_tiers_pass_a_declared_input_with_no_frame_as_none() {
        // The subprocess wire carries only the slots that HOLD a frame, so the child has to widen
        // the request back to its own declared slots — a different mechanism from the in-process
        // tier's direct list, which is exactly why the two need pinning against each other.
        let py = subproc_python();
        let p = ParamGroups::new();
        let mut here = PyNode::from_source(ABSENT, vec!["data"], vec!["out"]).expect("PyNode");
        here.setup(&mut NodeCtx::new(), &Params::new(&p)).expect("in-process setup");
        let (a_res, a) = once(&mut here, None, &p);
        let mut there = RemoteNode::new(&py, ABSENT, vec!["data"]);
        let (b_res, b) = once(&mut there, None, &p);

        assert!(a_res.is_ok(), "in-process tier errored on an absent input: {:?}", a_res.err());
        assert!(b_res.is_ok(), "subprocess tier errored on an absent input: {:?}", b_res.err());
        let a = a.expect("in-process emitted nothing — `process` was not called with None");
        let b = b.expect("subprocess emitted nothing — `process` was not called with None");
        assert_eq!(f32s(&a), f32s(&b), "the tiers must answer an absent input identically");
        assert_eq!(f32s(&a), vec![-1.0], "both took the node's own `data is None` branch");

        // The same nodes WITH a frame take the other branch, so the marker above is the node
        // choosing rather than the only thing this source can ever return.
        assert_eq!(f32s(&once(&mut here, Some(frame()), &p).1.unwrap()), vec![2.0, 4.0, 6.0]);
        assert_eq!(f32s(&once(&mut there, Some(frame()), &p).1.unwrap()), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn several_python_nodes_run_at_once_inside_the_live_graph() {
        // The rewrite's core thesis: with the GIL disabled, each Python node runs on its own
        // thread and they overlap. A node that SLEEPS makes that observable — serialized, four of
        // them would take four times as long as one.
        assert!(!PyNode::gil_enabled().unwrap(), "the interpreter must be free-threaded");
        const SLEEPER: &str = r#"
import time
import numpy as np
import goofi
class Sleeper(goofi.Node):
    INPUTS = {"data": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    def process(self, data):
        time.sleep(0.15)
        return {"out": data.data * 2.0}
"#;
        static IN: &[goofi_node::SlotDecl] = &[goofi_node::SlotDecl {
            name: "data", kind: goofi_core::SlotType::Array,
            trigger_process: true, multi: false, required: false }];
        static OUT: &[goofi_node::OutputDecl] =
            &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }];
        static SLEEPY: goofi_node::NodeManifest = goofi_node::NodeManifest {
            type_name: "Sleeper", category: "python", doc: "sleeps 150 ms per run",
            inputs: IN, outputs: OUT, params: &[],
            isolation: goofi_node::Isolation::InProcess, producer: false,
            factory: || unreachable!("a dyn type is built by its registered factory"),
        };

        let g = Goofi::new();
        g.register_dyn(&SLEEPY, Box::new(|_| {
            Box::new(PyNode::from_source(SLEEPER, vec!["data"], vec!["out"]).expect("PyNode"))
        }));
        let src = g.add("_TestCounter");

        // ONE sleeper first, to learn what a single 150 ms run costs on THIS machine — the wire's
        // own latency and whatever else the box is doing, included. Everything below is measured
        // against that rather than against a constant: a budget in milliseconds sits close to the
        // serialized answer by construction (4 × 150 ms is only 600), and the gap it has to
        // separate is the one that shrinks first when the machine is busy.
        let solo = g.add("Sleeper");
        let solo_probe = g.probe(solo, "out");
        g.link(src, "out", solo, "data");
        let t0 = Instant::now();
        g.until("the lone sleeper to emit", |_| solo_probe.latest());
        let one = t0.elapsed();
        g.call("remove_node", j!({ "node": hex(solo) }));

        let sleepers: Vec<_> = (0..4).map(|_| g.add("Sleeper")).collect();
        let probes: Vec<_> = sleepers.iter().map(|u| g.probe(*u, "out")).collect();
        for u in &sleepers {
            g.link(src, "out", *u, "data");
        }
        let t0 = Instant::now();
        for p in &probes {
            g.until("every sleeper to emit", |_| p.latest());
        }
        let four = t0.elapsed();
        // Overlapping, four cost about one; serialized they cost four. Two is the only bar that
        // sits between, whatever the machine.
        assert!(four < one * 2,
                "four python nodes took {four:?} against one node's {one:?} — they ran serialized");
        assert!(!PyNode::gil_enabled().unwrap(), "the GIL must stay disabled");
    }

    #[test]
    fn the_patch_rate_global_re_rates_every_producer_at_once() {
        // `common.max_frequency` is BOUND to `globals.default_ufreq` rather than copied from it, so
        // one global edit re-paces every producer live. The binding needs an EVALUATOR, which is
        // the pyo3 one — so this is the tier's own property as much as the graph's.
        let g = Goofi::new();
        g.state.graph.lock().unwrap().set_evaluator(std::sync::Arc::new(
            goofi_python::inproc::PyExprEvaluator::new().expect("the evaluator constructs")));
        g.call("set_global", j!({ "name": "default_ufreq", "value": 5.0, "type": "float" }));

        let osc = g.add("Oscillator");
        let probe = g.probe(osc, "out");
        g.ready(osc);
        let bound = g.doc()["nodes"][hex(osc)]["params"]["common"]["max_frequency"]["expr"].clone();
        assert_eq!((&bound["source"], &bound["enabled"]), (&j!("globals.default_ufreq"), &j!(true)),
                   "the manifest's declared binding was seeded live, not flattened to a literal");

        // Counting emitted frames is the only way to see a rate: the param's stated value reads
        // correct against a node that ignores it entirely.
        let runs = |window: Duration| {
            let (mut seen, mut last, end) = (0, None, std::time::Instant::now() + window);
            while std::time::Instant::now() < end {
                let now = probe.latest().map(|d| d.meta().index());
                if now.is_some() && now != last {
                    seen += 1;
                    last = now;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            seen
        };
        let slow = runs(Duration::from_millis(800));
        assert!(slow <= 8, "5 Hz produced {slow} frames in 0.8 s — the global is not pacing it");

        g.call("set_global", j!({ "name": "default_ufreq", "value": 60.0, "type": "float" }));
        g.until("every producer to be re-rated by one global edit",
                |_| (runs(Duration::from_millis(400)) > 8).then_some(()));
    }
}
