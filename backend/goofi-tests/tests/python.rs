//! Python nodes, driven the way a user drives them: a `.py` file in the patch's workspace, probed,
//! registered, added to the graph and run — on whichever tier its imports allow. The in-process
//! tier needs `--features embed`, which LINKS libpython.

use std::time::{Duration, Instant};

use goofi_tests::{f32s, install, require_python, j, Goofi, Uid};

/// Set a consumer to free-run, so it produces with nothing wired upstream.
fn free_run(g: &Goofi, uid: Uid, hz: f64) {
    g.set_param(uid, "common", "autotrigger", true);
    g.set_param(uid, "common", "max_frequency", hz);
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
    let py = require_python();
    let g = Goofi::new();
    let ty = install(&g, &py.py, "affine.py", AFFINE);
    assert_eq!(ty, "Affine", "the type is named after the file stem");

    let row = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|t| t["type"] == "Affine").expect("Affine is in the palette").clone();
    assert_eq!(row["input_slots"]["data"], "ARRAY", "{row}");
    assert_eq!(row["output_slots"]["out"], "ARRAY", "{row}");
    assert_eq!(row["params"]["gain"]["factor"]["value"], 1, "{row}");

    let src = g.add("_TestConst");
    g.set_param(src, "constant", "value", 1.0);
    let node = g.add("Affine");
    let probe = g.probe(node, "out");
    g.set_param(node, "gain", "factor", 3);
    g.link(src, "out", node, "data");

    // 1*3 + 10 = 13: the 10 proves `setup` ran in the child, the 3 that a live param crossed.
    g.until("the affine node's frame", |_| probe.latest().filter(|d| f32s(d)[0] == 13.0));

    g.set_param(node, "gain", "factor", 0);
    g.until("the re-parameterized node", |_| {
        probe.latest().filter(|d| f32s(d)[0] == 10.0).map(|_| ())
    });
    assert!(g.error(node).is_none(), "a healthy python node carries no error");
}

#[test]
fn a_python_node_that_raises_reports_it_and_the_child_carries_on() {
    // A raise inside `process` is a per-run error: the SAME child answers the next run.
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
    g.set_param(src, "constant", "value", -1.0);
    g.link(src, "out", node, "data");

    let why = g.until("the raise to surface", |g| g.error(node));
    assert!(why.contains("the sensor read negative"), "the Python exception text rides back: {why}");

    // A count above 1 proves the child never died — a respawn would reset `_runs`.
    g.set_param(src, "constant", "value", 1.0);
    let d = g.until("the recovered node", |_| probe.latest().filter(|d| f32s(d)[0] > 1.0));
    assert!(f32s(&d)[0] > 1.0, "the child survived the raise with its state: {:?}", f32s(&d));
    g.until("the error to clear", |g| g.error(node).is_none().then_some(()));
}

#[test]
fn a_python_node_writing_to_stdout_does_not_corrupt_the_transport() {
    // The child routes fd 1 to stderr, so a node that prints cannot inject bytes into the frames.
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

    // The frame VALUE is the oracle: a 2 says the whole initialization ran again on one instance.
    let d = g.until("the retried initialization", |_| probe.latest());
    assert_eq!(f32s(&d)[0], 2.0, "setup ran a second time and the node came up");
}

#[test]
fn a_node_missing_a_dependency_is_listed_greyed_rather_than_vanishing() {
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
    let row = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|t| t["type"] == "NeedsScipy").expect("the greyed row is in the palette").clone();
    assert_eq!(row["available"], false, "{row}");
    assert!(row["doc"].as_str().unwrap().contains("definitely_not_installed"), "{row}");
    g.refuse("node add", j!({ "type": "NeedsScipy" }));
}

#[test]
fn a_nodes_own_python_thread_runs_while_the_child_is_idle() {
    // A GIL held across the child's idle serve loop would starve a `setup()` thread for ever.
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
    // Slowly, so most of the window below is genuine idle.
    free_run(&g, node, 2.0);
    let first = f32s(&g.until("the ticker's first frame", |_| probe.latest()))[0];
    // A fixed window, not the next frame: a threshold calibrated on an idle machine reads a busy
    // machine as starved.
    let opened = Instant::now();
    let later = f32s(&g.until("a frame a second and a half later", |_| {
        (opened.elapsed() >= Duration::from_millis(1500)).then(|| probe.latest()).flatten()
    }))[0];
    assert!(later - first > 10.0, "the node's thread must run while the child idles: {first} → {later}");
}

#[cfg(feature = "embed")]
mod inproc {
    use std::time::Duration;

    use super::*;
    use goofi_tests::hex;
    use goofi_core::{Data, Meta, Param, Value};
    use goofi_node::{ParamGroups, Params};
use goofi_signal_sdk::{Inputs, Node, NodeCtx, Outputs};
    use goofi_python::inproc::PyNode;
    use goofi_python::subproc::RemoteNode;
    use indexmap::IndexMap;

    /// Run one node once with the named input and params, reading back `out`. Below the graph:
    /// what is compared is the MARSHALLING.
    fn once(node: &mut dyn Node, input: Option<Data>, params: &ParamGroups)
        -> (goofi_signal_sdk::NodeResult, Option<Data>) {
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

    /// The interpreter the subprocess half of a parity test spawns.
    fn subproc_python() -> String {
        let ft = goofi_python::inproc::interpreter_path()
            .expect("no FT interpreter (PYO3_PYTHON) — run `cargo run -p goofi-init`");
        std::env::var("GOOFI_SUBPROC_TEST_PYTHON").unwrap_or(ft)
    }

    /// One authored node exercised by both tiers. Its int32 output makes both run the shared
    /// cast-to-f32 and carry-input-meta paths.
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

    /// The same node, able to tell "called with None" from "never called at all".
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

        // In-process seeds params and runs setup; the child runs setup lazily on its first request.
        let mut here = PyNode::from_source(PARITY, vec!["data"], vec!["out"]).expect("PyNode");
        here.setup(&mut NodeCtx::new(), &Params::new(&params)).expect("in-process setup");
        let (_, a) = once(&mut here, Some(frame()), &params);
        let mut there = RemoteNode::new(&py, PARITY, vec!["data"]);
        let (_, b) = once(&mut there, Some(frame()), &params);

        let (a, b) = (a.expect("in-process frame"), b.expect("subprocess frame"));
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
        // The subprocess wire carries only the slots that HOLD a frame, so the child widens it back.
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

        assert_eq!(f32s(&once(&mut here, Some(frame()), &p).1.unwrap()), vec![2.0, 4.0, 6.0]);
        assert_eq!(f32s(&once(&mut there, Some(frame()), &p).1.unwrap()), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn several_python_nodes_run_at_once_inside_the_live_graph() {
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
        static SLEEPY_TIER: goofi_node::IsolationCell =
            goofi_node::IsolationCell::new(goofi_node::Isolation::InProcess);
        static SLEEPY: goofi_node::NodeManifest = goofi_node::NodeManifest {
            type_name: "Sleeper", category: "python", doc: "sleeps 150 ms per run",
            inputs: IN, outputs: OUT, params: &[],
            producer: false,
        };

        let g = Goofi::new();
        g.register_dyn(&SLEEPY, Box::new(|_| {
            Box::new(PyNode::from_source(SLEEPER, vec!["data"], vec!["out"]).expect("PyNode"))
        }), &SLEEPY_TIER);
        let src = g.add("_TestCounter");

        // ONE sleeper first, to learn what a single 150 ms run costs on THIS machine.
        let solo = g.add("Sleeper");
        let solo_probe = g.probe(solo, "out");
        g.link(src, "out", solo, "data");
        let t0 = Instant::now();
        g.until("the lone sleeper to emit", |_| solo_probe.latest());
        let one = t0.elapsed();
        g.call("node remove", j!({ "node": hex(solo) }));

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
        // Overlapping, four cost about one; serialized they cost four. Two is the only bar between.
        assert!(four < one * 2,
                "four python nodes took {four:?} against one node's {one:?} — they ran serialized");
        assert!(!PyNode::gil_enabled().unwrap(), "the GIL must stay disabled");
    }

    #[test]
    fn a_param_reference_reads_this_node_and_follows_its_edit() {
        let g = Goofi::new();
        g.state.graph.lock().unwrap().set_evaluator(std::sync::Arc::new(
            goofi_python::inproc::PyExprEvaluator::new().expect("the evaluator constructs")));
        let osc = g.add("Oscillator");
        let probe = g.probe(osc, "out");
        g.ready(osc);
        g.set_param(osc, "oscillator", "frequency", 8.0);
        // `me` is this node: the amplitude follows the node's OWN frequency param.
        let r = g.call("node param edit", j!({ "node": hex(osc), "param": "oscillator/amplitude",
            "expression": "me.params.oscillator.frequency / 4" }));
        assert!(r["error"].is_null(), "{r}");
        g.until("the amplitude to read 2 through `me`", |_| {
            probe.latest().filter(|d| f32s(d).iter().any(|v| v.abs() > 1.5)).map(|_| ())
        });
        // An authored edit of the referenced param re-binds its reader.
        g.set_param(osc, "oscillator", "frequency", 2.0);
        g.until("the edit to re-evaluate the reader", |_| {
            probe.latest().filter(|d| f32s(d).iter().all(|v| v.abs() < 0.9)).map(|_| ())
        });
        // The same reference across nodes, by name.
        let osc2 = g.add("Oscillator");
        let probe2 = g.probe(osc2, "out");
        g.ready(osc2);
        let name = g.doc()["nodes"][hex(osc)]["name"].as_str().unwrap().to_string();
        let r = g.call("node param edit", j!({ "node": hex(osc2), "param": "oscillator/amplitude",
            "expression": format!("nd('{name}').params.oscillator.frequency + 1") }));
        assert!(r["error"].is_null(), "{r}");
        g.until("the cross-node read to evaluate", |_| {
            probe2.latest().filter(|d| f32s(d).iter().any(|v| v.abs() > 2.0)).map(|_| ())
        });
    }

    #[test]
    fn the_patch_rate_global_re_rates_every_producer_at_once() {
        // `common.max_frequency` is BOUND to `globals.default_ufreq`, and a binding needs the evaluator.
        let g = Goofi::new();
        g.state.graph.lock().unwrap().set_evaluator(std::sync::Arc::new(
            goofi_python::inproc::PyExprEvaluator::new().expect("the evaluator constructs")));
        g.call("global edit", j!({ "name": "default_ufreq", "value": 5.0 }));

        let osc = g.add("Oscillator");
        let probe = g.probe(osc, "out");
        g.ready(osc);
        let bound = g.doc()["nodes"][hex(osc)]["params"]["common"]["max_frequency"].clone();
        assert_eq!((&bound["expr"], &bound["mode"]), (&j!("globals.default_ufreq"), &j!("expression")),
                   "the manifest's declared binding was seeded live, not flattened to a literal");

        // Counting emitted frames is the only way to see a rate: a stated value reads correct anyway.
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

        g.call("global edit", j!({ "name": "default_ufreq", "value": 60.0 }));
        g.until("every producer to be re-rated by one global edit",
                |_| (runs(Duration::from_millis(400)) > 8).then_some(()));

        // The evaluator's namespace: `math`'s names and `time()` are simply there, beside `np`.
        let r = g.call("node param edit", j!({ "node": hex(osc), "param": "oscillator/amplitude",
            "expression": "2 * sin(pi / 2) + exp(0) * (1 if time() > 0 else 0)" }));
        assert!(r["error"].is_null(), "the namespace compiles: {r}");
        // The doc keeps the STORED value; the evaluated one shows in `node state`'s text.
        // Observed in the output, where an amplitude IS observable: a ±1 sine cannot cross 2.
        g.until("math and time to evaluate into the output's swing", |_| {
            probe.latest().filter(|d| f32s(d).iter().any(|v| v.abs() > 2.0)).map(|_| ())
        });
    }
}
