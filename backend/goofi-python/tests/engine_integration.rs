//! End-to-end proof of the rewrite's core thesis: a real Python node (numpy,
//! free-threaded) is hosted inside the live engine `Graph` through the runtime
//! node-hosting seam (`register_dyn_type`), run by the same machinery as native
//! nodes, and — because the GIL is disabled — several Python nodes run
//! *concurrently*, each on its own thread.
//!
//! Runs only with the `embed` feature + a free-threaded interpreter, e.g.:
//!   PYO3_PYTHON=<python3.14t> LD_LIBRARY_PATH=<base>/lib \
//!     PYTHONPATH=<ft-venv site-packages> \
//!     cargo test -p goofi-python --features embed --test engine_integration
#![cfg(feature = "embed")]

use goofi_core::{Param, Value};
use goofi_engine::testing::OutputProbe;
use goofi_engine::Graph;
use goofi_node::{
    Isolation, Node, NodeManifest, OutputDecl, ParamDecl, SlotDecl,
};
use goofi_python::inproc::PyNode;

// A Python node type descriptor: one F32 "data" input (triggers), one "out".
static PY_IN: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: goofi_core::SlotType::Array,
    trigger_process: true,
    multi: false,
    required: false,
}];
static PY_OUT: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: goofi_core::SlotType::Array,
}];
static PY_PARAMS: &[ParamDecl] = &[];
fn py_stub_factory() -> Box<dyn Node> {
    unreachable!("PyNode instances come from the registered factory")
}
static PY_MANIFEST: NodeManifest = NodeManifest {
    type_name: "PyNode",
    category: "python",
    doc: "in-process free-threaded Python node",
    inputs: PY_IN,
    outputs: PY_OUT,
    params: PY_PARAMS,
    isolation: Isolation::InProcess,
    producer: false,
    factory: py_stub_factory,
};

fn first_f32(d: &goofi_core::Data) -> f32 {
    match d.value() {
        Value::Array(s) => f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()),
        _ => panic!("not an array"),
    }
}

/// A class-contract node body: doubles its `data` input into `out` (x*2 + 1).
const DOUBLE_SRC: &str = concat!(
    "import goofi\n",
    "import numpy as np\n",
    "class Double(goofi.Node):\n",
    "    def config_input_slots(self):\n",
    "        return {'data': goofi.DataType.ARRAY}\n",
    "    def config_output_slots(self):\n",
    "        return {'out': goofi.DataType.ARRAY}\n",
    "    def process(self, data):\n",
    "        return {'out': data.data * 2.0 + 1.0}\n",
);

/// Register `PyNode` as a runtime type whose factory compiles `source` (a `goofi.Node`
/// subclass with a single `data` input / `out` output).
fn register_py(g: &mut Graph, source: &'static str) {
    g.register_dyn_type(
        &PY_MANIFEST,
        Box::new(move |_p| Box::new(PyNode::from_source(source, vec!["data"], vec!["out"]).unwrap()) as Box<dyn Node>),
    );
}

#[test]
fn real_python_node_runs_inside_the_engine_graph() {
    assert!(!PyNode::gil_enabled().unwrap(), "interpreter must be free-threaded");

    let mut g = Graph::new();
    register_py(&mut g, DOUBLE_SRC);

    // Native _TestConst -> Python node, wired and ticked by the engine.
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "value", Param::float(3.0, -1e9, 1e9)).unwrap();
    g.update_param(src, "constant", "length", Param::int(4, 1, 1_000_000)).unwrap();
    let py = g.add_node("PyNode", None).unwrap();
    // Opened before the link: the data services keep no history, so a probe attached after the
    // node's first emit would have missed it.
    let out = OutputProbe::open(&g, py, "out");
    g.add_link(src, "out", py, "data").unwrap();

    // x=[3,3,3,3] -> x*2+1 = [7,7,7,7], produced by real numpy in-process. `wait_until` and not
    // "the first frame": the source's params and the link both reach it asynchronously, so an
    // earlier frame may carry the type defaults.
    let f = out.wait_until(&mut g, "carries x*2+1 of the source", |d| first_f32(d) == 7.0);
    if let Value::Array(s) = f.value() {
        assert_eq!(s.shape(), &[4]);
    } else {
        panic!("expected array");
    }
    assert!(!PyNode::gil_enabled().unwrap(), "GIL must stay disabled");
}

#[test]
fn lempel_ziv_runs_in_process_inline() {
    // The migrated class-contract LempelZiv fixture, hosted + ticked by the engine on the
    // in-process tier (probe-based discovery is covered in the discovery test).
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../nodes/lempel_ziv.py");
    let source: &'static str = Box::leak(std::fs::read_to_string(&path).unwrap().into_boxed_str());

    let mut g = Graph::new();
    g.register_dyn_type(
        &PY_MANIFEST,
        Box::new(move |_p| Box::new(PyNode::from_source(source, vec!["data"], vec!["out"]).unwrap()) as Box<dyn Node>),
    );
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "length", Param::int(8, 1, 1_000_000)).unwrap();
    let lz = g.add_node("PyNode", None).unwrap();
    let out = OutputProbe::open(&g, lz, "out");
    g.add_link(src, "out", lz, "data").unwrap();

    // LZ76 of a constant (mean-thresholded to all-zeros, length 8) is 2 — a finite result from
    // real numpy running in-process.
    out.wait_until(&mut g, "carries LZ76 of the constant", |d| first_f32(d) == 2.0);
    assert!(!PyNode::gil_enabled().unwrap(), "GIL stayed disabled");
}

#[test]
fn the_real_evaluator_runs_what_the_graphs_rewrite_produces() {
    // The user-facing chain, end to end and across the crate boundary: the graph REWRITES an
    // authored source (§5.3), the pyo3 evaluator compiles that rewritten form, and evaluating it
    // with the locals the rewrite named gives the answer. Driven through both halves rather than
    // through `set_expression` alone, because a graph that compiles cleanly proves only that the
    // text parsed — a rewrite that emitted a variable the evaluator cannot read would pass that.
    assert!(!PyNode::gil_enabled().unwrap(), "interpreter must be free-threaded");
    use goofi_node::ExprEvaluator;
    let ev = goofi_python::inproc::PyExprEvaluator::new().unwrap();

    let (rewritten, vars) = goofi_engine::expr_rewrite::rewrite("nd('src').out.mean() * globals.gain")
        .expect("the graph's rewrite");
    assert_eq!(rewritten, "__v0.mean() * __v1");
    let mut locals: std::collections::HashMap<String, Option<goofi_node::Local>> =
        std::collections::HashMap::new();
    locals.insert(vars[0].var().to_string(), Some(goofi_node::Local::Frame(f32_frame(&[3.0, 5.0]))));
    locals.insert(vars[1].var().to_string(), Some(goofi_node::Local::Value(Param::float(10.0, 0.0, 1e9))));

    let target = Param::float(0.0, -1e9, 1e9);
    let compiled = ev.compile(&rewritten).expect("the rewritten source compiles");
    let out = ev.eval(compiled.id, &goofi_node::EvalCtx { locals: &locals, t: 0.0, target: &target }).unwrap();
    assert_eq!(out.as_f64(), Some(40.0), "mean([3,5]) * 10");
    ev.release(compiled.id);

    // And a time expression names no variable at all — §2.1's "a binding with no variables".
    let (t_src, t_vars) = goofi_engine::expr_rewrite::rewrite("t*0 + 7").unwrap();
    assert!(t_vars.is_empty());
    let c = ev.compile(&t_src).unwrap();
    let out = ev
        .eval(c.id, &goofi_node::EvalCtx { locals: &std::collections::HashMap::new(), t: 4.0, target: &target })
        .unwrap();
    assert_eq!(out.as_f64(), Some(7.0));
    assert!(!PyNode::gil_enabled().unwrap(), "GIL stays disabled");
}

#[test]
fn renaming_a_producer_keeps_the_real_evaluator_expression_resolving() {
    // Renaming a node referenced by `nd('src')` must rewrite the AUTHORED source and have the real
    // pyo3 evaluator recompile the rewritten form — a rename that only edited the text would leave
    // the graph shipping a variable resolved against the old name.
    assert!(!PyNode::gil_enabled().unwrap(), "interpreter must be free-threaded");
    let mut g = Graph::new();
    g.set_evaluator(std::sync::Arc::new(goofi_python::inproc::PyExprEvaluator::new().unwrap()));

    let src = g.add_node("_TestConst", None).unwrap();
    g.rename_node(src, "src").unwrap();

    let host = g.add_node("_TestConst", None).unwrap();
    g.set_expression(host, "constant", "value", "nd('src')", true, false).unwrap();
    assert!(g.param_expression(host, "constant", "value").unwrap().error.is_none(), "resolves before rename");

    let touched = g.rename_node(src, "signal").unwrap();
    assert_eq!(touched, vec![host], "the referrer is reported for rebroadcast");
    let info = g.param_expression(host, "constant", "value").unwrap();
    assert_eq!(info.source, "nd('signal')", "authored source rewritten");
    assert!(info.error.is_none(), "and re-resolved + recompiled cleanly through the new name");

    // A name nothing answers is the control: the same call path must report it rather than
    // reporting healthy for every source it managed to compile.
    g.set_expression(host, "constant", "value", "nd('gone')", true, false).unwrap();
    assert_eq!(
        g.param_expression(host, "constant", "value").unwrap().error.as_deref(),
        Some("no node named `gone`"),
    );
    assert!(!PyNode::gil_enabled().unwrap(), "GIL stays disabled");
}

/// A 1-D f32 frame, the shape a producer's output takes across the locals seam.
fn f32_frame(vals: &[f32]) -> goofi_core::Data {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    goofi_core::Data::array_f32(vec![vals.len()], bytes, goofi_core::Meta::empty()).unwrap()
}

/// The FT interpreter with `goofi` importable, for the discovery probe. Prefers
/// $GOOFI_PYMOD_TEST_PYTHON, else the build-time PYO3_PYTHON — the `.gfivenv-ft` interpreter
/// `cargo run -p goofi-init` created, pointed cargo at, and installed the goofi wheel into.
fn probe_python() -> String {
    let mut cands: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("GOOFI_PYMOD_TEST_PYTHON") {
        if !p.is_empty() {
            cands.push(p);
        }
    }
    cands.extend(goofi_python::inproc::interpreter_path());
    for cand in &cands {
        let ok = std::process::Command::new(cand)
            .args(["-c", "import goofi"])
            .env_remove("PYTHONPATH")
            .env_remove("PYTHONHOME")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return cand.clone();
        }
    }
    // Check the precondition HERE: without it this fails far downstream as an empty discovery
    // result, which reads like a discovery bug rather than a missing wheel.
    panic!(
        "no python with `goofi` importable for the discovery probe (tried {cands:?}). \
         Run `cargo run -p goofi-init`, which creates both venvs and installs the goofi wheel \
         into them — or set GOOFI_PYMOD_TEST_PYTHON."
    )
}

#[test]
fn discovers_and_hosts_python_nodes_from_a_directory() {
    let py = probe_python();
    let dir = std::env::temp_dir().join(format!("goofi_pydisc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let triple = concat!(
        "import goofi\n",
        "import numpy as np\n",
        "class Triple(goofi.Node):\n",
        "    def config_input_slots(self):\n",
        "        return {'data': goofi.DataType.ARRAY}\n",
        "    def config_output_slots(self):\n",
        "        return {'out': goofi.DataType.ARRAY}\n",
        "    def process(self, data):\n",
        "        return {'out': data.data * 3.0}\n",
    );
    std::fs::write(dir.join("triple.py"), triple).unwrap();
    std::fs::write(dir.join("_hidden.py"), triple.replace("Triple", "Hidden")).unwrap();
    std::fs::write(dir.join("broken.py"), "import nonexistent_dep_xyz\n").unwrap();

    // One probe per file, exactly as the CLI's router walks a directory — so the three files a
    // real scan trips over are the three asserted here.
    let types: Vec<_> = ["triple.py", "_hidden.py", "broken.py"]
        .iter()
        .filter_map(|f| match goofi_python::inproc::probe(&dir.join(f), &py) {
            goofi_python::Discovery::Found(d) => Some(goofi_python::inproc::node_type_from(d)),
            _ => None,
        })
        .collect();
    let names: Vec<&str> = types.iter().map(|t| t.manifest.type_name).collect();
    assert_eq!(names, vec!["Triple"], "only the valid, non-hidden node discovers");

    let mut g = Graph::new();
    for t in types {
        g.register_dyn_type(t.manifest, t.factory);
    }
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "value", Param::float(2.0, -1e9, 1e9)).unwrap();
    g.update_param(src, "constant", "length", Param::int(3, 1, 1_000_000)).unwrap();
    let py_node = g.add_node("Triple", None).unwrap();
    let out = OutputProbe::open(&g, py_node, "out");
    g.add_link(src, "out", py_node, "data").unwrap();
    out.wait_until(&mut g, "carries 3x the source", |d| first_f32(d) == 6.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn python_nodes_run_concurrently() {
    // One source fans out to N Python nodes, each of which sleeps 25 ms inside Python on every
    // run. The source is a producer paced at the patch's `default_ufreq` (30 Hz), so a node that
    // overlaps with its siblings keeps up with it — while N nodes taking it in turns would manage
    // 1/(N*25ms) = 5 Hz each. The rate each node SUSTAINS is therefore the oracle, and it
    // discriminates by a factor of six.
    //
    // Restated from a per-tick duration, which no longer exists: every node has its own thread by
    // construction now. What is still worth pinning is that nothing SERIALIZES them — a shared
    // lock in the host, or a pyo3 attach that queued, would show up here exactly as the GIL did.
    const N: usize = 8;
    let mut g = Graph::new();
    register_py(
        &mut g,
        concat!(
            "import goofi\n",
            "import time\n",
            "import numpy as np\n",
            "class Sleep(goofi.Node):\n",
            "    def config_input_slots(self):\n",
            "        return {'data': goofi.DataType.ARRAY}\n",
            "    def config_output_slots(self):\n",
            "        return {'out': goofi.DataType.ARRAY}\n",
            "    def process(self, data):\n",
            "        time.sleep(0.025)\n",
            "        return {'out': data.data}\n",
        ),
    );

    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "value", Param::float(1.0, -1e9, 1e9)).unwrap();
    let mut probes = Vec::new();
    for _ in 0..N {
        let py = g.add_node("PyNode", None).unwrap();
        probes.push(OutputProbe::open(&g, py, "out"));
        g.add_link(src, "out", py, "data").unwrap();
    }

    // Counted from `meta["index"]`, which advances once per emit, rather than from how many frames
    // a probe catches: the data services are latest-wins one deep. Waiting for each node's FIRST
    // frame is also the warm-up — every interpreter is up and running by the time the clock starts.
    let index = |d: &goofi_core::Data| d.meta().index().expect("every emit is stamped");
    let first: Vec<u64> = probes
        .iter()
        .map(|p| index(&p.expect_frame(&mut g, "each python node to emit")))
        .collect();
    let t = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_secs(2));
    let secs = t.elapsed().as_secs_f64();

    let rates: Vec<f64> = probes
        .iter()
        .zip(&first)
        .map(|(p, start)| {
            let now = index(&p.latest().expect("a node that emitted once keeps emitting"));
            now.saturating_sub(*start) as f64 / secs
        })
        .collect();
    let slowest = rates.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        slowest >= 15.0,
        "{N} Python nodes each sleeping 25ms sustained {rates:?} frames/s; taking turns would be \
         ~{:.1} Hz each, overlapping keeps up with the 30 Hz source",
        1.0 / (N as f64 * 0.025),
    );
    assert!(!PyNode::gil_enabled().unwrap(), "GIL must stay disabled under concurrency");
}
