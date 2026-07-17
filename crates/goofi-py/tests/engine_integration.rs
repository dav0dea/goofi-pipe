//! End-to-end proof of the rewrite's core thesis: a real Python node (numpy,
//! free-threaded) is hosted inside the live engine `Graph` through the runtime
//! node-hosting seam (`register_dyn_type`), ticked by the same scheduler as
//! native nodes, and — because the GIL is disabled — several Python nodes run
//! *concurrently* on the scheduler's rayon pool.
//!
//! Runs only with the `embed` feature + a free-threaded interpreter, e.g.:
//!   PYO3_PYTHON=<python3.14t> LD_LIBRARY_PATH=<base>/lib \
//!     PYTHONPATH=<ft-venv site-packages> \
//!     cargo test -p goofi-py --features embed --test engine_integration
#![cfg(feature = "embed")]

use goofi_core::{DType, Param, Value};
use goofi_engine::Graph;
use goofi_node::{
    Isolation, Node, NodeManifest, OutputDecl, ParamDecl, SlotDecl,
};
use goofi_py::PyNode;

// A Python node type descriptor: one F32 "data" input (triggers), one "out".
static PY_IN: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: goofi_core::SlotType::Array,
    trigger_process: true,
    multi: false,
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
    factory: py_stub_factory,
};

fn first_f32(d: &goofi_core::Data) -> f32 {
    match d.value() {
        Value::Array(s) => f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()),
        _ => panic!("not an array"),
    }
}

/// Register `PyNode` as a runtime type whose factory compiles `source`.
fn register_py(g: &mut Graph, source: &'static str) {
    g.register_dyn_type(
        &PY_MANIFEST,
        Box::new(move |_p| Box::new(PyNode::from_source(source, "process").unwrap()) as Box<dyn Node>),
    );
}

#[test]
fn real_python_node_runs_inside_the_engine_graph() {
    assert!(!PyNode::gil_enabled().unwrap(), "interpreter must be free-threaded");

    let mut g = Graph::new();
    register_py(&mut g, "import numpy as np\ndef process(x):\n    return x * 2.0 + 1.0\n");

    // Native _TestConst -> Python node, wired and ticked by the engine.
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "value", Param::float(3.0, -1e9, 1e9)).unwrap();
    g.update_param(src, "constant", "length", Param::int(4, 1, 1_000_000)).unwrap();
    let py = g.add_node("PyNode", None).unwrap();
    g.add_link(src, "out", py, "data").unwrap();

    g.tick();

    // x=[3,3,3,3] -> x*2+1 = [7,7,7,7], produced by real numpy in-process.
    let f = g.latest_frame(py, "out").expect("python node produced a frame");
    if let Value::Array(s) = f.value() {
        assert_eq!(s.shape(), &[4]);
        assert_eq!(s.dtype(), DType::F32);
    } else {
        panic!("expected array");
    }
    assert_eq!(first_f32(&f), 7.0);
    assert!(!PyNode::gil_enabled().unwrap(), "GIL must stay disabled");
}

#[test]
fn real_evaluator_resolves_a_param_expression_end_to_end() {
    // The user-facing path: the pyo3 PyExprEvaluator injected via `set_evaluator` (exactly what
    // the CLI's `--features python` startup does) must actually evaluate a param expression each
    // tick and drive the node — not just compile. Without the evaluator wired the binding errors
    // "no expression evaluator available"; this proves the real evaluator resolves it.
    assert!(!PyNode::gil_enabled().unwrap(), "interpreter must be free-threaded");
    let mut g = Graph::new();
    g.set_evaluator(std::sync::Arc::new(goofi_py::PyExprEvaluator::new().unwrap()));

    let n = g.add_node("_TestConst", None).unwrap();
    g.update_param(n, "constant", "value", Param::float(1.0, -1e9, 1e9)).unwrap();
    g.update_param(n, "constant", "length", Param::int(3, 1, 1_000_000)).unwrap();

    // Bind `value` to a NON-literal expression: the engine must evaluate it via the real
    // evaluator and drive the output with the result (40+2 = 42, not the literal 1.0).
    g.set_expression(n, "constant", "value", "40 + 2", true, false).unwrap();
    assert!(
        g.param_expression(n, "constant", "value").unwrap().error.is_none(),
        "the real evaluator compiled the expression cleanly (got {:?})",
        g.param_expression(n, "constant", "value").unwrap().error
    );
    g.tick();
    assert_eq!(
        first_f32(&g.latest_frame(n, "out").unwrap()),
        42.0,
        "the expression evaluated and drove the node's output (not the literal 1.0)"
    );

    // A time expression resolves too (proves `t` is exposed, not only constants).
    g.set_expression(n, "constant", "value", "t*0 + 7", true, false).unwrap();
    g.tick();
    assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 7.0, "time expression resolves");
    assert!(!PyNode::gil_enabled().unwrap(), "GIL stays disabled");
}

#[test]
fn renaming_a_producer_keeps_the_real_evaluator_expression_resolving() {
    // Renaming a node referenced by `nd('src')` must rewrite the source AND have the real
    // pyo3 evaluator recompile it, so the expression still resolves through the new name
    // (not just a string rewrite that leaves the compiled refs pointing at the old name).
    assert!(!PyNode::gil_enabled().unwrap(), "interpreter must be free-threaded");
    let mut g = Graph::new();
    g.set_evaluator(std::sync::Arc::new(goofi_py::PyExprEvaluator::new().unwrap()));

    let src = g.add_node("_TestConst", None).unwrap();
    g.rename_node(src, "src").unwrap();
    g.update_param(src, "constant", "value", Param::float(5.0, -1e9, 1e9)).unwrap();
    g.update_param(src, "constant", "length", Param::int(1, 1, 1_000_000)).unwrap();

    let host = g.add_node("_TestConst", None).unwrap();
    g.set_expression(host, "constant", "value", "nd('src')", true, false).unwrap();
    g.tick();
    assert_eq!(first_f32(&g.latest_frame(host, "out").unwrap()), 5.0, "resolves before rename");

    let touched = g.rename_node(src, "signal").unwrap();
    assert_eq!(touched, vec![host], "the referrer is reported for rebroadcast");
    let info = g.param_expression(host, "constant", "value").unwrap();
    assert_eq!(info.source, "nd('signal')", "source rewritten");
    assert!(info.error.is_none(), "the real evaluator recompiled the rewritten source cleanly");
    g.tick();
    assert_eq!(
        first_f32(&g.latest_frame(host, "out").unwrap()),
        5.0,
        "still resolves end-to-end via nd('signal')"
    );
    assert!(!PyNode::gil_enabled().unwrap(), "GIL stays disabled");
}

#[test]
fn discovers_and_hosts_python_nodes_from_a_directory() {
    // A directory of node files, some valid, some not.
    let dir = std::env::temp_dir().join(format!("goofi_pydisc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("triple.py"),
        "import numpy as np\ndef process(x):\n    return x * 3.0\n",
    )
    .unwrap();
    std::fs::write(dir.join("_hidden.py"), "def process(x):\n    return x\n").unwrap();
    std::fs::write(dir.join("broken.py"), "def not_process(x):\n    return x\n").unwrap();

    let types = goofi_py::discover(&dir).unwrap();
    // Only triple.py is a valid, non-hidden node defining `process`.
    let names: Vec<&str> = types.iter().map(|t| t.manifest.type_name).collect();
    assert_eq!(names, vec!["Triple"]);

    // Register the discovered types into a graph and drive one end-to-end.
    let mut g = Graph::new();
    for t in types {
        g.register_dyn_type(t.manifest, t.factory);
    }
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "value", Param::float(2.0, -1e9, 1e9)).unwrap();
    g.update_param(src, "constant", "length", Param::int(3, 1, 1_000_000)).unwrap();
    let py = g.add_node("Triple", None).unwrap();
    g.add_link(src, "out", py, "data").unwrap();
    g.tick();
    assert_eq!(first_f32(&g.latest_frame(py, "out").unwrap()), 6.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn python_nodes_run_concurrently_in_the_scheduler() {
    // One source fans out to N Python nodes, all at topological level 1, so the
    // scheduler runs them concurrently. Each sleeps 25ms inside Python: with the
    // GIL disabled the whole level overlaps, so the tick finishes far under the
    // N*25ms a serialized (or GIL-bound) execution would take.
    const N: usize = 8;
    let mut g = Graph::new();
    register_py(
        &mut g,
        "import time\nimport numpy as np\ndef process(x):\n    time.sleep(0.025)\n    return x\n",
    );

    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "value", Param::float(1.0, -1e9, 1e9)).unwrap();
    for _ in 0..N {
        let py = g.add_node("PyNode", None).unwrap();
        g.add_link(src, "out", py, "data").unwrap();
    }

    g.tick(); // warm the pool + the interpreter threads
    let t = std::time::Instant::now();
    g.tick();
    let elapsed = t.elapsed();

    assert!(
        elapsed < std::time::Duration::from_millis(120),
        "{N} Python nodes each sleeping 25ms took {elapsed:?}; expected GIL-free concurrent \
         execution on the scheduler (sequential/GIL-bound would be ~{}ms)",
        N * 25
    );
    assert!(!PyNode::gil_enabled().unwrap(), "GIL must stay disabled under concurrency");
}
