//! Latency + concurrency-scaling probe for the in-process Python node path.
//! Not a test — run under the FT env:
//!   PYO3_PYTHON=<python3.14t> LD_LIBRARY_PATH=<base>/lib PYTHONPATH=<ft-sp> \
//!     cargo run -p goofi-python --features embed --example py_latency --release
//!
//! Measures (1) per-tick latency of a single Python node ticked by the engine and
//! (2) how an N-wide fan-out of Python nodes scales on the parallel scheduler,
//! which only overlaps because the interpreter is free-threaded (GIL off).
use std::time::Instant;

use goofi_core::Param;
use goofi_engine::Graph;
use goofi_node::{Isolation, Node, NodeManifest, OutputDecl, ParamDecl, ParamGroups, SlotDecl};
use goofi_python::inproc::PyNode;

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
    unreachable!()
}
static PY_MANIFEST: NodeManifest = NodeManifest {
    type_name: "PyNode",
    category: "python",
    doc: "",
    inputs: PY_IN,
    outputs: PY_OUT,
    params: PY_PARAMS,
    isolation: Isolation::InProcess,
    producer: false,
    factory: py_stub_factory,
};

fn build(n: usize, src: &'static str, len: i64) -> Graph {
    let mut g = Graph::new();
    g.register_dyn_type(
        &PY_MANIFEST,
        Box::new(move |_| Box::new(PyNode::from_source(src, vec!["data"], vec!["out"]).unwrap()) as Box<dyn Node>),
    );
    let osc = g.add_node("_TestConst", None).unwrap();
    g.update_param(osc, "constant", "value", Param::float(0.5, -1e9, 1e9)).unwrap();
    g.update_param(osc, "constant", "length", Param::int(len, 1, 10_000_000)).unwrap();
    for _ in 0..n {
        let py = g.add_node("PyNode", None).unwrap();
        g.add_link(osc, "out", py, "data").unwrap();
    }
    g
}

fn bench(label: &str, g: &mut Graph, iters: u32) {
    for _ in 0..50 {
        g.tick();
    }
    let t = Instant::now();
    for _ in 0..iters {
        g.tick();
    }
    let per = t.elapsed().as_secs_f64() / iters as f64;
    println!("{label:<44} {:>8.1} us/tick  ({:>7.0} ticks/s)", per * 1e6, 1.0 / per);
}

fn main() {
    println!("GIL enabled: {}", PyNode::gil_enabled().unwrap());

    // (0) Root-cause micro-probe: bare `Python::attach(|_| {})` cost on the main
    // thread vs a freshly spawned worker thread. Isolates the per-call attach
    // overhead from any numpy work.
    {
        use pyo3::prelude::*;
        let iters = 20_000u32;
        let t = Instant::now();
        for _ in 0..iters {
            Python::attach(|_py| {});
        }
        let main_ns = t.elapsed().as_nanos() as f64 / iters as f64;

        let worker_ns = std::thread::spawn(move || {
            // Prime one attach, then time the steady state on this worker thread.
            Python::attach(|_| {});
            let t = Instant::now();
            for _ in 0..iters {
                Python::attach(|_py| {});
            }
            t.elapsed().as_nanos() as f64 / iters as f64
        })
        .join()
        .unwrap();
        println!(
            "bare Python::attach  main-thread {main_ns:>8.0} ns   worker-thread {worker_ns:>8.0} ns"
        );
    }

    // Real numpy work, length-preserving (the common signal-node shape).
    let work = concat!(
        "import goofi\n",
        "import numpy as np\n",
        "class Work(goofi.Node):\n",
        "    def config_input_slots(self):\n",
        "        return {'data': goofi.DataType.ARRAY}\n",
        "    def config_output_slots(self):\n",
        "        return {'out': goofi.DataType.ARRAY}\n",
        "    def process(self, data):\n",
        "        return {'out': np.tanh(data.data) * 2.0 - data.data.mean()}\n",
    );

    // (1) Single-node per-tick latency at a few array sizes.
    for len in [64i64, 1024, 16384] {
        let mut g = build(1, work, len);
        bench(&format!("1 Python node, len={len}"), &mut g, 5000);
    }

    // (2) Concurrency scaling: fan out N nodes over one source (all at level 1).
    // Ideal free-threaded scaling keeps per-tick ~flat until cores saturate.
    println!("--- fan-out scaling via the engine scheduler (len=1024, real numpy) ---");
    for n in [1usize, 2, 4, 8, 16] {
        let mut g = build(n, work, 1024);
        bench(&format!("{n} Python nodes (parallel level)"), &mut g, 2000);
    }

    // (3) Isolation: run the SAME numpy work on N raw std::threads, no engine,
    // no rayon. If this also collapses, the contention is in Python/numpy itself
    // (free-threaded compute), not the scheduler.
    println!("--- same numpy work on raw std::threads (isolates Python/numpy contention) ---");
    let bytes: Vec<u8> = (0..1024).flat_map(|i| (i as f32).to_le_bytes()).collect();
    for n in [1usize, 2, 4, 8, 16] {
        // Each thread owns its own PyNode and calls process() `rounds` times.
        let rounds = 2000u32;
        // Warm.
        {
            let mut nd = PyNode::from_source(work, vec!["data"], vec!["out"]).unwrap();
            run_once(&mut nd, &bytes);
        }
        let t = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..n {
                let b = bytes.clone();
                s.spawn(move || {
                    let mut nd = PyNode::from_source(work, vec!["data"], vec!["out"]).unwrap();
                    for _ in 0..rounds {
                        run_once(&mut nd, &b);
                    }
                });
            }
        });
        // Per-call latency = wall / rounds (all N threads overlap the `rounds` loop).
        let per = t.elapsed().as_secs_f64() / rounds as f64;
        println!(
            "{n:>2} raw threads x {rounds} calls          {:>8.1} us/round  (round = one call on every thread)",
            per * 1e6
        );
    }
}

/// One process() call driving `bytes` through the node's `data` input.
fn run_once(node: &mut PyNode, bytes: &[u8]) {
    use goofi_core::{Data, Meta};
    use goofi_node::{Inputs, NodeCtx, Outputs, Params};
    use indexmap::IndexMap;
    let frame =
        Data::array_f32(vec![bytes.len() / 4], bytes.to_vec(), Meta::empty()).unwrap();
    let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    inmap.insert("data", Some(frame));
    let inp = Inputs::new(&inmap);
    let mut outbuf: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    outbuf.insert("out", None);
    let mut ctx = NodeCtx::new();
    let params = ParamGroups::new();
    let mut out = Outputs::new(&mut outbuf);
    node.process(&inp, &mut out, &mut ctx, &Params::new(&params)).unwrap();
}
