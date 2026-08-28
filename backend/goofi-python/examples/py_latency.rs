//! Latency + concurrency-scaling probe for the in-process Python node path.
//!   PYO3_PYTHON=<python3.14t> LD_LIBRARY_PATH=<base>/lib PYTHONPATH=<ft-sp> \
//!     cargo run -p goofi-python --features embed --example py_latency --release
use std::time::{Duration, Instant};

use goofi_core::globals::GlobalValue;
use goofi_core::Param;
use goofi_engine::testing::OutputProbe;
use goofi_engine::Graph;
use goofi_node::{Isolation, IsolationCell, Node, NodeManifest, OutputDecl, ParamDecl, ParamGroups, SlotDecl};
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
static PY_TIER: IsolationCell = IsolationCell::new(Isolation::InProcess);
static PY_MANIFEST: NodeManifest = NodeManifest {
    type_name: "PyNode",
    category: "python",
    doc: "",
    inputs: PY_IN,
    outputs: PY_OUT,
    params: PY_PARAMS,
    isolation: &PY_TIER,
    producer: false,
    factory: py_stub_factory,
};

fn build(n: usize, src: &'static str, len: i64) -> (Graph, Vec<OutputProbe>) {
    let mut g = Graph::new();
    // Every producer's rate cap is `globals.default_ufreq`; the patch default measures 30 Hz.
    g.apply_global_change("default_ufreq", Some(GlobalValue::Float(1e6)), None).unwrap();
    g.register_dyn_type(
        &PY_MANIFEST,
        Box::new(move |_| Box::new(PyNode::from_source(src, vec!["data"], vec!["out"]).unwrap()) as Box<dyn Node>),
    );
    let osc = g.add_node("_TestConst", None).unwrap();
    g.update_param(osc, "constant", "value", Param::float(0.5, -1e9, 1e9)).unwrap();
    g.update_param(osc, "constant", "length", Param::int(len, 1, 10_000_000)).unwrap();
    let mut probes = Vec::new();
    for _ in 0..n {
        let py = g.add_node("PyNode", None).unwrap();
        probes.push(OutputProbe::open(&g, py, "out"));
        g.add_link(osc, "out", py, "data").unwrap();
    }
    (g, probes)
}

/// The rate the fan-out sustains, counted from `meta["index"]` rather than from the frames a probe
/// catches — the data services are latest-wins one deep.
fn bench(label: &str, g: &mut Graph, probes: &[OutputProbe], window: Duration) {
    // A link attaches over a sequence that advances on acks, so nothing flows until status drains.
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_secs(2) {
        g.drain_status();
        std::thread::sleep(Duration::from_millis(1));
    }
    let index = |g: &mut Graph, p: &OutputProbe| {
        p.expect_frame(g, "a python node to emit").meta().index().unwrap_or(0)
    };
    let first: Vec<u64> = probes.iter().map(|p| index(g, p)).collect();
    let t = Instant::now();
    std::thread::sleep(window);
    let emitted: u64 = probes
        .iter()
        .zip(&first)
        .map(|(p, start)| index(g, p).saturating_sub(*start))
        .sum();
    let per_node = emitted as f64 / probes.len() as f64 / t.elapsed().as_secs_f64();
    println!("{label:<44} {per_node:>8.0} frames/s per node  ({:>7.0} total)", per_node * probes.len() as f64);
}

fn main() {
    println!("GIL enabled: {}", PyNode::gil_enabled().unwrap());

    // (0) Bare `Python::attach` cost, main thread against a fresh worker thread.
    {
        use pyo3::prelude::*;
        let iters = 20_000u32;
        let t = Instant::now();
        for _ in 0..iters {
            Python::attach(|_py| {});
        }
        let main_ns = t.elapsed().as_nanos() as f64 / iters as f64;

        let worker_ns = std::thread::spawn(move || {
            // Prime one attach, then time the steady state.
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
        "    INPUTS = {'data': goofi.DataType.ARRAY}\n",
        "    OUTPUTS = {'out': goofi.DataType.ARRAY}\n",
        "    def process(self, data):\n",
        "        return {'out': np.tanh(data.data) * 2.0 - data.data.mean()}\n",
    );

    // (1) Single-node sustained rate at a few array sizes.
    for len in [64i64, 1024, 16384] {
        let (mut g, probes) = build(1, work, len);
        bench(&format!("1 Python node, len={len}"), &mut g, &probes, Duration::from_secs(2));
    }

    // (2) Concurrency scaling: ideal free-threaded scaling holds the per-node rate flat.
    println!("--- fan-out scaling on the async runtime (len=1024, real numpy) ---");
    for n in [1usize, 2, 4, 8, 16] {
        let (mut g, probes) = build(n, work, 1024);
        bench(&format!("{n} Python nodes"), &mut g, &probes, Duration::from_secs(2));
    }

    // (3) The same work on raw threads: a collapse here is Python/numpy, not the scheduler.
    println!("--- same numpy work on raw std::threads (isolates Python/numpy contention) ---");
    let bytes: Vec<u8> = (0..1024).flat_map(|i| (i as f32).to_le_bytes()).collect();
    for n in [1usize, 2, 4, 8, 16] {
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
        // All N threads overlap the `rounds` loop.
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
