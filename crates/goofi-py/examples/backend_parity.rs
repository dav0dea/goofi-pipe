//! Unified backend-parity benchmark — the SAME `x*2+1` workload run through all
//! three node backends inside ONE engine, for a directly comparable measurement:
//!   1. native Rust           (an inline element-wise node)
//!   2. in-process FT Python  (pyo3, GIL off — `goofi_py::PyNode`)
//!   3. subprocess Python     (a separate GIL interpreter — `goofi_subproc::RemoteNode`)
//!
//! Each backend is hosted identically (_TestConst -> node), warmed up, then
//! timed per-tick, followed by a sustained stability run (no faulted error
//! channel). Not a test — run under the FT env (same vars as `py_latency`):
//!   PYO3_PYTHON=<python3.14t> LD_LIBRARY_PATH=<base>/lib PYTHONPATH=<ft-sp> \
//!     cargo run -p goofi-py --features embed --example backend_parity --release
//! The subprocess tier reuses PYO3_PYTHON (it inherits PYTHONPATH so numpy loads).

use std::time::Instant;

use goofi_core::{Data, DType, Param, Value};
use goofi_engine::Graph;
use goofi_node::{
    Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs, ParamDecl,
    ParamGroups, Params, SlotDecl,
};
use goofi_py::PyNode;
use goofi_subproc::RemoteNode;

/// The identical workload every backend computes, so the comparison is apples-to-apples.
const PY_SRC: &str = "def process(x):\n    return x * 2 + 1\n";

/// Native Rust equivalent of `x * 2 + 1`, element-wise over a float32 array.
struct NativeMul;
impl Node for NativeMul {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let Value::Array(s) = d.value() else {
            return Ok(());
        };
        let mut buf = Vec::with_capacity(s.as_bytes().len());
        for c in s.as_bytes().chunks_exact(4) {
            let x = f32::from_le_bytes(c.try_into().unwrap());
            buf.extend_from_slice(&(x * 2.0 + 1.0).to_le_bytes());
        }
        let data = Data::from_array_bytes(DType::F32, s.shape().to_vec(), buf, d.meta().clone())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }
}

static IN: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: goofi_core::SlotType::Array,
    trigger_process: true,
    multi: false,
}];
static OUT: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: goofi_core::SlotType::Array,
}];
static NO_PARAMS: &[ParamDecl] = &[];
fn stub_factory() -> Box<dyn Node> {
    unreachable!("dyn types build via their registered factory")
}
// One leaked-'static manifest per backend type (their `factory` is never called).
static NATIVE_M: NodeManifest = manifest("bench_native");
static FTPY_M: NodeManifest = manifest("bench_ftpy");
static SUBPY_M: NodeManifest = manifest("bench_subpy");
const fn manifest(type_name: &'static str) -> NodeManifest {
    NodeManifest {
        type_name,
        category: "bench",
        doc: "",
        inputs: IN,
        outputs: OUT,
        params: NO_PARAMS,
        isolation: Isolation::InProcess,
        factory: stub_factory,
    }
}

type Factory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// Build _TestConst(len) -> `n` fanned `type_name` nodes (all at one topo level,
/// so they run concurrently on the pool), warm up, gate correctness, time `iters`
/// whole-graph ticks, then a stability pass. Returns per-tick microseconds.
fn bench(manifest: &'static NodeManifest, factory: Factory, len: i64, n: usize, iters: u32) -> (f64, bool) {
    let mut g = Graph::new();
    g.register_dyn_type(manifest, factory);
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "value", Param::float(0.5, -1e9, 1e9)).unwrap();
    g.update_param(src, "constant", "length", Param::int(len, 1, 10_000_000)).unwrap();
    let mut nodes = Vec::new();
    for _ in 0..n {
        let node = g.add_node(manifest.type_name, None).unwrap();
        g.add_link(src, "out", node, "data").unwrap();
        nodes.push(node);
    }

    for _ in 0..100 {
        g.tick();
    }
    // Correctness + clean-error gate on every node: 0.5*2+1 = 2.0.
    for &node in &nodes {
        let frame = g.latest_frame(node, "out").expect("backend produced a frame");
        if let Value::Array(s) = frame.value() {
            let first = f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap());
            assert!((first - 2.0).abs() < 1e-4, "wrong result {first}");
        }
        assert!(g.last_error(node).is_none(), "faulted: {:?}", g.last_error(node));
    }

    let t = Instant::now();
    for _ in 0..iters {
        g.tick();
    }
    let per_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    for _ in 0..2000 {
        g.tick();
    }
    let stable = nodes.iter().all(|&node| g.last_error(node).is_none());
    (per_us, stable)
}

fn main() {
    let len: i64 = 1024;
    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let backends: [(&str, &'static NodeManifest); 3] = [
        ("1. native Rust", &NATIVE_M),
        ("2. in-process FT Python", &FTPY_M),
        ("3. subprocess Python", &SUBPY_M),
    ];

    println!("Backend parity — x*2+1 on a length-{len} f32 array, ticked by the one engine.\n");
    println!("{:<26} {:>14} {:>18}   stability", "backend", "1 node", "8-node fan-out");
    println!("{}", "-".repeat(74));
    for (label, m) in backends {
        // A fresh factory per pass (closures aren't Clone) via `rebuild`.
        let (one, s1) = bench(m, rebuild(m, &python), len, 1, 4000);
        let (eight, s8) = bench(m, rebuild(m, &python), len, 8, 2000);
        println!(
            "{label:<26} {one:>8.2} us/tick {eight:>10.2} us/tick   {} / {}",
            if s1 { "clean" } else { "FAULT" },
            if s8 { "clean" } else { "FAULT" }
        );
    }

    println!(
        "\nInterpretation: native is the RT floor and stays flat under fan-out (parallel, no\n\
         contention). In-process FT Python is fast for ONE node (it runs inline on the numpy\n\
         owner thread) but the 8-node fan-out exposes the free-threaded biased-refcount penalty\n\
         on worker threads. The subprocess tier pays a fixed pipe round-trip but its separate\n\
         interpreters DON'T contend, so it scales near-flat — the empirical basis for routing\n\
         hot fine-grained work to native/FT and GIL-unsafe-or-heavy work to the subprocess tier."
    );
}

/// Rebuild the per-backend factory (closures aren't Clone; the mapping mirrors the
/// `backends` vec above).
fn rebuild(m: &'static NodeManifest, python: &str) -> Factory {
    match m.type_name {
        "bench_native" => Box::new(|_| Box::new(NativeMul) as Box<dyn Node>),
        "bench_ftpy" => Box::new(|_| Box::new(PyNode::from_source(PY_SRC, "process").expect("PyNode")) as Box<dyn Node>),
        _ => {
            let py = python.to_string();
            Box::new(move |_| Box::new(RemoteNode::new(py.clone(), PY_SRC)) as Box<dyn Node>)
        }
    }
}
