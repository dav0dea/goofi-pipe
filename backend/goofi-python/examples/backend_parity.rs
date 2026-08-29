//! Backend-parity benchmark: the same `x*2+1` workload through native Rust, in-process FT Python
//! and subprocess Python, in one engine.
//!   PYO3_PYTHON=<python3.14t> LD_LIBRARY_PATH=<base>/lib PYTHONPATH=<ft-sp> \
//!     cargo run -p goofi-python --features embed --example backend_parity --release
//! GOOFI_SUBPROC_PYTHON names an iceoryx2-capable interpreter for the subprocess tier.

use std::time::{Duration, Instant};

use goofi_core::globals::GlobalValue;
use goofi_core::{Data, Param, Value};
use goofi_engine::testing::OutputProbe;
use goofi_engine::Graph;
use goofi_node::{Isolation, IsolationCell, NodeManifest, OutputDecl, ParamDecl, ParamGroups, Params, SlotDecl};
use goofi_signal::{Inputs, Node, NodeCtx, NodeResult, Outputs};
use goofi_python::inproc::PyNode;
use goofi_python::subproc::RemoteNode;

static BENCH_TIER: IsolationCell = IsolationCell::new(Isolation::InProcess);

/// The identical workload every backend computes.
const PY_SRC: &str = concat!(
    "import goofi\n",
    "import numpy as np\n",
    "class Bench(goofi.Node):\n",
    "    INPUTS = {'data': goofi.DataType.ARRAY}\n",
    "    OUTPUTS = {'out': goofi.DataType.ARRAY}\n",
    "    def process(self, data):\n",
    "        return {'out': data.data * 2 + 1}\n",
);

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
        let data = Data::array_f32(s.shape().to_vec(), buf, d.meta().clone())
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
    required: false,
}];
static OUT: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: goofi_core::SlotType::Array,
}];
static NO_PARAMS: &[ParamDecl] = &[];
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
        producer: false,
    }
}

type Factory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// Fan `n` nodes off one _TestConst and measure the rate each sustains, counted from
/// `meta["index"]`, plus whether every error channel stayed clean.
fn bench(manifest: &'static NodeManifest, factory: Factory, len: i64, n: usize, window: Duration) -> (f64, bool) {
    let mut g = Graph::new();
    // Every producer's rate cap is `globals.default_ufreq`; the patch default measures 30 Hz.
    g.apply_global_change("default_ufreq", Some(GlobalValue::Float(1e6)), None).unwrap();
    g.register_dyn_type(manifest, factory, &BENCH_TIER);
    let src = g.add_node("_TestConst", None).unwrap();
    g.update_param(src, "constant", "value", Param::float(0.5, -1e9, 1e9)).unwrap();
    g.update_param(src, "constant", "length", Param::int(len, 1, 10_000_000)).unwrap();
    let mut nodes = Vec::new();
    let mut probes = Vec::new();
    for _ in 0..n {
        let node = g.add_node(manifest.type_name, None).unwrap();
        probes.push(OutputProbe::open(&g, node, "out"));
        g.add_link(src, "out", node, "data").unwrap();
        nodes.push(node);
    }

    // A link attaches over a sequence that advances on acks, so nothing flows until status drains.
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_secs(3) {
        g.drain_status();
        std::thread::sleep(Duration::from_millis(1));
    }
    // 0.5*2+1 = 2.0.
    for (&node, probe) in nodes.iter().zip(&probes) {
        let frame = probe.expect_frame(&mut g, "the backend to emit");
        if let Value::Array(s) = frame.value() {
            let first = f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap());
            assert!((first - 2.0).abs() < 1e-4, "wrong result {first}");
        }
        assert!(g.last_error(node).is_none(), "faulted: {:?}", g.last_error(node));
    }

    let index = |g: &mut Graph, p: &OutputProbe| {
        p.expect_frame(g, "the backend to emit").meta().index().unwrap_or(0)
    };
    let first: Vec<u64> = probes.iter().map(|p| index(&mut g, p)).collect();
    let t = Instant::now();
    std::thread::sleep(window);
    let emitted: u64 = probes
        .iter()
        .zip(&first)
        .map(|(p, start)| index(&mut g, p).saturating_sub(*start))
        .sum();
    let per_node = emitted as f64 / n as f64 / t.elapsed().as_secs_f64();

    g.drain_status();
    let stable = nodes.iter().all(|&node| g.last_error(node).is_none());
    (per_node, stable)
}

fn main() {
    let len: i64 = 1024;
    // The subprocess tier needs an `iceoryx2` + `numpy` python, which PYO3_PYTHON usually is not.
    let python = std::env::var("GOOFI_SUBPROC_PYTHON")
        .or_else(|_| std::env::var("PYO3_PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());
    let backends: [(&str, &'static NodeManifest); 3] = [
        ("1. native Rust", &NATIVE_M),
        ("2. in-process FT Python", &FTPY_M),
        ("3. subprocess Python", &SUBPY_M),
    ];

    println!("Backend parity — x*2+1 on a length-{len} f32 array, in the one engine.\n");
    println!("{:<26} {:>16} {:>18}   stability", "backend", "1 node", "8-node fan-out");
    println!("{}", "-".repeat(78));
    for (label, m) in backends {
        let (one, s1) = bench(m, rebuild(m, &python), len, 1, Duration::from_secs(2));
        let (eight, s8) = bench(m, rebuild(m, &python), len, 8, Duration::from_secs(2));
        println!(
            "{label:<26} {one:>8.0} frames/s {eight:>10.0} frames/s   {} / {}",
            if s1 { "clean" } else { "FAULT" },
            if s8 { "clean" } else { "FAULT" }
        );
    }

    println!(
        "\nInterpretation: native is the RT ceiling and holds its per-node rate under fan-out (one\n\
         thread each, no contention). In-process FT Python is fast for ONE node but the 8-node\n\
         fan-out exposes the free-threaded biased-refcount penalty across threads. The subprocess\n\
         tier pays a fixed round-trip but its separate interpreters DON'T contend, so it scales\n\
         near-flat — the empirical basis for routing hot fine-grained work to native/FT and\n\
         GIL-unsafe-or-heavy work to the subprocess tier."
    );
}

/// Rebuild the per-backend factory, once per pass, because a closure is not `Clone`.
fn rebuild(m: &'static NodeManifest, python: &str) -> Factory {
    match m.type_name {
        "bench_native" => Box::new(|_| Box::new(NativeMul) as Box<dyn Node>),
        "bench_ftpy" => Box::new(|_| Box::new(PyNode::from_source(PY_SRC, vec!["data"], vec!["out"]).expect("PyNode")) as Box<dyn Node>),
        _ => {
            let py = python.to_string();
            Box::new(move |_| Box::new(RemoteNode::new(py.clone(), PY_SRC, vec!["data"])) as Box<dyn Node>)
        }
    }
}
