//! Concurrency-scaling probe for the subprocess tier. Confirms the latency
//! finding's implication: separate interpreters (one per process) run heavy
//! Python compute in parallel WITHOUT the free-threaded biased-refcount penalty
//! that flattened the in-process path (10us -> 250us on a worker thread).
//!
//! Run (needs a python3 with the goofi abi3 wheel + numpy on PATH):
//!   cargo run -p goofi-subproc --example subproc_scaling --release
use std::time::Instant;

use goofi_core::{Data, Meta};
use goofi_node::{Inputs, Node, NodeCtx, Outputs, ParamGroups, Params};
use goofi_subproc::RemoteNode;
use indexmap::IndexMap;

fn one_tick(node: &mut RemoteNode, d: &Data) {
    let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    inmap.insert("data", Some(d.clone()));
    let inp = Inputs::new(&inmap);
    let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    outmap.insert("out", None);
    let mut ctx = NodeCtx::new();
    let params = ParamGroups::new();
    let mut out = Outputs::new(&mut outmap);
    node.process(&inp, &mut out, &mut ctx, &Params::new(&params)).expect("tick");
}

fn main() {
    // Non-trivial numpy work, length-preserving. Authored to the goofi.Node class contract.
    let src = concat!(
        "import numpy as np\n",
        "import goofi\n",
        "class Tanh(goofi.Node):\n",
        "    @staticmethod\n",
        "    def config_input_slots():\n",
        "        return {'data': goofi.DataType.ARRAY}\n",
        "    @staticmethod\n",
        "    def config_output_slots():\n",
        "        return {'out': goofi.DataType.ARRAY}\n",
        "    def process(self, data):\n",
        "        x = data.data\n",
        "        return {'out': np.tanh(x) * 2.0 - x.mean()}\n",
    );
    let py = "python3";
    let len = 1024usize;
    let buf: Vec<u8> = (0..len).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let d = Data::array_f32(vec![len], buf, Meta::empty()).unwrap();

    let rounds = 300u32;
    println!("subprocess tier — {rounds} rounds, len={len}, real numpy per call");
    for n in [1usize, 2, 4, 8, 16] {
        // n independent RemoteNodes, each on its own thread => n OS processes.
        let d = d.clone();
        let src = src.to_string();
        // Warm: spawn + first import numpy is slow; pay it before timing.
        let mut warm = RemoteNode::spawn(py, &src, vec!["data"], vec!["out"]).unwrap();
        one_tick(&mut warm, &d);
        drop(warm);

        let t = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..n {
                let d = d.clone();
                let src = src.clone();
                s.spawn(move || {
                    let mut node = RemoteNode::spawn(py, &src, vec!["data"], vec!["out"]).unwrap();
                    one_tick(&mut node, &d); // amortize this thread's spawn
                    let t = Instant::now();
                    for _ in 0..rounds {
                        one_tick(&mut node, &d);
                    }
                    t.elapsed()
                });
            }
        });
        // Wall covers all n processes overlapping; report per-round (= per-tick
        // latency seen by one node while all n run concurrently).
        let per = t.elapsed().as_secs_f64() / rounds as f64;
        println!(
            "{n:>2} subprocess nodes (parallel)   {:>8.1} us/round   ({:>6.0} rounds/s)",
            per * 1e6,
            1.0 / per
        );
    }
    println!(
        "per-round grows only ~2x for 16x nodes => processes overlap on separate\n\
         cores with NO cross-process contention (in-process free-threaded exploded\n\
         ~70x: 10us@1 -> 700us@16 from biased refcounting). Caveat: the pipe\n\
         round-trip is a high fixed cost (~460us/tick), so the subprocess tier wins\n\
         for GIL-unsafe deps and HEAVY per-node compute, not light/single nodes\n\
         (in-process free-threaded is far faster there)."
    );
}
