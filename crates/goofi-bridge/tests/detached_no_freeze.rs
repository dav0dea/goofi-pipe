//! Regression guard for the load-bearing win: a Subprocess-isolated node runs off the
//! tick, so the bridge's `spawn_tick` (which holds `graph.lock()` across each `g.tick()`)
//! never strands control-plane work behind a slow node's `process()`. Before off-tick
//! execution, a `RemoteNode`'s multi-second iceoryx2 roundtrip froze the whole UI
//! (`goofi-bridge/src/lib.rs` — the documented freeze).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use goofi_bridge::spawn_tick;
use goofi_core::{Data, DType, Meta, SlotType};
use goofi_engine::Graph;
use goofi_node::{
    Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs, Params,
};

/// A native source marked `Subprocess`, so the engine detaches it. Its `process()` sleeps
/// 300 ms — standing in for a blocking subprocess roundtrip — WITHOUT needing python.
struct SlowNode;
impl Node for SlowNode {
    fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        std::thread::sleep(Duration::from_millis(300));
        let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
            .map_err(|e| e.to_string())?;
        out.set("out", d);
        Ok(())
    }
}

fn slow_stub() -> Box<dyn Node> {
    unreachable!("a runtime dyn type is built by its registered factory, not manifest.factory")
}

static SLOW_OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
static SLOW_MANIFEST: NodeManifest = NodeManifest {
    type_name: "SlowSubproc",
    category: "test",
    doc: "a slow (300ms) Subprocess-isolated source",
    inputs: &[],
    outputs: SLOW_OUT,
    params: &[],
    isolation: Isolation::Subprocess,
    factory: slow_stub,
};

#[test]
fn a_busy_detached_node_never_holds_the_graph_lock() {
    let mut g = Graph::new();
    g.register_dyn_type(&SLOW_MANIFEST, Box::new(|_| Box::new(SlowNode)));
    g.add_node("SlowSubproc", None).unwrap();
    let graph = Arc::new(Mutex::new(g));

    // The real bridge tick loop: locks the graph, ticks, releases, sleeps. It dispatches a
    // job to the slow worker, which then sleeps 300 ms OFF the lock.
    spawn_tick(graph.clone());
    std::thread::sleep(Duration::from_millis(60)); // let the worker be mid-process()

    // A control RPC does exactly this: acquire the graph lock. Even while the detached node
    // is deep in its 300 ms process(), the lock is only ever held for bounded inline work,
    // so acquisition stays fast. (If SlowNode were InProcess, `worst` would be ~300 ms.)
    let mut worst = Duration::ZERO;
    for _ in 0..25 {
        let t = Instant::now();
        {
            let _held = graph.lock().unwrap();
        }
        worst = worst.max(t.elapsed());
        std::thread::sleep(Duration::from_millis(12));
    }
    assert!(
        worst < Duration::from_millis(100),
        "graph lock stayed responsive (worst {worst:?}) despite a 300ms detached node"
    );
}
