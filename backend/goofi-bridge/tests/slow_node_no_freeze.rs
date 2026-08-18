//! Regression guard for the one lock property the async runtime does NOT get for free: a node
//! parked inside a long `process()` must not be able to hold up `remove_node`, which runs under the
//! graph mutex every control RPC needs.
//!
//! Its sibling — "a running node never strands the lock" — was deleted with the tick it described.
//! There is no loop holding `graph.lock()` across a node's `process()` any more, so an assertion
//! that lock acquisition stays fast now holds against every implementation, correct or not.
//! `remove_node` is different: it is real code that could wait, and `Graph::shutdown` (which waits
//! on purpose, at exit) is the proof that waiting is one line away.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use goofi_core::{Data, Meta, SlotType};
use goofi_engine::Graph;
use goofi_node::{
    Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs, Params,
};

/// A source whose `process()` sleeps 300 ms — standing in for a blocking subprocess roundtrip,
/// WITHOUT needing python. `InProcess` on purpose: that is the tier the old tick ran inline, so it
/// is the case the cutover had to make safe.
struct SlowNode;
impl Node for SlowNode {
    fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        std::thread::sleep(Duration::from_millis(300));
        let d = Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
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
    type_name: "SlowSource",
    category: "test",
    doc: "a source whose every run takes 300ms",
    inputs: &[],
    outputs: SLOW_OUT,
    params: &[],
    isolation: Isolation::InProcess,
    producer: true,
    factory: slow_stub,
};

#[test]
fn deleting_a_busy_node_never_waits_for_it_under_the_graph_lock() {
    // `remove_node` runs under the same lock every control RPC needs, and a node only observes its
    // halt flag between runs — so waiting on one parked inside `process()` would freeze the whole
    // app for that window (a real subprocess node waits out its 10 s timeout).
    let mut g = Graph::new();
    g.register_dyn_type(&SLOW_MANIFEST, Box::new(|_| Box::new(SlowNode)));
    let uid = g.add_node("SlowSource", None).unwrap();
    let graph = Arc::new(Mutex::new(g));

    // Long enough for the node's own thread to be inside its 300 ms run, and far short of its end.
    std::thread::sleep(Duration::from_millis(60));

    let t0 = Instant::now();
    graph.lock().unwrap().remove_node(uid).unwrap();
    let held = t0.elapsed();
    assert!(
        held < Duration::from_millis(100),
        "delete returned in {held:?} — it waited on the busy node under the graph lock"
    );
}
