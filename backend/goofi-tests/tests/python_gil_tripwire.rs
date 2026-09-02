//! The runtime GIL tripwire, in its own test BINARY: once a node re-enables the GIL it stays on
//! for the whole interpreter. Runs with `embed` and a free-threaded interpreter.
#![cfg(feature = "embed")]

use goofi_core::Data;
use goofi_node::{Isolation, IsolationCell, ParamGroups, Params};
use goofi_signal_sdk::{Inputs, Node, NodeCtx, Outputs};
use goofi_python::inproc::PyNode;
use indexmap::IndexMap;

/// A source node whose first `process` re-enables the GIL — and once on, it STAYS on.
const SRC: &str = r#"
import goofi, sys
import numpy as np
class Tripper(goofi.Node):
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    def process(self):
        sys._is_gil_enabled = lambda: True
        return {"out": (np.zeros(1, dtype=np.float32), {})}
"#;

fn tick(node: &mut PyNode, params: &ParamGroups) -> goofi_signal_sdk::NodeResult {
    let inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    let inp = Inputs::new(&inmap);
    let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    outmap.insert("out", None);
    let mut ctx = NodeCtx::new();
    let mut out = Outputs::new(&mut outmap);
    node.process(&inp, &mut out, &mut ctx, &Params::new(params))
}

#[test]
fn a_serialized_interpreter_is_reported_every_tick_and_demotes_its_type() {
    let p = ParamGroups::new();
    // The tier a registry would hold for this type: routed nodes read it at every build, so writing
    // it is the whole re-route.
    let tier = IsolationCell::leak(Isolation::InProcess);
    let mut node = PyNode::from_source(SRC, vec![], vec!["out"]).expect("PyNode").routed_by(tier);
    node.setup(&mut NodeCtx::new(), &Params::new(&p)).expect("setup");
    assert_eq!(tier.get(), Isolation::InProcess, "the probe cleared this type on its import");

    let first = tick(&mut node, &p);
    assert!(first.is_err(), "the tripwire must report the GIL being re-enabled");
    assert_eq!(
        tier.get(),
        Isolation::Subprocess,
        "tripping demotes the TYPE, so the next restart_node builds it in a subprocess"
    );

    // The condition is PERMANENT, and the 2 Hz stats sweep diffs SAMPLED state, so the error has
    // to persist while the condition does.
    let second = tick(&mut node, &p);
    assert!(second.is_err(), "a still-serialized interpreter must still be an error on the next tick");
    assert_eq!(format!("{:?}", first.unwrap_err()), format!("{:?}", second.unwrap_err()));
    assert_eq!(tier.get(), Isolation::Subprocess, "and the demotion holds rather than flapping");
}

/// A node built outside any registry has no tier to write, and must not panic reaching for one.
#[test]
fn an_unrouted_node_still_reports_the_trip() {
    let p = ParamGroups::new();
    let mut node = PyNode::from_source(SRC, vec![], vec!["out"]).expect("PyNode");
    node.setup(&mut NodeCtx::new(), &Params::new(&p)).expect("setup");
    assert!(tick(&mut node, &p).is_err(), "the error is the tripwire's, not the demotion's");
}
