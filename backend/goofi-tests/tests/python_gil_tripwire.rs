//! The runtime GIL tripwire, in its own test BINARY. A node that re-enables the GIL poisons
//! `sys._is_gil_enabled` for the whole interpreter, so this cannot share a process with the
//! other in-process host tests — every one of them would trip on its own first tick.
//!
//! Runs only with `embed` + a free-threaded interpreter:
//!   cargo test -p goofi-tests --features embed --test python_gil_tripwire
#![cfg(feature = "embed")]

use goofi_core::Data;
use goofi_node::{Inputs, Node, NodeCtx, Outputs, ParamGroups, Params};
use goofi_python::inproc::PyNode;
use indexmap::IndexMap;

/// A source node whose first `process` re-enables the GIL — the shape an FT-unsafe import at
/// call time produces. Stands in for the real thing: the tripwire reads `sys._is_gil_enabled`,
/// and once the GIL is on it STAYS on, so every later tick observes it too.
const SRC: &str = r#"
import goofi, sys
import numpy as np
class Tripper(goofi.Node):
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self):
        sys._is_gil_enabled = lambda: True
        return {"out": (np.zeros(1, dtype=np.float32), {})}
"#;

fn tick(node: &mut PyNode, params: &ParamGroups) -> goofi_node::NodeResult {
    let inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    let inp = Inputs::new(&inmap);
    let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    outmap.insert("out", None);
    let mut ctx = NodeCtx::new();
    let mut out = Outputs::new(&mut outmap);
    node.process(&inp, &mut out, &mut ctx, &Params::new(params))
}

#[test]
fn a_serialized_interpreter_keeps_being_reported_every_tick() {
    let p = ParamGroups::new();
    let mut node = PyNode::from_source(SRC, vec![], vec!["out"]).expect("PyNode");
    node.setup(&mut NodeCtx::new(), &Params::new(&p)).expect("setup");

    let first = tick(&mut node, &p);
    assert!(first.is_err(), "the tripwire must report the GIL being re-enabled");

    // The condition is PERMANENT — the interpreter is serialized for every in-process node
    // from here on. The only channel to the client is the 2 Hz stats sweep, which diffs
    // SAMPLED `last_error` state, so a single one-tick error at a 30 Hz tick is almost never
    // observed: the next successful tick clears it and the client never learns. The error has
    // to persist while the condition does.
    let second = tick(&mut node, &p);
    assert!(second.is_err(), "a still-serialized interpreter must still be an error on the next tick");
    assert_eq!(format!("{:?}", first.unwrap_err()), format!("{:?}", second.unwrap_err()));
}
