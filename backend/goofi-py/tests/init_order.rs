//! Interpreter-bootstrap ordering. A SEPARATE test binary (its own process) so the
//! embedded interpreter starts uninitialized and this test's ordering is deterministic —
//! it mirrors `goofi-cli main()`: the expression evaluator initializes the interpreter
//! FIRST (register_evaluator), then a Python node is built (register_python / add_node).
//!
//! Regression for the M2 append_to_inittab ordering bug: if the evaluator's
//! `Python::attach` inits the interpreter WITHOUT first registering `goofi` in the
//! inittab, the first `PyNode::from_source` -> `append_to_inittab!` panics ("a Python
//! interpreter is already running"), and that panic unwinds through the node factory
//! while the manager holds the graph mutex — poisoning it and downing the control plane.
#![cfg(feature = "embed")]

use goofi_py::{PyExprEvaluator, PyNode};

const NODE: &str = concat!(
    "import goofi\n",
    "import numpy as np\n",
    "class Double(goofi.Node):\n",
    "    def config_input_slots(self):\n",
    "        return {'data': goofi.DataType.ARRAY}\n",
    "    def config_output_slots(self):\n",
    "        return {'out': goofi.DataType.ARRAY}\n",
    "    def process(self, data):\n",
    "        return {'out': data.data * 2.0}\n",
);

#[test]
fn evaluator_first_then_python_node_does_not_panic() {
    // The evaluator inits the interpreter first (exactly as main() does).
    let _ev = PyExprEvaluator::new().expect("evaluator constructs");
    // Building a Python node AFTER the interpreter is initialized must not panic — the
    // `goofi` module must already be in the inittab (registered before the first attach).
    let _node = PyNode::from_source(NODE, vec!["data"], vec!["out"])
        .expect("PyNode built after the evaluator inited the interpreter must not panic");
}
