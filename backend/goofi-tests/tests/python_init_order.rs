//! Interpreter-bootstrap ordering, in its own process so the embedded interpreter starts
//! uninitialized: the evaluator initializes it FIRST, then a Python node is built.
#![cfg(feature = "embed")]

use goofi_python::inproc::{PyExprEvaluator, PyNode};

const NODE: &str = concat!(
    "import goofi\n",
    "import numpy as np\n",
    "class Double(goofi.Node):\n",
    "    INPUTS = {'data': goofi.DataType.ARRAY}\n",
    "    OUTPUTS = {'out': goofi.DataType.ARRAY}\n",
    "    def process(self, data):\n",
    "        return {'out': data.data * 2.0}\n",
);

#[test]
fn evaluator_first_then_python_node_does_not_panic() {
    let _ev = PyExprEvaluator::new().expect("evaluator constructs");
    // The `goofi` module must already be in the inittab, registered before the first attach.
    let _node = PyNode::from_source(NODE, vec![("data", false)], vec!["out"])
        .expect("PyNode built after the evaluator inited the interpreter must not panic");
}
