//! The `goofi.Node` base class Python node authors derive from
//! (`class MyNode(goofi.Node)`). The defaults make an un-overridden node valid
//! (no slots, no params, no-op setup/process); a subclass overrides any of them.
//! The Rust adapters (M2 in-process, M3 subprocess) call these methods; this crate
//! only defines the contract.

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(subclass)]
pub struct Node {}

#[pymethods]
impl Node {
    #[new]
    fn new() -> Node {
        Node {}
    }

    /// `{slot_name: DataType}` — the node's input slots. Default: none.
    fn config_input_slots<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }
    /// `{slot_name: DataType}` — the node's output slots. Default: none.
    fn config_output_slots<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }
    /// `{group: {name: <Param descriptor>}}` — the node's params. Default: none.
    fn config_params<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }

    /// One-time init after params are seeded. Default: no-op.
    fn setup(&self) -> PyResult<()> {
        Ok(())
    }
    /// The tick body. Default: emit nothing. `**inputs` are the present input slots.
    #[pyo3(signature = (**_inputs))]
    fn process(&self, _inputs: Option<&Bound<'_, PyDict>>) -> Option<Py<PyAny>> {
        None
    }
}
