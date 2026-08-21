//! The `goofi.Node` base class Python node authors derive from. A node declares itself in the
//! `INPUTS`/`OUTPUTS`/`PARAMS` class constants and the `PRODUCER` flag; each may be omitted.

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

    /// `{slot_name: DataType | InputSlot}` — the node's input slots.
    #[classattr]
    #[pyo3(name = "INPUTS")]
    fn inputs(py: Python<'_>) -> Bound<'_, PyDict> {
        PyDict::new(py)
    }

    /// `{slot_name: DataType}` — the node's output slots.
    #[classattr]
    #[pyo3(name = "OUTPUTS")]
    fn outputs(py: Python<'_>) -> Bound<'_, PyDict> {
        PyDict::new(py)
    }

    /// `{group: {name: <Param descriptor>}}` — the node's params.
    #[classattr]
    #[pyo3(name = "PARAMS")]
    fn params(py: Python<'_>) -> Bound<'_, PyDict> {
        PyDict::new(py)
    }

    /// Whether the node paces itself rather than waiting for a frame.
    #[classattr]
    #[pyo3(name = "PRODUCER")]
    fn producer() -> bool {
        false
    }

    /// Init after params are seeded, once it succeeds; a raise is retried on this same instance,
    /// so release what was acquired before failing.
    fn setup(&self) -> PyResult<()> {
        Ok(())
    }
    /// The tick body. `**inputs` is one keyword argument per DECLARED input slot, `None` where
    /// the slot holds no frame.
    #[pyo3(signature = (**_inputs))]
    fn process(&self, _inputs: Option<&Bound<'_, PyDict>>) -> Option<Py<PyAny>> {
        None
    }
}
