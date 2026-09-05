//! The `goofi.Node` base class Python node authors derive from. A node declares itself in the
//! `INPUTS`/`OUTPUTS`/`PARAMS`/`TAGS` class constants and the `PRODUCER` flag; each may be omitted.
//! A `pulse_<group>_<name>(self)` method answers the `goofi.PulseParam` it names.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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

    /// The palette facets, from goofi's closed tag vocabulary — `["analysis", "eeg"]`.
    #[classattr]
    #[pyo3(name = "TAGS")]
    fn tags(py: Python<'_>) -> Bound<'_, PyList> {
        PyList::empty(py)
    }

    /// Init after params are seeded, once it succeeds; a raise is retried on this same instance,
    /// so release what was acquired before failing.
    fn setup(&self) -> PyResult<()> {
        Ok(())
    }
    /// The tick body. `**inputs` is one keyword argument per DECLARED input slot: a `Data`, or
    /// `None` where the slot holds no frame; a `multi` slot is `list[tuple[str, Data]]`, each
    /// frame with the `node.slot` that sent it, in wire order.
    #[pyo3(signature = (**_inputs))]
    fn process(&self, _inputs: Option<&Bound<'_, PyDict>>) -> Option<Py<PyAny>> {
        None
    }
}
