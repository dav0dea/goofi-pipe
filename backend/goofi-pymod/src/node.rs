//! The `goofi.Node` base class Python node authors derive from
//! (`class MyNode(goofi.Node)`). The defaults make an un-overridden node valid — an empty
//! [`crate::manifest::Manifest`] and a no-op setup/process — and a subclass replaces any of them.
//! The Rust adapters (in-process and subprocess) read this contract; this crate only defines it.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::manifest::Manifest;

#[pyclass(subclass)]
pub struct Node {}

#[pymethods]
impl Node {
    #[new]
    fn new() -> Node {
        Node {}
    }

    /// What the node declares about itself. A subclass states its own:
    /// `manifest = goofi.Manifest(inputs={…}, outputs={…}, params={…}, producer=True)`.
    #[classattr]
    fn manifest(py: Python<'_>) -> Manifest {
        Manifest::new(py, None, None, None, false)
    }

    /// Init after params are seeded — once, if it succeeds. A raise leaves the node
    /// uninitialized: nothing runs against it until a later interaction retries this same
    /// instance, so release what was acquired before failing. Default: no-op.
    fn setup(&self) -> PyResult<()> {
        Ok(())
    }
    /// The tick body. Default: emit nothing. `**inputs` is one keyword argument per DECLARED
    /// input slot — a `goofi.Data` when the slot holds a frame and `None` when it does not, so the
    /// node decides for itself what an absent input means. A `required=True` slot never arrives
    /// empty; the engine refuses the tick upstream.
    #[pyo3(signature = (**_inputs))]
    fn process(&self, _inputs: Option<&Bound<'_, PyDict>>) -> Option<Py<PyAny>> {
        None
    }
}
