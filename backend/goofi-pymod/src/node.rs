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

    /// `{slot_name: DataType | InputSlot}` — the node's input slots. A bare `DataType` is still
    /// the whole of it for a slot with nothing to say beyond its type;
    /// `goofi.InputSlot(dtype, required=…, trigger=…)` carries the per-slot options. Default: none.
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
