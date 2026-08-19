//! The `goofi.Node` base class Python node authors derive from (`class MyNode(goofi.Node)`).
//!
//! A node declares itself in three CONSTANTS and one flag, stated in the class body:
//!
//! ```python
//! class Psd(goofi.Node):
//!     INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
//!     OUTPUTS = {"psd": goofi.DataType.ARRAY}
//!     PARAMS = {"welch": {"nperseg": goofi.IntParam(256, 16, 4096)}}
//! ```
//!
//! Each may be omitted; the defaults below make an un-overridden node valid. They are attributes
//! rather than the `config_*` HOOKS they replace, and that is the point: a hook had to be called to
//! be read, so the probe, the in-process host and the subprocess child each re-entered the node to
//! ask the same questions and nothing stopped it answering differently each time. A constant is
//! evaluated once, by the import — which is also when the probe samples the GIL, so a node's
//! declaration-time imports still land before the routing gate looks, with no call to arrange it.
//!
//! There is no `category` and no `isolation`: the palette groups by source, and the tier is decided
//! by the probe from whether the node's imports keep the GIL disabled. A node asking for either
//! would be asking to be believed.

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

    /// `{slot_name: DataType | InputSlot}` — the node's input slots. A bare `DataType` is the whole
    /// of it for a slot with nothing to say beyond its type; `goofi.InputSlot(dtype, required=…,
    /// trigger=…)` carries the per-slot options.
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

    /// Whether the node paces itself rather than waiting for a frame. A node that says nothing is
    /// not a source.
    #[classattr]
    #[pyo3(name = "PRODUCER")]
    fn producer() -> bool {
        false
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
