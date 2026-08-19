//! `goofi.Manifest` — everything a Python node declares about itself, in one place.
//!
//! It replaces three `config_*` hooks and a loose `producer` class attribute. Those were four
//! declarations of one thing, and being methods they had to be CALLED to be read — so the probe,
//! the in-process host and the subprocess child each re-entered the node to ask it the same
//! questions, and a node could answer differently each time.
//!
//! A class attribute is evaluated once, when the module is imported, which is also when the probe
//! samples the GIL — so a node's declaration-time imports still land before the routing gate looks,
//! with no hook call to arrange it.
//!
//! Parity with the Rust `NodeManifest` is deliberate, minus the two fields that are not an author's
//! to give: `category` (the palette groups by source, not by a string a node picks for itself) and
//! `isolation` (the tier is decided by the probe, from whether the node's imports keep the GIL
//! disabled — a node asking for one would be asking to be believed).

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `goofi.Manifest(inputs=…, outputs=…, params=…, producer=…)`.
#[pyclass]
pub struct Manifest {
    /// `{slot_name: DataType | InputSlot}`.
    #[pyo3(get)]
    pub inputs: Py<PyDict>,
    /// `{slot_name: DataType}`.
    #[pyo3(get)]
    pub outputs: Py<PyDict>,
    /// `{group: {name: <Param descriptor>}}`.
    #[pyo3(get)]
    pub params: Py<PyDict>,
    /// Whether the node paces itself rather than waiting for a frame. A node that says nothing is
    /// not a source.
    #[pyo3(get)]
    pub producer: bool,
}

#[pymethods]
impl Manifest {
    #[new]
    #[pyo3(signature = (inputs=None, outputs=None, params=None, producer=false))]
    pub(crate) fn new(
        py: Python<'_>,
        inputs: Option<Py<PyDict>>,
        outputs: Option<Py<PyDict>>,
        params: Option<Py<PyDict>>,
        producer: bool,
    ) -> Manifest {
        let empty = || PyDict::new(py).unbind();
        Manifest {
            inputs: inputs.unwrap_or_else(empty),
            outputs: outputs.unwrap_or_else(empty),
            params: params.unwrap_or_else(empty),
            producer,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Manifest(inputs={}, outputs={}, params={}, producer={})",
            self.inputs.bind(py).len(),
            self.outputs.bind(py).len(),
            self.params.bind(py).len(),
            self.producer
        )
    }
}
