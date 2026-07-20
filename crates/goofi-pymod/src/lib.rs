//! goofi-pymod — the `goofi` Python package, built two ways from one source:
//! an abi3 cdylib wheel for subprocess GIL pythons, and linked (rlib) into the
//! free-threaded host via `pyo3::append_to_inittab!(goofi)`. Under no features it
//! is an empty crate so the default python-free workspace build stays green.

#![cfg(any(feature = "extension-module", feature = "host"))]

mod node;
mod params;

use pyo3::prelude::*;

/// Whether the running interpreter has the GIL disabled (free-threaded proof).
#[pyfunction]
fn gil_disabled(py: Python<'_>) -> PyResult<bool> {
    match py.import("sys")?.getattr("_is_gil_enabled") {
        Ok(f) => Ok(!f.call0()?.extract::<bool>()?),
        Err(_) => Ok(false),
    }
}

/// The `goofi` Python module. The `[lib] name` is `goofi`, so the init symbol is
/// `PyInit_goofi` and the wheel imports as `goofi`. `pub` so the host can
/// `append_to_inittab!` it.
#[pymodule]
pub fn goofi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gil_disabled, m)?)?;
    m.add_class::<node::Node>()?;
    m.add_class::<params::DataType>()?;
    m.add_class::<params::IntParam>()?;
    m.add_class::<params::FloatParam>()?;
    m.add_class::<params::BoolParam>()?;
    m.add_class::<params::StringParam>()?;
    Ok(())
}
