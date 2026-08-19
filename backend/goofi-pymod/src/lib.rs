//! goofi-pymod — the `goofi` Python package, built two ways from one source:
//! an abi3 cdylib wheel for subprocess GIL pythons, and linked (rlib) into the
//! free-threaded host via `pyo3::append_to_inittab!(goofi)`. Under no features it
//! is an empty crate so the default python-free workspace build stays green.

#![cfg(any(feature = "extension-module", feature = "host"))]

mod data;
pub mod exec;
mod introspect;
pub mod loader;
mod node;
mod params;
// The subprocess child loop (Rust iceoryx2 + shared codec) — wheel only.
#[cfg(feature = "extension-module")]
mod serve;

pub use data::Data;

use pyo3::prelude::*;

/// The `goofi` Python module. The `[lib] name` is `goofi`, so the init symbol is
/// `PyInit_goofi` and the wheel imports as `goofi`. `pub` so the host can
/// `append_to_inittab!` it.
#[pymodule]
pub fn goofi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<data::Data>()?;
    m.add_class::<data::Ndims>()?;
    m.add_class::<node::Node>()?;
    m.add_class::<params::DataType>()?;
    m.add_class::<params::InputSlot>()?;
    m.add_class::<params::IntParam>()?;
    m.add_class::<params::FloatParam>()?;
    m.add_class::<params::BoolParam>()?;
    m.add_class::<params::StringParam>()?;
    m.add_function(wrap_pyfunction!(introspect::introspect, m)?)?;
    // The subprocess child entry point (`goofi.serve()`), only in the wheel.
    #[cfg(feature = "extension-module")]
    m.add_function(wrap_pyfunction!(serve::serve, m)?)?;
    Ok(())
}
