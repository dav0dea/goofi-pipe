//! The `goofi` Python package: an abi3 cdylib wheel for subprocess GIL pythons, and an
//! rlib linked into the free-threaded host via `pyo3::append_to_inittab!(goofi)`.

// An empty crate under no features, so the default python-free workspace build stays green.
#![cfg(any(feature = "extension-module", feature = "host"))]

mod data;
pub mod exec;
mod introspect;
pub mod loader;
mod node;
mod params;
#[cfg(feature = "extension-module")]
mod serve;
mod stream;

pub use data::Data;

use pyo3::prelude::*;

/// The `goofi` Python module; `pub` so the host can `append_to_inittab!` it.
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
    m.add_class::<params::PulseParam>()?;
    m.add_class::<stream::Stream>()?;
    m.add_function(wrap_pyfunction!(introspect::introspect, m)?)?;
    #[cfg(feature = "extension-module")]
    m.add_function(wrap_pyfunction!(serve::serve, m)?)?;
    Ok(())
}
