//! Loading a node module + finding its `goofi.Node` subclass — the one implementation
//! shared by the discovery probe (`introspect`) and the in-process executor
//! (`goofi-python`'s `inproc::PyNode`). A module is loaded from a file path (discovery) or an
//! in-memory source string (the host's factory + tests); either way `find_node_class`
//! returns the first strict `Node` subclass, and `instantiate` calls it.

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};

use crate::node::Node;

/// Compile a node module from an in-memory source string (the host factory path).
pub fn module_from_source<'py>(
    py: Python<'py>,
    name: &str,
    source: &str,
) -> PyResult<Bound<'py, PyModule>> {
    let file = format!("{name}.py");
    PyModule::from_code(
        py,
        std::ffi::CString::new(source)?.as_c_str(),
        std::ffi::CString::new(file)?.as_c_str(),
        std::ffi::CString::new(name)?.as_c_str(),
    )
}

/// Load a node module from an arbitrary file path via `importlib.util` (the probe path).
pub fn module_from_path<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyModule>> {
    let util = py.import("importlib.util")?;
    let spec = util.call_method1("spec_from_file_location", ("goofi_probe_mod", path))?;
    if spec.is_none() {
        return Err(pyo3::exceptions::PyImportError::new_err(format!("cannot load {path}")));
    }
    let module = util.call_method1("module_from_spec", (&spec,))?;
    spec.getattr("loader")?.call_method1("exec_module", (&module,))?;
    module.cast_into::<PyModule>().map_err(Into::into)
}

/// The first strict `goofi.Node` subclass defined in the module, else raise.
pub fn find_node_class<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyType>> {
    let node_ty = py.get_type::<Node>();
    for (_name, val) in module.dict().iter() {
        let Ok(ty) = val.cast_into::<PyType>() else {
            continue;
        };
        if !ty.is(&node_ty) && ty.is_subclass(&node_ty)? {
            return Ok(ty);
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err("no goofi.Node subclass in module"))
}

/// Find + instantiate the node module's `Node` subclass (no args to `__init__`).
pub fn instantiate<'py>(py: Python<'py>, module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyAny>> {
    find_node_class(py, module)?.call0()
}
