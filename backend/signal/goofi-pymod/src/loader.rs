//! Loading a node module + finding its `goofi.Node` subclass — the one implementation the
//! discovery probe and the in-process executor share.

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};

use crate::node::Node;

/// Held while a node module's body runs, so no two run at once.
///
/// A package whose `__init__` imports its submodules and whose submodules import back through the
/// package is legal under the GIL — a partially initialized module is visible — and deadlocks
/// without one: two threads entering the cycle at different points each hold the module lock the
/// other wants, and CPython raises `_DeadlockError` rather than hanging. biotuner is such a
/// package, and goofi hosts whatever the user installed, so this is the host's to prevent.
///
/// Only the FIRST import of a package can cycle; after it the module is in `sys.modules` and every
/// later import is a lookup. A node body is where a node's imports first run, so serializing the
/// bodies is enough — and it costs nothing after, since nothing here is on the frame path.
fn one_at_a_time(py: Python<'_>) -> std::sync::MutexGuard<'static, ()> {
    static BODIES: std::sync::Mutex<()> = std::sync::Mutex::new(());
    loop {
        if let Ok(held) = BODIES.try_lock() {
            return held;
        }
        // A guard cannot cross `detach`, so waiting and holding are two steps: block DETACHED
        // until it is free, then race for it attached. Waiting attached is what must not happen —
        // the GIL tripwire can turn the GIL back on, and a waiter holding it would stop the very
        // thread it waits for.
        py.detach(|| drop(BODIES.lock()));
    }
}

/// Compile a node module from an in-memory source string (the host factory path).
pub fn module_from_source<'py>(
    py: Python<'py>,
    name: &str,
    source: &str,
) -> PyResult<Bound<'py, PyModule>> {
    let file = format!("{name}.py");
    let _bodies = one_at_a_time(py);
    PyModule::from_code(
        py,
        std::ffi::CString::new(source)?.as_c_str(),
        std::ffi::CString::new(file)?.as_c_str(),
        std::ffi::CString::new(name)?.as_c_str(),
    )
}

/// Load a node module from an arbitrary file path via `importlib.util` (the probe path).
pub fn module_from_path<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyModule>> {
    let _bodies = one_at_a_time(py);
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
