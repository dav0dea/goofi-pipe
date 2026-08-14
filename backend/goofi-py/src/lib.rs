//! goofi-py — the in-process Python node host (pyo3, free-threaded CPython 3.14t).
//!
//! Gated behind the `embed` feature so the default workspace build needs no
//! Python. With the feature on and a free-threaded interpreter (`PYO3_PYTHON`
//! set to a python3.14t at build), a [`PyNode`] runs real Python node logic —
//! including numpy — in-process, in parallel with native nodes, behind the same
//! `Node` trait. The GIL stays disabled, verifiable via [`PyNode::gil_enabled`].
//!
//! Node values cross the boundary via numpy `frombuffer` / `tobytes` copies (the
//! zero-copy `rust-numpy` view path is a follow-on optimization).

/// Interpreter bootstrap: register the `goofi` module into the embedded interpreter's
/// inittab exactly once, BEFORE the interpreter is initialized. Every Python entry point
/// in this crate — the expression evaluator AND node building — attaches via [`attach`],
/// so whichever runs first registers `goofi` before pyo3's auto-initialize calls
/// `Py_Initialize`. Registering after init would panic (`append_to_inittab` requires an
/// un-initialized interpreter), which is exactly the M2 startup-order hazard this closes:
/// `main()` constructs the evaluator (initializing the interpreter) before any node.
#[cfg(feature = "embed")]
mod pyinit {
    use std::sync::Once;

    use goofi_pymod::goofi as goofi_module;
    use pyo3::prelude::*;

    fn ensure_goofi_module() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            // Runs before the first `Python::attach` in the process (all attaches funnel
            // through `attach`), so the interpreter is not yet initialized here.
            pyo3::append_to_inittab!(goofi_module);
        });
    }

    /// `Python::attach` preceded by the one-time inittab registration.
    pub(crate) fn attach<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        ensure_goofi_module();
        Python::attach(f)
    }
}

#[cfg(feature = "embed")]
pub(crate) use pyinit::attach;

/// The whole crate's tests share ONE embedded interpreter while cargo runs them on parallel
/// threads, so anything that reads process-global interpreter state (`sys._is_gil_enabled`) can
/// otherwise observe a sibling mid-import and fail spuriously. Every test that drives the
/// interpreter — in `host`, `expr` and `discover` alike — holds this for its duration.
#[cfg(all(test, feature = "embed"))]
pub(crate) mod testlock {
    static INTERP: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) fn interp() -> std::sync::MutexGuard<'static, ()> {
        // Recover from a poisoned lock: one failing test must not cascade into all the others.
        INTERP.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(feature = "embed")]
mod host;

#[cfg(feature = "embed")]
pub use host::{interpreter_path, PyNode};

#[cfg(feature = "embed")]
mod discover;

#[cfg(feature = "embed")]
pub use discover::{node_type_from, probe, PyNodeType};
pub use goofi_node::discover::Discovery;

#[cfg(feature = "embed")]
mod expr;

#[cfg(feature = "embed")]
pub use expr::PyExprEvaluator;
