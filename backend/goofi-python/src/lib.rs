//! goofi-python — the MANAGER side of both Python node tiers, behind one `Node` trait.
//!
//! A Python node runs in one of two places, and the choice is not a setting: the discovery probe
//! imports the module in a real interpreter and reports whether the GIL is still disabled
//! afterwards. Safe → [`inproc`], inline on the tick's rayon pool in an embedded free-threaded
//! CPython 3.14t. Otherwise → [`subproc`], a detached off-tick worker on a GIL interpreter,
//! reached over iceoryx2 shared memory.
//!
//! Both tiers drive the same `goofi.Node` class contract through the same
//! `goofi_pymod::exec` marshalling — proven by `tests/parity.rs`, which runs one source through
//! both — and both expose the same `probe` / `node_type_from` pair, so a caller routing between
//! them writes the same two calls and names only the tier.
//!
//! The tiers differ in ONE way that matters to this crate's shape: [`inproc`] **links**
//! libpython, so it sits behind the `embed` feature and the default build of this crate needs no
//! interpreter at all. [`subproc`] only ever **spawns** one, so it needs no gate. (`goofi-cli`
//! still takes the two together, under its own `python` feature — without a probe there is
//! nothing to route, so half a router would be no router.)

/// The subprocess tier: `RemoteNode`, the spawn/iceoryx2 round-trip, and the error frames a
/// raising node replies with. No pyo3 — this end speaks `goofi-codec` frames to a child running
pub mod subproc;

/// The in-process tier: `PyNode`, the pyo3 param-expression evaluator, and discovery. Links
/// libpython, hence the feature gate.
#[cfg(feature = "embed")]
pub mod inproc;

/// The probe outcome, shared by both tiers: `Found` / `Unavailable` (a real node that cannot load
/// here, named with its failing import) / `Skip` (not a node file at all). Re-exported at the
/// crate root because it is the one vocabulary a tier-routing caller needs from both halves.
pub use goofi_node::discover::Discovery;

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

