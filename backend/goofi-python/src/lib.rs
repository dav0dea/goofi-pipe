//! The manager side of both Python node tiers, behind one `Node` trait.

/// The subprocess tier: `RemoteNode` and the iceoryx2 round-trip to a child interpreter.
pub mod subproc;

/// The in-process tier: `PyNode`, the param-expression evaluator, and discovery. Links libpython.
#[cfg(feature = "embed")]
pub mod inproc;

pub use goofi_node::discover::{Discovered, Discovery};

/// Registers the `goofi` module into the inittab, which must happen BEFORE the interpreter
/// initializes — so every Python entry point in this crate attaches through [`attach`].
#[cfg(feature = "embed")]
mod pyinit {
    use std::sync::Once;

    use goofi_pymod::goofi as goofi_module;
    use pyo3::prelude::*;

    fn ensure_goofi_module() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
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


/// A discovered Python type registered as ONE type whose factory reads the type's tier cell at
/// build time. That cell is the only thing that decides the tier, so demoting a type is a single
/// write and the next `restart_node` honours it.
#[cfg(feature = "embed")]
pub fn routed_node_type(d: Discovered, subproc_python: &str) -> inproc::PyNodeType {
    let manifest = d.manifest;
    let tier = d.isolation;
    let in_slots: Vec<&'static str> = manifest.inputs.iter().map(|s| s.name).collect();
    let out_slots: Vec<&'static str> = manifest.outputs.iter().map(|o| o.name).collect();
    let source = std::fs::read_to_string(&d.source).unwrap_or_default();
    let python = subproc_python.to_string();
    let factory: goofi_node::discover::NodeFactory = Box::new(move |_p| {
        match tier.get() {
            goofi_node::Isolation::Subprocess => {
                Box::new(subproc::RemoteNode::new(&python, &source, in_slots.clone()))
                    as Box<dyn goofi_node::Node>
            }
            // A native tier cannot reach here: this factory only ever backs a discovered file.
            _ => inproc::build_routed(&source, in_slots.clone(), out_slots.clone(), tier),
        }
    });
    inproc::PyNodeType { manifest, isolation: tier, factory }
}
