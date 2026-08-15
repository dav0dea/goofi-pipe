//! The in-process Python tier: a node whose imports left the GIL disabled runs inline on the
//! tick's rayon pool, hosted by a pyo3-embedded free-threaded CPython 3.14t.
//!
//! Everything here is behind the crate's `embed` feature, because it LINKS libpython — which is
//! the whole reason this half is separable from [`crate::subproc`], whose child interpreter is
//! merely spawned. The two tiers run the same `goofi.Node` contract through the same
//! `goofi_pymod::exec` marshalling, and expose the same `probe` / `node_type_from` pair, so the
//! caller routing between them names only the tier.

mod discover;
mod expr;
mod host;

pub use discover::{node_type_from, probe, PyNodeType};
pub use expr::PyExprEvaluator;
pub use host::{interpreter_path, PyNode};
