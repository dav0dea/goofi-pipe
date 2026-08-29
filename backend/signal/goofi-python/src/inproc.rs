//! The in-process Python tier: a node whose imports left the GIL disabled, hosted by a
//! pyo3-embedded free-threaded CPython.

mod discover;
mod expr;
mod host;

pub use discover::{build_routed, node_type_from, probe, PyNodeType};
pub use expr::PyExprEvaluator;
pub use host::{interpreter_path, PyNode};
