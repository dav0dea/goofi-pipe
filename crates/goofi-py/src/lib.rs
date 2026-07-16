//! goofi-py — the in-process Python node host (pyo3, free-threaded CPython 3.14t).
//!
//! Gated behind the `embed` feature so the default workspace build needs no
//! Python. With the feature on and a free-threaded interpreter (`PYO3_PYTHON`
//! set to a python3.14t at build), a [`PyNode`] runs real Python node logic —
//! including numpy — in-process, in parallel with native nodes, behind the same
//! `Node` trait. The GIL stays disabled, verifiable via [`PyNode::gil_enabled`].
//!
//! This first cut uses a dependency-light bytes bridge (numpy `frombuffer` /
//! `tobytes`); the zero-copy `rust-numpy` view path is a follow-on optimization.

#[cfg(feature = "embed")]
mod host;

#[cfg(feature = "embed")]
pub use host::{interpreter_path, PyNode};

#[cfg(feature = "embed")]
mod discover;

#[cfg(feature = "embed")]
pub use discover::{discover, discover_one, PyNodeType};
