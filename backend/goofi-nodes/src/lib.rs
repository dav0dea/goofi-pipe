//! goofi-nodes — native Rust node implementations. Each is a thin adapter:
//! parse params into typed fields, compute, emit. Registered into the catalog
//! via `inventory`.
//!
//! Downstream binaries (engine/bridge) must reference this crate so the linker
//! keeps the `inventory::submit!` registrations — call [`native_node_count`].
//!
//! **Blank-slate reset:** the library is intentionally seeded with just two nodes,
//! `Oscillator` and `Buffer`, to be co-designed into an orthogonal set from here.
//! `test_nodes` is `_`-prefixed scaffolding behind the `test-nodes` feature, off by default, so a
//! release binary carries none of it. It is a FEATURE rather than `cfg(test)` because an
//! integration test is a separate crate linking the ordinary build: a node registered under
//! `cfg(test)` is invisible to it, which is what forced every integration test to hand-roll its own
//! fixtures.

mod buffer;
mod filter;
mod oscillator;
mod psd;
#[cfg(feature = "test-nodes")]
pub mod test_nodes;
#[cfg(feature = "test-nodes")]
mod test_source;

/// Force-links this crate's node registrations and reports how many native node
/// types are registered. Call once from a binary's startup so `inventory` keeps
/// the submissions.
pub fn native_node_count() -> usize {
    goofi_node::catalog().count()
}
