//! goofi-nodes — native Rust node implementations. Each is a thin adapter:
//! parse params into typed fields, compute, emit. Registered into the catalog
//! via `inventory`.
//!
//! Downstream binaries (engine/bridge) must reference this crate so the linker
//! keeps the `inventory::submit!` registrations — call [`native_node_count`].
//!
//! **Blank-slate reset:** the library is intentionally seeded with just two nodes,
//! `Oscillator` and `Buffer`, to be co-designed into an orthogonal set from here.
//! `test_source` is hidden test/bench scaffolding (`_TestConst`), not part of the
//! user-facing library.

mod buffer;
mod oscillator;
mod test_source;

/// Force-links this crate's node registrations and reports how many native node
/// types are registered. Call once from a binary's startup so `inventory` keeps
/// the submissions.
pub fn native_node_count() -> usize {
    goofi_node::catalog().count()
}
