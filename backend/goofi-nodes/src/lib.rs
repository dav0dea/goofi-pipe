//! Native Rust node implementations, registered into the catalog via `inventory`.
//! `test_nodes` sits behind a FEATURE, not `cfg(test)`: an integration test is a separate crate,
//! and a node registered under `cfg(test)` is invisible to it.

mod buffer;
mod filter;
mod oscillator;
mod psd;
#[cfg(feature = "test-nodes")]
pub mod test_nodes;
#[cfg(feature = "test-nodes")]
mod test_source;

/// Force-links this crate's node registrations, and reports how many are registered.
pub fn native_node_count() -> usize {
    goofi_node::catalog().count()
}
