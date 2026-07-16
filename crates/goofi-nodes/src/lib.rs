//! goofi-nodes — native Rust node implementations. Each is a thin adapter:
//! parse params into typed fields, compute, emit. Registered into the catalog
//! via `inventory`.
//!
//! Downstream binaries (engine/bridge) must reference this crate so the linker
//! keeps the `inventory::submit!` registrations — call [`native_node_count`].

mod autocorrelation;
mod buffer;
mod clip;
mod colorenhancer;
mod compass;
mod constant_array;
mod delay;
mod fft;
mod frequencyshift;
mod function;
mod ifft;
mod join;
mod joinstring;
mod math;
mod normalization;
mod oscillator;
mod padding;
mod powerband;
mod powerbandeeg;
mod psd;
mod reduce;
mod reshape;
mod setmeta;
mod smooth;
mod threshold;
mod timedelayembedding;
mod transpose;

/// Force-links this crate's node registrations and reports how many native node
/// types are registered. Call once from a binary's startup so `inventory` keeps
/// the submissions.
pub fn native_node_count() -> usize {
    goofi_node::catalog().count()
}
