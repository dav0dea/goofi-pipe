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

#[cfg(test)]
mod tests {
    /// Catalog validation: every `default_expr` a node declares must target a param that actually
    /// exists on that node — either one it declares, or a `common.*` scheduling param synthesized by
    /// `with_common`. Otherwise the fresh-add seeding would silently no-op (`set_expression` rejects
    /// an unknown param). Cheap, evaluator-free, and runs over the whole linked catalog.
    #[test]
    fn every_default_expr_targets_a_declared_param() {
        for m in goofi_node::catalog() {
            for decl in m.params {
                let Some(expr) = decl.default_expr else { continue };
                assert!(!expr.trim().is_empty(), "{}: {}/{} has an empty default_expr", m.type_name, decl.group, decl.name);
                let targets_declared = m.params.iter().any(|d| d.group == decl.group && d.name == decl.name);
                assert!(
                    targets_declared || decl.group == "common",
                    "{}: default_expr on undeclared param {}/{}",
                    m.type_name,
                    decl.group,
                    decl.name
                );
            }
        }
    }
}
