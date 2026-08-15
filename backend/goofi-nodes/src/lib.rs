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
    /// Catalog validation: every `expression` a node declares must READ ONLY GLOBALS THAT EXIST
    /// in a fresh patch — i.e. `goofi_core::globals::SYSTEM_GLOBALS`. Seeding runs on a fresh add
    /// (`seed_default_expressions`), where the only globals in the store are the system ones, so a
    /// typo'd `globals.defualt_ufreq` compiles, binds, and then errors at eval on every instance of
    /// that node type — the param falls back to its literal and the node wears an error badge.
    ///
    /// The "targets a declared param" check this test used to make cannot fail and is gone: an
    /// `expression` lives ON the decl it targets, and `with_common` keeps whatever `common.*`
    /// keys a node declared, so the target always exists by construction.
    ///
    /// Cheap, evaluator-free, and runs over the whole linked catalog PLUS the universal `common`
    /// group, which every node carries and which now declares one itself.
    #[test]
    fn every_declared_expression_reads_only_system_globals() {
        let decls = goofi_node::catalog()
            .map(|m| (m.type_name, m.params))
            .chain(std::iter::once(("common", goofi_node::COMMON_DECLS)));
        for (owner, params) in decls {
            for decl in params {
                let Some(expr) = decl.expression else { continue };
                assert!(!expr.trim().is_empty(), "{}: {}/{} has an empty expression", owner, decl.group, decl.name);
                for name in goofi_node::global_ref_names(expr) {
                    assert!(
                        goofi_core::globals::SYSTEM_GLOBALS.iter().any(|g| g.name == name),
                        "{}: the expression on {}/{} reads `globals.{}`, which no fresh patch has",
                        owner,
                        decl.group,
                        decl.name,
                        name
                    );
                }
            }
        }
    }
}
