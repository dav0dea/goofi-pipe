//! Runtime node-discovery scaffolding shared by every discovery backend (in-process Python,
//! subprocess Python, …): the runtime factory type, the `snake_case`→`CamelCase` type-name rule, the
//! fixed `process(x)` I/O shape, and the `'static` manifest leak. A backend supplies only its own
//! seam — the validate predicate + the factory closure + its category/isolation — so the same file
//! yields the same palette type name whichever backend hosts it.

use goofi_core::SlotType;

use crate::{Isolation, Node, NodeManifest, OutputDecl, ParamGroups, SlotDecl};

/// Builds a fresh boxed instance of a runtime-discovered node type from its params. A bare `fn`
/// pointer can't close over per-type state (a source string, a device handle), so this is a boxed
/// closure — shared by the engine's `register_dyn_type` and every discovery backend.
pub type NodeFactory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// `snake_case` file stem → `CamelCase` palette type name. One source of this rule, so the same file
/// yields the same type name whichever backend hosts it (in-process `PyNode` vs subprocess `RemoteNode`).
pub fn camel(stem: &str) -> String {
    stem.split('_')
        .filter(|s| !s.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// The fixed I/O of a discovered `process(x)` node: one ARRAY `data` input that triggers a tick,
/// one ARRAY `out` output.
pub static PROCESS_IN: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
pub static PROCESS_OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

/// Leak a `'static` [`NodeManifest`] for a discovered process-node type. The I/O shape, empty
/// params, and an unreachable stub `factory` (a runtime type is built by its registered
/// [`NodeFactory`], never `manifest.factory`) are fixed; `category` / `isolation` / `doc` vary per
/// backend. The leak is bounded — one manifest per discovered type, catalog-lifetime.
pub fn leak_process_manifest(
    type_name: String,
    doc: String,
    category: &'static str,
    isolation: Isolation,
) -> &'static NodeManifest {
    fn stub() -> Box<dyn Node> {
        unreachable!("a discovered node is built by its registered factory, not manifest.factory")
    }
    Box::leak(Box::new(NodeManifest {
        type_name: Box::leak(type_name.into_boxed_str()),
        category,
        doc: Box::leak(doc.into_boxed_str()),
        inputs: PROCESS_IN,
        outputs: PROCESS_OUT,
        params: &[],
        isolation,
        factory: stub,
    }))
}

#[cfg(test)]
mod tests {
    use super::camel;

    #[test]
    fn camel_case_conversion() {
        assert_eq!(camel("double"), "Double");
        assert_eq!(camel("my_band_filter"), "MyBandFilter");
        assert_eq!(camel(""), "");
        assert_eq!(camel("__weird__name"), "WeirdName");
    }
}
