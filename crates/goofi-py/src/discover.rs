//! Python node discovery: scan a directory of `process(x)` node files and turn
//! each into a runtime node type the engine can host via `register_dyn_type`.
//!
//! First cut: every discovered node has the single-array shape the [`PyNode`]
//! bytes bridge supports — one F32 `data` input, one F32 `out` output. Files
//! whose source fails to compile or that don't define `process` are skipped
//! (they'd otherwise panic at instantiation), mirroring how the Python backend
//! greys out a broken node rather than crashing the graph.

use std::path::Path;

use goofi_node::{Isolation, Node, NodeManifest, OutputDecl, ParamGroups, SlotDecl};

use crate::PyNode;

// Shared slot shape for a `process(x) -> array` node. Truly `'static` (no leak).
static PY_IN: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: goofi_core::SlotType::Array,
    trigger_process: true,
}];
static PY_OUT: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: goofi_core::SlotType::Array,
    length_preserving: true,
}];

fn py_params() -> ParamGroups {
    ParamGroups::new()
}
fn py_stub_make(_: &ParamGroups) -> Box<dyn Node> {
    unreachable!("a discovered Python node is built by its factory, not manifest.make")
}

/// Builds a node instance from its params (mirrors the engine's `NodeFactory`;
/// duplicated here so goofi-py needn't depend on goofi-engine).
pub type PyNodeFactory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// A discovered Python node type, ready to register into a `Graph`.
pub struct PyNodeType {
    /// The type's manifest (leaked to `'static`; discovery runs once at startup
    /// and the set of Python node types is bounded — catalog lifetime).
    pub manifest: &'static NodeManifest,
    /// Builds a fresh instance (recompiles the source per instance).
    pub factory: PyNodeFactory,
}

/// Convert a `snake_case` file stem to a `CamelCase` palette type name.
fn camel(stem: &str) -> String {
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

/// Scan `dir` for `*.py` node files (skipping `_`-prefixed) and return the ones
/// that compile and define `process`. Type names are the `CamelCase` file stem.
pub fn discover(dir: &Path) -> std::io::Result<Vec<PyNodeType>> {
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    // Deterministic order (readdir is unordered) so type names are stable.
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("py") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem.starts_with('_') {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(&path) else {
            continue;
        };
        // Validate: must compile and define `process`. Skip if not (fail-fast at
        // discovery instead of a panic when the node is first instantiated).
        if PyNode::from_source(&source, "process").is_err() {
            continue;
        }

        let type_name: &'static str = Box::leak(camel(stem).into_boxed_str());
        let doc: &'static str =
            Box::leak(format!("Python node from {}", path.display()).into_boxed_str());
        let manifest: &'static NodeManifest = Box::leak(Box::new(NodeManifest {
            type_name,
            category: "python",
            doc,
            inputs: PY_IN,
            outputs: PY_OUT,
            default_params: py_params,
            isolation: Isolation::InProcess,
            make: py_stub_make,
        }));
        let factory: PyNodeFactory = Box::new(move |_p| {
            Box::new(PyNode::from_source(&source, "process").expect("validated at discovery"))
                as Box<dyn Node>
        });
        out.push(PyNodeType { manifest, factory });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camel_case_conversion() {
        assert_eq!(camel("double"), "Double");
        assert_eq!(camel("my_band_filter"), "MyBandFilter");
        assert_eq!(camel("psd"), "Psd");
    }
}
