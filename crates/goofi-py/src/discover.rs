//! Python node discovery: scan a directory of `process(x)` node files and turn
//! each into a runtime node type the engine can host via `register_dyn_type`.
//!
//! First cut: every discovered node has the single-array shape the [`PyNode`]
//! bytes bridge supports — one F32 `data` input, one F32 `out` output. Files
//! whose source fails to compile or that don't define `process` are skipped
//! (they'd otherwise panic at instantiation), mirroring how the Python backend
//! greys out a broken node rather than crashing the graph.

use std::path::Path;

use goofi_node::discover::{camel, leak_process_manifest, NodeFactory};
use goofi_node::{Inputs, Isolation, Node, NodeCtx, NodeError, NodeManifest, NodeResult, Outputs, Params};

use crate::PyNode;

/// A stand-in for a Python node whose per-instance construction failed (its module re-exec raised).
/// It surfaces the error TERMINALLY from `setup()` — the node's bootstrap-error channel — instead of
/// the factory panicking. Discovery validates only the FIRST module exec; a repeat can still fail
/// (e.g. a top-level import acquiring an exclusive device/port on the 2nd instance). A panic in the
/// factory would be catastrophic: it runs under the manager's graph mutex (`add_node` holds it), so
/// it would POISON the mutex and take the whole control plane down.
struct FailedNode(String);

impl Node for FailedNode {
    fn setup(&mut self, _ctx: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Err(NodeError(self.0.clone()))
    }
    fn process(
        &mut self,
        _inp: &Inputs<'_>,
        _out: &mut Outputs<'_>,
        _ctx: &mut NodeCtx,
        _p: &Params<'_>,
    ) -> NodeResult {
        Err(NodeError(self.0.clone()))
    }
}

/// Build a Python node instance by re-execing its source. On failure returns a [`FailedNode`] that
/// reports the error via the bootstrap channel — it NEVER panics, because the factory runs under the
/// manager's graph mutex and a panic there poisons it (killing the control plane). This upholds the
/// module's stated contract: a broken node greys out / errors, it does not crash the graph.
fn build_py_node(source: &str) -> Box<dyn Node> {
    match PyNode::from_source(source, vec!["data"], vec!["out"]) {
        Ok(n) => Box::new(n),
        Err(e) => Box::new(FailedNode(format!("Python node construction failed: {e}"))),
    }
}

/// A discovered Python node type, ready to register into a `Graph`.
pub struct PyNodeType {
    /// The type's manifest (leaked to `'static`; discovery runs once at startup
    /// and the set of Python node types is bounded — catalog lifetime).
    pub manifest: &'static NodeManifest,
    /// Builds a fresh instance (recompiles the source per instance).
    pub factory: NodeFactory,
}

/// Build an in-process Python node type from a single file, or `None` if it is
/// not a node file: non-`.py`, `_`-prefixed (hidden), unreadable, or it doesn't
/// compile / lacks `process`. Used per-file by [`discover`] and by the CLI's
/// GIL-gate auto-router (which probes each file before choosing this backend).
pub fn discover_one(path: &Path) -> Option<PyNodeType> {
    if path.extension().and_then(|e| e.to_str()) != Some("py") {
        return None;
    }
    let stem = path.file_stem().and_then(|s| s.to_str())?;
    if stem.starts_with('_') {
        return None;
    }
    let source = std::fs::read_to_string(path).ok()?;
    // Validate: must compile and define a `goofi.Node` subclass (fail-fast at discovery).
    if PyNode::from_source(&source, vec!["data"], vec!["out"]).is_err() {
        return None;
    }

    let manifest = leak_process_manifest(
        camel(stem),
        format!("Python node from {}", path.display()),
        "python",
        Isolation::InProcess,
    );
    let factory: NodeFactory = Box::new(move |_p| build_py_node(&source));
    Some(PyNodeType { manifest, factory })
}

/// Scan `dir` for `*.py` node files (skipping `_`-prefixed) and return the ones
/// that compile and define `process`. Type names are the `CamelCase` file stem.
pub fn discover(dir: &Path) -> std::io::Result<Vec<PyNodeType>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    // Deterministic order (readdir is unordered) so type names are stable.
    entries.sort_by_key(|e| e.file_name());
    Ok(entries.iter().filter_map(|e| discover_one(&e.path())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_node::ParamGroups;

    // `camel` is unit-tested in goofi-node (its owner); no need to re-test the re-export here.

    #[test]
    fn a_broken_python_source_builds_an_error_node_instead_of_panicking() {
        // The factory runs under the manager's graph mutex; a panic on a per-instance construction
        // failure would poison it and kill the whole control plane. build_py_node must instead
        // return a node that surfaces the failure terminally on setup() (the bootstrap channel).
        let mut node = build_py_node("def process(:\n    pass\n"); // invalid syntax → from_source Err
        let mut ctx = NodeCtx::new();
        let params = ParamGroups::new();
        let err = node.setup(&mut ctx, &Params::new(&params)).expect_err("construction failure must error");
        assert!(err.0.contains("construction failed"), "the error surfaces on setup: {}", err.0);
    }
}
