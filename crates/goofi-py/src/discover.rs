//! Python node discovery: scan a directory of `goofi.Node`-subclass files and turn
//! each into an in-process runtime node type the engine hosts via `register_dyn_type`.
//!
//! Discovery runs the `goofi.introspect` probe (M1) on a free-threaded interpreter per
//! file: the probe's rich manifest carries the node's declared multi-slot inputs/outputs
//! + params, and the factory builds a class-contract [`PyNode`] bound to those slot names.
//! A file that isn't `.py`, is `_`-prefixed (hidden), or whose probe fails (missing dep /
//! no `Node` subclass) is skipped — greyed out, never a catalog crash.

use std::path::Path;

use goofi_node::discover::{discover as probe_discover, discover_one as probe_discover_one, Discovered, NodeFactory};
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

/// Build a class-contract [`PyNode`] bound to its manifest's slot names. On failure returns
/// a [`FailedNode`] that reports the error via the bootstrap channel — it NEVER panics,
/// because the factory runs under the manager's graph mutex and a panic there poisons it
/// (killing the control plane). A broken node greys out / errors, it does not crash the graph.
fn build_py_node(source: &str, in_slots: Vec<&'static str>, out_slots: Vec<&'static str>) -> Box<dyn Node> {
    match PyNode::from_source(source, in_slots, out_slots) {
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

/// Build an in-process Python node type from a single file by running the `goofi.introspect`
/// probe on `ft_python` (a free-threaded interpreter with `goofi` importable): the probe's
/// rich manifest gives multi-slot + params; the factory builds a class-contract [`PyNode`]
/// bound to that manifest's slot names. `None` if it is not a node file or the probe fails
/// (missing dep / no `Node` subclass) — greyed out, never a catalog crash.
pub fn discover_one(path: &Path, ft_python: &str) -> Option<PyNodeType> {
    let d = probe_discover_one(path, ft_python, "python", Isolation::InProcess)?;
    Some(py_type_from_discovered(path, d))
}

/// Scan `dir` for node files, probing each on `ft_python`; skips non-`.py`, `_`-prefixed,
/// and probe failures. Type names are the `CamelCase` file stem.
pub fn discover(dir: &Path, ft_python: &str) -> std::io::Result<Vec<PyNodeType>> {
    let discovered = probe_discover(dir, ft_python, "python", Isolation::InProcess)?;
    Ok(discovered
        .into_iter()
        .map(|d| {
            let path = d.source.clone();
            py_type_from_discovered(&path, d)
        })
        .collect())
}

/// Turn a probe-[`Discovered`] (rich manifest + `gil_safe` + source path) into an in-process
/// [`PyNodeType`]: the factory reads the file's source + builds a [`PyNode`] bound to the
/// manifest's slot names.
fn py_type_from_discovered(path: &Path, d: Discovered) -> PyNodeType {
    let manifest = d.manifest;
    let in_slots: Vec<&'static str> = manifest.inputs.iter().map(|s| s.name).collect();
    let out_slots: Vec<&'static str> = manifest.outputs.iter().map(|o| o.name).collect();
    let source = std::fs::read_to_string(path).unwrap_or_default();
    let factory: NodeFactory =
        Box::new(move |_p| build_py_node(&source, in_slots.clone(), out_slots.clone()));
    PyNodeType { manifest, factory }
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
        let mut node = build_py_node("def process(:\n    pass\n", vec!["data"], vec!["out"]); // invalid syntax → from_source Err
        let mut ctx = NodeCtx::new();
        let params = ParamGroups::new();
        let err = node.setup(&mut ctx, &Params::new(&params)).expect_err("construction failure must error");
        assert!(err.0.contains("construction failed"), "the error surfaces on setup: {}", err.0);
    }
}
