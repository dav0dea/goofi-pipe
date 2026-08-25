//! Python node discovery: turn a `goofi.Node`-subclass file into an in-process node type the
//! engine hosts via `register_dyn_type`.

use std::path::Path;

use goofi_node::discover::{discover_one as probe_discover_one, Discovered, Discovery, NodeFactory};
use goofi_node::{Inputs, Isolation, Node, NodeCtx, NodeError, NodeManifest, NodeResult, Outputs, Params};

use super::PyNode;

/// A stand-in for a node whose construction failed; it reports the error from `setup()`.
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

/// Build a [`PyNode`], or a [`FailedNode`] — never a panic, because this runs under the graph mutex.
fn build_py_node(source: &str, in_slots: Vec<&'static str>, out_slots: Vec<&'static str>) -> Box<dyn Node> {
    match PyNode::from_source(source, in_slots, out_slots) {
        Ok(n) => Box::new(n),
        Err(e) => Box::new(FailedNode(format!("Python node construction failed: {e}"))),
    }
}

/// As [`build_py_node`], with the node wired to demote its own type when the GIL tripwire fires.
pub fn build_routed(
    source: &str,
    in_slots: Vec<&'static str>,
    out_slots: Vec<&'static str>,
    tier: &'static goofi_node::IsolationCell,
) -> Box<dyn Node> {
    match PyNode::from_source(source, in_slots, out_slots) {
        Ok(n) => Box::new(n.routed_by(tier)),
        Err(e) => Box::new(FailedNode(format!("Python node construction failed: {e}"))),
    }
}

/// A discovered Python node type, ready to register into a `Graph`.
pub struct PyNodeType {
    pub manifest: &'static NodeManifest,
    pub factory: NodeFactory,
}

/// Probe one file for this tier, reporting all three outcomes; the [`Discovered`] it yields
/// carries the `gil_safe` flag that routes between tiers.
pub fn probe(path: &Path, ft_python: &str) -> Discovery {
    probe_discover_one(path, ft_python, "python", Isolation::InProcess)
}

/// Turn a probe-[`Discovered`] into an in-process [`PyNodeType`], without a second spawn.
pub fn node_type_from(d: Discovered) -> PyNodeType {
    let path = d.source.clone();
    py_type_from_discovered(&path, d)
}

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
    /// Serializes access to the process-global interpreter.
fn interp() -> std::sync::MutexGuard<'static, ()> {
    static INTERP: std::sync::Mutex<()> = std::sync::Mutex::new(());
    INTERP.lock().unwrap_or_else(|e| e.into_inner())
}
    use goofi_node::ParamGroups;

    #[test]
    fn a_broken_python_source_builds_an_error_node_instead_of_panicking() {
        let _interp = interp();
        let mut node = build_py_node("def process(:\n    pass\n", vec!["data"], vec!["out"]); // invalid syntax → from_source Err
        let mut ctx = NodeCtx::new();
        let params = ParamGroups::new();
        let err = node.setup(&mut ctx, &Params::new(&params)).expect_err("construction failure must error");
        assert!(err.0.contains("construction failed"), "the error surfaces on setup: {}", err.0);
    }
}
