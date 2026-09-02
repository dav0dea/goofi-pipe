//! The signal node's author contract: the `Node` trait a signal node implements, the views it is
//! handed, and the C boundary a built node crosses — shared by the engine that runs one and the
//! file that authors one, so the two halves cannot drift.

use std::fmt;

use goofi_core::{Data, Param};
use goofi_node::ParamGroups;
use indexmap::IndexMap;

pub mod abi;
#[cfg(feature = "host")]
pub mod host;

pub use goofi_core;
pub use goofi_node::{ExprDecl, ExprMode, OutputDecl, ParamDecl, ParamKey, ParamSpec, Params, SlotDecl};

/// What a node file declares: a `NodeManifest` less the type name, which is the FILE's.
pub struct Manifest {
    pub category: &'static str,
    pub doc: &'static str,
    pub inputs: &'static [SlotDecl],
    pub outputs: &'static [OutputDecl],
    pub params: &'static [ParamDecl],
    pub producer: bool,
}

/// A signal node's failure, propagated to the health plane rather than panicking.
#[derive(Debug, Clone)]
pub struct NodeError(pub String);

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NodeError {}
impl From<String> for NodeError {
    fn from(s: String) -> Self {
        NodeError(s)
    }
}
impl From<&str> for NodeError {
    fn from(s: &str) -> Self {
        NodeError(s.to_string())
    }
}

pub type NodeResult = std::result::Result<(), NodeError>;

/// Builds a fresh boxed instance of a runtime-registered node type from its params.
pub type NodeFactory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// The per-run input view; the two maps are keyed disjointly, so a slot is single XOR multi.
pub struct Inputs<'a> {
    singles: &'a IndexMap<&'static str, Option<Data>>,
    multis: Option<&'a IndexMap<&'static str, Vec<Data>>>,
}

impl<'a> Inputs<'a> {
    pub fn new(singles: &'a IndexMap<&'static str, Option<Data>>) -> Inputs<'a> {
        Inputs { singles, multis: None }
    }
    pub fn with_multi(
        singles: &'a IndexMap<&'static str, Option<Data>>,
        multis: &'a IndexMap<&'static str, Vec<Data>>,
    ) -> Inputs<'a> {
        Inputs { singles, multis: Some(multis) }
    }
    /// The latest frame on a single slot, or the first present frame on a `multi` slot.
    pub fn get(&self, name: &str) -> Option<&Data> {
        if let Some(o) = self.singles.get(name) {
            if let Some(d) = o.as_ref() {
                return Some(d);
            }
        }
        self.multis.and_then(|m| m.get(name)).and_then(|v| v.first())
    }
    /// The present frames on a `multi` slot, or a 0/1-element slice on a single slot.
    pub fn get_multi(&self, name: &str) -> &[Data] {
        if let Some(m) = self.multis {
            if let Some(v) = m.get(name) {
                return v.as_slice();
            }
        }
        match self.singles.get(name) {
            Some(Some(d)) => std::slice::from_ref(d),
            _ => &[],
        }
    }
}

/// A pre-sized output sink; slots left unset emit nothing.
pub struct Outputs<'a> {
    slots: &'a mut IndexMap<&'static str, Option<Data>>,
}

impl<'a> Outputs<'a> {
    pub fn new(slots: &'a mut IndexMap<&'static str, Option<Data>>) -> Outputs<'a> {
        Outputs { slots }
    }
    /// Set an output slot. Writing an unknown slot name is a no-op.
    pub fn set(&mut self, name: &str, data: Data) {
        if let Some(s) = self.slots.get_mut(name) {
            *s = Some(data);
        }
    }
}

/// Per-run engine context handed to a node.
#[derive(Debug, Default, Clone)]
pub struct NodeCtx {
    /// Monotonic seconds since the PATCH began — one clock across every node thread.
    pub now: f64,
}

impl NodeCtx {
    pub fn new() -> NodeCtx {
        NodeCtx::default()
    }
}

pub trait Node: Send {
    /// Derived init, after the params have been seeded. An `Err` leaves the node uninitialized and
    /// the next interaction retries the whole initialization on this same instance.
    fn setup(&mut self, _ctx: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
    /// The run body: read latest inputs + live params, write outputs.
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        ctx: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult;
    /// React to a param edit; the engine replays it per declared param at initialization.
    fn on_param_changed(&mut self, _key: &ParamKey, _v: &Param) -> NodeResult {
        Ok(())
    }
    /// Re-enumerate a `Str` param's options — the ⟳ button. `p` is the node's LIVE params.
    fn on_param_refreshed(&mut self, _key: &ParamKey, _p: &Params<'_>) -> Option<Vec<String>> {
        None
    }
    // Teardown is `impl Drop`, and does NOT fire between initialization retries: a `setup` that
    // fails partway must release what it acquired before returning `Err`.
}
