//! The engine seam: the trait an engine registers behind, and the settled view its settle reads.
//! The graph looks down at this and nothing engine-specific; an engine looks down at this and the
//! transport, and never at the graph.

use std::collections::HashMap;

use goofi_core::Param;

use crate::{BindingId, NodeManifest, ParamGroups, ParamKey, Status, Uid};

/// A doorbell id: `0` is a control message, `1..=64` the index of an input slot in
/// `manifest.inputs`, `65..=128` an expression channel the graph allocated at bind time.
pub type EventId = u8;

/// One resolved expression variable, graph-side: the model's spelling, which a view exposes and
/// an engine projects onto its own wire vocabulary.
#[derive(Clone, Debug)]
pub enum BoundVar {
    /// A producer's output slot, and the doorbell id it rings this consumer with.
    Stream { var: String, producer: Uid, slot: &'static str, event_id: EventId },
    /// A `globals.*` read, resolved and shipped inline — a globals edit re-sends the binding.
    Value { var: String, value: Param },
    /// The graph could not resolve it: an unknown node, a slot that does not exist, an ambiguous
    /// bare `nd()` on a multi-output producer, a global that is not defined.
    Missing { var: String, reason: String },
}

/// One port-resolved leaf-to-leaf edge, in link order — which IS a multi input's wire order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub producer: (Uid, &'static str),
    pub consumer: (Uid, &'static str),
}

/// One binding as settle ships it: the derived state an engine reads, never the authored record.
pub struct BindingView<'a> {
    pub key: &'a ParamKey,
    pub rewritten: &'a str,
    pub vars: &'a [BoundVar],
    pub trigger: bool,
    pub id: Option<BindingId>,
    /// Whether the graph ships it — a disabled or unbindable binding leaves the literal standing.
    pub live: bool,
}

/// One running node as the settled view carries it.
pub struct NodeView<'a> {
    pub engine: &'static str,
    pub generation: u64,
    /// Whether this node's engine wakes it by doorbell; a scheduled consumer is never rung.
    pub rings: bool,
    pub manifest: &'static NodeManifest,
    pub params: &'a ParamGroups,
    pub bindings: Vec<BindingView<'a>>,
}

/// The settled graph, as every engine reads it after a batch: the WHOLE graph — engines filter.
pub struct GraphView<'a> {
    pub instance: &'a str,
    pub edges: &'a [Edge],
    pub nodes: HashMap<Uid, NodeView<'a>>,
}

impl GraphView<'_> {
    /// A consumer input slot's desired producers, in wire order.
    pub fn wires_into(&self, uid: Uid, slot: &str) -> impl Iterator<Item = (Uid, &'static str)> + '_ {
        let want = (uid, slot.to_string());
        self.edges
            .iter()
            .filter(move |e| e.consumer.0 == want.0 && e.consumer.1 == want.1)
            .map(|e| e.producer)
    }
}

/// One thing a batch of ops changed, recorded by the op path for the settle that follows. The
/// delivery half of a write is deferred so one batch yields ONE decision, from settled state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Touched {
    /// A consumer input whose wire set may have moved.
    Slot(Uid, &'static str),
    /// A param whose value or binding moved and must reach its node.
    Param(Uid, ParamKey),
}

/// A one-time imperative a node must act on — what settled state cannot express.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    RefreshParam { key: ParamKey },
}

/// One node class an engine advertises: the shared manifest plus the display tier. The engine a
/// type belongs to is WHICH library advertises it — no tag field exists anywhere.
#[derive(Clone, Copy)]
pub struct LibraryEntry {
    pub manifest: &'static NodeManifest,
    pub tier: &'static str,
}

/// An engine: the runtime authority for its nodes. The graph applies every op to the MODEL and
/// propagates through these doors; an engine owns instances, health reporting, within-engine
/// transport and its own library, and the graph sees none of them.
pub trait Engine: Send {
    /// The id a registration is keyed by, and the palette's provenance for this library.
    fn id(&self) -> &'static str;
    /// Whether this engine's nodes wake on doorbells. A scheduled engine drains its boundary
    /// subscribers before each tick instead, and a producer facing it rings nothing.
    fn doorbell_driven(&self) -> bool;
    /// Every node class this engine can build, advertised on request.
    fn library(&self) -> Vec<LibraryEntry>;
    /// The record a fresh instance of `type_name` starts from: the declared defaults plus this
    /// engine's own universal groups, with `supplied` values folded in.
    fn normalize_params(
        &self,
        type_name: &str,
        supplied: Option<ParamGroups>,
    ) -> Result<ParamGroups, String>;
    /// Birth at `uid`, with the graph-minted generation. `Some` carries a boot error: the node
    /// then exists holding its place and saying why it is not running.
    fn insert(
        &mut self,
        uid: Uid,
        type_name: &str,
        generation: u64,
        params: &ParamGroups,
    ) -> Option<String>;
    fn remove(&mut self, uid: Uid);
    /// Deliver a settled batch: the touched items, plus whatever the engine's own drain marked
    /// pending. The ONLY place an engine composes messages — drain collects, settle decides.
    fn settle(&mut self, view: &GraphView<'_>, touched: &[Touched]);
    /// Hand over every queued health report. A pull: the caller owns the pace.
    fn drain(&mut self, apply: &mut dyn FnMut(Uid, Status)) -> usize;
    fn request(&mut self, uid: Uid, request: Request);
    fn shutdown(&mut self);
}
