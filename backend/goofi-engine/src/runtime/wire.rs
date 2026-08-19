//! The node's wire contract (spec §3.4): the messages a node exchanges with the graph, and the
//! transport seam that carries them.
//!
//! The seam is deliberately payload-shaped rather than iceoryx2-shaped: the node's scheduling is
//! decided from what a message SAYS, never from how it arrived, so the runtime can be driven by an
//! in-memory transport in a test and by shared memory in production without a second code path.
//! [`MemoryTransport`] is the test one; [`super::IoxTransport`] is the shipped one.
//!
//! There is no `Control::Terminate`. A node's manager-side thread is stopped through the
//! [`Halt`](super::Halt) flag it was born holding, because the one moment a removal has to work is
//! exactly the one the control channel cannot serve: a node deleted before it answered
//! [`Status::Ready`] has no sink, so a `Terminate` addressed to it would be dropped and the thread
//! would outlive its graph. The flag is also what a whole-graph `clear` sets, where there is no sequence to order.
//!
//! Both message sets are **msgpack** on the wire, over iceoryx2 rather than a Rust channel, so the
//! thread and subprocess cases are the same code — a param edit has no latency requirement, but a
//! second transport implementation is a correctness surface.

use std::time::Duration;

use goofi_core::{Data, Param};
use goofi_node::ParamKey;
use serde::{Deserialize, Serialize};

use super::NodeFault;

/// Why a node woke (spec §3.2): `0` is a control message, `1..=64` the index of an input slot in
/// `manifest.inputs`, `65..=128` an `nd()` channel the graph allocated at bind time.
pub type EventId = u8;

/// An iceoryx2 service name. A wire's identity IS its service name, which is why the slot messages
/// never carry a source uid.
pub type ServiceName = String;

/// The generated name of an expression variable (`__v0`), minted by the graph's rewrite (§5.3).
pub type VarName = String;

/// The messages the graph sends a node.
///
/// The slot messages are **declarative**: each carries the complete desired set for that slot and
/// the node diffs it, so wiring is idempotent, an empty set means disconnected, and a displaced
/// single-input wire falls out for free — the consumer's new set is simply the new producer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Control {
    /// Every producer service feeding this input slot, in wire order. That order IS
    /// `Inputs::get_multi`'s connection order (§3.5), which is why it survives a producer restart:
    /// the service name changes, the position does not.
    InSlot { slot: String, services: Vec<ServiceName> },
    /// Every doorbell to ring after publishing on this output slot, with the [`EventId`] that says
    /// WHY the far node woke. The union of this slot's wire consumers and its expression
    /// subscribers — one set, because a node cannot tell the two apart and does not need to.
    OutSlot { slot: String, targets: Vec<(ServiceName, EventId)> },
    /// Write a param: a literal, or the expression to bind it to. This is the NOTIFICATION path —
    /// a node parked with `next_wake() == None` is never rung by a bare param-record swap.
    SetParam { key: ParamKey, value: ParamValue },
    /// Re-enumerate a refreshable `Str` param's options. The answer comes back as
    /// [`Status::RefreshOptions`] rather than on this message's ack (§8.5): the hook runs on the
    /// node's own thread, so the RPC that asked cannot wait for it without re-introducing the very
    /// stall the tick's removal was for.
    RefreshParam { key: ParamKey },
}

/// A control message and the sequence number the node acks it with (§3.4). `seq` is graph-minted
/// and monotonic, and it is what makes the three-phase wire sequence orderable: the graph advances
/// only on the ack carrying the seq it is waiting for, so a cancelled sequence's late ack is inert
/// rather than a phase-skip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    pub seq: u64,
    pub control: Control,
}

impl Envelope {
    pub fn encode(&self) -> Vec<u8> {
        // Infallible for these shapes (no maps with non-string keys, no unsupported types), and a
        // control message that could not be encoded has nowhere to be reported to anyway.
        rmp_serde::to_vec(self).unwrap_or_default()
    }
    pub fn decode(bytes: &[u8]) -> Result<Envelope, String> {
        rmp_serde::from_slice(bytes).map_err(|e| format!("control decode: {e}"))
    }
}

/// A param is a literal or an expression. Sending `Literal` on a bound param unbinds it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Literal(Param),
    Expr {
        /// Graph-rewritten source: every `nd(..)` / `globals.*` term replaced by a variable the
        /// evaluator receives as a local, so the node never resolves a name (§5.3).
        source: String,
        vars: Vec<Var>,
        /// Whether an arrival on this binding also wakes `process()`. Inert on a `common.*` key,
        /// where re-pacing is never a reason to run (§1.1).
        trigger: bool,
        /// The evaluator's handle for [`Self::Expr::source`], compiled by the GRAPH. The graph
        /// compiles because `set_expression` has to answer the authoring RPC with a real compile
        /// error; the node evaluates because §2.1 puts the evaluation immediately before the run
        /// that reads it. `None` when there is no evaluator, or when the source did not compile —
        /// in either case the literal stands and the binding carries the error.
        id: Option<goofi_node::BindingId>,
    },
}

/// One variable of a rewritten expression, resolved by the graph (§5.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Var {
    /// An `nd()` reference — the node subscribes and keeps a latest-wins mailbox.
    Stream { name: VarName, service: ServiceName, event_id: EventId },
    /// A `globals.*` reference — resolved by the graph and delivered inline. A globals edit
    /// re-sends `SetParam`, which lands in the same mailbox.
    Value { name: VarName, value: Param },
    /// The graph could not resolve it: `nd('ghost')`, a deleted target, a removed global. The node
    /// constructs the binding error from this and the param falls back to its literal.
    Missing { name: VarName, reason: String },
}

impl Var {
    pub fn name(&self) -> &str {
        match self {
            Var::Stream { name, .. } | Var::Value { name, .. } | Var::Missing { name, .. } => name,
        }
    }
}

/// Where a node is in its own lifecycle. Two variants rather than the projection's four: `creating`
/// is the GRAPH's — a node it has built and not yet heard from — and `error` is derived from the
/// fault, so neither is a node's to claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStage {
    Setup,
    Ready,
}

impl NodeStage {
    /// The projection the editor draws, which is a string because `runtime_overlay` has always been
    /// one (§6: what changes is how the graph LEARNS a stage, not how it shows it).
    pub fn as_str(self) -> &'static str {
        match self {
            NodeStage::Setup => "setup",
            NodeStage::Ready => "ready",
        }
    }
}

/// What a node tells the graph about itself. Every variant is a TRANSITION — the report is the
/// change, so the status-drain worker needs no diffing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Status {
    /// The answer to one [`Envelope`], keyed by its `seq`. Acks are how the graph knows a node is
    /// in sync with it, and they are what orders the three phases of a wire change (§4).
    Ack { seq: u64, ok: Result<(), String> },
    /// This node's own end of its services exists and it is listening. §4's birth barrier: the
    /// graph addresses nothing before this arrives, because pub/sub has no history and a `Control`
    /// published to a node that has not subscribed is simply lost.
    Ready,
    Fault { fault: Option<NodeFault> },
    /// Where the node is in its own lifecycle — `setup` while it is initializing, `ready` once it
    /// has. The graph's `error` stage is DERIVED from a fault and is never reported here, so the
    /// two cannot disagree about which one wins.
    Stage { stage: NodeStage },
    /// The node's measured update rate (`meta["ufreq"]`), which is a MEASUREMENT rather than a
    /// transition — so unlike every other variant it is paced, at [`super::UFREQ_REPORT_MS`].
    Ufreq { hz: f64 },
    /// The answer to [`Control::RefreshParam`]: the freshly enumerated options, or `None` when the
    /// node implements no hook for that param.
    RefreshOptions { key: ParamKey, options: Option<Vec<String>> },
    /// Per-binding errors, `None` where one cleared. A map on the node, a delta on the wire: each
    /// renders on its own inspector field.
    BindingErrors { errors: Vec<(ParamKey, Option<String>)> },
    /// The evaluated values of the node's bound params — the sparse projection, never the full
    /// param record (§2).
    ParamValues { evaluated: Vec<(ParamKey, Param)> },
}

impl Status {
    /// Infallible for these shapes, as [`Envelope::encode`] is and for the same reason.
    pub fn encode(&self) -> Vec<u8> {
        rmp_serde::to_vec(self).unwrap_or_default()
    }
    pub fn decode(bytes: &[u8]) -> Result<Status, String> {
        rmp_serde::from_slice(bytes).map_err(|e| format!("status decode: {e}"))
    }
}

/// The node's end of its own services.
pub trait Transport: Send + Sync {
    /// Park until something rings, and answer with every [`EventId`] that did. A notification is a
    /// HINT (§3.3) — the truth is in the mailboxes — so a caller drains regardless of what it gets.
    fn wait(&self, timeout: Option<Duration>) -> Vec<EventId>;
    /// Take every pending control message.
    fn drain_control(&self) -> Vec<Envelope>;
    /// Subscribe this input slot to exactly `services` — the full desired set, in wire order. What
    /// is absent is dropped; what is already subscribed keeps whatever it holds.
    fn wire_in(&self, slot: &str, services: &[ServiceName]) -> Result<(), String>;
    /// Ring exactly `targets` after each emit on this output slot — again the full desired set.
    fn wire_out(&self, slot: &str, targets: &[(ServiceName, EventId)]) -> Result<(), String>;
    /// Take every frame waiting on every wire, as `(slot, wire index, frame)`. The wire index is
    /// the producer's position in the last `wire_in` set for that slot, which is what a `multi`
    /// slot's per-wire cells are keyed by.
    ///
    /// On the trait rather than on the transport alone because this is how a frame REACHES a node:
    /// the wake loop drains here, and nothing else delivers one.
    fn drain_inputs(&self) -> Vec<(String, usize, Data)>;
    /// Emit a frame on an output slot, to every consumer of that slot at once.
    fn publish(&self, slot: &str, frame: &Data);
    /// Report a transition to the graph.
    fn report(&self, status: Status);
}

/// The graph's end of ONE node's control channel: it hands over an [`Envelope`] and rings that
/// node's door. A trait because the wire planner is about ordering, not about iceoryx2 — the
/// sequence it drives is the same whether the far end is a thread or a subprocess.
pub trait ControlSink: Send + Sync {
    fn send(&self, envelope: Envelope);
}
