//! The node's wire contract (spec §3.4): the messages a node exchanges with the graph, and the
//! transport seam that carries them.

use std::time::Duration;

use goofi_core::{Data, Param};
use goofi_node::ParamKey;
use serde::{Deserialize, Serialize};

pub use goofi_node::EventId;

pub use goofi_transport::ServiceName;

/// The generated name of an expression variable (`__v0`), minted by the graph's rewrite (§5.3).
pub type VarName = String;

/// The messages the graph sends a node. The slot messages are DECLARATIVE: each carries the
/// complete desired set for that slot and the node diffs it, so wiring is idempotent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Control {
    /// Every producer service feeding this input slot, in wire order — a position survives a
    /// producer restart, a service name does not.
    InSlot { slot: String, services: Vec<ServiceName> },
    /// Every doorbell to ring after publishing on this output slot, with the [`EventId`] that
    /// says WHY the far node woke.
    OutSlot { slot: String, targets: Vec<(ServiceName, EventId)> },
    /// Write a param: a literal, or the expression to bind it to. The NOTIFICATION path — a bare
    /// param-record swap never rings a parked node.
    SetParam { key: ParamKey, value: ParamValue },
    /// Re-enumerate a refreshable `Str` param's options. The answer comes back as
    /// [`Status::RefreshOptions`] rather than on this message's ack, since the hook runs on the node.
    RefreshParam { key: ParamKey },
}

/// A control message and the sequence number the node acks it with. `seq` is graph-minted and
/// monotonic, so a cancelled sequence's late ack is inert rather than a phase-skip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    pub seq: u64,
    pub control: Control,
}

impl Envelope {
    pub fn encode(&self) -> Vec<u8> {
        // Infallible for these shapes, and a control message that failed to encode has nowhere to
        // be reported to anyway.
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
        /// evaluator receives as a local.
        source: String,
        vars: Vec<Var>,
        /// Whether an arrival on this binding also wakes `process()`. Inert on a `common.*` key.
        trigger: bool,
        /// The evaluator's handle for the source, compiled by the GRAPH so `set_expression` can
        /// answer with a real compile error. `None` leaves the literal standing.
        id: Option<goofi_node::BindingId>,
    },
}

/// One variable of a rewritten expression, resolved by the graph (§5.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Var {
    /// An `nd()` reference — the node subscribes and keeps a latest-wins mailbox.
    Stream { name: VarName, service: ServiceName, event_id: EventId },
    /// A `globals.*` reference — resolved by the graph and delivered inline.
    Value { name: VarName, value: Param },
    /// The graph could not resolve it; the node builds the binding error and falls back to the
    /// literal.
    Missing { name: VarName, reason: String },
}

impl Var {
    pub fn name(&self) -> &str {
        match self {
            Var::Stream { name, .. } | Var::Value { name, .. } | Var::Missing { name, .. } => name,
        }
    }
}

pub use goofi_node::{NodeStage, Status};

/// What the signal node's wire carries up: the async handshake plus the shared health vocabulary.
/// `Ack` and `Ready` are this engine's own and never cross the engine seam — the drain consumes
/// them and hands the graph only [`Status`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WireStatus {
    /// The answer to one [`Envelope`], keyed by its `seq` — what orders a wire change's phases.
    Ack { seq: u64, ok: Result<(), String> },
    /// This node's own end of its services exists and it is listening. §4's birth barrier: pub/sub
    /// has no history, so the graph addresses nothing before this arrives.
    Ready,
    Health(Status),
}

impl WireStatus {
    /// Infallible for these shapes, as [`Envelope::encode`] is and for the same reason.
    pub fn encode(&self) -> Vec<u8> {
        rmp_serde::to_vec(self).unwrap_or_default()
    }
    pub fn decode(bytes: &[u8]) -> Result<WireStatus, String> {
        rmp_serde::from_slice(bytes).map_err(|e| format!("status decode: {e}"))
    }
}

/// The node's end of its own services.
pub trait Transport: Send + Sync {
    /// Park until something rings, and answer with every [`EventId`] that did. A notification is a
    /// HINT — the truth is in the mailboxes — and a ZERO timeout takes what is already there.
    fn wait(&self, timeout: Option<Duration>) -> Vec<EventId>;
    /// Take every pending control message.
    fn drain_control(&self) -> Vec<Envelope>;
    /// Subscribe this input slot to exactly `services` — the full desired set, in wire order.
    fn wire_in(&self, slot: &str, services: &[ServiceName]) -> Result<(), String>;
    /// Ring exactly `targets` after each emit on this output slot — again the full desired set.
    fn wire_out(&self, slot: &str, targets: &[(ServiceName, EventId)]) -> Result<(), String>;
    /// Take every frame waiting on every wire, as `(slot, wire index, frame)`. The wire index is
    /// the producer's position in the last `wire_in` set for that slot.
    fn drain_inputs(&self) -> Vec<(String, usize, Data)>;
    /// Emit a frame on an output slot, to every consumer of that slot at once.
    fn publish(&self, slot: &str, frame: &Data);
    /// Report a transition to the graph.
    fn report(&self, status: WireStatus);
}

/// The graph's end of ONE node's control channel: it hands over an [`Envelope`] and rings that
/// node's door.
pub trait ControlSink: Send + Sync {
    fn send(&self, envelope: Envelope);
}
