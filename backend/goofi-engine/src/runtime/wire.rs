//! The node's wire contract (spec §3.4): the messages a node exchanges with the graph, and the
//! transport seam that carries them.
//!
//! The seam is deliberately payload-shaped rather than iceoryx2-shaped: the node's scheduling is
//! decided from what a message SAYS, never from how it arrived, so the runtime can be driven by an
//! in-memory transport in a test and by shared memory in production without a second code path.
//! Only [`MemoryTransport`] ships here — the iceoryx2 implementation is its own step.
//!
//! The message sets are the subset this runtime can honestly act on. `Control::{InSlot, OutSlot,
//! RefreshParam, Terminate}` and `Status::{Ready, Ufreq, Stage, RefreshOptions}` arrive with the
//! transport that gives a node subscribers to wire and a graph to answer.
//!
//! `Status::Ack` and the `seq` every `Control` carries are absent for the same reason and are
//! named here because they are the one omission that WIDENS a shipped type rather than adding a
//! variant beside it: an ack confirms an ordering the in-memory transport does not have, to a
//! graph that is not listening. `seq` belongs ON `Control`, so the transport that can honour it
//! is the one that should add it.

#[cfg(test)]
use std::sync::Mutex;
use std::time::Duration;

use goofi_core::{Data, Param};
use goofi_node::ParamKey;

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
#[derive(Clone, Debug, PartialEq)]
pub enum Control {
    /// Write a param: a literal, or the expression to bind it to. This is the NOTIFICATION path —
    /// a node parked with `next_wake() == None` is never rung by a bare param-record swap.
    SetParam { key: ParamKey, value: ParamValue },
}

/// A param is a literal or an expression. Sending `Literal` on a bound param unbinds it.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Literal(Param),
    Expr {
        /// Graph-rewritten source: every `nd(..)` / `globals.*` term replaced by a variable the
        /// evaluator receives as a local, so the node never resolves a name (§5.3).
        source: String,
        vars: Vec<Var>,
        /// Whether an arrival on this binding also wakes `process()`. Inert on a `common.*` key —
        /// see [`super::NodeRuntime::deliver_expr_arrival`].
        trigger: bool,
    },
}

/// One variable of a rewritten expression, resolved by the graph (§5.3).
#[derive(Clone, Debug, PartialEq)]
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

/// What a node tells the graph about itself. Every variant is a TRANSITION — the report is the
/// change, so the status-drain worker needs no diffing.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Fault { fault: Option<NodeFault> },
    /// Per-binding errors, `None` where one cleared. A map on the node, a delta on the wire: each
    /// renders on its own inspector field.
    BindingErrors { errors: Vec<(ParamKey, Option<String>)> },
    /// The evaluated values of the node's bound params — the sparse projection, never the full
    /// param record (§2).
    ParamValues { evaluated: Vec<(ParamKey, Param)> },
}

/// The seam the iceoryx2 implementation replaces.
pub trait Transport: Send + Sync {
    /// Park until something rings, and answer with every [`EventId`] that did. A notification is a
    /// HINT (§3.3) — the truth is in the mailboxes — so a caller drains regardless of what it gets.
    fn wait(&self, timeout: Option<Duration>) -> Vec<EventId>;
    /// Take every pending control message.
    fn drain_control(&self) -> Vec<Control>;
    /// Emit a frame on an output slot, to every consumer of that slot at once.
    fn publish(&self, slot: &str, frame: &Data);
    /// Report a transition to the graph.
    fn report(&self, status: Status);
}

/// The in-memory [`Transport`]: it never parks, and it records everything so a test can read what
/// the node emitted and told the graph. Test-only, because the transport that ships is the one
/// built on iceoryx2 — this exists to drive a node without one.
#[cfg(test)]
#[derive(Default)]
pub struct MemoryTransport {
    inner: Mutex<Inner>,
}

#[cfg(test)]
#[derive(Default)]
struct Inner {
    control: Vec<Control>,
    published: Vec<(String, Data)>,
    reported: Vec<Status>,
}

#[cfg(test)]
impl MemoryTransport {
    /// Queue a control message for the node — the graph side of [`Transport::drain_control`].
    pub fn send(&self, msg: Control) {
        self.inner.lock().unwrap().control.push(msg);
    }
    /// Every frame the node has published, in emission order.
    pub fn published(&self) -> Vec<(String, Data)> {
        self.inner.lock().unwrap().published.clone()
    }
    /// Every transition the node has reported, in order.
    pub fn reported(&self) -> Vec<Status> {
        self.inner.lock().unwrap().reported.clone()
    }
}

#[cfg(test)]
impl Transport for MemoryTransport {
    /// Nothing rings an in-memory doorbell: a node on this transport is driven directly by its
    /// caller, so there is never a wake reason to report and never anything to park on.
    fn wait(&self, _timeout: Option<Duration>) -> Vec<EventId> {
        Vec::new()
    }
    fn drain_control(&self) -> Vec<Control> {
        std::mem::take(&mut self.inner.lock().unwrap().control)
    }
    fn publish(&self, slot: &str, frame: &Data) {
        self.inner.lock().unwrap().published.push((slot.to_string(), frame.clone()));
    }
    fn report(&self, status: Status) {
        self.inner.lock().unwrap().reported.push(status);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::Meta;

    #[test]
    fn the_memory_transport_hands_each_queue_over_once() {
        // Draining is destructive, because a wake that re-delivered what it already delivered
        // would run the node again on a message it has consumed.
        let t = MemoryTransport::default();
        t.send(Control::SetParam {
            key: ParamKey::new("osc", "freq"),
            value: ParamValue::Literal(Param::float(1.0, 0.0, 2.0)),
        });
        assert_eq!(t.drain_control().len(), 1);
        assert!(t.drain_control().is_empty());

        t.publish("out", &Data::string("x", Meta::empty()));
        t.report(Status::Fault { fault: None });
        assert_eq!(t.published().len(), 1);
        assert_eq!(t.reported(), vec![Status::Fault { fault: None }]);
        assert_eq!(t.published().len(), 1, "reading what was emitted does not consume it");
    }
}
