//! The node's wire contract (spec §3.4): the messages a node exchanges with the graph, and the
//! transport seam that carries them.
//!
//! The seam is deliberately payload-shaped rather than iceoryx2-shaped: the node's scheduling is
//! decided from what a message SAYS, never from how it arrived, so the runtime can be driven by an
//! in-memory transport in a test and by shared memory in production without a second code path.
//! [`MemoryTransport`] is the test one; [`super::IoxTransport`] is the shipped one.
//!
//! The message sets are the subset this runtime can honestly act on. `Control::{RefreshParam,
//! Terminate}` and `Status::{Ready, Ufreq, Stage, RefreshOptions}` are still absent: each needs a
//! node lifecycle the graph does not own yet (a birth barrier to answer `Ready`, a manager-side
//! thread to terminate), and a variant nothing sends is a wire contract nothing honours.
//!
//! Both message sets are **msgpack** on the wire, over iceoryx2 rather than a Rust channel, so the
//! thread and subprocess cases are the same code — a param edit has no latency requirement, but a
//! second transport implementation is a correctness surface.

#[cfg(test)]
use std::sync::Mutex;
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

/// What a node tells the graph about itself. Every variant is a TRANSITION — the report is the
/// change, so the status-drain worker needs no diffing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Status {
    /// The answer to one [`Envelope`], keyed by its `seq`. Acks are how the graph knows a node is
    /// in sync with it, and they are what orders the three phases of a wire change (§4).
    Ack { seq: u64, ok: Result<(), String> },
    Fault { fault: Option<NodeFault> },
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
    /// Emit a frame on an output slot, to every consumer of that slot at once.
    fn publish(&self, slot: &str, frame: &Data);
    /// Report a transition to the graph.
    fn report(&self, status: Status);
}

/// The graph's end of ONE node's control channel: it hands over an [`Envelope`] and rings that
/// node's door. A trait because the wire planner is about ordering, not about iceoryx2 — the
/// sequence it drives is the same whether the far end is a thread, a subprocess, or a test double
/// that only writes down what it was sent.
pub trait ControlSink: Send + Sync {
    fn send(&self, envelope: Envelope);
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
    control: Vec<Envelope>,
    published: Vec<(String, Data)>,
    reported: Vec<Status>,
    wired_in: Vec<(String, Vec<ServiceName>)>,
    wired_out: Vec<(String, Vec<(ServiceName, EventId)>)>,
    next_seq: u64,
}

#[cfg(test)]
impl MemoryTransport {
    /// Queue a control message for the node — the graph side of [`Transport::drain_control`]. The
    /// seq is minted here because in production the graph mints it: a caller that does not care
    /// about ordering should not have to invent one, and one that does reads it back off the ack.
    pub fn send(&self, control: Control) {
        let mut inner = self.inner.lock().unwrap();
        inner.next_seq += 1;
        let seq = inner.next_seq;
        inner.control.push(Envelope { seq, control });
    }
    /// Every frame the node has published, in emission order.
    pub fn published(&self) -> Vec<(String, Data)> {
        self.inner.lock().unwrap().published.clone()
    }
    /// Every transition the node has reported, in order.
    pub fn reported(&self) -> Vec<Status> {
        self.inner.lock().unwrap().reported.clone()
    }
    /// Every input-slot set the node has applied, in order.
    pub fn wired_in(&self) -> Vec<(String, Vec<ServiceName>)> {
        self.inner.lock().unwrap().wired_in.clone()
    }
    /// Every output-slot target set the node has applied, in order.
    pub fn wired_out(&self) -> Vec<(String, Vec<(ServiceName, EventId)>)> {
        self.inner.lock().unwrap().wired_out.clone()
    }
}

#[cfg(test)]
impl Transport for MemoryTransport {
    /// Nothing rings an in-memory doorbell: a node on this transport is driven directly by its
    /// caller, so there is never a wake reason to report and never anything to park on.
    fn wait(&self, _timeout: Option<Duration>) -> Vec<EventId> {
        Vec::new()
    }
    fn drain_control(&self) -> Vec<Envelope> {
        std::mem::take(&mut self.inner.lock().unwrap().control)
    }
    /// Recorded rather than honoured: there is no shared memory here to subscribe to. What the node
    /// does with a slot message — dispatch it here and ack the result — is what these pin.
    fn wire_in(&self, slot: &str, services: &[ServiceName]) -> Result<(), String> {
        self.inner.lock().unwrap().wired_in.push((slot.to_string(), services.to_vec()));
        Ok(())
    }
    fn wire_out(&self, slot: &str, targets: &[(ServiceName, EventId)]) -> Result<(), String> {
        self.inner.lock().unwrap().wired_out.push((slot.to_string(), targets.to_vec()));
        Ok(())
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

    #[test]
    fn every_message_survives_the_wire_it_travels_on() {
        // Both message sets cross shared memory as msgpack, so anything a variant carries that the
        // codec cannot express is a message the far end silently never gets. Each one is checked
        // FULLY loaded — a variant with its collections empty round-trips even when the shape it
        // wraps does not.
        let messages = [
            Control::InSlot {
                slot: "in".to_string(),
                services: vec!["goofi_a_out_x".to_string(), "goofi_b_out_y".to_string()],
            },
            Control::OutSlot {
                slot: "out".to_string(),
                targets: vec![("goofi_c_door".to_string(), 1), ("goofi_d_door".to_string(), 65)],
            },
            Control::SetParam {
                key: ParamKey::new("osc", "freq"),
                value: ParamValue::Expr {
                    source: "__v0 * 2".to_string(),
                    vars: vec![
                        Var::Stream {
                            name: "__v0".to_string(),
                            service: "goofi_a_out_x".to_string(),
                            event_id: 65,
                        },
                        Var::Value { name: "__v1".to_string(), value: Param::float(30.0, 0.0, 100.0) },
                        Var::Missing { name: "__v2".to_string(), reason: "no node named `ghost`".to_string() },
                    ],
                    trigger: true,
                },
            },
        ];
        for control in messages {
            let sent = Envelope { seq: 7, control };
            assert_eq!(Envelope::decode(&sent.encode()), Ok(sent.clone()), "{sent:?}");
        }

        let statuses = [
            Status::Ack { seq: 7, ok: Ok(()) },
            Status::Ack { seq: 8, ok: Err("no such slot".to_string()) },
            Status::Fault { fault: Some(NodeFault::Process { msg: "boom".to_string(), since: 1.5 }) },
            Status::BindingErrors { errors: vec![(ParamKey::new("osc", "freq"), None)] },
            Status::ParamValues { evaluated: vec![(ParamKey::new("osc", "amp"), Param::boolean(true))] },
        ];
        for status in statuses {
            assert_eq!(Status::decode(&status.encode()), Ok(status.clone()), "{status:?}");
        }

        // And a payload that is not one of ours is refused rather than half-read: both decoders are
        // the far end of a shared-memory service anyone could have written to, and the transport
        // drops what it cannot read because there is no seq to answer with.
        assert!(Envelope::decode(b"not msgpack at all").is_err());
        assert!(Status::decode(&[0xc1]).is_err());
    }
}
