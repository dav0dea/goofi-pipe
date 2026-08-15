//! The [`Transport`] over real iceoryx2 (spec §3.1, §3.2, §3.5), and the graph's end of the same
//! wires.
//!
//! Every service a node owns is created at its birth from one base name, `<instance>_<uid>_<gen>`,
//! and every limit is set explicitly: iceoryx2 fixes them at CREATION, before a single wire exists,
//! so a default that is too small is not a slow path but a hard patch limit no later call can raise.
//!
//! The addressing model is that a service NAME is a wire's identity. A producer holds a
//! [`Doorbell`] per target and knows nothing else about it; a consumer opens a producer's output
//! service by name. That is why the slot messages carry names rather than uids, and why the same
//! code serves a thread and a subprocess.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use iceoryx2::prelude::*;

use goofi_core::Data;
use goofi_node::NodeManifest;

use super::wire::{ControlSink, Envelope, EventId, ServiceName, Status, Transport};
use crate::Uid;

/// The service variant every goofi port uses. `ipc_threadsafe` (rather than `ipc`) is what makes
/// the ports `Send + Sync`, which a `Transport` must be — the node thread publishes while the
/// status-drain worker reads.
type Svc = ipc_threadsafe::Service;
type BytePublisher = iceoryx2::port::publisher::Publisher<Svc, [u8], ()>;
type ByteSubscriber = iceoryx2::port::subscriber::Subscriber<Svc, [u8], ()>;
type ByteService = iceoryx2::service::port_factory::publish_subscribe::PortFactory<Svc, [u8], ()>;
type EventService = iceoryx2::service::port_factory::event::PortFactory<Svc>;

/// `EventId(0)` is a control message; `1..=64` an input slot; `65..=128` an `nd()` channel (§3.2).
/// 255 is the ceiling those three ranges are budgeted against.
const EVENT_ID_MAX: usize = 255;
/// The one id the graph itself rings with — every `Control`, whatever it says.
const CONTROL_EVENT_ID: EventId = 0;
/// Every producer feeding this node needs a notifier, plus the graph. The default 16 busts on a
/// 20-wire multi-input.
const MAX_NOTIFIERS: usize = 256;
/// Fan-out plus the `/data` reducer. The default 8 busts on a 9-consumer slot.
const MAX_SUBSCRIBERS: usize = 256;
/// The pool a publisher starts with. `PowerOfTwo` grows it for a larger frame; the initial size
/// only decides how many reallocations a big-frame patch pays before it settles.
const INITIAL_SLICE: usize = 64 * 1024;

/// The name every service of one node is derived from: `<instance>_<uid>_<gen>`. `gen` is bumped on
/// EVERY birth at a uid — restart, undo-of-delete and load alike — because teardown never blocks,
/// so without it a rebirth would race its predecessor's service names.
pub fn service_base(instance: &str, uid: Uid, gen: u64) -> String {
    format!("{instance}_{}_{gen}", uid.to_hex())
}

/// The one event service a node parks on for its whole life (§3.2).
pub fn door_service(base: &str) -> ServiceName {
    format!("goofi_{base}_door")
}

/// Graph → node control messages.
pub fn control_service(base: &str) -> ServiceName {
    format!("goofi_{base}_ctl")
}

/// Node → graph status transitions.
pub fn status_service(base: &str) -> ServiceName {
    format!("goofi_{base}_sts")
}

/// One output slot's data service — the name a consumer is given in its `InSlot` set.
pub fn output_service(base: &str, slot: &str) -> ServiceName {
    format!("goofi_{base}_out_{slot}")
}

/// A notifier onto one node's door. A producer holds one per target of an output slot and the graph
/// holds one per node it messages; neither knows anything else about the node it rings.
pub struct Doorbell {
    _node: iceoryx2::node::Node<Svc>,
    notifier: iceoryx2::port::notifier::Notifier<Svc>,
}

impl Doorbell {
    /// Open an existing door by name. `open_or_create` rather than `open` because the graph may ring
    /// a node whose own listener is still being built — the service is the rendezvous, not a proof
    /// of liveness.
    pub fn open(service: &str) -> Result<Doorbell, String> {
        let node = new_node()?;
        let door = event_service(&node, service)?;
        let notifier = door.notifier_builder().create().map_err(|e| format!("notifier `{service}`: {e}"))?;
        Ok(Doorbell { _node: node, notifier })
    }

    /// Ring it. `EventId` says WHY the node woke, and the node treats it as a hint (§3.3) — so a
    /// failed ring costs a wake, never a message: the payload is already in a queue the node drains.
    pub fn ring(&self, id: EventId) -> Result<(), String> {
        self.notifier
            .notify_with_custom_event_id(iceoryx2::prelude::EventId::new(id as usize))
            .map(|_| ())
            .map_err(|e| format!("notify: {e}"))
    }
}

/// One output slot: its publisher, and the doorbells to ring once a frame is out. The target list is
/// the full desired set the graph last sent (§3.4), so applying one is an assignment, not a diff.
struct OutputPort {
    service: ByteService,
    publisher: BytePublisher,
    targets: Mutex<Vec<(ServiceName, EventId)>>,
    /// One doorbell per target service, kept across re-wirings — opening one costs a service
    /// lookup, and a target that survives a re-plan should not pay it again.
    bells: Mutex<HashMap<ServiceName, Doorbell>>,
}

/// One wire feeding an input slot: the producer's service name (the wire's identity) and this end
/// of it. The order of these within a slot IS `Inputs::get_multi`'s connection order (§3.5).
struct InputWire {
    service: ServiceName,
    subscriber: ByteSubscriber,
}

/// A node's end of every service it owns.
pub struct IoxTransport {
    base: String,
    /// The iceoryx2 node must outlive every port built from it.
    node: iceoryx2::node::Node<Svc>,
    door: EventService,
    listener: iceoryx2::port::listener::Listener<Svc>,
    control: ByteSubscriber,
    status: BytePublisher,
    /// Fixed at birth from the manifest: an output slot cannot appear later.
    outputs: HashMap<&'static str, OutputPort>,
    /// Grown and shrunk by `InSlot`, which is why it is the one map behind a lock.
    inputs: Mutex<Vec<(String, Vec<InputWire>)>>,
}

impl IoxTransport {
    /// Create every service this node owns and open its own end of each. The order matters at the
    /// far end, not here: the control SUBSCRIBER exists before the graph is told the node is ready,
    /// which is what closes the attach window pub/sub leaves open (§4).
    pub fn create(
        instance: &str,
        uid: Uid,
        gen: u64,
        manifest: &NodeManifest,
    ) -> Result<IoxTransport, String> {
        let base = service_base(instance, uid, gen);
        let node = new_node()?;

        let door = event_service(&node, &door_service(&base))?;
        let listener = door.listener_builder().create().map_err(|e| format!("listener: {e}"))?;

        let control = data_service(&node, &control_service(&base))?
            .subscriber_builder()
            .create()
            .map_err(|e| format!("control subscriber: {e}"))?;
        let status = publisher(&data_service(&node, &status_service(&base))?, "status")?;

        let mut outputs = HashMap::new();
        for out in manifest.outputs {
            let service = data_service(&node, &output_service(&base, out.name))?;
            let publisher = publisher(&service, out.name)?;
            outputs.insert(
                out.name,
                OutputPort {
                    service,
                    publisher,
                    targets: Mutex::new(Vec::new()),
                    bells: Mutex::new(HashMap::new()),
                },
            );
        }

        Ok(IoxTransport { base, node, door, listener, control, status, outputs, inputs: Mutex::new(Vec::new()) })
    }

    /// The base every one of this node's service names is built from — what the graph puts in
    /// another node's slot message.
    pub fn base(&self) -> &str {
        &self.base
    }

    /// The doorbell service's creation-time limits, read back from iceoryx2 itself.
    pub fn event_config(&self) -> iceoryx2::service::static_config::event::StaticConfig {
        *self.door.static_config()
    }

    /// A declared output slot's creation-time limits, read back from iceoryx2 itself. `None` for a
    /// slot this node does not declare.
    pub fn data_config(
        &self,
        slot: &str,
    ) -> Option<iceoryx2::service::static_config::publish_subscribe::StaticConfig> {
        self.outputs.get(slot).map(|o| *o.service.static_config())
    }

    /// Take every frame waiting on every wire, as `(slot, wire index, frame)`. The wire index is the
    /// producer's position in the last `InSlot` set, which is what a `multi` slot's per-wire cells
    /// are keyed by — a single slot has only wire 0.
    ///
    /// Draining to empty (rather than one frame per wake) is the wake discipline of §3.3: the
    /// notification is a hint, so a node that drained everything can park knowing there is nothing
    /// left. A wire that produced several frames since the last drain keeps only its newest, which
    /// is what latest-wins means and what `subscriber_max_buffer_size(1)` already enforces.
    pub fn drain_inputs(&self) -> Vec<(String, usize, Data)> {
        let mut out = Vec::new();
        for (slot, wires) in self.inputs.lock().unwrap().iter() {
            for (index, wire) in wires.iter().enumerate() {
                let mut newest = None;
                while let Ok(Some(sample)) = wire.subscriber.receive() {
                    newest = Some(goofi_codec::decode(sample.payload()));
                }
                // A frame that cannot be decoded is a wire whose two ends disagree about the format
                // — dropping it keeps the node running on its other inputs, and the producer is the
                // only end that could report it anyway.
                if let Some(Ok(frame)) = newest {
                    out.push((slot.clone(), index, frame));
                }
            }
        }
        out
    }

    /// Open this end of one wire: the producer's output service, by the name that IS the wire's
    /// identity. `open_or_create` because a consumer may be wired before its producer has published
    /// anything — the service is the rendezvous.
    fn open_wire(&self, service: &ServiceName) -> Result<InputWire, String> {
        let subscriber = data_service(&self.node, service)?
            .subscriber_builder()
            .create()
            .map_err(|e| format!("subscriber `{service}`: {e}"))?;
        Ok(InputWire { service: service.clone(), subscriber })
    }

    /// The doorbells for one output slot, opened once and kept: `wire_out` is declarative, so a
    /// target that survives a re-plan must not pay another service lookup.
    fn ring_targets(&self, port: &OutputPort) {
        let targets = port.targets.lock().unwrap().clone();
        let mut bells = port.bells.lock().unwrap();
        bells.retain(|service, _| targets.iter().any(|(t, _)| t == service));
        for (service, id) in targets {
            let bell = match bells.entry(service.clone()) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => match Doorbell::open(&service) {
                    Ok(bell) => e.insert(bell),
                    // A door that cannot be opened is a node that has died or has not been born.
                    // The frame is already published, so the consumer gets it on its next wake.
                    Err(_) => continue,
                },
            };
            let _ = bell.ring(id);
        }
    }
}

impl Transport for IoxTransport {
    fn wait(&self, timeout: Option<Duration>) -> Vec<EventId> {
        let mut ids = Vec::new();
        // `wait_all` rather than `wait_one`: several producers can ring between two parks, and
        // taking one id per wake would leave the rest to be re-delivered a wake later.
        let waited = match timeout {
            Some(t) => self.listener.timed_wait_all(|id| ids.push(id.as_value() as EventId), t),
            None => self.listener.blocking_wait_all(|id| ids.push(id.as_value() as EventId)),
        };
        // A listener that errors reports no reason to wake; the caller drains regardless, which is
        // exactly what makes a lost notification cost nothing.
        let _ = waited;
        ids
    }

    fn drain_control(&self) -> Vec<Envelope> {
        let mut out = Vec::new();
        while let Ok(Some(sample)) = self.control.receive() {
            match Envelope::decode(sample.payload()) {
                Ok(envelope) => out.push(envelope),
                // Undecodable means the graph and this node disagree about the message format —
                // there is no seq to ack against, so it can only be dropped.
                Err(_) => continue,
            }
        }
        out
    }

    fn wire_in(&self, slot: &str, services: &[ServiceName]) -> Result<(), String> {
        let mut inputs = self.inputs.lock().unwrap();
        let mut held: Vec<Option<InputWire>> = match inputs.iter().position(|(name, _)| name == slot) {
            Some(at) => inputs.remove(at).1.into_iter().map(Some).collect(),
            None => Vec::new(),
        };
        let mut wires = Vec::with_capacity(services.len());
        let mut failed = Vec::new();
        for service in services {
            // A surviving wire keeps its subscriber: re-opening it would drop whatever the producer
            // has already sent and this node has not read yet. What is left in `held` at the end is
            // what the new set does not name, and dropping it IS the unsubscribe.
            let kept = held
                .iter_mut()
                .find(|w| w.as_ref().is_some_and(|w| &w.service == service))
                .and_then(Option::take);
            match kept {
                Some(wire) => wires.push(wire),
                None => match self.open_wire(service) {
                    Ok(wire) => wires.push(wire),
                    Err(e) => failed.push(e),
                },
            }
        }
        inputs.push((slot.to_string(), wires));
        if failed.is_empty() {
            Ok(())
        } else {
            Err(failed.join("; "))
        }
    }

    fn wire_out(&self, slot: &str, targets: &[(ServiceName, EventId)]) -> Result<(), String> {
        let port = self.outputs.get(slot).ok_or_else(|| format!("no output slot `{slot}`"))?;
        *port.targets.lock().unwrap() = targets.to_vec();
        Ok(())
    }

    fn publish(&self, slot: &str, frame: &Data) {
        let Some(port) = self.outputs.get(slot) else { return };
        let bytes = goofi_codec::encode(frame);
        match port.publisher.loan_slice_uninit(bytes.len()) {
            Ok(sample) => {
                let _ = sample.write_from_slice(&bytes).send();
            }
            // Nothing to report this to: a publish failure is a shared-memory condition, not a node
            // error, and the next emit re-tries it.
            Err(_) => return,
        }
        // The ring comes AFTER the send, always: a consumer woken before the frame is in its queue
        // drains nothing and parks again, and the frame then waits for an unrelated wake.
        self.ring_targets(port);
    }

    fn report(&self, status: Status) {
        let bytes = status.encode();
        if let Ok(sample) = self.status.loan_slice_uninit(bytes.len()) {
            let _ = sample.write_from_slice(&bytes).send();
        }
    }
}

/// The graph's end of one node's control channel: the control publisher, the status subscriber, and
/// that node's doorbell. Opened by name from the same base the node built its services from, which
/// is the whole of what the graph needs to know about where a node lives.
pub struct NodeChannel {
    _node: iceoryx2::node::Node<Svc>,
    control: BytePublisher,
    status: ByteSubscriber,
    door: Doorbell,
}

impl NodeChannel {
    pub fn open(base: &str) -> Result<NodeChannel, String> {
        let node = new_node()?;
        let control = publisher(&data_service(&node, &control_service(base))?, "control")?;
        let status = data_service(&node, &status_service(base))?
            .subscriber_builder()
            .create()
            .map_err(|e| format!("status subscriber: {e}"))?;
        let door = Doorbell::open(&door_service(base))?;
        Ok(NodeChannel { _node: node, control, status, door })
    }

    /// Every transition the node has reported since the last drain, in order. This is what the
    /// status-drain worker reads: acks that advance a wire sequence, and the node's own state.
    pub fn drain_status(&self) -> Vec<Status> {
        let mut out = Vec::new();
        while let Ok(Some(sample)) = self.status.receive() {
            if let Ok(status) = Status::decode(sample.payload()) {
                out.push(status);
            }
        }
        out
    }
}

impl ControlSink for NodeChannel {
    /// Publish, then ring `EventId(0)`. In that order for the same reason a frame is published
    /// before its doorbell: a node woken first would drain an empty mailbox and park again.
    fn send(&self, envelope: Envelope) {
        let bytes = envelope.encode();
        if let Ok(sample) = self.control.loan_slice_uninit(bytes.len()) {
            let _ = sample.write_from_slice(&bytes).send();
        }
        let _ = self.door.ring(CONTROL_EVENT_ID);
    }
}

/// One iceoryx2 node per port owner. They are cheap, and a shared one would tie every port's
/// lifetime to the longest-lived owner.
fn new_node() -> Result<iceoryx2::node::Node<Svc>, String> {
    NodeBuilder::new().create::<Svc>().map_err(|e| format!("iox node: {e}"))
}

/// The event service every door is: the three id ranges of §3.2 budgeted against one ceiling, and
/// one listener — the node itself.
fn event_service(node: &iceoryx2::node::Node<Svc>, name: &str) -> Result<EventService, String> {
    node.service_builder(&parse_name(name)?)
        .event()
        .event_id_max_value(EVENT_ID_MAX)
        .max_notifiers(MAX_NOTIFIERS)
        .max_listeners(1)
        .open_or_create()
        .map_err(|e| format!("event service `{name}`: {e}"))
}

/// The publish/subscribe service every data, control and status wire is. One publisher because a
/// slot has exactly one producer; no history because a link never replays a previous output; a
/// one-deep buffer because that is what latest-wins resolves to per wire.
fn data_service(node: &iceoryx2::node::Node<Svc>, name: &str) -> Result<ByteService, String> {
    node.service_builder(&parse_name(name)?)
        .publish_subscribe::<[u8]>()
        .enable_safe_overflow(true)
        .history_size(0)
        .subscriber_max_buffer_size(1)
        .max_publishers(1)
        .max_subscribers(MAX_SUBSCRIBERS)
        .open_or_create()
        .map_err(|e| format!("data service `{name}`: {e}"))
}

/// A publisher that can grow past its initial pool: a GOOF frame is variable-size, and `Static`
/// would refuse the first one larger than `INITIAL_SLICE` instead of reallocating.
fn publisher(service: &ByteService, what: &str) -> Result<BytePublisher, String> {
    service
        .publisher_builder()
        .initial_max_slice_len(INITIAL_SLICE)
        .allocation_strategy(AllocationStrategy::PowerOfTwo)
        .create()
        .map_err(|e| format!("publisher `{what}`: {e}"))
}

fn parse_name(name: &str) -> Result<iceoryx2::service::service_name::ServiceName, String> {
    name.try_into().map_err(|e| format!("bad service name `{name}`: {e:?}"))
}
