//! The [`Transport`] over real iceoryx2, and the graph's end of the same wires.
//!
//! Every service a node owns is created at its birth from one base name, and every limit is set
//! EXPLICITLY: iceoryx2 fixes them at creation, before a single wire exists, so a default that is
//! too small is a hard patch limit no later call can raise.
//!
//! A service NAME is a wire's identity. A producer holds a [`Doorbell`] per target and knows
//! nothing else about it; a consumer opens a producer's output service by name. That is why the
//! slot messages carry names rather than uids, and why the same code serves a thread and a
//! subprocess.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, Once};
use std::time::Duration;

use iceoryx2::config::Config;
use iceoryx2::node::NodeState;
use iceoryx2::prelude::*;
use iceoryx2_bb_posix::file_descriptor::FileDescriptorBased;

use goofi_core::Data;
use goofi_node::NodeManifest;

use super::wire::{ControlSink, Envelope, EventId, ServiceName, Status, Transport};
use crate::Uid;

/// The service variant every goofi port uses. `ipc_threadsafe` (rather than `ipc`) is what makes
/// the ports `Send + Sync`, which a `Transport` must be — the node thread publishes while the
/// status-drain worker reads.
type Svc = ipc_threadsafe::Service;
/// The iceoryx2 node every port of one owner is built from. It must outlive them, and it is what
/// `max_nodes` counts on each service — which is why owners share one rather than minting one per
/// port (see [`iox_node`]).
pub type IoxNode = iceoryx2::node::Node<Svc>;
type BytePublisher = iceoryx2::port::publisher::Publisher<Svc, [u8], ()>;
/// Public because [`crate::testing::OutputProbe`] holds one: a probe is a `/data` viewer and reads a
/// producer through the same port kind the reducer does.
pub type ByteSubscriber = iceoryx2::port::subscriber::Subscriber<Svc, [u8], ()>;
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
/// How many iceoryx2 NODES may open one service. §3.5 reads it as a per-process bound, which would
/// make the default 20 irrelevant here — but one graph node is one iceoryx2 node, so it is really a
/// per-peer bound, and it binds below both ceilings above: measured, the 20th consumer of a slot was
/// refused with `ExceedsMaxNumberOfNodes` while `max_subscribers` said 256. Raised to match.
const MAX_NODES: usize = 256;
/// How many messages a control or status subscriber may hold unread. The DATA services are one
/// deep on purpose — latest-wins is what a wire means — but control and status are message STREAMS:
/// an ack the graph never reads parks a wire sequence forever, and a fault the graph never reads is
/// a node that draws healthy while it is broken. Deep enough that a burst (a load's acks, a
/// restart's stage transitions) survives however long the drain takes to come round.
const MESSAGE_BUFFER: usize = 1024;
/// How many readers a control or status service admits — exactly one, by construction: the node
/// reads its own control and the graph reads that node's status, and nothing else has a reason to.
///
/// Not a cosmetic limit. iceoryx2 sizes a publisher's segment so every subscriber can hold its
/// whole buffer at once, so these three numbers MULTIPLY: 256 readers of a 1024-deep 64 KiB-sliced
/// stream is a 17 GB shared-memory file, and a node owns two of them. Measured before it was cut:
/// 7.5 s and 130 MB of resident shared memory per `add_node`.
const MESSAGE_READERS: usize = 1;
/// How long an empty `<root>/nodes/<id>` directory is left alone before [`remove_empty_node_dirs`]
/// takes it. iceoryx2 creates the directory and then writes into it, so a just-created one belongs
/// to a node another process is starting; a minute is far past that and far short of a session.
const NODE_DIR_GRACE: Duration = Duration::from_secs(60);
/// The pool a message publisher starts with. A control or status message is tens to hundreds of
/// bytes, and `PowerOfTwo` grows the segment for the rare large one (a `RefreshParam` answer naming
/// many devices) rather than reserving 64 KiB per slot against it.
const MESSAGE_SLICE: usize = 1024;
/// The pool a publisher starts with. `PowerOfTwo` grows it for a larger frame; the initial size
/// only decides how many reallocations a big-frame patch pays before it settles.
const INITIAL_SLICE: usize = 64 * 1024;

/// A fresh service-name scope for one graph. Random rather than the pid the first version used: a
/// pid is reused, every builder here is `open_or_create`, and a machine accumulates service records
/// — so a recycled pid would silently JOIN a stale service, taking whatever limits it was created
/// with, instead of failing. Carries no underscore, so a name splits back into its parts.
pub fn service_instance() -> String {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).expect("the OS random source");
    format!("{:016x}", u64::from_be_bytes(bytes))
}

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

/// Graph → node control messages. Private: both ends of this one are in this module.
fn control_service(base: &str) -> ServiceName {
    format!("goofi_{base}_ctl")
}

/// Node → graph status transitions. Private for the same reason as [`control_service`].
fn status_service(base: &str) -> ServiceName {
    format!("goofi_{base}_sts")
}

/// One output slot's data service — the name a consumer is given in its `InSlot` set.
pub fn output_service(base: &str, slot: &str) -> ServiceName {
    format!("goofi_{base}_out_{slot}")
}

/// A notifier onto one node's door. A producer holds one per target of an output slot and the graph
/// holds one per node it messages; neither knows anything else about the node it rings.
pub struct Doorbell {
    notifier: iceoryx2::port::notifier::Notifier<Svc>,
}

impl Doorbell {
    /// Open an existing door by name, on the ringer's OWN iceoryx2 node. Borrowed rather than minted
    /// per bell for two measured reasons: each node is a permanent `/tmp/iceoryx2/nodes/<id>`
    /// directory and a multi-kilobyte stderr dump when it drops, and each one counts against every
    /// service's `max_nodes`. A door is rung by one bell per producing NODE, not per wire.
    ///
    /// `open_or_create` rather than `open` because the graph may ring a node whose own listener is
    /// still being built — the service is the rendezvous, not a proof of liveness.
    pub fn open(node: &IoxNode, service: &str) -> Result<Doorbell, String> {
        let door = event_service(node, service)?;
        let notifier = door.notifier_builder().create().map_err(|e| format!("notifier `{service}`: {e}"))?;
        Ok(Doorbell { notifier })
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

/// One output slot: its publisher, and the doorbells to ring once a frame is out.
struct OutputPort {
    service: ByteService,
    publisher: BytePublisher,
    /// The full desired set the graph last sent (§3.4), already opened. ONE record rather than
    /// names beside bells: `wire_out` opens them, `publish` rings them, and nothing is reconciled on
    /// the per-frame path.
    targets: Mutex<Vec<(Doorbell, EventId)>>,
}

/// One wire feeding an input slot: the producer's service name (the wire's identity) and this end
/// of it. The order of these within a slot IS `Inputs::get_multi`'s connection order (§3.5).
struct InputWire {
    service: ServiceName,
    subscriber: ByteSubscriber,
}

/// A node's end of every service it owns.
pub struct IoxTransport {
    door: EventService,
    listener: iceoryx2::port::listener::Listener<Svc>,
    control: ByteSubscriber,
    status: BytePublisher,
    /// Fixed at birth from the manifest: an output slot cannot appear later.
    outputs: HashMap<&'static str, OutputPort>,
    /// Grown and shrunk by `InSlot`, which is why it is the one map behind a lock.
    inputs: Mutex<Vec<(String, Vec<InputWire>)>>,
    /// Must outlive every port built from it, this node's doorbells included — so it is declared
    /// LAST, because Rust drops a struct's fields in declaration order. Declared first, the node
    /// cleaned up while its own ports still held records inside its directory: it could not remove
    /// the directory, the ports then emptied it, and the empty directory stayed for ever. One
    /// `goofi-engine` suite run left 977 of them, and `IoxNode::list` walks every one at each
    /// process start — which took `goofi-python`'s subprocess tests from 8.8 s to 529 s.
    node: IoxNode,
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
        let node = iox_node()?;

        let door = event_service(&node, &door_service(&base))?;
        let listener = door.listener_builder().create().map_err(|e| format!("listener: {e}"))?;

        let control = message_service(&node, &control_service(&base))?
            .subscriber_builder()
            .create()
            .map_err(|e| format!("control subscriber: {e}"))?;
        let status = publisher(&message_service(&node, &status_service(&base))?, "status", MESSAGE_SLICE)?;

        let mut outputs = HashMap::new();
        for out in manifest.outputs {
            let service = data_service(&node, &output_service(&base, out.name))?;
            let publisher = publisher(&service, out.name, INITIAL_SLICE)?;
            outputs.insert(
                out.name,
                OutputPort { service, publisher, targets: Mutex::new(Vec::new()) },
            );
        }

        Ok(IoxTransport {
            node,
            door,
            listener,
            control,
            status,
            outputs,
            inputs: Mutex::new(Vec::new()),
        })
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
}

impl Transport for IoxTransport {
    fn wait(&self, timeout: Option<Duration>) -> Vec<EventId> {
        let mut ids = Vec::new();
        // `wait_all` rather than `wait_one`: several producers can ring between two parks, and
        // taking one id per wake would leave the rest to be re-delivered a wake later. A listener
        // that errors reports no reason to wake, and the caller drains regardless — which is
        // exactly what makes a lost notification cost nothing.
        // A BOUNDED wait parks on the listener's own descriptor rather than calling its
        // `timed_wait_all`. That one arms the timeout with `SO_RCVTIMEO`, which the kernel rounds
        // to a jiffie — about 1.3 ms added to every park, which is most of a 500 Hz node's period
        // and paced one at 300 Hz.
        //
        // `ppoll` and not `select`: `select` was tried and is a defect. Its `fd_set` is a 1024-bit
        // array, so a descriptor at or past `FD_SETSIZE` indexes off the end of it — a NON-
        // unwinding panic inside libc's `FD_ISSET`, which aborts the whole process rather than
        // failing one park. A goofi process holds several descriptors per node and reaches that
        // ceiling on a large patch; the `editing` suite reached it in thirteen tests. `ppoll`
        // takes the descriptor by value and has no such ceiling, and its `timespec` does not round.
        let _ = match timeout {
            Some(t) => {
                let mut pfd = libc::pollfd {
                    fd: unsafe { self.listener.file_descriptor().native_handle() },
                    events: libc::POLLIN,
                    revents: 0,
                };
                let ts = libc::timespec {
                    tv_sec: t.as_secs() as libc::time_t,
                    tv_nsec: t.subsec_nanos() as i64,
                };
                // An interrupted or failed park is a park that ended early: the caller drains its
                // mailboxes regardless, which is what makes a lost notification cost nothing.
                unsafe { libc::ppoll(&mut pfd, 1, &ts, std::ptr::null()) };
                self.listener.try_wait_all(|id| ids.push(id.as_value() as EventId))
            }
            None => self.listener.blocking_wait_all(|id| ids.push(id.as_value() as EventId)),
        };
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
        // Opened here, where a failure can still be reported: the ack carries it and the graph
        // abandons that sequence. The previous set stands until the whole new one opens, so a
        // refused message leaves the node in the state its ack describes.
        let mut opened = Vec::with_capacity(targets.len());
        for (service, id) in targets {
            opened.push((Doorbell::open(&self.node, service)?, *id));
        }
        *port.targets.lock().unwrap() = opened;
        Ok(())
    }

    /// Draining to empty (rather than one frame per wake) is the wake discipline of §3.3: the
    /// notification is a hint, so a node that drained everything can park knowing there is nothing
    /// left. A wire that produced several frames since the last drain keeps only its newest, which
    /// is what latest-wins means and what `subscriber_max_buffer_size(1)` already enforces.
    fn drain_inputs(&self) -> Vec<(String, usize, Data)> {
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

    fn publish(&self, slot: &str, frame: &Data) {
        let Some(port) = self.outputs.get(slot) else { return };
        let bytes = goofi_codec::encode(frame);
        // Nothing to report a loan failure to: it is a shared-memory condition rather than a node
        // error, and the next emit re-tries it. What must not happen is ringing anyway.
        let Ok(sample) = port.publisher.loan_slice_uninit(bytes.len()) else { return };
        let _ = sample.write_from_slice(&bytes).send();
        // The ring comes AFTER the send, always: a consumer woken before the frame is in its queue
        // drains nothing and parks again, and the frame then waits for an unrelated wake.
        for (bell, id) in port.targets.lock().unwrap().iter() {
            let _ = bell.ring(*id);
        }
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
    control: BytePublisher,
    status: ByteSubscriber,
    door: Doorbell,
    /// Last, for the reason [`IoxTransport::node`] states: the node has to drop AFTER every port
    /// built from it, or it leaves its own directory behind. The graph holds one of these for each
    /// node, so this half leaked at the same rate as the node's own half.
    _node: IoxNode,
}

impl NodeChannel {
    pub fn open(base: &str) -> Result<NodeChannel, String> {
        let node = iox_node()?;
        let control = publisher(&message_service(&node, &control_service(base))?, "control", MESSAGE_SLICE)?;
        let status = message_service(&node, &status_service(base))?
            .subscriber_builder()
            .create()
            .map_err(|e| format!("status subscriber: {e}"))?;
        let door = Doorbell::open(&node, &door_service(base))?;
        Ok(NodeChannel { _node: node, control, status, door })
    }

    /// Ring the node's door with no message behind it — how a parked node is made to look at
    /// something that is not on its control channel, which is exactly the [`super::Halt`] flag it
    /// was born holding. A wake is a hint (§3.3), so a failed ring costs at most one park.
    pub fn wake(&self) {
        let _ = self.door.ring(CONTROL_EVENT_ID);
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

/// One iceoryx2 node per port OWNER — a node's transport, or the graph's channel to one node —
/// never per port. Each one is a permanent `/tmp/iceoryx2/nodes/<id>` directory plus a multi-kilobyte
/// "unable to remove node resources" dump on drop, and each one is counted by every service's
/// `max_nodes`, so minting one per wire would both leak per re-plan and lower the real fan-out
/// ceiling below the configured one.
pub fn iox_node() -> Result<IoxNode, String> {
    sweep_once();
    NodeBuilder::new().config(iox_config()).create::<Svc>().map_err(|e| format!("iox node: {e}"))
}

/// The iceoryx2 configuration every goofi port is built against: the global defaults with the two
/// AUTOMATIC dead-node cleanup passes turned off.
///
/// They reclaim what a crashed run left, and they do it by rescanning every stale
/// `/tmp/iceoryx2/nodes/<id>` on every service open and every service creation — work that only
/// ever needs doing once per process. Measured on a machine that had accumulated 853 of them: a
/// multi-kilobyte `[W] SharedNodeState` dump per hit, one rescan per port, and 105 such blocks in a
/// short test run. [`reclaim_stale_resources`] does the same job once, at startup, where it can
/// also be reasoned about — which is the only reason turning these off is safe.
pub(crate) fn iox_config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let mut config = Config::global_config().clone();
        config.global.service.cleanup_dead_nodes_on_open = false;
        config.global.node.cleanup_dead_nodes_on_creation = false;
        // The third one is the one that hides the other two: it fires in `Node::drop`, so with it
        // left on every node teardown in the process quietly did the sweep's job — which made the
        // sweep's own test pass with the sweep gutted, as long as any other test ran beside it.
        config.global.node.cleanup_dead_nodes_on_destruction = false;
        config
    })
}

/// Remove the shared memory a CRASHED run left behind. A graceful exit needs none of it — a node
/// removes its own resources when it drops — but a killed process drops nothing, and its segments
/// stay allocated against RAM until something takes them.
pub fn reclaim_stale_resources() {
    let _ = IoxNode::list(iox_config(), |state| {
        if let NodeState::Dead(view) = state {
            let _ = view.try_remove_stale_resources();
        }
        CallbackProgression::Continue
    });
    remove_empty_node_dirs();
}

/// The inode half. iceoryx2 0.9.3 removes a dropped node's FILES and leaves its directory, so
/// `<root>/nodes/` grows by one empty entry per node for ever — 12 666 measured on one development
/// machine — and `IoxNode::list` walks every one of them at each process start, which makes startup
/// slower the longer the machine has been used.
///
/// Only EMPTY directories, and only ones nothing has touched for [`NODE_DIR_GRACE`]: iceoryx2
/// creates the directory and then writes its files, so a fresh empty one may belong to a node
/// another process is starting right now. Every error is ignored — this is housekeeping, and a
/// directory that cannot be removed simply stays.
fn remove_empty_node_dirs() {
    let Ok(root) = String::from_utf8(iox_config().global.root_path().as_bytes().to_vec()) else {
        return;
    };
    let Ok(nodes) = String::from_utf8(iox_config().global.node.directory.as_bytes().to_vec()) else {
        return;
    };
    let dir = format!("{}/{}", root.trim_end_matches('/'), nodes);
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for entry in entries.flatten() {
        let stale = entry
            .metadata()
            .ok()
            .filter(|m| m.is_dir())
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > NODE_DIR_GRACE);
        if stale {
            let _ = std::fs::remove_dir(entry.path());
        }
    }
}

/// Everything that happens once per PROCESS, before its first port exists. A `Once` rather than a
/// call in `main`, because a test binary has no `main` of ours — and once per process is the point
/// for the sweep: it is exactly what replaced iceoryx2's own once-per-service-open rescan.
fn sweep_once() {
    static SWEPT: Once = Once::new();
    SWEPT.call_once(|| {
        // iceoryx2 logs at Info by default and reads `IOX2_LOG_LEVEL` only when something asks it
        // to — so setting that variable alone changes nothing, which is worth knowing before
        // trying it. What Info prints here is not diagnostic: a multi-kilobyte `Notifier { .. }`
        // dump per `FailedToDeliverSignal`, which is a doorbell datagram dropped because the
        // consumer has not drained its socket yet. §3.3 makes that a NON-event — a ring is a hint,
        // and a lost one costs a wake rather than a message — but an uncapped producer causes it
        // at its own rate. Measured on one 306-test run: 41 753 blocks, 108 MB of stderr.
        set_log_level_from_env_or(LogLevel::Error);
        reclaim_stale_resources();
    });
}

/// The event service every door is: the three id ranges of §3.2 budgeted against one ceiling, and
/// one listener — the node itself.
fn event_service(node: &IoxNode, name: &str) -> Result<EventService, String> {
    node.service_builder(&parse_name(name)?)
        .event()
        .max_nodes(MAX_NODES)
        .event_id_max_value(EVENT_ID_MAX)
        .max_notifiers(MAX_NOTIFIERS)
        .max_listeners(1)
        .open_or_create()
        .map_err(|e| format!("event service `{name}`: {e}"))
}

/// The control and status services: the same publish/subscribe shape as a data wire, but a message
/// STREAM rather than a latest-wins cell — see [`MESSAGE_BUFFER`].
fn message_service(node: &IoxNode, name: &str) -> Result<ByteService, String> {
    node.service_builder(&parse_name(name)?)
        .publish_subscribe::<[u8]>()
        .max_nodes(MAX_NODES)
        .enable_safe_overflow(true)
        .history_size(0)
        .subscriber_max_buffer_size(MESSAGE_BUFFER)
        .max_publishers(1)
        .max_subscribers(MESSAGE_READERS)
        .open_or_create()
        .map_err(|e| format!("message service `{name}`: {e}"))
}

/// The publish/subscribe service every data wire is. One publisher because a
/// slot has exactly one producer; no history because a link never replays a previous output; a
/// one-deep buffer because that is what latest-wins resolves to per wire.
fn data_service(node: &IoxNode, name: &str) -> Result<ByteService, String> {
    node.service_builder(&parse_name(name)?)
        .publish_subscribe::<[u8]>()
        .max_nodes(MAX_NODES)
        .enable_safe_overflow(true)
        .history_size(0)
        .subscriber_max_buffer_size(1)
        .max_publishers(1)
        .max_subscribers(MAX_SUBSCRIBERS)
        .open_or_create()
        .map_err(|e| format!("data service `{name}`: {e}"))
}

/// Open a subscriber on an output slot's data service by name — a `/data` consumer's whole end of a
/// wire, and the one thing a viewer needs that is not already public.
pub fn open_output_subscriber(node: &IoxNode, service: &str) -> Result<ByteSubscriber, String> {
    data_service(node, service)?
        .subscriber_builder()
        .create()
        .map_err(|e| format!("subscriber `{service}`: {e}"))
}

/// A publisher that can grow past its initial pool: a GOOF frame is variable-size, and `Static`
/// would refuse the first one larger than `initial` instead of reallocating. The initial size is
/// the caller's because it is half of what a service's segment costs — see [`MESSAGE_READERS`].
fn publisher(service: &ByteService, what: &str, initial: usize) -> Result<BytePublisher, String> {
    service
        .publisher_builder()
        .initial_max_slice_len(initial)
        .allocation_strategy(AllocationStrategy::PowerOfTwo)
        .create()
        .map_err(|e| format!("publisher `{what}`: {e}"))
}

fn parse_name(name: &str) -> Result<iceoryx2::service::service_name::ServiceName, String> {
    name.try_into().map_err(|e| format!("bad service name `{name}`: {e:?}"))
}

