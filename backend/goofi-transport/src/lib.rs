//! Cross-engine transport: iceoryx2 names, rendezvous and endpoint machinery, one shared
//! mechanism for every engine. A phone book, not a switchboard — the resolver here is pure name
//! and config derivation, and whichever side settles first waits on `open_or_create`.

use std::sync::{Once, OnceLock};

use iceoryx2::config::Config;
use iceoryx2::node::{NodeState, NodeView};
use iceoryx2::prelude::*;

use goofi_node::Uid;
pub use goofi_node::EventId;

/// An iceoryx2 service name — a wire's identity, which is why slot messages carry no source uid.
pub type ServiceName = String;

/// The service variant every goofi port uses. `ipc_threadsafe` (rather than `ipc`) is what makes
/// the ports `Send + Sync`, which an engine's transport must be.
type Svc = ipc_threadsafe::Service;
/// The iceoryx2 node every port of one owner is built from. It must outlive them, and it is what
/// `max_nodes` counts on each service — so owners share one rather than minting one per port.
pub type IoxNode = iceoryx2::node::Node<Svc>;
pub type BytePublisher = iceoryx2::port::publisher::Publisher<Svc, [u8], ()>;
pub type ByteSubscriber = iceoryx2::port::subscriber::Subscriber<Svc, [u8], ()>;
pub type ByteService = iceoryx2::service::port_factory::publish_subscribe::PortFactory<Svc, [u8], ()>;
pub type EventService = iceoryx2::service::port_factory::event::PortFactory<Svc>;
pub type Listener = iceoryx2::port::listener::Listener<Svc>;

/// `EventId(0)` is a control message; `1..=64` an input slot; `65..=128` an `nd()` channel (§3.2).
/// 255 is the ceiling those three ranges are budgeted against.
const EVENT_ID_MAX: usize = 255;
/// Every producer feeding this node needs a notifier, plus the graph. The default 16 busts on a
/// 20-wire multi-input.
const MAX_NOTIFIERS: usize = 256;
/// Fan-out plus the `/data` reducer. The default 8 busts on a 9-consumer slot.
const MAX_SUBSCRIBERS: usize = 256;
/// How many iceoryx2 NODES may open one service. One graph node is one iceoryx2 node, so this is
/// really a per-peer bound, and it binds below both ceilings above.
const MAX_NODES: usize = 256;
/// How many messages a control or status subscriber may hold unread. Unlike the one-deep data
/// services, control and status are message STREAMS: an ack never read parks a wire sequence.
const MESSAGE_BUFFER: usize = 1024;
/// How many readers a control or status service admits — exactly one, by construction. Not
/// cosmetic: iceoryx2 sizes a segment as readers × buffer × slice, so these three numbers MULTIPLY.
const MESSAGE_READERS: usize = 1;
/// The pool a message publisher starts with; `PowerOfTwo` grows the segment for a rare large one.
pub const MESSAGE_SLICE: usize = 1024;
/// The pool a data publisher starts with; `PowerOfTwo` grows it for a larger frame.
pub const INITIAL_SLICE: usize = 64 * 1024;

/// A fresh service-name scope for one graph. Random rather than a pid, which is reused — and every
/// builder here is `open_or_create`, so a recycled pid would silently JOIN a stale service.
pub fn service_instance() -> String {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).expect("the OS random source");
    format!("{:016x}", u64::from_be_bytes(bytes))
}

/// The name every service of one node is derived from: `<instance>_<uid>_<gen>`. `gen` is bumped on
/// EVERY birth, because teardown never blocks and a rebirth would else race its predecessor.
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

/// A notifier onto one node's door; the ringer knows nothing else about the node it rings.
pub struct Doorbell {
    notifier: iceoryx2::port::notifier::Notifier<Svc>,
}

impl Doorbell {
    /// Open a door by name, on the ringer's OWN iceoryx2 node — one bell per producing NODE, since
    /// each node counts against `max_nodes`. `open_or_create`: the service is the rendezvous.
    pub fn open(node: &IoxNode, service: &str) -> Result<Doorbell, String> {
        let door = event_service(node, service)?;
        let notifier = door.notifier_builder().create().map_err(|e| format!("notifier `{service}`: {e}"))?;
        Ok(Doorbell { notifier })
    }

    /// Ring it. A failed ring costs a wake, never a message: the payload is already in a queue the
    /// node drains.
    pub fn ring(&self, id: EventId) -> Result<(), String> {
        self.notifier
            .notify_with_custom_event_id(iceoryx2::prelude::EventId::new(id as usize))
            .map(|_| ())
            .map_err(|e| format!("notify: {e}"))
    }
}

/// One iceoryx2 node per port OWNER, never per port: each is a permanent `/tmp/iceoryx2/nodes/<id>`
/// directory and each is counted by every service's `max_nodes`.
pub fn iox_node() -> Result<IoxNode, String> {
    sweep_once();
    NodeBuilder::new().config(iox_config()).create::<Svc>().map_err(|e| format!("iox node: {e}"))
}

/// The iceoryx2 configuration every goofi port is built against: the global defaults with the two
/// AUTOMATIC dead-node cleanup passes off, since [`reclaim_stale_resources`] does that job once.
fn iox_config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let mut config = Config::global_config().clone();
        config.global.service.cleanup_dead_nodes_on_open = false;
        config.global.node.cleanup_dead_nodes_on_creation = false;
        // The third one fires in `Node::drop`, so leaving it on lets every teardown quietly do the
        // sweep's job — which made the sweep's own test pass with the sweep gutted.
        config.global.node.cleanup_dead_nodes_on_destruction = false;
        config
    })
}

/// Ask iceoryx2 to reclaim the shared memory a CRASHED run left behind. A graceful exit needs none
/// of it, but a killed process drops nothing and its segments stay allocated.
///
/// It ASKS, and that is the whole of it. goofi used to delete `<root>/nodes/<id>` itself when the
/// ask failed — reaching into another library's private layout to finish its bookkeeping. That is
/// not goofi's to own, and on Windows it never worked anyway: the files carry a protected DACL
/// with no DELETE, so both the ask and the reach fail alike (upstream #1869).
pub fn reclaim_stale_resources() {
    let mut refused: Vec<u128> = Vec::new();
    let _ = IoxNode::list(iox_config(), |state| {
        if let NodeState::Dead(view) = state {
            // The id BEFORE the attempt: the reclaim consumes the view.
            let id = view.id().value();
            if view.try_remove_stale_resources().is_err() {
                refused.push(id);
            }
        }
        CallbackProgression::Continue
    });
    for id in refused {
        force_remove_refused(&id.to_string());
    }
}

/// WORKAROUND, and it is one: eclipse-iceoryx/iceoryx2#1869. On Windows a node's bookkeeping files
/// carry a PROTECTED DACL granting the owner no DELETE, so iceoryx2's own reclaim cannot take them
/// and the directory strands for good — measured at 1,561 entries a week old, and it is what turns
/// a later run's service open into `ServiceInCorruptedState`. The owner keeps implicit WRITE_DAC,
/// so granting first is what makes the removal possible at all.
///
/// Only a node iceoryx2 has ITSELF declared dead AND then refused reaches here. Never a live peer,
/// and never a blanket pass over the directory — that is what deleted a directory out from under a
/// live enumeration and took the process down with iceoryx2's own assert.
#[cfg(windows)]
fn force_remove_refused(id: &str) {
    use std::process::{Command, Stdio};
    let Some(dir) = nodes_dir().map(|d| format!("{d}/{id}")) else { return };
    if !std::path::Path::new(&dir).exists() {
        return;
    }
    // `*S-1-3-4` is OWNER RIGHTS: it grants the object's own owner, which is this user, and nobody
    // else. Shelling out for the same reason `proc::taskkill` does — `/T` has no one-call API.
    let _ = Command::new("icacls")
        .args([dir.as_str(), "/grant", "*S-1-3-4:(OI)(CI)F", "/T", "/C", "/Q"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Everywhere else iceoryx2's own reclaim is the whole of it, and a refusal is its business.
#[cfg(not(windows))]
fn force_remove_refused(_id: &str) {}

/// Make the root iceoryx2 is configured for. It fills the layout in — `nodes/`, `services/` —
/// but does not create the top directory, and on Windows nothing else does: every node then fails
/// with `NodeCreationFailure::InternalError` on a machine whose temp has been cleaned. Making a
/// directory the library goes on to own is not the same as reaching into one.
fn ensure_root() {
    if let Ok(root) = String::from_utf8(iox_config().global.root_path().as_bytes().to_vec()) {
        let _ = std::fs::create_dir_all(root);
    }
}

/// Where iceoryx2 keeps one directory per node. Read-only: goofi looks, and no longer reaches in
/// to delete — that layout is iceoryx2's to keep.
pub fn nodes_dir() -> Option<String> {
    let root = String::from_utf8(iox_config().global.root_path().as_bytes().to_vec()).ok()?;
    let nodes = String::from_utf8(iox_config().global.node.directory.as_bytes().to_vec()).ok()?;
    Some(format!("{}/{}", root.trim_end_matches('/'), nodes.trim_end_matches('/')))
}

/// Everything that happens once per PROCESS, before its first port exists. [`iox_node`] calls it
/// for correctness; a boot path calls it so the bill does not land on the user's first add.
pub fn sweep_once() {
    static SWEPT: Once = Once::new();
    SWEPT.call_once(|| {
        // iceoryx2 logs at Info by default and reads `IOX2_LOG_LEVEL` only when asked. What Info
        // prints here is a multi-kilobyte dump per dropped doorbell datagram — §3.3's non-event.
        set_log_level_from_env_or(LogLevel::Error);
        ensure_root();
        reclaim_stale_resources();
    });
}

/// The event service every door is: §3.2's three id ranges against one ceiling, and one listener.
pub fn event_service(node: &IoxNode, name: &str) -> Result<EventService, String> {
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
pub fn message_service(node: &IoxNode, name: &str) -> Result<ByteService, String> {
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

/// The publish/subscribe service every data wire is: one publisher, because a slot has exactly one
/// producer; no history, because a link never replays; a one-deep buffer, which is latest-wins.
pub fn data_service(node: &IoxNode, name: &str) -> Result<ByteService, String> {
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

/// Open a subscriber on an output slot's data service by name — a `/data` consumer's end of a wire.
pub fn open_output_subscriber(node: &IoxNode, service: &str) -> Result<ByteSubscriber, String> {
    data_service(node, service)?
        .subscriber_builder()
        .create()
        .map_err(|e| format!("subscriber `{service}`: {e}"))
}

/// A publisher that can grow past its initial pool: a GOOF frame is variable-size, and `Static`
/// would refuse the first one larger than `initial` instead of reallocating.
pub fn publisher(service: &ByteService, what: &str, initial: usize) -> Result<BytePublisher, String> {
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
