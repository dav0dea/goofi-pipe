//! The iceoryx2 transport, against real shared memory (spec §3).
//!
//! These live in an integration target rather than in-module because they need the real thing: a
//! process that creates services, parks on a listener and gets woken by another port in the same
//! binary. Every test picks its own [`Uid`], because one integration target is one process and a
//! service name is global to the machine — two tests sharing a uid would open each other's ports.

use std::sync::Arc;
use std::time::Duration;

use goofi_core::{Data, Meta, Param, SlotType, Value};
use goofi_engine::runtime::{
    door_service, output_service, service_base, Control, ControlSink, Doorbell, Envelope,
    IoxTransport, NodeChannel, NodeRuntime, ParamValue, Status, Transport,
};
use goofi_engine::Uid;
use goofi_node::{
    default_factory, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamKey, Params, SlotDecl,
};

/// Long enough that a park is a real park, short enough that a broken retention fails fast.
const MS200: Duration = Duration::from_millis(200);

/// A node with one input and one output — enough to give a transport a slot to publish on and a
/// slot to receive on. Nothing here runs it; the manifest is what the transport reads.
#[derive(Default)]
struct Passthrough;
impl Node for Passthrough {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
}
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "in",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: false,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
static MANIFEST: NodeManifest = NodeManifest {
    type_name: "_TransportTest",
    category: "test",
    doc: "transport fixture",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: &[],
    isolation: Isolation::InProcess,
    producer: false,
    factory: default_factory::<Passthrough>,
};

fn manifest() -> &'static NodeManifest {
    &MANIFEST
}

fn frame(values: &[f32]) -> Data {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Data::array_f32(vec![values.len()], bytes, Meta::empty()).unwrap()
}

fn values(frame: &Data) -> Vec<f32> {
    match frame.value() {
        Value::Array(store) => {
            store.as_bytes().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
        }
        other => panic!("expected an array frame, got {other:?}"),
    }
}

/// The service names of a node born at `uid` — what the graph puts in the other end's slot message.
fn base_of(uid: Uid) -> String {
    service_base("test", uid, 0)
}

#[test]
fn the_services_are_created_with_limits_the_defaults_do_not_give_us() {
    // iceoryx2 fixes these at CREATION, before any wire exists, so they are hard patch limits.
    // The defaults are all wrong for this design: max_notifiers 16 (a 20-wire multi-input busts
    // it), max_subscribers 8 (a 9-consumer slot busts it), and AllocationStrategy::Static
    // refuses any frame larger than initial_max_slice_len.
    let t = IoxTransport::create("test", Uid(1), 0, manifest()).expect("services");
    let cfg = t.event_config();
    assert_eq!(cfg.event_id_max_value(), 255);
    assert_eq!(cfg.max_notifiers(), 256);
    assert_eq!(cfg.max_listeners(), 1);
    let d = t.data_config("out").expect("the declared output slot has a service");
    assert_eq!(d.history_size(), 0, "a link NEVER replays a previous output");
    assert_eq!(d.max_subscribers(), 256);
    assert_eq!(d.max_publishers(), 1);
    assert_eq!(d.subscriber_max_buffer_size(), 1, "latest-wins, per wire");
    assert!(d.has_safe_overflow(), "drop-oldest, matching the no-queue delivery model");
}

/// The doorbell of a node born at `uid`, opened the way a producer or the graph opens one.
fn bell_for(uid: Uid) -> Doorbell {
    Doorbell::open(&door_service(&base_of(uid))).expect("doorbell")
}

#[test]
fn a_notify_landing_mid_drain_is_not_lost() {
    // The notification is only a HINT; the truth is in the subscriber queues and the control
    // mailbox. This is what lets a node drain everything and never park with work pending.
    let t = IoxTransport::create("test", Uid(2), 0, manifest()).unwrap();
    let bell = bell_for(Uid(2));
    bell.ring(1).unwrap();
    assert_eq!(t.wait(Some(MS200)), vec![1]);
    bell.ring(2).unwrap(); // lands while "draining"
    assert_eq!(t.wait(Some(MS200)), vec![2], "retained across the re-park");
}

#[test]
fn a_control_and_a_data_notification_both_survive() {
    let t = IoxTransport::create("test", Uid(3), 0, manifest()).unwrap();
    let bell = bell_for(Uid(3));
    bell.ring(0).unwrap();
    bell.ring(3).unwrap();
    let mut got = t.wait(Some(MS200));
    got.sort();
    assert_eq!(got, vec![0, 3]);
}

#[test]
fn a_control_message_crosses_shared_memory_and_comes_back_acked() {
    // The whole control plane in one pass: the graph publishes msgpack and rings id 0, the node
    // wakes on that, applies the message, and answers with the seq it was sent. The ack is the only
    // thing that orders a wire change, so a message that arrives without one is a stalled sequence.
    let transport = Arc::new(IoxTransport::create("test", Uid(4), 0, manifest()).unwrap());
    let mut node = NodeRuntime::new(manifest(), transport.clone());
    let channel = NodeChannel::open(&base_of(Uid(4))).unwrap();

    assert_eq!(node.next_wake(), None, "parked: nothing has asked this node to run");
    channel.send(Envelope {
        seq: 41,
        control: Control::SetParam {
            key: ParamKey::new("common", "autotrigger"),
            value: ParamValue::Literal(Param::boolean(true)),
        },
    });
    assert_eq!(transport.wait(Some(MS200)), vec![0], "the graph rang the control id");

    node.run_once();
    assert!(node.next_wake().is_some(), "the node applied what it was sent and re-paced");
    assert_eq!(channel.drain_status(), vec![Status::Ack { seq: 41, ok: Ok(()) }]);

    // And the ack carries the VERDICT, not a receipt: a slot this manifest does not declare is the
    // one refusal a node can state, and the graph abandons that sequence rather than waiting on it.
    channel.send(Envelope {
        seq: 42,
        control: Control::OutSlot { slot: "nope".to_string(), targets: Vec::new() },
    });
    node.run_once();
    assert_eq!(
        channel.drain_status(),
        vec![Status::Ack { seq: 42, ok: Err("no output slot `nope`".to_string()) }]
    );
}

#[test]
fn a_frame_reaches_a_wired_consumer_and_rings_its_slot() {
    // A wire is two declarations and nothing else: the producer is told a doorbell, the consumer is
    // told a service name. Neither knows the other's uid.
    let producer = IoxTransport::create("test", Uid(5), 0, manifest()).unwrap();
    let consumer = IoxTransport::create("test", Uid(6), 0, manifest()).unwrap();
    consumer.wire_in("in", &[output_service(&base_of(Uid(5)), "out")]).unwrap();
    producer.wire_out("out", &[(door_service(&base_of(Uid(6))), 1)]).unwrap();

    producer.publish("out", &frame(&[1.0, 2.0, 3.0]));
    assert_eq!(consumer.wait(Some(MS200)), vec![1], "woken by the slot's own event id");
    let got = consumer.drain_inputs();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].0.as_str(), got[0].1), ("in", 0), "slot, and its position in the wire order");
    assert_eq!(values(&got[0].2), vec![1.0, 2.0, 3.0]);
    assert!(consumer.drain_inputs().is_empty(), "a drained wire is empty");
}

#[test]
fn a_frame_larger_than_the_initial_slice_still_lands() {
    // `AllocationStrategy::Static` — the iceoryx2 default — refuses this outright, and a GOOF frame
    // is variable-size by construction: one HD video frame or one long buffer busts any fixed pool.
    let producer = IoxTransport::create("test", Uid(7), 0, manifest()).unwrap();
    let consumer = IoxTransport::create("test", Uid(8), 0, manifest()).unwrap();
    consumer.wire_in("in", &[output_service(&base_of(Uid(7)), "out")]).unwrap();

    let big: Vec<f32> = (0..80_000).map(|i| i as f32).collect(); // 320 KB, past the 64 KiB start
    producer.publish("out", &frame(&big));
    let got = consumer.drain_inputs();
    assert_eq!(got.len(), 1, "the oversized frame was published and received");
    assert_eq!(values(&got[0].2), big);
}

#[test]
fn a_wire_the_new_set_omits_stops_delivering() {
    // The slot set is DECLARATIVE: what a message does not name is dropped. This is the same
    // mechanism that displaces a single input's previous wire, so it has no special case anywhere.
    let producer = IoxTransport::create("test", Uid(9), 0, manifest()).unwrap();
    let consumer = IoxTransport::create("test", Uid(10), 0, manifest()).unwrap();
    let service = output_service(&base_of(Uid(9)), "out");
    consumer.wire_in("in", &[service]).unwrap();
    producer.publish("out", &frame(&[1.0]));
    assert_eq!(consumer.drain_inputs().len(), 1);

    consumer.wire_in("in", &[]).unwrap();
    producer.publish("out", &frame(&[2.0]));
    assert!(consumer.drain_inputs().is_empty(), "the dropped wire delivers nothing");
}
