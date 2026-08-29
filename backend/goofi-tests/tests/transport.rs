//! The iceoryx2 transport, against real shared memory (spec §3).
//!
//! Every test picks its own [`Uid`] and [`instance`] scopes the target by pid: a service name is
//! global to the MACHINE, and `open_or_create` means a colliding loser reads the winner's config.

use std::sync::Arc;
use std::time::Duration;

use goofi_core::{Param, SlotType};
use goofi_tests::{f32s, frame};
use goofi_graph::Uid;
use goofi_signal::runtime::{
    Control, ControlSink, Envelope, IoxTransport, NodeChannel, NodeEnv, NodeFault, NodeRuntime,
    ParamValue, Status, Transport, WireStatus,
};
use goofi_transport::{door_service, iox_node, output_service, service_base, Doorbell, IoxNode};
use goofi_node::{NodeManifest, OutputDecl, ParamKey, Params, SlotDecl};
use goofi_signal::{default_factory, Inputs, Node, NodeCtx, NodeResult, Outputs};

/// Long enough that a park is a real park, short enough that a broken retention fails fast.
const MS200: Duration = Duration::from_millis(200);

/// A node with one input and one output. Nothing runs it; the manifest is what the transport reads.
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
    producer: false,
};

fn manifest() -> &'static NodeManifest {
    &MANIFEST
}

/// This run's service-name scope.
fn instance() -> String {
    format!("t{:x}", std::process::id())
}

/// Status is asynchronous by design, so a test waits for it with a deadline rather than reading once.
fn status_within(channel: &NodeChannel, timeout: Duration) -> Vec<WireStatus> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let got = channel.drain_status();
        if !got.is_empty() || std::time::Instant::now() >= deadline {
            return got;
        }
        std::thread::yield_now();
    }
}

/// The service names of a node born at `uid` — what the graph puts in the other end's slot message.
fn base_of(uid: Uid) -> String {
    service_base(&instance(), uid, 0)
}

#[test]
fn the_services_are_created_with_limits_the_defaults_do_not_give_us() {
    // iceoryx2 fixes these at CREATION, so they are hard patch limits, and every default is wrong
    // for this design.
    let t = IoxTransport::create(&instance(), Uid(1), 0, manifest()).expect("services");
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

#[test]
fn an_undrained_control_mailbox_keeps_the_whole_burst() {
    // Control and status are message STREAMS, not the latest-wins CELL a data wire is. The count is
    // past any plausible drain interval: a node deep inside `process` is the burst this has to survive.
    const BURST: u64 = 200;
    let transport = IoxTransport::create(&instance(), Uid(30), 0, manifest()).unwrap();
    let channel = NodeChannel::open(&base_of(Uid(30))).unwrap();
    for seq in 1..=BURST {
        channel.send(Envelope {
            seq,
            control: Control::InSlot { slot: "in".to_string(), services: Vec::new() },
        });
    }

    let got = transport.drain_control();
    assert_eq!(
        got.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (1..=BURST).collect::<Vec<_>>(),
        "every message, in the order it was sent",
    );
}

/// The doorbell of a node born at `uid`, opened on the ringer's own iceoryx2 node.
fn bell_for(uid: Uid, ringer: &IoxNode) -> Doorbell {
    Doorbell::open(ringer, &door_service(&base_of(uid))).expect("doorbell")
}

#[test]
fn a_notify_landing_mid_drain_is_not_lost() {
    // The notification is only a HINT; the truth is in the subscriber queues and the control mailbox.
    let t = IoxTransport::create(&instance(), Uid(2), 0, manifest()).unwrap();
    let ringer = iox_node().unwrap();
    let bell = bell_for(Uid(2), &ringer);
    bell.ring(1).unwrap();
    assert_eq!(t.wait(Some(MS200)), vec![1]);
    bell.ring(2).unwrap(); // lands while "draining"
    assert_eq!(t.wait(Some(MS200)), vec![2], "retained across the re-park");
}

#[test]
fn a_control_and_a_data_notification_both_survive() {
    let t = IoxTransport::create(&instance(), Uid(3), 0, manifest()).unwrap();
    let ringer = iox_node().unwrap();
    let bell = bell_for(Uid(3), &ringer);
    bell.ring(0).unwrap();
    bell.ring(3).unwrap();
    let mut got = t.wait(Some(MS200));
    got.sort();
    assert_eq!(got, vec![0, 3]);
}

#[test]
fn a_control_message_crosses_shared_memory_and_comes_back_acked() {
    // The ack is the only thing that orders a wire change, so a message without one stalls the sequence.
    let transport = Arc::new(IoxTransport::create(&instance(), Uid(4), 0, manifest()).unwrap());
    let mut node = NodeRuntime::new(
        manifest(),
        default_factory::<Passthrough>(),
        manifest().default_params(),
        transport.clone(),
        NodeEnv::detached(),
    );
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
    assert_eq!(status_within(&channel, MS200), vec![WireStatus::Ack { seq: 41, ok: Ok(()) }]);

    // The ack carries the VERDICT, not a receipt: the graph abandons a refused sequence.
    channel.send(Envelope {
        seq: 42,
        control: Control::OutSlot { slot: "nope".to_string(), targets: Vec::new() },
    });
    node.run_once();
    assert_eq!(
        status_within(&channel, MS200),
        vec![WireStatus::Ack { seq: 42, ok: Err("no output slot `nope`".to_string()) }]
    );

    let fault = NodeFault::Process { msg: "boom".to_string(), since: 12.5 };
    transport.report(WireStatus::Health(Status::Fault { fault: Some(fault.clone()) }));
    assert_eq!(status_within(&channel, MS200), vec![WireStatus::Health(Status::Fault { fault: Some(fault) })]);
}

#[test]
fn a_frame_reaches_a_wired_consumer_and_rings_its_slot() {
    // A wire is two declarations and nothing else: neither end knows the other's uid.
    let producer = IoxTransport::create(&instance(), Uid(5), 0, manifest()).unwrap();
    let consumer = IoxTransport::create(&instance(), Uid(6), 0, manifest()).unwrap();
    consumer.wire_in("in", &[output_service(&base_of(Uid(5)), "out")]).unwrap();
    producer.wire_out("out", &[(door_service(&base_of(Uid(6))), 1)]).unwrap();

    producer.publish("out", &frame(&[1.0, 2.0, 3.0]));
    assert_eq!(consumer.wait(Some(MS200)), vec![1], "woken by the slot's own event id");
    let got = consumer.drain_inputs();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].0.as_str(), got[0].1), ("in", 0), "slot, and its position in the wire order");
    assert_eq!(f32s(&got[0].2), vec![1.0, 2.0, 3.0]);
    assert!(consumer.drain_inputs().is_empty(), "a drained wire is empty");
}

#[test]
fn a_frame_larger_than_the_initial_slice_still_lands() {
    // `AllocationStrategy::Static`, the iceoryx2 default, refuses this: a GOOF frame is variable-size.
    let producer = IoxTransport::create(&instance(), Uid(7), 0, manifest()).unwrap();
    let consumer = IoxTransport::create(&instance(), Uid(8), 0, manifest()).unwrap();
    consumer.wire_in("in", &[output_service(&base_of(Uid(7)), "out")]).unwrap();

    let big: Vec<f32> = (0..80_000).map(|i| i as f32).collect(); // 320 KB, past the 64 KiB start
    producer.publish("out", &frame(&big));
    let got = consumer.drain_inputs();
    assert_eq!(got.len(), 1, "the oversized frame was published and received");
    assert_eq!(f32s(&got[0].2), big);
}

#[test]
fn a_wire_the_new_set_omits_stops_delivering() {
    // The slot set is DECLARATIVE: what a message does not name is dropped, wire displacement included.
    let producer = IoxTransport::create(&instance(), Uid(9), 0, manifest()).unwrap();
    let consumer = IoxTransport::create(&instance(), Uid(10), 0, manifest()).unwrap();
    let service = output_service(&base_of(Uid(9)), "out");
    consumer.wire_in("in", &[service]).unwrap();
    producer.publish("out", &frame(&[1.0]));
    assert_eq!(consumer.drain_inputs().len(), 1);

    consumer.wire_in("in", &[]).unwrap();
    producer.publish("out", &frame(&[2.0]));
    assert!(consumer.drain_inputs().is_empty(), "the dropped wire delivers nothing");
}

#[test]
fn a_wire_the_new_set_still_names_keeps_what_it_is_holding() {
    // The FULL set is re-sent on every change to a slot, so a surviving wire must be kept rather than
    // rebuilt — a rebuild would discard whatever its producer has already sent.
    let producer = IoxTransport::create(&instance(), Uid(11), 0, manifest()).unwrap();
    let consumer = IoxTransport::create(&instance(), Uid(12), 0, manifest()).unwrap();
    let held = output_service(&base_of(Uid(11)), "out");
    let added = output_service(&base_of(Uid(13)), "out");
    consumer.wire_in("in", std::slice::from_ref(&held)).unwrap();
    producer.publish("out", &frame(&[1.0])); // in flight, unread

    consumer.wire_in("in", &[held, added]).unwrap();
    let got = consumer.drain_inputs();
    assert_eq!(got.len(), 1, "the second wire has nothing yet");
    assert_eq!(f32s(&got[0].2), vec![1.0], "and the first still holds what it was sent");
}

#[test]
fn a_slot_feeds_more_consumers_than_the_iceoryx2_defaults_allow() {
    // `max_subscribers` is inert on its own: a service is opened from one iceoryx2 node per graph
    // node, and `max_nodes` counts exactly those.
    const CONSUMERS: u64 = 24;
    let producer = IoxTransport::create(&instance(), Uid(20), 0, manifest()).unwrap();
    let service = output_service(&base_of(Uid(20)), "out");
    let consumers: Vec<IoxTransport> = (0..CONSUMERS)
        .map(|i| {
            let c = IoxTransport::create(&instance(), Uid(100 + i), 0, manifest()).unwrap();
            c.wire_in("in", std::slice::from_ref(&service)).expect("subscribe");
            c
        })
        .collect();

    producer.publish("out", &frame(&[7.0]));
    for (i, consumer) in consumers.iter().enumerate() {
        assert_eq!(consumer.drain_inputs().len(), 1, "consumer {i} of {CONSUMERS} got the frame");
    }
}

#[test]
fn a_multi_input_keeps_one_cell_per_wire_in_the_order_it_was_given() {
    // A multi slot's cells are keyed by service name and ordered by the SET, never by the producers.
    // The wire count is past the event service's `max_nodes`, which counts one node per producer.
    const WIRES: u64 = 40;
    let producers: Vec<IoxTransport> = (0..WIRES)
        .map(|i| IoxTransport::create(&instance(), Uid(200 + i), 0, manifest()).unwrap())
        .collect();
    let consumer = IoxTransport::create(&instance(), Uid(21), 0, manifest()).unwrap();
    // Reversed, so a wire index that follows the producers rather than the set is visible.
    let services: Vec<String> =
        (0..WIRES).rev().map(|i| output_service(&base_of(Uid(200 + i)), "out")).collect();
    consumer.wire_in("in", &services).expect("subscribe");

    let door = door_service(&base_of(Uid(21)));
    for (i, producer) in producers.iter().enumerate() {
        producer.wire_out("out", &[(door.clone(), 1)]).expect("ring this consumer");
        producer.publish("out", &frame(&[i as f32]));
    }
    // Twice on one wire before the drain: latest-wins keeps the second, per wire.
    producers[0].publish("out", &frame(&[100.0]));

    let got = consumer.drain_inputs();
    assert_eq!(got.len(), WIRES as usize, "one cell per wire, none merged");
    assert_eq!(
        got.iter().map(|(_, index, _)| *index).collect::<Vec<_>>(),
        (0..WIRES as usize).collect::<Vec<_>>(),
        "the cells are indexed by position in the set"
    );
    assert_eq!(
        got.iter().map(|(_, _, frame)| f32s(frame)[0]).collect::<Vec<_>>(),
        (0..WIRES).rev().map(|i| if i == 0 { 100.0 } else { i as f32 }).collect::<Vec<f32>>(),
        "each cell holds its own producer's newest frame"
    );
}

/// The env var that turns this binary into the child below. Its value is irrelevant.
const CRASH_HELPER: &str = "GOOFI_TRANSPORT_CRASH_HELPER";

/// The child: open two iceoryx2 nodes, NAME them, then wait to be killed. It names its own ids
/// because that directory is machine-global and a diff picks up other binaries' entries.
#[test]
fn crash_helper() {
    if std::env::var(CRASH_HELPER).is_err() {
        return; // the ordinary run: this test is only the child's entry point
    }
    let (a, b) = (iox_node().expect("node a"), iox_node().expect("node b"));
    println!("READY {} {}", a.id().value(), b.id().value());
    std::thread::sleep(Duration::from_secs(60));
}

#[test]
fn what_a_crash_left_behind_is_gone_by_the_next_start() {
    // A killed process drops NOTHING, so its node directories and shared memory stay allocated.
    let dirs = || -> std::collections::HashSet<String> {
        std::fs::read_dir(goofi_transport::nodes_dir().expect("the nodes directory"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect()
    };
    let mut child = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args(["crash_helper", "--exact", "--nocapture"])
        .env(CRASH_HELPER, "1")
        .stdout(std::process::Stdio::piped())
        // SIGKILLed mid-test, so its parting "broken pipe" would otherwise read as this test's failure.
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn the child");

    // Its nodes exist once it says so — not once it started, which would race their creation.
    let mut out = std::io::BufReader::new(child.stdout.take().expect("the child's stdout"));
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut line = String::new();
    while !line.contains("READY") && std::time::Instant::now() < deadline {
        line.clear();
        std::io::BufRead::read_line(&mut out, &mut line).expect("read the child");
    }
    let ids: Vec<String> = line.split_whitespace().skip(1).map(str::to_string).collect();
    assert_eq!(ids.len(), 2, "the child named the two nodes it opened: {line:?}");
    let present = dirs();
    for id in &ids {
        assert!(present.contains(id), "the child's node `{id}` has a directory to leave behind");
    }

    // SIGKILL, because the point is a process that drops nothing: a graceful exit would clean up.
    let _ = std::process::Command::new("kill").args(["-9", &child.id().to_string()]).status();
    let _ = child.wait();

    goofi_transport::reclaim_stale_resources();
    let left: Vec<&String> = ids.iter().filter(|id| dirs().contains(*id)).collect();
    assert!(left.is_empty(), "the sweep left {left:?} standing, for every later start to walk again");
}
