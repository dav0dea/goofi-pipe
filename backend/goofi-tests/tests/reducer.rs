//! The shared per-slot reducer: N viewers of one slot cost ONE reduce and encode per frame, and a
//! frame goes out when the producer EMITS, when a subscriber JOINS, or when the spec union CHANGES
//! — never because a deadline elapsed.
//!
//! Driven against the live `AppState`'s own reducers, which is what `/data` fans out from.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use goofi_bridge::reducer::SlotKey;
use goofi_tests::Goofi;
use goofi_view::{AxisReduce, DimCmp, ReduceMethod, ViewDtype, ViewSpec};
use tokio::sync::broadcast;

fn line_spec(width: usize) -> ViewSpec {
    ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Le, 2)],
        dims: vec![],
        reduce: vec![AxisReduce { dim: -1, max: width, method: ReduceMethod::Envelope }],
    }
}

/// A producer that emits exactly one frame per ARMING, and nothing at all until it is armed. The
/// Oscillator cannot express these tests: it free-runs, so "the producer emitted nothing" is a
/// state it is never in — and it is the SILENCE between emits that the send rule is about.
struct Armed {
    go: Arc<AtomicBool>,
    n: f32,
}

impl goofi_node::Node for Armed {
    fn process(&mut self, _i: &goofi_node::Inputs<'_>, o: &mut goofi_node::Outputs<'_>,
               _c: &mut goofi_node::NodeCtx, _p: &goofi_node::Params<'_>) -> goofi_node::NodeResult {
        if !self.go.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        self.n += 1.0;
        let body = self.n.to_le_bytes().to_vec();
        o.set("out", goofi_core::Data::array_f32(vec![1], body, goofi_core::Meta::empty()).unwrap());
        Ok(())
    }
}

static ARMED_OUT: &[goofi_node::OutputDecl] =
    &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }];
static ARMED: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "_TestArmed",
    category: "test",
    doc: "emits one frame per arming",
    inputs: &[],
    outputs: ARMED_OUT,
    params: &[],
    isolation: goofi_node::Isolation::InProcess,
    producer: true,
    factory: || unreachable!("a dyn type is built by its registered factory"),
};

/// One armed producer and the flag that fires it.
fn armed() -> (Goofi, SlotKey, Arc<AtomicBool>) {
    let g = Goofi::new();
    let go = Arc::new(AtomicBool::new(false));
    let flag = go.clone();
    g.register_dyn(&ARMED, Box::new(move |_| Box::new(Armed { go: flag.clone(), n: 0.0 })));
    let uid = g.add("_TestArmed");
    (g, (uid, "out".to_string()), go)
}

/// Fire the producer until a frame reaches `rx`, and answer whether one ever did.
///
/// A single arming cannot establish the stream: the reducer opens its subscriber on its OWN task,
/// while the producer is already running, and the data services keep no history. A frame emitted in
/// that window is gone for ever. Re-arming until one lands is also the positive counterpart each
/// silence assertion needs — silence alone cannot tell a quiet stream from one never connected.
async fn arm_until_received(go: &AtomicBool, rx: &mut broadcast::Receiver<bytes::Bytes>) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        go.store(true, Ordering::Release);
        if matches!(tokio::time::timeout(Duration::from_millis(50), rx.recv()).await, Ok(Ok(_))) {
            return true;
        }
    }
    false
}

/// Let a still-in-flight arming land and take it off the channel, so a silence window that follows
/// measures the send rule rather than the tail of the warm-up.
async fn settle(rx: &mut broadcast::Receiver<bytes::Bytes>) {
    tokio::time::sleep(Duration::from_millis(150)).await;
    while rx.try_recv().is_ok() {}
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_reducer_serves_every_subscriber_and_tears_down_on_last_leave() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let key: SlotKey = (osc, "out".to_string());
    let r = &g.state.reducers;

    let (c1, c2) = (r.new_conn(), r.new_conn());
    let mut r1 = r.subscribe(key.clone(), c1);
    let mut r2 = r.subscribe(key.clone(), c2);
    assert_eq!(r.active_slots(), 1, "one shared reducer for the slot");
    assert_eq!(r.subscribers(&key), 2);

    for (n, rx) in [("1", &mut r1), ("2", &mut r2)] {
        let got = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await;
        assert!(got.is_ok_and(|f| f.is_ok()), "subscriber {n} got a reduced frame");
    }

    r.unsubscribe(&key, c1);
    assert_eq!(r.active_slots(), 1, "one subscriber left → the reducer lives");
    r.unsubscribe(&key, c2);
    assert_eq!(r.active_slots(), 0, "the last left → the reducer is dropped");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unchanged_frame_is_not_rebroadcast() {
    // The wire-level bug behind "ufreq 30, fps 30, ~33 drops/s": the sweep re-encoded and re-sent
    // the SAME published frame every 16 ms — 375 frames against 180 fresh emits over 6 s on a live
    // socket.
    let (g, key, go) = armed();
    let r = &g.state.reducers;
    let c = r.new_conn();
    let mut rx = r.subscribe(key.clone(), c);

    assert!(arm_until_received(&go, &mut rx).await, "the emitted frame reached the subscriber");
    settle(&mut rx).await;

    // The producer emits nothing more, so the stream is SILENT. Ten sweep deadlines pass here; the
    // old loop shipped ten copies of the same frame.
    assert!(tokio::time::timeout(Duration::from_millis(200), rx.recv()).await.is_err(),
            "an unchanged frame was re-broadcast on a sweep deadline");

    // A spec change re-serves the CACHED frame at its new reduction, with no fresh emit — a
    // resized viewer of a slow producer must not stare at the stale size.
    r.set_specs(&key, c, vec![line_spec(64)]);
    assert!(tokio::time::timeout(Duration::from_millis(500), rx.recv()).await.is_ok_and(|f| f.is_ok()),
            "a spec change re-serves the cached frame");
    assert!(tokio::time::timeout(Duration::from_millis(200), rx.recv()).await.is_err(),
            "…as ONE frame, not a new steady stream");

    // A connection that DROPPED a frame re-offers: the skip holds the last frame back until the
    // next emit, so "giving up costs one frame" is only true if the frame comes around again.
    r.reoffer(&key);
    assert!(tokio::time::timeout(Duration::from_millis(500), rx.recv()).await.is_ok_and(|f| f.is_ok()),
            "a re-offer serves the cached frame again");

    go.store(true, Ordering::Release);
    assert!(tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.is_ok_and(|f| f.is_ok()),
            "a fresh emit flows to the subscriber");
    r.unsubscribe(&key, c);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restarted_node_re_homes_the_stream_instead_of_going_quiet() {
    // A node's service name carries its generation, so a restart publishes somewhere this task has
    // never opened. Nothing else in the loop can notice: the subscriber on the corpse's name
    // receives cleanly and for ever, it simply never receives anything again — a viewer frozen on
    // its last frame with no error anywhere.
    let (g, key, go) = armed();
    let r = &g.state.reducers;
    let c = r.new_conn();
    let mut rx = r.subscribe(key.clone(), c);
    assert!(arm_until_received(&go, &mut rx).await, "the stream is live before the restart");
    settle(&mut rx).await;

    let before = g.state.graph.lock().unwrap().output_service_of(key.0, "out");
    g.call("restart_node", serde_json::json!({ "node": goofi_tests::hex(key.0) }));
    assert_ne!(g.state.graph.lock().unwrap().output_service_of(key.0, "out"), before,
               "a rebirth is a new name");

    // The reborn node is a fresh instance, so it is disarmed. Arming until a frame lands is what
    // the re-home costs: for up to one rehome interval the reducer still listens on the dead name.
    assert!(arm_until_received(&go, &mut rx).await, "the reducer followed the restart");
    r.unsubscribe(&key, c);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reduction_cost_is_o1_in_subscriber_count() {
    // The headline claim: N tabs on one slot cost ONE reduce and encode per frame, not N. The pass
    // count must stay bounded by WALL-CLOCK — one task at ~16 ms — never multiplied by subscribers.
    const SUBS: usize = 50;
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let key: SlotKey = (osc, "out".to_string());
    let r = &g.state.reducers;

    let subs: Vec<_> = (0..SUBS).map(|_| {
        let c = r.new_conn();
        (c, r.subscribe(key.clone(), c))
    }).collect();
    assert_eq!(r.active_slots(), 1, "all {SUBS} tabs share ONE reducer");
    assert_eq!(r.subscribers(&key), SUBS);

    tokio::time::sleep(Duration::from_millis(200)).await; // ~12 passes at 16 ms
    let passes = r.reductions(&key);
    // Bounded by wall-clock with generous headroom, and CATEGORICALLY below the per-subscriber
    // count (~600) the old per-connection loop would have done.
    assert!(passes > 0 && passes < 40,
            "reduce passes {passes} bounded by wall-clock, not by {SUBS} subscribers");

    // Every subscriber is live. `len()` is what is QUEUED — nothing here recv()s, so an empty ring
    // means this subscriber never received, which is exactly the fan-out bug. A `<= 16` bound could
    // not catch it: an unreached subscriber has len 0 and passes.
    for (_, rx) in &subs {
        assert!(!rx.is_empty(), "every subscriber received the shared reducer's frames");
    }
}
