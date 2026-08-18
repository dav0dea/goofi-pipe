//! Shared per-slot data reducers (thalamus G1/G2). ONE reduction per active `(node, slot)`,
//! sized to the UNION of all subscribing connections' `ViewSpec`s, fanned out over a broadcast
//! so N tabs viewing the same slot cost one reduce+encode — not N. This replaces the former
//! per-connection reduce loop in `handle_data`, eliminating the multi-tab duplicate reduction.
//!
//! The reducer task runs at viewer rate (~16 ms), latest-wins, **subscribed to the producer's own
//! output service** — the same door a node's downstream consumer comes in through, and the only
//! door there is (§7: there is no privileged path into a node for a frame). It resolves that
//! service name under one brief graph lock and is lock-free after, so no number of viewers can
//! slow a `process()` down. A `viewers` refcount spawns the task on the first subscriber and
//! aborts it on the last leave.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use goofi_engine::{Graph, Uid};
use goofi_view::ViewSpec;
use tokio::sync::broadcast;

/// The physical stream a reducer serves: a node uid + one of its output slot names.
pub type SlotKey = (Uid, String);
/// A unique id per `/data` connection, so its spec contribution can be tracked + removed.
pub type ConnId = u64;

/// Flatten every connection's `ViewSpec`s into the single list the planner merges. Empty when
/// no connection has declared any spec yet (→ full-resolution passthrough).
fn union_specs(by_conn: &HashMap<ConnId, Vec<ViewSpec>>) -> Vec<ViewSpec> {
    by_conn.values().flatten().cloned().collect()
}

struct SlotReducer {
    /// Per-connection specs; the reduction plans against their union.
    specs: Arc<Mutex<HashMap<ConnId, Vec<ViewSpec>>>>,
    /// Encoded reduced-frame fan-out to every subscribing connection. `Bytes` so the socket
    /// task forwards the SHARED buffer — a per-subscriber copy would undo the dedup.
    tx: broadcast::Sender<Bytes>,
    /// The driving task (aborted on last-leave teardown).
    task: tokio::task::JoinHandle<()>,
    /// Count of reduce+encode passes — proves dedup in tests (one pass serves all subscribers).
    reductions: Arc<AtomicU64>,
    /// Serve generation: bumped on every spec change and subscriber join, so the loop re-serves
    /// the CURRENT frame once even when the producer has not emitted — a joiner must not stare
    /// at a blank viewer until a sparse producer's next emit, and a resized viewer must not
    /// stare at the stale reduction.
    gen: Arc<AtomicU64>,
}

/// Manages all per-slot reducers, spawning/dropping them by subscriber refcount. Cloneable
/// (shares one inner map); lives in `AppState`.
#[derive(Clone)]
pub struct SlotReducers {
    inner: Arc<Mutex<HashMap<SlotKey, SlotReducer>>>,
    graph: Arc<Mutex<Graph>>,
    next_conn: Arc<AtomicU64>,
}

impl SlotReducers {
    pub fn new(graph: Arc<Mutex<Graph>>) -> SlotReducers {
        SlotReducers {
            inner: Arc::new(Mutex::new(HashMap::new())),
            graph,
            next_conn: Arc::new(AtomicU64::new(1)),
        }
    }

    /// A fresh connection id.
    pub fn new_conn(&self) -> ConnId {
        self.next_conn.fetch_add(1, Ordering::Relaxed)
    }

    /// Subscribe `conn` to `key`'s reduced stream, spawning the slot's reducer task if this is
    /// the first subscriber. Returns the broadcast receiver of encoded frames.
    pub fn subscribe(&self, key: SlotKey, conn: ConnId) -> broadcast::Receiver<Bytes> {
        let mut map = self.inner.lock().unwrap();
        let reducer = map.entry(key.clone()).or_insert_with(|| {
            let specs = Arc::new(Mutex::new(HashMap::new()));
            let (tx, _) = broadcast::channel(16);
            let reductions = Arc::new(AtomicU64::new(0));
            let gen = Arc::new(AtomicU64::new(0));
            let task = spawn_reducer(
                key.clone(),
                specs.clone(),
                tx.clone(),
                self.graph.clone(),
                reductions.clone(),
                gen.clone(),
            );
            SlotReducer { specs, tx, task, reductions, gen }
        });
        reducer.specs.lock().unwrap().entry(conn).or_default();
        // A joiner is served the current frame on the next sweep (see `gen`). The receiver
        // MUST exist before the bump: bumped first, a sweep interleaving between the two
        // statements would broadcast the join-serve into a fan-out this joiner is not yet
        // part of, and the skip would then hold until the next emit or spec change.
        let rx = reducer.tx.subscribe();
        reducer.gen.fetch_add(1, Ordering::Release);
        rx
    }

    /// Replace `conn`'s declared specs for `key` (latest-wins). No-op if the slot is gone.
    pub fn set_specs(&self, key: &SlotKey, conn: ConnId, specs: Vec<ViewSpec>) {
        if let Some(r) = self.inner.lock().unwrap().get(key) {
            r.specs.lock().unwrap().insert(conn, specs);
            // The union changed shape: re-serve the current frame at its new reduction.
            r.gen.fetch_add(1, Ordering::Release);
        }
    }

    /// Ask the slot to serve its current frame again on the next sweep. For a connection that
    /// DROPPED a frame (a bounded write that timed out, a lagged broadcast ring): the skip
    /// holds an unchanged frame back, so without this the drop would cost every frame until
    /// the producer's next emit — indefinitely for a stopped one. Reaches every subscriber of
    /// the slot (the fan-out is shared), which is one duplicate frame on a rare path.
    pub fn reoffer(&self, key: &SlotKey) {
        if let Some(r) = self.inner.lock().unwrap().get(key) {
            r.gen.fetch_add(1, Ordering::Release);
        }
    }

    /// Remove `conn` from `key`; when the last subscriber leaves, abort the task and drop the
    /// slot so an idle stream costs nothing.
    pub fn unsubscribe(&self, key: &SlotKey, conn: ConnId) {
        let mut map = self.inner.lock().unwrap();
        let Some(r) = map.get(key) else { return };
        let empty = {
            let mut specs = r.specs.lock().unwrap();
            specs.remove(&conn);
            specs.is_empty()
        };
        if empty {
            if let Some(r) = map.remove(key) {
                r.task.abort();
            }
        }
    }

    /// Number of live slot reducers (test/diagnostic).
    pub fn active_slots(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Subscribers on a slot (test/diagnostic).
    pub fn subscribers(&self, key: &SlotKey) -> usize {
        self.inner.lock().unwrap().get(key).map(|r| r.specs.lock().unwrap().len()).unwrap_or(0)
    }

    /// Reduce+encode passes run for a slot so far (test/diagnostic — proves one pass serves
    /// every subscriber, not one-per-connection).
    pub fn reductions(&self, key: &SlotKey) -> u64 {
        self.inner.lock().unwrap().get(key).map(|r| r.reductions.load(Ordering::Relaxed)).unwrap_or(0)
    }
}

/// How often the slot's subscribe address is re-derived from the graph. A node's service name
/// carries its GENERATION, so a restart (an explicit `restart_node`, or the automatic one a node-file
/// rescan does) re-homes the stream to a name this task has never opened — and a subscriber left on
/// the corpse's name would simply go quiet for ever. Once a second rather than once a sweep because
/// §7's whole point is that the steady state does not touch the graph lock: a restart is a user
/// action, and a viewer following it within a second is following it.
const REHOME_INTERVAL: Duration = Duration::from_secs(1);

/// One end of a slot's data service: the subscriber, the iceoryx2 node it was built from (which
/// must outlive it), and the service name it was opened on — the thing a re-home compares against.
struct SlotFeed {
    subscriber: goofi_engine::runtime::ByteSubscriber,
    service: String,
    /// Last, because Rust drops a struct's fields in declaration order and the node must outlive
    /// the subscriber built from it. A node that drops first cannot remove its own directory, and
    /// a re-home builds one of these for each slot every second.
    _node: goofi_engine::runtime::IoxNode,
}

/// Open a subscriber on `(uid, slot)`'s current output service, or `None` while the node is not
/// addressable (it has been removed, or its services do not exist yet). A miss is retried on the
/// next re-home rather than being fatal: a viewer may legitimately subscribe to a node that is
/// still being born.
fn open_feed(graph: &Mutex<Graph>, uid: Uid, slot: &str) -> Option<SlotFeed> {
    let service = {
        let g = graph.lock().unwrap();
        g.manifest(uid)?;
        g.output_service_of(uid, slot)
    };
    let node = goofi_engine::runtime::iox_node().ok()?;
    let subscriber = goofi_engine::runtime::open_output_subscriber(&node, &service).ok()?;
    Some(SlotFeed { _node: node, subscriber, service })
}

/// Spawn the per-slot reducer loop: every ~16 ms take whatever the producer has published and —
/// ONLY when it emitted since the last send, a subscriber joined, or the spec union changed —
/// reduce it to the union of subscribers' specs (passthrough while none), encode once, and
/// broadcast to all. The sweep is a sampling deadline, never a send cadence: re-shipping an
/// unchanged frame every 16 ms put 62.5 frames/s on the wire for a 30 Hz producer, and every
/// viewer paid decode + (capped) paint for frames that carried nothing new.
///
/// The cached frame is also what a JOIN or a spec change is served from (§7): the producer's
/// service has no history, so a viewer arriving on a stream nobody was subscribed to waits for the
/// next emit — which is §3.5's no-replay rule and not a gap.
fn spawn_reducer(
    key: SlotKey,
    specs: Arc<Mutex<HashMap<ConnId, Vec<ViewSpec>>>>,
    tx: broadcast::Sender<Bytes>,
    graph: Arc<Mutex<Graph>>,
    reductions: Arc<AtomicU64>,
    gen: Arc<AtomicU64>,
) -> tokio::task::JoinHandle<()> {
    let (uid, slot) = key;
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(16));
        // Latest-wins: a deadline missed while this task was starved is a sample that no longer
        // exists, not a debt. Tokio's default (Burst) would repay every one of them back-to-back
        // the moment the worker is unblocked.
        //
        // What that repayment costs is LOOP ITERATIONS — a spin of non-blocking receives — and not
        // reduce+encode+broadcast passes. The de-dup gate below (`!fresh && served == Some(g_now)`)
        // returns before any pass is counted or sent, so a repaid deadline over an unchanged frame
        // already does no work. Measured: switching this line to `Burst` changes nothing any test
        // can see, and the burst counter reads 2 either way.
        //
        // So this line is cheap insurance against a spin, and the guarantee a reader cares about
        // lives in that gate, pinned by `an_unchanged_frame_is_not_rebroadcast`.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut feed = open_feed(&graph, uid, &slot);
        let mut rehomed = std::time::Instant::now();
        // The newest frame this task has seen, and the serve generation it was last broadcast at.
        // `None` for the generation means "never broadcast", which is what makes the first frame
        // go out without waiting for a bump.
        let mut cached: Option<goofi_core::Data> = None;
        let mut served: Option<u64> = None;
        loop {
            ticker.tick().await;
            if rehomed.elapsed() >= REHOME_INTERVAL {
                rehomed = std::time::Instant::now();
                let current = {
                    let g = graph.lock().unwrap();
                    g.manifest(uid).map(|_| g.output_service_of(uid, &slot))
                };
                // Only a CHANGED name reopens: re-creating the port every second would churn a
                // service that is working perfectly well.
                if current.is_some_and(|c| feed.as_ref().is_none_or(|f| f.service != c)) {
                    feed = open_feed(&graph, uid, &slot);
                }
            }
            let mut fresh = false;
            if let Some(f) = &feed {
                while let Ok(Some(sample)) = f.subscriber.receive() {
                    if let Ok(frame) = goofi_codec::decode(sample.payload()) {
                        cached = Some(frame);
                        fresh = true;
                    }
                }
            }
            let Some(d) = cached.clone() else { continue };
            let g_now = gen.load(Ordering::Acquire);
            if !fresh && served == Some(g_now) {
                continue; // nothing new to say — no emit, no joiner, no spec change
            }
            let merged = union_specs(&specs.lock().unwrap());
            let out = if merged.is_empty() {
                d
            } else {
                let plan = goofi_view::plan(&merged, &d);
                goofi_core::reduce::reduce_for_view(&d, &plan)
            };
            reductions.fetch_add(1, Ordering::Relaxed);
            let bytes = Bytes::from(goofi_codec::encode(&out));
            let _ = tx.send(bytes); // Err only if all receivers are momentarily gone — harmless.
            served = Some(g_now);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_view::{AxisReduce, DimCmp, ReduceMethod, ViewDtype};
    use std::sync::atomic::AtomicBool;

    fn line_spec(width: usize) -> ViewSpec {
        ViewSpec {
            dtype: ViewDtype::Array,
            ndim: vec![(DimCmp::Le, 2)],
            dims: vec![],
            reduce: vec![AxisReduce { dim: -1, max: width, method: ReduceMethod::Envelope }],
        }
    }

    /// Production's status worker, in the four lines these tests need of it: a node becomes
    /// addressable, and its params reach it, only once someone drains its status service. Without
    /// one running, an `add_node` here would leave the node on its manifest defaults for ever —
    /// including the `globals.default_ufreq` pacing every producer carries.
    struct Drainer {
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Drainer {
        fn spawn(graph: Arc<Mutex<Graph>>) -> Drainer {
            let stop = Arc::new(AtomicBool::new(false));
            let thread = {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        graph.lock().unwrap().drain_status();
                        std::thread::sleep(Duration::from_millis(1));
                    }
                })
            };
            Drainer { stop, thread: Some(thread) }
        }
    }

    impl Drop for Drainer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    /// A producer that emits exactly one frame per ARMING, and nothing at all until it is armed.
    /// The oscillator cannot express these tests any more: it free-runs, so "the producer emitted
    /// nothing" is a state it is never in — and it is the silence between emits that the sweep's
    /// send rule is about.
    struct Armed {
        go: Arc<AtomicBool>,
        n: f32,
    }

    impl goofi_node::Node for Armed {
        fn process(
            &mut self,
            _i: &goofi_node::Inputs<'_>,
            o: &mut goofi_node::Outputs<'_>,
            _c: &mut goofi_node::NodeCtx,
            _p: &goofi_node::Params<'_>,
        ) -> goofi_node::NodeResult {
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
        factory: || Box::new(Armed { go: Arc::new(AtomicBool::new(false)), n: 0.0 }),
    };

    /// A graph holding one armed producer, the flag that fires it, and the drainer that keeps the
    /// graph hearing from it.
    fn armed_graph() -> (Arc<Mutex<Graph>>, Uid, Arc<AtomicBool>, Drainer) {
        let go = Arc::new(AtomicBool::new(false));
        let mut g = Graph::new();
        let armed = go.clone();
        g.register_dyn_type(&ARMED, Box::new(move |_| Box::new(Armed { go: armed.clone(), n: 0.0 })));
        let uid = g.add_node("_TestArmed", None).unwrap();
        let graph = Arc::new(Mutex::new(g));
        let drainer = Drainer::spawn(graph.clone());
        (graph, uid, go, drainer)
    }

    /// Fire the armed producer until a frame reaches `rx`; answer whether one ever did.
    ///
    /// A single arming cannot establish the stream: the reducer opens its subscriber on its OWN
    /// task — after an `iox_node()` and a graph lock — while the producer is already running, and
    /// the data services keep no history (§3.5). A frame emitted in that window is gone for ever,
    /// and with it every assertion downstream. Re-arming until one lands is also the positive
    /// counterpart each silence assertion needs: `silent` alone cannot tell a quiet stream from a
    /// stream that was never connected.
    async fn arm_until_received(go: &AtomicBool, rx: &mut broadcast::Receiver<Bytes>) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            go.store(true, Ordering::Release);
            if matches!(tokio::time::timeout(Duration::from_millis(50), rx.recv()).await, Ok(Ok(_))) {
                return true;
            }
        }
        false
    }

    /// Let a still-in-flight arming land and take it off the channel, so a silence window that
    /// follows is measuring the sweep rule rather than the tail of the warm-up.
    async fn settle(rx: &mut broadcast::Receiver<Bytes>) {
        tokio::time::sleep(Duration::from_millis(150)).await;
        while rx.try_recv().is_ok() {}
    }

    #[test]
    fn union_specs_concatenates_every_connections_specs() {
        let mut by_conn: HashMap<ConnId, Vec<ViewSpec>> = HashMap::new();
        assert!(union_specs(&by_conn).is_empty(), "no connections → passthrough");
        by_conn.insert(1, vec![line_spec(64)]);
        by_conn.insert(2, vec![line_spec(128)]);
        assert_eq!(union_specs(&by_conn).len(), 2, "both connections' specs are merged");
        // A connection with no declared specs contributes nothing.
        by_conn.insert(3, vec![]);
        assert_eq!(union_specs(&by_conn).len(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn one_reducer_serves_multiple_subscribers_on_a_slot() {
        // Two connections view the SAME slot: exactly ONE reducer task exists, both receive
        // reduced frames, and it tears down on last-leave.
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let graph = Arc::new(Mutex::new(g));
        let _drainer = Drainer::spawn(graph.clone());

        let reducers = SlotReducers::new(graph.clone());
        let key: SlotKey = (osc, "out".to_string());
        let c1 = reducers.new_conn();
        let c2 = reducers.new_conn();
        let mut r1 = reducers.subscribe(key.clone(), c1);
        let mut r2 = reducers.subscribe(key.clone(), c2);
        assert_eq!(reducers.active_slots(), 1, "one shared reducer for the slot");
        assert_eq!(reducers.subscribers(&key), 2, "both connections subscribed");

        // Both subscribers receive a frame from the single reducer.
        let f1 = tokio::time::timeout(Duration::from_secs(5), r1.recv()).await;
        let f2 = tokio::time::timeout(Duration::from_secs(5), r2.recv()).await;
        assert!(f1.is_ok() && f1.unwrap().is_ok(), "subscriber 1 got a reduced frame");
        assert!(f2.is_ok() && f2.unwrap().is_ok(), "subscriber 2 got a reduced frame");

        // Last-leave tears the reducer down.
        reducers.unsubscribe(&key, c1);
        assert_eq!(reducers.active_slots(), 1, "still one subscriber → reducer alive");
        reducers.unsubscribe(&key, c2);
        assert_eq!(reducers.active_slots(), 0, "last subscriber left → reducer dropped");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unchanged_frame_is_not_rebroadcast() {
        // The wire-level bug behind "ufreq 30, fps 30, ~33 drops/s": the sweep re-encoded and
        // re-sent the SAME published frame every 16 ms (62.5/s for a 30 Hz producer — measured
        // 375 frames / 180 fresh emits over 6 s on a live socket). A frame goes out when the
        // producer EMITS, when a subscriber JOINS, or when the spec union CHANGES — never
        // because a deadline elapsed.
        let (graph, uid, go, _drainer) = armed_graph();
        let reducers = SlotReducers::new(graph.clone());
        let key: SlotKey = (uid, "out".to_string());
        let c = reducers.new_conn();
        let mut rx = reducers.subscribe(key.clone(), c);

        assert!(arm_until_received(&go, &mut rx).await, "the emitted frame reached the subscriber");
        settle(&mut rx).await;

        // The producer emits nothing more → the stream is SILENT. Ten sweep deadlines pass; the
        // old loop shipped ~10 copies of the same frame here.
        let silent = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await;
        assert!(silent.is_err(), "an unchanged frame was re-broadcast on a sweep deadline");

        // A spec change re-serves the CACHED frame at its new reduction, even with no fresh emit —
        // a resized viewer of a slow producer must not stare at the stale size.
        reducers.set_specs(&key, c, vec![line_spec(64)]);
        let respec = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
        assert!(respec.is_ok_and(|r| r.is_ok()), "a spec change re-serves the cached frame");
        let quiet = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await;
        assert!(quiet.is_err(), "the spec-change re-serve is one frame, not a new steady stream");

        // A connection that DROPPED a frame (write timeout, broadcast lag) re-offers: the skip
        // holds the last frame back until the next emit, so the drop must ask for it again —
        // "giving up costs one frame" is only true if the frame comes around again.
        reducers.reoffer(&key);
        let reoffered = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
        assert!(reoffered.is_ok_and(|r| r.is_ok()), "a re-offer serves the cached frame again");

        // A fresh emit flows through within a sweep.
        go.store(true, Ordering::Release);
        let fresh = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await;
        assert!(fresh.is_ok_and(|r| r.is_ok()), "a fresh emit flows to the subscriber");

        reducers.unsubscribe(&key, c);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_restarted_node_re_homes_the_stream_instead_of_going_quiet() {
        // A node's service name carries its generation, so a restart publishes somewhere this task
        // has never opened. Nothing else in the loop can notice: the subscriber on the corpse's
        // name receives cleanly and for ever, it simply never receives anything again — which is a
        // viewer frozen on its last frame with no error anywhere.
        let (graph, uid, go, _drainer) = armed_graph();
        let reducers = SlotReducers::new(graph.clone());
        let key: SlotKey = (uid, "out".to_string());
        let c = reducers.new_conn();
        let mut rx = reducers.subscribe(key.clone(), c);
        assert!(arm_until_received(&go, &mut rx).await, "the stream is live before the restart");
        settle(&mut rx).await;

        let before = graph.lock().unwrap().output_service_of(uid, "out");
        graph.lock().unwrap().restart_node(uid).unwrap();
        assert_ne!(graph.lock().unwrap().output_service_of(uid, "out"), before, "a rebirth is a new name");

        // The reborn node is a fresh instance, so it is disarmed. Arming until a frame lands is
        // what the re-home costs: for up to a `REHOME_INTERVAL` the reducer is still listening on
        // the corpse's name, and anything emitted meanwhile is emitted into it.
        assert!(
            arm_until_received(&go, &mut rx).await,
            "the reducer followed the restart to the new service"
        );

        reducers.unsubscribe(&key, c);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reduction_cost_is_o1_in_subscriber_count() {
        // The thalamus headline claim: N tabs on one slot cost ONE reduce+encode per frame, not
        // N. With many subscribers the reduce-pass count must stay bounded by WALL-CLOCK (one
        // ~16ms task), never multiplied by the subscriber count. Uses the internal counter — no
        // external load, so it can't leak load processes.
        const SUBS: usize = 50;
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let graph = Arc::new(Mutex::new(g));
        let _drainer = Drainer::spawn(graph.clone());

        let reducers = SlotReducers::new(graph.clone());
        let key: SlotKey = (osc, "out".to_string());
        let mut subs = Vec::new();
        for _ in 0..SUBS {
            let c = reducers.new_conn();
            subs.push((c, reducers.subscribe(key.clone(), c)));
        }
        assert_eq!(reducers.active_slots(), 1, "all {SUBS} tabs share ONE reducer");
        assert_eq!(reducers.subscribers(&key), SUBS);

        // Let the single reducer run for ~200 ms (~12 passes at 16 ms).
        tokio::time::sleep(Duration::from_millis(200)).await;
        let passes = reducers.reductions(&key);
        // Bounded by wall-clock (~12), with generous headroom — and CATEGORICALLY below the
        // per-subscriber count (SUBS * ~12 ≈ 600) the old per-connection loop would have done.
        assert!(
            passes > 0 && passes < 40,
            "reduce passes {passes} bounded by wall-clock (one shared reducer), not by {SUBS} subscribers"
        );

        // Every subscriber is live (the fan-out reaches all of them). `len()` is what is QUEUED —
        // nothing here recv()s, so an empty ring means this subscriber never received, which is
        // exactly the fan-out bug. The old `<= 16` bound could not catch it (an unreached
        // subscriber has len 0 and passes) and, being `tail - next`, tracked the pass count this
        // test deliberately tolerates up to 40 — so it went red on scheduler overrun instead.
        for (_, r) in &subs {
            assert!(!r.is_empty(), "every subscriber received the shared reducer's frames");
        }
    }
}
