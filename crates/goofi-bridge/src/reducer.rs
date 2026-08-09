//! Shared per-slot data reducers (thalamus G1/G2). ONE reduction per active `(node, slot)`,
//! sized to the UNION of all subscribing connections' `ViewSpec`s, fanned out over a broadcast
//! so N tabs viewing the same slot cost one reduce+encode — not N. This replaces the former
//! per-connection reduce loop in `handle_data`, eliminating the multi-tab duplicate reduction.
//!
//! The reducer task runs at viewer rate (~16 ms), latest-wins, reading the node's latest
//! published frame (`Graph::latest_frame`, a cheap `Arc` clone under a brief lock) — decoupled
//! from the node tick, so any number of viewers never throttle `process()`. A `viewers`
//! refcount spawns the task on the first subscriber and aborts it on the last leave.

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

/// Spawn the per-slot reducer loop: every ~16 ms read the latest frame and — ONLY when the
/// producer has emitted since the last send, a subscriber joined, or the spec union changed —
/// reduce it to the union of subscribers' specs (passthrough while none), encode once, and
/// broadcast to all. The sweep is a sampling deadline, never a send cadence: re-shipping an
/// unchanged frame every 16 ms put 62.5 frames/s on the wire for a 30 Hz producer, and every
/// viewer paid decode + (capped) paint for frames that carried nothing new.
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
        // Latest-wins: a deadline missed while the tick thread held the graph lock is a sample
        // that no longer exists, not a debt. Tokio's default (Burst) would repay every one of
        // them back-to-back over the same frame, the moment the worker is unblocked.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // What the last broadcast served: the exact frame handle (identity = "has the producer
        // emitted since", every emit mints a fresh allocation) and the serve generation. Holding
        // the handle keeps its allocation alive, so the identity check cannot alias.
        let mut sent: Option<(goofi_core::Data, u64)> = None;
        loop {
            ticker.tick().await;
            // Brief graph lock only for the latest-frame Arc clone; plan/reduce/encode run
            // off-lock so a large kHz/HD reduction never serializes against the scheduler tick.
            let d = {
                let g = graph.lock().unwrap();
                g.latest_frame(uid, &slot)
            };
            let Some(d) = d else { continue };
            let g_now = gen.load(Ordering::Acquire);
            if sent.as_ref().is_some_and(|(prev, g_sent)| prev.same_frame(&d) && *g_sent == g_now) {
                continue; // nothing new to say — no emit, no joiner, no spec change
            }
            let merged = union_specs(&specs.lock().unwrap());
            let out = if merged.is_empty() {
                d.clone()
            } else {
                let plan = goofi_view::plan(&merged, &d);
                goofi_core::reduce::reduce_for_view(&d, &plan)
            };
            reductions.fetch_add(1, Ordering::Relaxed);
            let bytes = Bytes::from(goofi_codec::encode(&out));
            let _ = tx.send(bytes); // Err only if all receivers are momentarily gone — harmless.
            sent = Some((d, g_now));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_view::{AxisReduce, DimCmp, ReduceMethod, ViewDtype};

    fn line_spec(width: usize) -> ViewSpec {
        ViewSpec {
            dtype: ViewDtype::Array,
            ndim: vec![(DimCmp::Le, 2)],
            dims: vec![],
            reduce: vec![AxisReduce { dim: -1, max: width, method: ReduceMethod::Envelope }],
        }
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
        use std::sync::atomic::AtomicBool;
        // Two connections view the SAME slot: exactly ONE reducer task exists, both receive
        // reduced frames, and it tears down on last-leave.
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let graph = Arc::new(Mutex::new(g));

        // Drive the graph continuously on a background thread (mirrors production spawn_tick),
        // so the reducer always has a fresh frame regardless of scheduling under load.
        let stop = Arc::new(AtomicBool::new(false));
        let ticker = {
            let (graph, stop) = (graph.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    graph.lock().unwrap().tick();
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        };

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

        stop.store(true, Ordering::Relaxed);
        ticker.join().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unchanged_frame_is_not_rebroadcast() {
        // The wire-level bug behind "ufreq 30, fps 30, ~33 drops/s": the sweep re-encoded and
        // re-sent the SAME published frame every 16 ms (62.5/s for a 30 Hz producer — measured
        // 375 frames / 180 fresh emits over 6 s on a live socket). A frame goes out when the
        // producer EMITS, when a subscriber JOINS, or when the spec union CHANGES — never
        // because a deadline elapsed.
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        // One published frame, then the graph goes quiet (nothing ticks it below).
        for _ in 0..200 {
            g.tick();
            if g.latest_frame(osc, "out").is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let first_frame = g.latest_frame(osc, "out").expect("the oscillator published a frame");
        let graph = Arc::new(Mutex::new(g));

        let reducers = SlotReducers::new(graph.clone());
        let key: SlotKey = (osc, "out".to_string());
        let c = reducers.new_conn();
        let mut rx = reducers.subscribe(key.clone(), c);

        // A joining subscriber is served the current frame without waiting for a fresh emit —
        // for a sparse producer (0.1 Hz) "wait for the next emit" would be a blank viewer.
        let joined = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
        assert!(joined.is_ok_and(|r| r.is_ok()), "a joining subscriber is served the current frame");

        // The producer emits nothing → the stream is SILENT. Ten sweep deadlines pass; the old
        // loop shipped ~10 copies of the same frame here.
        let silent = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await;
        assert!(silent.is_err(), "an unchanged frame was re-broadcast on a sweep deadline");

        // A spec change re-serves the CURRENT frame at its new reduction, even with no fresh
        // emit — a resized viewer of a slow producer must not stare at the stale size.
        reducers.set_specs(&key, c, vec![line_spec(64)]);
        let respec = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
        assert!(respec.is_ok_and(|r| r.is_ok()), "a spec change re-serves the current frame");
        let quiet = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await;
        assert!(quiet.is_err(), "the spec-change re-serve is one frame, not a new steady stream");

        // A connection that DROPPED a frame (write timeout, broadcast lag) re-offers: the skip
        // holds the last frame back until the next emit, so the drop must ask for it again —
        // "giving up costs one frame" is only true if the frame comes around again.
        reducers.reoffer(&key);
        let reoffered = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
        assert!(reoffered.is_ok_and(|r| r.is_ok()), "a re-offer serves the current frame again");

        // A fresh emit flows through within a sweep.
        for _ in 0..400 {
            std::thread::sleep(Duration::from_millis(2));
            let mut g = graph.lock().unwrap();
            g.tick();
            if !g.latest_frame(osc, "out").unwrap().same_frame(&first_frame) {
                break;
            }
        }
        assert!(
            !graph.lock().unwrap().latest_frame(osc, "out").unwrap().same_frame(&first_frame),
            "the oscillator emitted a fresh frame to flow"
        );
        let fresh = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
        assert!(fresh.is_ok_and(|r| r.is_ok()), "a fresh emit flows to the subscriber");

        reducers.unsubscribe(&key, c);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_starved_reducer_does_not_repay_the_deadlines_it_missed() {
        // The tick thread holds the graph mutex for the whole of an inline node's `process()`,
        // so a slow node starves this task past many 16 ms deadlines. A latest-wins sampler must
        // resume at the next deadline — repaying the backlog would fire a burst of full
        // reduce+encode+broadcast passes over ONE unchanged frame, on a worker that was just
        // unblocked, which is precisely when the process is least able to afford them.
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        // The oscillator paces itself off the wall clock, so tick it until it has published.
        for _ in 0..200 {
            g.tick();
            if g.latest_frame(osc, "out").is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(g.latest_frame(osc, "out").is_some(), "the oscillator published a frame to sample");
        let graph = Arc::new(Mutex::new(g));

        let reducers = SlotReducers::new(graph.clone());
        let key: SlotKey = (osc, "out".to_string());
        let c = reducers.new_conn();
        let _rx = reducers.subscribe(key.clone(), c);

        // ~20 missed deadlines while the lock is held, then sample a short window after release.
        let before = {
            let held = graph.lock().unwrap();
            std::thread::sleep(Duration::from_millis(320));
            let n = reducers.reductions(&key);
            drop(held);
            n
        };
        tokio::time::sleep(Duration::from_millis(48)).await;
        let burst = reducers.reductions(&key) - before;

        // 48 ms of resumed 16 ms sampling is 3-4 passes; the backlog would add ~20 on top. The
        // bound is one-sided on purpose — scheduler pressure can only make this SMALLER.
        assert!(burst <= 8, "{burst} passes in 48 ms after starvation: the missed deadlines were repaid");
        reducers.unsubscribe(&key, c);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reduction_cost_is_o1_in_subscriber_count() {
        use std::sync::atomic::AtomicBool;
        // The thalamus headline claim: N tabs on one slot cost ONE reduce+encode per frame, not
        // N. With many subscribers the reduce-pass count must stay bounded by WALL-CLOCK (one
        // ~16ms task), never multiplied by the subscriber count. Uses the internal counter — no
        // external load, so it can't leak load processes.
        const SUBS: usize = 50;
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let graph = Arc::new(Mutex::new(g));
        let stop = Arc::new(AtomicBool::new(false));
        let ticker = {
            let (graph, stop) = (graph.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    graph.lock().unwrap().tick();
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };

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
        stop.store(true, Ordering::Relaxed);
        ticker.join().unwrap();
    }
}
