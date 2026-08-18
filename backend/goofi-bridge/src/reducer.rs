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
