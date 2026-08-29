//! Shared per-slot data reducers: ONE reduction per active `(node, slot)`, sized to the union of
//! every subscribing connection's `ViewSpec`s and fanned out over a broadcast.
//!
//! The SLOT owns the reducer's lifetime, not the socket count — it lives until its node leaves the
//! graph, because a closing socket is no evidence that a slot stopped being watched.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use axum::body::Bytes;
use goofi_engine::{Graph, Uid};
use goofi_view::ViewSpec;
use tokio::sync::broadcast;

/// The physical stream a reducer serves: a node uid + one of its output slot names.
pub type SlotKey = (Uid, String);
/// A unique id per `/data` connection, so its spec contribution can be tracked + removed.
pub type ConnId = u64;

/// Flatten every connection's `ViewSpec`s into the single list the planner merges; empty means
/// full-resolution passthrough.
fn union_specs(by_conn: &HashMap<ConnId, Vec<ViewSpec>>) -> Vec<ViewSpec> {
    by_conn.values().flatten().cloned().collect()
}

struct SlotReducer {
    specs: Arc<Mutex<HashMap<ConnId, Vec<ViewSpec>>>>,
    /// `Bytes` so the socket task forwards the SHARED buffer — a per-subscriber copy undoes dedup.
    tx: broadcast::Sender<Bytes>,
    stop: Arc<AtomicBool>,
    reductions: Arc<AtomicU64>,
    /// Serve generation: bumped on every spec change and subscriber join, so the loop re-serves
    /// the current frame once even when the producer has not emitted.
    gen: Arc<AtomicU64>,
    /// The latest RAW frame, pre-reduction — what serves a re-attaching viewer and `node snapshot`.
    latest: Arc<Mutex<Option<goofi_core::Data>>>,
}

impl Drop for SlotReducer {
    fn drop(&mut self) {
        // Signalled, never joined: the slot's own loop is what removes its entry, and a join
        // from there would be a join on itself.
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Every per-slot reducer, one per watched `(node, slot)`; cloneable, sharing one inner map.
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

    /// The slot's reducer, spawning its task if it has none.
    fn ensure<'a>(
        &self,
        map: &'a mut HashMap<SlotKey, SlotReducer>,
        key: &SlotKey,
    ) -> &'a mut SlotReducer {
        let slots = Arc::downgrade(&self.inner);
        map.entry(key.clone()).or_insert_with(|| {
            let reducer = SlotReducer {
                specs: Arc::new(Mutex::new(HashMap::new())),
                tx: broadcast::channel(16).0,
                stop: Arc::new(AtomicBool::new(false)),
                reductions: Arc::new(AtomicU64::new(0)),
                gen: Arc::new(AtomicU64::new(0)),
                latest: Arc::new(Mutex::new(None)),
            };
            spawn_reducer(key.clone(), &reducer, self.graph.clone(), slots);
            reducer
        })
    }

    /// Subscribe `conn` to `key`'s reduced stream, spawning the slot's reducer task if it has none.
    pub fn subscribe(&self, key: SlotKey, conn: ConnId) -> broadcast::Receiver<Bytes> {
        let mut map = self.inner.lock().unwrap();
        let reducer = self.ensure(&mut map, &key);
        reducer.specs.lock().unwrap().entry(conn).or_default();
        // The receiver MUST exist before the bump, or a sweep between the two statements
        // broadcasts the join-serve into a fan-out this joiner is not yet part of.
        let rx = reducer.tx.subscribe();
        reducer.gen.fetch_add(1, Ordering::Release);
        rx
    }

    /// The slot's latest RAW frame, once — never reduced, never a subscription. Asking also makes
    /// sure the slot's reducer runs, so a never-watched slot starts warming on the first ask.
    pub fn latest(&self, key: SlotKey) -> Option<goofi_core::Data> {
        // The `inner` guard is released before `latest` is taken, mirroring the reducer's order.
        let latest = {
            let mut map = self.inner.lock().unwrap();
            self.ensure(&mut map, &key).latest.clone()
        };
        let frame = latest.lock().unwrap().clone();
        frame
    }

    /// Replace `conn`'s declared specs for `key` (latest-wins). No-op if the slot is gone.
    pub fn set_specs(&self, key: &SlotKey, conn: ConnId, specs: Vec<ViewSpec>) {
        if let Some(r) = self.inner.lock().unwrap().get(key) {
            r.specs.lock().unwrap().insert(conn, specs);
            r.gen.fetch_add(1, Ordering::Release);
        }
    }

    /// Ask the slot to serve its current frame again, for a connection that DROPPED one; it
    /// reaches every subscriber, which is one duplicate frame on a rare path.
    pub fn reoffer(&self, key: &SlotKey) {
        if let Some(r) = self.inner.lock().unwrap().get(key) {
            r.gen.fetch_add(1, Ordering::Release);
        }
    }

    /// Withdraw `conn`'s contribution to `key`'s spec union; the reducer itself STAYS.
    pub fn unsubscribe(&self, key: &SlotKey, conn: ConnId) {
        if let Some(r) = self.inner.lock().unwrap().get(key) {
            r.specs.lock().unwrap().remove(&conn);
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

    /// Reduce+encode passes run for a slot so far (test/diagnostic).
    pub fn reductions(&self, key: &SlotKey) -> u64 {
        self.inner.lock().unwrap().get(key).map(|r| r.reductions.load(Ordering::Relaxed)).unwrap_or(0)
    }
}

/// How often the slot's subscribe address is re-derived from the graph: a service name carries the
/// node's GENERATION, so a restart re-homes the stream to a name this task has never opened.
pub const REHOME_INTERVAL: Duration = Duration::from_secs(1);

/// One end of a slot's data service: the subscriber, its iceoryx2 node, and the service name it
/// was opened on.
struct SlotFeed {
    subscriber: goofi_transport::ByteSubscriber,
    service: String,
    /// Declared LAST: fields drop in order, and a node dropped before its subscriber cannot remove
    /// its own directory.
    _node: goofi_transport::IoxNode,
}

/// Open a subscriber on `(uid, slot)`'s current output service, or `None` while the node is not
/// addressable; a miss is retried on the next re-home rather than being fatal.
fn open_feed(graph: &Mutex<Graph>, uid: Uid, slot: &str) -> Option<SlotFeed> {
    let service = {
        let g = graph.lock().unwrap();
        g.manifest(uid)?;
        crate::output_service_of(&g, uid, slot)
    };
    let node = goofi_transport::iox_node().ok()?;
    let subscriber = goofi_transport::open_output_subscriber(&node, &service).ok()?;
    Some(SlotFeed { _node: node, subscriber, service })
}

/// Spawn the per-slot reducer loop, on a PLAIN thread so any transport's thread can open one:
/// every ~16 ms take whatever the producer has published and — only when it emitted, a subscriber
/// joined, or the spec union changed — reduce, encode once, and broadcast to all. The sweep is a
/// sampling deadline, never a send cadence.
fn spawn_reducer(
    key: SlotKey,
    reducer: &SlotReducer,
    graph: Arc<Mutex<Graph>>,
    slots: Weak<Mutex<HashMap<SlotKey, SlotReducer>>>,
) {
    let (specs, tx) = (reducer.specs.clone(), reducer.tx.clone());
    let (reductions, gen) = (reducer.reductions.clone(), reducer.gen.clone());
    let (latest, stop) = (reducer.latest.clone(), reducer.stop.clone());
    let (uid, slot) = key.clone();
    std::thread::spawn(move || {
        let mut feed = open_feed(&graph, uid, &slot);
        let mut rehomed = std::time::Instant::now();
        // `served: None` means "never broadcast", which is what sends the first frame without a bump.
        let mut served: Option<u64> = None;
        loop {
            std::thread::sleep(Duration::from_millis(16));
            if stop.load(Ordering::Relaxed) {
                return;
            }
            if rehomed.elapsed() >= REHOME_INTERVAL {
                rehomed = std::time::Instant::now();
                let current = {
                    let g = graph.lock().unwrap();
                    g.manifest(uid).map(|_| crate::output_service_of(&g, uid, &slot))
                };
                let Some(current) = current else {
                    // The node left the graph: the reducer's ONE death.
                    if let Some(slots) = slots.upgrade() {
                        // Only THIS task's entry: an undo puts a removed node back at the same uid,
                        // and a viewer that re-subscribed since holds a reducer this one must keep.
                        let mut map = slots.lock().unwrap();
                        if map.get(&key).is_some_and(|r| Arc::ptr_eq(&r.specs, &specs)) {
                            map.remove(&key);
                        }
                    }
                    return;
                };
                if feed.as_ref().is_none_or(|f| f.service != current) {
                    feed = open_feed(&graph, uid, &slot);
                }
            }
            let mut fresh = false;
            if let Some(f) = &feed {
                while let Ok(Some(sample)) = f.subscriber.receive() {
                    if let Ok(frame) = goofi_codec::decode(sample.payload()) {
                        *latest.lock().unwrap() = Some(frame);
                        fresh = true;
                    }
                }
            }
            // Nobody is watching: the receives above still keep the cache warm for whoever returns.
            if specs.lock().unwrap().is_empty() {
                continue;
            }
            let Some(d) = latest.lock().unwrap().clone() else { continue };
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
    });
}
