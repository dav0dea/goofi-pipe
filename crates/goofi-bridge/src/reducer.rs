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
    /// Encoded reduced-frame fan-out to every subscribing connection.
    tx: broadcast::Sender<Arc<[u8]>>,
    /// The driving task (aborted on last-leave teardown).
    task: tokio::task::JoinHandle<()>,
    /// Count of reduce+encode passes — proves dedup in tests (one pass serves all subscribers).
    reductions: Arc<AtomicU64>,
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
    pub fn subscribe(&self, key: SlotKey, conn: ConnId) -> broadcast::Receiver<Arc<[u8]>> {
        let mut map = self.inner.lock().unwrap();
        let reducer = map.entry(key.clone()).or_insert_with(|| {
            let specs = Arc::new(Mutex::new(HashMap::new()));
            let (tx, _) = broadcast::channel(16);
            let reductions = Arc::new(AtomicU64::new(0));
            let task = spawn_reducer(
                key.clone(),
                specs.clone(),
                tx.clone(),
                self.graph.clone(),
                reductions.clone(),
            );
            SlotReducer { specs, tx, task, reductions }
        });
        reducer.specs.lock().unwrap().entry(conn).or_default();
        reducer.tx.subscribe()
    }

    /// Replace `conn`'s declared specs for `key` (latest-wins). No-op if the slot is gone.
    pub fn set_specs(&self, key: &SlotKey, conn: ConnId, specs: Vec<ViewSpec>) {
        if let Some(r) = self.inner.lock().unwrap().get(key) {
            r.specs.lock().unwrap().insert(conn, specs);
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

/// Spawn the per-slot reducer loop: every ~16 ms read the latest frame, reduce it to the union
/// of subscribers' specs (passthrough while none), encode once, and broadcast to all.
fn spawn_reducer(
    key: SlotKey,
    specs: Arc<Mutex<HashMap<ConnId, Vec<ViewSpec>>>>,
    tx: broadcast::Sender<Arc<[u8]>>,
    graph: Arc<Mutex<Graph>>,
    reductions: Arc<AtomicU64>,
) -> tokio::task::JoinHandle<()> {
    let (uid, slot) = key;
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(16));
        loop {
            ticker.tick().await;
            // Brief graph lock only for the latest-frame Arc clone; plan/reduce/encode run
            // off-lock so a large kHz/HD reduction never serializes against the scheduler tick.
            let d = {
                let g = graph.lock().unwrap();
                g.latest_frame(uid, &slot)
            };
            let Some(d) = d else { continue };
            let merged = union_specs(&specs.lock().unwrap());
            let out = if merged.is_empty() {
                d
            } else {
                let plan = goofi_view::plan(&merged, &d);
                goofi_core::reduce::reduce_for_view(&d, &plan)
            };
            reductions.fetch_add(1, Ordering::Relaxed);
            let bytes: Arc<[u8]> = goofi_codec::encode(&out).into();
            let _ = tx.send(bytes); // Err only if all receivers are momentarily gone — harmless.
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
}
