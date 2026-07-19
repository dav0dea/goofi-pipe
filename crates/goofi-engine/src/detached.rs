//! The isolated-node execution tier: a per-node worker thread that runs a detached
//! (`Isolation::Subprocess`) node OFF the tick thread, fed and drained through
//! latest-wins mailboxes — so a blocking backend (a subprocess iceoryx2 roundtrip, a
//! device read) can never stall the tick or the graph lock. See
//! `docs/superpowers/specs/2026-07-19-isolated-node-tier-design.md`.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use goofi_core::Data;
use goofi_node::{Node, NodeCtx, NodeManifest, ParamGroups};
use indexmap::IndexMap;

use crate::{execute_node, seed_node, UfreqMeter};

/// A single-slot, latest-wins handoff: `post` overwrites any un-taken item (drop-oldest,
/// matching iceoryx2 and the no-queue delivery model). Carries a shutdown flag so a
/// blocked `wait` can be woken to exit.
pub(crate) struct Mailbox<T> {
    inner: Mutex<Slot<T>>,
    cv: Condvar,
}

struct Slot<T> {
    item: Option<T>,
    shutdown: bool,
}

impl<T> Mailbox<T> {
    pub(crate) fn new() -> Mailbox<T> {
        Mailbox { inner: Mutex::new(Slot { item: None, shutdown: false }), cv: Condvar::new() }
    }

    /// Latest-wins: replace any pending item and wake a waiter.
    pub(crate) fn post(&self, item: T) {
        let mut g = self.inner.lock().unwrap();
        g.item = Some(item);
        self.cv.notify_one();
    }

    /// Non-blocking drain (the tick side).
    pub(crate) fn take(&self) -> Option<T> {
        self.inner.lock().unwrap().item.take()
    }

    /// Block until an item OR shutdown. Shutdown is checked FIRST, so a pending item is
    /// dropped on shutdown and the worker exits promptly.
    pub(crate) fn wait(&self) -> Option<T> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if g.shutdown {
                return None;
            }
            if let Some(t) = g.item.take() {
                return Some(t);
            }
            g = self.cv.wait(g).unwrap();
        }
    }

    pub(crate) fn shutdown(&self) {
        let mut g = self.inner.lock().unwrap();
        g.shutdown = true;
        self.cv.notify_all();
    }
}

/// One unit of work handed to a detached worker: a snapshot of the node's live inputs +
/// params + clock, taken by the tick when the node is due. Latest-wins in the inbox.
pub(crate) struct Job {
    pub inputs: IndexMap<&'static str, Option<Data>>,
    pub multis: IndexMap<&'static str, Vec<Data>>,
    pub params: ParamGroups,
    pub now: f64,
}

/// A completed run's result, drained by the tick and integrated exactly like an inline
/// node's outputs (propagated in Phase B; the error rides the node's error channel).
pub(crate) struct Done {
    pub outputs: IndexMap<&'static str, Option<Data>>,
    pub error: Option<String>,
}

/// A node executing on its own OS thread. The tick posts `Job`s (latest-wins) and drains
/// `Done`s; it never runs the node's `process()` inline, so a blocking backend (a
/// subprocess iceoryx2 roundtrip, a device read) can't stall the tick or the graph lock.
pub(crate) struct DetachedHandle {
    inbox: Arc<Mailbox<Job>>,
    outbox: Arc<Mailbox<Done>>,
    thread: Option<JoinHandle<()>>,
}

impl DetachedHandle {
    /// Spawn the worker, moving the boxed node onto it. `params0`/`ctx0` seed it off-tick.
    pub(crate) fn spawn(
        node: Box<dyn Node>,
        manifest: &'static NodeManifest,
        params0: ParamGroups,
        ctx0: NodeCtx,
    ) -> DetachedHandle {
        let inbox = Arc::new(Mailbox::new());
        let outbox = Arc::new(Mailbox::new());
        let (ib, ob) = (inbox.clone(), outbox.clone());
        let thread = std::thread::Builder::new()
            .name(format!("goofi-detached-{}", manifest.type_name))
            .spawn(move || worker(node, manifest, params0, ctx0, ib, ob))
            .expect("spawn detached worker");
        DetachedHandle { inbox, outbox, thread: Some(thread) }
    }

    /// Post fresh inputs (latest-wins — a still-pending job is overwritten).
    pub(crate) fn dispatch(&self, job: Job) {
        self.inbox.post(job);
    }

    /// Non-blocking drain of a completed run.
    pub(crate) fn take_output(&self) -> Option<Done> {
        self.outbox.take()
    }
}

impl Drop for DetachedHandle {
    fn drop(&mut self) {
        // Signal shutdown, then join. An idle worker (waiting on the inbox) exits
        // instantly; a busy one finishes its one in-flight `process()` first (bounded by
        // the backend's own timeout) — vastly better than the pre-fix every-tick freeze.
        self.inbox.shutdown();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The worker loop: seed off-tick, then run each posted `Job` through the shared
/// [`execute_node`] (identical index/ufreq stamping to the inline path) and post the
/// result. Exits on shutdown; the node drops here, so its `Drop` reaps any child process.
fn worker(
    mut node: Box<dyn Node>,
    manifest: &'static NodeManifest,
    params0: ParamGroups,
    mut ctx: NodeCtx,
    inbox: Arc<Mailbox<Job>>,
    outbox: Arc<Mailbox<Done>>,
) {
    let mut index_counters: HashMap<&'static str, u64> = HashMap::new();
    let mut ufreq_meter = UfreqMeter::default();
    let mut outputs = manifest.output_buffer();
    let mut last_outputs: IndexMap<&'static str, Data> = IndexMap::new();
    // Seed off-tick (its `setup` / first-tick spawn may block). A failure surfaces via an
    // unsolicited Done so the node border reddens like an inline bootstrap error.
    if let Some(e) = seed_node(&mut *node, &params0, &mut ctx) {
        outbox.post(Done { outputs: IndexMap::new(), error: Some(e) });
    }
    while let Some(job) = inbox.wait() {
        ctx.now = job.now;
        let err = execute_node(
            manifest,
            &mut node,
            &job.params,
            &job.inputs,
            &job.multis,
            &mut outputs,
            &mut last_outputs,
            &mut ctx,
            &mut index_counters,
            &mut ufreq_meter,
        );
        let emitted: IndexMap<&'static str, Option<Data>> =
            outputs.iter().map(|(k, v)| (*k, v.clone())).collect();
        outbox.post(Done { outputs: emitted, error: err });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn post_overwrites_latest_wins() {
        let m: Mailbox<i32> = Mailbox::new();
        m.post(1);
        m.post(2);
        assert_eq!(m.take(), Some(2), "newest wins, oldest dropped");
        assert_eq!(m.take(), None, "drained");
    }

    #[test]
    fn wait_blocks_until_post_then_returns_it() {
        let m = Arc::new(Mailbox::<i32>::new());
        let m2 = m.clone();
        let h = std::thread::spawn(move || m2.wait());
        // Give the waiter a moment to park, then post.
        std::thread::sleep(std::time::Duration::from_millis(20));
        m.post(42);
        assert_eq!(h.join().unwrap(), Some(42));
    }

    #[test]
    fn shutdown_wakes_a_waiter_with_none() {
        let m = Arc::new(Mailbox::<i32>::new());
        let m2 = m.clone();
        let h = std::thread::spawn(move || m2.wait());
        std::thread::sleep(std::time::Duration::from_millis(20));
        m.shutdown();
        assert_eq!(h.join().unwrap(), None, "shutdown returns None");
    }
}
