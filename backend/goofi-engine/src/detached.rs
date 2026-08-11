//! The isolated-node execution tier: a per-node worker thread that runs a detached
//! (`Isolation::Subprocess`) node OFF the tick thread, fed and drained through
//! latest-wins mailboxes — so a blocking backend (a subprocess iceoryx2 roundtrip, a
//! device read) can never stall the tick or the graph lock. See
//! `docs/superpowers/specs/2026-07-19-isolated-node-tier-design.md`.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

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
    ch: Arc<Channels>,
}

/// Everything a handle and its worker share — the WHOLE seam between the tick thread and the
/// worker, stated once so neither side can grow a channel the other does not know about.
struct Channels {
    inbox: Mailbox<Job>,
    outbox: Mailbox<Done>,
    /// How far the worker has got through its own bootstrap. Unlike an inline node — seeded
    /// synchronously before `insert_node_at` returns — a detached node's `setup()` runs off-tick
    /// and can take a while (a Python child is spawn + import + setup, measured in hundreds of
    /// milliseconds), so the editor shows a spinner until this reaches `Ready`.
    stage: std::sync::atomic::AtomicU8,
    /// A failed bootstrap, latched write-once by the worker before it reaches `STAGE_READY`.
    ///
    /// It cannot ride `outbox`: that is a latest-wins single slot, so the first successful job's
    /// `Done` overwrites an un-drained bootstrap failure and the node reports healthy though its
    /// `setup` failed — silently, when the failure was a param `seed_node` folded in rather than
    /// a raise the following `process()` repeats.
    ///
    /// Sticky for THIS worker's lifetime, because `setup` runs exactly once per worker and never
    /// re-runs: the fact stays true for as long as the instance it describes exists. That cannot
    /// strand a healthy node, because the only way to get a new instance is
    /// [`crate::Graph::restart_node`], which installs a whole new handle whose latch starts empty.
    boot_error: OnceLock<String>,
}

/// `DetachedHandle::stage` values. Ordered: a worker only moves forward.
pub(crate) const STAGE_CREATING: u8 = 0;
pub(crate) const STAGE_SETUP: u8 = 1;
pub(crate) const STAGE_READY: u8 = 2;

impl DetachedHandle {
    /// Spawn the worker, moving the boxed node onto it. `params0`/`ctx0` seed it off-tick.
    pub(crate) fn spawn(
        node: Box<dyn Node>,
        manifest: &'static NodeManifest,
        params0: ParamGroups,
        ctx0: NodeCtx,
    ) -> DetachedHandle {
        let ch = Arc::new(Channels {
            inbox: Mailbox::new(),
            outbox: Mailbox::new(),
            stage: std::sync::atomic::AtomicU8::new(STAGE_CREATING),
            boot_error: OnceLock::new(),
        });
        let theirs = ch.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("goofi-detached-{}", manifest.type_name))
            .spawn(move || worker(node, manifest, params0, ctx0, theirs));
        if let Err(e) = spawned {
            // Thread creation fails under resource pressure — and this is called under the graph
            // mutex, so panicking here poisons it and takes the whole control plane down with one
            // node. Report it the way a failed bootstrap is already reported: latch the error and
            // publish READY, which is this type's terminal state (see `stage`'s Acquire pairing —
            // a reader that sees READY also sees the latch, so this can never read as healthy).
            let _ = ch.boot_error.set(format!("could not start worker thread: {e}"));
            ch.stage.store(STAGE_READY, std::sync::atomic::Ordering::Release);
        }
        DetachedHandle { ch }
    }

    /// How far the worker's bootstrap has got (`STAGE_*`).
    pub(crate) fn stage(&self) -> u8 {
        // Acquire, paired with the worker's Release store of `STAGE_READY`: a reader that sees
        // READY also sees the `boot_error` latch, so a failed bootstrap can never be read as
        // "ready, no error" in the window between the two writes.
        self.ch.stage.load(std::sync::atomic::Ordering::Acquire)
    }

    /// This worker's bootstrap failure, if its `setup` failed. See `Channels::boot_error`.
    pub(crate) fn boot_error(&self) -> Option<&str> {
        self.ch.boot_error.get().map(String::as_str)
    }

    /// Post fresh inputs (latest-wins — a still-pending job is overwritten).
    pub(crate) fn dispatch(&self, job: Job) {
        self.ch.inbox.post(job);
    }

    /// Non-blocking drain of a completed run.
    pub(crate) fn take_output(&self) -> Option<Done> {
        self.ch.outbox.take()
    }
}

impl Drop for DetachedHandle {
    fn drop(&mut self) {
        // Signal only — never wait. The worker checks shutdown at `wait()`, so it observes this
        // BETWEEN jobs; a worker parked inside a blocking `process()` (a subprocess roundtrip
        // waits out its 10s timeout) would otherwise hold up whoever dropped the handle. Every
        // teardown path — remove_node, clear, undo-of-add, load, restart — runs under the graph
        // mutex, so waiting here freezes the tick, every viewer and every other RPC. The worker
        // owns the boxed node and drops it on its own thread, which reaps any child process.
        self.ch.inbox.shutdown();
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
    ch: Arc<Channels>,
) {
    let mut index_counters: HashMap<&'static str, u64> = HashMap::new();
    let mut ufreq_meter = UfreqMeter::default();
    let mut outputs = manifest.output_buffer();
    let mut last_outputs: IndexMap<&'static str, Data> = IndexMap::new();
    // Seed off-tick (its `setup` / first-tick spawn may block). A failure is LATCHED, not posted
    // to `outbox` — the outbox is a per-tick latest-wins slot and would lose it (see `boot_error`).
    ch.stage.store(STAGE_SETUP, std::sync::atomic::Ordering::Relaxed);
    if let Some(e) = seed_node(&mut *node, &params0, &mut ctx) {
        ch.boot_error.set(e).expect("the bootstrap latch is written once, by this thread only");
    }
    // Bootstrapped either way — a failed setup reports through the latch, not by leaving the node
    // spinning forever. Release, so a reader that sees READY sees the latch (see `stage()`).
    ch.stage.store(STAGE_READY, std::sync::atomic::Ordering::Release);
    while let Some(job) = ch.inbox.wait() {
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
        ch.outbox.post(Done { outputs: emitted, error: err });
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
