//! The isolated-node execution tier: a per-node worker thread that runs a detached
//! (`Isolation::Subprocess`) node OFF the tick thread, fed and drained through
//! latest-wins mailboxes — so a blocking backend (a subprocess iceoryx2 roundtrip, a
//! device read) can never stall the tick or the graph lock. See
//! `docs/superpowers/specs/2026-07-19-isolated-node-tier-design.md`.

use std::sync::{Condvar, Mutex};

/// A single-slot, latest-wins handoff: `post` overwrites any un-taken item (drop-oldest,
/// matching iceoryx2 and the no-queue delivery model). Carries a shutdown flag so a
/// blocked `wait` can be woken to exit.
// `allow(dead_code)`: exercised by the tests now; wired into the tick in Task 4.
#[allow(dead_code)]
pub(crate) struct Mailbox<T> {
    inner: Mutex<Slot<T>>,
    cv: Condvar,
}

struct Slot<T> {
    item: Option<T>,
    shutdown: bool,
}

#[allow(dead_code)]
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
