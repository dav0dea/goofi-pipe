//! Watching an asynchronous graph from a test.
//!
//! There is no tick to step, so "run the graph and look" is no longer a thing a test can do. Every
//! observation is now one of two shapes, and this module is both of them:
//!
//! - [`wait_for`] — a node's own state (a fault, a stage, an evaluated value) reaches the graph
//!   through its status service, so a test drains that service until what it is waiting for holds.
//! - [`OutputProbe`] — a node's DATA reaches the world through `/data` and nowhere else (§7), so a
//!   test subscribes to the producer's output service exactly as the reducer does.
//!
//! **Every waiting call takes `&mut Graph`, including the probe's.** The graph learns a node exists
//! by draining its status — §4's birth barrier — and a wire attaches over three phases that advance
//! on acks, so a probe polled against a graph nobody is draining watches a slot that was never
//! wired. Taking the graph is what makes that impossible to forget; a nested pair of deadline loops
//! (a `wait_for` whose closure calls a waiting probe method) is what it replaced, and it cost two
//! full timeouts per failure.
//!
//! Both are deliberately public rather than `#[cfg(test)]`: the same two shapes are what the bridge
//! and `goofi-python` suites need, and a second copy of a deadline loop is a second set of timing
//! bugs. They are test scaffolding all the same — nothing in the product calls them.
//!
//! **A probe is opened BEFORE the frame it is waiting for.** The data services carry
//! `history_size(0)` (§3.5: a link never replays a previous output), so a probe opened after the
//! producer emitted has missed it and will wait for the next one — which, for a node that emits
//! once, never comes.

use std::time::{Duration, Instant};

use goofi_core::Data;

use crate::runtime::{iox_node, IoxNode};
use crate::{Graph, Uid};

/// How long a wait may take before it is a failure. Generous on purpose: it can only make a
/// genuinely broken test slower, never make a passing one flakier, and these run beside a
/// free-running node on every other core.
const WAIT: Duration = Duration::from_secs(5);

/// How long "nothing happened" is given to happen anyway. An absence can only ever be checked
/// against a window, and this one is long enough for a node to have run many times.
const SETTLE: Duration = Duration::from_millis(250);

/// How long a poll sleeps between looks. Short enough that a fast assertion is not paced by it,
/// long enough that a waiting test is not itself a busy loop competing with the node it watches.
const POLL: Duration = Duration::from_millis(1);

/// Drain every node's status until `done` holds. Panics with `what` if it never does.
///
/// This is what replaced `g.tick()` in an assertion: a node reports its own state asynchronously,
/// so the graph learns it only when someone drains — and in production that someone is the bridge's
/// status worker, which is exactly the same call.
pub fn wait_for(g: &mut Graph, what: &str, mut done: impl FnMut(&mut Graph) -> bool) {
    let deadline = Instant::now() + WAIT;
    loop {
        g.drain_status();
        if done(g) {
            return;
        }
        if Instant::now() >= deadline {
            panic!("timed out waiting for {what}");
        }
        std::thread::sleep(POLL);
    }
}

/// Drain status for a settle window and answer whether `holds` was true throughout. The oracle for
/// a NEGATIVE: a node that must not run, an error that must not appear.
///
/// A bare "check once" cannot express either — it holds trivially against a graph that has not got
/// round to the thing yet, which is the failure mode an asynchronous runtime makes routine.
pub fn stays(g: &mut Graph, mut holds: impl FnMut(&mut Graph) -> bool) -> bool {
    let deadline = Instant::now() + SETTLE;
    while Instant::now() < deadline {
        g.drain_status();
        if !holds(g) {
            return false;
        }
        std::thread::sleep(POLL);
    }
    true
}

/// A subscriber on one output slot — a viewer, with no privileged path into the node (§7).
pub struct OutputProbe {
    subscriber: crate::runtime::ByteSubscriber,
    /// The newest frame seen so far. Kept because the subscriber's queue is one deep and
    /// latest-wins: a caller that looks twice must not have to race the producer for the frame it
    /// already saw.
    latest: std::cell::RefCell<Option<Data>>,
    /// Must outlive the subscriber built from it, so it is declared LAST — Rust drops a struct's
    /// fields in declaration order, and a node that drops before its ports cannot remove its own
    /// directory. See [`crate::runtime::IoxTransport`], which had the same fault.
    _node: IoxNode,
}

impl OutputProbe {
    /// Open a probe on `(uid, slot)`. Panics for a node or slot that does not exist, because a
    /// probe on a slot that cannot emit would answer "silent" forever and read as a passing test.
    pub fn open(g: &Graph, uid: Uid, slot: &str) -> OutputProbe {
        let manifest = g.manifest(uid).unwrap_or_else(|| panic!("no node {uid}"));
        assert!(
            manifest.outputs.iter().any(|o| o.name == slot),
            "`{}` declares no output slot `{slot}`",
            manifest.type_name,
        );
        let node = iox_node().expect("an iceoryx2 node for the probe");
        let subscriber = crate::runtime::open_output_subscriber(&node, &g.output_service_of(uid, slot))
            .expect("a subscriber on the producer's output service");
        OutputProbe { _node: node, subscriber, latest: std::cell::RefCell::new(None) }
    }

    /// Take everything waiting, keeping the newest. Answers whether anything arrived.
    fn poll(&self) -> bool {
        let mut got = false;
        while let Ok(Some(sample)) = self.subscriber.receive() {
            if let Ok(frame) = goofi_codec::decode(sample.payload()) {
                *self.latest.borrow_mut() = Some(frame);
                got = true;
            }
        }
        got
    }

    /// The newest frame seen so far, looking once and waiting for nothing. This is the read a
    /// [`wait_for`] or [`stays`] closure makes: those already hold the graph and already pump it,
    /// so a second deadline loop inside one would only make a failure take twice as long.
    pub fn latest(&self) -> Option<Data> {
        self.poll();
        self.latest.borrow().clone()
    }

    /// The newest frame this slot has emitted, waiting for a first one while the graph is pumped.
    /// `None` when it stayed silent for the whole window.
    pub fn frame(&self, g: &mut Graph) -> Option<Data> {
        let deadline = Instant::now() + WAIT;
        loop {
            g.drain_status();
            if let Some(frame) = self.latest() {
                return Some(frame);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(POLL);
        }
    }

    /// The newest frame, or a failure naming what was being waited for.
    pub fn expect_frame(&self, g: &mut Graph, what: &str) -> Data {
        self.frame(g).unwrap_or_else(|| panic!("timed out waiting for a frame: {what}"))
    }

    /// Wait for a frame satisfying `want`, and answer it.
    ///
    /// The shape most assertions take now that a node runs on its own clock: a param edit, a link
    /// or a globals change reaches the node asynchronously, so the very next frame may still be the
    /// one that was already in flight. Asserting on "the next frame" would be a race; asserting on
    /// "a frame that reads this way, within the window" is the property the test actually means.
    pub fn wait_until(&self, g: &mut Graph, what: &str, want: impl Fn(&Data) -> bool) -> Data {
        let deadline = Instant::now() + WAIT;
        loop {
            g.drain_status();
            if let Some(frame) = self.latest() {
                if want(&frame) {
                    return frame;
                }
            }
            if Instant::now() >= deadline {
                panic!("timed out waiting for a frame that {what}: last was {:?}", self.latest());
            }
            std::thread::sleep(POLL);
        }
    }

    /// Whether the slot emitted nothing at all within the settle window — the "a refused run emits
    /// nothing" oracle. Frames already taken do not count against it, so a probe may be used to
    /// watch a node start, stop, and stay stopped.
    ///
    /// **This is the one call the module's `history_size(0)` hazard degrades SILENTLY.** A probe
    /// opened after its producer emitted has missed that frame for ever: [`Self::frame`] and
    /// [`Self::wait_until`] then fail loudly at their deadline, but `silent` simply answers `true`
    /// — a false PASS, and one that reads as the property holding. Open the probe before the node
    /// can have run, or give the assertion a positive counterpart on the same probe (a node that
    /// is silent here and emits once it is fed), which is what every caller in this crate does.
    pub fn silent(&self, g: &mut Graph) -> bool {
        let deadline = Instant::now() + SETTLE;
        while Instant::now() < deadline {
            g.drain_status();
            if self.poll() {
                return false;
            }
            std::thread::sleep(POLL);
        }
        true
    }
}
