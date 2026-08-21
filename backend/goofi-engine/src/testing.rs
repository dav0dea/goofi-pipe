//! Watching an asynchronous graph from a test.
//!
//! A probe must be opened BEFORE the frame it waits for — the data services carry
//! `history_size(0)`, so a probe opened later has missed what the producer already emitted.

use std::time::{Duration, Instant};

use goofi_core::Data;

use crate::runtime::{iox_node, IoxNode};
use crate::{Graph, Uid};

/// How long a wait may take before it is a failure. Generous on purpose.
const WAIT: Duration = Duration::from_secs(5);

/// How long "nothing happened" is given to happen anyway.
const SETTLE: Duration = Duration::from_millis(250);

/// How long a poll sleeps between looks.
const POLL: Duration = Duration::from_millis(1);

/// Drain every node's status until `done` holds. Panics with `what` if it never does.
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
    /// latest-wins.
    latest: std::cell::RefCell<Option<Data>>,
    /// Must outlive the subscriber built from it, so it is declared LAST — Rust drops a struct's
    /// fields in declaration order.
    _node: IoxNode,
}

impl OutputProbe {
    /// Open a probe on `(uid, slot)`. Panics for a missing node or slot, which would otherwise
    /// answer "silent" forever and read as a passing test.
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

    /// The newest frame seen so far, looking once and waiting for nothing.
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

    /// Whether the slot emitted nothing at all within the settle window. Frames already taken do
    /// not count. A probe opened after its producer emitted answers `true` — a false PASS.
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
