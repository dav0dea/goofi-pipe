//! The test harness — one live goofi, driven the way anything drives goofi.
//!
//! Every suite in `tests/` reaches the system through [`Goofi::call`], the same entry `/control`
//! and `/mcp` are transports over. That is the point: a test that reaches past it pins an
//! implementation detail, and this crate cannot reach past it — it is a separate crate and sees
//! only public API.
//!
//! Observation has exactly three doors, and no test needs a fourth:
//!   * [`Goofi::call`] — every op, including the `Surface::Internal` rows that exist for this;
//!   * [`Events`] — the broadcast a `/control` client hears;
//!   * [`OutputProbe`] — one subscriber on one output slot, the same door `/data` opens.

use std::time::{Duration, Instant};

use goofi_bridge::AppState;
use serde_json::{json, Value};

pub use goofi_engine::testing::OutputProbe;
pub use goofi_engine::Uid;
pub use serde_json::json as j;

/// How long [`Goofi::until`] waits before it calls a condition unmet. Generous: a node reaches
/// `Ready` on its own thread, and a loaded machine running the whole suite in parallel is the
/// normal case, not the exception.
const WAIT: Duration = Duration::from_secs(10);
/// How long [`Goofi::stays`] watches a negative. A bare "check once" holds trivially against a
/// runtime that has not got round to the thing yet.
const SETTLE: Duration = Duration::from_millis(250);

/// A running goofi: the graph, the runtime, the CRDT doc, the status-drain worker.
pub struct Goofi {
    pub state: AppState,
    session: String,
}

impl Default for Goofi {
    fn default() -> Self {
        Self::new()
    }
}

impl Goofi {
    /// Boot one. The status-drain worker runs, which is what makes a node addressable and what
    /// advances a wire's three-phase attach — without it nothing in the runtime ever completes.
    pub fn new() -> Goofi {
        let state = AppState::new();
        goofi_bridge::spawn_stats(state.graph.clone(), state.events.clone(), 2);
        Goofi { state, session: "test".into() }
    }

    /// A second client of the SAME instance, with its own undo stack — what two browser tabs are.
    pub fn client(&self, session: &str) -> Goofi {
        Goofi { state: self.state.clone(), session: session.into() }
    }

    /// Run an op and unwrap it. The common case: a test that expected a refusal says so with
    /// [`Goofi::refuse`], so an unexpected one is a failure here rather than a silent `Err`.
    #[track_caller]
    pub fn call(&self, op: &str, payload: Value) -> Value {
        self.try_call(op, payload.clone())
            .unwrap_or_else(|e| panic!("{op} {payload} was refused: {e}"))
    }

    pub fn try_call(&self, op: &str, payload: Value) -> Result<Value, String> {
        self.state.call(op, payload, &self.session)
    }

    /// Run an op that must be refused, and answer why.
    #[track_caller]
    pub fn refuse(&self, op: &str, payload: Value) -> String {
        match self.try_call(op, payload.clone()) {
            Err(e) => e,
            Ok(r) => panic!("{op} {payload} was accepted, answering {r}"),
        }
    }

    /// Add a node and answer its uid — three lines that appear in almost every test.
    #[track_caller]
    pub fn add(&self, type_name: &str) -> Uid {
        let r = self.call("add_node", json!({ "type": type_name }));
        let hex = r["uid"].as_str().unwrap_or_else(|| panic!("add_node answered {r}"));
        Uid::from_hex(hex).unwrap_or_else(|| panic!("add_node answered a malformed uid {hex}"))
    }

    /// Wire an output slot to an input slot.
    #[track_caller]
    pub fn link(&self, from: Uid, out: &str, to: Uid, into: &str) {
        self.call(
            "add_link",
            json!({ "node_out": hex(from), "slot_out": out, "node_in": hex(to), "slot_in": into }),
        );
    }

    /// The replicated projection every client mirrors — nodes, links, instances, globals,
    /// arrangement.
    pub fn doc(&self) -> Value {
        self.call("get_state", json!({}))
    }

    /// Subscribe to the event broadcast. Take this BEFORE the action that should emit: the channel
    /// keeps no history, so a receiver opened afterwards hears nothing and reads as a pass.
    pub fn events(&self) -> Events {
        Events { rx: self.state.events.subscribe() }
    }

    /// Open a subscriber on one output slot — the same door `/data` opens, and the only way to
    /// observe a node's output without a privileged path into the node.
    #[track_caller]
    pub fn probe(&self, node: Uid, slot: &str) -> OutputProbe {
        OutputProbe::open(&self.state.graph.lock().unwrap(), node, slot)
    }

    /// Poll `f` until it answers `Some`, or fail naming `what`. The runtime is asynchronous, so
    /// this — not a sleep, and not a single look — is how an integration test asserts a positive.
    #[track_caller]
    pub fn until<T>(&self, what: &str, mut f: impl FnMut(&Goofi) -> Option<T>) -> T {
        let deadline = Instant::now() + WAIT;
        loop {
            if let Some(v) = f(self) {
                return v;
            }
            if Instant::now() >= deadline {
                panic!("timed out waiting for {what}");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Watch a NEGATIVE for a settle window: a node that must not run, an error that must not
    /// appear. Answers whether `holds` was true throughout.
    pub fn stays(&self, mut holds: impl FnMut(&Goofi) -> bool) -> bool {
        let deadline = Instant::now() + SETTLE;
        while Instant::now() < deadline {
            if !holds(self) {
                return false;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        true
    }

    /// Wait until a node reports `Ready` — the birth barrier. A `Control` sent before a node's
    /// subscriber exists is simply lost, so a test that acts on a fresh node without this races it.
    #[track_caller]
    pub fn ready(&self, node: Uid) {
        self.until(&format!("{node} to report ready"), |g| {
            (g.stage(node) == "ready").then_some(())
        });
    }

    /// A node's runtime stage, as the status-drain worker filed it.
    pub fn stage(&self, node: Uid) -> String {
        let mut g = self.state.graph.lock().unwrap();
        g.drain_status();
        g.node_stage(node).to_string()
    }

    /// A node's standing error, if it has one.
    pub fn error(&self, node: Uid) -> Option<String> {
        let mut g = self.state.graph.lock().unwrap();
        g.drain_status();
        g.last_error(node).map(str::to_owned)
    }
}

/// The `/control` event broadcast, as a test consumes it.
pub struct Events {
    rx: tokio::sync::broadcast::Receiver<String>,
}

impl Events {
    /// The next event named `name`, skipping the others. Fails if none arrives — every event this
    /// suite waits on is emitted by an op it just called.
    #[track_caller]
    pub fn next(&mut self, name: &str) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            match self.rx.try_recv() {
                Ok(raw) => {
                    let v: Value = serde_json::from_str(&raw).expect("an event is JSON");
                    if v["event"] == name {
                        return v["payload"].clone();
                    }
                }
                Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(2)),
                Err(e) => panic!("no `{name}` event arrived: {e}"),
            }
        }
    }

    /// Answers whether an event named `name` arrives inside the settle window — the oracle for
    /// "this must NOT be broadcast".
    pub fn quiet(&mut self, name: &str) -> bool {
        let deadline = Instant::now() + SETTLE;
        while Instant::now() < deadline {
            if let Ok(raw) = self.rx.try_recv() {
                let v: Value = serde_json::from_str(&raw).expect("an event is JSON");
                if v["event"] == name {
                    return false;
                }
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        true
    }
}

/// A uid as the wire spells it.
pub fn hex(u: Uid) -> String {
    u.to_string()
}
