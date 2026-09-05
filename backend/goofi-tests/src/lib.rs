//! The test harness — one live goofi, driven through [`Goofi::call`], the entry `/control` and
//! `/mcp` are transports over. Observation is [`Goofi::call`], [`Events`] and [`OutputProbe`].

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::AppState;
use serde_json::{json, Value};

pub use goofi_graph::Uid;

pub mod fixtures;
mod probe;
pub use probe::OutputProbe;
pub use serde_json::json as j;

/// How long [`Goofi::until`] waits before it calls a condition unmet. Only a FAILING assertion
/// pays it, so the number clears the slowest machine that runs the suite rather than the fastest.
const WAIT: Duration = Duration::from_secs(90);
/// How long [`Goofi::stays`] watches a negative.
const SETTLE: Duration = Duration::from_millis(250);

/// A running goofi: the graph, the runtime, the document, the status-drain worker.
pub struct Goofi {
    pub state: AppState,
    actor: String,
    patience: Duration,
    /// The handle that minted the mount, and the only one whose drop is the session's end.
    owner: bool,
}

/// A situation ends the way a process exits: every node stopped and waited for, so no test leaves
/// the engine alive under libtest's `exit`.
impl Drop for Goofi {
    fn drop(&mut self) {
        if self.owner {
            self.state.graph.lock().unwrap_or_else(|e| e.into_inner()).shutdown();
            self.state.release_mount();
        }
    }
}

impl Default for Goofi {
    fn default() -> Self {
        Self::new()
    }
}

impl Goofi {
    /// Boot one, with the status-drain worker that makes a node addressable.
    pub fn new() -> Goofi {
        Goofi::with_mode(goofi_bridge::Mode::default())
    }

    /// Boot a HEADLESS one — the layout rows are not registered.
    pub fn headless() -> Goofi {
        Goofi::with_mode(goofi_bridge::Mode { headless: true, demo: false })
    }

    /// Boot a PUBLIC one — no host-facing ops, and no audio engine behind the catalog.
    pub fn demo() -> Goofi {
        Goofi::with_mode(goofi_bridge::Mode { headless: false, demo: true })
    }

    fn with_mode(mode: goofi_bridge::Mode) -> Goofi {
        // Every test process is WALLED OFF from the real `~/.goofi` — a developer's own config
        // or session records must not reach an assertion. A test that scoped its own home first
        // keeps it.
        static HOME: std::sync::Once = std::sync::Once::new();
        HOME.call_once(|| {
            if std::env::var_os("GOOFI_HOME").is_none() {
                let dir = std::env::temp_dir().join(format!("goofi-test-home-{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir); // a crashed run under a recycled pid
                std::env::set_var("GOOFI_HOME", dir);
            }
            // The product spawns agents under the user's own shell; the SUITE spawns them under
            // one known POSIX shell, so a fish or a loud profile cannot fail a test command.
            #[cfg(unix)]
            std::env::set_var("SHELL", "/bin/sh");
            // Both under THIS binary's target dir, which goofi's own build pre-warmed: a shell's own
            // target must never reach the nested cargo, and the machine's temp dir is every checkout's.
            let target = std::env::current_exe().ok().and_then(|e| e.ancestors().nth(3).map(Path::to_path_buf));
            if std::env::var_os("GOOFI_BUILD_DIR").is_none() {
                if let Some(target) = &target {
                    std::env::set_var("GOOFI_BUILD_DIR", target.join("goofi-build"));
                }
            }
            let nested = target.unwrap_or_else(std::env::temp_dir).join("goofi-test-cargo-target");
            std::env::set_var("CARGO_TARGET_DIR", nested);
        });
        let state = AppState::new(mode, goofi_bridge::Clock::External);
        {
            let mut g = state.graph.lock().unwrap();
            fixtures::register(&mut g);
            // The child the audio engine scans a bundle in — the suite's own stand-in for the
            // binary — and no platform folder, so an installed plugin never reaches a test. A demo
            // registers no audio engine at all, so there is nothing to hand it.
            if !mode.demo {
                goofi_bridge::audio_engine(&mut g).set_vst3(scanner(), Vec::new());
            }
            // The engine's Python door, as the CLI hands it at boot; a machine with none scans a
            // `.py` file as unavailable, which is what a test that needs one then reports.
            if let Some(subproc) = find_python() {
                goofi_bridge::signal_engine(&mut g).set_python(goofi_signal::Python::new(subproc));
            }
            // The shipped tree is a root like any other: scanned at boot, as the CLI scans it.
            let patch = state.mount();
            goofi_bridge::rescan(&state, &mut g, &patch);
        }
        goofi_bridge::spawn_workers(&state);
        Goofi { state, actor: "test".into(), patience: WAIT, owner: true }
    }

    /// Boot one whose `/data` sockets probe on a short clock. Through [`Goofi::with_mode`], so
    /// the home wall stands here too.
    pub fn impatient() -> Goofi {
        let mut g = Goofi::with_mode(goofi_bridge::Mode::default());
        g.state.data_liveness = goofi_bridge::DataLiveness {
            ping_interval: Duration::from_millis(100),
            // Wide enough that a CI runner's scheduling stall cannot read as a dead peer.
            pong_deadline: Duration::from_millis(3000),
            send_timeout: Duration::from_millis(200),
        };
        g
    }

    /// A second client of the SAME instance, with its own undo stack — what two browser tabs are.
    pub fn client(&self, actor: &str) -> Goofi {
        Goofi { state: self.state.clone(), actor: actor.into(), patience: self.patience, owner: false }
    }

    /// Run an op and unwrap it; an unexpected refusal is a failure here.
    #[track_caller]
    pub fn call(&self, op: &str, payload: Value) -> Value {
        self.try_call(op, payload.clone())
            .unwrap_or_else(|e| panic!("{op} {payload} was refused: {e}"))
    }

    pub fn try_call(&self, op: &str, payload: Value) -> Result<Value, String> {
        self.state.call(op, payload, &self.actor)
    }

    /// Run an op that must be refused, and answer why.
    #[track_caller]
    pub fn refuse(&self, op: &str, payload: Value) -> String {
        match self.try_call(op, payload.clone()) {
            Err(e) => e,
            Ok(r) => panic!("{op} {payload} was accepted, answering {r}"),
        }
    }

    /// Add a node and answer its uid.
    #[track_caller]
    pub fn add(&self, type_name: &str) -> Uid {
        let r = self.call("node add", json!({ "type": type_name }));
        let hex = r["uid"].as_str().unwrap_or_else(|| panic!("add_node answered {r}"));
        Uid::from_hex(hex).unwrap_or_else(|| panic!("add_node answered a malformed uid {hex}"))
    }

    /// Set one param's literal value, answering `{value, error}` — the value as STORED, coerced
    /// to the param's declared type.
    #[track_caller]
    pub fn set_param(&self, uid: Uid, group: &str, name: &str, value: impl Into<Value>) -> Value {
        self.call(
            "node param edit",
            json!({ "node": hex(uid), "param": format!("{group}/{name}"), "value": value.into() }),
        )
    }

    /// Wire an output slot to an input slot.
    #[track_caller]
    pub fn link(&self, from: Uid, out: &str, to: Uid, into: &str) {
        self.call("link add", json!({ "from": ep(hex(from), out), "to": ep(hex(to), into) }));
    }

    /// The replicated projection every client mirrors.
    pub fn doc(&self) -> Value {
        self.call("session state", json!({}))
    }

    /// Bind a real server on a free port and answer its `ws://host:port` base.
    pub async fn serve(&self) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // As the CLI does: `local_url` derives from the address actually bound.
        self.state.set_bound(addr);
        let served = self.state.clone();
        tokio::spawn(async move { goofi_bridge::serve_app(listener, served, &[], false).await.unwrap() });
        format!("ws://{addr}")
    }

    /// As [`Goofi::serve`], but also serving a bundle on the fallback route.
    pub async fn serve_spa(&self, spa: goofi_bridge::Spa) -> String {
        self.serve_spa_with(spa, false).await
    }

    /// Serve the SPA with `/dev/*` open or shut — what `--debug` decides.
    pub async fn serve_spa_with(&self, spa: goofi_bridge::Spa, dev_routes: bool) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        self.state.set_bound(addr);
        let served = self.state.clone();
        tokio::spawn(async move {
            goofi_bridge::serve_app(listener, served, spa, dev_routes).await.unwrap()
        });
        format!("ws://{addr}")
    }

    /// Register a node type with a per-instance factory, as the CLI does for a Python node.
    pub fn register_dyn(
        &self,
        manifest: &'static goofi_node::NodeManifest,
        factory: goofi_signal_sdk::NodeFactory,
        tier: &'static goofi_node::IsolationCell,
    ) {
        goofi_bridge::register_dyn_type(&mut self.state.graph.lock().unwrap(), manifest, factory, tier);
    }

    /// The LEAF node uids in the replicated projection, sorted. One map carries every entity, so
    /// which kind a record is is a question about its type.
    pub fn nodes(&self) -> Vec<String> {
        self.records(|ty| ty != goofi_graph::subpatch::SCOPE_TYPE
            && goofi_graph::subpatch::boundary_type(ty).is_none())
    }

    /// The live sub-patch facades, sorted.
    pub fn instances(&self) -> Vec<String> {
        self.records(|ty| ty == goofi_graph::subpatch::SCOPE_TYPE)
    }

    /// The boundary ports of one scope, in the order the doc carries them.
    pub fn ports(&self, scope: &str) -> Vec<String> {
        let doc = self.doc();
        let Some(nodes) = doc["nodes"].as_object() else { return vec![] };
        nodes
            .iter()
            .filter(|(_, n)| n["scope"] == scope && goofi_graph::subpatch::boundary_type(n["type"].as_str().unwrap_or("")).is_some())
            .map(|(u, _)| u.clone())
            .collect()
    }

    /// The direct members of a scope, sorted — every record naming it.
    pub fn members(&self, scope: &str) -> Vec<String> {
        let doc = self.doc();
        let mut v: Vec<String> = doc["nodes"]
            .as_object()
            .map(|m| m.iter().filter(|(_, n)| n["scope"] == scope).map(|(u, _)| u.clone()).collect())
            .unwrap_or_default();
        v.sort();
        v
    }

    /// The `(node, slot)` a port's inner wire names, read from `links` as any other cable is.
    pub fn inner(&self, port: &str) -> Option<(String, String)> {
        let doc = self.doc();
        // A port now wears a cable on BOTH sides, so which one is "inner" is its direction's
        // answer: an In port FEEDS the scope, an Out port drains it.
        let inward = doc["nodes"][port]["type"].as_str()?.starts_with("In");
        doc["links"].as_array()?.iter().find_map(|l| {
            let (a, b) = (l["node_out"].as_str()?, l["node_in"].as_str()?);
            match inward {
                true if a == port => Some((b.to_string(), l["slot_in"].as_str()?.to_string())),
                false if b == port => Some((a.to_string(), l["slot_out"].as_str()?.to_string())),
                _ => None,
            }
        })
    }

    fn records(&self, want: impl Fn(&str) -> bool) -> Vec<String> {
        let doc = self.doc();
        let mut v: Vec<String> = doc["nodes"]
            .as_object()
            .map(|m| m.iter().filter(|(_, n)| want(n["type"].as_str().unwrap_or(""))).map(|(u, _)| u.clone()).collect())
            .unwrap_or_default();
        v.sort();
        v
    }

    /// Subscribe to the event broadcast — take it BEFORE the action, the channel keeps no history.
    pub fn events(&self) -> Events {
        Events { rx: self.state.events.subscribe() }
    }

    /// Open a subscriber on one output slot — the same door `/data` opens.
    #[track_caller]
    pub fn probe(&self, node: Uid, slot: &str) -> OutputProbe {
        OutputProbe::open(&self.state.graph.lock().unwrap(), node, slot)
    }

    /// Poll `f` until it answers `Some`, or fail naming `what`.
    #[track_caller]
    pub fn until<T>(&self, what: &str, mut f: impl FnMut(&Goofi) -> Option<T>) -> T {
        let deadline = Instant::now() + self.patience;
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

    /// Watch a NEGATIVE for a settle window; answers whether `holds` was true throughout.
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

    /// Wait until a node reports `Ready`: a `Control` sent before that is lost. A node that files
    /// an error instead fails NOW, wearing that error — a timeout would bury the diagnosis.
    #[track_caller]
    pub fn ready(&self, node: Uid) {
        let deadline = Instant::now() + self.patience;
        loop {
            match self.stage(node).as_str() {
                "ready" => return,
                "error" => panic!("{node} failed instead of reporting ready: {}",
                                  self.error(node).unwrap_or_default()),
                stage if Instant::now() >= deadline => {
                    panic!("timed out waiting for {node} to report ready (stage: {stage})")
                }
                _ => std::thread::sleep(Duration::from_millis(2)),
            }
        }
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
    /// The next event named `name`, skipping the others.
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
}

/// `target/<profile>/vst3scan`: the package's bin, which cargo builds beside every test.
fn scanner() -> PathBuf {
    let exe = std::env::current_exe().expect("a test has a path");
    exe.ancestors().nth(2).expect("target/<profile>/deps/<test>").join(format!("vst3scan{}", std::env::consts::EXE_SUFFIX))
}

/// Render `frames` on the audio engine's external clock and hand back what the device would get,
/// interleaved, with its channel count.
pub fn drive(g: &Goofi, frames: usize) -> (Vec<f32>, u16) {
    let mut graph = g.state.graph.lock().unwrap();
    goofi_bridge::audio_engine(&mut graph).drive(frames)
}

/// A uid as the wire spells it.
pub fn hex(u: Uid) -> String {
    u.to_string()
}

/// A link endpoint as the wire spells it: `uid/slot`.
pub fn ep(uid: impl std::fmt::Display, slot: impl std::fmt::Display) -> String {
    format!("{uid}/{slot}")
}

pub use goofi_bridge::doc::{GraphDoc, Patch};
pub use tokio_tungstenite::tungstenite::Message;

pub type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

/// A `/control` client — what a browser tab is: RPC replies, events, and the doc replica it feeds.
pub struct Client {
    pub ws: Ws,
    next_id: i64,
    actor: String,
    doc: GraphDoc,
}

impl Client {
    /// Connect and take the `hello` snapshot, which every client is sent unprompted.
    pub async fn connect(base: &str) -> (Client, Value) {
        Client::connect_as(base, "test").await
    }

    pub async fn connect_as(base: &str, actor: &str) -> (Client, Value) {
        let (ws, _) = tokio_tungstenite::connect_async(format!("{base}/control")).await.unwrap();
        let mut c = Client { ws, next_id: 1, actor: actor.into(), doc: GraphDoc::new() };
        let hello = c.text().await;
        (c, hello["payload"].clone())
    }

    /// This client's replica of the manager's document.
    pub fn doc(&self) -> &GraphDoc {
        &self.doc
    }

    /// Read frames until the replica satisfies `want`. `want` must be POSITIVE: an absence
    /// predicate is already true of the empty replica.
    pub async fn until_doc(&mut self, want: impl Fn(&GraphDoc) -> bool) {
        for _ in 0..200 {
            if want(&self.doc) {
                return;
            }
            self.text().await;
        }
        panic!("the replica never reached the state this test waited for");
    }

    /// Send an RPC and return its result, skipping the events interleaved with the reply.
    pub async fn call(&mut self, op: &str, payload: Value) -> Value {
        match self.try_call(op, payload.clone()).await {
            Ok(r) => r,
            Err(e) => panic!("{op} {payload} was refused: {e}"),
        }
    }

    pub async fn try_call(&mut self, op: &str, payload: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let req = json!({ "id": id, "op": op, "payload": payload, "actor": self.actor });
        self.ws.send(Message::Text(req.to_string().into())).await.unwrap();
        loop {
            let m = self.text().await;
            if m.get("id").and_then(Value::as_i64) == Some(id) {
                return match m.get("error") {
                    Some(e) => Err(e.as_str().map(str::to_owned).unwrap_or_else(|| e.to_string())),
                    None => Ok(m["result"].clone()),
                };
            }
        }
    }

    /// The next event named `name`, skipping the others.
    pub async fn event(&mut self, name: &str) -> Value {
        loop {
            let m = self.text().await;
            if m.get("event").and_then(Value::as_str) == Some(name) {
                return m["payload"].clone();
            }
        }
    }

    /// The next TEXT frame, as JSON, feeding the replica on the way past. A GAP panics; a stale
    /// patch does not.
    pub async fn text(&mut self) -> Value {
        loop {
            let Message::Text(t) = self.next().await else { continue };
            let v: Value = serde_json::from_str(t.as_str()).expect("an event is JSON");
            match v.get("event").and_then(Value::as_str) {
                Some("doc_state") => {
                    let p = &v["payload"];
                    self.doc.reset_to(p["v"].as_u64().expect("a version"), p["doc"].clone());
                }
                Some("doc_patch") => {
                    let p = &v["payload"];
                    let out = self.doc.apply_patch(
                        p["from"].as_u64().expect("a base version"),
                        p["v"].as_u64().expect("a version"),
                        &p["patch"],
                    );
                    if let Patch::Gap { from, at } = out {
                        panic!("a delta was lost: this replica is at v{at} and the next patch is from v{from}");
                    }
                }
                _ => {}
            }
            return v;
        }
    }

    /// The next BINARY frame, skipping text.
    pub async fn binary(&mut self) -> Vec<u8> {
        loop {
            if let Message::Binary(b) = self.next().await {
                return b.to_vec();
            }
        }
    }

    async fn next(&mut self) -> Message {
        tokio::time::timeout(WAIT, self.ws.next())
            .await
            .expect("the socket said nothing before the deadline")
            .expect("the stream ended")
            .expect("a websocket error")
    }

}

impl Client {
    /// Send a raw text frame — for a test driving the envelope itself rather than an op.
    pub async fn send(&mut self, text: String) {
        self.ws.send(Message::Text(text.into())).await.unwrap();
    }
}

/// The panel ids in an arrangement, as a replica reads them.
pub fn panels(doc: &GraphDoc) -> Vec<String> {
    panel_ids(&doc.to_json()["arrangement"])
}

/// One node's JSON in an arrangement, by id.
pub fn arrangement_node<'a>(arrangement: &'a Value, id: &str) -> Option<&'a Value> {
    fn down<'a>(n: &'a Value, id: &str) -> Option<&'a Value> {
        if n["id"] == id {
            return Some(n);
        }
        n["children"].as_array()?.iter().find_map(|k| down(k, id))
    }
    arrangement["tabs"].as_array()?.iter().find_map(|t| down(&t["root"], id))
}

/// Every panel id in an arrangement JSON, depth-first.
pub fn panel_ids(arrangement: &Value) -> Vec<String> {
    fn down(n: &Value, out: &mut Vec<String>) {
        if n["kind"] == "panel" {
            out.push(n["id"].as_str().unwrap_or_default().to_string());
        }
        for k in n["children"].as_array().into_iter().flatten() {
            down(k, out);
        }
    }
    let mut out = Vec::new();
    for t in arrangement["tabs"].as_array().into_iter().flatten() {
        down(&t["root"], &mut out);
    }
    out
}

/// A `/data` viewer — one subscriber on one (node, slot) stream.
pub struct Viewer {
    pub ws: Ws,
}

impl Viewer {
    pub async fn open(base: &str, node: &str, slot: &str) -> Viewer {
        let (ws, _) = tokio_tungstenite::connect_async(format!("{base}/data/{node}/{slot}"))
            .await
            .unwrap();
        Viewer { ws }
    }

    /// Publish this viewer's constraints inband.
    pub async fn view(&mut self, specs: Value) {
        self.ws.send(Message::Text(json!({ "op": "view", "specs": specs }).to_string().into()))
            .await
            .unwrap();
    }

    /// The next GOOF frame, raw.
    pub async fn frame(&mut self) -> Vec<u8> {
        loop {
            match tokio::time::timeout(WAIT, self.ws.next()).await {
                Ok(Some(Ok(Message::Binary(b)))) => return b.to_vec(),
                Ok(Some(Ok(_))) => {}
                other => panic!("the data socket stopped before a frame arrived: {other:?}"),
            }
        }
    }

    /// The next frame, decoded — panics on anything the codec will not take.
    pub async fn decoded(&mut self) -> goofi_core::Data {
        let raw = self.frame().await;
        goofi_codec::decode(&raw).expect("a decodable GOOF frame")
    }

    /// Read frames until one satisfies `want`.
    pub async fn until(&mut self, mut want: impl FnMut(&goofi_core::Data) -> bool) -> goofi_core::Data {
        let deadline = Instant::now() + WAIT;
        loop {
            let d = self.decoded().await;
            if want(&d) {
                return d;
            }
            assert!(Instant::now() < deadline, "no frame matched before the deadline");
        }
    }

    /// The close code the bridge answered with, for a subscription it refuses.
    pub async fn close_code(&mut self) -> Option<u16> {
        loop {
            match tokio::time::timeout(WAIT, self.ws.next()).await {
                Ok(Some(Ok(Message::Close(Some(f))))) => return Some(u16::from(f.code)),
                Ok(Some(Ok(_))) => continue,
                _ => return None,
            }
        }
    }
}

/// Poll until `f` holds or `limit` elapses, and answer whether it held.
pub async fn holds_within(limit: Duration, mut f: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    f()
}

/// The `host:port` inside a `ws://` base, for the HTTP half of the same server.
pub fn host(base: &str) -> &str {
    base.trim_start_matches("ws://")
}

/// One HTTP request over a raw `TcpStream`, answering `(status, head, body)`. The body stays
/// bytes because a `.gfi` is a zip.
pub async fn http(
    addr: &str,
    method: &str,
    path: &str,
    headers: &str,
    body: &[u8],
) -> (u16, String, Vec<u8>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    // The address is the Host unless the caller named one — two Host headers is a 400, and the
    // origin guard is only reachable at all from a request whose Host is not this loopback.
    let host = match headers.to_ascii_lowercase().contains("host:") {
        true => String::new(),
        false => format!("Host: {addr}\r\n"),
    };
    let head = format!(
        "{method} {path} HTTP/1.1\r\n{host}{headers}Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(WAIT, s.read_to_end(&mut raw))
        .await
        .expect("the endpoint answered before the deadline")
        .unwrap();
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("a well-formed HTTP reply");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, head, raw[split + 4..].to_vec())
}

/// One `goofi_exec` command line over MCP, answering the rendered text. Fails if the tool
/// reported an error.
pub async fn tool(addr: &str, command: &str) -> String {
    let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": { "name": "goofi_exec", "arguments": { "commands": [command] } } });
    let (status, _, body) =
        http(addr, "POST", "/mcp", "Content-Type: application/json\r\n", req.to_string().as_bytes())
            .await;
    assert_eq!(status, 200, "`{command}` answered {status}");
    let reply: Value = serde_json::from_slice(&body).expect("a JSON-RPC reply");
    assert_eq!(reply["result"]["isError"], json!(false), "`{command}` failed: {}", reply["result"]);
    reply["result"]["content"][0]["text"].as_str().unwrap().to_string()
}

/// One ARRAY frame's payload as f32s — the LE decode every scenario reads frames with.
pub fn f32s(d: &goofi_core::Data) -> Vec<f32> {
    let goofi_core::Value::Array(a) = d.value() else { panic!("not an array: {d:?}") };
    a.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// A STRING frame's text; `None` for any other kind.
pub fn text(d: &goofi_core::Data) -> Option<&str> {
    match d.value() {
        goofi_core::Value::Str(s) => Some(s),
        _ => None,
    }
}

/// An ARRAY frame's shape.
pub fn shape(d: &goofi_core::Data) -> Vec<usize> {
    let goofi_core::Value::Array(a) = d.value() else { panic!("not an array: {d:?}") };
    a.shape().to_vec()
}

/// A 1-D ARRAY frame over `values`, metadata-less.
pub fn frame(values: &[f32]) -> goofi_core::Data {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    goofi_core::Data::array_f32(vec![values.len()], bytes, goofi_core::Meta::empty()).unwrap()
}

/// Serializes a binary's Python-tier tests: every one of them spawns an interpreter.
static TIER: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The interpreter to spawn children with, plus the tier lock — held for the rest of the test.
pub struct Tier {
    pub py: String,
    _lock: std::sync::MutexGuard<'static, ()>,
}

/// A python with BOTH goofi and numpy, or a PANIC naming the fix — these never skip. The probe
/// strips `PYTHONPATH` as the real child spawn does, so a host one cannot give a false negative.
/// The venv location is goofi-init's — the one owner of the layout.
pub fn require_python() -> Tier {
    // A panicking test poisons the mutex; recover rather than cascade onto every sibling.
    let _lock = TIER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(py) = find_python() {
        return Tier { py, _lock };
    }
    panic!(
        "no python with goofi + numpy found (checked $GOOFI_SUBPROC_TEST_PYTHON, ./{}, python3, \
         python). Run `cargo run -p goofi-init`, which creates the venvs and installs the goofi \
         wheel into them.",
        goofi_init::GIL_VENV
    );
}

/// The python found once per test process — the lookup spawns an interpreter, and every harness
/// boot asks.
fn find_python() -> Option<String> {
    static FOUND: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    FOUND
        .get_or_init(|| {
            let venv = goofi_init::repo_root().join(goofi_init::GIL_VENV);
            let cands = std::env::var("GOOFI_SUBPROC_TEST_PYTHON")
                .into_iter()
                .chain(goofi_init::venv_python(&venv).map(|p| p.to_string_lossy().into_owned()))
                .chain(["python3".to_string(), "python".to_string()]);
            cands.into_iter().find(|py| {
                std::process::Command::new(py)
                    .arg("-c")
                    .arg("import goofi, numpy")
                    .env_remove("PYTHONPATH")
                    .env_remove("PYTHONHOME")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .is_ok_and(|s| s.success())
            })
        })
        .clone()
}

/// Write `source` into the patch's own `nodes_signal/` and refresh the library through the op —
/// the door a user's file takes. Answers the type name the palette now offers; a file that scans
/// as unavailable is a panic naming why.
pub fn install(g: &Goofi, file: &str, source: &str) -> String {
    install_all(g, &[(file, source)]).remove(0)
}

/// [`install`] for several `(file, source)` pairs under ONE refresh — the scan probes them in
/// parallel itself, one interpreter per file.
pub fn install_all(g: &Goofi, files: &[(&str, &str)]) -> Vec<String> {
    let dir = g.state.mount().join("nodes_signal");
    std::fs::create_dir_all(&dir).unwrap();
    let names: Vec<String> = files
        .iter()
        .map(|(file, source)| {
            let path = dir.join(file);
            std::fs::write(&path, source).unwrap();
            goofi_node::type_name_of(&path).unwrap_or_else(|| panic!("{file} is not a node file"))
        })
        .collect();
    g.call("library refresh", j!({}));
    let graph = g.state.graph.lock().unwrap();
    for ((file, _), name) in files.iter().zip(&names) {
        if let Some((_, reason)) = graph.unavailable_types().find(|(n, _)| goofi_node::bare(n) == name) {
            panic!("{file} scanned as unavailable: {reason}");
        }
    }
    names
}

/// The one-variable evaluator a modulation step needs: the freshest frame's first sample,
/// coerced to the target's own type — no interpreter, so a scenario runs in the default suite.
pub struct FirstVar;

impl goofi_node::ExprEvaluator for FirstVar {
    fn compile(&self, _source: &str) -> Result<goofi_node::Compiled, goofi_node::ExprError> {
        Ok(goofi_node::Compiled { id: 1 })
    }
    fn eval(
        &self,
        _id: goofi_node::BindingId,
        ctx: &goofi_node::EvalCtx<'_>,
    ) -> Result<goofi_core::Param, goofi_node::ExprError> {
        let value = ctx
            .locals
            .values()
            .flatten()
            .find_map(|local| match local {
                goofi_node::Local::Frame(d) => f32s(d).first().map(|v| *v as f64),
                goofi_node::Local::Value(p) => p.as_f64(),
            })
            .ok_or_else(|| goofi_node::ExprError("no local arrived".into()))?;
        match ctx.target {
            goofi_core::Param::Float { vmin, vmax, .. } => {
                Ok(goofi_core::Param::float(value, *vmin, *vmax))
            }
            goofi_core::Param::Bool { .. } | goofi_core::Param::Pulse => {
                Ok(goofi_core::Param::boolean(value >= 0.5))
            }
            other => Ok(other.clone()),
        }
    }
    fn release(&self, _id: goofi_node::BindingId) {}
}
