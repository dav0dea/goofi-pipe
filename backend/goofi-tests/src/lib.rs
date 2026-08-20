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

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::AppState;
use serde_json::{json, Value};

pub use goofi_engine::testing::OutputProbe;
pub use goofi_engine::Uid;
pub use serde_json::json as j;

/// How long [`Goofi::until`] waits before it calls a condition unmet. Generous, and it has to be:
/// a node reaches `Ready` on its own thread, a loaded machine running the whole suite in parallel
/// is the normal case rather than the exception, and cargo runs the test BINARIES in parallel too —
/// so a scenario here waits behind whatever `nodes_shipped` is doing, which is starting eight
/// interpreters that each import numba. Ten seconds was calibrated before this repo shipped a node
/// with a heavy dependency, and it made three different tests fail on three different runs. Thirty
/// then did the same, twice, on a 32-core machine that was merely also running a build — and CI's
/// runner has four cores, so the number has to clear the SLOWEST machine that runs this suite, not
/// the fastest.
///
/// Only a FAILING assertion pays this; a passing one returns as soon as its condition holds. That
/// asymmetry is why the answer to "is this generous enough?" is always yes.
const WAIT: Duration = Duration::from_secs(90);
/// How long [`Goofi::stays`] watches a negative. A bare "check once" holds trivially against a
/// runtime that has not got round to the thing yet.
const SETTLE: Duration = Duration::from_millis(250);

/// A running goofi: the graph, the runtime, the CRDT doc, the status-drain worker.
pub struct Goofi {
    pub state: AppState,
    session: String,
    patience: Duration,
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
        Goofi { state, session: "test".into(), patience: WAIT }
    }

    /// Boot one whose `/data` sockets probe on a short clock: fast enough that the suite never
    /// waits on a production deadline, slow enough that ordinary scheduler jitter cannot trip it.
    pub fn impatient() -> Goofi {
        let mut state = AppState::new();
        state.data_liveness = goofi_bridge::DataLiveness {
            ping_interval: Duration::from_millis(100),
            pong_deadline: Duration::from_millis(1000),
            send_timeout: Duration::from_millis(200),
        };
        goofi_bridge::spawn_stats(state.graph.clone(), state.events.clone(), 2);
        Goofi { state, session: "test".into(), patience: WAIT }
    }

    /// A second client of the SAME instance, with its own undo stack — what two browser tabs are.
    pub fn client(&self, session: &str) -> Goofi {
        Goofi { state: self.state.clone(), session: session.into(), patience: self.patience }
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

    /// Bind a real server on a free port and answer its `ws://host:port` base. For the suite that
    /// tests the TRANSPORT — the `hello` snapshot, the binary sync relay, `/data`, `/term`, `/mcp`,
    /// the Origin guard. Everything else drives [`Goofi::call`] and needs no socket at all.
    pub async fn serve(&self) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // As the CLI does once it knows the bound port: a URL a spawned harness is handed has to be
        // one this very process answers, which is the whole claim `/mcp/<instance>` makes.
        self.state.set_mcp_port(addr.port());
        let served = self.state.clone();
        tokio::spawn(async move { goofi_bridge::serve_app(listener, served, &[]).await.unwrap() });
        format!("ws://{addr}")
    }

    /// As [`Goofi::serve`], but also serving a bundle on the fallback route — so the page is a
    /// real route rather than an argued one. Pass `goofi_bridge::SPA` for the one that ships.
    pub async fn serve_spa(&self, spa: goofi_bridge::Spa) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        self.state.set_mcp_port(addr.port());
        let served = self.state.clone();
        tokio::spawn(async move { goofi_bridge::serve_app(listener, served, spa).await.unwrap() });
        format!("ws://{addr}")
    }

    /// Register a node type with a PER-INSTANCE factory — what the CLI does for a discovered
    /// Python node. The only way to cover a node whose second instance differs from its first,
    /// which is what makes a restart observable at all.
    pub fn register_dyn(
        &self,
        manifest: &'static goofi_node::NodeManifest,
        factory: goofi_node::discover::NodeFactory,
    ) {
        self.state.graph.lock().unwrap().register_dyn_type(manifest, factory);
    }

    /// The uids in the replicated projection, sorted.
    pub fn nodes(&self) -> Vec<String> {
        keys(&self.doc()["nodes"])
    }

    /// The live sub-patch scopes, sorted.
    pub fn instances(&self) -> Vec<String> {
        keys(&self.doc()["instances"])
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

fn keys(v: &Value) -> Vec<String> {
    let mut k: Vec<String> = v.as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default();
    k.sort();
    k
}

// ---------------------------------------------------------------------------
// The transport half: a real WebSocket client, for the suite that tests the wire
// ---------------------------------------------------------------------------

pub use goofi_bridge::doc::{GraphDoc, Patch};
pub use tokio_tungstenite::tungstenite::Message;

pub type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

/// A `/control` client — what a browser tab is. One socket carrying JSON: RPC replies, events, and
/// the `doc_state` / `doc_patch` events that keep its replica equal to the manager's document.
///
/// The replica is the CLIENT's, and every frame this type reads feeds it — including the frames
/// read while waiting for an RPC reply. A browser tab has no way to skip those, so neither does
/// this: a harness that dropped them would show a client which edits as permanently behind one
/// that only watches, and it would look like a product defect rather than a test one.
pub struct Client {
    pub ws: Ws,
    next_id: i64,
    session: String,
    doc: GraphDoc,
}

impl Client {
    /// Connect and take the `hello` snapshot, which every client is sent unprompted.
    pub async fn connect(base: &str) -> (Client, Value) {
        Client::connect_as(base, "test").await
    }

    pub async fn connect_as(base: &str, session: &str) -> (Client, Value) {
        let (ws, _) = tokio_tungstenite::connect_async(format!("{base}/control")).await.unwrap();
        let mut c = Client { ws, next_id: 1, session: session.into(), doc: GraphDoc::new() };
        let hello = c.text().await;
        (c, hello["payload"].clone())
    }

    /// This client's replica of the manager's document.
    pub fn doc(&self) -> &GraphDoc {
        &self.doc
    }

    /// Read frames until the replica satisfies `want`, or fail. Every read feeds the replica, so a
    /// client that is also issuing RPCs converges on the same terms as one that only listens.
    ///
    /// `want` must be POSITIVE about something the document will deliver: an "absence" predicate is
    /// already true of the empty replica, before a single event lands.
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
        let req = json!({ "id": id, "op": op, "payload": payload, "session": self.session });
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

    /// The next TEXT frame, as JSON — feeding the replica on the way past, whatever the caller was
    /// waiting for. A GAP panics: the events arrive in order on one socket, so a replica missing one
    /// is a defect, not a condition to tolerate. A STALE patch does not — see
    /// [`goofi_bridge::doc::Patch`].
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

/// One node's JSON in an arrangement, by id — what a scenario reads a panel's type or binding off.
pub fn arrangement_node<'a>(arrangement: &'a Value, id: &str) -> Option<&'a Value> {
    fn down<'a>(n: &'a Value, id: &str) -> Option<&'a Value> {
        if n["id"] == id {
            return Some(n);
        }
        n["children"].as_array()?.iter().find_map(|k| down(k, id))
    }
    arrangement["tabs"].as_array()?.iter().find_map(|t| down(&t["root"], id))
}

/// Every panel id in an arrangement JSON, depth-first. The `.gfi` section and the document root
/// share this ONE shape, so both are read here.
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

/// A `/data` viewer — one subscriber on one (node, slot) stream. Binary frames are GOOF; a text
/// frame is the inband ViewSpec a viewer publishes to say what it can draw.
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

    /// Publish this viewer's constraints inband. The bridge folds every viewer's specs against the
    /// real frame and reduces ONCE, on its own subscription to the producer.
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

    /// Read frames until one satisfies `want`. A stream carries whatever the producer had emitted
    /// before this viewer's spec was folded in, so the frame a test is about is rarely the first.
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

/// Poll until `f` holds or `limit` elapses, and answer whether it held. Asserting the PROPERTY
/// ("reclaimed by T") rather than a window is what keeps a liveness test stable under cargo's
/// parallel runner.
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

/// One HTTP request over a raw `TcpStream`, answering `(status, head, body)`.
///
/// Hand-rolled for two reasons. `Connection: close` makes a reply "read to EOF", which is smaller
/// than the dev-dependency an HTTP client crate would add; and an HTTP client crate cannot ask the
/// WebSocket-upgrade question the Origin guard has to be asked. The body stays BYTES — a `.gfi` is
/// a zip, and lossy-decoding it to a string would corrupt exactly what those tests are about.
pub async fn http(
    addr: &str,
    method: &str,
    path: &str,
    headers: &str,
    body: &[u8],
) -> (u16, String, Vec<u8>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\n{headers}Content-Length: {}\r\n\
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

/// One MCP `tools/call`, answering the tool's rendered text. Fails if the tool reported an error.
pub async fn tool(addr: &str, name: &str, args: Value) -> String {
    let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": { "name": name, "arguments": args } });
    let (status, _, body) =
        http(addr, "POST", "/mcp", "Content-Type: application/json\r\n", req.to_string().as_bytes())
            .await;
    assert_eq!(status, 200, "{name} answered {status}");
    let reply: Value = serde_json::from_slice(&body).expect("a JSON-RPC reply");
    assert_eq!(reply["result"]["isError"], json!(false), "{name} failed: {}", reply["result"]);
    reply["result"]["content"][0]["text"].as_str().unwrap().to_string()
}
