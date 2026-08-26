//! The test harness — one live goofi, driven through [`Goofi::call`], the entry `/control` and
//! `/mcp` are transports over. Observation is [`Goofi::call`], [`Events`] and [`OutputProbe`].

use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::AppState;
use serde_json::{json, Value};

pub use goofi_engine::testing::OutputProbe;
pub use goofi_engine::Uid;
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
}

impl Default for Goofi {
    fn default() -> Self {
        Self::new()
    }
}

impl Goofi {
    /// Boot one, with the status-drain worker that makes a node addressable.
    pub fn new() -> Goofi {
        let state = AppState::new();
        goofi_bridge::spawn_stats(state.graph.clone(), state.events.clone(), 2);
        Goofi { state, actor: "test".into(), patience: WAIT }
    }

    /// Boot one whose `/data` sockets probe on a short clock.
    pub fn impatient() -> Goofi {
        let mut state = AppState::new();
        state.data_liveness = goofi_bridge::DataLiveness {
            ping_interval: Duration::from_millis(100),
            pong_deadline: Duration::from_millis(1000),
            send_timeout: Duration::from_millis(200),
        };
        goofi_bridge::spawn_stats(state.graph.clone(), state.events.clone(), 2);
        Goofi { state, actor: "test".into(), patience: WAIT }
    }

    /// A second client of the SAME instance, with its own undo stack — what two browser tabs are.
    pub fn client(&self, actor: &str) -> Goofi {
        Goofi { state: self.state.clone(), actor: actor.into(), patience: self.patience }
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

    /// Set one param's literal value, answering it AS STORED — a literal is coerced to the
    /// param's declared type.
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
        self.call(
            "link add",
            json!({ "node_out": hex(from), "slot_out": out, "node_in": hex(to), "slot_in": into }),
        );
    }

    /// The replicated projection every client mirrors.
    pub fn doc(&self) -> Value {
        self.call("session state", json!({}))
    }

    /// Bind a real server on a free port and answer its `ws://host:port` base.
    pub async fn serve(&self) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // As the CLI does: a `/mcp/<instance>` URL must name the process that answers it.
        self.state.set_mcp_port(addr.port());
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
        self.state.set_mcp_port(addr.port());
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
        factory: goofi_node::discover::NodeFactory,
    ) {
        self.state.graph.lock().unwrap().register_dyn_type(manifest, factory);
    }

    /// The LEAF node uids in the replicated projection, sorted. One map carries every entity, so
    /// which kind a record is is a question about its type.
    pub fn nodes(&self) -> Vec<String> {
        self.records(|ty| ty != goofi_engine::subpatch::SCOPE_TYPE
            && goofi_engine::subpatch::boundary_type(ty).is_none())
    }

    /// The live sub-patch facades, sorted.
    pub fn instances(&self) -> Vec<String> {
        self.records(|ty| ty == goofi_engine::subpatch::SCOPE_TYPE)
    }

    /// The boundary ports of one scope, in the order the doc carries them.
    pub fn ports(&self, scope: &str) -> Vec<String> {
        let doc = self.doc();
        let Some(nodes) = doc["nodes"].as_object() else { return vec![] };
        nodes
            .iter()
            .filter(|(_, n)| n["scope"] == scope && goofi_engine::subpatch::boundary_type(n["type"].as_str().unwrap_or("")).is_some())
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

    /// Wait until a node reports `Ready`: a `Control` sent before that is lost.
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

/// A uid as the wire spells it.
pub fn hex(u: Uid) -> String {
    u.to_string()
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
