//! End-to-end protocol test: a real WebSocket client drives the bridge exactly
//! as the frontend would — receives `hello`, lists nodes, adds a node (and gets
//! the `node_added` broadcast), then subscribes to the data plane and receives a
//! decodable GOOF frame. Proves the M1 vertical slice (engine + control + data).

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::{serve_app, spawn_stats, AppState};
use goofi_view::Reducible; // shape()/ndim() accessors on a decoded frame
use serde_json::{json, Value};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

// Read leaves through the generic CRDT reader (the typed getters were removed). A whole-number
// param comes back as an integer from `to_json`, so numeric reads compare via `as_f64`.
fn doc_node_pos(doc: &goofi_bridge::crdt::GraphDoc, uid: &str) -> Option<[f64; 2]> {
    let x = doc.read_at(&["nodes", uid, "pos", "x"])?.as_f64()?;
    let y = doc.read_at(&["nodes", uid, "pos", "y"])?.as_f64()?;
    Some([x, y])
}

async fn start_server() -> String {
    start_server_with_state().await.0
}

/// Like [`start_server`], but hands the state back as well — a test that reaches for the workspace
/// mount has to read it from the very `AppState` the server is answering from, since a load
/// REPLACES the path and a copy taken beforehand would name the mount that load just released.
async fn start_server_with_state() -> (String, AppState) {
    let state = AppState::new();
    spawn_stats(state.graph.clone(), state.events.clone(), 2);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = state.clone();
    tokio::spawn(async move {
        serve_app(listener, served, None).await.unwrap();
    });
    (format!("ws://{addr}"), state)
}

// A runtime type registered before serving — stands in for a discovered Python
// node (the CLI's `register_python` does exactly this against the live graph).
static SERVE_PARAMS: &[goofi_node::ParamDecl] = &[];
fn stub_factory() -> Box<dyn goofi_node::Node> {
    unreachable!("list_nodes never instantiates")
}
static SERVE_OUT: &[goofi_node::OutputDecl] = &[goofi_node::OutputDecl {
    name: "out",
    kind: goofi_core::SlotType::Array,
}];
static SERVE_MANIFEST: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "DiscoveredPyNode",
    category: "python",
    doc: "runtime type registered before serving",
    inputs: &[],
    outputs: SERVE_OUT,
    params: SERVE_PARAMS,
    isolation: goofi_node::Isolation::InProcess,
    producer: true,
    factory: stub_factory,
};

async fn start_server_with_runtime_type() -> String {
    let state = AppState::new();
    state
        .graph
        .lock()
        .unwrap()
        .register_dyn_type(&SERVE_MANIFEST, Box::new(|_| unreachable!()));
    spawn_stats(state.graph.clone(), state.events.clone(), 2);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    format!("ws://{addr}")
}

async fn recv_text(ws: &mut Ws) -> Value {
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("recv timed out")
            .expect("stream ended")
            .expect("ws error");
        if let Message::Text(t) = msg {
            return serde_json::from_str(t.as_str()).expect("json");
        }
    }
}

/// Receive the next BINARY frame (skipping text), with a timeout.
async fn recv_binary(ws: &mut Ws) -> Vec<u8> {
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("recv timed out")
            .expect("stream ended")
            .expect("ws error");
        if let Message::Binary(b) = msg {
            return b.to_vec();
        }
    }
}

/// Send an RPC and return the reply for its id (skipping interleaved events).
async fn call(ws: &mut Ws, id: i64, op: &str, payload: Value) -> Value {
    ws.send(Message::Text(
        json!({ "id": id, "op": op, "payload": payload }).to_string(),
    ))
    .await
    .unwrap();
    loop {
        let m = recv_text(ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(id) {
            return m;
        }
    }
}

/// Sync a FRESH CRDT replica from the server over `ws` and drain binary sync frames until
/// `ready(&doc)` holds, returning the replica for forest/graph reads. The structural broadcast
/// events (`subpatch_changed`/`node_removed`) are retired — the forest reaches clients via the doc,
/// so tests read it here (the pattern `leaf_write_expression`/`connecting_to_a_boundary_…` use). A
/// fresh replica advertises an empty state vector, so the server's `sync_hello` reply is the COMPLETE
/// current doc; `ready` is always satisfiable once the preceding RPC's effect has landed.
async fn sync_replica(ws: &mut Ws, ready: impl Fn(&goofi_bridge::crdt::GraphDoc) -> bool) -> goofi_bridge::crdt::GraphDoc {
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};
    let mut doc = GraphDoc::new();
    ws.send(Message::Binary(doc.sync_hello().into())).await.unwrap();
    for _ in 0..60 {
        if let Some(m) = SyncMsg::decode(&recv_binary(ws).await) {
            doc.on_sync(m);
        }
        if ready(&doc) {
            break;
        }
    }
    doc
}

fn panels(doc: &goofi_bridge::crdt::GraphDoc) -> Vec<String> {
    doc.to_json()["arrangement"]
        .as_object()
        .map(|m| {
            m.iter().filter(|(_, e)| e["kind"] == json!("panel")).map(|(id, _)| id.clone()).collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn a_layout_op_reaches_a_peers_replica_through_the_doc() {
    // Layout used to be client-owned: a peer learned an arrangement only on `hello`. As the fifth
    // doc root it rides the SAME delta broadcast as a node add — a LAYOUT-ONLY change ships, which
    // is what makes the frontend's parallel write authority removable at all.
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};
    let base = start_server().await;
    let (mut a, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut a).await;
    let (mut b, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut b).await;

    // B holds a replica and never asks again, so anything it learns below ARRIVED as a broadcast.
    let mut peer = GraphDoc::new();
    b.send(Message::Binary(peer.sync_hello())).await.unwrap();
    for _ in 0..10 {
        if let Some(m) = SyncMsg::decode(&recv_binary(&mut b).await) {
            peer.on_sync(m);
        }
        if !panels(&peer).is_empty() {
            break;
        }
    }
    let panel = panels(&peer).first().cloned().expect("the default page's one panel");

    let fresh = call(
        &mut a,
        1,
        "page_split_panel",
        json!({ "page": "Layout", "panel": panel, "direction": "row", "ratio": 0.5 }),
    )
    .await["result"]
        .as_str()
        .expect("the new panel's uid")
        .to_string();

    for _ in 0..20 {
        if let Some(m) = SyncMsg::decode(&recv_binary(&mut b).await) {
            peer.on_sync(m);
        }
        if peer.read_at(&["arrangement", fresh.as_str()]).is_some() {
            break;
        }
    }
    assert_eq!(
        peer.read_at(&["arrangement", fresh.as_str(), "panel_type"]),
        Some(json!("empty")),
        "the peer converged on the split, and a split births an EMPTY panel"
    );
    assert_eq!(panels(&peer).len(), 2);

    // The read a caller navigates by: the tree, which names the page, the panel and its type.
    let tree = call(&mut a, 2, "inspect_layout", json!({})).await["result"]["text"]
        .as_str()
        .expect("inspect_layout answers text")
        .to_string();
    assert!(tree.contains("Layout") && tree.contains(&fresh), "the tree names the page and the panel: {tree}");
    assert!(tree.contains("empty"), "…and what a fresh split birthed there: {tree}");
    // A page is addressed by NAME, so an unknown one has to say which exist.
    let miss = call(&mut a, 3, "inspect_layout", json!({ "page": "Nope" })).await;
    assert!(miss["error"].as_str().is_some_and(|e| e.contains("Layout")), "{miss}");
}

#[tokio::test]
async fn runtime_registered_type_reaches_the_palette_over_the_wire() {
    // The full serving path a browser sees: a runtime type registered into the
    // live graph (as the CLI's node scan does) surfaces via list_nodes.
    let base = start_server_with_runtime_type().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let hello = recv_text(&mut ws).await;
    assert_eq!(hello["event"], "hello");

    let reply = call(&mut ws, 1, "list_nodes", json!({})).await;
    let types = reply["result"]["types"].as_array().expect("types array");
    assert!(
        types.iter().any(|t| t["type"] == "DiscoveredPyNode"),
        "runtime-registered type must appear in the palette; got {:?}",
        types.iter().map(|t| &t["type"]).collect::<Vec<_>>()
    );
    // Its category rides along so the palette can group it.
    let entry = types.iter().find(|t| t["type"] == "DiscoveredPyNode").unwrap();
    assert_eq!(entry["category"], "python");
}

#[tokio::test]
async fn control_and_data_plane_end_to_end() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();

    // 1. hello: protocol_version + instance_id + an empty runtime overlay.
    let hello = recv_text(&mut ws).await;
    assert_eq!(hello["event"], "hello");
    assert_eq!(hello["payload"]["protocol_version"], 1);
    assert!(hello["payload"]["instance_id"].is_string());
    // No graph projection rides the snapshot — structure is the doc's alone.
    assert!(hello["payload"]["runtime"].as_object().is_some_and(|m| m.is_empty()));
    // The backend advertises the pillars it hosts (signal-only for now).
    assert_eq!(hello["payload"]["pillars"], json!(["signal"]));

    // 2. list_nodes returns the catalog incl. Oscillator.
    ws.send(Message::Text(
        json!({"id": 1, "op": "list_nodes", "payload": {}})
            .to_string(),
    ))
    .await
    .unwrap();
    let reply = loop {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(1) {
            break m;
        }
    };
    let types = reply["result"]["types"].as_array().unwrap();
    // All native node types must survive linkage into a dependent binary (guards
    // against inventory registrations being dropped / stale-build confusion).
    for expected in ["Oscillator", "Buffer"] {
        assert!(
            types.iter().any(|t| t["type"] == expected),
            "catalog must contain {expected}; got {:?}",
            types.iter().map(|t| &t["type"]).collect::<Vec<_>>()
        );
    }
    // Test-only nodes are hidden.
    assert!(!types.iter().any(|t| t["type"] == "_TestEcho"));

    // 3. add_node -> uid result + node_added announcement.
    ws.send(Message::Text(
        json!({"id": 2, "op": "add_node", "payload": {"type": "Oscillator", "category": "inputs", "pos": [10.0, 20.0]}})
            .to_string(),
    ))
    .await
    .unwrap();

    let mut uid: Option<String> = None;
    let mut saw_added = false;
    for _ in 0..6 {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(2) {
            uid = Some(m["result"]["uid"].as_str().unwrap().to_string());
        } else if m["event"] == "node_added" {
            saw_added = true;
            // A bare uid announcement — the node's type/pos/params reach clients via the doc.
            assert_eq!(m["payload"].as_object().map(|o| o.len()), Some(1));
            assert!(m["payload"]["uid"].is_string());
        }
        if uid.is_some() && saw_added {
            break;
        }
    }
    let uid = uid.expect("add_node returned a uid");
    assert!(saw_added, "node_added must be broadcast");

    // 4. data plane: subscribe and receive a decodable GOOF frame.
    let (mut data, _) = connect_async(format!("{base}/data/{uid}/out"))
        .await
        .unwrap();
    let frame = loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), data.next())
            .await
            .expect("data frame timed out")
            .expect("stream ended")
            .expect("ws error");
        if let Message::Binary(b) = msg {
            break b;
        }
    };
    assert_eq!(&frame[0..4], b"GOOF", "magic");
    assert_eq!(frame[4], 2, "version");
    assert_eq!(frame[5], 0, "dtype tag ARRAY");

    // 5. unknown node/slot -> terminal close 4004.
    let (mut bad, _) = connect_async(format!("{base}/data/{uid}/nope"))
        .await
        .unwrap();
    let closed = loop {
        match tokio::time::timeout(Duration::from_secs(5), bad.next()).await {
            Ok(Some(Ok(Message::Close(Some(f))))) => break Some(u16::from(f.code)),
            Ok(Some(Ok(_))) => continue,
            _ => break None,
        }
    };
    assert_eq!(closed, Some(4004), "unknown slot closes with 4004");
}

#[tokio::test]
async fn native_chain_streams_frames_over_the_data_plane() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);

    // Bound the buffer so it fills quickly.
    call(
        &mut ws,
        3,
        "update_param",
        json!({ "node": buf, "group": "buffer", "name": "size", "value": 128 }),
    )
    .await;
    // Oscillator -> Buffer
    call(
        &mut ws,
        4,
        "add_link",
        json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" }),
    )
    .await;

    // The buffered output streams real array frames through the data plane.
    let (mut data, _) = connect_async(format!("{base}/data/{buf}/out"))
        .await
        .unwrap();
    let frame = loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), data.next())
            .await
            .expect("data frame timed out")
            .expect("stream ended")
            .expect("ws error");
        if let Message::Binary(b) = msg {
            break b;
        }
    };
    assert_eq!(&frame[0..4], b"GOOF");
    assert_eq!(frame[4], 2, "version");
    assert_eq!(frame[5], 0, "Buffer emits an ARRAY");
    let body_len = u32::from_le_bytes(frame[10..14].try_into().unwrap());
    assert!(body_len > 8, "non-trivial buffered body ({body_len} bytes)");
}

#[tokio::test]
async fn data_plane_sustains_streaming_over_a_window() {
    // Stability/throughput smoke: a live Oscillator→Buffer chain must keep delivering frames
    // over a wall-clock window (not just one), proving the node threads + data plane sustain streaming
    // without stalling. Loose lower bound so it's not CI-timing-flaky; the measured rate is
    // logged for a latency/throughput read.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    call(&mut ws, 3, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" })).await;

    let (mut data, _) = connect_async(format!("{base}/data/{buf}/out")).await.unwrap();
    let window = Duration::from_millis(400);
    let mut frames = 0u32;
    let deadline = tokio::time::Instant::now() + window;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), data.next()).await {
            Ok(Some(Ok(Message::Binary(b)))) if &b[0..4] == b"GOOF" => frames += 1,
            Ok(Some(Ok(_))) => {}
            _ => break,
        }
    }
    eprintln!("data-plane throughput: {frames} frames in {}ms", window.as_millis());
    assert!(frames >= 3, "the data plane must sustain streaming (got {frames} frames in {window:?})");
}

#[tokio::test]
async fn data_plane_reduces_to_the_declared_viewspec() {
    // A viewer declares its need inband on the /data socket (line: array, ≤2-D, envelope
    // dim -1 → 32). The bridge reduces the buffered frame ONCE for this connection and
    // stamps `meta.reduced` — proving reduction runs on the data plane, off the node's own thread,
    // never in the node process.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    // A 128-sample buffer is well over the 2·32 envelope floor, so once it fills the
    // last axis actually shrinks.
    call(
        &mut ws,
        3,
        "update_param",
        json!({ "node": buf, "group": "buffer", "name": "size", "value": 128 }),
    )
    .await;
    call(
        &mut ws,
        4,
        "add_link",
        json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" }),
    )
    .await;

    let (mut data, _) = connect_async(format!("{base}/data/{buf}/out"))
        .await
        .unwrap();
    // Inband ViewSpec: one line viewer wanting the last axis enveloped to 32.
    data.send(Message::Text(
        json!({
            "op": "view",
            "specs": [{
                "dtype": "array",
                "ndim": [["le", 2]],
                "reduce": [{ "dim": -1, "max": 32, "method": "envelope" }]
            }]
        })
        .to_string(),
    ))
    .await
    .unwrap();

    // Wait for a frame that carries reduced meta AND genuinely shrank on the last axis —
    // the definitive proof the plan was applied (passthrough never stamps it). Requiring a
    // real shrink (not merely `reduced.is_some()`) avoids a boundary race: envelope fires at
    // axis len ≥ 2·W = 64 producing exactly 64 samples, so a frame caught with the Buffer at
    // *exactly* 64 (which happens when the node's thread is starved under parallel test load)
    // has orig_len == output == 64 — reduced-meta present but no shrink. Keep consuming until
    // the buffer has grown past the envelope floor, so the assertions below see a true
    // reduction. Bounded so a stuck plane still fails loudly.
    let last_dim_shrank = |d: &goofi_core::Data| -> bool {
        let Some(goofi_core::MetaValue::Map(dims)) = d.meta().reduced().as_ref() else {
            return false;
        };
        let last = d.ndim() - 1;
        let Some(goofi_core::MetaValue::Map(entry)) = dims.get(&last.to_string()) else {
            return false;
        };
        let orig = match entry.get("orig_len") {
            Some(goofi_core::MetaValue::Uint(n)) => *n as i64,
            Some(goofi_core::MetaValue::Int(n)) => *n,
            _ => return false,
        };
        (d.shape()[last] as i64) < orig
    };
    let reduced = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            let msg = data.next().await.expect("stream ended").expect("ws error");
            if let Message::Binary(b) = msg {
                let d = goofi_codec::decode(&b).expect("decodable GOOF frame");
                if last_dim_shrank(&d) {
                    break d;
                }
            }
        }
    })
    .await
    .expect("a genuinely-reduced frame must arrive");

    let goofi_core::MetaValue::Map(dims) = reduced.meta().reduced().as_ref().unwrap() else {
        panic!("reduced meta is a per-dim map");
    };
    let last = reduced.ndim() - 1;
    let goofi_core::MetaValue::Map(entry) = dims.get(&last.to_string()).expect("last dim reduced")
    else {
        panic!("dim entry is a map");
    };
    assert_eq!(entry.get("method"), Some(&goofi_core::MetaValue::Str("envelope".into())));
    // The pre-reduction axis is recorded. msgpack round-trips a small +int as Int, so read
    // the value not the discriminant. Envelope only fires past its 2·32 floor, so the buffer
    // had grown to ≥ 64 by the frame we caught.
    let orig_len = match entry.get("orig_len") {
        Some(goofi_core::MetaValue::Uint(n)) => *n as i64,
        Some(goofi_core::MetaValue::Int(n)) => *n,
        other => panic!("orig_len is an integer; got {other:?}"),
    };
    assert!(orig_len >= 64, "envelope fires only on a large axis; orig_len {orig_len}");
    // Envelope emits (min,max) per bin → ≤ 2·32 samples, and strictly fewer than the source.
    assert!(
        reduced.shape()[last] <= 64 && (reduced.shape()[last] as i64) < orig_len,
        "axis shrank to envelope width; got {} from {orig_len}",
        reduced.shape()[last]
    );
}

#[tokio::test]
async fn a_client_replica_converges_via_the_binary_sync_relay() {
    // Phase 2: a browser Yjs replica (here a Rust GraphDoc standing in for it) mounts, syncs
    // the current graph over the /control binary channel, and receives live deltas as the
    // graph mutates — the reader half of the CRDT control plane.
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};

    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await; // JSON hello (existing snapshot event)
    let _server_sv = recv_binary(&mut ws).await; // server's sync_hello (its state vector)

    // The client mounts an empty replica and advertises its state vector; the server replies
    // with the diff it lacks (the full current doc — empty graph so far).
    let mut client = GraphDoc::new();
    ws.send(Message::Binary(client.sync_hello().into())).await.unwrap();
    let diff = recv_binary(&mut ws).await;
    client.on_sync(SyncMsg::decode(&diff).expect("a sync frame"));
    assert!(client.node_ids().is_empty(), "converged to the empty graph");

    // Mutate the graph via a normal RPC; the server broadcasts the delta on the binary channel.
    ws.send(Message::Text(
        json!({ "id": 1, "op": "add_node", "payload": { "type": "Oscillator" } }).to_string(),
    ))
    .await
    .unwrap();

    // Collect frames until the client's replica reflects the new node (the text reply carries
    // its uid; the binary delta carries the CRDT change).
    let mut uid: Option<String> = None;
    for _ in 0..20 {
        let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timeout")
            .expect("stream")
            .expect("ws");
        match msg {
            Message::Text(t) => {
                let v: Value = serde_json::from_str(t.as_str()).unwrap();
                if v.get("id").and_then(|x| x.as_i64()) == Some(1) {
                    uid = v["result"]["uid"].as_str().map(str::to_string);
                }
            }
            Message::Binary(b) => {
                if let Some(m) = SyncMsg::decode(&b) {
                    client.on_sync(m);
                }
            }
            _ => {}
        }
        if let Some(u) = &uid {
            if client.node_ids().contains(u) {
                assert_eq!(
                    client.read_at(&["nodes", u.as_str(), "type"]).as_ref().and_then(|v| v.as_str()),
                    Some("Oscillator"),
                    "delta carried the node"
                );
                return;
            }
        }
    }
    panic!("client replica never converged on the added node (uid={uid:?})");
}

#[tokio::test]
async fn two_tabs_on_one_slot_share_the_reducer_over_the_wire() {
    // Thalamus G1/G2 end-to-end: two /data connections to the SAME (node, slot) each receive
    // reduced frames from a single shared reducer (not one reduce loop per connection).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    call(&mut ws, 3, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" })).await;

    // Two independent viewers of buf/out (two browser tabs).
    let (mut a, _) = connect_async(format!("{base}/data/{buf}/out")).await.unwrap();
    let (mut b, _) = connect_async(format!("{base}/data/{buf}/out")).await.unwrap();

    // Both receive a decodable frame — the shared reducer fans out to every subscriber.
    for (name, sock) in [("a", &mut a), ("b", &mut b)] {
        let got = tokio::time::timeout(Duration::from_secs(8), async {
            loop {
                if let Message::Binary(bytes) = sock.next().await.unwrap().unwrap() {
                    if goofi_codec::decode(&bytes).is_ok() {
                        return true;
                    }
                }
            }
        })
        .await
        .unwrap_or(false);
        assert!(got, "viewer {name} received a frame from the shared reducer");
    }
}


#[tokio::test]
async fn data_plane_streams_an_output_boundary_via_the_inner_leaf() {
    // Group a Buffer whose output is wired downstream → the instance gains an output boundary.
    // A viewer subscribing to /data/{inst}/{bnd} must receive the inner Buffer's frames (spec
    // §5: a boundary resolves chain-to-leaf to exactly one physical stream).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    let sink = uid(&call(&mut ws, 3, "add_node", json!({ "type": "Buffer" })).await);
    call(&mut ws, 4, "update_param", json!({ "node": buf, "group": "buffer", "name": "size", "value": 64 })).await;
    call(&mut ws, 5, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" })).await;
    // buf.out → sink makes buf's output a CUT link when buf is grouped → an output boundary.
    call(&mut ws, 6, "add_link", json!({ "node_out": buf, "slot_out": "out", "node_in": sink, "slot_in": "data" })).await;

    let reply = call(&mut ws, 7, "group_nodes", json!({ "members": [buf], "pos": [0.0, 0.0] })).await;
    let inst = reply["result"]["inst_id"].as_str().unwrap().to_string();
    // The scope must expose out0 (buf.out) as an output stub — read it from the doc forest
    // (subpatch_changed retired). This also barriers the group before the /data subscription below.
    let doc = sync_replica(&mut ws, |d| d.read_at(&["instances", inst.as_str(), "stubs", "out0"]).is_some()).await;
    assert!(
        doc.read_at(&["instances", inst.as_str(), "stubs", "out0"]).is_some(),
        "output stub out0 present in the doc forest"
    );

    // Subscribe to the boundary port; frames come from the inner Buffer leaf.
    let (mut data, _) = connect_async(format!("{base}/data/{inst}/out0")).await.unwrap();
    let frame = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Message::Binary(b) = data.next().await.unwrap().unwrap() {
                break b;
            }
        }
    })
    .await
    .expect("a frame must arrive via the boundary");
    assert_eq!(&frame[0..4], b"GOOF", "the inner leaf's frame streams through the boundary");
}

/// Connect a `/control` client whose TCP receive buffer is pinned tiny, so a stalled reader
/// deterministically lags the server's broadcast ring (setting SO_RCVBUF also disables the
/// kernel's autotuning that would otherwise absorb the whole flood).
async fn connect_small_rcvbuf(base: &str) -> Ws {
    let addr: std::net::SocketAddr = base.trim_start_matches("ws://").parse().unwrap();
    let sock =
        socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, Some(socket2::Protocol::TCP))
            .unwrap();
    sock.set_recv_buffer_size(2048).unwrap();
    sock.connect(&addr.into()).unwrap();
    sock.set_nonblocking(true).unwrap();
    let std_stream: std::net::TcpStream = sock.into();
    let tokio_stream = tokio::net::TcpStream::from_std(std_stream).unwrap();
    let (ws, _) = tokio_tungstenite::client_async(
        format!("{base}/control"),
        tokio_tungstenite::MaybeTlsStream::Plain(tokio_stream),
    )
    .await
    .unwrap();
    ws
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lagged_control_client_recovers_via_a_fresh_snapshot() {
    // The JSON `events` plane must recover a client that lagged past the shared broadcast ring,
    // exactly as the sync_updates plane does — otherwise a dropped structural event permanently
    // desyncs its mirror. Victim A has a tiny receive buffer and STOPS reading; flooder B pumps
    // state_update events (constant value → the sync plane stays quiet, isolating the events
    // plane) far past A's 256-slot ring; when A resumes it must receive a fresh `hello` snapshot.
    let base = start_server().await;

    // Victim A: tiny recv buffer, read the initial hello (+ its sync SV), then stall.
    let mut a = connect_small_rcvbuf(&base).await;
    let h0 = recv_text(&mut a).await;
    assert_eq!(h0["event"], "hello", "initial hello");
    let _a_sv = recv_binary(&mut a).await;

    // Flooder B: add one node, then background-drain so B itself never blocks the server.
    let (mut b, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut b).await;
    let _ = recv_binary(&mut b).await;
    let osc = call(&mut b, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    let (mut btx, mut brx) = b.split();
    let drain = tokio::spawn(async move { while let Some(Ok(_)) = brx.next().await {} });

    // Flood: id-less (no reply) set_expression re-binding the SAME constant expression — each pushes
    // a state_update on the events plane, while the unchanged binding leaves the sync plane quiet
    // (isolating the events plane). A's 2 KB receive buffer blocks its server task after a few
    // frames, so while A stays idle the ring only needs ~256 of these to overflow. The stall below
    // (not any single wall-clock value) is what forces the lag: A must NOT drain while the ring
    // overflows, or it would keep pace and never lag.
    for _ in 0..2000 {
        btx.send(Message::Text(
            json!({ "op": "set_expression", "payload": {
                "node": osc, "group": "common", "name": "max_frequency",
                "expression": "7", "enabled": true, "triggers": false
            }})
            .to_string(),
        ))
        .await
        .unwrap();
    }
    // Hold A idle long enough for the server to broadcast past the 256-slot ring even under a
    // saturated parallel-suite; it only needs to process ~256 of the flood, far less than all.
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // A resumes: among the buffered frames it MUST receive a recovery hello (the second one).
    let recovered = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            let m = recv_text(&mut a).await;
            if m.get("event").and_then(|v| v.as_str()) == Some("hello") {
                return true;
            }
        }
    })
    .await
    .unwrap_or(false);
    drain.abort();
    assert!(recovered, "a lagged control client must recover via a fresh hello snapshot");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_clients_concurrently_drag_and_all_converge() {
    // Stress the POSITION command path against the re-mirror — the exact interleaving the audit
    // found losing drags before upsert_node was made idempotent. N clients each own a node and
    // hammer ROUNDS `set_node_pos` commands concurrently; EACH triggers a manager re-mirror that
    // re-asserts EVERY node's pos (upsert_node). With the wholesale pos-map replacement this test
    // would drop drags (a fresh reader would not converge on all N final positions); with the
    // idempotent in-place upsert_node every concurrent drag survives.
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};

    const N: usize = 8;
    const ROUNDS: usize = 5;

    let base = start_server().await;

    let (mut setup, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut setup).await;
    let _ = recv_binary(&mut setup).await;
    let mut uids = Vec::new();
    for i in 0..N {
        let u = call(&mut setup, i as i64 + 1, "add_node", json!({ "type": "Oscillator" })).await
            ["result"]["uid"]
            .as_str()
            .unwrap()
            .to_string();
        uids.push(u);
    }

    let mut handles = Vec::new();
    for i in 0..N {
        let base = base.clone();
        let uids = uids.clone();
        handles.push(tokio::spawn(async move {
            let (mut w, _) = connect_async(format!("{base}/control")).await.unwrap();
            let _ = recv_text(&mut w).await;
            let _ = recv_binary(&mut w).await;
            // Ramp this node's position 1 → ROUNDS via the set_node_pos command op; awaiting each
            // reply proves the re-mirror re-asserted every node's pos without orphaning a concurrent
            // writer's drag.
            for r in 1..=ROUNDS {
                call(
                    &mut w,
                    r as i64,
                    "set_node_pos",
                    json!({ "node": uids[i], "pos": [r as f64, r as f64] }),
                )
                .await;
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let (mut r, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut r).await;
    let _ = recv_binary(&mut r).await;
    let mut rdoc = GraphDoc::new();
    r.send(Message::Binary(rdoc.sync_hello().into())).await.unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let mut converged = false;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(500), r.next()).await {
            Ok(Some(Ok(Message::Binary(b)))) => {
                if let Some(m) = SyncMsg::decode(&b) {
                    rdoc.on_sync(m);
                }
            }
            Ok(Some(Ok(_))) => continue,
            Ok(_) => break,
            Err(_) => {}
        }
        if uids.iter().all(|u| doc_node_pos(&rdoc, u) == Some([ROUNDS as f64, ROUNDS as f64])) {
            converged = true;
            break;
        }
    }
    if !converged {
        let got: Vec<_> = uids.iter().map(|u| doc_node_pos(&rdoc, u)).collect();
        panic!("not converged; final position per node = {got:?}");
    }
}







// ---------------------------------------------------------------------------
// Data-plane peer liveness (audit item 10)
// ---------------------------------------------------------------------------

/// Serve with a deliberately tiny liveness policy, and hand the state back so the test can watch
/// the shared reducer's refcount — the thing a stalled peer used to pin open forever.
async fn start_server_with_liveness(live: goofi_bridge::DataLiveness) -> (String, AppState) {
    let mut state = AppState::new();
    state.data_liveness = live;
    spawn_stats(state.graph.clone(), state.events.clone(), 2);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = state.clone();
    tokio::spawn(async move {
        serve_app(listener, served, None).await.unwrap();
    });
    (format!("ws://{addr}"), state)
}

/// A short policy: fast enough that the suite never waits on a production deadline, slow enough
/// that it cannot be tripped by ordinary scheduler jitter.
fn test_liveness() -> goofi_bridge::DataLiveness {
    goofi_bridge::DataLiveness {
        ping_interval: Duration::from_millis(100),
        pong_deadline: Duration::from_millis(1000),
        send_timeout: Duration::from_millis(200),
    }
}

/// Poll `f` until it holds or `limit` elapses; returns whether it held. Asserting the PROPERTY
/// ("torn down by T") rather than a window is what keeps this stable under cargo's parallel runner.
async fn holds_within(limit: Duration, mut f: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + limit;
    while std::time::Instant::now() < deadline {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    f()
}

#[tokio::test]
async fn a_data_peer_that_never_pongs_is_torn_down_and_its_reducer_reclaimed() {
    // The dead-but-not-closed peer: a viewer that completed the handshake and then went silent —
    // it never sends Close, never reads, never pongs. Nothing on the socket errors, so without an
    // active probe the connection lives forever and the SHARED per-slot reducer keeps reducing and
    // encoding for a viewer that is not there. The fix must reach the EXISTING `unsubscribe`.
    let (base, state) = start_server_with_liveness(test_liveness()).await;
    let osc = state.graph.lock().unwrap().add_node("Oscillator", None).unwrap();
    let key = (osc, "out".to_string());

    // `connect_async` performs the handshake but starts NO background task, so simply never
    // polling this stream is a faithful frozen peer: no auto-pong, no reads, no Close.
    let dead = connect_async(format!("{base}/data/{}/out", osc.to_hex())).await.unwrap().0;

    assert!(
        holds_within(Duration::from_secs(2), || state.reducers.subscribers(&key) == 1).await,
        "the peer subscribed to the slot's reducer"
    );

    // Generous bound: ~10x the 300 ms deadline. The assertion is the property (reclaimed BY T),
    // not a window, and never a median.
    assert!(
        holds_within(Duration::from_secs(3), || state.reducers.active_slots() == 0).await,
        "a peer that never pongs must be torn down past the deadline and its reducer reclaimed \
         (active_slots={}, subscribers={})",
        state.reducers.active_slots(),
        state.reducers.subscribers(&key),
    );
    drop(dead);
}

#[tokio::test]
async fn an_idle_dead_peer_is_reclaimed_because_a_probe_is_not_its_own_proof_of_life() {
    // The other half of the dead-peer case, and the one that constrains the design: a viewer of a
    // slot that publishes NOTHING (an unwired Buffer). No frame write can ever vouch for this
    // peer, and its socket buffer happily swallows every 2-byte ping — so if a sent probe were
    // allowed to count as evidence, this connection would pin its reducer open forever. Only the
    // ANSWER counts.
    let (base, state) = start_server_with_liveness(test_liveness()).await;
    let buf = state.graph.lock().unwrap().add_node("Buffer", None).unwrap();
    let key = (buf, "out".to_string());

    let dead = connect_async(format!("{base}/data/{}/out", buf.to_hex())).await.unwrap().0;
    assert!(
        holds_within(Duration::from_secs(2), || state.reducers.subscribers(&key) == 1).await,
        "the idle peer subscribed to the slot's reducer"
    );
    assert!(
        holds_within(Duration::from_secs(3), || state.reducers.active_slots() == 0).await,
        "an idle peer that never pongs must still be reclaimed (active_slots={})",
        state.reducers.active_slots(),
    );
    drop(dead);
}

#[tokio::test]
async fn a_slow_but_alive_viewer_that_pongs_keeps_its_reducer_and_its_frames() {
    // The regression this fix is most likely to cause: killing a HEALTHY viewer. Modelled on the
    // real client — `dataWorker` drains the socket while `frames.ts` coalesces to rAF — as a tab
    // that repeatedly stalls for longer than the reducer's 16-slot ring holds, then catches up.
    // It must survive several deadlines, keep receiving, and still visibly LAG: dropping frames
    // (latest-wins, the `Lagged` contract) is what a slow viewer is supposed to do; dropping the
    // connection is not.
    let (base, state) = start_server_with_liveness(test_liveness()).await;
    let osc = state.graph.lock().unwrap().add_node("Oscillator", None).unwrap();
    let key = (osc, "out".to_string());

    let mut slow = connect_async(format!("{base}/data/{}/out", osc.to_hex())).await.unwrap().0;

    let start = std::time::Instant::now();
    let mut received = 0usize;
    let mut received_past_the_deadline = 0usize;
    // 5 rounds x 600 ms = 3 s, three times the 1 s deadline.
    for _ in 0..5 {
        // Stall ~400 ms: at the reducer's ~62 Hz that is ~25 frames against a 16-slot ring, so
        // this viewer provably falls behind and provably backs the socket up.
        tokio::time::sleep(Duration::from_millis(400)).await;
        // Then drain, exactly as the real worker does. Draining is what makes tungstenite answer
        // the pings that queued up behind the backlog.
        let drain = std::time::Instant::now();
        while drain.elapsed() < Duration::from_millis(200) {
            match tokio::time::timeout(Duration::from_millis(50), slow.next()).await {
                Ok(Some(Ok(Message::Binary(_)))) => {
                    received += 1;
                    if start.elapsed() > Duration::from_millis(1000) {
                        received_past_the_deadline += 1;
                    }
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(e))) => panic!("a slow-but-alive viewer was disconnected: {e}"),
                Ok(None) => panic!("a slow-but-alive viewer's stream was closed by the bridge"),
                Err(_) => {}
            }
        }
    }

    assert_eq!(state.reducers.subscribers(&key), 1, "the slow viewer is still subscribed");
    assert_eq!(state.reducers.active_slots(), 1, "its reducer is still running");
    assert!(
        received_past_the_deadline > 0,
        "a slow-but-alive viewer keeps receiving frames well past the pong deadline"
    );
    // …and it really was outpaced while it stalled: the reducer produced far more than the
    // 16-slot ring holds during each pause, so this viewer was repeatedly behind rather than
    // quietly keeping up. (How many frames it ultimately *lost* is not asserted — with small
    // frames on loopback the socket buffers can absorb a whole stall — because pinning an
    // environment-dependent drop count is exactly the kind of wall-clock assertion that flakes.)
    let produced = state.reducers.reductions(&key);
    assert!(
        produced > 50 && received > 0,
        "the reducer outpaced the stalling viewer ({received} received of {produced} produced)"
    );
}
