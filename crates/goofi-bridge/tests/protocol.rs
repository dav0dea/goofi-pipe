//! End-to-end protocol test: a real WebSocket client drives the bridge exactly
//! as the frontend would — receives `hello`, lists nodes, adds a node (and gets
//! the `node_added` broadcast), then subscribes to the data plane and receives a
//! decodable GOOF frame. Proves the M1 vertical slice (engine + control + data).

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::{serve_listener, spawn_tick, AppState};
use serde_json::{json, Value};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

async fn start_server() -> String {
    let state = AppState::new();
    spawn_tick(state.graph.clone(), 240);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_listener(listener, state).await.unwrap();
    });
    format!("ws://{addr}")
}

// A runtime type registered before serving — stands in for a discovered Python
// node (the CLI's `register_python` does exactly this against the live graph).
fn params() -> goofi_node::ParamGroups {
    goofi_node::ParamGroups::new()
}
fn stub_make(_: &goofi_node::ParamGroups) -> Box<dyn goofi_node::Node> {
    unreachable!("list_nodes never instantiates")
}
static SERVE_OUT: &[goofi_node::OutputDecl] = &[goofi_node::OutputDecl {
    name: "out",
    kind: goofi_core::SlotType::Array,
    length_preserving: false,
}];
static SERVE_MANIFEST: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "DiscoveredPyNode",
    category: "python",
    doc: "runtime type registered before serving",
    inputs: &[],
    outputs: SERVE_OUT,
    default_params: params,
    isolation: goofi_node::Isolation::InProcess,
    make: stub_make,
};

async fn start_server_with_runtime_type() -> String {
    let state = AppState::new();
    state
        .graph
        .lock()
        .unwrap()
        .register_dyn_type(&SERVE_MANIFEST, Box::new(|_| unreachable!()));
    spawn_tick(state.graph.clone(), 240);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_listener(listener, state).await.unwrap();
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

#[tokio::test]
async fn runtime_registered_type_reaches_the_palette_over_the_wire() {
    // The full serving path a browser sees: a runtime type registered into the
    // live graph (as the CLI's --python-nodes does) surfaces via list_nodes.
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

    // 1. hello: protocol_version + instance_id + ROOT instance.
    let hello = recv_text(&mut ws).await;
    assert_eq!(hello["event"], "hello");
    assert_eq!(hello["payload"]["protocol_version"], 1);
    assert!(hello["payload"]["instance_id"].is_string());
    assert!(hello["payload"]["instances"]["__root__"].is_object());
    assert_eq!(hello["payload"]["nodes"].as_array().unwrap().len(), 0);

    // 2. list_nodes returns the catalog incl. ConstantArray.
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
    for expected in ["ConstantArray", "Oscillator", "Buffer", "Math"] {
        assert!(
            types.iter().any(|t| t["type"] == expected),
            "catalog must contain {expected}; got {:?}",
            types.iter().map(|t| &t["type"]).collect::<Vec<_>>()
        );
    }
    // Test-only nodes are hidden.
    assert!(!types.iter().any(|t| t["type"] == "_TestEcho"));

    // 3. add_node -> uid result + node_added broadcast.
    ws.send(Message::Text(
        json!({"id": 2, "op": "add_node", "payload": {"type": "ConstantArray", "category": "inputs", "pos": [10.0, 20.0]}})
            .to_string(),
    ))
    .await
    .unwrap();

    let mut uid: Option<String> = None;
    let mut saw_added = false;
    for _ in 0..6 {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(2) {
            uid = Some(m["result"].as_str().unwrap().to_string());
        } else if m["event"] == "node_added" {
            saw_added = true;
            assert_eq!(m["payload"]["type"], "ConstantArray");
            assert_eq!(m["payload"]["pos"][0], 10.0);
        }
        if uid.is_some() && saw_added {
            break;
        }
    }
    let uid = uid.expect("add_node returned a uid");
    assert!(saw_added, "node_added must be broadcast");

    // 4. data plane: subscribe and receive a decodable GOOF frame.
    let (mut data, _) = connect_async(format!("{base}/data/{uid}/out/line"))
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
    let (mut bad, _) = connect_async(format!("{base}/data/{uid}/nope/line"))
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
async fn native_dsp_chain_streams_a_spectrum() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    let psd = uid(&call(&mut ws, 3, "add_node", json!({ "type": "PSD" })).await);

    // Bound the buffer so the spectrum length settles quickly.
    call(
        &mut ws,
        4,
        "update_param",
        json!({ "node": buf, "group": "buffer", "name": "size", "value": 128 }),
    )
    .await;
    // Oscillator -> Buffer -> PSD
    call(
        &mut ws,
        5,
        "add_link",
        json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" }),
    )
    .await;
    call(
        &mut ws,
        6,
        "add_link",
        json!({ "node_out": buf, "slot_out": "out", "node_in": psd, "slot_in": "data" }),
    )
    .await;

    // The PSD output streams a real one-sided spectrum through the data plane.
    let (mut data, _) = connect_async(format!("{base}/data/{psd}/psd/line"))
        .await
        .unwrap();
    let frame = loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), data.next())
            .await
            .expect("spectrum frame timed out")
            .expect("stream ended")
            .expect("ws error");
        if let Message::Binary(b) = msg {
            break b;
        }
    };
    assert_eq!(&frame[0..4], b"GOOF");
    assert_eq!(frame[4], 2, "version");
    assert_eq!(frame[5], 0, "PSD emits an ARRAY");
    let body_len = u32::from_le_bytes(frame[10..14].try_into().unwrap());
    assert!(body_len > 8, "non-trivial spectrum body ({body_len} bytes)");
}

#[tokio::test]
async fn serialize_and_load_roundtrip() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let ser = call(&mut ws, 2, "serialize", json!({})).await;
    let yaml = ser["result"]["yaml"].as_str().unwrap().to_string();
    assert!(yaml.contains("version: 3"), "gfi v3 header");
    assert!(yaml.contains("Oscillator"), "node persisted");

    // load_text replaces the graph and broadcasts graph_replaced.
    ws.send(Message::Text(
        json!({ "id": 3, "op": "load_text", "payload": { "content": yaml } }).to_string(),
    ))
    .await
    .unwrap();
    let replaced = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("graph_replaced") {
            break m;
        }
    };
    let nodes = replaced["payload"]["nodes"].as_array().unwrap();
    assert!(
        nodes.iter().any(|n| n["type"] == "Oscillator"),
        "graph_replaced snapshot contains the restored node"
    );
}
