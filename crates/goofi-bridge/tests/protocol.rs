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

async fn recv_text(ws: &mut (impl StreamExt<Item = tokio_tungstenite::tungstenite::Result<Message>> + Unpin)) -> Value {
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

#[tokio::test]
async fn control_and_data_plane_end_to_end() {
    let state = AppState::new();
    spawn_tick(state.graph.clone(), 240);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_listener(listener, state).await.unwrap();
    });

    let base = format!("ws://{addr}");
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
