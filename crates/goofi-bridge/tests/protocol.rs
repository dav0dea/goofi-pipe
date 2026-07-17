//! End-to-end protocol test: a real WebSocket client drives the bridge exactly
//! as the frontend would — receives `hello`, lists nodes, adds a node (and gets
//! the `node_added` broadcast), then subscribes to the data plane and receives a
//! decodable GOOF frame. Proves the M1 vertical slice (engine + control + data).

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::{serve_listener, spawn_tick, AppState};
use goofi_view::Reducible; // shape()/ndim() accessors on a decoded frame
use serde_json::{json, Value};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

async fn start_server() -> String {
    let state = AppState::new();
    spawn_tick(state.graph.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_listener(listener, state).await.unwrap();
    });
    format!("ws://{addr}")
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
    factory: stub_factory,
};

async fn start_server_with_runtime_type() -> String {
    let state = AppState::new();
    state
        .graph
        .lock()
        .unwrap()
        .register_dyn_type(&SERVE_MANIFEST, Box::new(|_| unreachable!()));
    spawn_tick(state.graph.clone());
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

/// Receive until the named event arrives, returning it (skipping RPC replies/other events).
async fn drain_event(ws: &mut Ws, event: &str) -> Value {
    loop {
        let m = recv_text(ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some(event) {
            return m;
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

    // 3. add_node -> uid result + node_added broadcast.
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
            uid = Some(m["result"].as_str().unwrap().to_string());
        } else if m["event"] == "node_added" {
            saw_added = true;
            assert_eq!(m["payload"]["type"], "Oscillator");
            assert_eq!(m["payload"]["pos"][0], 10.0);
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

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
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
async fn data_plane_reduces_to_the_declared_viewspec() {
    // A viewer declares its need inband on the /data socket (line: array, ≤2-D, envelope
    // dim -1 → 32). The bridge reduces the buffered frame ONCE for this connection and
    // stamps `meta.reduced` — proving reduction runs on the data plane, off the node tick,
    // never in the node process.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
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

    // Wait for a frame that carries reduced meta — the definitive proof the plan was
    // applied (passthrough never stamps it). Bounded so a stuck plane fails loudly.
    let reduced = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            let msg = data.next().await.expect("stream ended").expect("ws error");
            if let Message::Binary(b) = msg {
                let d = goofi_codec::decode(&b).expect("decodable GOOF frame");
                if d.meta().reduced.is_some() {
                    break d;
                }
            }
        }
    })
    .await
    .expect("a reduced frame must arrive");

    let goofi_core::MetaValue::Map(dims) = reduced.meta().reduced.as_ref().unwrap() else {
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
async fn set_expression_binds_and_reflects_over_the_wire() {
    // Regression guard for the "unknown op `set_expression`" bug: the op must dispatch
    // (not 404), store the binding, and echo it back in the node's param descriptor —
    // including `expression_error` (the field indicator) and with NO `expression_autoeval`
    // key (auto-eval is always on, so there is no autoeval flag on the wire).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }))
        .await["result"]
        .as_str()
        .unwrap()
        .to_string();

    // Bind an expression on the universal common.max_frequency param.
    ws.send(Message::Text(
        json!({
            "id": 2,
            "op": "set_expression",
            "payload": {
                "node": osc,
                "group": "common",
                "name": "max_frequency",
                "expression": "1 + 2",
                "expression_enabled": true,
                "expression_triggers_process": false
            }
        })
        .to_string(),
    ))
    .await
    .unwrap();

    // Collect both the id=2 reply and the state_update broadcast (either order).
    let mut ok = false;
    let mut descriptor: Option<Value> = None;
    for _ in 0..10 {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(2) {
            assert_eq!(m["result"]["ok"], true, "set_expression must dispatch, not 404 as unknown op");
            ok = true;
        } else if m["event"] == "state_update" && m["payload"]["node"] == json!(osc) {
            descriptor = Some(m["payload"]["params"]["common"]["max_frequency"].clone());
        }
        if ok && descriptor.is_some() {
            break;
        }
    }
    assert!(ok, "set_expression reply must arrive");
    let d = descriptor.expect("state_update carrying the param descriptor");
    assert_eq!(d["expression"], "1 + 2", "source round-trips");
    assert_eq!(d["expression_enabled"], true);
    assert_eq!(d["expression_triggers_process"], false);
    // This harness injects no evaluator, so the binding round-trips WITH an error — the
    // point is the field exists as a string to drive the per-param red indicator.
    assert!(
        d["expression_error"].is_string(),
        "expression_error must be present for the field indicator; got {:?}",
        d["expression_error"]
    );
    assert!(
        d.get("expression_autoeval").is_none(),
        "no autoeval flag on the wire (auto-eval is always on)"
    );
}

#[tokio::test]
async fn group_and_expand_project_the_instance_forest() {
    // Grouping two nodes surfaces one instance in the snapshot (ROOT membership re-tagged,
    // the members moved into the instance's scope); expanding restores them to ROOT.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    call(
        &mut ws,
        3,
        "add_link",
        json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" }),
    )
    .await;

    // Group both into a sub-patch.
    let reply = call(&mut ws, 4, "group_nodes", json!({ "members": [osc, buf], "pos": [50.0, 50.0] })).await;
    let inst = reply["result"]["inst_id"].as_str().expect("inst_id returned").to_string();

    // The subpatch_changed snapshot: ROOT holds the instance (not the members); the instance
    // scope holds both members.
    let snap = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("subpatch_changed") {
            break m;
        }
    };
    let root = &snap["payload"]["instances"]["__root__"];
    assert!(root["members"].as_object().unwrap().values().any(|v| v["uid"] == json!(inst) && v["is_instance"] == true),
        "ROOT lists the instance; got {:?}", root["members"]);
    let inst_info = &snap["payload"]["instances"][&inst];
    assert_eq!(inst_info["kind"], "unique", "one reference ⇒ unique");
    assert_eq!(inst_info["members"].as_object().unwrap().len(), 2, "both members in the instance scope");
    // The osc member's node info reports its new membership.
    let nodes = snap["payload"]["nodes"].as_array().unwrap();
    let osc_node = nodes.iter().find(|n| n["uid"] == json!(osc)).unwrap();
    assert_eq!(osc_node["membership"]["instance"], json!(inst), "member membership re-tagged");

    // Expand restores both members to ROOT.
    let ex = call(&mut ws, 5, "expand_instance", json!({ "inst_id": inst })).await;
    let restored = ex["result"]["restored"].as_array().unwrap();
    assert_eq!(restored.len(), 2, "both members restored");
    let snap2 = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("subpatch_changed") {
            break m;
        }
    };
    assert!(snap2["payload"]["instances"].get(&inst).is_none() || snap2["payload"]["instances"][&inst].is_null(),
        "instance gone after expand");
    let osc_after = snap2["payload"]["nodes"].as_array().unwrap().iter().find(|n| n["uid"] == json!(osc)).unwrap();
    assert_eq!(osc_after["membership"]["instance"], "__root__", "member back at ROOT");
}

#[tokio::test]
async fn duplicate_shared_then_make_unique_over_the_wire() {
    // Group → duplicate_shared surfaces a sibling and marks both instances "shared";
    // make_unique on one returns it to "unique".
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    call(&mut ws, 3, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" })).await;
    let inst = call(&mut ws, 4, "group_nodes", json!({ "members": [osc, buf], "pos": [0.0, 0.0] })).await["result"]["inst_id"]
        .as_str().unwrap().to_string();
    drain_event(&mut ws, "subpatch_changed").await;

    // Duplicate → a sibling instance, both now "shared".
    let dup = call(&mut ws, 5, "duplicate_shared", json!({ "inst_id": inst, "pos": [200.0, 0.0] })).await;
    let sib = dup["result"]["inst_id"].as_str().expect("sibling inst_id").to_string();
    assert_ne!(sib, inst, "a fresh sibling instance");
    let snap = drain_event(&mut ws, "subpatch_changed").await;
    assert_eq!(snap["payload"]["instances"][&inst]["kind"], "shared", "original now shared");
    assert_eq!(snap["payload"]["instances"][&sib]["kind"], "shared", "sibling shared");
    // The sibling has its own two members, distinct from the original's.
    let sib_members = snap["payload"]["instances"][&sib]["members"].as_object().unwrap();
    assert_eq!(sib_members.len(), 2, "sibling has both members");
    // Four flat leaves total now (2 original + 2 sibling).
    assert_eq!(snap["payload"]["nodes"].as_array().unwrap().len(), 4, "sibling leaves spawned");

    // make_unique the sibling → back to "unique".
    call(&mut ws, 6, "make_unique", json!({ "inst_id": sib })).await;
    let snap2 = drain_event(&mut ws, "subpatch_changed").await;
    assert_eq!(snap2["payload"]["instances"][&sib]["kind"], "unique", "sibling forked to unique");
    assert_eq!(snap2["payload"]["instances"][&inst]["kind"], "unique", "original back to unique too");

    // Undo-of-duplicate routes remove_node on the sibling INSTANCE uid → the whole subtree is
    // torn down (the undo/redo executor relies on this).
    call(&mut ws, 7, "remove_node", json!({ "node": sib })).await;
    let snap3 = drain_event(&mut ws, "subpatch_changed").await;
    assert!(snap3["payload"]["instances"].get(&sib).is_none() || snap3["payload"]["instances"][&sib].is_null(),
        "sibling instance removed");
    assert_eq!(snap3["payload"]["nodes"].as_array().unwrap().len(), 2, "only the original's two leaves remain");
}

#[tokio::test]
async fn connecting_to_a_boundary_creates_a_flat_leaf_link() {
    // Wiring a top-level node to an instance's input boundary must resolve to a flat leaf→leaf
    // link on the inner member — the boundary is a naming indirection, the runtime link is flat.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    // Group the buffer alone (no links yet → no auto boundaries), then author an input port.
    let inst = call(&mut ws, 3, "group_nodes", json!({ "members": [buf], "pos": [0.0, 0.0] })).await["result"]["inst_id"]
        .as_str()
        .unwrap()
        .to_string();
    let bnd = call(&mut ws, 4, "add_boundary", json!({ "inst_id": inst, "dir": "in", "dtype": "ARRAY", "pos": [0.0, 0.0] })).await
        ["result"]["bnd_id"].as_str().unwrap().to_string();
    call(&mut ws, 5, "wire_boundary", json!({ "inst_id": inst, "bnd_id": bnd, "inner_node": buf, "inner_slot": "data" })).await;

    // Connect osc.out → inst::in0. The bridge translates it to osc.out → buf.data.
    call(&mut ws, 6, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": inst, "slot_in": bnd })).await;
    let added = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("link_added") {
            break m;
        }
    };
    assert_eq!(added["payload"]["node_in"], json!(buf), "resolved to the inner buffer leaf, not the instance");
    assert_eq!(added["payload"]["slot_in"], "data", "resolved to the inner slot");
    assert_eq!(added["payload"]["node_out"], json!(osc), "the plain endpoint passes through");
}

#[tokio::test]
async fn boundary_authoring_over_the_wire() {
    // add_boundary → wire_boundary → rename_boundary, reflected in the snapshot interface.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    call(&mut ws, 3, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" })).await;
    // Group only the Buffer: osc→buf.data is a cut → auto input boundary; buf.out is UNexposed
    // (no downstream), so we can author a fresh output boundary onto it.
    let inst = call(&mut ws, 4, "group_nodes", json!({ "members": [buf], "pos": [0.0, 0.0] })).await["result"]["inst_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Add an unwired OUTPUT boundary, then wire it to the buffer's out.
    let add = call(&mut ws, 5, "add_boundary", json!({ "inst_id": inst, "dir": "out", "dtype": "ARRAY", "pos": [0.0, 0.0] })).await;
    let bnd = add["result"]["bnd_id"].as_str().expect("bnd_id").to_string();
    call(&mut ws, 6, "wire_boundary", json!({ "inst_id": inst, "bnd_id": bnd, "inner_node": buf, "inner_slot": "out" })).await;
    let rn = call(&mut ws, 7, "rename_boundary", json!({ "inst_id": inst, "bnd_id": bnd, "name": "wave" })).await;
    assert_eq!(rn["result"]["ok"], true);

    // The latest snapshot's interface carries the wired, renamed boundary (bnd_id unchanged).
    let snap = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("subpatch_changed") {
            let iface = &m["payload"]["instances"][&inst]["interface"];
            if iface.get(&bnd).and_then(|b| b.get("name")) == Some(&json!("wave")) {
                break m;
            }
        }
    };
    let port = &snap["payload"]["instances"][&inst]["interface"][&bnd];
    assert_eq!(port["dir"], "out");
    assert_eq!(port["inner_node"], json!(buf), "wired to the buffer leaf");
    assert_eq!(port["inner_slot"], "out");
    assert_eq!(port["name"], "wave", "renamed; bnd_id preserved");
}

#[tokio::test]
async fn data_plane_streams_an_output_boundary_via_the_inner_leaf() {
    // Group a Buffer whose output is wired downstream → the instance gains an output boundary.
    // A viewer subscribing to /data/{inst}/{bnd} must receive the inner Buffer's frames (spec
    // §5: a boundary resolves chain-to-leaf to exactly one physical stream).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    let sink = uid(&call(&mut ws, 3, "add_node", json!({ "type": "Buffer" })).await);
    call(&mut ws, 4, "update_param", json!({ "node": buf, "group": "buffer", "name": "size", "value": 64 })).await;
    call(&mut ws, 5, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" })).await;
    // buf.out → sink makes buf's output a CUT link when buf is grouped → an output boundary.
    call(&mut ws, 6, "add_link", json!({ "node_out": buf, "slot_out": "out", "node_in": sink, "slot_in": "data" })).await;

    let reply = call(&mut ws, 7, "group_nodes", json!({ "members": [buf], "pos": [0.0, 0.0] })).await;
    let inst = reply["result"]["inst_id"].as_str().unwrap().to_string();
    // drain the subpatch_changed event
    loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("subpatch_changed") {
            // The interface must expose out0 (buf.out) as an output boundary.
            let iface = &m["payload"]["instances"][&inst]["interface"];
            assert!(iface.get("out0").is_some(), "output boundary out0 present; got {iface:?}");
            break;
        }
    }

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

#[tokio::test]
async fn set_node_viewers_persists_and_echoes_the_view_state() {
    // The editor's per-slot viewer view-state (kind/settings) is server-authoritative:
    // set_node_viewers stores it, echoes it back, and it survives a serialize round-trip.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();

    let viewers = json!({ "out": { "collapsed": false, "kind": "line", "settings": { "yScale": 2 } } });
    ws.send(Message::Text(
        json!({ "id": 2, "op": "set_node_viewers", "payload": { "node": osc, "viewers": viewers } })
            .to_string(),
    ))
    .await
    .unwrap();

    // The reply dispatches (not 404) and the change is echoed as node_viewers.
    let mut ok = false;
    let mut echoed: Option<Value> = None;
    for _ in 0..10 {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(2) {
            assert_eq!(m["result"]["ok"], true, "set_node_viewers must dispatch");
            ok = true;
        } else if m["event"] == "node_viewers" && m["payload"]["node"] == json!(osc) {
            echoed = Some(m["payload"]["viewers"].clone());
        }
        if ok && echoed.is_some() {
            break;
        }
    }
    assert!(ok, "reply must arrive");
    assert_eq!(echoed.expect("node_viewers echo"), viewers, "view-state echoed verbatim");

    // It persists into the serialized .gfi.
    let yaml = call(&mut ws, 3, "serialize", json!({})).await["result"]["yaml"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(yaml.contains("yScale"), "view-state persisted to .gfi; got:\n{yaml}");
}

#[tokio::test]
async fn serialize_and_load_roundtrip() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let ser = call(&mut ws, 2, "serialize", json!({})).await;
    let yaml = ser["result"]["yaml"].as_str().unwrap().to_string();
    assert!(yaml.contains("version: 4"), "gfi v4 header");
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
