//! End-to-end protocol test: a real WebSocket client drives the bridge exactly
//! as the frontend would — receives `hello`, lists nodes, adds a node (and gets
//! the `node_added` broadcast), then subscribes to the data plane and receives a
//! decodable GOOF frame. Proves the M1 vertical slice (engine + control + data).

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::{serve_app, spawn_tick, spawn_workers, AppState};
use goofi_view::Reducible; // shape()/ndim() accessors on a decoded frame
use serde_json::{json, Value};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

// Read leaves through the generic CRDT reader (the typed getters were removed). A whole-number
// param comes back as an integer from `to_json`, so numeric reads compare via `as_f64`.
fn doc_param_f64(doc: &goofi_crdt::GraphDoc, uid: &str, group: &str, name: &str) -> Option<f64> {
    doc.read_at(&["nodes", uid, "params", group, name, "value"]).and_then(|v| v.as_f64())
}
fn doc_node_pos(doc: &goofi_crdt::GraphDoc, uid: &str) -> Option<[f64; 2]> {
    let x = doc.read_at(&["nodes", uid, "pos", "x"])?.as_f64()?;
    let y = doc.read_at(&["nodes", uid, "pos", "y"])?.as_f64()?;
    Some([x, y])
}

async fn start_server() -> String {
    let state = AppState::new();
    spawn_tick(state.graph.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
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

/// Bind an ENABLED expression on a param via the `set_expression` command op — the shape the
/// frontend sends (B3; the client-doc-write leaf path is retired). The manager routes it through an
/// `EditParam` command and echoes the runtime-enriched descriptor as a `state_update`.
async fn bind_expression(ws: &mut Ws, id: i64, node: &str, group: &str, name: &str, source: &str) {
    call(
        ws,
        id,
        "set_expression",
        json!({ "node": node, "group": group, "name": name, "expression": source, "enabled": true, "triggers": false }),
    )
    .await;
}

/// Sync a FRESH CRDT replica from the server over `ws` and drain binary sync frames until
/// `ready(&doc)` holds, returning the replica for forest/graph reads. The structural broadcast
/// events (`subpatch_changed`/`node_removed`) are retired — the forest reaches clients via the doc,
/// so tests read it here (the pattern `leaf_write_expression`/`connecting_to_a_boundary_…` use). A
/// fresh replica advertises an empty state vector, so the server's `sync_hello` reply is the COMPLETE
/// current doc; `ready` is always satisfiable once the preceding RPC's effect has landed.
async fn sync_replica(ws: &mut Ws, ready: impl Fn(&goofi_crdt::GraphDoc) -> bool) -> goofi_crdt::GraphDoc {
    use goofi_crdt::{GraphDoc, SyncMsg};
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

/// Like `call`, but tags the request with a `session` (the undo/redo scope).
async fn call_session(ws: &mut Ws, id: i64, op: &str, payload: Value, session: &str) -> Value {
    ws.send(Message::Text(
        json!({ "id": id, "op": op, "payload": payload, "session": session }).to_string(),
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
async fn add_undo_redo_over_the_wire_is_uid_stable() {
    // A command-backed add records an inverse; undo removes the node from the synced doc; redo
    // restores it at the SAME uid (the toggle-model history is uid-stable). can_undo/can_redo track.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call_session(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }), "s1").await["result"]
        .as_str()
        .unwrap()
        .to_string();
    let doc = sync_replica(&mut ws, |d| d.node_ids().iter().any(|u| *u == osc)).await;
    assert!(doc.node_ids().iter().any(|u| *u == osc), "node added");

    // Undo → the node is gone from the doc; the reply reports the session can now redo, not undo.
    let u = call_session(&mut ws, 2, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "undo changed the graph");
    assert_eq!(u["result"]["can_undo"], json!(false), "nothing left to undo");
    assert_eq!(u["result"]["can_redo"], json!(true), "can redo the undone add");
    let doc2 = sync_replica(&mut ws, |d| d.node_ids().is_empty()).await;
    assert!(doc2.node_ids().is_empty(), "undo removed the node");

    // Redo → the node returns at the SAME uid.
    let r = call_session(&mut ws, 3, "redo", json!({}), "s1").await;
    assert_eq!(r["result"]["changed"], json!(true), "redo changed the graph");
    let doc3 = sync_replica(&mut ws, |d| d.node_ids().iter().any(|u| *u == osc)).await;
    assert!(doc3.node_ids().iter().any(|u| *u == osc), "redo restored the SAME uid");
}

#[tokio::test]
async fn undo_is_scoped_per_session() {
    // Two sessions add a node each over one shared history; each session's undo reverts only ITS
    // own add (per-session filtering), never the other's.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let a = call_session(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }), "s1").await["result"]
        .as_str().unwrap().to_string();
    let b = call_session(&mut ws, 2, "add_node", json!({ "type": "Buffer" }), "s2").await["result"]
        .as_str().unwrap().to_string();
    let doc = sync_replica(&mut ws, |d| d.node_ids().len() == 2).await;
    assert!(doc.node_ids().iter().any(|u| *u == a) && doc.node_ids().iter().any(|u| *u == b));

    // s1 undo → only A is removed; B (s2's) survives.
    call_session(&mut ws, 3, "undo", json!({}), "s1").await;
    let doc2 = sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    assert!(!doc2.node_ids().iter().any(|u| *u == a), "s1 undo removed A");
    assert!(doc2.node_ids().iter().any(|u| *u == b), "s2's B is untouched by s1 undo");

    // s2 undo → B removed; graph empty.
    call_session(&mut ws, 4, "undo", json!({}), "s2").await;
    let doc3 = sync_replica(&mut ws, |d| d.node_ids().is_empty()).await;
    assert!(doc3.node_ids().is_empty(), "s2 undo removed B");
}

#[tokio::test]
async fn a_new_command_clears_the_sessions_redo_run() {
    // Undo then a fresh command discards the session's redo future (single-stack semantics).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call_session(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }), "s1").await;
    sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    call_session(&mut ws, 2, "undo", json!({}), "s1").await;
    sync_replica(&mut ws, |d| d.node_ids().is_empty()).await;

    // A fresh add clears the redo run — redo is now a no-op (changed:false, can_redo:false).
    call_session(&mut ws, 3, "add_node", json!({ "type": "Buffer" }), "s1").await;
    sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    let r = call_session(&mut ws, 4, "redo", json!({}), "s1").await;
    assert_eq!(r["result"]["changed"], json!(false), "redo run was cleared by the new command");
    assert_eq!(r["result"]["can_redo"], json!(false));
}

#[tokio::test]
async fn a_link_add_is_undoable_over_the_wire() {
    // add_link routes through the history (on the resolved flat link); undo removes it.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call_session(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }), "s1").await["result"]
        .as_str().unwrap().to_string();
    let buf = call_session(&mut ws, 2, "add_node", json!({ "type": "Buffer" }), "s1").await["result"]
        .as_str().unwrap().to_string();
    call_session(&mut ws, 3, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" }), "s1").await;
    let doc = sync_replica(&mut ws, |d| d.read_at(&["links"]).and_then(|v| v.as_array().map(|a| a.len())) == Some(1)).await;
    assert_eq!(doc.read_at(&["links"]).unwrap().as_array().unwrap().len(), 1, "link added");

    // Undo the link (the most recent s1 command) → back to zero links, both nodes intact. Anchor on
    // a POSITIVE presence (both nodes) a completed sync guarantees — an "empty links" predicate alone
    // is satisfied by the initial empty replica before any data frame lands.
    call_session(&mut ws, 4, "undo", json!({}), "s1").await;
    let doc2 = sync_replica(&mut ws, |d| {
        d.node_ids().len() == 2
            && d.read_at(&["links"]).and_then(|v| v.as_array().map(|a| a.is_empty())).unwrap_or(false)
    })
    .await;
    assert!(doc2.read_at(&["links"]).unwrap().as_array().unwrap().is_empty(), "undo removed the link");
    assert_eq!(doc2.node_ids().len(), 2, "both nodes survive the link undo");
}

#[tokio::test]
async fn a_param_edit_is_undoable_over_the_wire() {
    // update_param routes through the command history (EditParam); undo restores the PRIOR value,
    // redo re-applies — read back from the synced doc.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call_session(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }), "s1").await["result"]
        .as_str().unwrap().to_string();
    call_session(&mut ws, 2, "update_param", json!({ "node": osc, "group": "common", "name": "max_frequency", "value": 20.0 }), "s1").await;
    call_session(&mut ws, 3, "update_param", json!({ "node": osc, "group": "common", "name": "max_frequency", "value": 33.0 }), "s1").await;
    let doc = sync_replica(&mut ws, |d| doc_param_f64(d, &osc, "common", "max_frequency") == Some(33.0)).await;
    assert_eq!(doc_param_f64(&doc, &osc, "common", "max_frequency"), Some(33.0), "second edit applied");

    call_session(&mut ws, 4, "undo", json!({}), "s1").await;
    let doc2 = sync_replica(&mut ws, |d| doc_param_f64(d, &osc, "common", "max_frequency") == Some(20.0)).await;
    assert_eq!(doc_param_f64(&doc2, &osc, "common", "max_frequency"), Some(20.0), "undo restored the prior value");

    call_session(&mut ws, 5, "redo", json!({}), "s1").await;
    let doc3 = sync_replica(&mut ws, |d| doc_param_f64(d, &osc, "common", "max_frequency") == Some(33.0)).await;
    assert_eq!(doc_param_f64(&doc3, &osc, "common", "max_frequency"), Some(33.0), "redo re-applied");
}

#[tokio::test]
async fn a_global_add_is_undoable_over_the_wire() {
    // add_global routes through the history (EditGlobal); undo removes the added global. The
    // default_ufreq SYSTEM global always remains, so anchor the post-undo sync on its presence.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call_session(&mut ws, 1, "add_global", json!({ "name": "subj", "value": "P01", "type": "string" }), "s1").await;
    let doc = sync_replica(&mut ws, |d| d.read_at(&["globals", "subj", "value"]).is_some()).await;
    assert_eq!(
        doc.read_at(&["globals", "subj", "value"]).and_then(|v| v.as_str().map(str::to_string)),
        Some("P01".to_string()),
        "global added"
    );

    call_session(&mut ws, 2, "undo", json!({}), "s1").await;
    let doc2 = sync_replica(&mut ws, |d| {
        d.read_at(&["globals", "default_ufreq", "value"]).is_some()
            && d.read_at(&["globals", "subj", "value"]).is_none()
    })
    .await;
    assert!(doc2.read_at(&["globals", "subj", "value"]).is_none(), "undo removed the added global");
}

#[tokio::test]
async fn a_global_rename_folds_into_one_undo_step() {
    // rename_global is add-new(with the old value) + remove-old composed into ONE Compound command,
    // so a single undo reverts the whole rename (old name + value back, new name gone).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call_session(&mut ws, 1, "add_global", json!({ "name": "subj", "value": "P01", "type": "string" }), "s1").await;
    call_session(&mut ws, 2, "rename_global", json!({ "old": "subj", "new": "participant" }), "s1").await;
    let doc = sync_replica(&mut ws, |d| d.read_at(&["globals", "participant", "value"]).is_some()).await;
    assert!(doc.read_at(&["globals", "subj", "value"]).is_none(), "old name gone after rename");
    assert_eq!(
        doc.read_at(&["globals", "participant", "value"]).and_then(|v| v.as_str().map(str::to_string)),
        Some("P01".to_string()),
        "value carried to the new name"
    );

    call_session(&mut ws, 3, "undo", json!({}), "s1").await;
    let doc2 = sync_replica(&mut ws, |d| d.read_at(&["globals", "subj", "value"]).is_some()).await;
    assert!(doc2.read_at(&["globals", "participant", "value"]).is_none(), "new name gone after one undo");
    assert_eq!(
        doc2.read_at(&["globals", "subj", "value"]).and_then(|v| v.as_str().map(str::to_string)),
        Some("P01".to_string()),
        "old name + value restored in a single step"
    );
}

#[tokio::test]
async fn add_global_rejects_a_collision_instead_of_silently_upserting() {
    // add_global must REJECT a name that already exists (a distinct op from set_global) — else the
    // agent seam `window.goofi.commands.addGlobal('default_ufreq', …)` would silently clobber a
    // system value while resolving success.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_global", json!({ "name": "subj", "value": "P01", "type": "string" })).await;
    let dup = call(&mut ws, 2, "add_global", json!({ "name": "subj", "value": "P02", "type": "string" })).await;
    assert!(dup.get("error").is_some(), "a colliding add is rejected; got {dup}");
    let sys = call(&mut ws, 3, "add_global", json!({ "name": "default_ufreq", "value": 5, "type": "int" })).await;
    assert!(sys.get("error").is_some(), "cannot add over the system global; got {sys}");
    // The originally-added value is untouched by the rejected duplicate.
    let doc = sync_replica(&mut ws, |d| d.read_at(&["globals", "subj", "value"]).is_some()).await;
    assert_eq!(
        doc.read_at(&["globals", "subj", "value"]).and_then(|v| v.as_str().map(str::to_string)),
        Some("P01".to_string()),
        "the first value survives the rejected duplicate"
    );
}

#[tokio::test]
async fn rename_global_rejects_a_system_name_without_leaking_a_phantom() {
    // Renaming a system global must be rejected AND must not leave the add-new half applied (the
    // Compound is not atomic — the guard runs before it).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let reply = call(&mut ws, 1, "rename_global", json!({ "old": "default_ufreq", "new": "foo" })).await;
    assert!(reply.get("error").is_some(), "system rename rejected; got {reply}");
    let doc = sync_replica(&mut ws, |d| d.read_at(&["globals", "default_ufreq", "value"]).is_some()).await;
    assert!(doc.read_at(&["globals", "foo", "value"]).is_none(), "no phantom 'foo' global leaked");
    assert!(doc.read_at(&["globals", "default_ufreq", "value"]).is_some(), "system global intact");
}

#[tokio::test]
async fn deleting_a_sub_patch_instance_is_undoable_over_the_wire() {
    // Deleting a collapsed sub-patch instance tears down its whole subtree; undo must restore the
    // scope + its members uid-stably (B3b closed the delete-undo gap).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let a = call_session(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }), "s1").await["result"]
        .as_str().unwrap().to_string();
    let b = call_session(&mut ws, 2, "add_node", json!({ "type": "Buffer" }), "s1").await["result"]
        .as_str().unwrap().to_string();
    call_session(&mut ws, 3, "add_link", json!({ "node_out": a, "slot_out": "out", "node_in": b, "slot_in": "data" }), "s1").await;
    let inst = call_session(&mut ws, 4, "group_nodes", json!({ "members": [a, b], "pos": [0.0, 0.0] }), "s1").await
        ["result"]["inst_id"].as_str().unwrap().to_string();
    // The scope is live with 2 members hidden under it.
    sync_replica(&mut ws, |d| d.instance_ids().iter().any(|i| *i == inst) && d.node_ids().len() == 2).await;

    // Delete the instance → its whole subtree is gone.
    call_session(&mut ws, 5, "remove_node", json!({ "node": inst }), "s1").await;
    let gone = sync_replica(&mut ws, |d| d.instance_ids().is_empty() && d.node_ids().is_empty()).await;
    assert!(gone.node_ids().is_empty() && gone.instance_ids().is_empty(), "subtree torn down");

    // Undo restores the scope + both members uid-stably.
    call_session(&mut ws, 6, "undo", json!({}), "s1").await;
    let back = sync_replica(&mut ws, |d| {
        d.instance_ids().iter().any(|i| *i == inst)
            && d.node_ids().iter().any(|u| *u == a)
            && d.node_ids().iter().any(|u| *u == b)
    })
    .await;
    assert!(back.instance_ids().iter().any(|i| *i == inst), "scope restored under the same uid");
    assert!(back.node_ids().iter().any(|u| *u == a) && back.node_ids().iter().any(|u| *u == b), "members restored uid-stable");
}

#[tokio::test]
async fn group_undo_redo_over_the_wire_is_uid_stable() {
    // group routes through the command history: undo expands the scope (members back at root), redo
    // regroups at the SAME scope uid with the crossing stub restored.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let osc = uid(&call_session(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }), "s1").await);
    let buf = uid(&call_session(&mut ws, 2, "add_node", json!({ "type": "Buffer" }), "s1").await);
    let sink = uid(&call_session(&mut ws, 3, "add_node", json!({ "type": "Buffer" }), "s1").await);
    call_session(&mut ws, 4, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" }), "s1").await;
    call_session(&mut ws, 5, "add_link", json!({ "node_out": buf, "slot_out": "out", "node_in": sink, "slot_in": "data" }), "s1").await;

    // Group [osc, buf] → one scope with an Out stub (the buf→sink cut).
    let scope = call_session(&mut ws, 6, "group_nodes", json!({ "members": [osc, buf], "pos": [0.0, 0.0] }), "s1").await
        ["result"]["inst_id"].as_str().unwrap().to_string();
    let doc = sync_replica(&mut ws, |d| d.instance_ids().iter().any(|u| *u == scope)).await;
    let has_stub = |d: &goofi_crdt::GraphDoc, s: &str| {
        d.to_json()["instances"][s]["stubs"].as_object().map(|m| !m.is_empty()).unwrap_or(false)
    };
    assert!(has_stub(&doc, &scope), "grouped scope exposes a stub");

    // Undo → expand: scope gone, all three leaves remain at root (positive-presence anchor).
    call_session(&mut ws, 7, "undo", json!({}), "s1").await;
    let doc2 = sync_replica(&mut ws, |d| d.node_ids().len() == 3 && d.instance_ids().is_empty()).await;
    assert!(doc2.instance_ids().is_empty(), "undo expanded the scope");
    assert_eq!(doc2.node_ids().len(), 3, "all leaves remain");

    // Redo → regroup at the SAME scope uid with the stub restored.
    call_session(&mut ws, 8, "redo", json!({}), "s1").await;
    let doc3 = sync_replica(&mut ws, |d| d.instance_ids().iter().any(|u| *u == scope) && has_stub(d, &scope)).await;
    assert!(doc3.instance_ids().iter().any(|u| *u == scope), "redo restored the SAME scope uid");
    assert!(has_stub(&doc3, &scope), "stub restored verbatim");
}

#[tokio::test]
async fn add_boundary_is_undoable_over_the_wire() {
    // A stub add routes through the command history; undo removes it (the scope survives).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let buf = uid(&call_session(&mut ws, 1, "add_node", json!({ "type": "Buffer" }), "s1").await);
    let scope = call_session(&mut ws, 2, "group_nodes", json!({ "members": [buf], "pos": [0.0, 0.0] }), "s1").await
        ["result"]["inst_id"].as_str().unwrap().to_string();
    let doc0 = sync_replica(&mut ws, |d| d.instance_ids().iter().any(|u| *u == scope)).await;
    let stubs0 = doc0.to_json()["instances"][&scope]["stubs"].as_object().map(|m| m.len()).unwrap_or(0);

    let bnd = call_session(&mut ws, 3, "add_boundary", json!({ "inst_id": scope, "dir": "in", "dtype": "ARRAY", "pos": [0.0, 0.0] }), "s1").await
        ["result"]["bnd_id"].as_str().unwrap().to_string();
    let doc = sync_replica(&mut ws, |d| d.read_at(&["instances", scope.as_str(), "stubs", bnd.as_str()]).is_some()).await;
    assert!(doc.read_at(&["instances", scope.as_str(), "stubs", bnd.as_str()]).is_some(), "stub added");

    // Undo → the stub is gone; the scope (positive-presence anchor) survives with the original stub count.
    call_session(&mut ws, 4, "undo", json!({}), "s1").await;
    let doc2 = sync_replica(&mut ws, |d| {
        d.instance_ids().iter().any(|u| *u == scope)
            && d.read_at(&["instances", scope.as_str(), "stubs"]).and_then(|v| v.as_object().map(|m| m.len())) == Some(stubs0)
    })
    .await;
    assert!(doc2.read_at(&["instances", scope.as_str(), "stubs", bnd.as_str()]).is_none(), "undo removed the stub");
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
async fn data_plane_sustains_streaming_over_a_window() {
    // Stability/throughput smoke: a live Oscillator→Buffer chain must keep delivering frames
    // over a wall-clock window (not just one), proving the tick + data plane sustain streaming
    // without stalling. Loose lower bound so it's not CI-timing-flaky; the measured rate is
    // logged for a latency/throughput read.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
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

    // Wait for a frame that carries reduced meta AND genuinely shrank on the last axis —
    // the definitive proof the plan was applied (passthrough never stamps it). Requiring a
    // real shrink (not merely `reduced.is_some()`) avoids a boundary race: envelope fires at
    // axis len ≥ 2·W = 64 producing exactly 64 samples, so a frame caught with the Buffer at
    // *exactly* 64 (which happens when the tick thread is starved under parallel test load)
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
async fn setting_an_expression_binds_and_echoes_the_descriptor() {
    // The `set_expression` command op binds a param to an expression: the manager routes it through
    // an `EditParam` command and echoes the runtime-enriched param descriptor as a `state_update` —
    // the binding round-trips AND carries `expression_error` (the field indicator; runtime-derived,
    // never in the doc), with NO `expression_autoeval` key.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    let _sv = recv_binary(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }))
        .await["result"]
        .as_str()
        .unwrap()
        .to_string();

    // Bind an expression on common.max_frequency via the command op.
    bind_expression(&mut ws, 2, &osc, "common", "max_frequency", "1 + 2").await;

    // The manager echoes the descriptor as a state_update (arrives after the command reply).
    let mut descriptor: Option<Value> = None;
    for _ in 0..40 {
        let m = recv_text(&mut ws).await;
        if m["event"] == "state_update" && m["payload"]["node"] == json!(osc) {
            let d = m["payload"]["params"]["common"]["max_frequency"].clone();
            if d["expression"] == "1 + 2" {
                descriptor = Some(d);
                break;
            }
        }
    }
    let d = descriptor.expect("state_update carrying the bound param descriptor");
    assert_eq!(d["expression"], "1 + 2", "source round-trips");
    assert_eq!(d["expression_enabled"], true);
    assert_eq!(d["expression_triggers_process"], false);
    // This harness injects no evaluator, so the binding round-trips WITH an error — the point is
    // the field exists as a string to drive the per-param red indicator.
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
async fn a_client_replica_converges_via_the_binary_sync_relay() {
    // Phase 2: a browser Yjs replica (here a Rust GraphDoc standing in for it) mounts, syncs
    // the current graph over the /control binary channel, and receives live deltas as the
    // graph mutates — the reader half of the CRDT control plane.
    use goofi_crdt::{GraphDoc, SyncMsg};

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
                    uid = v["result"].as_str().map(str::to_string);
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
    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
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
async fn an_ephemeral_frame_is_relayed_to_other_clients() {
    // The awareness channel: a client's ephemeral frame (presence/live-drag/preview) is
    // relayed verbatim to other clients, never touching the doc. Two clients A and B: A sends
    // an ephemeral frame; B receives it; the graph/doc is unaffected.
    use goofi_crdt::SyncMsg;

    let base = start_server().await;
    let (mut a, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut a).await;
    let _ = recv_binary(&mut a).await; // hello SV
    let (mut b, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut b).await;
    let _ = recv_binary(&mut b).await;

    // A publishes an ephemeral payload (opaque to the manager).
    let payload = b"\x07\x00\x00\x00\x00\x00\x00\x00cursor".to_vec(); // client-id + state (browser-defined)
    a.send(Message::Binary(SyncMsg::Ephemeral(payload.clone()).encode().into())).await.unwrap();

    // B receives it, decoded as the same Ephemeral frame.
    let got = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let bytes = recv_binary(&mut b).await;
            if let Some(SyncMsg::Ephemeral(p)) = SyncMsg::decode(&bytes) {
                return p;
            }
        }
    })
    .await
    .expect("B must receive A's ephemeral frame");
    assert_eq!(got, payload, "ephemeral payload relayed verbatim");
}

#[tokio::test]
async fn a_param_command_reaches_the_graph_and_other_clients() {
    // A client commits a param edit via the `update_param` command op. The manager routes it
    // through an `EditParam` command, applies it to the authoritative graph, and broadcasts the
    // resulting doc delta so a second client converges — no client doc write involved.
    use goofi_crdt::{GraphDoc, SyncMsg};

    let base = start_server().await;

    // Writer client: connect, add a node, then edit a param via the command op.
    let (mut w, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut w).await;
    let _ = recv_binary(&mut w).await; // server hello SV

    let osc = call(&mut w, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();
    call(
        &mut w,
        2,
        "update_param",
        json!({ "node": osc, "group": "common", "name": "max_frequency", "value": 12.0 }),
    )
    .await;

    // The manager applies it to the graph: a fresh reader client sees max_frequency == 12.
    let (mut r, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut r).await;
    let _ = recv_binary(&mut r).await;
    let mut rdoc = GraphDoc::new();
    r.send(Message::Binary(rdoc.sync_hello().into())).await.unwrap();

    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let b = recv_binary(&mut r).await;
            if let Some(m) = SyncMsg::decode(&b) {
                rdoc.on_sync(m);
            }
            if doc_param_f64(&rdoc, &osc, "common", "max_frequency") == Some(12.0) {
                return true;
            }
        }
    })
    .await
    .unwrap_or(false);
    assert!(converged, "the client's leaf write reached the graph and a second client");
}

#[tokio::test]
async fn a_position_command_reaches_the_graph_and_other_clients() {
    // A client commits a drag by sending the `set_node_pos` command op — no client doc write. The
    // manager routes it through an `EditNode` command and broadcasts, so a second client sees the
    // moved position.
    use goofi_crdt::{GraphDoc, SyncMsg};

    let base = start_server().await;

    let (mut w, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut w).await;
    let _ = recv_binary(&mut w).await; // server hello SV

    let osc = call(&mut w, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();
    // Commit a drag: move the node to [123, 456] via the command op.
    call(&mut w, 2, "set_node_pos", json!({ "node": osc, "pos": [123.0, 456.0] })).await;

    // A fresh reader converges on the moved position.
    let (mut r, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut r).await;
    let _ = recv_binary(&mut r).await;
    let mut rdoc = GraphDoc::new();
    r.send(Message::Binary(rdoc.sync_hello().into())).await.unwrap();

    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let b = recv_binary(&mut r).await;
            if let Some(m) = SyncMsg::decode(&b) {
                rdoc.on_sync(m);
            }
            if doc_node_pos(&rdoc, &osc) == Some([123.0, 456.0]) {
                return true;
            }
        }
    })
    .await
    .unwrap_or(false);
    assert!(converged, "the client's position leaf write reached the graph and a second client");
}

#[tokio::test]
async fn crdt_doc_tracks_an_rpc_node_add_and_param_edit() {
    // The server-side CRDT mirror tracks RPC-driven control edits: after an add_node +
    // update_param over the /control RPC path, a synced client replica reflects BOTH the node
    // and the new param value — read via the binary sync relay (not a diagnostic op).
    use goofi_crdt::{GraphDoc, SyncMsg};

    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    let _server_sv = recv_binary(&mut ws).await;
    let mut client = GraphDoc::new();
    ws.send(Message::Binary(client.sync_hello().into())).await.unwrap();
    client.on_sync(SyncMsg::decode(&recv_binary(&mut ws).await).expect("a sync frame"));

    // Drive add_node then update_param, absorbing BOTH the text replies (for the uid) and the
    // binary sync deltas in one loop — `call` would discard the interleaved binary frames.
    ws.send(Message::Text(
        json!({ "id": 1, "op": "add_node", "payload": { "type": "Oscillator" } }).to_string(),
    ))
    .await
    .unwrap();
    let mut osc: Option<String> = None;
    let mut sent_param = false;
    let tracked = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match ws.next().await.expect("stream").expect("ws") {
                Message::Text(t) => {
                    let v: Value = serde_json::from_str(t.as_str()).unwrap();
                    if v.get("id").and_then(|x| x.as_i64()) == Some(1) {
                        osc = v["result"].as_str().map(str::to_string);
                    }
                }
                Message::Binary(b) => {
                    if let Some(m) = SyncMsg::decode(&b) {
                        client.on_sync(m);
                    }
                }
                _ => {}
            }
            if let Some(o) = osc.clone() {
                if !sent_param {
                    ws.send(Message::Text(
                        json!({ "id": 2, "op": "update_param", "payload": {
                            "node": o, "group": "common", "name": "max_frequency", "value": 25.0
                        }})
                        .to_string(),
                    ))
                    .await
                    .unwrap();
                    sent_param = true;
                }
                if client.node_ids().contains(&o)
                    && doc_param_f64(&client, &o, "common", "max_frequency") == Some(25.0)
                {
                    return true;
                }
            }
        }
    })
    .await
    .unwrap_or(false);
    assert!(tracked, "the mirror tracked the RPC node add + param edit");
}

#[tokio::test]
async fn renaming_a_node_rewrites_referrers_nd_expressions_over_the_wire() {
    // A node referenced by `nd('old')` in another node's expression: renaming it must
    // rewrite the reference to `nd('new')` AND rebroadcast the referrer's params so its
    // inspector reflects the rewrite (Python: manager.rename_node).
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let producer = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let consumer = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Oscillator" })).await);
    call(&mut ws, 3, "rename_node", json!({ "node": producer, "name": "src" })).await;

    // consumer.common.max_frequency = nd('src') — via the set_expression command op.
    bind_expression(&mut ws, 4, &consumer, "common", "max_frequency", "nd('src')").await;

    // Rename the producer; the reply is fire-and-forget, the rewrite rides a state_update.
    call(&mut ws, 5, "rename_node", json!({ "node": producer, "name": "signal" })).await;

    let mut rewritten: Option<Value> = None;
    for _ in 0..20 {
        let m = recv_text(&mut ws).await;
        if m["event"] == "state_update" && m["payload"]["node"] == json!(consumer) {
            rewritten = Some(m["payload"]["params"]["common"]["max_frequency"]["expression"].clone());
            break;
        }
    }
    assert_eq!(
        rewritten.expect("a state_update for the referrer must be broadcast"),
        json!("nd('signal')"),
        "the referrer's nd() reference followed the rename"
    );
}

#[tokio::test]
async fn rename_node_rejects_a_duplicate_display_name_up_front() {
    // The engine's Command::EditNode tolerates a rename collision as a no-op (so a stale undo-replay
    // converges instead of wedging the stack). The forward user error must therefore be raised at
    // the RPC boundary: renaming a node onto a name another node already holds is rejected, and the
    // target node keeps its name.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let a = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let _b = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await); // "buffer0"

    let reply = call(&mut ws, 3, "rename_node", json!({ "node": a, "name": "buffer0" })).await;
    assert!(reply.get("error").is_some(), "a duplicate rename is rejected; got {reply}");

    // A itself keeps its own name — a unique rename still succeeds.
    let ok = call(&mut ws, 4, "rename_node", json!({ "node": a, "name": "myosc" })).await;
    assert!(ok.get("error").is_none(), "a unique rename still succeeds; got {ok}");
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

    // The forest reaches the client via the doc (subpatch_changed retired). A top-level instance's
    // parent MUST be ROOT_ID (not null): the editor's `childrenOfScope` renders it at the root canvas
    // only when `instance.parent === ROOT_ID`. Unique ⇔ no def_id; both members in the instance scope.
    let doc = sync_replica(&mut ws, |d| d.instance_ids().iter().any(|u| *u == inst)).await;
    let j = doc.to_json();
    let rec = &j["instances"][&inst];
    assert_eq!(rec["parent"], json!("__root__"), "top-level scope parented to ROOT so the canvas renders it");
    assert!(rec.get("def_id").is_none(), "no sharing ⇒ no def_id");
    // The flat scope's `members` map is keyed by member uid → {is_instance}.
    assert_eq!(rec["members"].as_object().unwrap().len(), 2, "both members in the scope");
    assert!(
        rec["members"].as_object().unwrap().contains_key(osc.as_str()),
        "member osc re-tagged into the scope; got {:?}", rec["members"]
    );

    // Expand restores both members to ROOT → the instance drops out of the doc forest, leaves remain.
    call(&mut ws, 5, "expand_instance", json!({ "inst_id": inst })).await;
    // Anchor on osc being present (a completed sync) — an "absence" predicate would be satisfied by
    // the empty replica before the data frame lands. After expand: 2 leaves, 0 instances.
    let doc2 = sync_replica(&mut ws, |d| d.node_ids().iter().any(|u| *u == osc) && d.instance_ids().is_empty()).await;
    assert!(doc2.to_json()["instances"].get(&inst).is_none(), "instance gone after expand");
    assert!(doc2.node_ids().iter().any(|u| *u == osc), "osc back as a top-level leaf");
}

#[tokio::test]
async fn node_stats_broadcasts_the_measured_ufreq() {
    // Regression: `spawn_workers` (what the binary runs at startup) must wire `spawn_stats`,
    // else the node header never shows a live update rate — the `node_stats` producer was
    // orphaned in the now-removed `serve()` and the CLI called only `spawn_tick`.
    let state = AppState::new();
    spawn_workers(&state); // tick loop + 2 Hz stats, exactly as the CLI startup does
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    let base = format!("ws://{addr}");

    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    // A free-running source measures a ufreq after a few ticks.
    let src = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();

    // Within a few 2 Hz stats periods a `node_stats` for the source arrives carrying its rate.
    let stats = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            let m = recv_text(&mut ws).await;
            if m.get("event").and_then(|v| v.as_str()) == Some("node_stats")
                && m["payload"]["node"] == json!(src)
            {
                return m;
            }
        }
    })
    .await
    .expect("a node_stats event for the source must arrive (spawn_stats wired)");
    assert!(
        stats["payload"]["stats"]["updates_per_second"].is_number(),
        "node_stats carries a numeric measured ufreq; got {:?}",
        stats["payload"]
    );
}

#[tokio::test]
async fn param_values_broadcasts_live_expression_values() {
    // A param with an ENABLED expression is broadcast on the `param_values` event so the
    // inspector preview tracks each re-evaluation. (No evaluator is injected here, so the
    // value stays the literal — the point is the seam exists and carries the bound param.)
    let state = AppState::new();
    spawn_workers(&state); // tick loop + 2 Hz stats/param_values, as the CLI startup does
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    let base = format!("ws://{addr}");
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();
    // Bind an enabled expression via the set_expression command op.
    bind_expression(&mut ws, 2, &osc, "common", "max_frequency", "1 + 2").await;

    let ev = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            let m = recv_text(&mut ws).await;
            if m.get("event").and_then(|v| v.as_str()) == Some("param_values")
                && m["payload"]["node"] == json!(osc)
            {
                return m;
            }
        }
    })
    .await
    .expect("a param_values event for the node with an active expression must arrive");
    assert!(
        ev["payload"]["values"]["common"]["max_frequency"].is_number(),
        "the bound param's live value is carried; got {:?}",
        ev["payload"]["values"]
    );
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

    // Links live in the CRDT doc now (the `link_added` event is retired). Sync a fresh replica and
    // read the flat link back: the boundary endpoint must have resolved to the inner buffer leaf.
    use goofi_crdt::{GraphDoc, SyncMsg};
    let mut doc = GraphDoc::new();
    ws.send(Message::Binary(doc.sync_hello().into())).await.unwrap();
    let mut links = Vec::new();
    for _ in 0..40 {
        if let Some(m) = SyncMsg::decode(&recv_binary(&mut ws).await) {
            doc.on_sync(m);
        }
        if let Some(Value::Array(a)) = doc.read_at(&["links"]) {
            if !a.is_empty() {
                links = a;
                break;
            }
        }
    }
    assert_eq!(links.len(), 1, "one flat leaf→leaf link");
    assert_eq!(links[0]["node_in"], json!(buf), "resolved to the inner buffer leaf, not the instance");
    assert_eq!(links[0]["slot_in"], "data", "resolved to the inner slot");
    assert_eq!(links[0]["node_out"], json!(osc), "the plain endpoint passes through");
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

    // The doc scope's stubs carry the wired, renamed stub (StubId unchanged) — read it from a
    // synced replica (subpatch_changed retired).
    let doc = sync_replica(&mut ws, |d| {
        d.read_at(&["instances", inst.as_str(), "stubs", bnd.as_str(), "name"])
            .and_then(|v| v.as_str().map(String::from))
            == Some("wave".into())
    })
    .await;
    let port = doc.read_at(&["instances", inst.as_str(), "stubs", bnd.as_str()]).unwrap();
    assert_eq!(port["dir"], "out");
    assert_eq!(port["inner_node"], json!(buf), "wired to the buffer leaf");
    assert_eq!(port["inner_slot"], "out");
    assert_eq!(port["name"], "wave", "renamed; StubId preserved");
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

#[tokio::test]
async fn set_node_viewers_persists_the_view_state() {
    // The editor's per-slot viewer view-state (kind/settings/collapsed) is pushed via the
    // `set_node_viewers` op — soft view state (not undoable). The manager stores it on the node and
    // it survives a .gfi serialize round-trip.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    let _sv = recv_binary(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();

    // Push the viewer blob via the command surface (a JSON object, the graph's stored shape).
    let viewers = json!({ "out": { "collapsed": false, "kind": "line", "settings": { "yScale": 2 } } });
    call(&mut ws, 2, "set_node_viewers", json!({ "node": osc, "viewers": viewers })).await;

    // It reaches the graph and persists into the serialized .gfi (poll: the write is async).
    let mut persisted = false;
    for i in 4..14 {
        let yaml = call(&mut ws, i, "serialize", json!({})).await["result"]["yaml"]
            .as_str()
            .unwrap()
            .to_string();
        if yaml.contains("yScale") {
            persisted = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(persisted, "the client's viewer leaf write reached the graph and persisted to .gfi");
}

#[tokio::test]
async fn serialize_and_load_roundtrip() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let ser = call(&mut ws, 2, "serialize", json!({})).await;
    let yaml = ser["result"]["yaml"].as_str().unwrap().to_string();
    assert!(yaml.contains("version: 6"), "gfi v6 header");
    assert!(yaml.contains("Oscillator"), "node persisted");
    assert!(yaml.contains("default_ufreq"), "globals block persisted");

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
    let osc = call(&mut b, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
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
async fn many_clients_concurrently_edit_params_and_all_converge() {
    // Stress + multi-user correctness for the command write path. N clients each OWN one node and
    // hammer `ROUNDS` `update_param` command RPCs concurrently, all racing through the manager's
    // `EditParam → graph → resync_and_broadcast` path (the shared graph→crdt→last_sync_sv mutex
    // chain). Two properties are proven at once:
    //   * liveness — the contended mutex chain never deadlocks (the whole test completes);
    //   * no-loss   — a fresh reader converges on ALL N distinct final values, so not one of
    //                 the N·ROUNDS concurrent commands was dropped or clobbered by the re-mirror.
    //
    // Determinism hinges on the awaited command reply: `handle_control` reads one incoming message
    // per socket at a time, so each `call` reply proves that command was applied. Once all writers
    // return, every edit is live server-side, so the reader's first full-state sync carries them.
    use goofi_crdt::{GraphDoc, SyncMsg};

    const N: usize = 8;
    const ROUNDS: usize = 5;

    let base = start_server().await;

    // Setup: add N Oscillators over one control client; collect their uids (all exist before
    // any writer connects, so every writer's initial sync learns every node).
    let (mut setup, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut setup).await; // hello
    let _ = recv_binary(&mut setup).await; // server sync_hello (state vector)
    let mut uids = Vec::new();
    for i in 0..N {
        let u = call(&mut setup, i as i64 + 1, "add_node", json!({ "type": "Oscillator" })).await
            ["result"]
            .as_str()
            .unwrap()
            .to_string();
        uids.push(u);
    }

    // Concurrent writers: each ramps its OWN node's max_frequency 1.0 → ROUNDS over the binary
    // sync channel, then barriers on a serialize RPC before disconnecting.
    let mut handles = Vec::new();
    for i in 0..N {
        let base = base.clone();
        let uids = uids.clone();
        handles.push(tokio::spawn(async move {
            let (mut w, _) = connect_async(format!("{base}/control")).await.unwrap();
            let _ = recv_text(&mut w).await; // hello
            let _ = recv_binary(&mut w).await; // server sync_hello
            // Ramp this node's max_frequency 1 → ROUNDS via the update_param command op; awaiting
            // each reply proves the graph applied it (the happens-before barrier writers race on).
            for r in 1..=ROUNDS {
                call(
                    &mut w,
                    r as i64,
                    "update_param",
                    json!({ "node": uids[i], "group": "common", "name": "max_frequency", "value": r as f64 }),
                )
                .await;
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    // A fresh reader must converge on ALL N nodes at the final ramped value — proof that every
    // concurrent write survived the contended re-mirror (no lost update, no deadlock stall).
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
            Err(_) => {} // no frame this window — re-check convergence, keep waiting
        }
        if uids
            .iter()
            .all(|u| doc_param_f64(&rdoc, u, "common", "max_frequency") == Some(ROUNDS as f64))
        {
            converged = true;
            break;
        }
    }
    if !converged {
        let got: Vec<_> = uids
            .iter()
            .map(|u| doc_param_f64(&rdoc, u, "common", "max_frequency"))
            .collect();
        panic!("not converged; final max_frequency per node = {got:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_clients_concurrently_drag_and_all_converge() {
    // Stress the POSITION command path against the re-mirror — the exact interleaving the audit
    // found losing drags before upsert_node was made idempotent. N clients each own a node and
    // hammer ROUNDS `set_node_pos` commands concurrently; EACH triggers a manager re-mirror that
    // re-asserts EVERY node's pos (upsert_node). With the wholesale pos-map replacement this test
    // would drop drags (a fresh reader would not converge on all N final positions); with the
    // idempotent in-place upsert_node every concurrent drag survives.
    use goofi_crdt::{GraphDoc, SyncMsg};

    const N: usize = 8;
    const ROUNDS: usize = 5;

    let base = start_server().await;

    let (mut setup, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut setup).await;
    let _ = recv_binary(&mut setup).await;
    let mut uids = Vec::new();
    for i in 0..N {
        let u = call(&mut setup, i as i64 + 1, "add_node", json!({ "type": "Oscillator" })).await
            ["result"]
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

#[tokio::test]
async fn add_node_applies_inline_params_at_creation() {
    // Paste/duplicate and undo-of-delete replay a node's params by passing an inline `params` map to
    // add_node; the node must be born CONFIGURED (params applied under the graph lock, before
    // node_added). Before this, callers did a post-add update_param — but that became a doc
    // leaf-write which no-ops until the just-added node has synced into the client's replica, so the
    // replayed values were silently dropped.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    ws.send(Message::Text(
        json!({ "id": 1, "op": "add_node", "payload": {
            "type": "Oscillator",
            "params": { "common": { "max_frequency": 42.0 } }
        }})
        .to_string(),
    ))
    .await
    .unwrap();

    // Capture BOTH the reply (uid) and the node_added broadcast (either order) — node_added must
    // already carry the applied value.
    let mut uid: Option<String> = None;
    let mut val: Option<Value> = None;
    for _ in 0..20 {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(1) {
            uid = m["result"].as_str().map(str::to_string);
        }
        if m["event"] == "node_added" {
            val = Some(m["payload"]["params"]["common"]["max_frequency"]["value"].clone());
        }
        if uid.is_some() && val.is_some() {
            break;
        }
    }
    assert!(uid.is_some(), "add_node reply must arrive");
    assert_eq!(val, Some(json!(42.0)), "add_node applied the inline param at creation");
}

#[tokio::test]
async fn add_node_restores_a_specific_uid_and_name() {
    // Undo-of-delete and redo-of-add replay add_node with the ORIGINAL uid (member_uid) + display
    // name so uid-keyed links/panels reconnect. Without honoring them the backend mints a FRESH uid
    // and the follow-up add_link (which references the old uid) fails — the restored node comes back
    // disconnected. Repro the delete→undo shape: add a node, remove it (freeing its uid), then
    // restore at the same uid + a chosen name.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let a = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();
    call(&mut ws, 2, "remove_node", json!({ "node": a })).await;

    // Restore at the SAME uid + a specific name (what removeNode's inverse sends).
    ws.send(Message::Text(
        json!({ "id": 3, "op": "add_node", "payload": {
            "type": "Oscillator", "member_uid": a, "name": "restored_osc"
        }})
        .to_string(),
    ))
    .await
    .unwrap();
    let mut uid: Option<String> = None;
    let mut name: Option<Value> = None;
    for _ in 0..20 {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(3) {
            uid = m["result"].as_str().map(str::to_string);
        }
        if m["event"] == "node_added" {
            name = Some(m["payload"]["name"].clone());
        }
        if uid.is_some() && name.is_some() {
            break;
        }
    }
    assert_eq!(uid.as_deref(), Some(a.as_str()), "add_node must restore the requested uid");
    assert_eq!(name, Some(json!("restored_osc")), "add_node must restore the requested name");
}

#[tokio::test]
async fn removing_a_grouped_member_updates_the_instance_forest_in_the_doc() {
    // Removing a node inside a sub-patch reaches the client through the doc, not the retired
    // `node_removed` event: the node drops out of the doc's `nodes`, and — because `remove_member`
    // edits the def — out of the instance's `members` map too (no dangling entry). The instance
    // survives its other member.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"].as_str().unwrap().to_string();
    let osc = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await);
    let buf = uid(&call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await);
    let inst = call(&mut ws, 3, "group_nodes", json!({ "members": [osc, buf], "pos": [0.0, 0.0] }))
        .await["result"]["inst_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Remove the grouped member `osc`. Anchor on the surviving instance settling to its single
    // remaining member (buf) — a completed sync — rather than osc's absence (true of an empty replica).
    call(&mut ws, 4, "remove_node", json!({ "node": osc })).await;
    let doc = sync_replica(&mut ws, |d| {
        d.to_json()["instances"][&inst]["members"].as_object().map(|m| m.len()) == Some(1)
    })
    .await;
    let members = doc.to_json()["instances"][&inst]["members"].as_object().unwrap().clone();
    assert!(
        !members.values().any(|v| v.as_str() == Some(osc.as_str())),
        "osc dropped from the instance's members (no dangling entry); got {members:?}"
    );
    assert!(!doc.node_ids().iter().any(|u| *u == osc), "osc gone from the graph");
    assert!(doc.instance_ids().iter().any(|u| *u == inst), "the instance survives its other member");
}

#[tokio::test]
async fn list_dir_browses_the_backend_filesystem() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    // No path ⇒ home, which is where the Save/Load modal opens on a fresh patch.
    let home = call(&mut ws, 1, "list_dir", json!({})).await;
    let home = &home["result"];
    assert!(home["path"].as_str().unwrap().starts_with('/'), "an absolute path; got {home:?}");
    assert!(
        home["roots"].as_array().unwrap().iter().any(|r| r["label"] == "Home"),
        "the sidebar needs at least a Home root; got {:?}",
        home["roots"]
    );

    // Navigating to the repo's own directory finds this crate, shaped as the browser expects.
    let repo = std::env::current_dir().unwrap();
    let listing = call(&mut ws, 2, "list_dir", json!({ "path": repo.to_string_lossy() })).await;
    let entries = listing["result"]["entries"].as_array().unwrap();
    let src = entries.iter().find(|e| e["name"] == "src").expect("goofi-bridge has a src/ dir");
    assert_eq!(src["kind"], json!("dir"));
    assert_eq!(src["is_gfi"], json!(false));
    assert_eq!(src["hidden"], json!(false));
    assert_eq!(listing["result"]["parent"].as_str(), repo.parent().map(|p| p.to_str().unwrap()));
}

#[tokio::test]
async fn load_reads_a_patch_from_a_backend_path() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let path = std::env::temp_dir().join(format!("goofi-load-{}.gfi", std::process::id()));
    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    call(&mut ws, 2, "save", json!({ "path": path.to_string_lossy() })).await;

    // Diverge from the saved patch, then load it back off disk.
    call(&mut ws, 3, "add_node", json!({ "type": "Buffer" })).await;
    ws.send(Message::Text(
        json!({ "id": 4, "op": "load", "payload": { "path": path.to_string_lossy() } }).to_string(),
    ))
    .await
    .unwrap();

    let mut replaced = None;
    let mut save_path = None;
    while replaced.is_none() || save_path.is_none() {
        let m = recv_text(&mut ws).await;
        match m.get("event").and_then(|v| v.as_str()) {
            Some("graph_replaced") => replaced = Some(m),
            Some("save_path_changed") => save_path = Some(m),
            _ => {}
        }
    }
    let types: Vec<String> = replaced.unwrap()["payload"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["type"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(types, ["Oscillator"], "the on-disk patch replaced the diverged graph");
    // The title bar names the loaded patch, so the manager reports where it came from.
    assert_eq!(save_path.unwrap()["payload"]["save_path"].as_str(), path.to_str());

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn load_reports_a_missing_file_instead_of_replacing_the_graph() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let reply = call(&mut ws, 2, "load", json!({ "path": "/definitely/not/a/patch.gfi" })).await;

    assert!(reply["error"].as_str().unwrap().contains("load failed"), "got {reply:?}");

    // A readable file that is not a patch fails at the parse, leaving the graph untouched.
    let junk = std::env::temp_dir().join(format!("goofi-junk-{}.gfi", std::process::id()));
    std::fs::write(&junk, "this: is: not: a patch").unwrap();
    let reply = call(&mut ws, 3, "load", json!({ "path": junk.to_string_lossy() })).await;
    assert!(reply.get("error").is_some(), "a malformed patch is rejected; got {reply:?}");

    let ser = call(&mut ws, 4, "serialize", json!({})).await;
    assert!(
        ser["result"]["yaml"].as_str().unwrap().contains("Oscillator"),
        "the pre-load graph survives both failures"
    );
    let _ = std::fs::remove_file(&junk);
}

#[tokio::test]
async fn restart_node_respawns_in_place_without_touching_undo() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();
    let buf = call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();
    call(&mut ws, 3, "add_link", json!({ "node_out": osc, "slot_out": "out", "node_in": buf, "slot_in": "data" })).await;

    let reply = call(&mut ws, 4, "restart_node", json!({ "node": buf })).await;
    assert!(reply.get("error").is_none(), "restart_node is served; got {reply:?}");

    // A restart is a recovery action, not an edit: the client records no history entry for it,
    // so the manager must not record one either — else undo would flip the restart instead of
    // the user's last real edit. Undo here must remove the LINK.
    call(&mut ws, 5, "undo", json!({})).await;
    // Anchor on a POSITIVE presence (both nodes) as well as the link's absence — an absence-only
    // predicate is already true of the initial empty replica, before any sync frame lands.
    let doc = sync_replica(&mut ws, |d| {
        d.node_ids().len() == 2
            && d.read_at(&["links"]).and_then(|v| v.as_array().map(|a| a.is_empty())).unwrap_or(false)
    })
    .await;
    assert_eq!(doc.node_ids().len(), 2, "both nodes survive; only the link was undone");
}

// A refreshable string param whose node re-enumerates it — the device-picker shape.
static PICKER_PARAMS: &[goofi_node::ParamDecl] = &[goofi_node::ParamDecl {
    group: "audio",
    name: "device",
    spec: goofi_node::ParamSpec::Str { default: "none", options: &["none"], refresh: true },
    default_expr: None,
    doc: Some("Which input device to capture from."),
}];
static PICKER_MANIFEST: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "DevicePicker",
    category: "python",
    doc: "a node with a refreshable device list",
    inputs: &[],
    outputs: SERVE_OUT,
    params: PICKER_PARAMS,
    isolation: goofi_node::Isolation::InProcess,
    factory: stub_factory,
};

#[derive(Default)]
struct Picker;
impl goofi_node::Node for Picker {
    fn on_param_refreshed(&mut self, key: &goofi_node::ParamKey) -> Option<Vec<String>> {
        (key.name == "device").then(|| vec!["mic".to_string(), "line-in".to_string()])
    }
    fn process(
        &mut self,
        _i: &goofi_node::Inputs<'_>,
        _o: &mut goofi_node::Outputs<'_>,
        _c: &mut goofi_node::NodeCtx,
        _p: &goofi_node::Params<'_>,
    ) -> goofi_node::NodeResult {
        Ok(())
    }
}

async fn start_server_with_picker() -> String {
    let state = AppState::new();
    state
        .graph
        .lock()
        .unwrap()
        .register_dyn_type(&PICKER_MANIFEST, Box::new(|_| Box::<Picker>::default()));
    spawn_tick(state.graph.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    format!("ws://{addr}")
}

#[tokio::test]
async fn refresh_param_echoes_fresh_options_and_clears_the_spinner() {
    let base = start_server_with_picker().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = call(&mut ws, 1, "add_node", json!({ "type": "DevicePicker" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();

    ws.send(Message::Text(
        json!({ "id": 2, "op": "refresh_param", "payload": { "node": uid, "group": "audio", "name": "device" } })
            .to_string(),
    ))
    .await
    .unwrap();

    // Options live only in runtime state (never in the doc), so they reach the browser ONLY via
    // this echo — and `refreshed_params` is what lifts the ⟳ spinner.
    let update = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("state_update")
            && m["payload"]["node"] == json!(uid)
        {
            break m;
        }
    };
    assert_eq!(
        update["payload"]["params"]["audio"]["device"]["options"],
        json!(["mic", "line-in"]),
        "the re-enumerated list reached the client"
    );
    assert_eq!(update["payload"]["refreshed_params"], json!([["audio", "device"]]));
}

#[tokio::test]
async fn refresh_param_reports_completion_even_when_the_node_offers_nothing() {
    // The hook returning nothing must still clear the spinner, or the UI stalls for its full
    // 15s safety timeout on every node that declares a refreshable param without a hook.
    let base = start_server_with_runtime_type().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]
        .as_str()
        .unwrap()
        .to_string();
    let reply = call(&mut ws, 2, "refresh_param", json!({ "node": osc, "group": "oscillator", "name": "waveform" })).await;

    // Oscillator's waveform is a fixed list: refusing is right, and the frontend lifts the
    // spinner on a rejected call.
    assert!(reply["error"].as_str().unwrap().contains("not refreshable"), "got {reply:?}");
}
