//! End-to-end protocol test: a real WebSocket client drives the bridge exactly
//! as the frontend would — receives `hello`, lists nodes, adds a node (and gets
//! the `node_added` broadcast), then subscribes to the data plane and receives a
//! decodable GOOF frame. Proves the M1 vertical slice (engine + control + data).

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::{serve_app, spawn_stats, spawn_workers, AppState};
use goofi_view::Reducible; // shape()/ndim() accessors on a decoded frame
use serde_json::{json, Value};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

/// A path as goofi spells it back: `/` on every platform (see `goofi_core::path`). A test that
/// compared against the platform's own spelling would pass on unix and pin the Windows bug — it
/// once asserted `C:\Users\…` against the `C:/Users/…` the wire actually carries.
fn spelled(p: &std::path::Path) -> String {
    goofi_core::path::to_slash(p)
}

// Read leaves through the generic CRDT reader (the typed getters were removed). A whole-number
// param comes back as an integer from `to_json`, so numeric reads compare via `as_f64`.
fn doc_param_f64(doc: &goofi_bridge::crdt::GraphDoc, uid: &str, group: &str, name: &str) -> Option<f64> {
    doc.read_at(&["nodes", uid, "params", group, name, "value"]).and_then(|v| v.as_f64())
}
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

/// The panel ids in a synced replica's arrangement root, in id order — how a client reads the flat
/// layout now that the manager owns it.
/// The arrangement's ENTRIES. The root also carries the manager's monotone id counter under a
/// reserved key, which no minted `{prefix}-{n}` id can take — a reader walks entries, not keys.
fn arrangement(doc: &goofi_bridge::crdt::GraphDoc) -> Value {
    let mut m = doc.to_json()["arrangement"].as_object().cloned().unwrap_or_default();
    m.retain(|_, e| e.get("kind").is_some());
    Value::Object(m)
}

fn entry_count(doc: &goofi_bridge::crdt::GraphDoc) -> usize {
    arrangement(doc).as_object().map_or(0, serde_json::Map::len)
}

/// A layout write ACCEPTED. Every one answers the arrangement it produced, so what a caller checks
/// is that something came back and nothing was refused — the two ops that mint an id answer that
/// instead, and one shape of "accepted" beats two.
fn accepted(r: &Value) -> bool {
    r.get("error").is_none() && r.get("result").is_some()
}

/// The page names `inspect_layout` draws, in tab order — the read that used to have an op of its
/// own. One line per page, `page \`name\`  [id]`, and the header holds no such line.
async fn page_names(ws: &mut Ws, id: i64) -> Vec<String> {
    let tree = call(ws, id, "inspect_layout", json!({})).await["result"]["text"]
        .as_str()
        .expect("inspect_layout answers text")
        .to_string();
    tree.lines()
        .filter_map(|l| Some(l.trim().strip_prefix("page `")?.split_once('`')?.0.to_string()))
        .collect()
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
async fn deleting_a_node_empties_the_panels_bound_to_it_and_an_undo_binds_them_back() {
    // A panel bound to a node that is gone renders empty and explains nothing, so the binding has
    // to go with the node. The manager does it INSIDE `RemoveNode` — the client used to, back when
    // it owned the layout, and doing it there now would cost a second command per delete and leave
    // a peer's copy of the panel pointing at a uid that no longer exists.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let uid = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .expect("the new node's uid")
        .to_string();
    let d = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let panel = panels(&d).first().cloned().expect("the default page's one panel");
    call(
        &mut ws,
        2,
        "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "type": "viewer", "state": { "node": uid } }),
    )
    .await;

    call(&mut ws, 3, "remove_node", json!({ "node": uid })).await;
    let after = call(&mut ws, 4, "inspect_layout", json!({})).await;
    let d = sync_replica(&mut ws, |d| d.node_ids().is_empty()).await;
    let state = d.read_at(&["arrangement", panel.as_str(), "state"]);
    assert_eq!(
        state.as_ref().and_then(Value::as_str),
        Some("{\"node\":null}"),
        "the deleted node's binding went with it: {state:?} / {after}"
    );

    // …and comes back with it: the manager owns the inverse, so ONE undo restores both.
    call(&mut ws, 5, "undo", json!({})).await;
    let d = sync_replica(&mut ws, |d| !d.node_ids().is_empty()).await;
    let state = d.read_at(&["arrangement", panel.as_str(), "state"]);
    assert_eq!(
        state.as_ref().and_then(Value::as_str),
        Some(format!("{{\"node\":\"{uid}\"}}").as_str()),
        "an undone delete re-binds the panel it emptied: {state:?}"
    );
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
async fn inspect_layout_narrows_to_one_page_and_states_each_entrys_share() {
    // `page_list_panels` was a SECOND read of the same fact in a second shape. What it genuinely
    // carried was cheapness: on a four-page patch the whole tree costs a caller three pages it did
    // not ask about. So the narrow read survives as an ARGUMENT — `inspect_layout {page}`, the
    // idiom `inspect_patch {scope}` already set — and there is one layout read again.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let first = panels(&doc)[0].clone();

    call(&mut ws, 1, "session_add_page", json!({ "name": "Signals" })).await;
    let fresh = call(&mut ws, 2, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "row", "ratio": 0.25 })).await
        ["result"].as_str().expect("the new panel's id").to_string();

    let text = |r: &Value| r["result"]["text"].as_str().expect("inspect_layout answers text").to_string();
    let whole = text(&call(&mut ws, 3, "inspect_layout", json!({})).await);
    assert!(whole.contains("Layout") && whole.contains("Signals"), "no arg is still every page: {whole}");

    let one = text(&call(&mut ws, 4, "inspect_layout", json!({ "page": "Layout" })).await);
    assert!(one.contains(&first) && one.contains(&fresh), "the page's own panels: {one}");
    assert!(!one.contains("Signals"), "…and NOT a page the caller did not ask about: {one}");
    // The share each entry takes of its parent — what `page_resize_split` sets, and the one thing
    // about the arrangement that its shape does not say.
    assert!(one.contains("0.25"), "each entry's share is annotated: {one}");

    // A page that is not there is refused by naming the ones that are, exactly as the page ops are.
    let miss = call(&mut ws, 5, "inspect_layout", json!({ "page": "Nope" })).await;
    assert!(
        miss["error"].as_str().is_some_and(|e| e.contains("Layout") && e.contains("Signals")),
        "{miss}"
    );
}

#[tokio::test]
async fn undo_of_a_layout_op_restores_the_arrangement_it_found() {
    // Layout undo is manager-owned now, in the same per-session history as a graph edit — so a
    // split ping-pongs uid-stably, exactly like an add.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let panel = panels(&doc)[0].clone();

    let fresh = call_session(
        &mut ws,
        1,
        "page_split_panel",
        json!({ "page": "Layout", "panel": panel, "direction": "column" }),
        "s1",
    )
    .await["result"]
        .as_str()
        .unwrap()
        .to_string();
    let after = sync_replica(&mut ws, |d| d.read_at(&["arrangement", fresh.as_str()]).is_some()).await;
    assert_eq!(entry_count(&after), 4, "page + split + 2 panels");

    let u = call_session(&mut ws, 2, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "undo flipped the layout entry: {u}");
    // The predicate has to be POSITIVE about something: a fresh replica trivially "lacks" the new
    // panel before it has synced a single frame, which would pass on an empty doc.
    let undone = sync_replica(&mut ws, |d| {
        !panels(d).is_empty() && d.read_at(&["arrangement", fresh.as_str()]).is_none()
    })
    .await;
    assert_eq!(panels(&undone), vec![panel.clone()], "the arrangement is exactly what it was");
    assert_eq!(entry_count(&undone), 2, "the wrapper split went too");

    call_session(&mut ws, 3, "redo", json!({}), "s1").await;
    let redone = sync_replica(&mut ws, |d| d.read_at(&["arrangement", fresh.as_str()]).is_some()).await;
    assert!(redone.read_at(&["arrangement", fresh.as_str()]).is_some(), "redo re-splits at the SAME id");
}

#[tokio::test]
async fn a_layout_undo_leaves_a_peers_panel_standing() {
    // TWO sessions, which is where this bug lives: a single session's undo is provably clean, so
    // nothing single-session can see it. Restoring the exact slots the split displaced deletes the
    // wrapper split a PEER has since hung a panel off — its panel is orphaned, the live arrangement
    // is one the manager's own loader refuses, and the peer's visible work is gone. The inverse is
    // therefore planned through the same close-with-promote the remove path uses, against the
    // arrangement as it stands at undo time.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let panel = panels(&doc)[0].clone();

    let mine = call_session(
        &mut ws,
        1,
        "page_split_panel",
        json!({ "page": "Layout", "panel": panel, "direction": "row" }),
        "s1",
    )
    .await["result"]
        .as_str()
        .expect("the new panel's id")
        .to_string();
    // The peer splits the panel A just made, so its own panel hangs off A's wrapper split.
    let theirs = call_session(
        &mut ws,
        2,
        "page_split_panel",
        json!({ "page": "Layout", "panel": mine, "direction": "row" }),
        "s2",
    )
    .await["result"]
        .as_str()
        .expect("the peer's new panel id")
        .to_string();

    let u = call_session(&mut ws, 3, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "undo flipped something: {u}");
    let after = sync_replica(&mut ws, |d| {
        !panels(d).is_empty() && d.read_at(&["arrangement", mine.as_str()]).is_none()
    })
    .await;
    let arrangement = after.to_json()["arrangement"].clone();
    let up = after
        .read_at(&["arrangement", theirs.as_str(), "parent"])
        .and_then(|v| v.as_str().map(str::to_string));
    assert!(
        up.as_deref().is_some_and(|p| arrangement.get(p).is_some()),
        "the peer's panel survived a foreign undo, still hanging off something: {arrangement}"
    );
    assert!(panels(&after).contains(&panel), "and so did the panel that was split");

    // The other birth is a PAGE, and there the semantics differ: undoing "add page" closes the page
    // WHOLE, so a peer's panel on it goes too — a lost update, but a convergent one, where restoring
    // the slots would leave the peer's panel hanging off a page that no longer exists.
    let standing = panels(&after);
    let entries = entry_count(&after);
    call_session(&mut ws, 6, "session_add_page", json!({ "name": "Second" }), "s1").await;
    let d = sync_replica(&mut ws, |d| panels(d).len() == standing.len() + 1).await;
    let second = panels(&d).into_iter().find(|p| !standing.contains(p)).expect("the new page's panel");
    call_session(&mut ws, 7, "page_split_panel",
        json!({ "page": "Second", "panel": second, "direction": "row" }), "s2").await;
    let u = call_session(&mut ws, 8, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");
    let closed = sync_replica(&mut ws, |d| {
        !panels(d).is_empty() && d.read_at(&["arrangement", second.as_str()]).is_none()
    })
    .await;
    let pages = page_names(&mut ws, 9).await;
    assert_eq!(pages.len(), 1, "the page went whole: {pages:?}");
    assert_eq!(entry_count(&closed), entries, "and took the peer's split with it, leaving no orphan");

    // …and it still REACHES a page. The manager's own loader is the judge: an orphan makes the
    // patch it just saved open on the default arrangement instead.
    let yaml =
        call(&mut ws, 4, "serialize", json!({})).await["result"]["yaml"].as_str().unwrap().to_string();
    let r = call(&mut ws, 5, "load_text", json!({ "content": yaml })).await;
    assert_eq!(r["result"]["ok"], json!(true), "{r}");
    assert_eq!(
        r["result"]["layout_warning"],
        Value::Null,
        "the manager saved an arrangement it cannot itself open: {r}"
    );
}

/// The id of the entry the whole page hangs off, and how many there are — a page holds exactly one,
/// so a second root IS the corruption a resurrected container makes.
fn page_roots(doc: &goofi_bridge::crdt::GraphDoc, name: &str) -> Vec<String> {
    let arr = arrangement(doc);
    let obj = arr.as_object().expect("the arrangement root");
    let page = obj.iter().find(|(_, e)| e["name"] == json!(name)).map(|(id, _)| id.clone());
    let Some(page) = page else { return Vec::new() };
    obj.iter().filter(|(_, e)| e["parent"] == json!(page)).map(|(id, _)| id.clone()).collect()
}

#[tokio::test]
async fn a_layout_undo_moves_a_panel_back_rather_than_resurrecting_its_split() {
    // The MOVE half of the class `a_layout_undo_leaves_a_peers_panel_standing` closed for a birth,
    // and again only TWO sessions can see it. A move that empties a split promotes the survivor and
    // drops the split; restoring the slots the move displaced puts that DEAD split back at the page
    // root — while the wrapper a PEER has since hung its own panel off stays where it is, so the
    // page ends up with two roots and an arrangement the manager's own loader refuses. The inverse
    // is therefore another MOVE, planned at undo time against whatever still stands.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let first = panels(&doc)[0].clone();
    // Accepted: every layout write answers `{ok:true}`, save `session_add_page`, which answers
    // with the ids it minted.
    let ok = |r: &Value| {
        assert!(r.get("error").is_none(), "{r}");
        assert!(accepted(r), "{r}");
    };

    // `Layout` holds a two-child split; `Signals` holds another, to move into.
    let mine = call(&mut ws, 1, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "row" })).await["result"]
        .as_str().unwrap().to_string();
    ok(&call(&mut ws, 2, "session_add_page", json!({ "name": "Signals" })).await);
    let d = sync_replica(&mut ws, |d| panels(d).len() == 3).await;
    let theirs = panels(&d).into_iter().find(|p| *p != first && *p != mine).expect("its panel");
    let far = call(&mut ws, 3, "page_split_panel",
        json!({ "page": "Signals", "panel": theirs, "direction": "row" })).await["result"]
        .as_str().unwrap().to_string();
    let d = sync_replica(&mut ws, |d| d.read_at(&["arrangement", far.as_str()]).is_some()).await;
    let dest = d.read_at(&["arrangement", far.as_str(), "parent"])
        .and_then(|v| v.as_str().map(str::to_string)).expect("the split on Signals");

    // s1 moves its panel across, which leaves `Layout`'s split one-armed: it is promoted away and
    // `first` becomes the page's root.
    ok(&call_session(&mut ws, 4, "page_move_panel",
        json!({ "page": "Layout", "panel": mine, "new_parent": dest, "order_index": 0 }), "s1").await);
    let moved = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", mine.as_str(), "parent"]) == Some(json!(dest))
    })
    .await;
    assert_eq!(page_roots(&moved, "Layout"), vec![first.clone()], "the survivor took the page root");

    // s2 then splits that survivor, so its panel hangs off a wrapper sitting in the very slot the
    // dead split wants back.
    let peers = call_session(&mut ws, 5, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "column" }), "s2").await["result"]
        .as_str().expect("the peer's panel").to_string();

    let u = call_session(&mut ws, 6, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "undo flipped something: {u}");
    let after = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", mine.as_str(), "parent"]).is_some_and(|v| v != json!(dest))
    })
    .await;
    let arr = arrangement(&after);
    assert_eq!(page_roots(&after, "Layout").len(), 1, "a dead split did not come back: {arr}");
    assert!(panels(&after).contains(&peers), "the peer's panel survived a foreign undo: {arr}");
    let page = call(&mut ws, 7, "inspect_layout", json!({ "page": "Layout" })).await["result"]
        ["text"].as_str().unwrap().to_string();
    assert!(page.contains(&mine), "and the undo did move the panel back: {page}");

    // The manager's own loader is the judge of whether that page still holds together.
    let yaml = call(&mut ws, 8, "serialize", json!({})).await["result"]["yaml"]
        .as_str().unwrap().to_string();
    let r = call(&mut ws, 9, "load_text", json!({ "content": yaml })).await;
    assert_eq!(r["result"]["layout_warning"], Value::Null,
        "the manager saved an arrangement it cannot itself open: {r}");
}

#[tokio::test]
async fn each_frozen_drags_undo_leaves_a_peers_panel_standing_too() {
    // The two gestures that CARRY a subtree — dropping it on a panel, and tearing it off onto the
    // tab bar — are the same class: each lifts a subtree, which can promote its split away, and a
    // slot-restore undo brings that dead split back on top of whatever the peer built where it stood.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let first = panels(&doc)[0].clone();
    // Accepted: every layout write answers `{ok:true}`, save `session_add_page`, which answers
    // with the ids it minted.
    let ok = |r: &Value| {
        assert!(r.get("error").is_none(), "{r}");
        assert!(accepted(r), "{r}");
    };

    let mine = call(&mut ws, 1, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "row" })).await["result"]
        .as_str().unwrap().to_string();
    ok(&call(&mut ws, 2, "session_add_page", json!({ "name": "Signals" })).await);
    let d = sync_replica(&mut ws, |d| panels(d).len() == 3).await;
    let target = panels(&d).into_iter().find(|p| *p != first && *p != mine).expect("its panel");

    ok(&call_session(&mut ws, 3, "page_insert_at_panel",
        json!({ "page": "Signals", "subtree": mine, "target": target, "direction": "column" }),
        "s1").await);
    let dropped = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", mine.as_str(), "parent"])
            .is_some_and(|p| d.read_at(&["arrangement", first.as_str(), "parent"]) != Some(p))
    })
    .await;
    assert_eq!(page_roots(&dropped, "Layout"), vec![first.clone()], "the survivor took the page root");

    let peers = call_session(&mut ws, 4, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "column" }), "s2").await["result"]
        .as_str().expect("the peer's panel").to_string();
    let u = call_session(&mut ws, 5, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");
    let after = sync_replica(&mut ws, |d| panels(d).contains(&peers) && panels(d).len() == 4).await;
    let arr = arrangement(&after);
    assert_eq!(page_roots(&after, "Layout").len(), 1, "a dead split did not come back: {arr}");
    assert!(panels(&after).contains(&peers), "the peer's panel survived a foreign undo: {arr}");

    // The tab-bar tear-off carries a subtree onto a page of its own, and lifting it can promote a
    // split away just the same — so its undo is a move back, not the slots the tear-off displaced.
    ok(&call_session(&mut ws, 6, "session_add_page",
        json!({ "name": "Torn off", "subtree": mine }), "s1").await);
    let torn = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", mine.as_str(), "size"]).and_then(|v| v.as_f64()) == Some(1.0)
    })
    .await;
    let survivor = page_roots(&torn, "Layout");
    assert_eq!(survivor.len(), 1, "the page it left kept exactly one root: {:?}", arrangement(&torn));
    let theirs = call_session(&mut ws, 7, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "row" }), "s2").await["result"]
        .as_str().expect("the peer's second panel").to_string();
    let u = call_session(&mut ws, 8, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");
    let back = sync_replica(&mut ws, |d| panels(d).contains(&theirs) && panels(d).len() == 5).await;
    let arr = arrangement(&back);
    assert_eq!(page_roots(&back, "Layout").len(), 1, "a dead split did not come back: {arr}");
    assert!(panels(&back).contains(&theirs), "the peer's panel survived the tear-off's undo: {arr}");

    let yaml = call(&mut ws, 9, "serialize", json!({})).await["result"]["yaml"]
        .as_str().unwrap().to_string();
    let r = call(&mut ws, 10, "load_text", json!({ "content": yaml })).await;
    assert_eq!(r["result"]["layout_warning"], Value::Null,
        "the manager saved an arrangement it cannot itself open: {r}");
}

#[tokio::test]
async fn undoing_a_move_puts_the_panel_back_at_the_index_and_share_it_had() {
    // The plain move case, undisturbed: re-planning the inverse must not cost the single-session
    // expectation that ctrl-Z puts a panel back exactly — same split, same position among its
    // siblings, same shares for all three.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let first = panels(&doc)[0].clone();
    // Accepted: every layout write answers `{ok:true}`, save `session_add_page`, which answers
    // with the ids it minted.
    let ok = |r: &Value| {
        assert!(r.get("error").is_none(), "{r}");
        assert!(accepted(r), "{r}");
    };

    // A THREE-child split, so the move leaves it standing and the inverse is a move back INTO it.
    let last = call(&mut ws, 1, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "row" })).await["result"]
        .as_str().unwrap().to_string();
    let mid = call(&mut ws, 2, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "row" })).await["result"]
        .as_str().unwrap().to_string();
    ok(&call(&mut ws, 3, "session_add_page", json!({ "name": "Signals" })).await);
    let d = sync_replica(&mut ws, |d| panels(d).len() == 4).await;
    let theirs = panels(&d).into_iter()
        .find(|p| *p != first && *p != mid && *p != last).expect("the new page's panel");
    let far = call(&mut ws, 4, "page_split_panel",
        json!({ "page": "Signals", "panel": theirs, "direction": "row" })).await["result"]
        .as_str().unwrap().to_string();
    let d = sync_replica(&mut ws, |d| d.read_at(&["arrangement", far.as_str()]).is_some()).await;
    let dest = d.read_at(&["arrangement", far.as_str(), "parent"])
        .and_then(|v| v.as_str().map(str::to_string)).expect("the split on Signals");
    let before = arrangement(&d);

    ok(&call_session(&mut ws, 5, "page_move_panel",
        json!({ "page": "Layout", "panel": mid, "new_parent": dest, "order_index": 1 }), "s1").await);
    let u = call_session(&mut ws, 6, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");
    let back = sync_replica(&mut ws, |d| entry_count(d) > 0 && arrangement(d) == before).await;
    assert_eq!(arrangement(&back), before, "one ctrl-Z put the panel back exactly where it was");
}

/// A panel's share of its split, as a replica reads it.
fn size_of(doc: &goofi_bridge::crdt::GraphDoc, id: &str) -> f64 {
    doc.read_at(&["arrangement", id, "size"]).and_then(|v| v.as_f64()).unwrap_or(f64::NAN)
}

/// The manager's own loader, asked to open what the manager just saved — `Null` when the arrangement
/// is one it can. The judge every no-raw-restore test appeals to.
async fn reload_warning(ws: &mut Ws, id: i64) -> Value {
    let yaml =
        call(ws, id, "serialize", json!({})).await["result"]["yaml"].as_str().unwrap().to_string();
    call(ws, id + 1, "load_text", json!({ "content": yaml })).await["result"]["layout_warning"]
        .clone()
}

#[tokio::test]
async fn a_type_change_undone_after_a_peers_split_leaves_the_peer_its_slot() {
    // The other half of the rule, and the half the guard FOUND. `page_set_panel` edits what a panel
    // HOLDS, not where it sits — but its inverse restored the WHOLE entry, `order` among it, so the
    // undo pinned the panel back into the slot a peer's adjacent split had since taken. Two children
    // of one split at one order is an arrangement the manager's own loader refuses. A single session
    // cannot see this, which is why it is two over the wire.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let a = panels(&doc)[0].clone();
    let b = call(&mut ws, 1, "page_split_panel",
        json!({ "page": "Layout", "panel": a, "direction": "row" })).await["result"]
        .as_str().expect("the split's new panel").to_string();

    let r = call_session(&mut ws, 2, "page_set_panel",
        json!({ "page": "Layout", "panel": b, "type": "console" }), "s1").await;
    assert!(accepted(&r), "{r}");
    // The peer splits `a` along the same axis, so its new panel is inserted adjacent and takes the
    // order `b` held — the slot the undo must not reclaim.
    let peer = call_session(&mut ws, 3, "page_split_panel",
        json!({ "page": "Layout", "panel": a, "direction": "row" }), "s2").await["result"]
        .as_str().expect("the peer's panel").to_string();
    let u = call_session(&mut ws, 4, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");

    let after = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", b.as_str(), "panel_type"]) == Some(json!("empty"))
    })
    .await;
    let arr = arrangement(&after);
    assert_eq!(
        after.read_at(&["arrangement", b.as_str(), "panel_type"]),
        Some(json!("empty")),
        "the undo gave the panel back the type it had: {arr}"
    );
    assert_ne!(
        after.read_at(&["arrangement", b.as_str(), "order"]),
        after.read_at(&["arrangement", peer.as_str(), "order"]),
        "and left the peer's panel the order it took: {arr}"
    );
    assert_eq!(
        reload_warning(&mut ws, 5).await,
        Value::Null,
        "the manager saved an arrangement it cannot itself open: {arr}"
    );
}

#[tokio::test]
async fn a_resize_undone_after_a_peers_split_re_asserts_shares_without_re_pinning_slots() {
    // `page_resize_split` is the same shape: a set of shares is CONTENTS too, and restoring each
    // whole entry to undo them puts the orders back with them. The undo re-asserts the shares it
    // found and renormalizes around whatever the peer added, so the split still divides one slot.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let a = panels(&doc)[0].clone();
    let b = call(&mut ws, 1, "page_split_panel",
        json!({ "page": "Layout", "panel": a, "direction": "row" })).await["result"]
        .as_str().expect("the split's new panel").to_string();
    let d = sync_replica(&mut ws, |d| d.read_at(&["arrangement", b.as_str()]).is_some()).await;
    let near = d.read_at(&["arrangement", b.as_str(), "parent"])
        .and_then(|v| v.as_str().map(str::to_string)).expect("the wrapper split");

    let r = call_session(&mut ws, 2, "page_resize_split",
        json!({ "page": "Layout", "split": near, "fractions": [0.3, 0.7] }), "s1").await;
    assert!(accepted(&r), "{r}");
    let peer = call_session(&mut ws, 3, "page_split_panel",
        json!({ "page": "Layout", "panel": a, "direction": "row" }), "s2").await["result"]
        .as_str().expect("the peer's panel").to_string();
    let u = call_session(&mut ws, 4, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");

    let (a2, b2) = (a.clone(), b.clone());
    let after =
        sync_replica(&mut ws, |d| (size_of(d, &a2) - size_of(d, &b2)).abs() < 1e-9).await;
    let arr = arrangement(&after);
    assert!(
        (size_of(&after, &a) - size_of(&after, &b)).abs() < 1e-9,
        "the undo put back the equal shares it found: {arr}"
    );
    let total = size_of(&after, &a) + size_of(&after, &b) + size_of(&after, &peer);
    assert!(
        (total - 1.0).abs() < 1e-9,
        "and the split still divides exactly one slot around the peer's own share: {arr}"
    );
    assert_eq!(
        reload_warning(&mut ws, 5).await,
        Value::Null,
        "the manager saved an arrangement it cannot itself open: {arr}"
    );
}

#[tokio::test]
async fn a_contents_undo_follows_the_panel_a_peer_has_since_carried_off() {
    // A slot is not only an order. A peer may have carried the panel to another split entirely — and
    // the two-child split it left promoted away behind it. Restoring the entry's own `parent` hangs
    // the panel off a container that is no longer there, which reaches no page at all.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let a = panels(&doc)[0].clone();
    let b = call(&mut ws, 1, "page_split_panel",
        json!({ "page": "Layout", "panel": a, "direction": "row" })).await["result"]
        .as_str().expect("the split's new panel").to_string();
    call(&mut ws, 2, "session_add_page", json!({ "name": "Two" })).await;
    let d = sync_replica(&mut ws, |d| panels(d).len() == 3).await;
    let c = panels(&d).into_iter().find(|p| *p != a && *p != b).expect("the second page's panel");
    let e = call(&mut ws, 3, "page_split_panel",
        json!({ "page": "Two", "panel": c, "direction": "row" })).await["result"]
        .as_str().expect("the second page's second panel").to_string();
    let d = sync_replica(&mut ws, |d| d.read_at(&["arrangement", e.as_str()]).is_some()).await;
    let far = d.read_at(&["arrangement", e.as_str(), "parent"])
        .and_then(|v| v.as_str().map(str::to_string)).expect("the split on Two");

    let r = call_session(&mut ws, 4, "page_set_panel",
        json!({ "page": "Layout", "panel": b, "type": "console" }), "s1").await;
    assert!(accepted(&r), "{r}");
    let r = call_session(&mut ws, 5, "page_move_panel",
        json!({ "page": "Layout", "panel": b, "new_parent": far, "order_index": 0 }), "s2").await;
    assert!(accepted(&r), "{r}");
    let u = call_session(&mut ws, 6, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");

    let after = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", b.as_str(), "panel_type"]) == Some(json!("empty"))
    })
    .await;
    let arr = arrangement(&after);
    assert_eq!(
        after.read_at(&["arrangement", b.as_str(), "parent"]),
        Some(json!(far)),
        "the type came back where the peer had carried the panel to: {arr}"
    );
    assert_eq!(
        reload_warning(&mut ws, 7).await,
        Value::Null,
        "the manager saved an arrangement it cannot itself open: {arr}"
    );
}

#[tokio::test]
async fn a_rename_undone_after_a_peers_reorder_keeps_the_tab_index_it_finds() {
    // A page's NAME is contents and its tab index is the slot, so the same rule binds: a peer's
    // reorder renumbers the strip, and restoring the whole page entry puts back an index another
    // tab now holds. The guard's own interleaving never reaches this (an added page appends rather
    // than renumbering), which is why it is drawn here explicitly.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    // Accepted: every layout write answers `{ok:true}`, save `session_add_page`, which answers
    // with the ids it minted.
    let ok = |r: &Value| {
        assert!(r.get("error").is_none(), "{r}");
        assert!(accepted(r), "{r}");
    };
    ok(&call(&mut ws, 1, "session_add_page", json!({ "name": "Two" })).await);
    ok(&call(&mut ws, 2, "session_add_page", json!({ "name": "Three" })).await);

    ok(&call_session(&mut ws, 3, "session_rename_page",
        json!({ "from": "Two", "to": "Deux" }), "s1").await);
    // The peer pulls the last tab to the front, so every index behind it shifts by one.
    ok(&call_session(&mut ws, 4, "session_reorder_page",
        json!({ "name": "Three", "to_index": 0 }), "s2").await);
    let u = call_session(&mut ws, 5, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");

    let after = sync_replica(&mut ws, |d| {
        arrangement(d).as_object().is_some_and(|m| m.values().any(|e| e["name"] == json!("Two")))
    })
    .await;
    let arr = arrangement(&after);
    let mut tabs: Vec<i64> = arr.as_object().expect("the arrangement")
        .values().filter(|e| e["kind"] == json!("page"))
        .filter_map(|e| e["order"].as_i64()).collect();
    tabs.sort_unstable();
    tabs.dedup();
    assert_eq!(tabs.len(), 3, "the undo renamed the page without taking a tab index twice: {arr}");
    assert_eq!(
        reload_warning(&mut ws, 6).await,
        Value::Null,
        "the manager saved an arrangement it cannot itself open: {arr}"
    );
}

/// **The rule, enforced rather than remembered: no layout inverse restores raw state — every inverse
/// re-plans through the forward planners.** Putting a slot back resurrects a container a peer's
/// children no longer belong under, which strands them and corrupts the arrangement on the next
/// save. Three rounds found three instances of it BY HAND, which is two too many, so every layout
/// write op is driven here through the one interleaving that makes the class visible — a peer edits
/// between the op and its undo — and the manager's own loader is the judge. The op list comes from
/// the registry and the match below has no catch-all, so a NEW layout op is red until it is driven
/// too.
#[tokio::test]
async fn no_layout_undo_puts_back_a_slot_a_peer_has_since_built_over() {
    let ops: Vec<&str> = goofi_bridge::ops::REGISTRY
        .iter()
        .filter(|o| o.writes && (o.name.starts_with("page_") || o.name.starts_with("session_")))
        .map(|o| o.name)
        .collect();
    assert!(
        ops.contains(&"page_remove_panel") && ops.contains(&"session_remove_page"),
        "the registry filter still finds the layout write ops: {ops:?}"
    );
    let mut stranded = Vec::new();
    for op in &ops {
        // A fresh manager per op, so each one meets the same arrangement: `Layout` holds a two-child
        // split and `Two` holds another. Between them every op has an argument that exists, and each
        // one leaves `a` standing for the peer to build on.
        let base = start_server().await;
        let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
        let _ = recv_text(&mut ws).await;
        let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
        let a = panels(&doc)[0].clone();
        let b = call(&mut ws, 1, "page_split_panel",
            json!({ "page": "Layout", "panel": a, "direction": "row" })).await["result"]
            .as_str().expect("the split's new panel").to_string();
        call(&mut ws, 2, "session_add_page", json!({ "name": "Two" })).await;
        let d = sync_replica(&mut ws, |d| panels(d).len() == 3).await;
        let c = panels(&d).into_iter().find(|p| *p != a && *p != b).expect("the second page's panel");
        let e = call(&mut ws, 3, "page_split_panel",
            json!({ "page": "Two", "panel": c, "direction": "row" })).await["result"]
            .as_str().expect("the second page's second panel").to_string();
        let d = sync_replica(&mut ws, |d| d.read_at(&["arrangement", e.as_str()]).is_some()).await;
        let far = d.read_at(&["arrangement", e.as_str(), "parent"])
            .and_then(|v| v.as_str().map(str::to_string)).expect("the split on Two");
        let near = d.read_at(&["arrangement", b.as_str(), "parent"])
            .and_then(|v| v.as_str().map(str::to_string)).expect("the split on Layout");

        let payload = match *op {
            "session_add_page" => json!({ "name": "Fresh" }),
            "session_remove_page" => json!({ "name": "Two" }),
            "session_rename_page" => json!({ "from": "Two", "to": "Deux" }),
            "session_reorder_page" => json!({ "name": "Two", "to_index": 0 }),
            "page_split_panel" => json!({ "page": "Layout", "panel": a }),
            "page_set_panel" => json!({ "page": "Layout", "panel": b, "type": "console" }),
            "page_move_panel" =>
                json!({ "page": "Layout", "panel": b, "new_parent": far, "order_index": 0 }),
            "page_insert_at_panel" => json!({ "page": "Two", "subtree": b, "target": c }),
            "page_resize_split" =>
                json!({ "page": "Layout", "split": near, "fractions": [0.3, 0.7] }),
            "page_remove_panel" => json!({ "page": "Layout", "panel": b }),
            new => panic!("`{new}` is a layout write op with no case here — drive it through this \
                           guard, and say why if its inverse may restore a slot"),
        };
        let r = call_session(&mut ws, 4, op, payload, "s1").await;
        assert!(r["error"].is_null(), "{op}: {r}");
        // The peer then builds exactly where that op's slot-restore inverse would want to write: over
        // the survivor `a` for a structural op, over the tab index the page ops renumber.
        let (peer_op, peer) = if op.starts_with("session_") {
            ("session_add_page", json!({ "name": "Peer" }))
        } else {
            ("page_split_panel", json!({ "page": "Layout", "panel": a }))
        };
        let r = call_session(&mut ws, 5, peer_op, peer, "s2").await;
        assert!(r["error"].is_null(), "{op}, the peer's edit: {r}");
        let u = call_session(&mut ws, 6, "undo", json!({}), "s1").await;
        assert_eq!(u["result"]["changed"], json!(true), "{op}: the undo flipped nothing: {u}");

        let yaml = call(&mut ws, 7, "serialize", json!({})).await["result"]["yaml"]
            .as_str().unwrap().to_string();
        let r = call(&mut ws, 8, "load_text", json!({ "content": yaml })).await;
        if r["result"]["layout_warning"] != Value::Null {
            stranded.push(*op);
        }
    }
    // EMPTY, and it stays empty: the two this guard found (`page_set_panel`, `page_resize_split`)
    // now invert through `Command::LayoutContents`, which re-reads each slot at flip time. The one
    // op whose inverse still restores an `order` is `session_reorder_page` — where the order IS the
    // content, so carrying the live one over would make its undo a no-op; it needs a re-planning
    // command of its own and no interleaving here reaches it. It is driven above regardless, so the
    // day it does strand, this list is what says so.
    let empty: [&str; 0] = [];
    assert_eq!(
        stranded, empty,
        "an undo left an arrangement the manager cannot itself open"
    );
}

#[tokio::test]
async fn a_redo_after_a_peers_edit_re_plans_rather_than_replaying_the_slots_it_found() {
    // The narrower half of the same class, and the one an undo test cannot see: what a REDO replays
    // is the close's own inverse. Handing it the slots the close found puts the dead split back on
    // top of whatever the peer built where it stood — undo, peer edit, redo, two roots on one page.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let a = panels(&doc)[0].clone();

    let born = call_session(&mut ws, 1, "page_split_panel",
        json!({ "page": "Layout", "panel": a, "direction": "row" }), "s1").await["result"]
        .as_str().expect("the new panel's id").to_string();
    // The undo closes it with promote, so `a` takes the page root and the wrapper dies.
    let u = call_session(&mut ws, 2, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");
    let gone = sync_replica(&mut ws, |d| {
        !panels(d).is_empty() && d.read_at(&["arrangement", born.as_str()]).is_none()
    })
    .await;
    assert_eq!(page_roots(&gone, "Layout"), vec![a.clone()], "the survivor took the page root");

    let peer = call_session(&mut ws, 3, "page_split_panel",
        json!({ "page": "Layout", "panel": a, "direction": "column" }), "s2").await["result"]
        .as_str().expect("the peer's panel").to_string();
    let r = call_session(&mut ws, 4, "redo", json!({}), "s1").await;
    assert_eq!(r["result"]["changed"], json!(true), "{r}");

    let after =
        sync_replica(&mut ws, |d| panels(d).contains(&born) && panels(d).contains(&peer)).await;
    let arr = arrangement(&after);
    assert_eq!(page_roots(&after, "Layout").len(), 1, "a dead split did not come back: {arr}");
    assert!(panels(&after).contains(&peer), "the peer's panel survived a foreign redo: {arr}");
    let yaml = call(&mut ws, 5, "serialize", json!({})).await["result"]["yaml"]
        .as_str().unwrap().to_string();
    let r = call(&mut ws, 6, "load_text", json!({ "content": yaml })).await;
    assert_eq!(r["result"]["layout_warning"], Value::Null,
        "the manager saved an arrangement it cannot itself open: {r}");
}

#[tokio::test]
async fn one_pass_over_every_session_and_page_write_op() {
    // The dispatch arms are pure argument plumbing over planners that are unit-tested, so what is
    // NOT otherwise checked is that each arm reads the argument name its registry row advertises.
    // One pass exercises all of them, including a subtree move across pages.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let first = panels(&doc)[0].clone();
    // Accepted: every layout write answers `{ok:true}`, save `session_add_page`, which answers
    // with the ids it minted.
    let ok = |r: &Value| {
        assert!(r.get("error").is_none(), "{r}");
        assert!(accepted(r), "{r}");
    };

    ok(&call(&mut ws, 1, "session_add_page", json!({ "name": "Second" })).await);
    ok(&call(&mut ws, 2, "session_rename_page", json!({ "from": "Second", "to": "Signals" })).await);
    ok(&call(&mut ws, 3, "session_reorder_page", json!({ "name": "Signals", "to_index": 0 })).await);
    let pages = page_names(&mut ws, 4).await;
    assert_eq!(pages, ["Signals", "Layout"], "rename and reorder both landed");

    // The new page's own panel, and a split on it to serve as a move destination.
    let d = sync_replica(&mut ws, |d| panels(d).len() == 2).await;
    let theirs = panels(&d).into_iter().find(|p| *p != first).expect("the new page's panel");
    let sibling = call(&mut ws, 5, "page_split_panel",
        json!({ "page": "Signals", "panel": theirs, "direction": "row" })).await["result"]
        .as_str().unwrap().to_string();
    let d = sync_replica(&mut ws, |d| d.read_at(&["arrangement", sibling.as_str()]).is_some()).await;
    let dest = d.read_at(&["arrangement", sibling.as_str(), "parent"]).and_then(|v| v.as_str().map(str::to_string))
        .expect("the wrapper split the wrap created");

    // Move a panel off the OTHER page into that split — the cross-page subtree move.
    let mine = call(&mut ws, 6, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "column", "ratio": 0.25 })).await["result"]
        .as_str().unwrap().to_string();
    ok(&call(&mut ws, 7, "page_move_panel",
        json!({ "page": "Layout", "panel": mine, "new_parent": dest, "order_index": 0 })).await);
    let page = call(&mut ws, 8, "inspect_layout", json!({ "page": "Signals" })).await["result"]["text"]
        .as_str().unwrap().to_string();
    assert!(page.contains(&mine), "the moved panel is on the destination page now: {page}");

    ok(&call(&mut ws, 9, "page_remove_panel", json!({ "page": "Signals", "panel": mine })).await);
    ok(&call(&mut ws, 10, "session_remove_page", json!({ "name": "Signals" })).await);
    let left = page_names(&mut ws, 11).await;
    assert_eq!(left, ["Layout"], "the page and its panels went");
    // The last page and a page's last panel both refuse, rather than leaving nothing to look at.
    let last = call(&mut ws, 12, "session_remove_page", json!({ "name": "Layout" })).await;
    assert!(last["error"].as_str().is_some_and(|e| e.contains("last page")), "{last}");
}

#[tokio::test]
async fn each_frozen_drag_gesture_is_one_op_and_therefore_one_undo() {
    // The drag feel is FROZEN UX. Expressed as the primitive ops, a drop costs three to five
    // commands — three to five ctrl-Z for one drag, and every peer watching two arrangements that
    // were never on anybody's screen. Each gesture is one op, and undo puts back exactly what it
    // found.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let first = panels(&doc)[0].clone();
    // Accepted: every layout write answers `{ok:true}`, save `session_add_page`, which answers
    // with the ids it minted.
    let ok = |r: &Value| {
        assert!(r.get("error").is_none(), "{r}");
        assert!(accepted(r), "{r}");
    };

    // Two panels on `Layout`, and a second page holding the drop target.
    let mine = call(&mut ws, 1, "page_split_panel",
        json!({ "page": "Layout", "panel": first, "direction": "row" })).await["result"]
        .as_str().unwrap().to_string();
    ok(&call(&mut ws, 2, "session_add_page", json!({ "name": "Signals", "index": 0 })).await);
    let d = sync_replica(&mut ws, |d| panels(d).len() == 3).await;
    assert_eq!(
        page_names(&mut ws, 3).await.first().map(String::as_str),
        Some("Signals"),
        "the page landed at the tab index asked for"
    );
    let target = panels(&d).into_iter().find(|p| *p != first && *p != mine).expect("its panel");

    // dropOnPanel — one op, and one undo.
    let before = arrangement(&d);
    ok(&call_session(&mut ws, 4, "page_insert_at_panel",
        json!({ "page": "Signals", "subtree": mine, "target": target,
                "direction": "column", "place_before": true, "ratio": 0.3 }), "s1").await);
    // Positive about something: a fresh replica trivially "differs" from `before` on a doc it has
    // not synced a single frame of.
    let dropped = sync_replica(&mut ws, |d| {
        !panels(d).is_empty()
            && d.read_at(&["arrangement", mine.as_str(), "parent"])
                != Some(before[&mine]["parent"].clone())
    })
    .await;
    let page = call(&mut ws, 5, "inspect_layout", json!({ "page": "Signals" })).await["result"]
        ["text"].as_str().unwrap().to_string();
    assert!(page.contains(&mine), "the panel crossed pages in ONE op: {page}");
    assert_ne!(arrangement(&dropped), before, "the drop actually moved something");

    let u = call_session(&mut ws, 6, "undo", json!({}), "s1").await;
    assert_eq!(u["result"]["changed"], json!(true), "{u}");
    let back = sync_replica(&mut ws, |d| entry_count(d) > 0 && arrangement(d) == before).await;
    assert_eq!(arrangement(&back), before, "ONE ctrl-Z put the whole drag back");

    // dropPanelOnTabBar — a page built around an existing panel, also one op and one undo.
    ok(&call_session(&mut ws, 7, "session_add_page",
        json!({ "name": "Torn off", "index": 0, "subtree": mine }), "s1").await);
    let torn = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", mine.as_str(), "size"]).and_then(|v| v.as_f64()) == Some(1.0)
    })
    .await;
    assert_eq!(
        torn.read_at(&["arrangement", mine.as_str(), "size"]).and_then(|v| v.as_f64()),
        Some(1.0),
        "the dragged panel is the new page's whole root"
    );
    call_session(&mut ws, 8, "undo", json!({}), "s1").await;
    let back = sync_replica(&mut ws, |d| entry_count(d) > 0 && arrangement(d) == before).await;
    assert_eq!(arrangement(&back), before, "and one ctrl-Z put that back too");

    // page_resize_split — the drag-commit, and the only op that sizes anything.
    let split = back.read_at(&["arrangement", mine.as_str(), "parent"])
        .and_then(|v| v.as_str().map(str::to_string)).expect("the wrapper split");
    ok(&call(&mut ws, 9, "page_resize_split",
        json!({ "page": "Layout", "split": split, "fractions": [0.2, 0.8] })).await);
    let sized = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", first.as_str(), "size"]).and_then(|v| v.as_f64()) == Some(0.2)
    })
    .await;
    assert_eq!(
        sized.read_at(&["arrangement", mine.as_str(), "size"]).and_then(|v| v.as_f64()),
        Some(0.8),
        "both children landed on the fractions the drag drew"
    );
    let bad = call(&mut ws, 10, "page_resize_split",
        json!({ "page": "Layout", "split": split, "fractions": [0.5] })).await;
    assert!(bad["error"].as_str().is_some_and(|e| e.contains("children")), "{bad}");
}

#[tokio::test]
async fn page_set_panel_lands_a_combined_type_and_binding_and_refuses_an_unknown_node() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let panel = panels(&doc)[0].clone();

    // Type is applied BEFORE state: switching type clears the old type's state, so a combined
    // `{type, state}` that landed them the other way round would store a wiped binding.
    let r = call(
        &mut ws,
        2,
        "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "type": "viewer",
                "state": { "node": osc, "slot": "out" } }),
    )
    .await;
    assert!(accepted(&r), "{r}");
    let d2 = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", panel.as_str(), "panel_type"]) == Some(json!("viewer"))
    })
    .await;
    let state = d2
        .read_at(&["arrangement", panel.as_str(), "state"])
        .and_then(|v| v.as_str().map(str::to_string))
        .expect("the state leaf");
    assert!(state.contains(&osc), "the binding survived the type change: {state}");

    // Two state writes back to back, with no delta between them — the shape every panel-state caller
    // has (read the bag, edit one key, write it back), and the one an agent driving the ops reaches
    // at once. The second must not replace a bag it has not seen the first land in.
    for (id, patch) in [(20, json!({ "kind": "line" })), (21, json!({ "slot": "out" }))] {
        let r = call(&mut ws, id, "page_set_panel",
            json!({ "page": "Layout", "panel": panel, "state": patch })).await;
        assert!(accepted(&r), "{r}");
    }
    let merged = sync_replica(&mut ws, |d| {
        d.read_at(&["arrangement", panel.as_str(), "state"])
            .and_then(|v| v.as_str().map(|s| s.contains("\"slot\":\"out\"")))
            .unwrap_or(false)
    })
    .await;
    let state = merged
        .read_at(&["arrangement", panel.as_str(), "state"])
        .and_then(|v| v.as_str().map(str::to_string))
        .expect("the state leaf");
    assert!(state.contains(&osc) && state.contains("line"),
        "a state write merges, so neither earlier key was dropped: {state}");

    // A bind to a node that is not there renders an EMPTY panel and says nothing — so refuse it
    // where the answer can teach.
    let bad = call(
        &mut ws,
        3,
        "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "state": { "node": "deadbeefdead" } }),
    )
    .await;
    assert!(bad["error"].as_str().is_some_and(|e| e.contains("deadbeefdead")), "{bad}");

    // A DISPLAY NAME is not a binding. It resolves today and stops resolving the moment the node is
    // renamed, which is a binding that silently empties a panel — the uid is the identity, and it
    // is what the frontend stores.
    let name = d2
        .read_at(&["nodes", osc.as_str(), "name"])
        .and_then(|v| v.as_str().map(str::to_string))
        .expect("the node's display name");
    let by_name = call(
        &mut ws,
        4,
        "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "state": { "node": name } }),
    )
    .await;
    assert!(
        by_name["error"].as_str().is_some_and(|e| e.contains(&name)),
        "a panel binds by uid, never by name: {by_name}"
    );
}

#[tokio::test]
async fn a_layout_write_answers_with_the_arrangement_it_produced() {
    // `{ok: true}` told a caller its write was accepted and nothing about what it made, so an agent
    // editing the layout had to follow every single op with an `inspect_layout` to see the tree it
    // was building. The write already knows: it is holding the arrangement it just planned against.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let panel = panels(&doc)[0].clone();

    let typed = call(&mut ws, 1, "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "type": "console" })).await;
    let text = typed["result"]["text"].as_str().unwrap_or_default().to_string();
    assert!(text.contains("console") && text.contains(&panel), "{typed}");
    // …and it is the arrangement AFTER the write, not the one the op was handed.
    let renamed = call(&mut ws, 2, "session_rename_page", json!({ "from": "Layout", "to": "Signals" })).await;
    let text = renamed["result"]["text"].as_str().unwrap_or_default().to_string();
    assert!(text.contains("Signals") && !text.contains("Layout"), "{renamed}");
    // Every family of layout write answers the same way — the close and move planners too, which
    // are separate code paths from the contents one the two above take.
    let page = call(&mut ws, 3, "session_add_page", json!({ "name": "Second" })).await;
    let moved = call(&mut ws, 4, "page_insert_at_panel",
        json!({ "page": "Second", "subtree": panel, "target": page["result"]["panel"] })).await;
    assert!(moved["result"]["text"].as_str().is_some_and(|t| t.contains("Second")), "{moved}");
    let _ = call(&mut ws, 5, "session_add_page", json!({ "name": "Third" })).await;
    let closed = call(&mut ws, 6, "session_remove_page", json!({ "name": "Third" })).await;
    let text = closed["result"]["text"].as_str().unwrap_or_default().to_string();
    assert!(!text.contains("Third") && text.contains("Second"), "{closed}");
}

#[tokio::test]
async fn page_set_panel_refuses_a_word_outside_the_vocabulary_and_names_the_set() {
    // The user's own repro (2026-08-10), driving a real agent against the live system: it guessed
    // `params` for the panel type — the real one is `parameters` — and was answered `{ok: true}`
    // while the panel dropped into an "Unknown panel type: params" state. A plausible guess told it
    // succeeded is worse than a refusal: nothing downstream can teach the caller it guessed.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let panel = panels(&doc)[0].clone();

    let bad = call(
        &mut ws,
        2,
        "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "type": "params", "state": { "node": osc } }),
    )
    .await;
    let err = bad["error"].as_str().unwrap_or_default().to_string();
    assert!(err.contains("params"), "the refusal names what was asked for: {bad}");
    assert!(err.contains("parameters"), "…and the set it could have meant: {bad}");
    // And it refused BEFORE writing: a panel left holding a type nothing renders is the whole bug.
    let after = sync_replica(&mut ws, |_| true).await;
    assert_ne!(
        after.read_at(&["arrangement", panel.as_str(), "panel_type"]),
        Some(json!("params")),
        "the refused type was not stored"
    );

    // A viewer's `kind` is the same vocabulary problem one level down — a free string inside the
    // state bag, and the one an agent reaches for right after binding the node.
    let bad_kind = call(
        &mut ws,
        3,
        "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "type": "viewer",
                "state": { "node": osc, "kind": "waveform" } }),
    )
    .await;
    let err = bad_kind["error"].as_str().unwrap_or_default().to_string();
    assert!(err.contains("waveform") && err.contains("line"), "{bad_kind}");

    // …and a slot the bound node does not have, which renders the panel's own empty state.
    let bound = call(
        &mut ws,
        4,
        "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "type": "viewer", "state": { "node": osc } }),
    )
    .await;
    assert!(bound.get("error").is_none(), "{bound}");
    let bad_slot = call(
        &mut ws,
        5,
        "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "state": { "slot": "spectrum" } }),
    )
    .await;
    let err = bad_slot["error"].as_str().unwrap_or_default().to_string();
    assert!(err.contains("spectrum") && err.contains("out"), "{bad_slot}");
}

#[tokio::test]
async fn the_viewpoint_persists_across_a_reload_without_dirtying_the_patch() {
    // Where a client is LOOKING is per-client, so it is deliberately not a doc root — it cannot
    // drag a peer or raise the unsaved dot. Persistence is the other axis: it still rides the
    // `.gfi` and comes back on hello, so reopening a patch restores the saver's viewpoint.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    let vp = json!({ "activePage": "Layout", "maximized": null, "subpatchPath": { "panel-2": ["a1b2c3"] } });
    call(&mut ws, 1, "set_viewpoint", json!({ "viewpoint": vp })).await;
    assert!(!is_dirty(&base).await, "looking around is not authoring, on any platform");

    let (mut fresh, _) = connect_async(format!("{base}/control")).await.unwrap();
    assert_eq!(recv_text(&mut fresh).await["payload"]["viewpoint"], vp, "hello carries it back");
    let ser = call(&mut ws, 2, "serialize", json!({})).await;
    assert!(ser["result"]["yaml"].as_str().unwrap().contains("a1b2c3"), "and it rides the .gfi");
}

#[tokio::test]
async fn a_corrupt_arrangement_still_opens_the_patch_and_says_what_it_dropped() {
    // The graph is the value, the arrangement is chrome: a layout the flat model admits but cannot
    // render must never make a patch unopenable — and the fallback must be stated, not silent.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut ws).await;
    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let yaml = call(&mut ws, 2, "serialize", json!({})).await["result"]["yaml"]
        .as_str()
        .unwrap()
        .to_string();
    // A panel parented to nothing — a class the nested tree could not even express.
    let broken = yaml.replace("parent: page-1", "parent: gone");
    assert_ne!(broken, yaml, "the fixture actually corrupted something");

    let r = call(&mut ws, 3, "load_text", json!({ "content": broken })).await;
    assert_eq!(r["result"]["ok"], json!(true), "the patch still opens: {r}");
    assert!(
        r["result"]["layout_warning"].as_str().is_some_and(|w| w.contains("reaches no page")),
        "the reply says why the arrangement was dropped: {r}"
    );
    let d = sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    assert_eq!(panels(&d).len(), 1, "opened on the default arrangement");
    assert_eq!(d.node_ids().len(), 1, "with the graph intact");
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
        .await["result"]["uid"]
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
async fn a_param_command_reaches_the_graph_and_other_clients() {
    // A client commits a param edit via the `update_param` command op. The manager routes it
    // through an `EditParam` command, applies it to the authoritative graph, and broadcasts the
    // resulting doc delta so a second client converges — no client doc write involved.
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};

    let base = start_server().await;

    // Writer client: connect, add a node, then edit a param via the command op.
    let (mut w, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut w).await;
    let _ = recv_binary(&mut w).await; // server hello SV

    let osc = call(&mut w, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
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
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};

    let base = start_server().await;

    let (mut w, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _ = recv_text(&mut w).await;
    let _ = recv_binary(&mut w).await; // server hello SV

    let osc = call(&mut w, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
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
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};

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
                        osc = v["result"]["uid"].as_str().map(str::to_string);
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

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
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

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
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

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
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

/// The member uids of a scope, as the client reads them out of the doc forest.
fn scope_members(doc: &goofi_bridge::crdt::GraphDoc, scope: &str) -> Vec<String> {
    doc.to_json()["instances"][scope]["members"]
        .as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

#[tokio::test]
async fn add_node_with_inst_id_lands_inside_the_scope_and_survives_undo_redo() {
    // A node added while the user has ENTERED a sub-patch must become a MEMBER of that scope, not a
    // ROOT node: the canvas renders only the entered scope's children, so a dropped `inst_id` makes
    // the node invisible exactly where it was placed. The placement rides on the COMMAND, so it has
    // to survive undo→redo too — a missing command field shows up there first.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
    let osc = uid(&call_session(&mut ws, 1, "add_node", json!({ "type": "Oscillator" }), "s1").await);
    let buf = uid(&call_session(&mut ws, 2, "add_node", json!({ "type": "Buffer" }), "s1").await);
    let scope = call_session(&mut ws, 3, "group_nodes", json!({ "members": [osc, buf], "pos": [0.0, 0.0] }), "s1")
        .await["result"]["inst_id"].as_str().unwrap().to_string();

    // Add a third node INSIDE the scope — the payload an entered editor sends.
    let inner = uid(
        &call_session(
            &mut ws,
            4,
            "add_node",
            json!({ "type": "Buffer", "inst_id": scope, "pos": [10.0, 20.0] }),
            "s1",
        )
        .await,
    );
    // Anchor every sync on the node's PRESENCE, never on the membership under test — an unsatisfiable
    // predicate would drain the helper's frame budget and report a recv timeout instead of the
    // assertion that actually failed. (A scope member stays in `node_ids`; only `members` moves.)
    let present = |d: &goofi_bridge::crdt::GraphDoc| d.node_ids().iter().any(|u| *u == inner);
    let doc = sync_replica(&mut ws, present).await;
    assert!(
        scope_members(&doc, &scope).contains(&inner),
        "the new node is a DIRECT member of the entered scope; got {:?}",
        scope_members(&doc, &scope)
    );

    // Undo drops it back out of the scope (and out of the graph).
    call_session(&mut ws, 5, "undo", json!({}), "s1").await;
    let gone = sync_replica(&mut ws, |d| {
        d.instance_ids().iter().any(|u| *u == scope) && !present(d)
    })
    .await;
    assert!(!gone.node_ids().iter().any(|u| *u == inner), "undo removed the node");

    // Redo must put it back INSIDE the scope. A command that never carried the scope re-roots it here.
    call_session(&mut ws, 6, "redo", json!({}), "s1").await;
    let back = sync_replica(&mut ws, present).await;
    assert!(back.node_ids().iter().any(|u| *u == inner), "redo restored the node");
    assert!(
        scope_members(&back, &scope).contains(&inner),
        "redo restored it INSIDE the scope, not at ROOT; got {:?}",
        scope_members(&back, &scope)
    );
}

#[tokio::test]
async fn add_node_rejects_an_inst_id_it_cannot_honour_and_creates_nothing() {
    // No partial mutation, and no silent rooting: an `inst_id` that is malformed, names nothing, or
    // names a plain node is REFUSED — before anything is created. Dropping it would put the node
    // where the canvas cannot show it; creating it and THEN failing would leave the graph and its
    // CRDT mirror disagreeing.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str().unwrap().to_string();

    // Well-formed hex, but no such scope.
    let unknown = call(&mut ws, 2, "add_node", json!({ "type": "Buffer", "inst_id": "deadbeef" })).await;
    assert!(unknown.get("error").is_some(), "an unknown scope is refused; got {unknown}");
    // Not hex at all.
    let malformed = call(&mut ws, 3, "add_node", json!({ "type": "Buffer", "inst_id": "not-a-uid" })).await;
    assert!(malformed.get("error").is_some(), "a malformed inst_id is refused; got {malformed}");
    // A live LEAF uid is not a scope either.
    let leaf = call(&mut ws, 4, "add_node", json!({ "type": "Buffer", "inst_id": osc })).await;
    assert!(leaf.get("error").is_some(), "a plain node uid is not a scope; got {leaf}");

    let doc = sync_replica(&mut ws, |d| d.node_ids().iter().any(|u| *u == osc)).await;
    assert_eq!(
        doc.node_ids().len(),
        1,
        "no refused add left a node behind; got {:?}",
        doc.node_ids()
    );
}

#[tokio::test]
async fn node_stats_broadcasts_the_measured_ufreq() {
    // Regression: `spawn_workers` (what the binary runs at startup) must wire `spawn_stats`,
    // else the node header never shows a live update rate — the `node_stats` producer was
    // orphaned in the now-removed `serve()` and the CLI called only `spawn_tick`.
    let state = AppState::new();
    spawn_workers(&state); // the status-drain worker, exactly as the CLI startup does
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    let base = format!("ws://{addr}");

    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    // A free-running source measures a ufreq after a few runs.
    let src = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
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
    // inspector preview tracks each re-evaluation. The value now comes from the NODE
    // (`Status::ParamValues`, spec §6.2) rather than from a graph-side evaluation, so the report
    // is what this drives — reporting nothing and asserting the event never arrives would pass
    // against a broadcaster that was deleted.
    let state = AppState::new();
    spawn_workers(&state); // the status-drain worker, as the CLI startup does
    let graph = state.graph.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    let base = format!("ws://{addr}");
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    // Bind an enabled expression via the set_expression command op.
    bind_expression(&mut ws, 2, &osc, "common", "max_frequency", "1 + 2").await;
    // …and report an evaluation of it, as the node's own status service will.
    {
        let uid = goofi_engine::Uid::from_hex(&osc).expect("a uid");
        let mut g = graph.lock().unwrap();
        g.apply_status(
            uid,
            goofi_engine::runtime::Status::ParamValues {
                evaluated: vec![(
                    goofi_node::ParamKey::new("common", "max_frequency"),
                    goofi_core::Param::float(3.0, 0.0, 100.0),
                )],
            },
        );
    }

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
    assert_eq!(
        ev["payload"]["values"]["common"]["max_frequency"].as_f64(),
        Some(3.0),
        "the EVALUATED value is carried, not the literal (Oscillator's own default is 30.0, so \
         `is_number()` could not tell the two apart); got {:?}",
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

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
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
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};
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

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
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
async fn a_boundary_wires_to_a_nested_sub_patch_port_over_the_wire() {
    // Inside an entered scope, a nested sub-patch shows its collapsed facade whose handles ARE its
    // own StubIds — so dropping a cable from the parent's boundary pill onto one sends exactly this
    // wire_boundary. `Command::WireStub` swallows a failed set_stub_inner as a recoverable no-op, so
    // a refusal here replies ok:true and the cable silently vanishes on the next reconcile; the
    // assertion that matters is therefore the mirrored inner, not the RPC's ok.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
    let inst_of = |v: &Value| v["result"]["inst_id"].as_str().unwrap().to_string();
    let buf = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Buffer" })).await);
    let inner = inst_of(&call(&mut ws, 2, "group_nodes", json!({ "members": [buf], "pos": [0.0, 0.0] })).await);
    let outer = inst_of(&call(&mut ws, 3, "group_nodes", json!({ "members": [inner], "pos": [0.0, 0.0] })).await);

    // Expose the buffer on the nested scope, then expose THAT port on the outer one.
    let ib = call(&mut ws, 4, "add_boundary", json!({ "inst_id": inner, "dir": "out", "dtype": "ARRAY", "pos": [0.0, 0.0] })).await
        ["result"]["bnd_id"].as_str().unwrap().to_string();
    call(&mut ws, 5, "wire_boundary", json!({ "inst_id": inner, "bnd_id": ib, "inner_node": buf, "inner_slot": "out" })).await;
    let ob = call(&mut ws, 6, "add_boundary", json!({ "inst_id": outer, "dir": "out", "dtype": "ARRAY", "pos": [0.0, 0.0] })).await
        ["result"]["bnd_id"].as_str().unwrap().to_string();
    call(&mut ws, 7, "wire_boundary", json!({ "inst_id": outer, "bnd_id": ob, "inner_node": inner, "inner_slot": ib })).await;

    let doc = sync_replica(&mut ws, |d| {
        d.read_at(&["instances", outer.as_str(), "stubs", ob.as_str(), "dir"]).is_some()
    })
    .await;
    let port = doc.read_at(&["instances", outer.as_str(), "stubs", ob.as_str()]).expect("outer stub");
    assert_eq!(port["inner_node"], json!(inner), "wired to the nested scope's facade, not dropped");
    assert_eq!(port["inner_slot"], json!(ib), "…at its own boundary port");
}

#[tokio::test]
async fn unwiring_a_boundary_over_the_wire_prunes_its_inner_target() {
    // Deleting an In→member / member→Out edge is an UNWIRE: the pill survives, its inner target
    // clears. `Command::WireStub` models exactly that (`inner: Option<…>`, "an unwire always
    // applies") and the delete path is the only door to it — so the wire must be able to say it.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
    let buf = uid(&call(&mut ws, 1, "add_node", json!({ "type": "Buffer" })).await);
    let inst = call(&mut ws, 2, "group_nodes", json!({ "members": [buf], "pos": [0.0, 0.0] })).await["result"]["inst_id"]
        .as_str()
        .unwrap()
        .to_string();
    let bnd = call(&mut ws, 3, "add_boundary", json!({ "inst_id": inst, "dir": "in", "dtype": "ARRAY", "pos": [0.0, 0.0] })).await
        ["result"]["bnd_id"].as_str().unwrap().to_string();
    let wired = call(&mut ws, 4, "wire_boundary", json!({ "inst_id": inst, "bnd_id": bnd, "inner_node": buf, "inner_slot": "data" })).await;
    assert_eq!(wired["result"]["ok"], true, "wire lands");

    // The frontend sends nulls for both halves to clear the target (graph.svelte.ts wireBoundary).
    let unwired = call(&mut ws, 5, "wire_boundary", json!({ "inst_id": inst, "bnd_id": bnd, "inner_node": null, "inner_slot": null })).await;
    assert_eq!(unwired["result"]["ok"], true, "unwire is accepted, not rejected as a missing uid");

    // The mirror writes inner_node/inner_slot only for a WIRED stub, so the leaf must be pruned.
    // The unwire already returned, so a fresh replica syncs the cleared state directly — but the
    // predicate must still see the STUB before judging its leaf, or an empty replica passes.
    let doc = sync_replica(&mut ws, |d| {
        d.read_at(&["instances", inst.as_str(), "stubs", bnd.as_str(), "dir"]).is_some()
            && d.read_at(&["instances", inst.as_str(), "stubs", bnd.as_str(), "inner_node"]).is_none()
    })
    .await;
    let port = doc.read_at(&["instances", inst.as_str(), "stubs", bnd.as_str()]).expect("stub survives");
    assert_eq!(port["inner_node"], Value::Null, "inner_node leaf pruned");
    assert_eq!(port["inner_slot"], Value::Null, "inner_slot leaf pruned");
    assert_eq!(port["dir"], "in", "the pill itself survives the unwire");

    // Naming ONE half is the third state the pair must not admit — it is neither a wire nor an
    // unwire, so it is an error rather than a silently-narrowed one of the two.
    let half = call(&mut ws, 6, "wire_boundary", json!({ "inst_id": inst, "bnd_id": bnd, "inner_node": buf })).await;
    assert!(half["error"].is_string(), "a half-specified pair is rejected, got {half}");
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

#[tokio::test]
async fn set_node_viewers_persists_the_view_state() {
    // The editor's per-slot viewer view-state (kind/settings/collapsed) is pushed via the
    // `set_node_viewers` op — soft view state (not undoable). The manager stores it on the node and
    // it survives a .gfi serialize round-trip.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;
    let _sv = recv_binary(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
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
async fn set_node_viewers_refuses_a_word_outside_the_vocabulary_and_names_the_set() {
    // The bug `page_set_panel` was taught out of, one door over. This bag carries the SAME viewer
    // kinds, keyed by the SAME output slot names, and the manager stored both as free strings — so
    // a plausible guess was answered `{ok: true}` and the stored view drew something else, or hung
    // off a slot that does not exist.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();

    let bad = call(&mut ws, 2, "set_node_viewers",
        json!({ "node": osc, "viewers": { "out": { "kind": "waveform" } } })).await;
    assert!(
        bad["error"].as_str().is_some_and(|e| e.contains("waveform") && e.contains("line")
            && e.contains("topomap")),
        "the refusal names what was asked for AND the set it could have meant: {bad}"
    );

    // …and the slot the bag is keyed by, which is its other free string.
    let slot = call(&mut ws, 3, "set_node_viewers",
        json!({ "node": osc, "viewers": { "psd": { "kind": "line" } } })).await;
    assert!(
        slot["error"].as_str().is_some_and(|e| e.contains("psd") && e.contains("out")),
        "an unknown slot is refused by naming the ones that exist: {slot}"
    );

    // A real kind on a real slot still lands, so the check is a gate and not a wall.
    let ok = call(&mut ws, 4, "set_node_viewers",
        json!({ "node": osc, "viewers": { "out": { "kind": "line" } } })).await;
    assert_eq!(ok["result"]["ok"], json!(true), "{ok}");

    // A uid naming no node stays the ENGINE's refusal, which says so by name — the slot check must
    // not shadow it with "has no output slot" on a node that is not there at all.
    let gone = call(&mut ws, 5, "set_node_viewers",
        json!({ "node": "0000000000ff", "viewers": { "out": { "kind": "line" } } })).await;
    assert!(gone["error"].as_str().is_some_and(|e| e.contains("no such node")), "{gone}");

    // …and a bag that is not a map at all is refused by saying what one looks like, rather than
    // being stored for the editor to trip over.
    let junk = call(&mut ws, 6, "set_node_viewers", json!({ "node": osc, "viewers": 7 })).await;
    assert!(junk["error"].as_str().is_some_and(|e| e.contains("map")), "{junk}");
}

#[tokio::test]
async fn serialize_and_load_roundtrip() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let ser = call(&mut ws, 2, "serialize", json!({})).await;
    let yaml = ser["result"]["yaml"].as_str().unwrap().to_string();
    assert!(yaml.contains("version: 7"), "gfi v7 header");
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
    assert!(replaced["payload"]["runtime"].is_object(), "graph_replaced seeds the runtime overlay");
    // The restored GRAPH arrives through the doc — the snapshot deliberately carries no second
    // projection of it.
    let doc = sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    let uid = doc.node_ids()[0].clone();
    assert_eq!(
        doc.read_at(&["nodes", uid.as_str(), "type"]).as_ref().and_then(|v| v.as_str()),
        Some("Oscillator"),
        "the loaded node reaches the client through the doc"
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
    use goofi_bridge::crdt::{GraphDoc, SyncMsg};

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
            ["result"]["uid"]
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

    // Capture BOTH the reply (uid) and the node_added announcement (either order); the VALUE is
    // then read from the doc, the single source clients render from.
    let mut uid: Option<String> = None;
    let mut announced = false;
    for _ in 0..20 {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(1) {
            uid = m["result"]["uid"].as_str().map(str::to_string);
        }
        if m["event"] == "node_added" {
            assert_eq!(m["payload"]["uid"], json!(uid.clone().unwrap_or_default()));
            announced = true;
        }
        if uid.is_some() && announced {
            break;
        }
    }
    let uid = uid.expect("add_node reply must arrive");
    let doc = sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    assert_eq!(
        doc_param_f64(&doc, &uid, "common", "max_frequency"),
        Some(42.0),
        "add_node applied the inline param at creation"
    );
}

#[tokio::test]
async fn add_node_restores_a_specific_uid_and_name() {
    // `add_node` can be asked for a SPECIFIC uid + display name, so a caller reconstructing a known
    // graph gets its uid-keyed links and panels to reconnect. Undo/redo do NOT come through here —
    // they are manager-owned and restore via `capture_subtree_restore` — so this pins the RPC's own
    // automation door: add a node, remove it (freeing its uid), then re-add at that uid and name.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let a = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    call(&mut ws, 2, "remove_node", json!({ "node": a })).await;

    // Re-add at the SAME uid + a specific name.
    ws.send(Message::Text(
        json!({ "id": 3, "op": "add_node", "payload": {
            "type": "Oscillator", "member_uid": a, "name": "restored_osc"
        }})
        .to_string(),
    ))
    .await
    .unwrap();
    let mut uid: Option<String> = None;
    for _ in 0..20 {
        let m = recv_text(&mut ws).await;
        if m.get("id").and_then(|v| v.as_i64()) == Some(3) {
            uid = m["result"]["uid"].as_str().map(str::to_string);
            break;
        }
    }
    assert_eq!(uid.as_deref(), Some(a.as_str()), "add_node must restore the requested uid");
    let doc = sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    assert_eq!(
        doc.read_at(&["nodes", a.as_str(), "name"]).as_ref().and_then(|v| v.as_str()),
        Some("restored_osc"),
        "add_node must restore the requested name"
    );
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

    let uid = |v: &Value| v["result"]["uid"].as_str().unwrap().to_string();
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
    assert!(
        std::path::Path::new(home["path"].as_str().unwrap()).is_absolute(),
        "an absolute path; got {home:?}"
    );
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
    let parent = repo.parent().map(spelled);
    assert_eq!(listing["result"]["parent"].as_str(), parent.as_deref());
}

#[tokio::test]
async fn save_without_a_path_is_refused() {
    // "Save in browser" is gone (user decision, 2026-08-08): a save's ONLY job is writing the
    // patch to a backend path. The old no-path form quietly returned the YAML for a browser
    // download and left the dirty flag standing — a second save semantics that C38's design
    // work would have had to carry. A save with no path is now a malformed request, not a mode.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let reply = call(&mut ws, 2, "save", json!({})).await;
    let err = reply["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("save") && err.contains("path"),
        "a path-less save is refused by name, got: {reply}"
    );
}

#[tokio::test]
async fn save_writes_a_gfi_archive() {
    // A `.gfi` is a zip holding the manifest beside the mounted workspace tree, so the saved file
    // is readable only through `read_gfi` — a bare-YAML write would leave it "not a zip archive".
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let reply = call(&mut ws, 2, "save", json!({ "path": path.to_string_lossy() })).await;
    assert!(reply.get("error").is_none(), "the save is accepted; got {reply}");

    let dest = dir.path().join("unpacked");
    let manifest = goofi_engine::archive::read_gfi(&path, &dest).unwrap();
    assert!(manifest.contains("Oscillator"), "the manifest is the serialized patch: {manifest}");
    assert!(dest.is_dir(), "the workspace tree rides along, empty or not");
}

/// C38: the MANAGER owns where the patch lives, so every client agrees about it.
///
/// It used to own none — the snapshot hard-coded `save_path: null` and only the `load` arm ever
/// announced one — so a save named the patch in the tab that performed it and nowhere else, and a
/// reload forgot the file it had just written to. Both halves are pinned here, because they fail
/// separately: the live broadcast converges the tabs that are already open, the snapshot converges
/// the ones that connect afterwards.
#[tokio::test]
async fn a_save_names_the_patch_for_every_tab_and_for_the_next_one() {
    let base = start_server().await;
    let (mut a, _) = connect_async(format!("{base}/control")).await.unwrap();
    let hello = recv_text(&mut a).await;
    assert!(hello["payload"]["save_path"].is_null(), "an unsaved patch has no home yet");
    let (mut b, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello_b = recv_text(&mut b).await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    let reply = call(&mut a, 1, "save", json!({ "path": path.to_string_lossy() })).await;
    assert!(reply.get("error").is_none(), "the save is accepted; got {reply}");

    // The other open tab learns it live…
    let ev = loop {
        let m = recv_text(&mut b).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("save_path_changed") {
            break m;
        }
    };
    assert_eq!(ev["payload"]["save_path"].as_str(), Some(spelled(&path).as_str()), "the peer is told where");

    // …and a tab opened afterwards — a reload — learns it from the snapshot alone, with no event
    // to catch. This is the half `save_path: null` made impossible.
    let (mut c, _) = connect_async(format!("{base}/control")).await.unwrap();
    let hello_c = recv_text(&mut c).await;
    assert_eq!(hello_c["payload"]["save_path"].as_str(), Some(spelled(&path).as_str()), "a reload remembers");
}

/// The stored path always names a file this patch was really written to or read from — because a
/// plain Save overwrites it silently, from any tab, with no second prompt. The two ways it could
/// come to name something else are a save that did not happen and a patch that arrived without a
/// file behind it at all.
#[tokio::test]
async fn only_a_patch_with_a_file_behind_it_keeps_a_name() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    call(&mut ws, 2, "save", json!({ "path": path.to_string_lossy() })).await;

    // A save that fails leaves the previous home standing: the patch has never been written to
    // the file it was refused, so naming it would aim the next silent overwrite at it.
    let nowhere = dir.path().join("no-such-dir").join("patch.gfi");
    let refused = call(&mut ws, 3, "save", json!({ "path": nowhere.to_string_lossy() })).await;
    assert!(refused.get("error").is_some(), "the save fails; got {refused}");
    assert_eq!(save_path_on_connect(&base).await.as_deref(), Some(spelled(&path).as_str()), "the old home stands");

    // An upload (`load_text`) carries no file, so the patch it replaces the open one with is
    // UNNAMED. Inheriting the previous path here is the silent-overwrite hazard in its purest
    // form: a different patch entirely, saved over a file it never came from.
    let yaml = call(&mut ws, 4, "serialize", json!({})).await["result"]["yaml"]
        .as_str()
        .unwrap()
        .to_string();
    call(&mut ws, 5, "load_text", json!({ "content": yaml })).await;
    assert_eq!(save_path_on_connect(&base).await, None, "an uploaded patch has no home");
}

/// Where the manager says the patch lives, read the way a joining client reads it — a fresh
/// `hello`. Peer of [`is_dirty`], and free of the same event-ordering race.
async fn save_path_on_connect(base: &str) -> Option<String> {
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let hello = recv_text(&mut ws).await;
    hello["payload"]["save_path"].as_str().map(str::to_string)
}

/// A viewer panel's binding is a node UID, and a load does not remap it — so the load has to bring
/// the uid back. It must survive into an instance that has already held OTHER nodes, which is the
/// only arrangement that can fail: a load into a fresh instance renumbers to the very values it
/// saved, and looks perfect.
#[tokio::test]
async fn a_panel_binding_survives_a_load_into_an_instance_that_held_other_nodes() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    let doc = sync_replica(&mut ws, |d| !panels(d).is_empty()).await;
    let panel = panels(&doc)[0].clone();
    let r = call(&mut ws, 2, "page_set_panel",
        json!({ "page": "Layout", "panel": panel, "type": "viewer",
                "state": { "node": osc, "slot": "out" } })).await;
    assert!(accepted(&r), "{r}");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    call(&mut ws, 3, "save", json!({ "path": path.to_string_lossy() })).await;

    // Now make the instance a USED one: three more nodes, whose uids the old load would have handed
    // the saved patch on the way back in.
    for id in 4..7 {
        call(&mut ws, id, "add_node", json!({ "type": "Buffer" })).await;
    }
    let r = call(&mut ws, 7, "load", json!({ "path": path.to_string_lossy() })).await;
    assert!(accepted(&r), "{r}");

    let d = sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    assert_eq!(d.node_ids(), vec![osc.clone()], "the patch came back with the uid it was saved with");
    let state = d
        .read_at(&["arrangement", panel.as_str(), "state"])
        .and_then(|v| v.as_str().map(str::to_string))
        .expect("the panel's state leaf");
    assert!(state.contains(&osc), "…so the viewer panel still names a node that exists: {state}");
}

/// The round trip, which is the whole point of the container: a `.gfi` written by `save` loads
/// back — both the graph and the workspace files the patch was saved with.
#[tokio::test]
async fn load_restores_the_graph_and_the_workspace_from_an_archive() {
    let (base, state) = start_server_with_state().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    std::fs::write(state.mount().join("agent.md"), b"notes").unwrap();
    // The orientation is workspace content like any other, so what the patch carries is whatever
    // its own workspace holds: here an `AGENTS.md` the agent rewrote, and NO `CLAUDE.md` at all —
    // deleted, as a patch saved before goofi ever seeded one would have none. The load below must
    // return both exactly as the archive has them.
    let learned = b"goofi-pipe: this patch's EEG source is on channel 3.\n";
    std::fs::write(state.mount().join("AGENTS.md"), learned).unwrap();
    std::fs::remove_file(state.mount().join("CLAUDE.md")).unwrap();
    // The packaging ignore list is the patch's on the same terms, and it is the ONE file the pack
    // consults as it packs: narrowed here to prove it rides its own archive rather than filtering
    // itself out of it, and that a load returns the author's list rather than the seeded default.
    let ignores = "*.wav\n";
    std::fs::write(state.mount().join(goofi_engine::archive::IGNORE_FILE), ignores).unwrap();
    call(&mut ws, 2, "save", json!({ "path": path.to_string_lossy() })).await;

    // Diverge from the saved patch on BOTH planes — a node it does not have, and a workspace that
    // no longer matches the one it packed — then load it back off disk.
    let stale = state.mount();
    call(&mut ws, 3, "add_node", json!({ "type": "Buffer" })).await;
    std::fs::remove_file(stale.join("agent.md")).unwrap();
    std::fs::write(stale.join("scratch.txt"), b"written since the save").unwrap();
    ws.send(Message::Text(
        json!({ "id": 4, "op": "load", "payload": { "path": path.to_string_lossy() } }).to_string(),
    ))
    .await
    .unwrap();

    // Collected in whichever order they arrive: the order stopped being load-bearing when the
    // manager took ownership of the save path (C38, task 5), because the snapshot the client
    // applies wholesale now names the same file, so neither message can clobber the other.
    let mut replaced = None;
    let mut save_path = None;
    while replaced.is_none() || save_path.is_none() {
        let m = recv_text(&mut ws).await;
        assert!(m.get("error").is_none(), "the archive loads; got {m}");
        match m.get("event").and_then(|v| v.as_str()) {
            Some("graph_replaced") => replaced = Some(m),
            Some("save_path_changed") => save_path = Some(m),
            _ => {}
        }
    }
    assert!(replaced.unwrap()["payload"]["runtime"].is_object());
    // The replaced graph itself arrives through the doc (the snapshot carries no node list).
    let doc = sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    let uid = doc.node_ids()[0].clone();
    assert_eq!(
        doc.read_at(&["nodes", uid.as_str(), "type"]).as_ref().and_then(|v| v.as_str()),
        Some("Oscillator"),
        "the on-disk patch replaced the diverged graph"
    );
    // The title bar names the loaded patch, so the manager reports where it came from.
    assert_eq!(save_path.unwrap()["payload"]["save_path"].as_str(), Some(spelled(&path).as_str()));

    // The workspace came back with the patch, into a mount of the load's OWN — so nothing the
    // diverged patch had written can survive into the one that replaced it.
    let mount = state.mount();
    assert_ne!(mount, stale, "a load mounts fresh");
    assert_eq!(std::fs::read(mount.join("agent.md")).unwrap(), b"notes");
    assert!(!mount.join("scratch.txt").exists(), "the diverged workspace did not follow");
    // A load seeds NOTHING. The orientation is the patch's, not goofi's: the one the agent rewrote
    // came back as it left it, and the file the patch does not have stays missing rather than being
    // conjured — goofi initialises a workspace it created, never one it unpacked someone's patch
    // into. Together these two are what stops the seed call being put back on this path.
    assert_eq!(std::fs::read(mount.join("AGENTS.md")).unwrap(), learned,
               "the load seeded over the orientation the patch was saved with");
    assert!(!mount.join("CLAUDE.md").exists(), "the load invented a file the archive never held");
    assert_eq!(
        std::fs::read_to_string(mount.join(goofi_engine::archive::IGNORE_FILE)).unwrap(),
        ignores,
        "the patch's own ignore list did not survive its round trip through the archive"
    );
    assert!(!stale.exists(), "the mount the load replaced is released, not leaked");
}

#[tokio::test]
async fn a_refused_load_leaves_the_graph_and_the_workspace_untouched() {
    let (base, state) = start_server_with_state().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let mount = state.mount();
    std::fs::write(mount.join("agent.md"), b"notes").unwrap();

    // The three ways a load is refused, in the order the arm reaches them: no such file, a file
    // that is not an archive, and — the one that pins commit-AFTER-parse — a perfectly good
    // archive, workspace and all, whose manifest the engine will not accept.
    let dir = tempfile::tempdir().unwrap();
    let junk = dir.path().join("junk.gfi");
    std::fs::write(&junk, "this: is: not: a patch").unwrap();
    let packed = dir.path().join("ws");
    std::fs::create_dir(&packed).unwrap();
    std::fs::write(packed.join("intruder.txt"), b"from the refused archive").unwrap();
    let bad = dir.path().join("bad.gfi");
    goofi_engine::archive::write_gfi(&bad, "this: is: not: a patch", &packed).unwrap();

    for (id, target) in [(2, dir.path().join("absent.gfi")), (3, junk), (4, bad)] {
        let reply = call(&mut ws, id, "load", json!({ "path": target.to_string_lossy() })).await;
        assert!(reply.get("error").is_some(), "`{}` is refused; got {reply}", target.display());
    }

    let ser = call(&mut ws, 5, "serialize", json!({})).await;
    assert!(
        ser["result"]["yaml"].as_str().unwrap().contains("Oscillator"),
        "the pre-load graph survives every failure"
    );
    assert_eq!(state.mount(), mount, "the live mount is still the one the open patch was using");
    assert_eq!(std::fs::read(mount.join("agent.md")).unwrap(), b"notes");
    assert!(!mount.join("intruder.txt").exists(), "nothing from a refused archive landed in it");
}

/// New is reached from a patch that had grown all three things a patch can have — a graph, an
/// editor arrangement and a file on disk — and it must inherit none of them. Each half fails
/// separately: the arrangement is not graph content, and the dispatch tail dirties any op it does
/// not recognise, so a New patch would be born asking to be saved over the last real one.
#[tokio::test]
async fn a_new_patch_is_empty_clean_and_unnamed() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let page = call(&mut ws, 2, "session_add_page", json!({ "name": "Second" })).await;
    assert!(page.get("error").is_none(), "the page is added; got {page}");
    call(&mut ws, 3, "save", json!({ "path": path.to_string_lossy() })).await;

    let reply = call(&mut ws, 4, "new", json!({})).await;
    assert!(reply.get("error").is_none(), "New is accepted; got {reply}");

    // The canvas of a tab that was ALREADY open when New fired. `graph_replaced` carries no node
    // list, so the only thing that empties it is the dispatch tail's re-mirror — which `new` gets
    // solely by not being in `read_only`, a silent, string-keyed omission. Read on THIS socket
    // rather than a fresh one: the broadcast deltas queued on it carry the Oscillator, so the
    // replica has to be walked back to empty rather than merely starting there.
    let doc = sync_replica(&mut ws, |d| d.node_ids().is_empty()).await;
    assert!(doc.node_ids().is_empty(), "an open tab's canvas is emptied too: {:?}", doc.node_ids());

    // Read back the way a joining client reads it — a fresh `hello`, as `is_dirty`/`save_path_on_connect`
    // do, which also keeps the dirty assertion clear of the save's own `unsaved_changes` event.
    let (mut next, _) = connect_async(format!("{base}/control")).await.unwrap();
    let hello = recv_text(&mut next).await;
    assert_eq!(hello["payload"]["unsaved_changes"], json!(false), "a New patch has nothing to save");
    assert!(hello["payload"]["save_path"].is_null(), "…and no file behind it");
    let pages = page_names(&mut next, 1).await;
    assert_eq!(pages.len(), 1, "…and none of the previous patch's panels: {pages:?}");
    let ser = call(&mut next, 2, "serialize", json!({})).await;
    let yaml = ser["result"]["yaml"].as_str().unwrap();
    assert!(!yaml.contains("Oscillator"), "…and none of its nodes: {yaml}");

    // The manager's command history goes with the patch too: an entry belonging to the graph that
    // just went away has nothing left to flip against, and its redo would put the node back. Last,
    // because the dispatch tail dirties any op it does not recognise as read-only — including an
    // undo that changed nothing — which would otherwise perturb the assertions above.
    let undo = call(&mut ws, 5, "undo", json!({})).await;
    assert_eq!(undo["result"]["changed"], json!(false), "nothing to undo across a New; got {undo}");
    assert_eq!(undo["result"]["can_undo"], json!(false), "…and none offered");
}

/// The workspace is half of what a patch is, so New mounts one of its own. `open_workspace` is how
/// a client learns where that is at all: the mount is a per-run temp directory under a random
/// name, so nothing outside the manager can derive it.
#[tokio::test]
async fn a_new_patch_mounts_an_empty_workspace_of_its_own() {
    let (base, state) = start_server_with_state().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let before = std::path::PathBuf::from(
        call(&mut ws, 1, "open_workspace", json!({})).await["result"]["path"].as_str().unwrap(),
    );
    assert_eq!(before, state.mount(), "open_workspace names the LIVE mount");
    std::fs::write(before.join("agent.md"), b"notes").unwrap();

    call(&mut ws, 2, "new", json!({})).await;

    let after = std::path::PathBuf::from(
        call(&mut ws, 3, "open_workspace", json!({})).await["result"]["path"].as_str().unwrap(),
    );
    assert_eq!(after, state.mount(), "…and follows it when New swaps it");
    assert_ne!(after, before, "New mounts fresh");
    assert!(!after.join("agent.md").exists(), "so nothing the previous patch wrote survives");
    assert!(!before.exists(), "and the mount it replaced is released, not leaked");
    // Empty of the previous patch, but not of the orientation: `new` mints the workspace, so `new`
    // is exactly the case that initialises it. It shares its dispatch arm with `load`, which must
    // NOT be seeded, and the two are one line apart — this is the half that says which is which.
    let agents = std::fs::read_to_string(after.join("AGENTS.md"))
        .expect("a New patch's workspace is seeded with the orientation");
    assert!(agents.contains("goofi-pipe is a live"), "…the real one: {agents}");
    assert_eq!(std::fs::read_to_string(after.join("CLAUDE.md")).unwrap(), "@AGENTS.md\n");
    // Asking where the workspace is is a question, not an edit — `read_only` is what keeps the
    // dispatch tail from dirtying the patch for having been asked.
    assert!(!is_dirty(&base).await, "and asking where it is did not dirty anything");
}

// A node whose FIRST instance fails to boot and whose second succeeds — so a restart is
// observable over the wire rather than just "the op did not error".
static FLAKY_MANIFEST: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "FlakyBoot",
    category: "python",
    doc: "fails setup once, then succeeds",
    inputs: &[],
    outputs: SERVE_OUT,
    params: SERVE_PARAMS,
    isolation: goofi_node::Isolation::InProcess,
    producer: true,
    factory: stub_factory,
};

struct FlakyBoot {
    fail: bool,
}
impl goofi_node::Node for FlakyBoot {
    fn setup(&mut self, _c: &mut goofi_node::NodeCtx, _p: &goofi_node::Params<'_>) -> goofi_node::NodeResult {
        if self.fail {
            return Err("boot failed".into());
        }
        Ok(())
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

async fn start_server_with_flaky_type() -> String {
    let state = AppState::new();
    let builds = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    state.graph.lock().unwrap().register_dyn_type(
        &FLAKY_MANIFEST,
        Box::new(move |_| {
            let n = builds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::new(FlakyBoot { fail: n == 0 })
        }),
    );
    spawn_stats(state.graph.clone(), state.events.clone(), 2);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    format!("ws://{addr}")
}

#[tokio::test]
async fn restart_node_rebuilds_the_instance_and_clears_the_error() {
    // The button exists to rescue a crashed node, so the wire-level proof is that the node's
    // error goes away — not merely that the op returned Ok.
    let base = start_server_with_flaky_type().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = call(&mut ws, 1, "add_node", json!({ "type": "FlakyBoot" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    ws.send(Message::Text(
        json!({ "id": 2, "op": "restart_node", "payload": { "node": uid } }).to_string(),
    ))
    .await
    .unwrap();

    let update = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("state_update")
            && m["payload"]["node"] == json!(uid)
            && m["payload"]["error"].is_null()
        {
            break m;
        }
    };
    assert!(update["payload"]["error"].is_null(), "the second instance booted clean");
}

#[tokio::test]
async fn restart_node_respawns_in_place_without_touching_undo() {
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    let buf = call(&mut ws, 2, "add_node", json!({ "type": "Buffer" })).await["result"]["uid"]
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
    expression: None,
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
    producer: true,
    factory: stub_factory,
};

#[derive(Default)]
struct Picker;
impl goofi_node::Node for Picker {
    fn on_param_refreshed(
        &mut self,
        key: &goofi_node::ParamKey,
        _p: &goofi_node::Params<'_>,
    ) -> Option<Vec<String>> {
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

// The same refreshable param, on a node that implements NO refresh hook.
static MUTE_MANIFEST: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "MutePicker",
    category: "python",
    doc: "declares a refreshable device list but implements no hook",
    inputs: &[],
    outputs: SERVE_OUT,
    params: PICKER_PARAMS,
    isolation: goofi_node::Isolation::InProcess,
    producer: true,
    factory: stub_factory,
};

#[derive(Default)]
struct MutePicker;
impl goofi_node::Node for MutePicker {
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

async fn start_server_with_picker_lacking_a_hook() -> String {
    let state = AppState::new();
    state
        .graph
        .lock()
        .unwrap()
        .register_dyn_type(&MUTE_MANIFEST, Box::new(|_| Box::<MutePicker>::default()));
    spawn_stats(state.graph.clone(), state.events.clone(), 2);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    format!("ws://{addr}")
}

async fn start_server_with_picker() -> String {
    let state = AppState::new();
    state
        .graph
        .lock()
        .unwrap()
        .register_dyn_type(&PICKER_MANIFEST, Box::new(|_| Box::<Picker>::default()));
    spawn_stats(state.graph.clone(), state.events.clone(), 2);
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

    let uid = call(&mut ws, 1, "add_node", json!({ "type": "DevicePicker" })).await["result"]["uid"]
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
async fn refresh_param_rejects_a_param_that_is_not_refreshable() {
    let base = start_server_with_runtime_type().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let osc = call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    let reply = call(&mut ws, 2, "refresh_param", json!({ "node": osc, "group": "oscillator", "name": "waveform" })).await;

    // Oscillator's waveform is a fixed list: refusing is right, and the frontend lifts the
    // spinner on a rejected call.
    assert!(reply["error"].as_str().unwrap().contains("not refreshable"), "got {reply:?}");
}

#[tokio::test]
async fn refresh_param_reports_completion_even_when_the_node_offers_nothing() {
    // A node that declares a refreshable param but implements no hook must still get its echo:
    // the ⟳ spinner is cleared by `refreshed_params`, so without it the button spins for its
    // full 15s safety timeout on every such node.
    let base = start_server_with_picker_lacking_a_hook().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    let uid = call(&mut ws, 1, "add_node", json!({ "type": "MutePicker" })).await["result"]["uid"]
        .as_str()
        .unwrap()
        .to_string();
    ws.send(Message::Text(
        json!({ "id": 2, "op": "refresh_param", "payload": { "node": uid, "group": "audio", "name": "device" } })
            .to_string(),
    ))
    .await
    .unwrap();

    let update = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("state_update") && m["payload"]["node"] == json!(uid) {
            break m;
        }
    };
    assert_eq!(update["payload"]["refreshed_params"], json!([["audio", "device"]]), "the spinner is cleared");
    assert_eq!(
        update["payload"]["params"]["audio"]["device"]["options"],
        json!(["none"]),
        "and the declared options are left as they were"
    );
}

#[tokio::test]
async fn unsaved_changes_tracks_mutations_and_clears_on_save() {
    // The title-bar dot and the unload guard both read this. It is derived, not stored: any
    // successful mutation dirties the patch, saving it (or loading another) makes it clean.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let hello = recv_text(&mut ws).await;
    assert_eq!(hello["payload"]["unsaved_changes"], json!(false), "a fresh session is clean");

    ws.send(Message::Text(
        json!({ "id": 1, "op": "add_node", "payload": { "type": "Oscillator" } }).to_string(),
    ))
    .await
    .unwrap();
    let dirty = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("unsaved_changes") {
            break m;
        }
    };
    assert_eq!(dirty["payload"]["unsaved_changes"], json!(true), "adding a node dirties the patch");

    let path = std::env::temp_dir().join(format!("goofi-dirty-{}.gfi", std::process::id()));
    ws.send(Message::Text(
        json!({ "id": 2, "op": "save", "payload": { "path": path.to_string_lossy() } }).to_string(),
    ))
    .await
    .unwrap();
    let clean = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("unsaved_changes") {
            break m;
        }
    };
    assert_eq!(clean["payload"]["unsaved_changes"], json!(false), "saving makes it clean");

    // A read-only op must not re-dirty it.
    call(&mut ws, 3, "list_nodes", json!({})).await;
    let listing = call(&mut ws, 4, "serialize", json!({})).await;
    assert!(listing["result"]["yaml"].is_string());
    // …and neither must the dispatch tail, which is why `save` is in `read_only` despite writing a
    // file: the tail's default sets the flag on any op it does not recognise, including the one
    // that just cleared it.
    assert!(!is_dirty(&base).await, "and it STAYS clean — the dispatch tail must not re-dirty a save");

    let _ = std::fs::remove_file(&path);
}

/// A save clears BOTH halves of the dirty flag, so it has to announce that on both.
///
/// `unsaved_changes` is a composite — the graph flag OR a workspace that drifted from its archive —
/// but the announcement used to ride the *flag's transition* alone. So a patch dirtied only by a
/// file written into the mount saved silently: the flag was already false, nothing was broadcast,
/// and every tab that had read `unsaved_changes: true` kept its dot and its unload guard armed on a
/// patch that was by then entirely on disk. (This is invisible to the mutation test above, whose
/// `add_node` sets the flag and so gets the transition for free.)
#[tokio::test]
async fn a_save_announces_a_clean_patch_though_only_the_workspace_was_dirty() {
    let (base, state) = start_server_with_state().await;
    std::fs::write(state.mount().join("agent.md"), b"notes").unwrap();

    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let hello = recv_text(&mut ws).await;
    assert_eq!(hello["payload"]["unsaved_changes"], json!(true), "an unpacked file is unsaved work");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    ws.send(Message::Text(
        json!({ "id": 1, "op": "save", "payload": { "path": path.to_string_lossy() } }).to_string(),
    ))
    .await
    .unwrap();
    let clean = loop {
        let m = recv_text(&mut ws).await;
        if m.get("event").and_then(|v| v.as_str()) == Some("unsaved_changes") {
            break m;
        }
    };
    assert_eq!(clean["payload"]["unsaved_changes"], json!(false), "every tab is told it is saved");
}

/// The manager's authoritative dirty flag, read the way a joining client reads it: a fresh
/// `hello`. Free of the event-ordering race — an `unsaved_changes` broadcast reaches the socket
/// through a separate task, so "no event arrived yet" is not the same as "the patch is clean".
async fn is_dirty(base: &str) -> bool {
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let hello = recv_text(&mut ws).await;
    hello["payload"]["unsaved_changes"].as_bool().expect("hello carries unsaved_changes")
}

#[tokio::test]
async fn restarting_a_node_recovers_it_without_dirtying_the_patch() {
    // The dirty gate derives "the patch differs from disk" from "the op could have mutated the
    // graph". `restart_node` is the one op where that inference is simply false: it respawns the
    // instance in place and replays the node's own ParamGroups verbatim, leaving name, position,
    // bindings, viewers, links and scopes untouched — so `serialize()` is byte-identical.
    //
    // It matters where it is reached from. A Python node raises, the inspector offers Restart, and
    // one click arms the unsaved dot and the beforeunload guard on a patch identical to the file —
    // on the RECOVERY path, where a user is least able to tell a spurious dot from a real one.
    let base = start_server().await;
    let (mut ws, _) = connect_async(format!("{base}/control")).await.unwrap();
    let _hello = recv_text(&mut ws).await;

    call(&mut ws, 1, "add_node", json!({ "type": "Oscillator" })).await;
    let yaml = call(&mut ws, 2, "serialize", json!({})).await["result"]["yaml"]
        .as_str()
        .unwrap()
        .to_string();
    // A load is how a patch becomes "the same as disk" without a filesystem.
    call(&mut ws, 3, "load_text", json!({ "content": yaml })).await;
    assert!(!is_dirty(&base).await, "a freshly loaded patch matches disk");

    let doc = sync_replica(&mut ws, |d| d.node_ids().len() == 1).await;
    let uid = doc.node_ids()[0].clone();
    let before = call(&mut ws, 4, "serialize", json!({})).await["result"]["yaml"]
        .as_str()
        .unwrap()
        .to_string();

    let restarted = call(&mut ws, 5, "restart_node", json!({ "node": uid })).await;
    assert!(restarted.get("error").is_none(), "restart succeeded: {restarted}");

    let after = call(&mut ws, 6, "serialize", json!({})).await["result"]["yaml"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(before, after, "a restart changes nothing that reaches the .gfi");
    assert!(!is_dirty(&base).await, "recovering a node must not dirty the patch");
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
