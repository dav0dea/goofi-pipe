//! The programmatic API — goofi driven from another crate with no socket and no server.
//!
//! `/control`, `/mcp` and this test are three transports over ONE seam, [`AppState::call`]. A test
//! that opens a WebSocket proves the transport; this proves the system.

use goofi_bridge::AppState;
use serde_json::{json, Value};

fn uid(v: &Value, key: &str) -> String {
    v.get(key).and_then(|u| u.as_str()).expect("a uid").to_string()
}

#[test]
fn the_whole_control_plane_is_reachable_without_a_transport() {
    let state = AppState::new();

    let src = uid(&state.call("add_node", json!({ "type": "_TestCounter" }), "s1").unwrap(), "uid");
    let dst = uid(&state.call("add_node", json!({ "type": "_TestEcho" }), "s1").unwrap(), "uid");
    state
        .call("add_link", json!({ "node_out": src, "slot_out": "out", "node_in": dst, "slot_in": "in" }), "s1")
        .expect("the wire attaches");

    let wired = |s: &AppState| {
        let p = s.call("inspect_patch", json!({}), "s1").unwrap();
        p["text"].as_str().expect("the patch view").contains("-- out\u{2192}in -->")
    };
    assert!(wired(&state), "the wire is in the patch");

    // Undo is session-scoped, and the session travels with the call rather than with a socket.
    state.call("undo", json!({}), "s1").expect("the wire detaches again");
    assert!(!wired(&state), "and undo took it out again");
}

#[test]
fn a_caller_hears_the_same_events_a_socket_would() {
    let state = AppState::new();
    let mut rx = state.events.subscribe();

    state.call("add_node", json!({ "type": "_TestEcho" }), "s1").unwrap();

    let ev: Value = serde_json::from_str(&rx.try_recv().expect("an event was broadcast")).unwrap();
    assert_eq!(ev["event"], "node_added", "{ev}");
}

#[test]
fn an_unknown_op_is_refused_by_name() {
    let state = AppState::new();
    let err = state.call("no_such_op", json!({}), "s1").unwrap_err();
    assert!(err.contains("no_such_op"), "{err}");
}

/// The replicated projection — the five doc roots a browser replica mirrors — read as plain JSON
/// through the ordinary op path. Without this a test can only reach it by speaking the CRDT sync
/// protocol, which pins the projection to the transport that happens to carry it today.
#[test]
fn the_state_clients_replicate_is_readable_as_plain_json() {
    let state = AppState::new();
    let n = uid(&state.call("add_node", json!({ "type": "_TestEcho" }), "s1").unwrap(), "uid");

    let doc = state.call("get_state", json!({}), "s1").unwrap();
    assert_eq!(doc["nodes"][&n]["type"], "_TestEcho", "{doc}");
    assert!(doc["globals"]["default_ufreq"].is_object(), "the seeded system globals: {doc}");

    state.call("remove_node", json!({ "node": n }), "s1").unwrap();
    let doc = state.call("get_state", json!({}), "s1").unwrap();
    assert!(doc["nodes"].get(&n).is_none(), "and a removal leaves no tombstone behind: {doc}");
}
