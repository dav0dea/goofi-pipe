//! The wire itself — everything that is a fact about the SOCKET rather than about goofi.
//!
//! Two interleaved channels on `/control`: JSON for RPC and events, binary for CRDT sync frames.
//! One `/data` stream per (node, slot), whatever the number of viewers. Every other suite drives
//! `Goofi::call` and needs none of this; what is here needs it and nothing else does.

use std::time::Duration;

use futures_util::StreamExt;
use goofi_tests::{hex, holds_within, j, panels, Client, GraphDoc, Goofi, Message, SyncMsg, Viewer};
use goofi_view::Reducible; // shape()/ndim() on a decoded frame
use serde_json::Value;

#[tokio::test]
async fn a_client_is_greeted_with_the_session_frame_and_nothing_else() {
    let g = Goofi::new();
    let (mut c, hello) = Client::connect(&g.serve().await).await;

    assert_eq!(hello["protocol_version"], 1);
    assert!(hello["instance_id"].is_string());
    assert_eq!(hello["pillars"], j!(["signal"]), "the backend advertises what it hosts");
    // No graph projection rides the snapshot — structure is the doc's alone. What DOES ride is the
    // runtime overlay, the one per-node truth the doc never holds.
    assert!(hello["runtime"].as_object().is_some_and(|m| m.is_empty()));

    let types = c.call("list_nodes", j!({})).await["types"].as_array().cloned().unwrap();
    for want in ["Oscillator", "Buffer"] {
        assert!(types.iter().any(|t| t["type"] == want),
                "every native type must survive linkage into a dependent binary: {types:?}");
    }
    assert!(!types.iter().any(|t| t["type"] == "_TestEcho"), "test nodes stay out of the palette");
}

#[tokio::test]
async fn an_add_is_announced_as_a_bare_uid_and_the_node_streams() {
    let g = Goofi::new();
    let base = g.serve().await;
    let (mut c, _) = Client::connect(&base).await;

    let uid = c.call("add_node", j!({ "type": "Oscillator", "pos": [10.0, 20.0] })).await["uid"]
        .as_str().unwrap().to_string();
    let added = c.event("node_added").await;
    assert_eq!(added.as_object().map(|o| o.len()), Some(1),
               "a bare uid — type, pos and params reach clients via the doc: {added}");
    assert_eq!(added["uid"], uid);

    let mut v = Viewer::open(&base, &uid, "out").await;
    let frame = v.frame().await;
    assert_eq!(&frame[0..4], b"GOOF", "magic");
    assert_eq!(frame[4], 2, "version");
    assert_eq!(frame[5], 0, "dtype tag ARRAY");

    // A slot that does not exist is a terminal refusal, not a stream that never speaks.
    let mut bad = Viewer::open(&base, &uid, "nope").await;
    assert_eq!(bad.close_code().await, Some(4004));
}

#[tokio::test]
async fn a_native_chain_streams_and_keeps_streaming() {
    let g = Goofi::new();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 128 }));
    g.link(osc, "out", buf, "data");

    let mut v = Viewer::open(&base, &hex(buf), "out").await;
    let frame = v.frame().await;
    assert_eq!(&frame[0..4], b"GOOF");
    assert_eq!(frame[5], 0, "Buffer emits an ARRAY");
    let body = u32::from_le_bytes(frame[10..14].try_into().unwrap());
    assert!(body > 8, "a non-trivial buffered body ({body} bytes)");

    // …and it SUSTAINS: one frame proves the plumbing, a window proves the node threads and the
    // data plane do not stall. A loose bound, so this is not a wall-clock assertion in disguise.
    let window = Duration::from_millis(400);
    let deadline = tokio::time::Instant::now() + window;
    let mut frames = 0u32;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), v.ws.next()).await {
            Ok(Some(Ok(Message::Binary(b)))) if &b[0..4] == b"GOOF" => frames += 1,
            Ok(Some(Ok(_))) => {}
            _ => break,
        }
    }
    assert!(frames >= 3, "the data plane must sustain streaming (got {frames} in {window:?})");
}

#[tokio::test]
async fn the_bridge_reduces_a_frame_to_the_viewspec_the_viewer_declared() {
    // A viewer declares its need inband; the bridge reduces on ITS OWN subscription to the
    // producer and stamps `meta.reduced` — so reduction runs off the node's thread, never in the
    // node process.
    let g = Goofi::new();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    // Well over the 2·32 envelope floor, so once it fills the last axis really shrinks.
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 128 }));
    g.link(osc, "out", buf, "data");

    let mut v = Viewer::open(&base, &hex(buf), "out").await;
    v.view(j!([{ "dtype": "array", "ndim": [["le", 2]],
                 "reduce": [{ "dim": -1, "max": 32, "method": "envelope" }] }])).await;

    // Wait for a frame that carries reduced meta AND genuinely shrank. Requiring a real shrink
    // avoids a boundary race: envelope fires at axis len ≥ 2·W = 64 producing exactly 64 samples,
    // so a frame caught with the Buffer at exactly 64 has orig_len == output — reduced meta
    // present, no shrink, and nothing proven.
    let orig_of = |d: &goofi_core::Data| -> Option<i64> {
        let goofi_core::MetaValue::Map(dims) = d.meta().reduced().as_ref()? else { return None };
        let goofi_core::MetaValue::Map(e) = dims.get(&(d.ndim() - 1).to_string())? else { return None };
        match e.get("orig_len")? {
            goofi_core::MetaValue::Uint(n) => Some(*n as i64),
            goofi_core::MetaValue::Int(n) => Some(*n),
            _ => None,
        }
    };
    let (reduced, orig) = loop {
        let d = v.decoded().await;
        if let Some(orig) = orig_of(&d) {
            if (d.shape()[d.ndim() - 1] as i64) < orig {
                break (d, orig);
            }
        }
    };

    let last = reduced.ndim() - 1;
    let goofi_core::MetaValue::Map(dims) = reduced.meta().reduced().as_ref().unwrap() else {
        panic!("reduced meta is a per-dim map")
    };
    let goofi_core::MetaValue::Map(entry) = dims.get(&last.to_string()).unwrap() else {
        panic!("a dim entry is a map")
    };
    assert_eq!(entry.get("method"), Some(&goofi_core::MetaValue::Str("envelope".into())));
    assert!(orig >= 64, "envelope fires only on a large axis; orig_len {orig}");
    // Envelope emits (min, max) per bin → at most 2·32 samples, and strictly fewer than the source.
    assert!(reduced.shape()[last] <= 64, "got {} from {orig}", reduced.shape()[last]);
}

#[tokio::test]
async fn two_viewers_of_one_slot_share_the_reducer() {
    let g = Goofi::new();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");

    let mut a = Viewer::open(&base, &hex(buf), "out").await;
    let mut b = Viewer::open(&base, &hex(buf), "out").await;
    a.decoded().await;
    b.decoded().await;
    assert_eq!(g.state.reducers.active_slots(), 1, "one reduction, fanned out to both");
}

#[tokio::test]
async fn a_boundary_port_streams_the_inner_leafs_frames() {
    // A boundary resolves chain-to-leaf to exactly one physical stream, so a viewer on the
    // instance's port gets the inner node's frames.
    let g = Goofi::new();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let sink = g.add("Buffer");
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 64 }));
    g.link(osc, "out", buf, "data");
    g.link(buf, "out", sink, "data"); // makes buf.out a CUT link when buf is grouped
    let inst = g.call("group_nodes", j!({ "members": [hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    assert!(!g.doc()["instances"][&inst]["stubs"]["out0"].is_null(), "the scope exposes out0");

    let mut v = Viewer::open(&base, &inst, "out0").await;
    assert_eq!(&v.frame().await[0..4], b"GOOF");
}

#[tokio::test]
async fn a_replica_converges_off_the_binary_sync_relay() {
    // The reader half of the CRDT control plane: a replica mounts, syncs the current graph over the
    // binary channel, and receives live deltas as the graph mutates.
    //
    // Both channels are read in ONE loop, as a real client does — the reply carries the uid and the
    // delta carries the change, and a reader that drains only one of them throws the other away.
    use futures_util::SinkExt;

    let g = Goofi::new();
    let (mut c, _) = Client::connect(&g.serve().await).await;
    let _server_sv = c.binary().await;

    let mut replica = GraphDoc::new();
    c.ws.send(Message::Binary(replica.sync_hello().into())).await.unwrap();
    replica.on_sync(SyncMsg::decode(&c.binary().await).expect("a sync frame"));
    assert!(replica.node_ids().is_empty(), "converged on the empty graph");

    c.send(j!({ "id": 1, "op": "add_node", "payload": { "type": "Oscillator" } }).to_string()).await;
    let mut uid: Option<String> = None;
    for _ in 0..20 {
        match tokio::time::timeout(Duration::from_secs(5), c.ws.next()).await {
            Ok(Some(Ok(Message::Text(txt)))) => {
                let v: Value = serde_json::from_str(txt.as_str()).unwrap();
                if v.get("id").and_then(Value::as_i64) == Some(1) {
                    uid = v["result"]["uid"].as_str().map(str::to_string);
                }
            }
            Ok(Some(Ok(Message::Binary(b)))) => {
                if let Some(m) = SyncMsg::decode(&b) {
                    replica.on_sync(m);
                }
            }
            Ok(Some(Ok(_))) => {}
            other => panic!("the socket stopped: {other:?}"),
        }
        if let Some(u) = &uid {
            if replica.node_ids().contains(u) {
                assert_eq!(replica.read_at(&["nodes", u.as_str(), "type"]).as_ref().and_then(Value::as_str),
                           Some("Oscillator"), "the delta carried the node");
                return;
            }
        }
    }
    panic!("the replica never converged on the added node ({uid:?})");
}

#[tokio::test]
async fn a_removal_reaches_a_replica_too_and_is_not_skipped_as_an_empty_delta() {
    // The broadcast gate once used a state-vector empty-diff check, which is DELETION-BLIND: a Yjs
    // delete does not advance the state vector, so a delete-only diff was byte-identical to the
    // empty baseline and every removal silently never reached a client. The gate compares the doc's
    // logical state instead.
    let g = Goofi::new();
    let (mut c, _) = Client::connect(&g.serve().await).await;
    let _server_sv = c.binary().await;
    let n = g.add("Buffer");

    let mut replica = c.replica(|d| d.node_ids().contains(&hex(n))).await;
    assert!(replica.node_ids().contains(&hex(n)), "the add reached the replica");

    g.call("remove_node", j!({ "node": hex(n) }));
    for _ in 0..20 {
        if let Some(m) = SyncMsg::decode(&c.binary().await) {
            replica.on_sync(m);
        }
        if !replica.node_ids().contains(&hex(n)) {
            return;
        }
    }
    panic!("a removal was never broadcast: the replica still holds {}", hex(n));
}

#[tokio::test]
async fn a_layout_change_reaches_a_peers_replica_as_an_ordinary_delta() {
    // Layout used to be client-owned: a peer learned an arrangement only on `hello`. As the fifth
    // doc root it rides the SAME delta broadcast as a node add, which is what made the frontend's
    // parallel write authority removable at all.
    let g = Goofi::new();
    let base = g.serve().await;
    let (mut a, _) = Client::connect(&base).await;
    let (mut b, _) = Client::connect(&base).await;

    // B holds a replica and never asks again, so anything it learns below ARRIVED as a broadcast.
    let mut peer = b.replica(|d| !panels(d).is_empty()).await;
    let panel = panels(&peer).first().cloned().expect("the default page's one panel");

    let fresh = a.call("page_split_panel", j!({ "page": "Layout", "panel": panel,
                                               "direction": "row", "ratio": 0.5 }))
        .await.as_str().unwrap().to_string();
    for _ in 0..20 {
        if let Some(m) = SyncMsg::decode(&b.binary().await) {
            peer.on_sync(m);
        }
        if peer.read_at(&["arrangement", fresh.as_str()]).is_some() {
            break;
        }
    }
    assert_eq!(peer.read_at(&["arrangement", fresh.as_str(), "panel_type"]), Some(j!("empty")),
               "the peer converged on the split, and a split births an EMPTY panel");
    assert_eq!(panels(&peer).len(), 2);
}

#[tokio::test]
async fn a_runtime_registered_type_reaches_the_palette() {
    // The full serving path a browser sees, for a type the binary was not compiled with — what the
    // CLI's node scan produces.
    static OUT: &[goofi_node::OutputDecl] =
        &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }];
    static DISCOVERED: goofi_node::NodeManifest = goofi_node::NodeManifest {
        type_name: "DiscoveredPyNode",
        category: "python",
        doc: "a runtime type registered before serving",
        inputs: &[],
        outputs: OUT,
        params: &[],
        isolation: goofi_node::Isolation::InProcess,
        producer: true,
        factory: || unreachable!("list_nodes never instantiates"),
    };

    let g = Goofi::new();
    g.register_dyn(&DISCOVERED, Box::new(|_| unreachable!()));
    let (mut c, _) = Client::connect(&g.serve().await).await;
    let types = c.call("list_nodes", j!({})).await["types"].as_array().cloned().unwrap();
    assert!(types.iter().any(|t| t["type"] == "DiscoveredPyNode"), "{types:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lagged_client_recovers_through_a_fresh_snapshot() {
    // The JSON events plane must recover a client that lagged past the shared broadcast ring,
    // exactly as the sync plane does — otherwise a dropped structural event permanently desyncs
    // its mirror. The victim has a tiny receive buffer and STOPS reading; a flooder pumps events
    // far past the 256-slot ring; on resuming, the victim must be sent a fresh `hello`.
    let g = Goofi::new();
    let base = g.serve().await;

    let addr: std::net::SocketAddr = base.trim_start_matches("ws://").parse().unwrap();
    let sock = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM,
                                    Some(socket2::Protocol::TCP)).unwrap();
    sock.set_recv_buffer_size(2048).unwrap(); // also disables the kernel's autotuning
    sock.connect(&addr.into()).unwrap();
    sock.set_nonblocking(true).unwrap();
    let stream = tokio::net::TcpStream::from_std(std::net::TcpStream::from(sock)).unwrap();
    let (mut victim, _) = tokio_tungstenite::client_async(
        format!("{base}/control"), tokio_tungstenite::MaybeTlsStream::Plain(stream)).await.unwrap();
    victim.next().await; // the initial hello, then stall

    // Re-binding the SAME constant expression pushes a `state_update` on the events plane while
    // leaving the doc unchanged, so the sync plane stays quiet and this isolates the events plane.
    let osc = g.add("Oscillator");
    let flood = std::thread::spawn(move || {
        for _ in 0..1200 {
            g.call("set_expression", j!({ "node": hex(osc), "group": "common",
                                         "name": "max_frequency", "expression": "7",
                                         "enabled": true, "triggers": false }));
        }
    });
    tokio::time::sleep(Duration::from_millis(2000)).await;

    let recovered = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            if let Some(Ok(Message::Text(t))) = victim.next().await {
                let v: Value = serde_json::from_str(t.as_str()).unwrap();
                if v["event"] == "hello" {
                    return true;
                }
            }
        }
    })
    .await
    .unwrap_or(false);
    flood.join().unwrap();
    assert!(recovered, "a lagged control client must recover via a fresh hello snapshot");
}

// ---------------------------------------------------------------------------
// `/data` peer liveness — the dead-but-not-closed viewer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_peer_that_never_pongs_is_torn_down_and_its_reducer_reclaimed() {
    // A viewer that completed the handshake and then went silent: it never sends Close, never
    // reads, never pongs. Nothing on the socket errors, so without an active probe the connection
    // lives forever and the SHARED reducer keeps reducing for a viewer that is not there.
    let g = Goofi::impatient();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    let key = (osc, "out".to_string());

    // `connect_async` performs the handshake and starts NO background task, so simply never
    // polling this stream is a faithful frozen peer: no auto-pong, no reads, no Close.
    let dead = Viewer::open(&base, &hex(osc), "out").await;
    assert!(holds_within(Duration::from_secs(2), || g.state.reducers.subscribers(&key) == 1).await,
            "the peer subscribed to the slot's reducer");
    // The assertion is the PROPERTY (reclaimed by T), not a window, and never a median.
    assert!(holds_within(Duration::from_secs(3), || g.state.reducers.active_slots() == 0).await,
            "a peer that never pongs must be torn down and its reducer reclaimed \
             (active_slots={}, subscribers={})",
            g.state.reducers.active_slots(), g.state.reducers.subscribers(&key));
    drop(dead);
}

#[tokio::test]
async fn an_idle_dead_peer_is_reclaimed_because_a_probe_is_not_its_own_proof_of_life() {
    // The half that constrains the design: a viewer of a slot that publishes NOTHING. No frame
    // write can ever vouch for this peer, and its socket buffer happily swallows every 2-byte
    // ping — so if a SENT probe counted as evidence, this connection would pin its reducer open
    // forever. Only the ANSWER counts.
    let g = Goofi::impatient();
    let base = g.serve().await;
    let buf = g.add("Buffer"); // unwired, so it never emits
    let key = (buf, "out".to_string());

    let dead = Viewer::open(&base, &hex(buf), "out").await;
    assert!(holds_within(Duration::from_secs(2), || g.state.reducers.subscribers(&key) == 1).await,
            "the idle peer subscribed");
    assert!(holds_within(Duration::from_secs(3), || g.state.reducers.active_slots() == 0).await,
            "an idle peer that never pongs must still be reclaimed (active_slots={})",
            g.state.reducers.active_slots());
    drop(dead);
}

#[tokio::test]
async fn a_slow_but_alive_viewer_keeps_its_reducer_and_its_frames() {
    // The regression this guard is most likely to cause: killing a HEALTHY viewer. Modelled on the
    // real client — the worker drains the socket while rAF coalesces — as a tab that repeatedly
    // stalls for longer than the reducer's ring holds, then catches up. Dropping FRAMES is what a
    // slow viewer is supposed to do; dropping the connection is not.
    let g = Goofi::impatient();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    let key = (osc, "out".to_string());
    let mut slow = Viewer::open(&base, &hex(osc), "out").await;

    let start = std::time::Instant::now();
    let (mut received, mut past_deadline) = (0usize, 0usize);
    for _ in 0..5 {
        // ~400 ms at the reducer's ~62 Hz is ~25 frames against a 16-slot ring, so this viewer
        // provably falls behind and provably backs the socket up.
        tokio::time::sleep(Duration::from_millis(400)).await;
        // Then drain, exactly as the real worker does — which is what makes tungstenite answer the
        // pings queued up behind the backlog.
        let drain = std::time::Instant::now();
        while drain.elapsed() < Duration::from_millis(200) {
            match tokio::time::timeout(Duration::from_millis(50), slow.ws.next()).await {
                Ok(Some(Ok(Message::Binary(_)))) => {
                    received += 1;
                    if start.elapsed() > Duration::from_millis(1000) {
                        past_deadline += 1;
                    }
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(e))) => panic!("a slow-but-alive viewer was disconnected: {e}"),
                Ok(None) => panic!("a slow-but-alive viewer's stream was closed by the bridge"),
                Err(_) => {}
            }
        }
    }

    assert_eq!(g.state.reducers.subscribers(&key), 1, "the slow viewer is still subscribed");
    assert_eq!(g.state.reducers.active_slots(), 1, "its reducer is still running");
    assert!(past_deadline > 0, "it keeps receiving frames well past the pong deadline");
    // …and it really WAS outpaced: the reducer produced far more than the ring holds during each
    // pause. How many frames it ultimately lost is not asserted — on loopback the socket buffers
    // can absorb a whole stall, and pinning an environment-dependent drop count is exactly the
    // kind of wall-clock assertion that flakes.
    let produced = g.state.reducers.reductions(&key);
    assert!(produced > 50 && received > 0,
            "the reducer outpaced the stalling viewer ({received} received of {produced} produced)");
}
