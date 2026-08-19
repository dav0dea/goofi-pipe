//! What a browser tab meets: the socket, the replica, the file door, the page — and the guard that
//! decides which pages are allowed to ask at all.
//!
//! Everything else in this crate drives `Goofi::call` and needs no transport. What is here is about
//! the wire itself: two interleaved channels on `/control` (JSON for RPC and events, binary for
//! CRDT sync), one `/data` stream per slot, `/patch.gfi` for bytes, and the SPA on the fallback.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_tests::{hex, host, http, j, panels, tool, Client, Goofi, GraphDoc, Message, SyncMsg, Viewer};
use serde_json::Value;

#[tokio::test]
async fn a_tab_is_greeted_with_the_session_frame_and_the_palette_it_can_build_from() {
    // The `hello` carries the session frame and NOTHING of the graph — structure is the doc's
    // alone. What does ride it is the runtime overlay, the one per-node truth the doc never holds:
    // its live stream carries only transitions, so a tab joining a running patch would otherwise
    // draw an errored node as healthy.
    static DISCOVERED: goofi_node::NodeManifest = goofi_node::NodeManifest {
        type_name: "DiscoveredPyNode",
        category: "python",
        doc: "a runtime type registered before serving",
        inputs: &[],
        outputs: &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }],
        params: &[],
        isolation: goofi_node::Isolation::InProcess,
        producer: true,
        factory: || unreachable!("list_nodes never instantiates"),
    };
    let g = Goofi::new();
    g.register_dyn(&DISCOVERED, Box::new(|_| unreachable!()));
    let base = g.serve().await;
    let (mut c, hello) = Client::connect(&base).await;

    assert_eq!(hello["protocol_version"], 1);
    assert!(hello["instance_id"].is_string());
    assert_eq!(hello["pillars"], j!(["signal"]), "the backend advertises what it hosts");
    assert!(hello["runtime"].as_object().is_some_and(|m| m.is_empty()), "{hello}");

    let types = c.call("list_nodes", j!({})).await["types"].as_array().cloned().unwrap();
    for want in ["Oscillator", "Buffer", "DiscoveredPyNode"] {
        // The native pair proves linkage — a crate nothing NAMES is a crate rustc drops, taking
        // every `inventory` registration with it. The third is a type this binary never compiled.
        assert!(types.iter().any(|t| t["type"] == want), "`{want}` is missing: {types:?}");
    }
    assert!(!types.iter().any(|t| t["type"] == "_TestEcho"), "test nodes stay out of the palette");

    // An add is announced as a BARE uid: everything about the node reaches clients via the doc.
    let uid = c.call("add_node", j!({ "type": "Oscillator", "pos": [10.0, 20.0] })).await["uid"]
        .as_str().unwrap().to_string();
    let added = c.event("node_added").await;
    assert_eq!(added, j!({ "uid": uid }), "no graph state rides an event: {added}");

    let mut v = Viewer::open(&base, &uid, "out").await;
    let frame = v.frame().await;
    assert_eq!((&frame[0..4], frame[4], frame[5]), (&b"GOOF"[..], 2, 0), "magic, version, ARRAY tag");
    // A slot that does not exist is a terminal refusal, not a stream that never speaks.
    let mut bad = Viewer::open(&base, &uid, "nope").await;
    assert_eq!(bad.close_code().await, Some(4004));
}

#[tokio::test]
async fn a_tab_mirrors_the_graph_off_the_binary_relay_and_follows_a_peer_editing_it() {
    // The reader half of the control plane. A replica mounts, syncs, and then learns everything
    // else as a delta — including a PEER's edits, which is what let the frontend stop writing.
    let g = Goofi::new();
    let base = g.serve().await;
    let (mut c, _) = Client::connect(&base).await;
    let mut peer = Client::connect(&base).await.0;
    let _server_sv = c.binary().await;

    let mut replica = GraphDoc::new();
    c.ws.send(Message::Binary(replica.sync_hello().into())).await.unwrap();
    replica.on_sync(SyncMsg::decode(&c.binary().await).expect("a sync frame"));
    assert!(replica.node_ids().is_empty(), "converged on the empty graph");

    // Its OWN add. Both channels are read in one loop, as a real client does: the reply carries the
    // uid and the delta carries the change, and a reader draining one throws the other away.
    c.send(j!({ "id": 1, "op": "add_node", "payload": { "type": "Oscillator" } }).to_string()).await;
    let mut uid: Option<String> = None;
    for _ in 0..20 {
        match tokio::time::timeout(Duration::from_secs(5), c.ws.next()).await {
            Ok(Some(Ok(Message::Text(t)))) => {
                let v: Value = serde_json::from_str(t.as_str()).unwrap();
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
        if uid.as_ref().is_some_and(|u| replica.node_ids().contains(u)) {
            break;
        }
    }
    let uid = uid.expect("the add was answered");
    assert_eq!(replica.read_at(&["nodes", uid.as_str(), "type"]).as_ref().and_then(Value::as_str),
               Some("Oscillator"), "the delta carried the node");

    // A PEER's layout edit. Layout is the fifth doc root, so it rides the same delta broadcast as
    // a node add — it used to reach a tab only on `hello`.
    let panel = panels(&replica).first().cloned().expect("the default page's one panel");
    let fresh = peer.call("page_split_panel", j!({ "page": "Layout", "panel": panel,
                                                  "direction": "row", "ratio": 0.5 }))
        .await.as_str().unwrap().to_string();
    for _ in 0..20 {
        if let Some(m) = SyncMsg::decode(&c.binary().await) {
            replica.on_sync(m);
        }
        if replica.read_at(&["arrangement", fresh.as_str()]).is_some() {
            break;
        }
    }
    assert_eq!(replica.read_at(&["arrangement", fresh.as_str(), "panel_type"]), Some(j!("empty")),
               "the peer's split converged, and a split births an EMPTY panel");

    // And a REMOVAL. The broadcast gate once compared state vectors, which is deletion-blind: a Yjs
    // delete does not advance the vector, so a delete-only diff was byte-identical to the empty
    // baseline and no removal ever reached a client. The gate compares logical state instead.
    peer.call("remove_node", j!({ "node": uid.clone() })).await;
    for _ in 0..20 {
        if let Some(m) = SyncMsg::decode(&c.binary().await) {
            replica.on_sync(m);
        }
        if !replica.node_ids().contains(&uid) {
            return;
        }
    }
    panic!("a removal was never broadcast: the replica still holds {uid}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tab_that_fell_behind_is_recovered_with_a_fresh_snapshot() {
    // The events plane must recover a lagged client exactly as the sync plane does, or one dropped
    // structural event desyncs its mirror for good. The victim has a tiny receive buffer and STOPS
    // reading; a flood pushes events past the shared 256-slot ring.
    let g = Goofi::new();
    let base = g.serve().await;
    let addr: std::net::SocketAddr = host(&base).parse().unwrap();
    let sock = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM,
                                    Some(socket2::Protocol::TCP)).unwrap();
    sock.set_recv_buffer_size(2048).unwrap(); // also turns off the kernel's autotuning
    sock.connect(&addr.into()).unwrap();
    sock.set_nonblocking(true).unwrap();
    let stream = tokio::net::TcpStream::from_std(std::net::TcpStream::from(sock)).unwrap();
    let (mut victim, _) = tokio_tungstenite::client_async(
        format!("{base}/control"), tokio_tungstenite::MaybeTlsStream::Plain(stream)).await.unwrap();
    victim.next().await; // the initial hello, then stall

    // Re-binding the SAME constant expression pushes a `state_update` while leaving the doc
    // unchanged, so the sync plane stays quiet and this isolates the events plane.
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
                if serde_json::from_str::<Value>(t.as_str()).unwrap()["event"] == "hello" {
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

#[tokio::test]
async fn a_patch_travels_as_bytes_between_two_instances_and_a_bad_upload_changes_nothing() {
    // `/patch.gfi` is the door onto locations no mount reaches: a container sees only what was
    // bind-mounted, while the browser runs on the host and its file dialogs reach anywhere. Two
    // servers, because a round trip through one would pass against a route that packed nothing.
    // Deliberately not a `/control` op — this is a byte stream, and the registry describes JSON.
    let src = Goofi::new();
    let source = host(&src.serve().await).to_string();
    tool(&source, "add_node", j!({ "type": "Oscillator" })).await;
    tool(&source, "add_node", j!({ "type": "Buffer" })).await;

    let (status, head, gfi) = http(&source, "GET", "/patch.gfi", "", b"").await;
    assert_eq!(status, 200, "{head}");
    assert_eq!(&gfi[..2], b"PK", "a .gfi is a zip, so it starts with the zip magic");
    assert!(head.to_ascii_lowercase().contains("content-disposition"),
            "the reply names a filename, or the browser invents one from the URL: {head}");

    let dst = Goofi::new();
    let dest = host(&dst.serve().await).to_string();
    let octet = "Content-Type: application/octet-stream\r\n";
    assert!(!tool(&dest, "inspect_patch", j!({})).await.contains("Oscillator"), "it starts empty");
    let (status, head, _) = http(&dest, "POST", "/patch.gfi", octet, &gfi).await;
    assert_eq!(status, 200, "{head}");

    let after = tool(&dest, "inspect_patch", j!({})).await;
    assert!(after.contains("Oscillator") && after.contains("Buffer"), "the whole patch: {after}");
    // …and it did NOT adopt the staging file as its home: that path is deleted the moment the load
    // returns, so the next silent Ctrl-S would aim at a file that is gone. The patch's real home is
    // on the USER's machine, which this process cannot name.
    assert!(after.contains("(never saved)"), "an uploaded patch has no server-side home: {after}");

    // A POST of something else is the wrong file in a file dialog — a caller's error, answered
    // readably, with the running patch left alone. A 500, or a wipe on the way to failing, are the
    // two ways this goes wrong.
    let (status, head, body) = http(&dest, "POST", "/patch.gfi", octet, b"not a zip").await;
    assert_eq!(status, 400, "a bad upload is the caller's error, not the server's: {head}");
    assert!(!body.is_empty(), "the refusal says why");
    assert!(tool(&dest, "inspect_patch", j!({})).await.contains("Oscillator"),
            "the live patch survived a refused upload");
}

#[tokio::test]
async fn the_app_is_served_out_of_the_binary_and_the_client_router_owns_the_rest() {
    // goofi ships as ONE file: the SPA is embedded, not read from a `frontend/build/` that has to
    // travel beside the binary and can silently go stale against it.
    let g = Goofi::new();
    let addr = host(&g.serve_spa(goofi_bridge::SPA).await).to_string();

    let (status, head, body) = http(&addr, "GET", "/index.html", "", b"").await;
    assert_eq!(status, 200, "{head}");
    assert!(String::from_utf8_lossy(&body).contains("<!doctype html"), "the real index");
    assert!(head.contains("text/html"), "…served with its content type: {head}");

    let (status, head, _) = http(&addr, "GET", "/_app/version.json", "", b"").await;
    assert_eq!(status, 200, "a hashed asset the page links: {head}");
    assert!(head.contains("application/json"), "{head}");

    let (status, _, body) = http(&addr, "GET", "/some/client/route", "", b"").await;
    assert_eq!(status, 200, "an unknown path is the SPA's own route, not a 404");
    assert!(String::from_utf8_lossy(&body).contains("<!doctype html"));
}

// ---------------------------------------------------------------------------
// The Origin/Host guard — asked of every route, WebSocket UPGRADES included
// ---------------------------------------------------------------------------

/// `/control`'s socket is CORS-exempt, so before this guard any page the user merely visited could
/// open one and read AND write the patch. `/mcp` was measured taking a no-preflight cross-origin
/// POST that created a node. `/term` is worse than both: a PTY running the user's own shell, which
/// makes a drive-by page an RCE. A guard scoped to the HTTP routes would have shut the smallest of
/// the three doors — so it is one layer over the whole router, and this asks every route.
///
/// **A drive-by guard, not authentication.** goofi is deliberately single-user and unauthenticated;
/// what is stopped is a page in the user's browser reaching a server it was never served by. A
/// client with no `Origin` — curl, an MCP client, a spawned harness, this suite — is not a browser
/// and is served, which is why every case below states exactly what it sends.
const ROUTES: &[(&str, &str)] = &[
    ("/control", "WS"),
    ("/data/deadbeef/out", "WS"),
    ("/term/no-such-instance", "WS"),
    ("/mcp", "POST"),
    ("/mcp/no-such-instance", "POST"),
    // The SPA is in this list because the page must sit behind the same door as the sockets it
    // opens — and so a route added later inherits the guard by construction.
    ("/index.html", "GET"),
];

/// One request, hand-rolled: an HTTP client crate cannot ask the WS question and a WS client crate
/// cannot ask the HTTP one, and the point is that ONE guard answers both the same way. `origin` and
/// `host` are separate arguments because a DNS-rebound name makes them AGREE.
async fn ask(addr: &str, path: &str, method: &str, headers: &str, host: &str) -> u16 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (head, body) = match method {
        "WS" => (format!(
            "GET {path} HTTP/1.1\r\nHost: {host}\r\n{headers}Connection: Upgrade\r\n\
             Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n\
             Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"), String::new()),
        "POST" => {
            let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#.to_string();
            (format!("POST {path} HTTP/1.1\r\nHost: {host}\r\n{headers}\
                      Content-Type: application/json\r\nContent-Length: {}\r\n\
                      Connection: close\r\n\r\n", body.len()), body)
        }
        _ => (format!("GET {path} HTTP/1.1\r\nHost: {host}\r\n{headers}Connection: close\r\n\r\n"),
              String::new()),
    };
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body.as_bytes()).await.unwrap();
    let mut buf = [0u8; 512];
    let n = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf))
        .await.expect("the route answered within 5s").unwrap();
    String::from_utf8_lossy(&buf[..n]).split_whitespace().nth(1)
        .expect("a status line").parse().expect("a status code")
}

#[tokio::test]
async fn every_route_answers_the_same_origin_question_the_same_way() {
    let g = Goofi::new();
    let addr = host(&g.serve_spa(&[("index.html", b"<!doctype html>goofi")]).await).to_string();
    let port = addr.rsplit(':').next().unwrap().to_string();
    // `served` rather than "not 403", so a route refusing for its own reasons cannot read as a pass.
    let served = |m: &str| if m == "WS" { 101 } else { 200 };

    // (what the caller sends, the Host it sends it to, whether it is served, why)
    let rebound = format!("evil.example:{port}");
    let lan = format!("192.168.7.5:{port}");
    let cases: Vec<(String, String, bool, &str)> = vec![
        (String::new(), addr.clone(), true,
         "a client with no Origin is not a browser: curl, an MCP client, a harness goofi spawned"),
        (format!("Origin: http://{addr}\r\n"), addr.clone(), true,
         "the page goofi served itself — every request a tab makes"),
        ("Origin: https://evil.example\r\n".into(), addr.clone(), false,
         "the load-bearing negative: a page the user merely visited gets nothing, /term least of all"),
        (format!("Origin: http://{rebound}\r\n"), rebound.clone(), false,
         "DNS rebinding is why this is an allowlist: Origin and Host AGREE and it is still refused"),
        ("Origin: http://localhost:5173\r\n".into(), addr.clone(), true,
         "another loopback port is a developer — `npm run dev` on :5173 talking to :8000"),
        ("Origin: http://[::1]:5173\r\n".into(), addr.clone(), true, "…including over IPv6"),
        (format!("Origin: http://{lan}\r\n"), lan.clone(), true,
         "the documented trusted-LAN case: goofi bound to a LAN address, serving its own page"),
        (format!("Origin: http://192.168.7.9:{port}\r\n"), lan.clone(), false,
         "…but not a neighbour on that LAN"),
        // A cross-site form POST is a page driving goofi, and a browser that puts no Origin on one
        // would sail through the rule above as "not a browser" — Safari did exactly that until
        // 15.4. `Sec-Fetch-Site` answers it: every modern browser sends it, script cannot set it
        // (it is forbidden), and no non-browser sends it at all.
        ("Sec-Fetch-Site: cross-site\r\n".into(), addr.clone(), false, "a cross-site form POST"),
        ("Sec-Fetch-Site: same-site\r\n".into(), addr.clone(), false, "a sibling subdomain"),
        ("Sec-Fetch-Site: none\r\n".into(), addr.clone(), true, "the user typing the address"),
        ("Sec-Fetch-Site: same-origin\r\n".into(), addr.clone(), true, "everything that page asks for"),
    ];

    for (headers, sent_host, ok, why) in cases {
        for (path, method) in ROUTES {
            let got = ask(&addr, path, method, &headers, &sent_host).await;
            let want = if ok { served(method) } else { 403 };
            assert_eq!(got, want, "`{path}` answered {got} for [{headers:?}] — {why}");
        }
    }
}
