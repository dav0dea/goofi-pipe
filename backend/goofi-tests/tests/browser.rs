//! What a browser tab meets: the socket, the replica, the file door, the tab — and the guard that
//! decides which tabs are allowed to ask at all.
//!
//! Everything else in this crate drives `Goofi::call` and needs no transport. What is here is about
//! the wire itself: two interleaved channels on `/control` (JSON for RPC and events, binary for
//! CRDT sync), one `/data` stream per slot, `/patch.gfi` for bytes, and the SPA on the fallback.

use std::time::Duration;

use futures_util::StreamExt;
use goofi_tests::{hex, host, http, j, panels, tool, Client, Goofi, Message, Viewer};
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

    // Bumped in lockstep with `frontend/src/lib/api/control.ts` — a literal here on purpose, so a
    // wire change that forgets one of the two shows up as a failing test rather than as a browser
    // reconciling against a vocabulary the manager no longer speaks.
    assert_eq!(hello["protocol_version"], 2);
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
async fn a_tab_mirrors_the_graph_off_the_document_events_and_follows_a_peer_editing_it() {
    // The reader half of the control plane. A replica is seeded whole on connect and learns
    // everything after that as a delta — including a PEER's edits, which is what let the frontend
    // stop writing.
    let g = Goofi::new();
    let base = g.serve().await;
    let (mut c, _) = Client::connect(&base).await;
    let mut peer = Client::connect(&base).await.0;

    c.until_doc(|d| d.version() > 0).await; // the unprompted `doc_state` every connection is sent
    assert!(c.doc().node_ids().is_empty(), "seeded from the empty graph");
    assert_eq!(c.doc().version(), 1, "and at the manager's version, not at its own zero");

    // Its OWN add. The reply carries the uid and the delta carries the change — on ONE socket, in
    // order, which is what lets a replica apply a patch onto the version it names instead of
    // reconciling two streams. `call` reads past the delta on the way to the reply, and the replica
    // takes it as it goes by: a client that edits is never behind one that only watches.
    let uid = c.call("add_node", j!({ "type": "Oscillator" })).await["uid"].as_str().unwrap().to_string();
    c.until_doc(|d| d.node_ids().contains(&uid)).await;
    assert_eq!(c.doc().read_at(&["nodes", uid.as_str(), "type"]).as_ref().and_then(Value::as_str),
               Some("Oscillator"), "the delta carried the node");

    // A PEER's layout edit. Layout is the fifth document root, so it rides the same delta broadcast
    // as a node add — it used to reach a tab only on `hello`.
    let panel = panels(c.doc()).first().cloned().expect("the default tab's one panel");
    let fresh = peer.call("split_panel", j!({ "panel": panel,
                                                  "direction": "row", "ratio": 0.5 }))
        .await.as_str().unwrap().to_string();
    c.until_doc(|d| d.read_at(&["arrangement", fresh.as_str()]).is_some()).await;
    assert_eq!(c.doc().read_at(&["arrangement", fresh.as_str(), "panel_type"]), Some(j!("empty")),
               "the peer's split converged, and a split births an EMPTY panel");

    // And the GLOBALS root, with the system flag a client gates rename and delete on — the document
    // is where they reach a tab, so a replica that carried nodes and no globals would look healthy.
    assert_eq!(c.doc().read_at(&["globals", "default_ufreq", "system"]), Some(j!(true)));
    peer.call("add_global", j!({ "name": "subject", "value": "P07", "type": "string" })).await;
    c.until_doc(|d| d.read_at(&["globals", "subject", "value"]).is_some()).await;
    assert_eq!(c.doc().read_at(&["globals", "subject", "value"]), Some(j!("P07")));
    assert_eq!(c.doc().read_at(&["globals", "subject", "system"]), Some(j!(false)),
               "a user global is distinguishable from a system one in the replica");

    // And a REMOVAL, which is the half a delta format is easiest to get wrong: the gate that
    // decides whether to broadcast once compared CRDT state vectors, which are deletion-blind, so
    // no removal ever reached a client. A merge patch spells a delete as an explicit `null`, and
    // the gate compares the whole projection.
    peer.call("remove_node", j!({ "node": uid.clone() })).await;
    c.until_doc(|d| !d.node_ids().contains(&uid)).await;
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
    assert_eq!(status, 200, "a hashed asset the tab links: {head}");
    assert!(head.contains("application/json"), "{head}");

    let (status, _, body) = http(&addr, "GET", "/some/client/route", "", b"").await;
    assert_eq!(status, 200, "an unknown path is the SPA's own route, not a 404");
    assert!(String::from_utf8_lossy(&body).contains("<!doctype html"));
}

// ---------------------------------------------------------------------------
// The Origin/Host guard — asked of every route, WebSocket UPGRADES included
// ---------------------------------------------------------------------------

/// `/control`'s socket is CORS-exempt, so before this guard any tab the user merely visited could
/// open one and read AND write the patch. `/mcp` was measured taking a no-preflight cross-origin
/// POST that created a node. `/term` is worse than both: a PTY running the user's own shell, which
/// makes a drive-by tab an RCE. A guard scoped to the HTTP routes would have shut the smallest of
/// the three doors — so it is one layer over the whole router, and this asks every route.
///
/// **A drive-by guard, not authentication.** goofi is deliberately single-user and unauthenticated;
/// what is stopped is a tab in the user's browser reaching a server it was never served by. A
/// client with no `Origin` — curl, an MCP client, a spawned harness, this suite — is not a browser
/// and is served, which is why every case below states exactly what it sends.
const ROUTES: &[(&str, &str)] = &[
    ("/control", "WS"),
    ("/data/deadbeef/out", "WS"),
    ("/term/no-such-instance", "WS"),
    ("/mcp", "POST"),
    ("/mcp/no-such-instance", "POST"),
    // The SPA is in this list because the tab must sit behind the same door as the sockets it
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
         "the tab goofi served itself — every request a tab makes"),
        ("Origin: https://evil.example\r\n".into(), addr.clone(), false,
         "the load-bearing negative: a tab the user merely visited gets nothing, /term least of all"),
        (format!("Origin: http://{rebound}\r\n"), rebound.clone(), false,
         "DNS rebinding is why this is an allowlist: Origin and Host AGREE and it is still refused"),
        ("Origin: http://localhost:5173\r\n".into(), addr.clone(), true,
         "another loopback port is a developer — `npm run dev` on :5173 talking to :8000"),
        ("Origin: http://[::1]:5173\r\n".into(), addr.clone(), true, "…including over IPv6"),
        (format!("Origin: http://{lan}\r\n"), lan.clone(), true,
         "the documented trusted-LAN case: goofi bound to a LAN address, serving its own tab"),
        (format!("Origin: http://192.168.7.9:{port}\r\n"), lan.clone(), false,
         "…but not a neighbour on that LAN"),
        // A cross-site form POST is a tab driving goofi, and a browser that puts no Origin on one
        // would sail through the rule above as "not a browser" — Safari did exactly that until
        // 15.4. `Sec-Fetch-Site` answers it: every modern browser sends it, script cannot set it
        // (it is forbidden), and no non-browser sends it at all.
        ("Sec-Fetch-Site: cross-site\r\n".into(), addr.clone(), false, "a cross-site form POST"),
        ("Sec-Fetch-Site: same-site\r\n".into(), addr.clone(), false, "a sibling subdomain"),
        ("Sec-Fetch-Site: none\r\n".into(), addr.clone(), true, "the user typing the address"),
        ("Sec-Fetch-Site: same-origin\r\n".into(), addr.clone(), true, "everything that tab asks for"),
    ];

    for (headers, sent_host, ok, why) in cases {
        for (path, method) in ROUTES {
            let got = ask(&addr, path, method, &headers, &sent_host).await;
            let want = if ok { served(method) } else { 403 };
            assert_eq!(got, want, "`{path}` answered {got} for [{headers:?}] — {why}");
        }
    }

    // A NAVIGATION is not a subresource, and the three cases below cannot share the loop above
    // because the method is what separates them. This was the guard's own bug: a browser replays
    // the first navigation's `Sec-Fetch-Site` on every later reload, so a tab opened from anywhere
    // but the address bar answered `cross-site` for ever after — measured in Chromium — and the
    // only way back in was to retype the URL. Restarting goofi and hitting reload is when a user
    // meets it.
    const NAV: &str = "Sec-Fetch-Site: cross-site\r\nSec-Fetch-Mode: navigate\r\n";
    let doc = format!("{NAV}Sec-Fetch-Dest: document\r\n");
    assert_eq!(ask(&addr, "/index.html", "GET", &doc, &addr).await, 200,
               "reloading the tab is the user arriving, not a tab reaching in");
    assert_eq!(ask(&addr, "/index.html", "GET", &format!("{NAV}Sec-Fetch-Dest: iframe\r\n"), &addr).await,
               403, "…but a tab FRAMING goofi is the drive-by, and only `Dest` tells them apart");
    assert_eq!(ask(&addr, "/mcp", "POST", &doc, &addr).await, 403,
               "…and a cross-site form POST is a navigation too, which is why the method is asked");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_devices_edit_one_patch_at_once_and_end_on_the_same_document() {
    // The multi-client promise, stated as a scenario rather than as reasoning: one goofi, one
    // patch, several frontends editing it SIMULTANEOUSLY, and no replica left behind.
    //
    // Nothing merges here, and that is the design. Every mutation is an RPC the manager applies
    // under the graph lock, and the delta it broadcasts is computed and sent while the document
    // lock is still held — so two writers cannot interleave into out-of-order versions no matter
    // how they race. A replica only ever follows.
    //
    // The two devices edit from SPAWNED TASKS, not one after another: awaiting each reply in turn
    // would let every delta land before the next call went out, and the race this is about would
    // never happen. A third device connects while they run, which is the window the manager pays
    // for by subscribing a socket BEFORE it snapshots — that client is seeded with a document
    // already holding edits whose deltas are also in its buffer, and it has to read those as stale
    // rather than as a gap.
    const BURST: usize = 12;
    let g = Goofi::new();
    let base = g.serve().await;
    let (mut a, _) = Client::connect(&base).await;
    let (mut b, _) = Client::connect(&base).await;
    a.until_doc(|d| d.version() > 0).await;
    b.until_doc(|d| d.version() > 0).await;

    let ta = tokio::spawn(async move {
        for i in 0..BURST {
            let uid = a.call("add_node", j!({ "type": "Oscillator" })).await["uid"]
                .as_str().unwrap().to_string();
            a.call("rename_node", j!({ "node": uid, "name": format!("osc{i}") })).await;
            a.call("update_param", j!({ "node": uid, "group": "oscillator", "name": "amplitude",
                                        "value": 0.1 * i as f64 })).await;
        }
        a
    });
    let tb = tokio::spawn(async move {
        for i in 0..BURST {
            b.call("add_global", j!({ "name": format!("g{i}"), "value": i as f64, "type": "float" })).await;
        }
        b
    });

    // Mid-flight, with both bursts in the air.
    let (mut c, _) = Client::connect(&base).await;
    c.until_doc(|d| d.version() > 0).await;

    let mut a = ta.await.expect("device A's task");
    let mut b = tb.await.expect("device B's task");

    // Every device ends on the manager's document, and therefore on each other's.
    let want = g.call("get_state", j!({}));
    for (device, client) in [("built the graph", &mut a), ("edited the globals", &mut b),
                             ("joined mid-flight", &mut c)] {
        client.until_doc(|d| d.to_json() == want).await;
        assert_eq!(client.doc().to_json(), want, "the device that {device}");
    }

    // …and it is the document BOTH of them authored, not one of the two halves.
    let d = c.doc();
    assert_eq!(d.node_ids().len(), BURST, "every node device A added");
    for i in 0..BURST {
        let uid = d.to_json()["nodes"].as_object().unwrap().iter()
            .find(|(_, n)| n["name"] == j!(format!("osc{i}"))).map(|(u, _)| u.clone())
            .unwrap_or_else(|| panic!("osc{i} is missing from the replica"));
        assert_eq!(d.read_at(&["nodes", uid.as_str(), "params", "oscillator", "amplitude", "value"]),
                   Some(j!(0.1 * i as f64)), "A's rename and its param edit both landed on osc{i}");
        assert_eq!(d.read_at(&["globals", &format!("g{i}"), "value"]), Some(j!(i as f64)),
                   "device B's global g{i}");
    }
}
