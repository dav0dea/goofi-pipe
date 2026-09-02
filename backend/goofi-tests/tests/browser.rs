//! What a browser tab meets: the socket, the replica, the file door, the tab — and the guard that

use std::time::Duration;

use futures_util::StreamExt;
use goofi_tests::{hex, host, http, j, panels, tool, Client, Goofi, Message, Viewer};
use serde_json::Value;

#[tokio::test]
async fn a_tab_is_greeted_with_the_session_frame_and_the_palette_it_can_build_from() {
    // The `hello` carries the runtime overlay, the one per-node truth the doc never holds.
    static DISCOVERED: goofi_node::NodeManifest = goofi_node::NodeManifest {
        type_name: "DiscoveredPyNode",
        category: "python",
        doc: "a runtime type registered before serving",
        inputs: &[],
        outputs: &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }],
        params: &[],
        producer: true,
    };
    let g = Goofi::new();
    g.register_dyn(&DISCOVERED, Box::new(|_| unreachable!()), &goofi_node::NATIVE);
    let base = g.serve().await;
    let (mut c, hello) = Client::connect(&base).await;

    // Bumped in lockstep with `frontend/src/lib/api/control.ts`; a literal here on purpose.
    assert_eq!(hello["protocol_version"], 3);
    assert!(hello["instance_id"].is_string());
    assert!(hello["runtime"].as_object().is_some_and(|m| m.is_empty()), "{hello}");

    let types = c.call("library list", j!({})).await["types"].as_array().cloned().unwrap();
    for want in ["Oscillator", "Buffer", "DiscoveredPyNode"] {
        // Two shipped nodes, loaded from the artifacts built into the binary, beside a scanned one.
        assert!(types.iter().any(|t| t["type"] == want), "`{want}` is missing: {types:?}");
    }
    assert!(!types.iter().any(|t| t["type"] == "_TestEcho"), "test nodes stay out of the palette");

    let uid = c.call("node add", j!({ "type": "Oscillator", "pos": [10.0, 20.0] })).await["uid"]
        .as_str().unwrap().to_string();
    let added = c.event("node_added").await;
    assert_eq!(added, j!({ "uid": uid }), "no graph state rides an event: {added}");

    let mut v = Viewer::open(&base, &uid, "out").await;
    let frame = v.frame().await;
    assert_eq!((&frame[0..4], frame[4], frame[5]), (&b"GOOF"[..], 2, 0), "magic, version, ARRAY tag");
    let mut bad = Viewer::open(&base, &uid, "nope").await;
    assert_eq!(bad.close_code().await, Some(4004));
}

#[tokio::test]
async fn a_tab_mirrors_the_graph_off_the_document_events_and_follows_a_peer_editing_it() {
    let g = Goofi::new();
    let base = g.serve().await;
    let (mut c, _) = Client::connect(&base).await;
    let mut peer = Client::connect(&base).await.0;

    c.until_doc(|d| d.version() > 0).await; // the unprompted `doc_state` every connection is sent
    assert!(c.doc().node_ids().is_empty(), "seeded from the empty graph");
    assert_eq!(c.doc().version(), 1, "and at the manager's version, not at its own zero");

    // `call` reads past the delta on the way to the reply, and the replica takes it as it goes by.
    let uid = c.call("node add", j!({ "type": "Oscillator" })).await["uid"].as_str().unwrap().to_string();
    c.until_doc(|d| d.node_ids().contains(&uid)).await;
    assert_eq!(c.doc().read_at(&["nodes", uid.as_str(), "type"]).as_ref().and_then(Value::as_str),
               Some("Oscillator"), "the delta carried the node");

    let panel = panels(c.doc()).first().cloned().expect("the default tab's one panel");
    let fresh = peer.call("layout panel add", j!({ "beside": panel,
                                                   "side": "right", "ratio": 0.5 }))
        .await["id"].as_str().unwrap().to_string();
    c.until_doc(|d| panels(d).contains(&fresh)).await;
    let born = goofi_tests::arrangement_node(&c.doc().to_json()["arrangement"], &fresh).cloned();
    assert_eq!(born.map(|n| n["panel_type"].clone()), Some(j!("empty")),
               "the peer's split converged, and a split births an EMPTY panel");

    assert_eq!(c.doc().read_at(&["globals", "default_ufreq", "system"]), Some(j!(true)));
    peer.call("global add", j!({ "name": "subject", "value": "P07", "type": "string" })).await;
    c.until_doc(|d| d.read_at(&["globals", "subject", "value"]).is_some()).await;
    assert_eq!(c.doc().read_at(&["globals", "subject", "value"]), Some(j!("P07")));
    assert_eq!(c.doc().read_at(&["globals", "subject", "system"]), Some(j!(false)),
               "a user global is distinguishable from a system one in the replica");

    // A merge patch spells a delete as an explicit `null`, and the gate compares the whole projection.
    peer.call("node remove", j!({ "node": uid.clone() })).await;
    c.until_doc(|d| !d.node_ids().contains(&uid)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tab_that_fell_behind_is_recovered_with_a_fresh_snapshot() {
    // The victim has a tiny receive buffer and STOPS reading; a flood pushes past the 256-slot ring.
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

    // Re-binding the SAME expression pushes a `state_update` without touching the doc, isolating events.
    let osc = g.add("Oscillator");
    let flood = std::thread::spawn(move || {
        for _ in 0..1200 {
            g.call("node param edit", j!({ "node": hex(osc), "param": "common/max_frequency",
                                           "expression": "7" }));
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
    // Two servers, because a round trip through one would pass against a route that packed nothing.
    let src = Goofi::new();
    let source = host(&src.serve().await).to_string();
    tool(&source, "node add --type Oscillator").await;
    tool(&source, "node add --type Buffer").await;

    let (status, head, gfi) = http(&source, "GET", "/patch.gfi", "", b"").await;
    assert_eq!(status, 200, "{head}");
    assert_eq!(&gfi[..2], b"PK", "a .gfi is a zip, so it starts with the zip magic");
    assert!(head.to_ascii_lowercase().contains("content-disposition"),
            "the reply names a filename, or the browser invents one from the URL: {head}");

    let dst = Goofi::new();
    let dest = host(&dst.serve().await).to_string();
    let octet = "Content-Type: application/octet-stream\r\n";
    assert!(!tool(&dest, "nodes inspect").await.contains("Oscillator"), "it starts empty");
    let (status, head, _) = http(&dest, "POST", "/patch.gfi", octet, &gfi).await;
    assert_eq!(status, 200, "{head}");

    let after = tool(&dest, "nodes inspect").await;
    assert!(after.contains("Oscillator") && after.contains("Buffer"), "the whole patch: {after}");
    // The staging path is deleted the moment the load returns, so it must not become the patch's home.
    assert_eq!(dst.call("session status", goofi_tests::j!({}))["save_path"], serde_json::Value::Null,
               "an uploaded patch has no server-side home");

    let (status, head, body) = http(&dest, "POST", "/patch.gfi", octet, b"not a zip").await;
    assert_eq!(status, 400, "a bad upload is the caller's error, not the server's: {head}");
    assert!(!body.is_empty(), "the refusal says why");
    assert!(tool(&dest, "nodes inspect").await.contains("Oscillator"),
            "the live patch survived a refused upload");
}

#[tokio::test]
async fn the_app_is_served_out_of_the_binary_and_the_client_router_owns_the_rest() {
    // Asked FIRST and by name: an empty table makes every assertion below fail as a 404 on the route.
    assert!(
        !goofi_bridge::SPA.is_empty(),
        "no app is compiled in — build the frontend, and do not set GOOFI_HEADLESS for a run \
         that asserts the app is served"
    );
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

    // `/dev/*` is the one family the client router does NOT get handed by default. Refusing it has
    // to happen here: answering with the page would let the router mount the route anyway.
    let (status, _, _) = http(&addr, "GET", "/dev/ui", "", b"").await;
    assert_eq!(status, 404, "a development route is shut without --debug");

    let addr = host(&g.serve_spa_with(goofi_bridge::SPA, true).await).to_string();
    let (status, _, body) = http(&addr, "GET", "/dev/ui", "", b"").await;
    assert_eq!(status, 200, "…and --debug opens it");
    assert!(String::from_utf8_lossy(&body).contains("<!doctype html"), "served as the app's own route");
}

/// The Origin/Host guard, asked of every route including the WebSocket upgrades. A drive-by guard,
/// not authentication: a client with no `Origin` is not a browser and is served.
const ROUTES: &[(&str, &str, u16)] = &[
    ("/control", "WS", 101),
    ("/data/deadbeef/out", "WS", 101),
    ("/term/no-such-instance", "WS", 101),
    ("/mcp", "POST", 200),
    // 400, not 200: `/exec` refuses the probe's empty body for its own reason, and that reason
    // must stay distinguishable from the guard's 403.
    ("/exec", "POST", 400),
    // The SPA is in this list so a route added later inherits the guard by construction.
    ("/index.html", "GET", 200),
];

/// One request, hand-rolled: ONE guard answers both the HTTP and the WS question. `origin` and
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
        // `Sec-Fetch-Site` catches a cross-site form POST carrying no Origin; script cannot set it.
        ("Sec-Fetch-Site: cross-site\r\n".into(), addr.clone(), false, "a cross-site form POST"),
        ("Sec-Fetch-Site: same-site\r\n".into(), addr.clone(), false, "a sibling subdomain"),
        ("Sec-Fetch-Site: none\r\n".into(), addr.clone(), true, "the user typing the address"),
        ("Sec-Fetch-Site: same-origin\r\n".into(), addr.clone(), true, "everything that tab asks for"),
    ];

    for (headers, sent_host, ok, why) in cases {
        // The EXACT served status per route, never "not 403", so a route refusing for its own
        // reasons cannot read as a pass.
        for (path, method, served) in ROUTES {
            let got = ask(&addr, path, method, &headers, &sent_host).await;
            let want = if ok { *served } else { 403 };
            assert_eq!(got, want, "`{path}` answered {got} for [{headers:?}] — {why}");
        }
    }

    // A browser replays the first navigation's `Sec-Fetch-Site` on every reload, so navigation is
    // its own case.
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
    // The two devices edit from SPAWNED TASKS: awaiting each reply in turn would remove the race.
    // A third connects mid-flight, and reads the deltas already in its buffer as stale, not a gap.
    const BURST: usize = 12;
    let g = Goofi::new();
    let base = g.serve().await;
    let (mut a, _) = Client::connect(&base).await;
    let (mut b, _) = Client::connect(&base).await;
    a.until_doc(|d| d.version() > 0).await;
    b.until_doc(|d| d.version() > 0).await;

    let ta = tokio::spawn(async move {
        for i in 0..BURST {
            let uid = a.call("node add", j!({ "type": "Oscillator" })).await["uid"]
                .as_str().unwrap().to_string();
            a.call("node edit", j!({ "node": uid, "name": format!("osc{i}") })).await;
            a.call("node param edit", j!({ "node": uid, "param": "oscillator/amplitude",
                                           "value": 0.1 * i as f64 })).await;
        }
        a
    });
    let tb = tokio::spawn(async move {
        for i in 0..BURST {
            b.call("global add", j!({ "name": format!("g{i}"), "value": i as f64, "type": "float" })).await;
        }
        b
    });

    let (mut c, _) = Client::connect(&base).await;
    c.until_doc(|d| d.version() > 0).await;

    let mut a = ta.await.expect("device A's task");
    let mut b = tb.await.expect("device B's task");

    let want = g.call("session state", j!({}));
    for (device, client) in [("built the graph", &mut a), ("edited the globals", &mut b),
                             ("joined mid-flight", &mut c)] {
        client.until_doc(|d| d.to_json() == want).await;
        assert_eq!(client.doc().to_json(), want, "the device that {device}");
    }

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
