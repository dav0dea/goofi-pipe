//! The Origin/Host guard, asked of every route this server has — **including the three WebSocket
//! UPGRADES**, which is the whole reason this file exists rather than a `#[test]` beside the
//! function.
//!
//! `/control` is CORS-exempt (a WebSocket handshake is not a preflighted fetch), so before this
//! guard any page the user merely visited could open a socket to goofi and read AND write the
//! patch. `/mcp` was measured taking a no-preflight `text/plain` cross-origin POST that created a
//! node. `/term` is worse than either: it is a PTY running the user's own shell, which makes a
//! drive-by page an RCE. A guard scoped to the HTTP routes would have shut the smallest of the
//! three doors.
//!
//! **This is a drive-by guard, not authentication.** goofi is deliberately single-user and
//! unauthenticated; what is being stopped is a page in the user's browser reaching a server it was
//! never served by. A client with no `Origin` at all — curl, an MCP client, a spawned harness, the
//! rest of this test suite — is not a browser and is served, which is exactly why every assertion
//! below states the Origin it is sending.
//!
//! The requests are hand-rolled over a `TcpStream` for one reason: an HTTP client crate cannot ask
//! the WS question and a WS client crate cannot ask the HTTP one, and the point here is that ONE
//! guard answers both the same way.

use goofi_bridge::{serve_app, AppState};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Every route the server serves, and how it is reached. The instance and node paths name nothing
/// live on purpose: the guard runs BEFORE routing reaches a handler, so what is being proven is
/// that the route pattern is covered, and spawning a harness would only slow it down.
const ROUTES: &[(&str, &str)] = &[
    ("/control", "WS"),
    ("/data/deadbeef/out", "WS"),
    ("/term/no-such-instance", "WS"),
    ("/mcp", "POST"),
    ("/mcp/no-such-instance", "POST"),
    // The SPA the CLI mounts on the fallback. It is in this list because the guard is one layer
    // over the whole router rather than a decoration on each route — so the page is behind the
    // same door as the sockets it would open, and a route added later inherits it by construction.
    ("/index.html", "GET"),
];

/// What each route answers a request the guard let through — asserted rather than merely
/// "not 403", so a route that started refusing for its own reasons could not read as a pass.
fn served(method: &str) -> u16 {
    if method == "WS" {
        101
    } else {
        200
    }
}

/// A one-file SPA build, so the static fallback above is a real route here and not an argued one.
/// Leaked into a `OnceLock` because it has to outlive every server these tests start.
fn spa() -> std::path::PathBuf {
    static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let dir = DIR.get_or_init(|| {
        let d = tempfile::tempdir().expect("a temp dir");
        std::fs::write(d.path().join("index.html"), "<!doctype html>goofi").expect("an index");
        d
    });
    dir.path().to_path_buf()
}

async fn start_server() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        serve_app(listener, AppState::new(), Some(spa())).await.unwrap();
    });
    addr
}

/// Ask one route the guard's question and return the HTTP status. `method` is `WS` for an upgrade
/// handshake; `origin` and `host` are exactly what a browser would put on the wire, and are
/// separate arguments because a DNS-rebound name makes them AGREE.
async fn ask(addr: &str, path: &str, method: &str, origin: Option<&str>, host: &str) -> u16 {
    let o = origin.map(|o| format!("Origin: {o}\r\n")).unwrap_or_default();
    let (head, body) = match method {
        "WS" => (
            format!(
                "GET {path} HTTP/1.1\r\nHost: {host}\r\n{o}Connection: Upgrade\r\n\
                 Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n\
                 Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
            ),
            String::new(),
        ),
        "POST" => {
            let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#.to_string();
            (
                format!(
                    "POST {path} HTTP/1.1\r\nHost: {host}\r\n{o}Content-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                ),
                body,
            )
        }
        _ => (
            format!("GET {path} HTTP/1.1\r\nHost: {host}\r\n{o}Connection: close\r\n\r\n"),
            String::new(),
        ),
    };
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body.as_bytes()).await.unwrap();
    let mut buf = [0u8; 512];
    let n = tokio::time::timeout(std::time::Duration::from_secs(5), s.read(&mut buf))
        .await
        .expect("the route answered within 5s")
        .unwrap();
    let text = String::from_utf8_lossy(&buf[..n]).into_owned();
    text.split_whitespace().nth(1).expect("a status line").parse().expect("a status code")
}

/// The load-bearing negative. A page the user merely VISITED gets nothing — not a socket on
/// `/control`, not a stream on `/data`, and above all not a shell on `/term`.
#[tokio::test]
async fn every_route_refuses_a_page_that_was_served_somewhere_else() {
    let addr = start_server().await;
    for (path, method) in ROUTES {
        let got = ask(&addr, path, method, Some("https://evil.example"), &addr).await;
        assert_eq!(got, 403, "`{path}` served a foreign origin");
    }
}

/// DNS rebinding is the reason this is an ALLOWLIST and not an is-it-same-origin comparison: an
/// attacker who points `evil.example` at 127.0.0.1 makes `Origin` and `Host` agree, and every
/// same-origin test above would go on passing while the door stood open.
#[tokio::test]
async fn a_rebound_name_is_refused_even_though_it_matches_the_host_it_sent() {
    let addr = start_server().await;
    let port = addr.rsplit(':').next().unwrap();
    let rebound = format!("evil.example:{port}");
    for (path, method) in ROUTES {
        let got = ask(&addr, path, method, Some(&format!("http://{rebound}")), &rebound).await;
        assert_eq!(got, 403, "`{path}` served a rebound name");
    }
}

/// The half that keeps the app working: the page goofi served itself is served back. This is the
/// browser's every request — the `/control` socket a tab opens, the `/data` stream a viewer opens,
/// the `/term` socket the agent panel opens, and the SPA that opened all three.
#[tokio::test]
async fn every_route_serves_the_page_goofi_served_itself() {
    let addr = start_server().await;
    for (path, method) in ROUTES {
        let got = ask(&addr, path, method, Some(&format!("http://{addr}")), &addr).await;
        assert_eq!(got, served(method), "`{path}` refused its own page");
    }
}

/// A client with no `Origin` is not a browser and cannot be driven by one — curl, an MCP client, a
/// harness goofi spawned itself, and the rest of this suite. Refusing it would break every agent
/// this whole sub-project exists to serve while stopping nothing.
#[tokio::test]
async fn every_route_serves_a_client_that_is_not_a_browser_at_all() {
    let addr = start_server().await;
    for (path, method) in ROUTES {
        let got = ask(&addr, path, method, None, &addr).await;
        assert_eq!(got, served(method), "`{path}` refused a tool");
    }
}

/// A page on another LOOPBACK port is a developer's own machine, not a drive-by: nothing can serve
/// a page from loopback without already running code there. This is `npm run dev`, which proxies
/// nothing — `:5173` talking to `:8000` is the whole frontend workflow.
#[tokio::test]
async fn a_second_loopback_port_is_a_developer_not_an_attacker() {
    let addr = start_server().await;
    for origin in ["http://localhost:5173", "http://127.0.0.1:5173", "http://[::1]:5173"] {
        for (path, method) in ROUTES {
            let got = ask(&addr, path, method, Some(origin), &addr).await;
            assert_eq!(got, served(method), "`{path}` refused {origin}");
        }
    }
}

/// The trusted-LAN case the app is documented for: goofi bound to a LAN address, reached from
/// another machine's browser at that address. The page came FROM goofi, so it is served; a page
/// served by a different host on the same LAN is not, which is as far as an unauthenticated app
/// can go without becoming one that authenticates.
#[tokio::test]
async fn a_lan_address_serves_its_own_page_and_not_its_neighbours() {
    let addr = start_server().await;
    let port = addr.rsplit(':').next().unwrap();
    let lan = format!("192.168.7.5:{port}");
    for (path, method) in ROUTES {
        let mine = ask(&addr, path, method, Some(&format!("http://{lan}")), &lan).await;
        assert_eq!(mine, served(method), "`{path}` refused the LAN page it served");
        let neighbour = format!("http://192.168.7.9:{port}");
        let theirs = ask(&addr, path, method, Some(&neighbour), &lan).await;
        assert_eq!(theirs, 403, "`{path}` served a neighbour on the LAN");
    }
}
