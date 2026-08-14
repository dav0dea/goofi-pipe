//! `/patch.gfi` — the patch as a FILE the browser carries, in both directions.
//!
//! This is the door onto locations no mount reaches. A container sees only what was bind-mounted,
//! and that boundary cannot be lifted from inside; the browser, however, runs on the host and its
//! own file dialogs reach anywhere. So the bytes travel over HTTP and the browser does the
//! filesystem part.
//!
//! Deliberately NOT a `/control` op: this is a byte stream, and the ops registry describes JSON
//! request/reply pairs. What the route must not do is grow its own idea of what a `.gfi` is — both
//! directions go through the very same `archive` functions a disk save and a disk load use, so the
//! two routes cannot drift from the two ops.
//!
//! The HTTP client is hand-rolled over a `TcpStream` for the reason `mcp.rs` gives: `Connection:
//! close` makes a reply "read to EOF", which is smaller than the dev-dependency an HTTP client
//! crate would add.

use goofi_bridge::{serve_app, spawn_tick, AppState};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn start_server() -> String {
    let state = AppState::new();
    spawn_tick(state.graph.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_app(listener, state, None).await.unwrap();
    });
    addr.to_string()
}

/// One HTTP request, returning `(status, headers, body-bytes)`. The body stays BYTES: a `.gfi` is
/// a zip, and lossy-decoding it to a string would corrupt exactly what these tests are about.
async fn request(addr: &str, method: &str, path: &str, ctype: &str, body: &[u8]) -> (u16, String, Vec<u8>) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: {ctype}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(std::time::Duration::from_secs(10), s.read_to_end(&mut raw))
        .await
        .expect("the endpoint answered within 10s")
        .unwrap();
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("a well-formed HTTP reply");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, head, raw[split + 4..].to_vec())
}

/// An MCP `tools/call`, used here only to put something in the graph and to read it back.
async fn tool(addr: &str, name: &str, args: Value) -> String {
    let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": { "name": name, "arguments": args } });
    let (status, _, body) = request(addr, "POST", "/mcp", "application/json", req.to_string().as_bytes()).await;
    assert_eq!(status, 200, "{name} answered {status}");
    let reply: Value = serde_json::from_slice(&body).expect("a JSON-RPC reply");
    assert_eq!(reply["result"]["isError"], json!(false), "{name} failed: {}", reply["result"]);
    reply["result"]["content"][0]["text"].as_str().unwrap().to_string()
}

/// The whole point, end to end: a patch leaves one goofi as bytes and arrives in ANOTHER one as
/// the same graph. Two servers rather than one, because a round trip through the same instance
/// would pass against a route that returned a cached manifest and never packed anything.
#[tokio::test]
async fn a_patch_exported_as_bytes_loads_into_a_different_instance() {
    let source = start_server().await;
    tool(&source, "add_node", json!({ "type": "Oscillator" })).await;
    tool(&source, "add_node", json!({ "type": "Buffer" })).await;

    let (status, head, gfi) = request(&source, "GET", "/patch.gfi", "text/plain", b"").await;
    assert_eq!(status, 200, "{head}");
    assert_eq!(&gfi[..2], b"PK", "a .gfi is a zip, so it starts with the zip magic");
    assert!(
        head.to_ascii_lowercase().contains("content-disposition"),
        "the reply names a filename, or the browser saves it as `patch.gfi` from the URL: {head}"
    );

    // A DIFFERENT instance, whose graph starts empty.
    let dest = start_server().await;
    let before = tool(&dest, "inspect_patch", json!({})).await;
    assert!(!before.contains("Oscillator"), "the destination starts empty: {before}");

    let (status, head, _) = request(&dest, "POST", "/patch.gfi", "application/octet-stream", &gfi).await;
    assert_eq!(status, 200, "{head}");

    let after = tool(&dest, "inspect_patch", json!({})).await;
    assert!(after.contains("Oscillator"), "the imported patch is live: {after}");
    assert!(after.contains("Buffer"), "…all of it: {after}");

    // …and it did NOT adopt the staging file as its home. The upload was written to a temp path
    // and deleted the moment the load returned, so a patch that adopted it would aim the next
    // silent Ctrl-S at a file that no longer exists — overwriting nothing, or worse, recreating a
    // stray `.gfi` in the system temp directory. The patch's real home is on the USER's machine,
    // which this process cannot name, so "never saved" is the only honest answer.
    assert!(
        after.contains("(never saved)"),
        "an uploaded patch has no server-side home: {after}"
    );
}

/// A POST of something that is not an archive is a USER error — the wrong file in a file dialog —
/// so it answers 400 with a readable reason and leaves the running patch alone. Returning 500, or
/// wiping the graph on the way to failing, are the two ways this goes wrong.
#[tokio::test]
async fn a_post_that_is_not_an_archive_is_refused_and_changes_nothing() {
    let addr = start_server().await;
    tool(&addr, "add_node", json!({ "type": "Oscillator" })).await;

    let (status, head, body) = request(&addr, "POST", "/patch.gfi", "application/octet-stream", b"not a zip").await;
    assert_eq!(status, 400, "a bad upload is the caller's error, not the server's: {head}");
    assert!(!body.is_empty(), "the refusal says why");

    let after = tool(&addr, "inspect_patch", json!({})).await;
    assert!(after.contains("Oscillator"), "the live patch survived a refused upload: {after}");
}
