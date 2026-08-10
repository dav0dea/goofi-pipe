//! The `/mcp` endpoint driven the way a model provider's client drives it: JSON-RPC 2.0 over plain
//! HTTP POST, against the SAME live backend `/control` answers from. There is one MCP server per
//! goofi instance (user, 2026-08-10 — "Claude never spawns a server, it only creates its client"),
//! so these tests drive one server from several clients rather than one server per client.
//!
//! The HTTP client is hand-rolled over a `TcpStream`: the requests are four fixed lines and
//! `Connection: close` makes the reply "read to EOF", which is smaller than the dev-dependency an
//! HTTP client crate would add for it.

use goofi_bridge::ops::{Surface, MCP_PREFIX, REGISTRY};
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

/// One HTTP request to the MCP endpoint, returning `(status, body)`. `Connection: close` is what
/// makes the reply readable without parsing `Content-Length`.
async fn request(addr: &str, method: &str, body: &str) -> (u16, String) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let head = format!(
        "{method} /mcp HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nMCP-Protocol-Version: 2025-06-18\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(std::time::Duration::from_secs(5), s.read_to_end(&mut raw))
        .await
        .expect("the endpoint answered within 5s")
        .unwrap();
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").expect("a well-formed HTTP reply");
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, body.to_string())
}

/// A JSON-RPC call, asserting the reply carries the id it was sent with — the property that makes
/// concurrent callers safe to interleave.
async fn rpc(addr: &str, id: i64, method: &str, params: Value) -> Value {
    let req = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
    let (status, body) = request(addr, "POST", &req.to_string()).await;
    assert_eq!(status, 200, "{method} answered {status}: {body}");
    let reply: Value = serde_json::from_str(&body).expect("a JSON-RPC reply");
    assert_eq!(reply["jsonrpc"], "2.0");
    assert_eq!(reply["id"].as_i64(), Some(id), "reply carried another caller's id: {reply}");
    reply
}

/// A `tools/call`, returning the single text block a goofi tool answers with.
async fn call_tool(addr: &str, id: i64, name: &str, args: Value) -> String {
    let reply = rpc(addr, id, "tools/call", json!({ "name": name, "arguments": args })).await;
    let result = &reply["result"];
    assert_eq!(result["isError"], json!(false), "{name} failed: {result}");
    result["content"][0]["text"].as_str().expect("a text content block").to_string()
}

async fn tool_names(addr: &str) -> Vec<String> {
    let reply = rpc(addr, 1, "tools/list", json!({})).await;
    reply["result"]["tools"]
        .as_array()
        .expect("a tools array")
        .iter()
        .map(|t| t["name"].as_str().expect("every tool is named").to_string())
        .collect()
}

/// The generated list is the registry's `Mcp` rows — no more, no less. Both directions matter: a
/// tool with no row would be an op an agent can call and nothing declares, and a row with no tool
/// is an op the registry advertises and no agent can reach. Adding an op to the registry without
/// a surface decision fails HERE rather than shipping silently.
#[tokio::test]
async fn the_tool_list_is_exactly_the_registrys_mcp_surface() {
    let addr = start_server().await;
    let mut served = tool_names(&addr).await;
    let mut want: Vec<String> = REGISTRY
        .iter()
        .filter(|o| o.surface == Surface::Mcp)
        .map(|o| o.name.to_string())
        .collect();
    served.sort();
    want.sort();
    assert_eq!(served, want);
    // The exclusions are the point of the filter, so name the ones with a safety consequence:
    // each replaces the patch the agent is working inside — with, for the three sharing the `load`
    // arm, its undo history — or is the human file browser's half of that door. Both halves are
    // asserted, so flipping one of these rows to `Mcp` fails HERE rather than in review.
    // (`set_layout` is absent from both lists because the layout ops replaced it outright.)
    for off in ["load", "load_text", "new", "save", "serialize", "list_dir", "set_viewpoint"] {
        assert!(
            REGISTRY.iter().any(|o| o.name == off && o.surface == Surface::ControlOnly),
            "`{off}` is no longer a control-only row"
        );
        assert!(!served.iter().any(|t| t == off), "`{off}` reached the agent surface");
    }
}

/// A name that pushes `mcp__goofi__<name>` past 64 characters makes a provider reject the WHOLE
/// tool list with a 400 — every tool, not just that one. `ops.rs` pins the invariant on the
/// registry; this pins it on the artifact actually served, which is what a provider reads.
#[tokio::test]
async fn every_served_tool_name_fits_the_prefixed_budget() {
    let addr = start_server().await;
    for name in tool_names(&addr).await {
        assert!(
            name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "`{name}` is not [a-z0-9_]+"
        );
        assert!(
            MCP_PREFIX.len() + name.len() <= 64,
            "`{MCP_PREFIX}{name}` is {} characters — over the 64 a tool name may have",
            MCP_PREFIX.len() + name.len()
        );
    }
}

/// Every tool carries the schema a model fills in: an object with the op's arguments, and the
/// required ones marked. Advertising a required argument as optional is how a model omits the one
/// the op cannot run without.
#[tokio::test]
async fn each_tool_advertises_its_arguments_and_which_are_required() {
    let addr = start_server().await;
    let reply = rpc(&addr, 1, "tools/list", json!({})).await;
    let tools = reply["result"]["tools"].as_array().unwrap().clone();
    let add = tools.iter().find(|t| t["name"] == json!("add_node")).expect("add_node is a tool");
    assert!(add["description"].as_str().is_some_and(|d| !d.is_empty()));
    let schema = &add["inputSchema"];
    assert_eq!(schema["type"], json!("object"));
    assert_eq!(schema["properties"]["type"]["type"], json!("string"));
    assert_eq!(schema["properties"]["pos"]["type"], json!("array"));
    assert_eq!(schema["required"], json!(["type"]));
    // An op with no arguments still advertises an object schema, or a client rejects the tool.
    let list = tools.iter().find(|t| t["name"] == json!("list_nodes")).unwrap();
    assert_eq!(list["inputSchema"]["type"], json!("object"));
    assert_eq!(list["inputSchema"]["required"], json!([]));
}

/// The round trip: a client initializes, calls a tool that MUTATES the graph, and a second tool
/// call reads the mutation back out of the live engine.
#[tokio::test]
async fn a_tool_call_round_trips_against_a_live_backend() {
    let addr = start_server().await;
    let init = rpc(&addr, 1, "initialize", json!({
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": { "name": "test", "version": "0" }
    }))
    .await;
    assert!(init["result"]["capabilities"]["tools"].is_object(), "tools are advertised: {init}");
    assert!(init["result"]["serverInfo"]["name"].as_str().is_some_and(|n| n.contains("goofi")));

    let uid = call_tool(&addr, 2, "add_node", json!({ "type": "Oscillator" })).await;
    assert!(!uid.is_empty() && !uid.contains('{'), "add_node answered a bare uid, got {uid:?}");

    // …and the engine really holds it: the patch read finds the node the tool created.
    let patch = call_tool(&addr, 3, "inspect_patch", json!({})).await;
    assert!(patch.contains("Oscillator"), "inspect_patch does not see the new node:\n{patch}");
    assert!(patch.contains(&uid), "inspect_patch does not name the new uid {uid}:\n{patch}");

    // A tool that fails answers an MCP error result, not a transport failure — the model has to be
    // able to read the reason and correct itself.
    let reply = rpc(&addr, 4, "tools/call", json!({
        "name": "add_node", "arguments": { "type": "NoSuchNodeType" }
    }))
    .await;
    assert_eq!(reply["result"]["isError"], json!(true), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap_or_default();
    assert!(text.contains("NoSuchNodeType"), "the refusal names the type it refused: {text:?}");

    // An unknown tool is refused by name rather than dispatched.
    let reply = rpc(&addr, 5, "tools/call", json!({ "name": "load", "arguments": {} })).await;
    assert_eq!(reply["result"]["isError"], json!(true), "`load` is not on the agent surface: {reply}");
}

/// The user's stated requirement: ONE server per goofi instance, many clients. Two of them drive it
/// at once — their replies must not cross (each id comes back to its sender) and both mutations
/// must land, since they share the one graph rather than one graph each.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_clients_share_the_one_server() {
    let addr = start_server().await;
    let a = {
        let addr = addr.clone();
        tokio::spawn(async move {
            let mut uids = Vec::new();
            for i in 0..4 {
                uids.push(call_tool(&addr, 100 + i, "add_node", json!({ "type": "Oscillator" })).await);
            }
            uids
        })
    };
    let b = {
        let addr = addr.clone();
        tokio::spawn(async move {
            let mut uids = Vec::new();
            for i in 0..4 {
                uids.push(call_tool(&addr, 200 + i, "add_node", json!({ "type": "Buffer" })).await);
            }
            uids
        })
    };
    let (mut a, b) = (a.await.unwrap(), b.await.unwrap());
    a.extend(b);
    assert_eq!(a.len(), 8);
    // Both clients' work survived: one graph, eight distinct nodes, none overwritten by the other.
    let unique: std::collections::HashSet<&String> = a.iter().collect();
    assert_eq!(unique.len(), 8, "two clients minted a colliding uid: {a:?}");
    let patch = call_tool(&addr, 300, "inspect_patch", json!({})).await;
    for uid in &a {
        assert!(patch.contains(uid), "{uid} is missing from the shared patch:\n{patch}");
    }
}

/// `initialize` NEGOTIATES the revision rather than echoing whatever was asked for. Claiming one
/// this server does not implement is worse than declining it: `2026-07-28` requires `resultType` on
/// every result, `ttlMs`/`cacheScope` on `tools/list` and a `server/discover` method — none of which
/// are here — so a client told "yes" fails later on the missing fields instead of falling back now.
#[tokio::test]
async fn initialize_answers_with_a_revision_this_server_implements() {
    let addr = start_server().await;
    // A supported ask comes back unchanged — the half that makes this a negotiation, not a constant.
    for asked in ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"] {
        let reply = rpc(&addr, 1, "initialize", json!({ "protocolVersion": asked })).await;
        assert_eq!(reply["result"]["protocolVersion"], json!(asked), "{reply}");
    }
    // A future or malformed ask is answered with the latest revision this server DOES implement.
    for asked in ["2026-07-28", "not-a-version", "9999-99-99"] {
        let reply = rpc(&addr, 2, "initialize", json!({ "protocolVersion": asked })).await;
        assert_eq!(
            reply["result"]["protocolVersion"],
            json!("2025-11-25"),
            "`{asked}` came back verbatim: {reply}"
        );
    }
    // No ask at all keeps the documented default for a client that predates the version field.
    let reply = rpc(&addr, 3, "initialize", json!({})).await;
    assert_eq!(reply["result"]["protocolVersion"], json!("2025-06-18"), "{reply}");
}

/// A JSON-RPC batch is a JSON ARRAY, which carries no top-level `id` — so without a guard it takes
/// the notification path and the client gets 202 and no replies, hanging on requests that will never
/// be answered. Batching was removed in 2025-06-18; the honest answer is a readable refusal.
#[tokio::test]
async fn a_batch_body_is_refused_rather_than_swallowed_as_a_notification() {
    let addr = start_server().await;
    let batch = json!([
        { "jsonrpc": "2.0", "id": 4, "method": "tools/list" },
        { "jsonrpc": "2.0", "id": 5, "method": "ping" }
    ]);
    let (status, body) = request(&addr, "POST", &batch.to_string()).await;
    assert_eq!(status, 200, "a batch is answered, not swallowed: {body}");
    let reply: Value = serde_json::from_str(&body).expect("a JSON-RPC reply");
    assert_eq!(reply["error"]["code"], json!(-32600), "{reply}");
    assert!(reply["id"].is_null(), "a request that is not one object has no id to echo: {reply}");
}

/// The transport's edges. A GET is the retired standalone SSE stream — 405 says so, which is what
/// the spec tells a server that does not offer one to answer. A notification carries no id and so
/// has no reply: 202 with an empty body, or a client hangs waiting for one.
#[tokio::test]
async fn the_endpoint_answers_a_get_and_a_notification_as_the_transport_requires() {
    let addr = start_server().await;
    let (status, _) = request(&addr, "GET", "").await;
    assert_eq!(status, 405, "a GET must be refused, not answered with a stream");
    let (status, body) =
        request(&addr, "POST", &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }).to_string())
            .await;
    assert_eq!(status, 202, "a notification is accepted, not answered");
    assert!(body.is_empty(), "a 202 carries no body, got {body:?}");
}
