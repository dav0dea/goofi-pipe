//! The MCP endpoint — the op registry's third consumer, after `dispatch` and the frontend's
//! `OpName` union. Every `Surface::Mcp` row becomes one tool; nothing is hand-listed, because a
//! hand-list is exactly how a name drifts past the 64-character budget that makes a provider reject
//! the WHOLE tool list with a 400.
//!
//! **Why a route and not a process.** The user's lifecycle decision (2026-08-10) is ONE MCP server
//! per goofi instance — "Claude never spawns a server, it only creates its client". A stdio server
//! is spawned *by* its client and is therefore one per client, so the transport has to be HTTP; and
//! once goofi owns the server's lifetime, a sidecar process buys nothing and costs a process, a
//! port and a supervision path. So it is `/mcp` on the axum server that already serves `/control`
//! and `/data`, and there is no lazy connect, no "goofi isn't running" path and no reconnect.
//!
//! **Why there is no async seam.** A tool call is a SYNCHRONOUS [`crate::dispatch`] inside an async
//! task — byte for byte what the `/control` socket does — so nothing awaits while the graph lock is
//! held. That is load-bearing: an async dispatch would re-open W0's save-on-lock and W1's
//! rescan-on-lock, and moving the save off-lock once cost a discarded attempt ~450 lines of race
//! machinery.
//!
//! **Protocol shape.** One JSON object per POST, no session state, no SSE, no `Mcp-Session-Id`. The
//! MCP revisions split into two eras — `2025-11-25` and earlier open with an `initialize`
//! handshake, `2026-07-28` and later carry version and identity per request and have no handshake
//! at all — and a stateless tools-only server serves both by answering `initialize` when it comes
//! and never requiring it. Nothing behaves differently across the revisions in
//! [`SUPPORTED_PROTOCOLS`], which is why one code path serves them all; a revision outside that set
//! is negotiated DOWN rather than echoed, because claiming one this server does not implement only
//! moves the failure later.
//!
//! **One deliberate deviation.** An unknown or withheld tool name comes back as a `CallToolResult`
//! with `isError: true`, where the spec classes it as a protocol error (`-32602`). That is on
//! purpose: `isError` puts the reason in front of the MODEL, which is the only party that can
//! correct the call, whereas a JSON-RPC error is swallowed by the client's transport layer.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::ops::{self, Surface};
use crate::AppState;

/// The undo/redo scope every MCP call runs in. Undo is per-session so that one browser tab cannot
/// undo another's edit; an agent has no session identity to present (the modern transport is
/// stateless by design, and the legacy one's session id is optional and client-opaque), so every
/// agent call shares one stack. That is the isolation that matters: an agent's undo reaches its own
/// edits and never into a human tab's stack. Two agents on one patch are collaborators on one
/// graph — the same thing two humans in two tabs are.
const AGENT_SESSION: &str = "mcp";

/// The revision to claim when a client names none. `2025-06-18` is the last one that a client which
/// omits the field can be assumed to speak (the transport spec says to assume `2025-03-26` for a
/// missing header, and every client that sends none at all predates the split).
const DEFAULT_PROTOCOL: &str = "2025-06-18";

/// The newest revision this server implements, and what an unsupported ask is answered with.
const LATEST_PROTOCOL: &str = "2025-11-25";

/// Every revision whose `initialize`/`tools/list`/`tools/call` shapes this server actually speaks.
/// `2026-07-28` is deliberately absent: it requires `resultType` on every result, `ttlMs`/
/// `cacheScope` on `tools/list` and a `server/discover` method, none of which are here.
const SUPPORTED_PROTOCOLS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18", LATEST_PROTOCOL];

/// One registry argument type as JSON Schema. The list `ops.rs` admits is pinned by its own schema
/// test, so the string arm is a genuine default (`uid` and `string` share it) rather than a
/// fallback for a type that could slip through.
fn json_schema(ty: &str) -> Value {
    if let Some(item) = ty.strip_suffix("[]") {
        return json!({ "type": "array", "items": json_schema(item) });
    }
    match ty {
        "float" => json!({ "type": "number" }),
        "int" => json!({ "type": "integer" }),
        "bool" => json!({ "type": "boolean" }),
        // A canvas position. Bounded, so a model cannot send one number and get a silent default.
        "float2" => {
            json!({ "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 })
        }
        // An opaque value the engine round-trips (a param value, a panel's state bag): any JSON.
        // An empty schema is how JSON Schema spells "anything", and the doc line says what it is.
        "json" => json!({}),
        _ => json!({ "type": "string" }),
    }
}

/// The tool list, generated from the registry's agent surface.
pub fn tools() -> Vec<Value> {
    ops::REGISTRY
        .iter()
        .filter(|op| op.surface == Surface::Mcp)
        .map(|op| {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();
            for (name, ty, req) in op.args() {
                properties.insert(name.to_string(), json_schema(ty));
                if req {
                    required.push(json!(name));
                }
            }
            json!({
                "name": op.name,
                // The result schema rides in the description: a model that knows the reply's shape
                // chains the next call instead of probing for it.
                "description": format!("{}\n\nReturns: {}", op.doc, op.result),
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        })
        .collect()
}

/// A tool's answer as the text a model reads. The inspect ops already produce prose (`{text: …}`)
/// and several ops answer a bare uid, so re-encoding either as JSON would only add quoting for the
/// model to strip; everything else is pretty-printed, which reads in a transcript.
fn render(result: &Value) -> String {
    if let Value::String(s) = result {
        return s.clone();
    }
    match result.as_object() {
        Some(o) if o.len() == 1 => match o.get("text").and_then(|t| t.as_str()) {
            Some(t) => t.to_string(),
            None => serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string()),
        },
        _ => serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string()),
    }
}

/// A `CallToolResult`. A failed op is `isError` inside a successful JSON-RPC reply, not a transport
/// error: that is what puts the manager's refusal in front of the model, which is the only way it
/// can correct itself.
fn tool_result(text: String, is_error: bool) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

/// Run one tool by handing the registry op straight to the control-plane dispatcher.
fn call_tool(state: &AppState, params: &Value) -> Value {
    let name = params.get("name").and_then(|v| v.as_str()).unwrap_or_default();
    let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
    // The registry is the gate, checked BEFORE dispatch: `load`/`new`/`save` have live arms and
    // would run perfectly well if asked for by name, and each replaces the patch the agent is
    // working inside — with, for the three sharing the `load` arm, its undo history.
    if !ops::find(name).is_some_and(|op| op.surface == Surface::Mcp) {
        return tool_result(format!("unknown tool `{name}`"), true);
    }
    let req = json!({ "id": 1, "op": name, "payload": arguments, "session": AGENT_SESSION });
    // Synchronous, exactly as the `/control` handler calls it — see the module note. `dispatch`
    // answers `None` only for a request with no numeric id, and this one always has one.
    let Some(reply) = crate::dispatch(state, &req.to_string()) else {
        return tool_result(format!("`{name}` returned no reply"), true);
    };
    let reply: Value = serde_json::from_str(&reply).unwrap_or(Value::Null);
    if let Some(err) = reply.get("error") {
        let text = err.as_str().map(str::to_string).unwrap_or_else(|| err.to_string());
        return tool_result(text, true);
    }
    match reply.get("result") {
        Some(result) => tool_result(render(result), false),
        None => tool_result(format!("`{name}` answered neither a result nor an error"), true),
    }
}

fn ok(id: Value, result: Value) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })).into_response()
}

fn rpc_error(id: Value, code: i64, message: String) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }))
        .into_response()
}

/// The MCP endpoint. Registered with `post`, so axum answers the retired GET stream and the retired
/// DELETE session-teardown with the 405 the transport spec asks a server without them to give.
pub async fn endpoint(State(state): State<AppState>, body: String) -> Response {
    let req: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return rpc_error(Value::Null, -32700, format!("parse error: {e}")),
    };
    // A batch is a JSON ARRAY, and an array has no top-level `id` — so without this it would fall
    // into the notification branch and every request in it would go unanswered. Batching was
    // retired in 2025-06-18, so refusing it readably beats leaving a client waiting forever.
    let Some(req) = req.as_object() else {
        return rpc_error(Value::Null, -32600, "invalid request: expected one JSON-RPC object".into());
    };
    // A notification carries no id and therefore has no reply. 202-with-no-body is what the
    // transport requires; a JSON-RPC object here leaves the client matching a response to nothing.
    let Some(id) = req.get("id").filter(|v| !v.is_null()).cloned() else {
        return StatusCode::ACCEPTED.into_response();
    };
    let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
    match req.get("method").and_then(|v| v.as_str()).unwrap_or_default() {
        // Answered for a legacy client, never required of a modern one. A supported ask is echoed;
        // anything else is answered with the newest revision this server really implements, which
        // is what the spec asks of a server that cannot speak what it was offered.
        "initialize" => ok(
            id,
            json!({
                "protocolVersion": match params.get("protocolVersion").and_then(|v| v.as_str()) {
                    None => DEFAULT_PROTOCOL,
                    Some(v) if SUPPORTED_PROTOCOLS.contains(&v) => v,
                    Some(_) => LATEST_PROTOCOL,
                },
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "goofi-pipe", "version": env!("CARGO_PKG_VERSION") },
                "instructions": "Build and inspect the running goofi patch. inspect_patch draws \
                                 the graph and lists standing errors; inspect_node reads one \
                                 node's params, output health and error. Every edit is undoable \
                                 with undo/redo.",
            }),
        ),
        "tools/list" => ok(id, json!({ "tools": tools() })),
        "tools/call" => ok(id, call_tool(&state, &params)),
        "ping" => ok(id, json!({})),
        method => rpc_error(id, -32601, format!("unknown method `{method}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rendering is what a model actually reads, and each of the three shapes has a caller: the
    /// inspect ops answer `{text}`, `add_node` answers a bare uid, and the rest answer an object.
    #[test]
    fn a_tool_answer_is_rendered_as_the_shape_it_came_in() {
        assert_eq!(render(&json!("a1b2c3")), "a1b2c3");
        assert_eq!(render(&json!({ "text": "flowchart TD" })), "flowchart TD");
        // A one-key object that is NOT `text` stays JSON — otherwise `{ok: true}` would render as
        // nothing a model could tell from an empty answer.
        assert_eq!(render(&json!({ "ok": true })), "{\n  \"ok\": true\n}");
        assert!(render(&json!({ "text": "x", "more": 1 })).contains("\"more\""));
    }

    /// A list type is the item type in an array — the shape a model has to produce for
    /// `group_nodes`'s members or `page_resize_split`'s fractions.
    #[test]
    fn a_list_argument_advertises_its_item_type() {
        assert_eq!(json_schema("uid[]"), json!({ "type": "array", "items": { "type": "string" } }));
        assert_eq!(
            json_schema("float[]"),
            json!({ "type": "array", "items": { "type": "number" } })
        );
        // `json` is deliberately unconstrained: it is whatever the op round-trips.
        assert_eq!(json_schema("json"), json!({}));
    }
}
