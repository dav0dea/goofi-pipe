//! The MCP endpoint: every `Surface::Mcp` registry row becomes one tool, over one JSON object per
//! POST — no session state, no SSE.
//!
//! A tool call is a SYNCHRONOUS [`crate::dispatch`] inside an async task, so nothing awaits while
//! the graph lock is held. And a refused tool name comes back as an `isError` result where the spec
//! says `-32602`, because only the `isError` shape reaches the model that can correct the call.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::ops::{self, Surface};
use crate::AppState;

/// The undo scope every central MCP call runs in: the transport is stateless, so agents share one
/// stack, which is still isolated from every human tab's.
const AGENT_SESSION: &str = "mcp";

/// The revision to claim when a client names none.
const DEFAULT_PROTOCOL: &str = "2025-06-18";

/// The newest revision this server implements, and what an unsupported ask is answered with.
const LATEST_PROTOCOL: &str = "2025-11-25";

/// Every revision this server actually speaks; `2026-07-28` is absent because `resultType`,
/// `ttlMs`/`cacheScope` and `server/discover` are not implemented here.
const SUPPORTED_PROTOCOLS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18", LATEST_PROTOCOL];

/// One registry argument type as JSON Schema; `uid` and `string` share the default arm.
fn json_schema(ty: &str) -> Value {
    if let Some(item) = ty.strip_suffix("[]") {
        return json!({ "type": "array", "items": json_schema(item) });
    }
    match ty {
        "float" => json!({ "type": "number" }),
        "int" => json!({ "type": "integer" }),
        "bool" => json!({ "type": "boolean" }),
        "float2" => {
            json!({ "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 })
        }
        // An empty schema is how JSON Schema spells "anything".
        "json" => json!({}),
        // Advertised as the SET it may take, so a model reads the choices instead of guessing.
        "panel_type" => json!({ "type": "string", "enum": crate::vocab::panel_type_ids() }),
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
                "description": format!("{}\n\nReturns: {}", op.doc(), op.result),
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        })
        .collect()
}

/// A tool's answer as the text a model reads: prose and bare strings verbatim, everything else
/// pretty-printed.
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

/// A `CallToolResult`: a failed op is `isError` inside a successful JSON-RPC reply.
fn tool_result(text: String, is_error: bool) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

/// Run one tool by handing the registry op straight to the control-plane dispatcher.
fn call_tool(state: &AppState, session: &str, params: &Value) -> Value {
    let name = params.get("name").and_then(|v| v.as_str()).unwrap_or_default();
    let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
    // The registry is the gate, checked BEFORE dispatch: a ControlOnly op has a live arm and would
    // run perfectly well if asked for by name.
    if !ops::find(name).is_some_and(|op| op.surface == Surface::Mcp) {
        return tool_result(format!("unknown tool `{name}`"), true);
    }
    match state.call(name, arguments, session) {
        Ok(result) => tool_result(render(&result), false),
        Err(e) => tool_result(e, true),
    }
}

fn ok(id: Value, result: Value) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })).into_response()
}

fn rpc_error(id: Value, code: i64, message: String) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }))
        .into_response()
}

/// The central MCP endpoint — the address an external agent connects to. Registered with `post`,
/// so axum answers the retired GET stream and DELETE teardown with the 405 the spec asks for.
pub async fn endpoint(State(state): State<AppState>, body: String) -> Response {
    serve(&state, AGENT_SESSION, None, &body).await
}

/// The address `spawn_harness` minted for ONE harness. Identity is the route itself, so there is
/// nothing to spoof and nothing to validate, and the undo session follows the address.
pub async fn instance_endpoint(
    axum::extract::Path(id): axum::extract::Path<String>,
    State(state): State<AppState>,
    body: String,
) -> Response {
    let gone = !state.harnesses.serves_mcp(&id);
    serve(&state, &id, gone.then_some(id.as_str()), &body).await
}

/// One JSON-RPC request, in the undo session the address names. `gone` names an instance whose
/// address has been dropped, and is answered rather than 404'd so a model can read the refusal.
async fn serve(state: &AppState, session: &str, gone: Option<&str>, body: &str) -> Response {
    let req: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return rpc_error(Value::Null, -32700, format!("parse error: {e}")),
    };
    // A batch is a JSON ARRAY with no top-level `id`, so without this it falls into the
    // notification branch below and every request in it goes unanswered.
    let Some(req) = req.as_object() else {
        return rpc_error(Value::Null, -32600, "invalid request: expected one JSON-RPC object".into());
    };
    // A notification carries no id and therefore has no reply: 202 with no body.
    let Some(id) = req.get("id").filter(|v| !v.is_null()).cloned() else {
        return StatusCode::ACCEPTED.into_response();
    };
    let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
    let method = req.get("method").and_then(|v| v.as_str()).unwrap_or_default();
    if let Some(instance) = gone {
        let why = format!(
            "harness instance `{instance}` has been stopped, so this address no longer serves \
             goofi's tools. The patch itself is unchanged and still reachable at /mcp."
        );
        // A call is refused as a tool ERROR, the only shape the model reads; anything else, which
        // no model waits on, is a plain JSON-RPC error.
        return match method {
            "tools/call" => ok(id, tool_result(why, true)),
            _ => rpc_error(id, -32001, why),
        };
    }
    match method {
        // Answered for a legacy client, never required of a modern one.
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
                // No `instructions`: the orientation is `AGENTS.md` in the harness's cwd.
            }),
        ),
        "tools/list" => ok(id, json!({ "tools": tools() })),
        "tools/call" => ok(id, call_tool(state, session, &params)),
        "ping" => ok(id, json!({})),
        method => rpc_error(id, -32601, format!("unknown method `{method}`")),
    }
}
