//! The MCP endpoint: ONE tool, `goofi_exec`, whose input is a list of command lines — the same
//! lines the CLI speaks, parsed by the same [`crate::phrase`] layer, so the two surfaces cannot
//! drift. One JSON object per POST — no session state, no SSE.
//!
//! A tool call is a SYNCHRONOUS [`crate::AppState::call`] inside an async task, so nothing awaits
//! while the graph lock is held. And a refused call comes back as an `isError` result where the
//! spec says `-32602`, because only the `isError` shape reaches the model that can correct it.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::{phrase, AppState};

/// The undo ACTOR an MCP call runs as when it names none: the transport is stateless, so such
/// agents share one stack, which is still isolated from every human tab's.
const AGENT_ACTOR: &str = "mcp";

/// The revision to claim when a client names none.
const DEFAULT_PROTOCOL: &str = "2025-06-18";

/// The newest revision this server implements, and what an unsupported ask is answered with.
const LATEST_PROTOCOL: &str = "2025-11-25";

/// Every revision this server actually speaks; `2026-07-28` is absent because `resultType`,
/// `ttlMs`/`cacheScope` and `server/discover` are not implemented here.
const SUPPORTED_PROTOCOLS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18", LATEST_PROTOCOL];

/// What the one tool teaches a model BEFORE its first call: the line grammar, the index, and the
/// batch rule.
const DESCRIPTION: &str = "\
Drive goofi with command lines. Each entry in `commands` is one op: `<op> [--arg value …]`, with \
bash's own quoting rules. Call `op list` first — it answers every op with its arguments, its \
result and its kind. A bool arg is `--x` or `--no-x`; a list arg repeats its flag; a `json` arg \
takes one JSON string.\n\n\
The reply is every command's result, in order, as one JSON list. SEVERAL commands execute as \
one batch and ONE undo step: a step is a read or an undoable write — an effect runs alone — and \
a refused step takes the whole batch back. To wire nodes made in the same batch, choose their \
uids yourself with `node add --member_uid`. `actor` names your undo stack; calls that share it \
share undo.";

/// The tool list: the one tool.
fn tools() -> Vec<Value> {
    vec![json!({
        "name": "goofi_exec",
        "description": DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "One command per entry: `<op> [--arg value …]`.",
                },
                "actor": {
                    "type": "string",
                    "description": "The undo stack these commands land in (default \"mcp\").",
                },
            },
            "required": ["commands"],
        },
    })]
}

/// A `CallToolResult`: a failed op is `isError` inside a successful JSON-RPC reply.
fn tool_result(text: String, is_error: bool) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

/// Run the one tool: parse every line first, then execute — one command directly, several as one
/// compound, so a batch is one undo step and a refused step takes the others back.
fn call_tool(state: &AppState, params: &Value) -> Value {
    let actor = params
        .get("arguments")
        .and_then(|a| a.get("actor"))
        .and_then(|v| v.as_str())
        .unwrap_or(AGENT_ACTOR);
    let name = params.get("name").and_then(|v| v.as_str()).unwrap_or_default();
    if name != "goofi_exec" {
        return tool_result(format!("unknown tool `{name}` — this server has one: goofi_exec"), true);
    }
    let Some(lines) = params
        .get("arguments")
        .and_then(|a| a.get("commands"))
        .and_then(|c| c.as_array())
        .map(|c| c.iter().map(|l| l.as_str().unwrap_or_default().to_string()).collect::<Vec<_>>())
        .filter(|c: &Vec<String>| !c.is_empty())
    else {
        return tool_result("goofi_exec: `commands` is a non-empty list of command lines".into(), true);
    };
    match phrase::exec_lines(state, &lines, actor) {
        // ONE shape whatever the count — the list of results, in order — so no data is
        // reachable at one arity and paraphrased at another.
        Ok(results) => {
            let list = Value::Array(results);
            tool_result(
                serde_json::to_string_pretty(&list).unwrap_or_else(|_| list.to_string()),
                false,
            )
        }
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
    serve(&state, &body).await
}

/// One JSON-RPC request.
async fn serve(state: &AppState, body: &str) -> Response {
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
        "tools/call" => {
            // Off the async workers, as `/exec` is: a batch can hold the graph lock for seconds.
            let state = state.clone();
            let ran = tokio::task::spawn_blocking(move || call_tool(&state, &params)).await;
            ok(id, ran.unwrap_or_else(|e| tool_result(format!("the exec task died: {e}"), true)))
        }
        "ping" => ok(id, json!({})),
        method => rpc_error(id, -32601, format!("unknown method `{method}`")),
    }
}
