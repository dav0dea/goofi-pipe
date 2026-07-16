//! goofi-bridge — the axum HTTP/WebSocket server that exposes the engine to the
//! browser: the `/control` JSON RPC + broadcast-event plane and the
//! `/data/<node>/<slot>/<kind>` binary GOOF plane. Event-sourced: RPCs return
//! thin acks; real state changes arrive as broadcast events the client applies.
//!
//! M1 scope: hello/snapshot, list_nodes, add_node/remove_node/add_link/
//! remove_link/update_param/set_node_pos/rename_node, and full-resolution GOOF
//! frame streaming. Inbound ViewSpec negotiation, log SSE, sub-patches, and
//! static SPA serving arrive in later milestones.

mod schemas;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::response::Response;
use axum::routing::{any, get};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use goofi_core::Param;
use goofi_engine::{Graph, Uid};
use serde_json::{json, Value};
use tokio::sync::broadcast;

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<Mutex<Graph>>,
    pub events: broadcast::Sender<String>,
    pub instance_id: Arc<str>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> AppState {
        let (events, _) = broadcast::channel(256);
        let iid = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        AppState {
            graph: Arc::new(Mutex::new(Graph::new())),
            events,
            instance_id: Arc::from(format!("{iid:x}").as_str()),
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/control", any(control_ws))
        .route("/data/{node}/{slot}/{kind}", any(data_ws))
        .route("/api/healthz", get(healthz))
        .with_state(state)
}

/// Spawn the background tick loop at `hz` (grows into the real scheduler in M2).
pub fn spawn_tick(graph: Arc<Mutex<Graph>>, hz: u64) {
    std::thread::spawn(move || {
        let period = Duration::from_secs_f64(1.0 / hz as f64);
        loop {
            {
                let mut g = graph.lock().unwrap();
                g.tick();
            }
            std::thread::sleep(period);
        }
    });
}

/// Bind and serve (used by the CLI). Ticks at 60 Hz.
pub async fn serve(bind: &str, port: u16, state: AppState) -> std::io::Result<()> {
    spawn_tick(state.graph.clone(), 60);
    let listener = tokio::net::TcpListener::bind((bind, port)).await?;
    serve_listener(listener, state).await
}

/// Serve on an already-bound listener (used by tests to grab an ephemeral port).
pub async fn serve_listener(
    listener: tokio::net::TcpListener,
    state: AppState,
) -> std::io::Result<()> {
    axum::serve(listener, router(state)).await
}

async fn healthz() -> Json<Value> {
    Json(json!({ "ok": true }))
}

// ---------------------------------------------------------------------------
// Control plane
// ---------------------------------------------------------------------------

async fn control_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_control(socket, state))
}

async fn handle_control(socket: WebSocket, state: AppState) {
    let (mut tx, mut rx) = socket.split();

    let hello = {
        let g = state.graph.lock().unwrap();
        event("hello", schemas::snapshot(&g, &state.instance_id, true))
    };
    if tx.send(Message::Text(hello.into())).await.is_err() {
        return;
    }

    let mut events = state.events.subscribe();
    loop {
        tokio::select! {
            incoming = rx.next() => match incoming {
                Some(Ok(Message::Text(t))) => {
                    if let Some(reply) = dispatch(&state, t.as_str()) {
                        if tx.send(Message::Text(reply.into())).await.is_err() {
                            break;
                        }
                    }
                }
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {}
            },
            broadcasted = events.recv() => match broadcasted {
                Ok(e) => {
                    if tx.send(Message::Text(e.into())).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            },
        }
    }
}

fn event(name: &str, payload: Value) -> String {
    json!({ "event": name, "payload": payload }).to_string()
}

fn parse_uid(payload: &Value, key: &str) -> Result<Uid, String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .and_then(Uid::from_hex)
        .ok_or_else(|| format!("missing/invalid uid `{key}`"))
}

fn parse_pos(v: &Value) -> Option<[f64; 2]> {
    let a = v.as_array()?;
    if a.len() != 2 {
        return None;
    }
    Some([a[0].as_f64()?, a[1].as_f64()?])
}

fn parse_link(p: &Value) -> Result<(Uid, String, Uid, String), String> {
    let node_out = parse_uid(p, "node_out")?;
    let node_in = parse_uid(p, "node_in")?;
    let slot_out = p
        .get("slot_out")
        .and_then(|v| v.as_str())
        .ok_or("missing slot_out")?
        .to_string();
    let slot_in = p
        .get("slot_in")
        .and_then(|v| v.as_str())
        .ok_or("missing slot_in")?
        .to_string();
    Ok((node_out, slot_out, node_in, slot_in))
}

/// Build a `Param` from a JSON value, keeping the existing param's type + bounds.
fn param_from_json(existing: &Param, v: &Value) -> Param {
    match existing {
        Param::Float { vmin, vmax, .. } => Param::Float {
            value: v.as_f64().unwrap_or(0.0),
            vmin: *vmin,
            vmax: *vmax,
        },
        Param::Int { vmin, vmax, .. } => Param::Int {
            value: v.as_i64().unwrap_or(0),
            vmin: *vmin,
            vmax: *vmax,
        },
        Param::Bool { .. } => Param::Bool {
            value: v.as_bool().unwrap_or(false),
        },
        Param::Trigger { .. } => Param::Trigger {
            fired: v.as_bool().unwrap_or(false),
        },
        Param::Str {
            options, refresh, ..
        } => Param::Str {
            value: v.as_str().unwrap_or("").to_string(),
            options: options.clone(),
            refresh: *refresh,
        },
    }
}

/// Dispatch one control RPC. Mutates the graph, queues broadcast events, and
/// returns the `{id,result}`/`{id,error}` reply (only when `id` is numeric).
fn dispatch(state: &AppState, text: &str) -> Option<String> {
    let req: Value = serde_json::from_str(text).ok()?;
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let op = req.get("op")?.as_str()?.to_string();
    let payload = req.get("payload").cloned().unwrap_or_else(|| json!({}));

    let mut events: Vec<String> = Vec::new();
    let result: Result<Value, String> = (|| {
        let mut g = state.graph.lock().unwrap();
        match op.as_str() {
            "list_nodes" => Ok(json!({ "types": schemas::catalog_types() })),
            "add_node" => {
                let ty = payload
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or("add_node: missing type")?;
                let uid = g.add_node(ty, None)?;
                if let Some(pos) = payload.get("pos").and_then(parse_pos) {
                    let _ = g.set_node_pos(uid, pos);
                }
                events.push(event("node_added", schemas::node_instance_info(&g, uid)));
                Ok(json!(uid.to_hex()))
            }
            "remove_node" => {
                let uid = parse_uid(&payload, "node")?;
                let name = g.name(uid).unwrap_or("").to_string();
                g.remove_node(uid)?;
                events.push(event(
                    "node_removed",
                    json!({
                        "node": uid.to_hex(),
                        "membership": { "instance": schemas::ROOT_ID, "local_name": name },
                    }),
                ));
                Ok(json!({ "ok": true }))
            }
            "add_link" => {
                let (a, so, b, si) = parse_link(&payload)?;
                g.add_link(a, &so, b, &si)?;
                events.push(event(
                    "link_added",
                    json!({ "node_out": a.to_hex(), "slot_out": so, "node_in": b.to_hex(), "slot_in": si }),
                ));
                Ok(json!({ "ok": true }))
            }
            "remove_link" => {
                let (a, so, b, si) = parse_link(&payload)?;
                g.remove_link(a, &so, b, &si)?;
                events.push(event(
                    "link_removed",
                    json!({ "node_out": a.to_hex(), "slot_out": so, "node_in": b.to_hex(), "slot_in": si }),
                ));
                Ok(json!({ "ok": true }))
            }
            "update_param" => {
                let uid = parse_uid(&payload, "node")?;
                let group = payload.get("group").and_then(|v| v.as_str()).ok_or("missing group")?;
                let name = payload.get("name").and_then(|v| v.as_str()).ok_or("missing name")?;
                let vjson = payload.get("value").ok_or("missing value")?;
                let existing = g
                    .params(uid)
                    .and_then(|p| goofi_node::param(p, group, name))
                    .cloned()
                    .ok_or("no such param")?;
                let newp = param_from_json(&existing, vjson);
                g.update_param(uid, group, name, newp)?;
                events.push(event(
                    "state_update",
                    json!({
                        "node": uid.to_hex(),
                        "params": schemas::describe_params(g.params(uid).unwrap()),
                        "output_subscribers": {},
                        "stage": "ready",
                        "error": g.last_error(uid),
                        "log_endpoint": Value::Null,
                        "refreshed_params": [],
                    }),
                ));
                Ok(json!({ "ok": true }))
            }
            "set_node_pos" => {
                let uid = parse_uid(&payload, "node")?;
                let pos = payload.get("pos").and_then(parse_pos).ok_or("missing pos")?;
                g.set_node_pos(uid, pos)?;
                events.push(event("node_moved", json!({ "node": uid.to_hex(), "pos": pos })));
                Ok(json!({ "ok": true }))
            }
            "rename_node" => {
                let uid = parse_uid(&payload, "node")?;
                let name = payload.get("name").and_then(|v| v.as_str()).ok_or("missing name")?;
                g.rename_node(uid, name)?;
                events.push(event("node_renamed", json!({ "node": uid.to_hex(), "name": name })));
                Ok(json!({ "ok": true }))
            }
            other => Err(format!("unknown op `{other}`")),
        }
    })();

    for e in events {
        let _ = state.events.send(e);
    }

    match id {
        Value::Number(_) => Some(match result {
            Ok(r) => json!({ "id": id, "result": r }).to_string(),
            Err(e) => json!({ "id": id, "error": e }).to_string(),
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Data plane
// ---------------------------------------------------------------------------

async fn data_ws(
    Path((node, slot, kind)): Path<(String, String, String)>,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_data(socket, state, node, slot, kind))
}

fn close(code: u16, reason: &str) -> Message {
    Message::Close(Some(CloseFrame {
        code,
        reason: reason.into(),
    }))
}

async fn handle_data(socket: WebSocket, state: AppState, node: String, slot: String, _kind: String) {
    let (mut tx, mut rx) = socket.split();

    let uid = match Uid::from_hex(&node) {
        Some(u) => u,
        None => {
            let _ = tx.send(close(4004, "bad node uid")).await;
            return;
        }
    };
    let valid = {
        let g = state.graph.lock().unwrap();
        g.manifest(uid)
            .map(|m| m.outputs.iter().any(|o| o.name == slot))
            .unwrap_or(false)
    };
    if !valid {
        let _ = tx.send(close(4004, "unknown node/slot")).await;
        return;
    }

    let mut ticker = tokio::time::interval(Duration::from_millis(16));
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let frame = {
                    let g = state.graph.lock().unwrap();
                    g.latest_frame(uid, &slot).map(|d| goofi_codec::encode(&d))
                };
                if let Some(bytes) = frame {
                    if tx.send(Message::Binary(bytes.into())).await.is_err() {
                        break;
                    }
                }
            }
            incoming = rx.next() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                // Inbound ViewSpec `{op:"view"}` is ignored in M1 (full-resolution frames).
                _ => {}
            },
        }
    }
}
