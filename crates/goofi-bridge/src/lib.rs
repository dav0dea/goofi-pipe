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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tower_http::services::{ServeDir, ServeFile};

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

/// Spawn the background tick loop. It paces itself to the graph's fastest node via
/// [`Graph::next_run_delay`] — a producer with `max_frequency <= 0` runs as fast as
/// possible, a capped producer sleeps its remaining period, and an idle graph falls
/// back to `IDLE_POLL` so control-plane edits are picked up promptly. There is NO fixed
/// rate ceiling — `max_frequency` is the only cap (0 = unbounded).
///
/// `LOCK_CEDE` is a sub-millisecond floor applied only to the run-now (unbounded) case:
/// the tick holds the single shared graph mutex, so a truly flat-out spin would starve
/// the /control and /data planes (which lock the same mutex). It is a lock-fairness
/// cede for today's single-mutex architecture (~10 kHz, 166× the old 60 Hz ceiling),
/// NOT a rate policy; genuinely unbounded ticking wants the data plane decoupled from
/// the graph lock (future work).
pub fn spawn_tick(graph: Arc<Mutex<Graph>>) {
    std::thread::spawn(move || {
        const IDLE_POLL: Duration = Duration::from_millis(50);
        const LOCK_CEDE: Duration = Duration::from_micros(100);
        loop {
            let delay = {
                let mut g = graph.lock().unwrap();
                g.tick();
                g.next_run_delay(std::time::Instant::now()).unwrap_or(IDLE_POLL)
            };
            // Clamp to [LOCK_CEDE, IDLE_POLL]: never spin the lock flat-out, never sleep
            // so long that graph edits lag.
            std::thread::sleep(delay.clamp(LOCK_CEDE, IDLE_POLL));
        }
    });
}

/// Given each node's current error and the last-broadcast errors, return the uids whose
/// error state changed (appeared, cleared, or message changed) and update `last`. A node
/// first seen HEALTHY is not a change (so startup doesn't push a `state_update` for every
/// node); removed nodes are forgotten so a re-created uid re-broadcasts fresh.
fn error_transitions(
    current: &[(String, Option<String>)],
    last: &mut HashMap<String, Option<String>>,
) -> Vec<String> {
    let seen: HashSet<&String> = current.iter().map(|(u, _)| u).collect();
    let mut changed = Vec::new();
    for (uid, err) in current {
        let is_changed = match last.get(uid) {
            Some(prev) => prev != err,
            None => err.is_some(),
        };
        if is_changed {
            changed.push(uid.clone());
        }
        last.insert(uid.clone(), err.clone());
    }
    last.retain(|k, _| seen.contains(k));
    changed
}

/// Broadcast each node's measured update frequency (`node_stats`) at `hz`, and push an
/// `error` event whenever a node's error state changes. The tick loop emits nothing, so
/// without this a RUNTIME error that appears mid-run (an expression that compiles but
/// fails on later data, a process error) would not turn the node border red until an
/// unrelated RPC or a reconnect. De-duped: one push per transition, not per tick.
///
/// The transition push is the identity-only `error` event (node + error), NOT a
/// full-params `state_update`: this async 2 Hz snapshot must never carry params, or a
/// stale snapshot could arrive after — and clobber — a concurrent `update_param` edit on
/// the same node (both ride the one broadcast channel, and the frontend replaces params
/// wholesale). The per-param expression-error field refreshes on the next RPC; the node
/// border + console update live here.
pub fn spawn_stats(graph: Arc<Mutex<Graph>>, events: broadcast::Sender<String>, hz: u64) {
    std::thread::spawn(move || {
        let period = Duration::from_secs_f64(1.0 / hz as f64);
        let mut last_errors: HashMap<String, Option<String>> = HashMap::new();
        loop {
            std::thread::sleep(period);
            let (rates, errs) = {
                let g = graph.lock().unwrap();
                let mut rates: Vec<(String, f64)> = Vec::new();
                let mut errs: Vec<(String, Option<String>)> = Vec::new();
                for u in g.node_uids() {
                    let hex = u.to_hex();
                    if let Some(f) = g.node_ufreq(u) {
                        rates.push((hex.clone(), f));
                    }
                    errs.push((hex, g.last_error(u).map(str::to_string)));
                }
                (rates, errs)
            };
            // Diff + build payloads after releasing the lock (both inputs are owned).
            let changed = error_transitions(&errs, &mut last_errors);
            for (node, ufreq) in rates {
                // Only send what we actually measure — no fabricated process-time /
                // tick-count placeholders (the frontend treats those as optional).
                let ev = json!({
                    "event": "node_stats",
                    "payload": { "node": node, "stats": { "updates_per_second": ufreq } }
                });
                let _ = events.send(ev.to_string());
            }
            for hex in changed {
                let err = errs.iter().find(|(h, _)| *h == hex).and_then(|(_, e)| e.clone());
                let _ = events.send(event("error", json!({ "node": hex, "error": err })));
            }
        }
    });
}

/// The full router, optionally serving the built SPA (SPA-fallback to index.html)
/// for any non-API path.
pub fn app(state: AppState, static_dir: Option<PathBuf>) -> Router {
    let base = router(state);
    match static_dir {
        Some(dir) => {
            let index = dir.join("index.html");
            base.fallback_service(ServeDir::new(&dir).not_found_service(ServeFile::new(index)))
        }
        None => base,
    }
}

/// Resolve the built SPA directory: `$GOOFI_FRONTEND_BUILD` or `./frontend/build`.
/// Returns a canonical absolute path so static + SPA-fallback serving is
/// independent of the process working directory.
pub fn resolve_frontend_dir() -> Option<PathBuf> {
    let candidate = match std::env::var("GOOFI_FRONTEND_BUILD") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from("frontend/build"),
    };
    if candidate.is_dir() {
        Some(std::fs::canonicalize(&candidate).unwrap_or(candidate))
    } else {
        None
    }
}

/// Bind and serve (used by the CLI). Ticks adaptively (paced by node `max_frequency`,
/// no fixed ceiling); serves the SPA if found.
pub async fn serve(bind: &str, port: u16, state: AppState) -> std::io::Result<()> {
    spawn_tick(state.graph.clone());
    spawn_stats(state.graph.clone(), state.events.clone(), 2); // node-header update rate

    let listener = tokio::net::TcpListener::bind((bind, port)).await?;
    serve_app(listener, state, resolve_frontend_dir()).await
}

/// Serve on an already-bound listener, API only (used by tests for an ephemeral port).
pub async fn serve_listener(
    listener: tokio::net::TcpListener,
    state: AppState,
) -> std::io::Result<()> {
    axum::serve(listener, router(state)).await
}

/// Serve on an already-bound listener with optional static SPA serving.
pub async fn serve_app(
    listener: tokio::net::TcpListener,
    state: AppState,
    static_dir: Option<PathBuf>,
) -> std::io::Result<()> {
    axum::serve(listener, app(state, static_dir)).await
}

async fn healthz() -> Json<Value> {
    Json(json!({ "ok": true }))
}

/// Native node type names visible in the catalog (diagnostic; ensures linkage).
pub fn catalog_type_names() -> Vec<String> {
    let _ = goofi_nodes::native_node_count();
    goofi_node::catalog()
        .map(|m| m.type_name.to_string())
        .collect()
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

/// Coerce a JSON number to i64, ROUNDING a fractional value to nearest. serde's
/// `as_i64()` returns `None` for a float like `5.5`, which the old `unwrap_or(0)`
/// silently snapped to 0 — so editing an Int param to a fractional value reset it.
fn json_to_i64(v: &Value) -> i64 {
    v.as_i64()
        .or_else(|| v.as_f64().map(|f| f.round() as i64))
        .unwrap_or(0)
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
            value: json_to_i64(v),
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
            "list_nodes" => Ok(json!({ "types": schemas::catalog_types(&g) })),
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
                        "params": schemas::describe_node_params(&g, uid),
                        "output_subscribers": {},
                        "stage": "ready",
                        "error": g.last_error(uid),
                        "log_endpoint": Value::Null,
                        "refreshed_params": [],
                    }),
                ));
                Ok(json!({ "ok": true }))
            }
            "set_expression" => {
                let uid = parse_uid(&payload, "node")?;
                let group = payload.get("group").and_then(|v| v.as_str()).ok_or("missing group")?;
                let name = payload.get("name").and_then(|v| v.as_str()).ok_or("missing name")?;
                let expression = payload.get("expression").and_then(|v| v.as_str()).unwrap_or("");
                let enabled = payload
                    .get("expression_enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let triggers = payload
                    .get("expression_triggers_process")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                g.set_expression(uid, group, name, expression, enabled, triggers)?;
                events.push(event(
                    "state_update",
                    json!({
                        "node": uid.to_hex(),
                        "params": schemas::describe_node_params(&g, uid),
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
            "serialize" => Ok(json!({ "yaml": g.serialize() })),
            "save" => {
                let yaml = g.serialize();
                let path = payload.get("path").and_then(|v| v.as_str());
                if let Some(p) = path {
                    std::fs::write(p, &yaml).map_err(|e| format!("save failed: {e}"))?;
                }
                Ok(json!({ "path": path, "yaml": yaml }))
            }
            "load_text" => {
                let content = payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or("load_text: missing content")?;
                g.load_doc(content)?;
                events.push(event(
                    "graph_replaced",
                    schemas::snapshot(&g, &state.instance_id, false),
                ));
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
                // Hold the graph lock only for the cheap Arc clone; encode the
                // (immutable) frame AFTER releasing it, so a viewer copying a large
                // kHz/HD body never serializes against the scheduler tick or the
                // other viewers.
                let d = {
                    let g = state.graph.lock().unwrap();
                    g.latest_frame(uid, &slot)
                };
                if let Some(bytes) = d.map(|d| goofi_codec::encode(&d)) {
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

#[cfg(test)]
mod param_coerce_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn int_param_rounds_fractional_instead_of_zeroing() {
        let p = Param::int(3, 0, 100);
        // The bug: a fractional value snapped the Int param to 0. Now it rounds.
        assert_eq!(param_from_json(&p, &json!(5.5)).as_i64(), Some(6));
        assert_eq!(param_from_json(&p, &json!(5.4)).as_i64(), Some(5));
        assert_eq!(param_from_json(&p, &json!(7)).as_i64(), Some(7), "plain int unaffected");
    }

    fn ev(uid: &str, err: Option<&str>) -> (String, Option<String>) {
        (uid.to_string(), err.map(str::to_string))
    }

    #[test]
    fn error_transitions_fires_only_on_change() {
        let mut last = HashMap::new();
        // First sight: a healthy node is NOT a transition; an errored one IS.
        let t = error_transitions(&[ev("a", None), ev("b", Some("boom"))], &mut last);
        assert_eq!(t, vec!["b".to_string()], "only the newly-errored node");
        // Steady state: no repeat pushes.
        assert!(error_transitions(&[ev("a", None), ev("b", Some("boom"))], &mut last).is_empty());
        // Recovery of b + a newly errors -> both transition.
        let mut t2 = error_transitions(&[ev("a", Some("x")), ev("b", None)], &mut last);
        t2.sort();
        assert_eq!(t2, vec!["a".to_string(), "b".to_string()]);
        // A changed message is a transition.
        assert_eq!(error_transitions(&[ev("a", Some("y")), ev("b", None)], &mut last), vec!["a".to_string()]);
    }

    #[test]
    fn error_transitions_forgets_removed_nodes() {
        let mut last = HashMap::new();
        error_transitions(&[ev("a", Some("boom"))], &mut last);
        assert!(last.contains_key("a"));
        // 'a' gone next poll -> forgotten, so a re-created 'a' re-broadcasts fresh.
        error_transitions(&[ev("b", None)], &mut last);
        assert!(!last.contains_key("a"), "removed node forgotten");
        assert_eq!(error_transitions(&[ev("a", Some("boom"))], &mut last), vec!["a".to_string()]);
    }
}
