//! goofi-bridge — the axum HTTP/WebSocket server that exposes the engine to the
//! browser: the `/control` JSON RPC + broadcast-event plane and the
//! `/data/<node>/<slot>` binary GOOF plane — ONE reduced stream per (node, slot);
//! the viewer kind is NOT in the path (viewers send their ViewSpec inband via
//! `{op:"view"}`). Event-sourced: RPCs return thin acks; real state changes arrive
//! as broadcast events the client applies. The built SPA is served from disk
//! (`frontend/build`, or `GOOFI_FRONTEND_BUILD`) via `ServeDir`.

mod crdt_mirror;
mod reducer;
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
    /// Server-side CRDT mirror of the graph's control state, re-synced after every
    /// successful control op. The shared source of truth clients replicate (Phase 2+).
    pub crdt: Arc<Mutex<goofi_crdt::GraphDoc>>,
    /// Binary sync-update fan-out: each mutation broadcasts the CRDT delta as a framed
    /// [`goofi_crdt::SyncMsg::Update`] to every connected client's replica.
    pub sync_updates: broadcast::Sender<Vec<u8>>,
    /// Ephemeral/awareness fan-out: presence, live-drag values, previews, active viewer specs.
    /// Separate from `sync_updates` because it is fire-and-forget — a lagged ephemeral frame is
    /// simply dropped (never triggers doc recovery). Relayed verbatim; peers self-filter their
    /// own client id.
    pub ephemeral: broadcast::Sender<Vec<u8>>,
    /// The doc's state vector as of the last broadcast delta — the baseline the next delta
    /// is computed against (guarded together with `crdt`: always lock `crdt` first).
    pub last_sync_sv: Arc<Mutex<Vec<u8>>>,
    /// Shared per-slot data reducers (thalamus G1/G2): one reduction per active (node, slot),
    /// fanned out to every viewer, so N tabs on one slot cost one reduce+encode, not N.
    pub reducers: reducer::SlotReducers,
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
        let (sync_updates, _) = broadcast::channel(256);
        let (ephemeral, _) = broadcast::channel(256);
        let crdt = goofi_crdt::GraphDoc::new();
        let last_sync_sv = Arc::new(Mutex::new(crdt.state_vector()));
        let graph = Arc::new(Mutex::new(Graph::new()));
        let reducers = reducer::SlotReducers::new(graph.clone());
        AppState {
            graph,
            events,
            instance_id: Arc::from(format!("{iid:x}").as_str()),
            crdt: Arc::new(Mutex::new(crdt)),
            sync_updates,
            ephemeral,
            last_sync_sv,
            reducers,
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/control", any(control_ws))
        // One stream per (node, slot) — the kind segment is gone; a single reduced stream
        // serves every viewer kind. Each connection sends its viewers' ViewSpecs inband.
        .route("/data/{node}/{slot}", any(data_ws))
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
///
/// KNOWN LIMITATION (architectural, tracked in `docs/analysis/tick-lock-subprocess-stall.md`):
/// `g.tick()` runs every node's `process()` inline while holding this mutex — including a
/// [`goofi_subproc::RemoteNode`], whose `process()` blocks on an iceoryx2 roundtrip. A subprocess
/// node's FIRST tick pays child cold-start (python+numpy+iceoryx2 import, ~1-2 s) and a hung
/// child strands up to `DEFAULT_TIMEOUT` (10 s) + 1 s. For that whole window the graph lock is
/// held, so control dispatch, the reducer's `latest_frame`, and stats all block — the UI freezes.
/// The bound (kill-on-timeout) keeps it finite but a multi-second freeze on a normal action is
/// real. A proper fix is the same lock decoupling as above (release the lock across node
/// `process()` / async node bootstrap staging), not a patch here — a rushed change to this
/// load-bearing path would risk correctness.
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
///
/// It also pushes the live values of expression-driven params as a `param_values` event,
/// so the inspector preview tracks each re-evaluation instead of freezing at the
/// edit-time value. That is safe where a full-params snapshot is not: it carries ONLY the
/// evaluated values (not descriptors) and the frontend applies them surgically to
/// expression-bound params — which are never user-editable literals, so there is no
/// concurrent edit to clobber.
pub fn spawn_stats(graph: Arc<Mutex<Graph>>, events: broadcast::Sender<String>, hz: u64) {
    std::thread::spawn(move || {
        let period = Duration::from_secs_f64(1.0 / hz as f64);
        let mut last_errors: HashMap<String, Option<String>> = HashMap::new();
        loop {
            std::thread::sleep(period);
            let (rates, errs, expr_vals) = {
                let g = graph.lock().unwrap();
                let mut rates: Vec<(String, f64)> = Vec::new();
                let mut errs: Vec<(String, Option<String>)> = Vec::new();
                let mut expr_vals: Vec<(String, Value)> = Vec::new();
                for u in g.node_uids() {
                    let hex = u.to_hex();
                    if let Some(f) = g.node_ufreq(u) {
                        rates.push((hex.clone(), f));
                    }
                    let vals = schemas::expression_value_map(&g, u);
                    if vals.as_object().is_some_and(|o| !o.is_empty()) {
                        expr_vals.push((hex.clone(), vals));
                    }
                    errs.push((hex, g.last_error(u).map(str::to_string)));
                }
                (rates, errs, expr_vals)
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
            for (node, values) in expr_vals {
                let _ = events.send(event("param_values", json!({ "node": node, "values": values })));
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

/// Start the background workers a live server needs: the adaptive tick loop AND the 2 Hz
/// node-stats broadcaster — the latter pushes each node's measured ufreq to the node header
/// and the async error-transition that reddens a node border mid-run. The binary calls this
/// once at startup (alongside its own bind + `serve_app`), so both are wired in one place;
/// `serve_app` itself stays pure, letting a test bind an ephemeral listener without the stats
/// thread when it doesn't need it. (This replaces the old bundled `serve()`, which the CLI
/// couldn't use — it must register evaluators/nodes and print the URL around the bind — so the
/// stats worker rotted into dead code and the header rate silently stopped updating.)
pub fn spawn_workers(state: &AppState) {
    spawn_tick(state.graph.clone());
    spawn_stats(state.graph.clone(), state.events.clone(), 2);
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

    // Subscribe to every broadcast plane BEFORE snapshotting. If we snapshotted first and
    // subscribed second, a mutation landing in that window would be neither in the snapshot nor
    // delivered — the client's mirror would silently desync. Subscribing first can at worst
    // re-deliver an event already reflected in the snapshot, which every apply branch absorbs
    // idempotently (node_added filters its uid; link_added dedups; removes/moves reconcile). The
    // CRDT plane subscribes first for the same reason.
    let mut events = state.events.subscribe();
    let mut sync_updates = state.sync_updates.subscribe();
    let mut ephemeral = state.ephemeral.subscribe();

    let hello = {
        let g = state.graph.lock().unwrap();
        event("hello", schemas::snapshot(&g, &state.instance_id, true))
    };
    if tx.send(Message::Text(hello.into())).await.is_err() {
        return;
    }

    // CRDT sync handshake: advertise the server replica's state vector as a binary frame.
    // The client answers with its own state vector; `on_sync` then ships the diff it lacks.
    {
        let hello_sv = state.crdt.lock().unwrap().sync_hello();
        if tx.send(Message::Binary(hello_sv.into())).await.is_err() {
            return;
        }
    }

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
                Some(Ok(Message::Binary(b))) => {
                    // A CRDT sync frame from the client. A StateVector drives the pairwise
                    // handshake (reply with the diff it lacks); an Update is a client leaf
                    // write — apply it to the graph and reconcile/broadcast to all clients.
                    match goofi_crdt::SyncMsg::decode(&b) {
                        Some(msg @ goofi_crdt::SyncMsg::StateVector(_)) => {
                            let replies = state.crdt.lock().unwrap().on_sync(msg);
                            for r in replies {
                                if tx.send(Message::Binary(r.encode().into())).await.is_err() {
                                    return;
                                }
                            }
                        }
                        Some(goofi_crdt::SyncMsg::Update(u)) => apply_client_write(&state, &u),
                        Some(eph @ goofi_crdt::SyncMsg::Ephemeral(_)) => {
                            // Presence/preview: relay verbatim to every client (peers self-filter
                            // their own id). Fire-and-forget — no doc write, no recovery.
                            let _ = state.ephemeral.send(eph.encode());
                        }
                        None => {}
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
                // A slow client lagged past the shared 256-slot channel and dropped events the
                // ring already evicted. A dropped structural event (node/link add/remove) would
                // permanently desync its JSON mirror — there is no gap detection. Recover exactly
                // as the sync_updates plane does: re-send a full `hello` snapshot (the frontend
                // applies it as a full reset; idempotent apply branches absorb any still-buffered
                // events delivered after it).
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let hello = {
                        let g = state.graph.lock().unwrap();
                        event("hello", schemas::snapshot(&g, &state.instance_id, true))
                    };
                    if tx.send(Message::Text(hello.into())).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
            sync = sync_updates.recv() => match sync {
                Ok(update) => {
                    if tx.send(Message::Binary(update.into())).await.is_err() {
                        break;
                    }
                }
                // A lagged client missed one or more deltas the broadcast channel already
                // dropped. Send the FULL current state (idempotent, resolves any gap incl.
                // pending updates) — NOT the server's state vector, which a reader answers
                // with an empty diff and so never actually catches up (permanent desync).
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let full = state.crdt.lock().unwrap().full_state_frame();
                    if tx.send(Message::Binary(full.into())).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
            eph = ephemeral.recv() => match eph {
                Ok(frame) => {
                    if tx.send(Message::Binary(frame.into())).await.is_err() {
                        break;
                    }
                }
                // Fire-and-forget: a lagged ephemeral frame is simply skipped (presence/preview
                // is latest-wins, never recovered).
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            },
        }
    }
}

fn event(name: &str, payload: Value) -> String {
    json!({ "event": name, "payload": payload }).to_string()
}

/// A per-node `state_update` event carrying a node's current params + error. Emitted for every
/// peer a §4.5 shared-member edit touches (param value, position, expression), so any observer
/// reconciles each mirrored sibling.
fn param_state_update(g: &Graph, peer: Uid) -> String {
    event(
        "state_update",
        json!({
            "node": peer.to_hex(),
            "params": schemas::describe_node_params(g, peer),
            "output_subscribers": {},
            "stage": "ready",
            "error": g.last_error(peer),
            "log_endpoint": Value::Null,
            "refreshed_params": [],
        }),
    )
}

fn parse_uid(payload: &Value, key: &str) -> Result<Uid, String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .and_then(Uid::from_hex)
        .ok_or_else(|| format!("missing/invalid uid `{key}`"))
}

fn parse_slot_type(s: &str) -> Option<goofi_core::SlotType> {
    match s {
        "ARRAY" => Some(goofi_core::SlotType::Array),
        "STRING" => Some(goofi_core::SlotType::String),
        "TABLE" => Some(goofi_core::SlotType::Table),
        _ => None,
    }
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

/// Translate a link endpoint that names a sub-patch instance's boundary port into the flat
/// inner leaf it resolves to. A top-level node wired to `inst::bnd` becomes a real leaf→leaf
/// link — the boundary is a naming indirection resolved here, so the runtime/persisted link is
/// always flat. A plain `(node, slot)` passes through unchanged.
fn resolve_link_endpoint(g: &goofi_engine::Graph, uid: Uid, slot: &str) -> (Uid, String) {
    if g.instance(uid).is_some() {
        if let Some(leaf) = g.resolve_boundary(uid, slot) {
            return leaf;
        }
    }
    (uid, slot.to_string())
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
                // Delete-on-an-instance (and the inverse of duplicate_shared) routes here with an
                // instance uid — tear down the whole subtree and broadcast the new snapshot.
                if g.instance(uid).is_some() {
                    g.remove_instance(uid)?;
                    events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                    Ok(json!({ "ok": true }))
                } else {
                    // Capture the node's REAL scope before removal — the frontend drops the
                    // member from this scope's index, so hardcoding ROOT would leave a member of
                    // a sub-patch scope stale (inflated count badge + latent index entry).
                    let membership = schemas::membership(&g, uid);
                    g.remove_node(uid)?;
                    events.push(event(
                        "node_removed",
                        json!({ "node": uid.to_hex(), "membership": membership }),
                    ));
                    Ok(json!({ "ok": true }))
                }
            }
            "add_link" => {
                let (a, so, b, si) = parse_link(&payload)?;
                // Resolve either endpoint through a sub-patch boundary → flat leaf→leaf.
                let (a, so) = resolve_link_endpoint(&g, a, &so);
                let (b, si) = resolve_link_endpoint(&g, b, &si);
                g.add_link(a, &so, b, &si)?;
                events.push(event(
                    "link_added",
                    json!({ "node_out": a.to_hex(), "slot_out": so, "node_in": b.to_hex(), "slot_in": si }),
                ));
                Ok(json!({ "ok": true }))
            }
            "remove_link" => {
                let (a, so, b, si) = parse_link(&payload)?;
                let (a, so) = resolve_link_endpoint(&g, a, &so);
                let (b, si) = resolve_link_endpoint(&g, b, &si);
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
                // Re-project to every shared sibling (§4.5): a shared member's edit hits all its
                // instances. A ROOT / unique-member edit updates only itself.
                let updated = g.update_member_param(uid, group, name, newp)?;
                for peer in updated {
                    events.push(param_state_update(&g, peer));
                }
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
                // Re-project to every shared sibling (§4.5): a shared member's expression edit
                // hits all its instances. A ROOT / unique-member edit updates only itself.
                let updated = g.set_member_expression(uid, group, name, expression, enabled, triggers)?;
                for peer in updated {
                    events.push(param_state_update(&g, peer));
                }
                Ok(json!({ "ok": true }))
            }
            // `set_node_pos` / `set_node_viewers` retired (Phase 3): node/instance position and the
            // per-slot viewer blob are merge-safe leaves the client writes directly to the doc
            // (`apply_client_write` → `set_member_pos` / `set_node_viewers`).
            "rename_node" => {
                let uid = parse_uid(&payload, "node")?;
                let name = payload.get("name").and_then(|v| v.as_str()).ok_or("missing name")?;
                let referrers = g.rename_node(uid, name)?;
                events.push(event("node_renamed", json!({ "node": uid.to_hex(), "name": name })));
                // Any expression that referenced the old name was rewritten to nd('new'):
                // push each referrer's fresh params so its inspector reflects the rewrite.
                for r in referrers {
                    events.push(param_state_update(&g, r));
                }
                Ok(json!({ "ok": true }))
            }
            "group_nodes" => {
                let members = payload
                    .get("members")
                    .and_then(|v| v.as_array())
                    .ok_or("group_nodes: missing members")?;
                let uids: Vec<Uid> = members.iter().filter_map(|m| m.as_str().and_then(Uid::from_hex)).collect();
                if uids.len() != members.len() {
                    return Err("group_nodes: malformed member uid".into());
                }
                let pos = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
                let inst = g.group_nodes(&uids, pos)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "inst_id": inst.to_hex() }))
            }
            "expand_instance" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let restored = g.expand_instance(inst)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "restored": restored.iter().map(|u| u.to_hex()).collect::<Vec<_>>() }))
            }
            "add_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let dir = match payload.get("dir").and_then(|v| v.as_str()) {
                    Some("in") => goofi_engine::subpatch::Dir::In,
                    Some("out") => goofi_engine::subpatch::Dir::Out,
                    _ => return Err("add_boundary: dir must be \"in\" or \"out\"".into()),
                };
                let dtype = parse_slot_type(payload.get("dtype").and_then(|v| v.as_str()).unwrap_or("ARRAY"))
                    .ok_or("add_boundary: bad dtype")?;
                let pos = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
                let bnd = g.add_boundary(inst, dir, dtype, pos)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "bnd_id": bnd }))
            }
            "wire_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = payload.get("bnd_id").and_then(|v| v.as_str()).ok_or("wire_boundary: missing bnd_id")?;
                let inner = parse_uid(&payload, "inner_node")?;
                let slot = payload.get("inner_slot").and_then(|v| v.as_str()).ok_or("wire_boundary: missing inner_slot")?;
                g.wire_boundary(inst, bnd, inner, slot)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "ok": true }))
            }
            "remove_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = payload.get("bnd_id").and_then(|v| v.as_str()).ok_or("remove_boundary: missing bnd_id")?;
                g.remove_boundary(inst, bnd)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "ok": true }))
            }
            "rename_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = payload.get("bnd_id").and_then(|v| v.as_str()).ok_or("rename_boundary: missing bnd_id")?;
                let name = payload.get("name").and_then(|v| v.as_str()).ok_or("rename_boundary: missing name")?;
                g.rename_boundary(inst, bnd, name)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "ok": true }))
            }
            "set_boundary_pos" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = payload.get("bnd_id").and_then(|v| v.as_str()).ok_or("set_boundary_pos: missing bnd_id")?;
                let pos = payload.get("pos").and_then(parse_pos).ok_or("set_boundary_pos: missing pos")?;
                g.set_boundary_pos(inst, bnd, pos)?;
                events.push(event("boundary_moved", json!({ "inst_id": inst.to_hex(), "bnd_id": bnd, "pos": pos })));
                Ok(json!({ "ok": true }))
            }
            "duplicate_shared" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let pos = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
                let sib = g.duplicate_shared(inst, pos)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "inst_id": sib.to_hex() }))
            }
            "make_unique" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let def = g.make_unique(inst)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "def_id": def.to_hex() }))
            }
            "re_share_instance" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let def = payload
                    .get("def_id")
                    .and_then(|v| v.as_str())
                    .and_then(goofi_engine::subpatch::DefId::from_hex)
                    .ok_or("re_share_instance: bad def_id")?;
                let out = g.re_share_instance(inst, def)?;
                events.push(event("subpatch_changed", schemas::snapshot(&g, &state.instance_id, false)));
                Ok(json!({ "inst_id": out.to_hex() }))
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

    // Keep the server-side CRDT doc in agreement with the graph after any successful MUTATING
    // control op, then broadcast the resulting delta so every connected client's replica
    // converges. A non-empty `events` is exactly the "this op changed the graph" signal — every
    // mutating arm pushes at least one event, while read-only ops (list_nodes, serialize, save)
    // push none, so this skips a full-graph re-mirror walk on every read.
    if result.is_ok() && !events.is_empty() {
        resync_and_broadcast(state);
    }

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

/// Re-mirror the (already-locked) graph into the (already-locked) doc and broadcast the
/// resulting delta to every connected client, advancing the shared broadcast baseline. The
/// caller must hold `graph` then `crdt` (the canonical order); passing the guards in keeps the
/// whole apply→re-mirror critical section atomic so no concurrent writer can observe a doc
/// leaf the graph has not yet caught up to.
fn remirror_and_broadcast_locked(state: &AppState, g: &Graph, doc: &mut goofi_crdt::GraphDoc) {
    crdt_mirror::sync_graph_to_doc(g, doc);
    let mut last_sv = state.last_sync_sv.lock().unwrap();
    let delta = doc.diff(&last_sv);
    // A no-op mutation yields a trivial empty-diff; only broadcast a real change.
    if !doc.is_empty_diff(&delta) {
        *last_sv = doc.state_vector();
        let _ = state.sync_updates.send(goofi_crdt::SyncMsg::Update(delta).encode());
    }
}

/// Re-sync the CRDT doc from the (authoritative) graph and broadcast the resulting delta to
/// every connected client, advancing the shared broadcast baseline. Called after any graph
/// mutation — an RPC dispatch or an applied client doc-write. The re-mirror also RECONCILES
/// the doc back to the graph's authoritative structure, so a client's out-of-band structural
/// write (a bogus node/link) is reverted here rather than diverging the graph.
fn resync_and_broadcast(state: &AppState) {
    let g = state.graph.lock().unwrap();
    let mut doc = state.crdt.lock().unwrap();
    remirror_and_broadcast_locked(state, &g, &mut doc);
}

/// Apply a client's CRDT leaf write (a binary `SyncMsg::Update`) to the graph: apply it to
/// the server replica, learn exactly which param values changed, push each into the engine
/// `Graph` (coerced to the param's type, re-projected to shared siblings), then re-mirror +
/// broadcast so every client — including the writer — converges on the authoritative result.
/// Only param VALUES are honored from clients here; structural writes are reverted by the
/// re-mirror (see [`resync_and_broadcast`]). Expression edits still flow via the RPC path.
///
/// The ENTIRE apply→graph-push→re-mirror→broadcast runs under a single `graph`+`crdt` critical
/// section. This is load-bearing under concurrent writers: if the `crdt` lock were released
/// after `apply_client_update` (so another writer could apply its leaf to the doc) before this
/// writer's graph push landed, the subsequent blanket re-mirror would read a graph still behind
/// that other leaf and clobber it back to the stale value — a lost update. Holding both locks
/// throughout keeps graph and doc consistent at every release point.
fn apply_client_write(state: &AppState, update: &[u8]) {
    let mut g = state.graph.lock().unwrap();
    let mut doc = state.crdt.lock().unwrap();
    let changed = doc.apply_client_update(update).unwrap_or_default();
    if changed.is_empty() {
        return;
    }
    for (uid_hex, group, name, value) in &changed.params {
        let Some(uid) = Uid::from_hex(uid_hex) else { continue };
        let Some(existing) = g.params(uid).and_then(|p| goofi_node::param(p, group, name)).cloned()
        else {
            continue; // unknown param (e.g. a stale/bogus client write) — ignore
        };
        let newp = param_from_json(&existing, value);
        let _ = g.update_member_param(uid, group, name, newp);
    }
    for (uid_hex, pos) in &changed.positions {
        let Some(uid) = Uid::from_hex(uid_hex) else { continue };
        // `set_member_pos` moves a ROOT node, an instance box, or a shared member (mirroring to
        // siblings) — the same authority the retired `set_node_pos` RPC used.
        let _ = g.set_member_pos(uid, *pos);
    }
    for (uid_hex, blob) in &changed.viewers {
        let Some(uid) = Uid::from_hex(uid_hex) else { continue };
        // Opaque per-slot view-state — stored + persisted to .gfi verbatim (the retired
        // `set_node_viewers` RPC's authority).
        let _ = g.set_node_viewers(uid, blob.clone());
    }
    remirror_and_broadcast_locked(state, &g, &mut doc);
}

// ---------------------------------------------------------------------------
// Data plane
// ---------------------------------------------------------------------------

async fn data_ws(
    Path((node, slot)): Path<(String, String)>,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_data(socket, state, node, slot))
}

/// The inband `{op:"view", specs:[…]}` message a viewer sends on the `/data` socket to
/// declare (or update) what it can draw + wants reduced. Latest-wins: the newest list
/// replaces the connection's prior specs.
#[derive(serde::Deserialize)]
struct ViewMsg {
    op: String,
    #[serde(default)]
    specs: Vec<goofi_view::ViewSpec>,
}

fn close(code: u16, reason: &str) -> Message {
    Message::Close(Some(CloseFrame {
        code,
        reason: reason.into(),
    }))
}

async fn handle_data(socket: WebSocket, state: AppState, node: String, slot: String) {
    let (mut tx, mut rx) = socket.split();

    let uid = match Uid::from_hex(&node) {
        Some(u) => u,
        None => {
            let _ = tx.send(close(4004, "bad node uid")).await;
            return;
        }
    };
    // Resolve the physical stream target. Either `(node, slot)` is a real output slot, or
    // `node` is a sub-patch instance and `slot` is a wired OUTPUT boundary — chain-resolved to
    // its single inner leaf `(uid, slot)`. Either way exactly one physical leaf slot is streamed,
    // so a boundary viewer and an inner-scope viewer coalesce onto the same reducer (spec §5).
    let target = {
        let g = state.graph.lock().unwrap();
        if g.manifest(uid).map(|m| m.outputs.iter().any(|o| o.name == slot)).unwrap_or(false) {
            Some((uid, slot.clone()))
        } else {
            g.resolve_boundary(uid, &slot)
        }
    };
    let Some((stream_uid, stream_slot)) = target else {
        let _ = tx.send(close(4004, "unknown node/slot")).await;
        return;
    };

    // Subscribe to the SHARED per-slot reducer: the frame is reduced ONCE for this slot (to
    // the union of every subscriber's ViewSpecs) and fanned out, so N tabs on one slot cost
    // one reduce+encode, not N. This connection just forwards the reduced frames to its socket
    // and pushes its own ViewSpecs into the union (latest-wins) on each inband `{op:"view"}`.
    let key: reducer::SlotKey = (stream_uid, stream_slot);
    let conn = state.reducers.new_conn();
    let mut frames = state.reducers.subscribe(key.clone(), conn);
    loop {
        tokio::select! {
            frame = frames.recv() => match frame {
                Ok(bytes) => {
                    if tx.send(Message::Binary(bytes.to_vec().into())).await.is_err() {
                        break;
                    }
                }
                // A slow viewer that lagged the reducer's fan-out simply drops frames (latest-
                // wins, like the node↔node plane) — never stalls the shared reducer.
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            },
            incoming = rx.next() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                // Inband ViewSpec negotiation: latest-wins replace this connection's contribution
                // to the slot's spec union.
                Some(Ok(Message::Text(t))) => {
                    if let Ok(m) = serde_json::from_str::<ViewMsg>(t.as_str()) {
                        if m.op == "view" {
                            state.reducers.set_specs(&key, conn, m.specs);
                        }
                    }
                }
                _ => {}
            },
        }
    }
    // Deregister so the reducer tears down when the last viewer of this slot leaves.
    state.reducers.unsubscribe(&key, conn);
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
