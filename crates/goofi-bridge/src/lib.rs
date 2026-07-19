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
    /// The single central per-session command history (unified-command API). A command-backed op
    /// applies through here (recording its inverse tagged with the caller's session); `undo`/`redo`
    /// replay the inverse/forward for that session. Locked AFTER `graph`, BEFORE `crdt`.
    pub history: Arc<Mutex<goofi_engine::CommandHistory>>,
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
        // Mirror the INITIAL graph (empty nodes + the seeded system globals) into the doc so a client
        // connecting to a fresh backend syncs the current state immediately (e.g. `default_ufreq`),
        // rather than an empty doc that stays blank until the first mutation re-mirrors.
        let graph_val = Graph::new();
        let mut crdt = goofi_crdt::GraphDoc::new();
        crdt_mirror::sync_graph_to_doc(&graph_val, &mut crdt);
        let last_sync_sv = Arc::new(Mutex::new(crdt.state_vector()));
        let graph = Arc::new(Mutex::new(graph_val));
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
            history: Arc::new(Mutex::new(goofi_engine::CommandHistory::new())),
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

/// Serve on an already-bound listener with optional static SPA serving. Passing `None` serves the
/// API only (`app(state, None)` is exactly `router(state)`) — what tests use for an ephemeral port.
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
                    // A CRDT sync frame from the client. The client replica is READ-ONLY (B3):
                    // every mutation is a command RPC, so a StateVector drives the pairwise sync
                    // handshake (reply with the diff it lacks) and an Ephemeral relays presence —
                    // a client `Update` is never expected and is IGNORED (the doc is manager-authored;
                    // an out-of-band leaf write would just be reverted by the next re-mirror anyway).
                    match goofi_crdt::SyncMsg::decode(&b) {
                        Some(msg @ goofi_crdt::SyncMsg::StateVector(_)) => {
                            let replies = state.crdt.lock().unwrap().on_sync(msg);
                            for r in replies {
                                if tx.send(Message::Binary(r.encode().into())).await.is_err() {
                                    return;
                                }
                            }
                        }
                        Some(goofi_crdt::SyncMsg::Update(_)) => {} // read-only client — ignored
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

/// A required string field from an RPC payload, erroring `missing {key}` if absent or non-string —
/// the sibling of [`parse_uid`] for plain string args. (The op is recoverable from the request, so
/// the error needs no per-op prefix.)
fn parse_str<'a>(payload: &'a Value, key: &str) -> Result<&'a str, String> {
    payload.get(key).and_then(|v| v.as_str()).ok_or_else(|| format!("missing {key}"))
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
    if g.scope(uid).is_some() {
        if let Some(leaf) = g.resolve_stub(uid, slot) {
            return leaf;
        }
    }
    (uid, slot.to_string())
}


/// Dispatch one control RPC. Mutates the graph, queues broadcast events, and
/// returns the `{id,result}`/`{id,error}` reply (only when `id` is numeric).
fn dispatch(state: &AppState, text: &str) -> Option<String> {
    let req: Value = serde_json::from_str(text).ok()?;
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let op = req.get("op")?.as_str()?.to_string();
    let payload = req.get("payload").cloned().unwrap_or_else(|| json!({}));
    // The caller's session tag (a browser tab's stable id) scopes the command history's undo/redo.
    // Absent ⇒ a single shared "default" session, so a client that never presents one still works.
    let session = req.get("session").and_then(|v| v.as_str()).unwrap_or("default").to_string();

    let mut events: Vec<String> = Vec::new();
    let result: Result<Value, String> = (|| {
        let mut g = state.graph.lock().unwrap();
        match op.as_str() {
            "list_nodes" => Ok(json!({ "types": schemas::catalog_types(&g) })),
            "add_node" => {
                let ty = payload
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or("add_node: missing type")?
                    .to_string();
                // Redo-of-add / undo-of-delete replay the ORIGINAL uid (member_uid) + name so
                // uid-keyed links + panels reconnect to the same node; a plain add mints a fresh uid.
                // (inst_id sub-patch member placement is not yet restored here — ROOT nodes only.)
                let restore = payload.get("member_uid").and_then(|v| v.as_str()).and_then(Uid::from_hex);
                let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let pos = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
                // Route through the command history so the add is undoable (its inverse is a
                // RemoveNode). Inline params are applied AFTER (below): RemoveNode's inverse
                // capture_restores the LIVE node — INCLUDING those params — so an undo→redo restores
                // the configured values without threading them through the command here.
                let cmd = goofi_engine::Command::AddNode {
                    type_name: ty,
                    pos,
                    uid: restore,
                    name: (!name.is_empty()).then_some(name),
                    params: None,
                    exprs: vec![],
                    viewers: None,
                };
                let uid = match state.history.lock().unwrap().apply(&mut g, &session, cmd)? {
                    goofi_engine::Outcome::Uid(u) => u,
                    _ => return Err("add_node: no uid returned".into()),
                };
                // Optional inline params (paste/duplicate replay + undo-of-delete): apply at creation
                // UNDER THE GRAPH LOCK so the node is born configured (same coercion as update_param).
                // node_added is emitted after, so it carries the configured values, and the resync
                // mirrors them into the doc.
                if let Some(groups) = payload.get("params").and_then(|v| v.as_object()) {
                    for (group, names) in groups {
                        let Some(names) = names.as_object() else { continue };
                        for (name, vjson) in names {
                            if let Some(existing) =
                                g.params(uid).and_then(|p| goofi_node::param(p, group, name)).cloned()
                            {
                                let newp = goofi_engine::param_from_json(&existing, vjson, true);
                                let _ = g.update_param(uid, group, name, newp);
                            }
                        }
                    }
                }
                events.push(event("node_added", schemas::node_instance_info(&g, uid)));
                Ok(json!(uid.to_hex()))
            }
            "remove_node" => {
                let uid = parse_uid(&payload, "node")?;
                // A top-level leaf, a sub-patch member (leaf or nested instance), or a collapsed
                // instance — RemoveNode dispatches internally and CAPTURES the whole subtree
                // (members + params + links + stubs + membership) so its inverse restores it
                // uid-stably (undoable; B3b closed the delete-undo gap). The result reaches clients
                // via the post-dispatch re-mirror.
                state
                    .history
                    .lock()
                    .unwrap()
                    .apply(&mut g, &session, goofi_engine::Command::RemoveNode { uid })?;
                Ok(json!({ "ok": true }))
            }
            // Links are read from the CRDT doc (Phase 2) — the resolved flat link rides the re-mirror
            // after dispatch. The old `link_added`/`link_removed` events had no client consumer.
            "add_link" => {
                let (a, so, b, si) = parse_link(&payload)?;
                // Resolve either endpoint through a sub-patch boundary → flat leaf→leaf, THEN route
                // the resolved flat link through the history (undoable; inverse is a RemoveLink).
                let (a, so) = resolve_link_endpoint(&g, a, &so);
                let (b, si) = resolve_link_endpoint(&g, b, &si);
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::AddLink { node_out: a, slot_out: so, node_in: b, slot_in: si },
                )?;
                Ok(json!({ "ok": true }))
            }
            "remove_link" => {
                let (a, so, b, si) = parse_link(&payload)?;
                let (a, so) = resolve_link_endpoint(&g, a, &so);
                let (b, si) = resolve_link_endpoint(&g, b, &si);
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::RemoveLink { node_out: a, slot_out: so, node_in: b, slot_in: si },
                )?;
                Ok(json!({ "ok": true }))
            }
            // The leaf edits (param value / expression / node+instance pos / rename / globals) route
            // through the command history so each is undoable (B3a). The mutation reaches clients via
            // the post-dispatch re-mirror; only the runtime-derived, doc-invisible bits (a param's
            // `expression_error`, a rename's nd()-rewrite echo) are pushed as `state_update` events.
            "update_param" => {
                let uid = parse_uid(&payload, "node")?;
                let group = parse_str(&payload, "group")?.to_string();
                let name = parse_str(&payload, "name")?.to_string();
                let vjson = payload.get("value").ok_or("missing value")?;
                let existing = g
                    .params(uid)
                    .and_then(|p| goofi_node::param(p, &group, &name))
                    .cloned()
                    .ok_or("no such param")?;
                let newp = goofi_engine::param_from_json(&existing, vjson, true);
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditParam { uid, group, name, value: Some(newp), expr: None },
                )?;
                Ok(json!({ "ok": true }))
            }
            "set_expression" => {
                let uid = parse_uid(&payload, "node")?;
                let group = parse_str(&payload, "group")?.to_string();
                let name = parse_str(&payload, "name")?.to_string();
                // An absent/null/empty `expression` clears the binding (revert to the literal);
                // `enabled`/`triggers` default false.
                let source = payload.get("expression").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let enabled = payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
                let triggers = payload.get("triggers").and_then(|v| v.as_bool()).unwrap_or(false);
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditParam {
                        uid,
                        group,
                        name,
                        value: None,
                        expr: Some(goofi_engine::ExprState { source, enabled, triggers }),
                    },
                )?;
                // The binding source rides the doc re-mirror; the runtime `expression_error` is
                // doc-invisible, so echo the enriched descriptor (what the retired leaf path did).
                events.push(param_state_update(&g, uid));
                Ok(json!({ "ok": true }))
            }
            "set_node_pos" => {
                let uid = parse_uid(&payload, "node")?;
                let pos = payload.get("pos").and_then(parse_pos).ok_or("set_node_pos: missing pos")?;
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditNode { uid, name: None, pos: Some(pos) },
                )?;
                Ok(json!({ "ok": true }))
            }
            "set_node_viewers" => {
                // Soft per-slot view-state (kind/settings/collapse) persisted to `.gfi` — NOT a
                // command (not undoable). Written to the graph; the re-mirror persists + broadcasts.
                let uid = parse_uid(&payload, "node")?;
                let viewers = payload.get("viewers").cloned().ok_or("set_node_viewers: missing viewers")?;
                g.set_node_viewers(uid, viewers)?;
                Ok(json!({ "ok": true }))
            }
            "rename_node" => {
                let uid = parse_uid(&payload, "node")?;
                let name = parse_str(&payload, "name")?.to_string();
                let out = state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditNode { uid, name: Some(name), pos: None },
                )?;
                // The new name rides the re-mirror; each referrer whose nd() expression was rewritten
                // needs its runtime-enriched descriptor re-pushed (the source is in the doc, the
                // runtime error is not).
                if let goofi_engine::Outcome::Nodes(referrers) = out {
                    for r in referrers {
                        events.push(param_state_update(&g, r));
                    }
                }
                Ok(json!({ "ok": true }))
            }
            // Globals validation is server-side now (the retired client `docAddGlobal`/`docRename`
            // guards moved here): `add_global` REJECTS a collision, `set_global` edits an EXISTING
            // one, `rename_global` refuses a system/colliding/invalid target up front (its Compound
            // is not atomic, so a mid-sequence failure would leave a phantom). Wire shape carries the
            // typed value as `{ name, value, type }`.
            "add_global" => {
                let name = parse_str(&payload, "name")?.to_string();
                let val = payload.get("value").ok_or("add_global: missing value")?;
                let ty = payload.get("type").and_then(|v| v.as_str()).ok_or("add_global: missing type")?;
                if g.globals().contains(&name) {
                    return Err(format!("add_global: global `{name}` already exists"));
                }
                // On an ABSENT name, EditGlobal routes through GlobalStore::add, which validates the
                // name (an invalid name still rejects).
                let value = goofi_engine::global_from_json(&json!({ "value": val, "type": ty }))
                    .ok_or("add_global: malformed value")?;
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditGlobal { name, value: Some(value) },
                )?;
                Ok(json!({ "ok": true }))
            }
            "set_global" => {
                // EDIT an existing global's value (system or user); rejects a non-existent name so it
                // cannot silently create one (that is `add_global`'s job).
                let name = parse_str(&payload, "name")?.to_string();
                let val = payload.get("value").ok_or("set_global: missing value")?;
                let ty = payload.get("type").and_then(|v| v.as_str()).ok_or("set_global: missing type")?;
                if !g.globals().contains(&name) {
                    return Err(format!("set_global: no such global `{name}`"));
                }
                let value = goofi_engine::global_from_json(&json!({ "value": val, "type": ty }))
                    .ok_or("set_global: malformed value")?;
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditGlobal { name, value: Some(value) },
                )?;
                Ok(json!({ "ok": true }))
            }
            "remove_global" => {
                let name = parse_str(&payload, "name")?.to_string();
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditGlobal { name, value: None },
                )?;
                Ok(json!({ "ok": true }))
            }
            "rename_global" => {
                let old = parse_str(&payload, "old")?.to_string();
                let new = parse_str(&payload, "new")?.to_string();
                // Validate the WHOLE rename up front (the Compound is NOT atomic — a mid-sequence
                // failure would leave the add-new applied as a phantom). Refuse a missing/system
                // source and a colliding/invalid target, so both children are guaranteed to succeed.
                let value = g.globals().get(&old).cloned().ok_or("rename_global: no such global")?;
                if g.globals().is_system(&old) {
                    return Err(format!("rename_global: cannot rename system global `{old}`"));
                }
                if g.globals().contains(&new) {
                    return Err(format!("rename_global: `{new}` already exists"));
                }
                if !goofi_core::globals::is_valid_global_name(&new) {
                    return Err(format!("rename_global: invalid name `{new}`"));
                }
                // A rename = add-new(with the old value) + remove-old, folded into one undo step.
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::Compound(vec![
                        goofi_engine::Command::EditGlobal { name: new, value: Some(value) },
                        goofi_engine::Command::EditGlobal { name: old, value: None },
                    ]),
                )?;
                Ok(json!({ "ok": true }))
            }
            // The sub-patch structural ops (group/expand/boundary authoring/share) mutate the forest
            // and return; the mutated forest reaches every client via the post-dispatch re-mirror,
            // which the frontend reconciles from the doc. The old `subpatch_changed` snapshot echo is
            // retired (Phase 4) — the doc read-path covers it.
            // The structural sub-patch ops route through the command history (undoable, uid-stable on
            // the flat model). Each parses a Command, applies it, and maps the Outcome to the reply.
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
                let out = state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::Group { members: uids, pos, restore: None },
                )?;
                let inst = match out {
                    goofi_engine::Outcome::Uid(u) => u,
                    _ => return Err("group_nodes: no scope uid returned".into()),
                };
                Ok(json!({ "inst_id": inst.to_hex() }))
            }
            "expand_instance" => {
                let inst = parse_uid(&payload, "inst_id")?;
                // Capture the members BEFORE the command dissolves the scope — the legacy client
                // expand executor (until Task B3) reads `restored` to record its own inverse.
                let restored = g.scope_members(inst);
                state
                    .history
                    .lock()
                    .unwrap()
                    .apply(&mut g, &session, goofi_engine::Command::Expand { scope: inst })?;
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
                let out = state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::AddStub { scope: inst, dir, dtype, pos, restore: None },
                )?;
                let bnd = match out {
                    goofi_engine::Outcome::StubId(id) => id,
                    _ => return Err("add_boundary: no stub id returned".into()),
                };
                Ok(json!({ "bnd_id": bnd }))
            }
            "wire_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = parse_str(&payload, "bnd_id")?.to_string();
                let inner = parse_uid(&payload, "inner_node")?;
                let slot = parse_str(&payload, "inner_slot")?.to_string();
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::WireStub { scope: inst, stub_id: bnd, inner: Some((inner, slot)), dtype: None },
                )?;
                Ok(json!({ "ok": true }))
            }
            "remove_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = parse_str(&payload, "bnd_id")?.to_string();
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::RemoveStub { scope: inst, stub_id: bnd },
                )?;
                Ok(json!({ "ok": true }))
            }
            "rename_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = parse_str(&payload, "bnd_id")?.to_string();
                let name = parse_str(&payload, "name")?.to_string();
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditStub { scope: inst, stub_id: bnd, name: Some(name), pos: None },
                )?;
                Ok(json!({ "ok": true }))
            }
            "set_boundary_pos" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = parse_str(&payload, "bnd_id")?.to_string();
                let pos = payload.get("pos").and_then(parse_pos).ok_or("set_boundary_pos: missing pos")?;
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditStub { scope: inst, stub_id: bnd, name: None, pos: Some(pos) },
                )?;
                Ok(json!({ "ok": true }))
            }
            // duplicate_shared / make_unique / re_share_instance are gone — sub-patch sharing was
            // dropped in the flat-scope re-architecture (sub-patches are organizational facades now).
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
                // A load fully resets the session — there is nothing to undo across it (spec §3:
                // no load command / no checkpoint), so drop every session's command history.
                state.history.lock().unwrap().clear();
                events.push(event(
                    "graph_replaced",
                    schemas::snapshot(&g, &state.instance_id, false),
                ));
                Ok(json!({ "ok": true }))
            }
            // Session-scoped undo/redo over the central command history. The graph mutation reaches
            // clients via the post-dispatch re-mirror (doc-authoritative); the reply carries the
            // session's fresh can-undo/can-redo so the UI can enable its buttons.
            "undo" => {
                let mut hist = state.history.lock().unwrap();
                let changed = hist.undo(&mut g, &session)?;
                Ok(json!({ "changed": changed, "can_undo": hist.can_undo(&session), "can_redo": hist.can_redo(&session) }))
            }
            "redo" => {
                let mut hist = state.history.lock().unwrap();
                let changed = hist.redo(&mut g, &session)?;
                Ok(json!({ "changed": changed, "can_undo": hist.can_undo(&session), "can_redo": hist.can_redo(&session) }))
            }
            other => Err(format!("unknown op `{other}`")),
        }
    })();

    // Keep the server-side CRDT doc in agreement with the graph after any successful MUTATING
    // control op, then broadcast the resulting delta so every connected client's replica converges.
    // The re-mirror is gated on whether the op *could* have mutated the graph — NOT on `events`,
    // because link/boundary writes mutate the doc-read graph while emitting no client event (their
    // `link_added`/`boundary_moved` events are retired). Read-only ops touch nothing and skip the
    // expensive full-graph walk; any other op re-mirrors (an unchanged re-mirror is a no-op empty
    // diff that broadcasts nothing, so defaulting a new op to re-mirror is safe).
    let read_only = matches!(op.as_str(), "list_nodes" | "serialize" | "save");
    if result.is_ok() && !read_only {
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
    // Gate the broadcast on whether the mirror changed the doc's LOGICAL state (`to_json` before vs
    // after). `is_empty_diff` cannot do this: it is deletion-blind — a Yjs delete does not advance the
    // state vector, so a delete-only `diff(last_sv)` is byte-identical to the empty baseline
    // `diff(current_sv)`, and every node/link/instance/global REMOVAL was silently dropped from the
    // broadcast. `to_json` equality catches adds, edits, and deletes alike (the same lesson the
    // frontend `SyncClient.commit` learned about the always-embedded delete set).
    let before = doc.to_json();
    crdt_mirror::sync_graph_to_doc(g, doc);
    if doc.to_json() == before {
        return; // no logical change → nothing to broadcast (no tombstone churn)
    }
    let mut last_sv = state.last_sync_sv.lock().unwrap();
    // `diff(last_sv)` carries the missing structs AND the full delete set — so a peer at `last_sv`
    // applies the removal even though the state vector is unchanged by it.
    let delta = doc.diff(&last_sv);
    *last_sv = doc.state_vector();
    let _ = state.sync_updates.send(goofi_crdt::SyncMsg::Update(delta).encode());
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
    // `node` is a sub-patch scope and `slot` is a wired OUTPUT stub — chain-resolved to its single
    // inner leaf `(uid, slot)`. Either way exactly one physical leaf slot is streamed, so a stub
    // viewer and an inner-scope viewer coalesce onto the same reducer (spec §5).
    let target = {
        let g = state.graph.lock().unwrap();
        if g.manifest(uid).map(|m| m.outputs.iter().any(|o| o.name == slot)).unwrap_or(false) {
            Some((uid, slot.clone()))
        } else {
            g.resolve_stub(uid, &slot)
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
    use goofi_core::Param;
    use serde_json::json;

    #[test]
    fn removing_a_node_broadcasts_a_delta() {
        // A node REMOVAL must broadcast a delta to clients. Regression: the broadcast gate used
        // `is_empty_diff`, which is deletion-blind (a Yjs delete doesn't advance the state vector, so
        // a delete-only delta looked identical to the empty baseline) — so removals silently never
        // reached clients in the doc read-path. Caught by the e2e undo flow (undo didn't remove).
        let state = AppState::new();
        let mut rx = state.sync_updates.subscribe();

        let uid = {
            let mut g = state.graph.lock().unwrap();
            let uid = g.add_node("Buffer", None).unwrap();
            let mut doc = state.crdt.lock().unwrap();
            remirror_and_broadcast_locked(&state, &g, &mut doc);
            uid
        };
        rx.try_recv().expect("adding a node broadcasts a delta");

        {
            let mut g = state.graph.lock().unwrap();
            g.remove_node(uid).unwrap();
            let mut doc = state.crdt.lock().unwrap();
            remirror_and_broadcast_locked(&state, &g, &mut doc);
        }
        assert!(rx.try_recv().is_ok(), "removing a node must broadcast a delta, not be skipped as empty");
    }

    #[test]
    fn fresh_appstate_mirrors_seeded_globals_into_the_doc() {
        // A fresh backend must serve a doc that already carries the seeded system globals, so a
        // client connecting BEFORE any mutation syncs `default_ufreq` — not an empty doc that stays
        // blank until the first edit. (Regression: the e2e globals-panel flow saw an empty doc.)
        let state = AppState::new();
        let doc = state.crdt.lock().unwrap();
        let j = doc.to_json();
        assert!(
            j.get("globals").and_then(|g| g.get("default_ufreq")).is_some(),
            "fresh doc must carry the seeded system global; got {j}"
        );
    }

    #[test]
    fn int_param_rounds_fractional_instead_of_zeroing() {
        // The coercion now lives in goofi_engine (SSOT); the bridge's RPC/CRDT writes go through it
        // with fire_triggers=true. This locks the bridge's dependency on the rounding behavior.
        let p = Param::int(3, 0, 100);
        assert_eq!(goofi_engine::param_from_json(&p, &json!(5.5), true).as_i64(), Some(6));
        assert_eq!(goofi_engine::param_from_json(&p, &json!(5.4), true).as_i64(), Some(5));
        assert_eq!(goofi_engine::param_from_json(&p, &json!(7), true).as_i64(), Some(7), "plain int unaffected");
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
