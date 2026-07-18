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
                // Redo-of-add / undo-of-delete replay the ORIGINAL uid (member_uid) + name so
                // uid-keyed links + panels reconnect to the same node; a plain add mints a fresh uid.
                // (inst_id sub-patch member placement is not yet restored here — ROOT nodes only.)
                let uid = match payload.get("member_uid").and_then(|v| v.as_str()).and_then(Uid::from_hex) {
                    Some(restore) => {
                        let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        g.add_node_at(ty, None, restore, name)?
                    }
                    None => g.add_node(ty, None)?,
                };
                // Optional inline params (paste/duplicate replay + undo-of-delete): apply at creation
                // UNDER THE GRAPH LOCK so the node is born configured. A post-add update_param would
                // now be a doc leaf-write that no-ops until the node has synced into the client's
                // replica — silently dropping the replayed values (same coercion as the update_param
                // arm). node_added is emitted after, so it carries the configured values, and the
                // resync mirrors them into the doc.
                if let Some(groups) = payload.get("params").and_then(|v| v.as_object()) {
                    for (group, names) in groups {
                        let Some(names) = names.as_object() else { continue };
                        for (name, vjson) in names {
                            if let Some(existing) =
                                g.params(uid).and_then(|p| goofi_node::param(p, group, name)).cloned()
                            {
                                let newp = goofi_engine::param_from_json(&existing, vjson, true);
                                let _ = g.update_member_param(uid, group, name, newp);
                            }
                        }
                    }
                }
                if let Some(pos) = payload.get("pos").and_then(parse_pos) {
                    let _ = g.set_node_pos(uid, pos);
                }
                events.push(event("node_added", schemas::node_instance_info(&g, uid)));
                Ok(json!(uid.to_hex()))
            }
            "remove_node" => {
                let uid = parse_uid(&payload, "node")?;
                // Route by what the uid is. A sub-patch MEMBER (leaf or nested instance living in an
                // instance's scope) is removed from its def AND every strict-mirror sibling via
                // remove_member — so deleting a node inside a sub-patch doesn't leave a dangling
                // member the def would resurrect on reload. A TOP-LEVEL instance (collapsed sub-patch
                // delete, or the inverse of duplicate_shared) tears down its subtree; a top-level leaf
                // is a plain remove. Every result reaches clients via the post-dispatch re-mirror
                // (the node_removed / subpatch_changed echoes are retired — the frontend reconciles
                // the whole forest from the doc).
                if g.scope_of(uid).is_some() {
                    g.remove_member(uid)?;
                } else if g.scope(uid).is_some() {
                    g.remove_instance(uid)?;
                } else {
                    g.remove_node(uid)?;
                }
                Ok(json!({ "ok": true }))
            }
            // Links are read from the CRDT doc (Phase 2) — the resolved flat link rides the re-mirror
            // after dispatch. The old `link_added`/`link_removed` events had no client consumer.
            "add_link" => {
                let (a, so, b, si) = parse_link(&payload)?;
                // Resolve either endpoint through a sub-patch boundary → flat leaf→leaf.
                let (a, so) = resolve_link_endpoint(&g, a, &so);
                let (b, si) = resolve_link_endpoint(&g, b, &si);
                g.add_link(a, &so, b, &si)?;
                Ok(json!({ "ok": true }))
            }
            "remove_link" => {
                let (a, so, b, si) = parse_link(&payload)?;
                let (a, so) = resolve_link_endpoint(&g, a, &so);
                let (b, si) = resolve_link_endpoint(&g, b, &si);
                g.remove_link(a, &so, b, &si)?;
                Ok(json!({ "ok": true }))
            }
            // Retained deliberately, unlike its 3 leaf-write siblings (set_node_pos/viewers/
            // expression), whose handlers were removed: the frontend no longer calls update_param
            // (params are leaf-written to the doc), but this handler is the tested reference for the
            // uniform mutating-RPC → graph → re-mirror invariant (crdt_doc_tracks_an_rpc_node_add_
            // and_param_edit) and the constant-value push-flood test — behaviours a leaf-write's
            // skip-if-unchanged deliberately can't reproduce. It runs under the graph lock and
            // re-mirrors like the structural RPCs, so it carries no lost-update risk.
            "update_param" => {
                let uid = parse_uid(&payload, "node")?;
                let group = parse_str(&payload, "group")?;
                let name = parse_str(&payload, "name")?;
                let vjson = payload.get("value").ok_or("missing value")?;
                let existing = g
                    .params(uid)
                    .and_then(|p| goofi_node::param(p, group, name))
                    .cloned()
                    .ok_or("no such param")?;
                let newp = goofi_engine::param_from_json(&existing, vjson, true);
                // Re-project to every shared sibling (§4.5): a shared member's edit hits all its
                // instances. A ROOT / unique-member edit updates only itself.
                let updated = g.update_member_param(uid, group, name, newp)?;
                for peer in updated {
                    events.push(param_state_update(&g, peer));
                }
                Ok(json!({ "ok": true }))
            }
            // `set_expression` retired (Phase 3): the expression binding is a merge-safe leaf the
            // client writes directly to the doc. `apply_client_write` applies it via
            // `set_member_expression` (re-projecting to shared siblings) and echoes the same
            // runtime-enriched `state_update` (carrying `expression_error`) this handler used to.
            // `set_node_pos` / `set_node_viewers` retired (Phase 3): node/instance position and the
            // per-slot viewer blob are merge-safe leaves the client writes directly to the doc
            // (`apply_client_write` → `set_member_pos` / `set_node_viewers`).
            "rename_node" => {
                let uid = parse_uid(&payload, "node")?;
                let name = parse_str(&payload, "name")?;
                let referrers = g.rename_node(uid, name)?;
                // The new name reaches clients via the post-dispatch re-mirror (node_renamed retired).
                // The nd('new') rewrite of any referring expression is itself a doc leaf, but push each
                // referrer's fresh params so a runtime `expression_error` from re-evaluating the rewrite
                // still surfaces on its inspector (the doc carries the source, not the runtime error).
                for r in referrers {
                    events.push(param_state_update(&g, r));
                }
                Ok(json!({ "ok": true }))
            }
            // The sub-patch structural ops (group/expand/boundary authoring/share) mutate the forest
            // and return; the mutated forest reaches every client via the post-dispatch re-mirror,
            // which the frontend reconciles from the doc. The old `subpatch_changed` snapshot echo is
            // retired (Phase 4) — the doc read-path covers it.
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
                Ok(json!({ "inst_id": inst.to_hex() }))
            }
            "expand_instance" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let restored = g.expand_instance(inst)?;
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
                Ok(json!({ "bnd_id": bnd }))
            }
            "wire_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = parse_str(&payload, "bnd_id")?;
                let inner = parse_uid(&payload, "inner_node")?;
                let slot = parse_str(&payload, "inner_slot")?;
                g.wire_boundary(inst, bnd, inner, slot)?;
                Ok(json!({ "ok": true }))
            }
            "remove_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = parse_str(&payload, "bnd_id")?;
                g.remove_boundary(inst, bnd)?;
                Ok(json!({ "ok": true }))
            }
            "rename_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = parse_str(&payload, "bnd_id")?;
                let name = parse_str(&payload, "name")?;
                g.rename_boundary(inst, bnd, name)?;
                Ok(json!({ "ok": true }))
            }
            "set_boundary_pos" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let bnd = parse_str(&payload, "bnd_id")?;
                let pos = payload.get("pos").and_then(parse_pos).ok_or("set_boundary_pos: missing pos")?;
                g.set_boundary_pos(inst, bnd, pos)?;
                // Boundary positions are read from the CRDT doc forest (retired `boundary_moved`).
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
        let newp = goofi_engine::param_from_json(&existing, value, true);
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
    // Expression bindings: the doc is the SSOT for the binding (source/enabled/triggers), but the
    // per-param `expression_error` is RUNTIME-derived and never enters the doc. So after applying
    // each binding (re-projected to shared siblings), echo the runtime-enriched param descriptor as
    // the same `state_update` the retired `set_expression` RPC emitted — the client's fx toggle and
    // field error indicator refresh exactly as before, with no read-path change.
    let mut expr_peers: Vec<Uid> = Vec::new();
    for (uid_hex, group, name, binding) in &changed.expressions {
        let Some(uid) = Uid::from_hex(uid_hex) else { continue };
        let (source, enabled, triggers) = match binding {
            Some(e) => (e.source.as_str(), e.enabled, e.triggers),
            None => ("", false, false), // a cleared binding — revert to the literal value
        };
        if let Ok(updated) = g.set_member_expression(uid, group, name, source, enabled, triggers) {
            expr_peers.extend(updated);
        }
    }
    // Globals: `Some(entry)` sets/adds, `None` deletes. A system-delete is rejected by the engine and
    // re-asserted by the re-mirror below — so a client's attempt to delete a system global reappears.
    for (name, entry) in &changed.globals {
        let value = match entry {
            Some(e) => match goofi_engine::global_from_json(e) {
                Some(v) => Some(v),
                None => continue, // malformed entry — ignore
            },
            None => None,
        };
        let _ = g.apply_global_change(name, value);
    }
    remirror_and_broadcast_locked(state, &g, &mut doc);
    // Emit after the re-mirror so the doc + graph are consistent; dedup so one descriptor per node.
    expr_peers.sort_by_key(|u| u.to_hex());
    expr_peers.dedup();
    for peer in expr_peers {
        let _ = state.events.send(param_state_update(&g, peer));
    }
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
