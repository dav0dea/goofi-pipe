//! goofi-bridge — the axum HTTP/WebSocket server that exposes the engine to the
//! browser: the `/control` JSON RPC + broadcast-event plane and the
//! `/data/<node>/<slot>` binary GOOF plane — ONE reduced stream per (node, slot);
//! the viewer kind is NOT in the path (viewers send their ViewSpec inband via
//! `{op:"view"}`). Event-sourced: RPCs return thin acks; real state changes arrive
//! as broadcast events the client applies. The built SPA is served from disk
//! (`frontend/build`, or `GOOFI_FRONTEND_BUILD`) via `ServeDir`.

mod crdt_mirror;
mod fsbrowse;
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
use axum::routing::any;
use axum::Router;
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
    /// The doc's state vector as of the last broadcast delta — the baseline the next delta
    /// is computed against (guarded together with `crdt`: always lock `crdt` first).
    pub last_sync_sv: Arc<Mutex<Vec<u8>>>,
    /// Whether the patch has been mutated since it was last saved or loaded — the title-bar dot
    /// and the unload guard. DERIVED, not stored: nothing persists it, so a fresh session starts
    /// clean and every successful mutating op sets it.
    dirty: Arc<std::sync::atomic::AtomicBool>,
    /// Shared per-slot data reducers (thalamus G1/G2): one reduction per active (node, slot),
    /// fanned out to every viewer, so N tabs on one slot cost one reduce+encode, not N.
    pub reducers: reducer::SlotReducers,
    /// The single central per-session command history (unified-command API). A command-backed op
    /// applies through here (recording its inverse tagged with the caller's session); `undo`/`redo`
    /// replay the inverse/forward for that session. Locked AFTER `graph`, BEFORE `crdt`.
    pub history: Arc<Mutex<goofi_engine::CommandHistory>>,
    /// Liveness policy for `/data` sockets. Injectable so a test need not sit through a
    /// production-length deadline.
    pub data_liveness: DataLiveness,
    /// Where the open patch's workspace files live while it is open — the tree a `.gfi` packs and
    /// unpacks. Created at boot, dropped by [`AppState::release_mount`] on a graceful exit; after a
    /// crash it simply stays, because a reboot clears the system temp directory.
    ///
    /// Shared and private because a load REPLACES it (the loaded patch brings its own workspace)
    /// while every handler holds its own clone of the state — one stored path, read through
    /// [`AppState::mount`], is the single source of truth for where the workspace is right now.
    mount: Arc<Mutex<PathBuf>>,
}

/// Timings that govern how a `/data` socket detects a **dead-but-not-closed** peer — a laptop that
/// slept, a NAT that dropped the flow, a killed tab that never sent Close. Such a peer holds its
/// connection (and therefore its share of the slot's reducer) open forever unless the bridge
/// actively probes it, because a socket with no traffic produces no error.
#[derive(Clone, Copy, Debug)]
pub struct DataLiveness {
    /// How often an otherwise-idle peer is probed with a WS Ping.
    pub ping_interval: Duration,
    /// How long an un-answered ping may stand before the peer is declared dead. Deliberately
    /// several ping intervals, so a couple of lost round-trips on a bad mobile link do not
    /// disconnect a healthy viewer.
    pub pong_deadline: Duration,
    /// The longest a single outgoing write may block before the loop gives up on it. This is a
    /// *non-parking* bound, not a liveness verdict — see `handle_data`.
    pub send_timeout: Duration,
}

impl DataLiveness {
    /// Production timings: probe every 10 s, declare dead after 30 s (three missed round-trips),
    /// never let one write park the loop for more than 5 s. A peer that walks out of WiFi is
    /// reclaimed within [30 s, 40 s] — small next to a session, large next to a hiccup.
    pub const DEFAULT: DataLiveness = DataLiveness {
        ping_interval: Duration::from_secs(10),
        pong_deadline: Duration::from_secs(30),
        send_timeout: Duration::from_secs(5),
    };
}

impl Default for DataLiveness {
    fn default() -> Self {
        DataLiveness::DEFAULT
    }
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
            last_sync_sv,
            dirty: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            reducers,
            history: Arc::new(Mutex::new(goofi_engine::CommandHistory::new())),
            data_liveness: DataLiveness::DEFAULT,
            mount: Arc::new(Mutex::new(new_mount())),
        }
    }

    /// Where the open patch's workspace files live *right now*. Copied out rather than borrowed:
    /// the lock guards only the swap, and no filesystem walk may run while holding it.
    pub fn mount(&self) -> PathBuf {
        self.mount.lock().unwrap().clone()
    }

    /// Drop the workspace mount, nonce directory and all. Deleting a *parent* needs no shape check
    /// to be safe: the field is private and its only two writers — boot, and the load swap — both
    /// store a `new_mount()` result, so what it names is always that nonce directory. Best-effort by
    /// decision: a failure leaves one directory in the system temp dir, which a reboot clears — no
    /// registry, no retry, no reporting.
    pub fn release_mount(&self) {
        remove_mount(&self.mount());
    }
}

/// A fresh, empty workspace mount: `<temp>/goofi-<128-bit hex>/workspace`. The nonce directory
/// wraps it so that loading a patch can rename an extracted tree onto `workspace` wholesale, and
/// so one `remove_dir_all` reclaims the pair. A failed mkdir is not worth reporting at boot: the
/// first save into an unwritable temp dir surfaces the real IO error, naming the path.
fn new_mount() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("goofi-{}", nonce_hex())).join("workspace");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Reclaim a mount: the nonce directory, not just `workspace` — otherwise every released mount
/// leaves an empty husk behind.
fn remove_mount(mount: &std::path::Path) {
    let _ = std::fs::remove_dir_all(mount.parent().unwrap_or(mount));
}

/// A 128-bit random name, hex. Only has to be unguessable-enough to keep two concurrent goofis —
/// or two concurrent saves onto one target — from colliding.
fn nonce_hex() -> String {
    let mut nonce = [0u8; 16];
    getrandom::fill(&mut nonce).expect("the OS random source");
    format!("{:032x}", u128::from_be_bytes(nonce))
}

/// Pack the patch to `target` as a `.gfi`: `manifest` beside the live workspace `mount`.
///
/// The pack goes to a temp sibling and is renamed onto the target, so a write that dies part-way —
/// a full disk, a workspace file that vanished mid-walk — leaves the PREVIOUS `.gfi` standing. The
/// bare `fs::write` this replaces could mostly only lose the new content; a multi-entry zip
/// truncates the old file first and then has many more chances to fail.
///
/// Runs under the graph lock the caller already holds, so the tick stalls for the duration (tens of
/// ms for a workspace of code files) — accepted by decision, because taking the pack off-lock means
/// guarding a graph-versus-workspace race that only exists once it is off-lock.
fn save_archive(target: &std::path::Path, manifest: &str, mount: &std::path::Path) -> Result<(), String> {
    // The mount's nonce directory is deleted when the patch closes, so a save into it is a save
    // into nothing. Both sides go through `resolve` for the same reason the arm's path does: they
    // must agree on what a path means, and only one of the two arrived normalized.
    let owned = fsbrowse::resolve(&mount.parent().unwrap_or(mount).to_string_lossy());
    if std::path::Path::new(&fsbrowse::resolve(&target.to_string_lossy())).starts_with(&owned) {
        return Err("save failed: that folder is the patch's own temporary workspace".into());
    }
    // Suffix appended, not substituted, so the temp is a sibling of the target and the rename
    // below stays within one filesystem.
    let tmp = PathBuf::from({
        let mut s = target.as_os_str().to_owned();
        s.push(format!(".tmp-{}", nonce_hex()));
        s
    });
    let packed = goofi_engine::archive::write_gfi(&tmp, manifest, mount)
        .and_then(|()| std::fs::rename(&tmp, target).map_err(|e| format!("{}: {e}", target.display())));
    if packed.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    packed.map_err(|e| format!("save failed: {e}"))
}

/// The fallible half of a load, run against a mount that is not yet live: unpack (or, for
/// `load_text`, simply accept) the patch and apply its manifest to `g`, reporting the file it came
/// from. `load` unpacks the archive's workspace tree into `mount`; `load_text` is a browser upload
/// that cannot carry a workspace, so its `mount` stays the empty one the caller made.
///
/// Nothing here touches state the caller has not already agreed to lose: `g` is replaced only by
/// `load_doc`, which is the last thing that can fail, and `mount` is not yet the live one. That is
/// what lets the caller commit unconditionally once this returns `Ok`.
fn load_into(
    mount: &std::path::Path,
    g: &mut Graph,
    op: &str,
    payload: &Value,
) -> Result<Option<String>, String> {
    let (content, from_path) = if op == "load" {
        // Expand `~` exactly as the browser does — the two must agree on what a path means.
        let path =
            fsbrowse::resolve(payload.get("path").and_then(|v| v.as_str()).ok_or("load: missing path")?);
        let manifest = goofi_engine::archive::read_gfi(std::path::Path::new(&path), mount)
            .map_err(|e| format!("load failed: {e}"))?;
        (manifest, Some(path))
    } else {
        let content =
            payload.get("content").and_then(|v| v.as_str()).ok_or("load_text: missing content")?;
        (content.to_string(), None)
    };
    // Parse BEFORE the caller announces anything: a rejected patch must not leave the title bar
    // naming a file the graph was never loaded from.
    g.load_doc(&content)?;
    Ok(from_path)
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/control", any(control_ws))
        // One stream per (node, slot) — the kind segment is gone; a single reduced stream
        // serves every viewer kind. Each connection sends its viewers' ViewSpecs inband.
        .route("/data/{node}/{slot}", any(data_ws))
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
/// An INLINE node — native or in-process Python — runs its `process()` under this mutex, so a slow
/// one paces the lock for every other holder. A Subprocess-isolated node does not: it ticks on its
/// own detached worker, which `tests/detached_no_freeze.rs` pins.
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
/// full-params `state_update`: this async 2 Hz snapshot must never carry params. The
/// original reason no longer holds — the frontend stopped replacing params wholesale at
/// the doc cutover, so a stale snapshot can no longer clobber a concurrent `update_param`
/// (values are doc-owned; `_mergeParamRuntime` takes only the runtime bits). The shape
/// still stands on its own: a late snapshot would overwrite a fresher `expression_error`,
/// and every param of every node twice a second is bandwidth for a payload neither the
/// node border nor the console reads. The per-param expression-error field refreshes on
/// the next RPC; the node border + console update live here.
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
        // A detached node bootstraps off-tick, so its stage changes with no RPC to ride on —
        // same reason the error transition is pushed here.
        let mut last_stages: HashMap<String, &'static str> = HashMap::new();
        loop {
            std::thread::sleep(period);
            let (rates, errs, expr_vals, stages) = {
                let g = graph.lock().unwrap();
                let mut rates: Vec<(String, f64)> = Vec::new();
                let mut errs: Vec<(String, Option<String>)> = Vec::new();
                let mut stages: Vec<(String, &'static str)> = Vec::new();
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
                    stages.push((hex.clone(), g.node_stage(u)));
                    errs.push((hex, g.last_error(u).map(str::to_string)));
                }
                (rates, errs, expr_vals, stages)
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
            for (node, stage) in stages {
                if last_stages.insert(node.clone(), stage) != Some(stage) {
                    let _ = events.send(event("node_stage", json!({ "node": node, "stage": stage })));
                }
            }
            last_stages.retain(|h, _| errs.iter().any(|(e, _)| e == h));
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

/// Native node type names visible in the catalog (`--list-nodes`; also ensures linkage).
/// Hides `_`-prefixed test nodes, exactly as the palette projection does.
pub fn catalog_type_names() -> Vec<String> {
    let _ = goofi_nodes::native_node_count();
    goofi_node::catalog()
        .filter(|m| !m.type_name.starts_with('_'))
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
    // idempotently. The CRDT plane subscribes first for the same reason.
    let mut events = state.events.subscribe();
    let mut sync_updates = state.sync_updates.subscribe();

    let hello = {
        let g = state.graph.lock().unwrap();
        event("hello", schemas::snapshot(&g, &state.instance_id, true, state.is_dirty()))
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
                    // handshake (reply with the diff it lacks), and a client `Update` is never
                    // expected and is IGNORED (the doc is manager-authored; an out-of-band leaf
                    // write would just be reverted by the next re-mirror anyway).
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
                        event("hello", schemas::snapshot(&g, &state.instance_id, true, state.is_dirty()))
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
        }
    }
}

impl AppState {
    /// Whether the patch differs from its last saved/loaded state.
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Set the dirty flag, returning an `unsaved_changes` event ONLY when it actually changed —
    /// every mutation would otherwise re-broadcast the same value.
    fn set_dirty(&self, dirty: bool) -> Option<String> {
        let was = self.dirty.swap(dirty, std::sync::atomic::Ordering::Relaxed);
        (was != dirty).then(|| event("unsaved_changes", json!({ "unsaved_changes": dirty })))
    }
}

fn event(name: &str, payload: Value) -> String {
    json!({ "event": name, "payload": payload }).to_string()
}

/// A per-node `state_update` event carrying a node's current params + error. Emitted for every
/// peer a §4.5 shared-member edit touches (param value, position, expression), so any observer
/// reconciles each mirrored sibling.
fn param_state_update(g: &Graph, peer: Uid) -> String {
    param_state_update_refreshed(g, peer, &[])
}

/// As [`param_state_update`], naming the params whose ⟳ refresh just completed. The frontend
/// clears each one's spinner off this list, so it must be sent on EVERY outcome — including a
/// refresh that turned up nothing — or the button spins until its 15s safety timeout.
fn param_state_update_refreshed(g: &Graph, peer: Uid, refreshed: &[(&str, &str)]) -> String {
    event(
        "state_update",
        json!({
            "node": peer.to_hex(),
            "params": schemas::describe_node_params(g, peer),
            "stage": g.node_stage(peer),
            "error": g.last_error(peer),
            "refreshed_params": refreshed.iter().map(|(g, n)| json!([g, n])).collect::<Vec<_>>(),
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

/// A boundary wire's inner target. Both halves named = wire; both absent or `null` = UNWIRE — the
/// `None` [`goofi_engine::Command::WireStub`] already models ("an unwire always applies"), and the
/// only shape the edge-delete path sends. Parsing the pair as ONE value is what keeps the
/// half-specified third state unconstructible: name either half and both are required.
fn parse_inner(payload: &Value) -> Result<Option<(Uid, String)>, String> {
    let named = |k: &str| payload.get(k).is_some_and(|v| !v.is_null());
    if !named("inner_node") && !named("inner_slot") {
        return Ok(None);
    }
    Ok(Some((parse_uid(payload, "inner_node")?, parse_str(payload, "inner_slot")?.to_string())))
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

/// Did the user AUTHOR the editor arrangement, or merely navigate it? Persistence is a separate
/// axis: the layout rides the `.gfi` whichever way it changed, but only *authoring* it — splitting
/// a panel, picking a viewer kind or a slot — makes the patch differ from disk. Navigation
/// (entering a sub-patch, switching a layout tab, an undo/redo re-orientation) and the manager's
/// own layout echoed back on hello leave the file's meaning intact.
///
/// Two consumers read this one axis, which is why it names the classification rather than either
/// effect: the unsaved dot (navigation must not raise it, nor the unload guard) and the `layout`
/// broadcast (navigation must not move a peer's view — it is where *this* client is looking).
///
/// The client owns the classification — it is the only side that knows what the user did — and
/// declares it as `intent`. A payload that declares nothing is authoring, so forgetting to
/// classify can only cost a spurious dot, never a lost change.
fn layout_write_is_authored(payload: &Value) -> bool {
    payload.get("intent").and_then(|v| v.as_str()) != Some("navigation")
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
        // Ops that read no graph state are served WITHOUT the graph mutex. `list_dir` walks a
        // directory and stats every child, which can block for a long time on a huge or network
        // path — under the lock that would stall the tick thread for the whole walk.
        if op == "list_dir" {
            return Ok(fsbrowse::list_dir(payload.get("path").and_then(|v| v.as_str())));
        }
        let mut g = state.graph.lock().unwrap();
        match op.as_str() {
            "list_nodes" => Ok(json!({ "types": schemas::catalog_types(&g) })),
            "add_node" => {
                let ty = payload
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or("add_node: missing type")?
                    .to_string();
                // `member_uid` + `name` place the new node at a CHOSEN uid and display name instead
                // of minting fresh ones. This is NOT the undo path — undo/redo are manager-owned
                // and a restore goes through `Command::AddNode { uid: Some, name: Some }` built by
                // `capture_subtree_restore`, never through this RPC. It is an automation/restore
                // door: a caller reconstructing a known graph (a script, a fixture) gets the
                // uid-keyed links and panels to reconnect to the same node.
                let restore = payload.get("member_uid").and_then(|v| v.as_str()).and_then(Uid::from_hex);
                let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let pos = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
                // `inst_id` is the sub-patch the editor has ENTERED: the node is born INSIDE it.
                // Absent/null = ROOT. A malformed id is refused here and an id naming no live scope
                // is refused by the command's pre-mutation check — never silently rooted, because
                // the canvas draws only the entered scope's children, so a rooted node is invisible
                // exactly where the user placed it (while the panel still selects it).
                let scope = match payload.get("inst_id").filter(|v| !v.is_null()) {
                    Some(v) => {
                        Some(v.as_str().and_then(Uid::from_hex).ok_or("add_node: malformed inst_id")?)
                    }
                    None => None,
                };
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
                    scope,
                };
                let uid = match state.history.lock().unwrap().apply(&mut g, &session, cmd)? {
                    goofi_engine::Outcome::Uid(u) => u,
                    _ => return Err("add_node: no uid returned".into()),
                };
                // Optional inline params (paste/duplicate replay + undo-of-delete): apply at creation
                // UNDER THE GRAPH LOCK so the node is born configured (same coercion as update_param),
                // before the resync mirrors them into the doc.
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
                // A bare uid announcement: the node itself arrives via the doc mirror, so anything
                // more would be a second, drift-prone projection of it.
                events.push(event("node_added", json!({ "uid": uid.to_hex() })));
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
            // Recovery, not an edit: respawn the node's instance in place, keeping its uid, name,
            // params, expressions, viewers, scope and links. NOT routed through the command history
            // — the client records no `graph_cmd` for a restart, and the two stacks must stay 1:1.
            "restart_node" => {
                let uid = parse_uid(&payload, "node")?;
                g.restart_node(uid)?;
                // Push the cleared error straight away so the node's red border lifts on the click
                // rather than on the next 2 Hz error-transition sweep.
                events.push(param_state_update(&g, uid));
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
            // Re-enumerate a refreshable string param (a device/stream picker). NOT a command —
            // options are runtime-only, never persisted, so there is nothing to undo. They are
            // also invisible to the CRDT doc, so this echo is the ONLY way they reach the client.
            "refresh_param" => {
                let uid = parse_uid(&payload, "node")?;
                let group = parse_str(&payload, "group")?.to_string();
                let name = parse_str(&payload, "name")?.to_string();
                g.refresh_param(uid, &group, &name)?;
                events.push(param_state_update_refreshed(&g, uid, &[(&group, &name)]));
                Ok(json!({ "ok": true }))
            }
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
            // Patch-scoped editor layout, stored opaquely (the node-`viewers` rule). NOT a command
            // — view state is not undoable — and both dirtying and broadcasting only when the
            // client says the user AUTHORED the arrangement (`layout_write_is_authored`).
            "set_layout" => {
                let layout = payload.get("layout").cloned().unwrap_or(Value::Null);
                // The layout is deliberately NOT a CRDT doc root, so the post-dispatch re-mirror
                // cannot carry it and a peer would learn the arrangement only on `hello`. This
                // event is how every other client converges live. It names its author so that
                // client can ignore its own echo rather than re-applying it.
                if layout_write_is_authored(&payload) {
                    events.push(event("layout", json!({ "layout": layout, "session": session })));
                }
                g.set_layout(layout);
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
                // Reject a duplicate display name up front (mirrors `rename_global`). The engine's
                // `Command::EditNode` tolerates a rename collision as a no-op so a stale undo-replay
                // converges instead of wedging the stack — so the user-facing error must be raised
                // here, at the forward RPC boundary.
                if g.name_taken(&name, uid) {
                    return Err(format!("rename_node: display name `{name}` already in use"));
                }
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
                    goofi_engine::Command::EditGlobal { name, value: Some(value), at: None },
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
                    goofi_engine::Command::EditGlobal { name, value: Some(value), at: None },
                )?;
                Ok(json!({ "ok": true }))
            }
            "remove_global" => {
                let name = parse_str(&payload, "name")?.to_string();
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::EditGlobal { name, value: None, at: None },
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
                        goofi_engine::Command::EditGlobal { name: new, value: Some(value), at: None },
                        goofi_engine::Command::EditGlobal { name: old, value: None, at: None },
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
                state
                    .history
                    .lock()
                    .unwrap()
                    .apply(&mut g, &session, goofi_engine::Command::Expand { scope: inst })?;
                Ok(json!({ "ok": true }))
            }
            "add_boundary" => {
                let inst = parse_uid(&payload, "inst_id")?;
                let dir = match payload.get("dir").and_then(|v| v.as_str()) {
                    Some("in") => goofi_engine::subpatch::Dir::In,
                    Some("out") => goofi_engine::subpatch::Dir::Out,
                    _ => return Err("add_boundary: dir must be \"in\" or \"out\"".into()),
                };
                let dtype = goofi_core::SlotType::from_name(
                    payload.get("dtype").and_then(|v| v.as_str()).unwrap_or("ARRAY"),
                )
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
                let inner = parse_inner(&payload)?;
                state.history.lock().unwrap().apply(
                    &mut g,
                    &session,
                    goofi_engine::Command::WireStub { scope: inst, stub_id: bnd, inner, dtype: None },
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
                // Expand `~` exactly as the browser does, or a path the user could navigate to
                // would not be writable — the two must agree on what a path means. The path is
                // REQUIRED: the old no-path form quietly returned the YAML for a browser
                // download ("Save in browser"), a second save semantics that left the dirty
                // flag standing and that the save-path design (C38) would have had to carry.
                // The user removed the feature; a save writes a file or it is malformed.
                let path = payload
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(fsbrowse::resolve)
                    .ok_or("save: missing path")?;
                save_archive(std::path::Path::new(&path), &g.serialize(), &state.mount())?;
                // Written to disk ⇒ clean.
                events.extend(state.set_dirty(false));
                Ok(json!({ "path": path }))
            }
            // One load path for both sources: `load_text` carries the YAML inline (a browser
            // upload), `load` names a `.gfi` the BACKEND reads. Everything after the read —
            // replace, reset history, announce — must not drift between them, so they share an arm.
            "load_text" | "load" => {
                // Both sources mount FRESH, and the live mount is swapped for it only once the
                // manifest has parsed. So a refused load leaves the open patch untouched on both
                // planes — its graph AND its workspace files — and a loaded patch never inherits
                // the files of the patch it replaced.
                let fresh = new_mount();
                let from_path =
                    load_into(&fresh, &mut g, &op, &payload).inspect_err(|_| remove_mount(&fresh))?;
                // Commit, now that nothing left can fail: the loaded patch's workspace becomes the
                // live one and the mount it replaced is reclaimed — after the lock drops, since
                // deleting a tree is a walk and the lock guards only the swap.
                let replaced = std::mem::replace(&mut *state.mount.lock().unwrap(), fresh);
                remove_mount(&replaced);
                // A load fully resets the session — there is nothing to undo across it (spec §3:
                // no load command / no checkpoint), so drop every session's command history.
                state.history.lock().unwrap().clear();
                events.extend(state.set_dirty(false));
                events.push(event(
                    "graph_replaced",
                    schemas::snapshot(&g, &state.instance_id, false, false),
                ));
                if let Some(path) = from_path {
                    // The patch now has a home on disk — the title bar names it and a later plain
                    // Save overwrites it without re-prompting. AFTER `graph_replaced`, whose
                    // snapshot carries `save_path: null` (the manager keeps no save-path state)
                    // and is applied wholesale by the client — announcing first would be
                    // immediately clobbered.
                    events.push(event("save_path_changed", json!({ "save_path": path })));
                }
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
    let read_only = matches!(op.as_str(), "list_nodes" | "serialize" | "save" | "list_dir");
    if result.is_ok() && !read_only {
        resync_and_broadcast(state);
        // "Could this have changed the graph?" is a good enough answer to "does the patch now
        // differ from disk?" for most ops that the two share a gate — but it is an INFERENCE, and
        // these four are where it is wrong:
        //   `load`/`load_text` clear the flag inside their arm, which runs first and is then
        //     re-set here; re-clear it.
        //   `set_layout` — persistence and dirtiness are separate axes (`layout_write_is_authored`).
        //   `restart_node` respawns an instance in place, replaying the node's own ParamGroups
        //     verbatim and touching neither name, position, bindings, viewers, links nor scopes, so
        //     `serialize()` is byte-identical. It is RECOVERY, not an edit, and it is reached by one
        //     click on the inspector's Restart button after a node raised — exactly where a spurious
        //     unsaved dot is least distinguishable from a real one.
        //   `refresh_param` re-enumerates a device/stream picker's options, which are runtime-only
        //     and never persisted. Latent today (no shipped node declares `refresh: true`, and the
        //     engine rejects the op for any param that does not, so the `Err` skips this gate
        //     entirely) — listed here because it is the same op-is-not-an-edit case, not a
        //     prediction that it currently misfires.
        // Both stay OUT of `read_only`: neither is an edit, but both still need the re-mirror.
        match op.as_str() {
            "load" | "load_text" => events.extend(state.set_dirty(false)),
            "set_layout" if !layout_write_is_authored(&payload) => {}
            "restart_node" | "refresh_param" => {}
            _ => events.extend(state.set_dirty(true)),
        }
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
    // after). A state-vector empty-diff check cannot do this: it is deletion-blind — a Yjs delete
    // does not advance the state vector, so a delete-only `diff(last_sv)` is byte-identical to the
    // empty baseline `diff(current_sv)`, and every node/link/instance/global REMOVAL would be
    // silently dropped from the broadcast. `to_json` equality catches adds, edits, and deletes alike
    // (the same lesson the frontend `SyncClient.commit` learned about the always-embedded delete set).
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
/// every connected client, advancing the shared broadcast baseline. Called after an RPC dispatch
/// mutates the graph. The re-mirror also RECONCILES the doc back to the graph's authoritative
/// structure (idempotent), so any stale doc leaf converges to the graph.
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

/// How a bounded `/data` write ended: delivered, given up on (peer stalled past the bound), or
/// the socket itself failed.
#[derive(Debug, PartialEq)]
enum SendOutcome {
    Sent,
    Dropped,
    Gone,
}

/// Write one message to a `/data` socket, giving up after `bound`.
///
/// The bound is not a policy about slow viewers — it is what keeps the caller's `tokio::select!`
/// from parking. An `.await` inside a select BRANCH BODY runs to completion with no other branch
/// polled, so an unbounded write to a peer whose TCP window stopped draining would freeze the
/// keepalive beat: the liveness probe would be dead code on exactly the socket it exists to catch.
/// Giving up on a message costs one frame (latest-wins, as `Lagged` does) — the caller re-offers
/// a `Dropped` frame through the reducer, since the skip-unchanged sweep will not send it again
/// on its own; parking costs the connection forever.
async fn send_bounded<S>(tx: &mut S, msg: Message, bound: Duration) -> SendOutcome
where
    S: futures_util::Sink<Message> + Unpin,
{
    // A timeout leaves at most this one message buffered: the sink's `poll_ready` gates the NEXT
    // write on the same unfinished flush, so nothing accumulates behind a peer that stopped
    // reading.
    match tokio::time::timeout(bound, tx.send(msg)).await {
        Ok(Ok(())) => SendOutcome::Sent,
        Ok(Err(_)) => SendOutcome::Gone,
        Err(_) => SendOutcome::Dropped,
    }
}

/// What the `/data` keepalive timer should do on this beat.
#[derive(Debug, PartialEq, Eq)]
enum Beat {
    /// Probe the peer, and start (or keep) the pong deadline running.
    Ping,
    /// A ping is outstanding but still inside the deadline — say nothing, keep waiting.
    Wait,
    /// The deadline lapsed unanswered: the peer is gone. Leave the loop so the socket's
    /// existing `unsubscribe` teardown runs.
    Dead,
}

/// Whether the peer on one `/data` socket has shown, within the deadline, that its receive path is
/// moving. A stalled *write* is not itself evidence of death — a slow phone stalls writes too —
/// so the verdict rests on the peer failing to make **any** progress for a whole deadline.
struct PeerLiveness {
    cfg: DataLiveness,
    /// When the oldest un-answered probe was sent; `None` while the peer is known to be moving.
    awaiting_pong_since: Option<std::time::Instant>,
}

impl PeerLiveness {
    fn new(cfg: DataLiveness) -> PeerLiveness {
        PeerLiveness { cfg, awaiting_pong_since: None }
    }

    /// The verdict for this beat. A probe is marked outstanding on the ATTEMPT, not on a
    /// successful write: a peer whose receive path is jammed cannot be pinged at all, and that is
    /// precisely the condition the deadline exists to catch — crediting it for the write we could
    /// not make would leave the stalled peer undetectable.
    fn beat(&mut self, now: std::time::Instant) -> Beat {
        match self.awaiting_pong_since {
            // Measured from the OLDEST unanswered probe, so beating faster than the deadline
            // (the normal case) cannot keep postponing the verdict.
            Some(sent) if now.duration_since(sent) >= self.cfg.pong_deadline => Beat::Dead,
            Some(_) => Beat::Wait,
            None => {
                self.awaiting_pong_since = Some(now);
                Beat::Ping
            }
        }
    }

    /// The peer answered — it read our probe, so its receive path is moving. This is the ONLY
    /// thing that keeps a connection alive: a *sent* probe cannot be its own proof of life, and a
    /// flushed frame is not proof either (the socket buffer of a peer that stopped reading keeps
    /// swallowing frames until it is full).
    fn pong(&mut self) {
        self.awaiting_pong_since = None;
    }
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

    // Peer liveness. A dead-but-not-closed peer (slept laptop, dropped NAT flow, killed tab that
    // never sent Close) produces NO socket error, so without an active probe this connection —
    // and its share of the shared slot reducer — would live forever.
    let cfg = state.data_liveness;
    let mut live = PeerLiveness::new(cfg);
    let mut keepalive = tokio::time::interval(cfg.ping_interval);

    loop {
        tokio::select! {
            frame = frames.recv() => match frame {
                Ok(bytes) => {
                    // BOUNDED (see `send_bounded`), because this `.await` sits in a select BRANCH
                    // BODY: an unbounded write to a peer that stopped draining would park here
                    // and starve the keepalive beat below.
                    //
                    // Giving up on a frame is deliberately NOT a liveness signal in either
                    // direction. A timeout is not death — a slow phone stalls writes too, and
                    // dropping the frame is the same latest-wins contract as `Lagged` just below.
                    // A flush is not life either: the socket buffer of a peer that stopped reading
                    // keeps swallowing frames until it is full, which on a low-rate slot takes
                    // minutes. Only the pong decides.
                    match send_bounded(&mut tx, Message::Binary(bytes), cfg.send_timeout).await {
                        SendOutcome::Sent => {}
                        // The sweep will not resend an unchanged frame, so a dropped one must be
                        // asked for again — otherwise the drop costs every frame until the next
                        // emit (a one-shot join/spec serve would simply be lost).
                        SendOutcome::Dropped => state.reducers.reoffer(&key),
                        SendOutcome::Gone => break, // the socket really is gone
                    }
                }
                // A slow viewer that lagged the reducer's fan-out simply drops frames (latest-
                // wins, like the node↔node plane) — never stalls the shared reducer. Re-offer for
                // the same reason as a Dropped write: the missed frame may have been the last.
                Err(broadcast::error::RecvError::Lagged(_)) => state.reducers.reoffer(&key),
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
                // The peer answered our probe: the one and only thing that clears the deadline.
                Some(Ok(Message::Pong(_))) => live.pong(),
                _ => {}
            },
            // The keepalive beat. Complementary to the bounded send above, not redundant with it:
            // the bounded send stops the loop parking on a BACKED-UP peer, this catches an IDLE
            // dead one — no frames means no send, so a write timeout alone would never fire.
            _ = keepalive.tick() => match live.beat(std::time::Instant::now()) {
                // Bounded for the same reason as the frame send: a jammed sink must not park us.
                // The probe's own write succeeding proves nothing — only the answer does.
                Beat::Ping => {
                    // A Dropped probe needs no re-offer — only the missing pong means anything,
                    // and the deadline below is already counting.
                    if send_bounded(&mut tx, Message::Ping(Default::default()), cfg.send_timeout).await
                        == SendOutcome::Gone
                    {
                        break;
                    }
                }
                Beat::Wait => {}
                // Fall out so the EXISTING unsubscribe below runs and the shared reducer is
                // reclaimed once its last real viewer is gone.
                Beat::Dead => break,
            },
        }
    }
    // Deregister so the reducer tears down when the last viewer of this slot leaves.
    state.reducers.unsubscribe(&key, conn);
}

#[cfg(test)]
mod save_archive_tests {
    use super::*;

    /// The headline: a save packs the LIVE workspace mount, not merely some directory. Nothing
    /// else pins which tree reaches `write_gfi` — the wire test's target has an empty mount, so it
    /// cannot tell a correct one from a wrong one.
    #[test]
    fn packs_the_mount_alongside_the_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let mount = tmp.path().join("goofi-0123").join("workspace");
        std::fs::create_dir_all(&mount).unwrap();
        std::fs::write(mount.join("agent.md"), b"notes").unwrap();

        let target = tmp.path().join("patch.gfi");
        save_archive(&target, "version: 7\n", &mount).unwrap();

        let dest = tmp.path().join("unpacked");
        assert_eq!(goofi_engine::archive::read_gfi(&target, &dest).unwrap(), "version: 7\n");
        assert_eq!(std::fs::read(dest.join("agent.md")).unwrap(), b"notes");
    }

    #[test]
    fn a_failed_pack_leaves_the_previous_archive_intact() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("patch.gfi");
        std::fs::write(&target, b"the previous save").unwrap();

        // A mount that is not on disk fails the workspace walk — which happens AFTER the zip's
        // first entry is written, so it is exactly the window in which packing straight onto the
        // target would truncate a good `.gfi` into a half-written one. It has to sit OUTSIDE the
        // target's own directory, or the mount refusal answers first and the pack never runs
        // (which is how this test first passed against a version that had no temp+rename at all).
        let mount = tmp.path().join("mnt").join("gone").join("workspace");
        let err = save_archive(&target, "version: 7\n", &mount).unwrap_err();
        assert!(err.contains("save failed"), "the refusal names the operation: {err}");
        assert_eq!(std::fs::read(&target).unwrap(), b"the previous save");
        let left: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().map(|e| e.unwrap().path()).collect();
        assert_eq!(left, [target], "the half-written temp sibling is cleaned up too");
    }

    #[test]
    fn refuses_a_target_inside_the_workspace_mount() {
        let tmp = tempfile::tempdir().unwrap();
        let mount = tmp.path().join("goofi-0123").join("workspace");
        std::fs::create_dir_all(&mount).unwrap();

        for target in [mount.join("patch.gfi"), mount.parent().unwrap().join("patch.gfi")] {
            let err = save_archive(&target, "version: 7\n", &mount).unwrap_err();
            assert!(err.contains("temporary workspace"), "the refusal says why: {err}");
            assert!(!target.exists(), "a refused save writes nothing");
        }
    }
}

#[cfg(test)]
mod param_coerce_tests {
    use super::*;
    use goofi_core::Param;
    use serde_json::json;

    #[test]
    fn removing_a_node_broadcasts_a_delta() {
        // A node REMOVAL must broadcast a delta to clients. Regression: the broadcast gate once used
        // a state-vector empty-diff check, which is deletion-blind (a Yjs delete doesn't advance the
        // state vector, so a delete-only delta looked identical to the empty baseline) — so removals
        // silently never reached clients in the doc read-path. Caught by the e2e undo flow (undo
        // didn't remove); the gate now compares `to_json` before/after instead.
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

#[cfg(test)]
mod peer_liveness_tests {
    use super::*;
    use std::time::Instant;

    fn cfg() -> DataLiveness {
        DataLiveness {
            ping_interval: Duration::from_millis(100),
            pong_deadline: Duration::from_millis(300),
            send_timeout: Duration::from_millis(200),
        }
    }

    #[test]
    fn an_idle_peer_is_probed_then_left_alone_until_the_deadline() {
        // The first beat probes; subsequent beats inside the deadline stay quiet rather than
        // stacking pings on a peer that may simply be between round-trips.
        let t0 = Instant::now();
        let mut live = PeerLiveness::new(cfg());
        assert_eq!(live.beat(t0), Beat::Ping, "an unprobed peer is pinged");
        assert_eq!(live.beat(t0 + Duration::from_millis(100)), Beat::Wait, "the probe stands");
        assert_eq!(live.beat(t0 + Duration::from_millis(299)), Beat::Wait, "still inside deadline");
    }

    #[test]
    fn a_probe_unanswered_past_the_deadline_declares_the_peer_dead() {
        // The dead-but-not-closed case: nothing errored, nothing closed, the peer just stopped
        // answering. Only the elapsed deadline can distinguish it from an idle healthy viewer.
        let t0 = Instant::now();
        let mut live = PeerLiveness::new(cfg());
        assert_eq!(live.beat(t0), Beat::Ping);
        assert_eq!(live.beat(t0 + Duration::from_millis(300)), Beat::Dead, "deadline lapsed");
    }

    #[test]
    fn a_pong_clears_the_deadline_so_an_alive_peer_is_never_declared_dead() {
        // The regression guard in pure form: however long a viewer is watched, as long as it
        // answers it is only ever re-probed — never declared dead.
        let t0 = Instant::now();
        let mut live = PeerLiveness::new(cfg());
        for cycle in 0..20 {
            let now = t0 + Duration::from_millis(100 * cycle);
            assert_eq!(live.beat(now), Beat::Ping, "cycle {cycle}: an answered peer is re-probed");
            live.pong();
        }
    }

    #[test]
    fn a_pong_arriving_late_in_the_deadline_still_saves_the_peer() {
        // A backlogged viewer answers only just before the deadline — it must be credited in
        // full, not merely granted a stay: the clock restarts from the next probe.
        let t0 = Instant::now();
        let mut live = PeerLiveness::new(cfg());
        assert_eq!(live.beat(t0), Beat::Ping);
        live.pong();
        let late = t0 + Duration::from_millis(299);
        assert_eq!(live.beat(late), Beat::Ping, "the next beat re-probes rather than condemning");
        assert_eq!(live.beat(late + Duration::from_millis(299)), Beat::Wait, "clock ran from `late`");
    }

    #[test]
    fn the_deadline_runs_from_the_oldest_unanswered_probe_not_the_latest_beat() {
        // A `Wait` beat must not refresh the clock, or the deadline could never expire on a peer
        // that is beaten more often than the deadline is long — which is the normal case.
        let t0 = Instant::now();
        let mut live = PeerLiveness::new(cfg());
        assert_eq!(live.beat(t0), Beat::Ping);
        for step in 1..3 {
            assert_eq!(live.beat(t0 + Duration::from_millis(100 * step)), Beat::Wait);
        }
        assert_eq!(live.beat(t0 + Duration::from_millis(300)), Beat::Dead, "measured from t0");
    }
}

#[cfg(test)]
mod send_bounded_tests {
    use super::*;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// A sink that never becomes ready — a peer whose TCP window stopped draining, modelled
    /// directly so the test does not depend on the OS socket-buffer sizes it would take to
    /// reproduce that against a real socket.
    struct StalledSink;

    impl futures_util::Sink<Message> for StalledSink {
        type Error = axum::Error;
        fn poll_ready(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }
        fn start_send(self: Pin<&mut Self>, _: Message) -> Result<(), Self::Error> {
            unreachable!("a stalled sink is never ready to be given a message")
        }
        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }
        fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }
    }

    #[tokio::test]
    async fn a_write_to_a_stalled_peer_gives_up_instead_of_parking_the_loop() {
        // The property the whole fix rests on. `handle_data`'s send sits in a select BRANCH BODY,
        // so if it never returns no other branch is ever polled again — the keepalive beat would
        // be starved on precisely the dead-but-not-closed socket it exists to catch. Bounding the
        // write is what lets the loop go round.
        // Small, so the test is fast; the outer bound is 40x it, so this asserts the PROPERTY
        // (it returned) with room to spare rather than a tight window.
        let bound = Duration::from_millis(50);
        let mut sink = StalledSink;
        // The outer bound makes an unbounded send FAIL cleanly rather than hang the suite.
        let outcome = tokio::time::timeout(
            bound * 40,
            send_bounded(&mut sink, Message::Binary(Default::default()), bound),
        )
        .await;
        assert_eq!(
            outcome.ok(),
            Some(SendOutcome::Dropped),
            "a write to a stalled peer must return as Dropped (not Gone — the socket is not \
             dead) so the keepalive beat gets polled and the frame is re-offered"
        );
    }

    #[tokio::test]
    async fn a_probe_to_a_stalled_peer_gives_up_too() {
        // The beat's own write is bounded for the same reason: a jammed sink must not park the
        // loop on the very branch that is supposed to declare the peer dead.
        let bound = Duration::from_millis(50);
        let mut sink = StalledSink;
        let outcome = tokio::time::timeout(
            bound * 40,
            send_bounded(&mut sink, Message::Ping(Default::default()), bound),
        )
        .await;
        assert_eq!(outcome.ok(), Some(SendOutcome::Dropped), "the probe write is bounded as well");
    }
}
