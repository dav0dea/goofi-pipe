//! The axum server: `/control` (JSON RPC + broadcast events, doc state and doc deltas among them),
//! `/data/<node>/<slot>` (ONE reduced GOOF stream per slot, whatever the viewer count — the kind
//! is not in the path, since viewers publish their ViewSpec inband), `/term`, `/mcp`, and the SPA
//! compiled into the binary.

/// The control-plane document and its deltas — shape-agnostic.
pub mod doc;
mod projection;
mod fsbrowse;
mod inspect;
mod mcp;
pub mod ops;
mod origin;
mod patchfile;
mod proc;
pub mod reducer;
pub mod schemas;
pub mod term;
pub mod vocab;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a `/term` socket waits for the PTY's end-of-stream after the child is reaped: ConPTY
/// keeps its pseudoconsole open past the child's death, so on Windows that end never comes.
const EXIT_SETTLE: Duration = Duration::from_millis(250);


use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::response::Response;
use axum::routing::{any, get, post};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use goofi_engine::{Graph, Uid};
use serde_json::{json, Value};
use tokio::sync::broadcast;

/// Where the development surfaces live. One literal, so the gate and the app agree on the prefix.
pub const DEV_ROUTE_PREFIX: &str = "/dev/";

/// The built SPA as it ships: a URL path and its bytes, compiled into the binary. Empty when the
/// crate was built without a frontend, which [`HEADLESS_BUILD`] says whether anyone asked for.
pub type Spa = &'static [(&'static str, &'static [u8])];
include!(concat!(env!("OUT_DIR"), "/spa.rs"));

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<Mutex<Graph>>,
    pub events: broadcast::Sender<String>,
    pub instance_id: Arc<str>,
    /// The control-plane document every client replicates, re-projected from the graph after each
    /// successful op; its deltas ride the `events` channel.
    pub doc: Arc<Mutex<crate::doc::GraphDoc>>,
    /// Whether the patch has been mutated since it was last saved or loaded. Nothing persists it,
    /// so a fresh session starts clean.
    dirty: Arc<std::sync::atomic::AtomicBool>,
    /// One reduction per active (node, slot), fanned out to every viewer.
    pub reducers: reducer::SlotReducers,
    /// The central per-session command history. Locked AFTER `graph`, BEFORE `doc`.
    pub history: Arc<Mutex<goofi_engine::CommandHistory>>,
    /// Liveness policy for `/data` sockets, injectable so a test need not sit through a
    /// production-length deadline.
    pub data_liveness: DataLiveness,
    /// How a directory of node files becomes registered node types; the default discovers nothing.
    pub scan_nodes: NodeScan,
    /// The shipped node directories — `nodes/`, then every `--extra-nodes`.
    pub system_nodes: Vec<PathBuf>,
    /// What the last scan found, by type name → the file's stamp: the baseline the next [`rescan`]
    /// diffs against, and the only list it removes from.
    node_index: Arc<Mutex<std::collections::BTreeMap<String, Option<Stamp>>>>,
    /// The tree a `.gfi` packs and unpacks. Behind a lock because a LOAD replaces it while every
    /// handler holds its own clone of the state.
    mount: Arc<Mutex<PathBuf>>,
    /// The workspace as it was last packed or unpacked — what [`AppState::is_dirty`] compares the
    /// live mount against. Re-taken at BOTH ends.
    workspace_baseline: Arc<Mutex<std::collections::BTreeMap<PathBuf, (u64, std::time::SystemTime)>>>,
    /// Where the open patch lives on disk. Manager-owned rather than per tab, so it rides the
    /// snapshot every client connects with.
    save_path: Arc<Mutex<Option<String>>>,
    /// Only the PORT: a harness is a CHILD of this process, so `127.0.0.1` is right whatever
    /// `--bind` says.
    mcp_port: Arc<std::sync::atomic::AtomicU16>,
    /// The spawned agent harnesses, their PTYs, and the detection cache.
    pub harnesses: Arc<term::Harnesses>,
}

/// How a `/data` socket detects a dead-but-not-closed peer, which a socket with no traffic cannot
/// report on its own.
#[derive(Clone, Copy, Debug)]
pub struct DataLiveness {
    /// How often an otherwise-idle peer is probed with a WS Ping.
    pub ping_interval: Duration,
    /// How long an un-answered ping may stand before the peer is declared dead; several ping
    /// intervals, so a few lost round-trips do not disconnect a healthy viewer.
    pub pong_deadline: Duration,
    /// The longest one outgoing write may block: a NON-PARKING bound, not a liveness verdict.
    pub send_timeout: Duration,
}

impl DataLiveness {
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
        // Project the INITIAL graph — no nodes, but the seeded system globals — so a client that
        // connects to a fresh backend has the current state at once.
        let graph_val = Graph::new();
        let mut doc = crate::doc::GraphDoc::new();
        doc.reconcile_root(&projection::of(&graph_val));
        let graph = Arc::new(Mutex::new(graph_val));
        let reducers = reducer::SlotReducers::new(graph.clone());
        // Seeded BEFORE the baseline is taken, or the patch is dirty from boot, having written
        // the seed itself.
        let mount = new_mount();
        term::seed_orientation(&mount);
        let workspace_baseline = goofi_engine::archive::fingerprint(&mount);
        AppState {
            graph,
            events,
            instance_id: Arc::from(format!("{iid:x}").as_str()),
            doc: Arc::new(Mutex::new(doc)),
            dirty: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            reducers,
            history: Arc::new(Mutex::new(goofi_engine::CommandHistory::new())),
            data_liveness: DataLiveness::DEFAULT,
            scan_nodes: Arc::new(|_, _| Vec::new()),
            system_nodes: Vec::new(),
            node_index: Arc::new(Mutex::new(Default::default())),
            mount: Arc::new(Mutex::new(mount)),
            workspace_baseline: Arc::new(Mutex::new(workspace_baseline)),
            save_path: Arc::new(Mutex::new(None)),
            mcp_port: Arc::new(std::sync::atomic::AtomicU16::new(8000)),
            harnesses: Arc::new(term::Harnesses::default()),
        }
    }

    /// Point a spawned harness's MCP config at the port this server actually bound.
    pub fn set_mcp_port(&self, port: u16) {
        self.mcp_port.store(port, std::sync::atomic::Ordering::Relaxed);
    }

    /// The base URL a spawned harness reaches this server's MCP surface at — see [`mcp_port`].
    ///
    /// [`mcp_port`]: AppState::mcp_port
    fn mcp_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.mcp_port.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Where the open patch lives on disk, if anywhere.
    fn save_path(&self) -> Option<String> {
        self.save_path.lock().unwrap().clone()
    }

    /// Where the open patch's workspace files live right now. Copied out rather than borrowed: no
    /// filesystem walk may run while holding the lock.
    pub fn mount(&self) -> PathBuf {
        self.mount.lock().unwrap().clone()
    }

    /// Drop the workspace mount, nonce directory and all.
    pub fn release_mount(&self) {
        self.retire_mount(&self.mount());
    }

    /// Reclaim one mount and everything living IN it: the harnesses spawned into it are asked to
    /// leave FIRST, or one survives editing a patch out of a directory the next line deletes.
    fn retire_mount(&self, mount: &std::path::Path) {
        self.harnesses.stop_all();
        remove_mount(mount);
    }
}

/// A fresh, empty workspace mount: `<temp>/goofi-<128-bit hex>/workspace`. The nonce directory
/// wraps it so a load can rename an extracted tree onto `workspace` wholesale.
fn new_mount() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("goofi-{}", nonce_hex())).join("workspace");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Reclaim a mount: the nonce directory, not just `workspace`, which would leave an empty husk.
fn remove_mount(mount: &std::path::Path) {
    let _ = std::fs::remove_dir_all(mount.parent().unwrap_or(mount));
}

/// A 128-bit random name, hex — enough to keep two concurrent goofis from colliding.
pub(crate) fn nonce_hex() -> String {
    let mut nonce = [0u8; 16];
    getrandom::fill(&mut nonce).expect("the OS random source");
    format!("{:032x}", u128::from_be_bytes(nonce))
}

/// Pack the patch to `target`: `manifest` beside the live workspace `mount`. Written to a temp
/// sibling and RENAMED, so a write that dies part-way leaves the previous `.gfi` standing.
pub fn save_archive(target: &std::path::Path, manifest: &str, mount: &std::path::Path) -> Result<(), String> {
    // The mount's nonce directory is deleted when the patch closes, so a save into it saves into
    // nothing. Both sides go through `resolve`, or they disagree on what a path means.
    let owned = fsbrowse::resolve(&mount.parent().unwrap_or(mount).to_string_lossy());
    if std::path::Path::new(&fsbrowse::resolve(&target.to_string_lossy())).starts_with(&owned) {
        return Err("save failed: that folder is the patch's own temporary workspace".into());
    }
    // Suffix appended, not substituted, so the rename below stays within one filesystem.
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

/// The front half of a load, against a mount that is not yet live. It stops AT the manifest,
/// because the patch's own node types must be registered before `load_doc` resolves the graph.
fn stage_load(
    mount: &std::path::Path,
    op: &str,
    payload: &Value,
) -> Result<(String, Option<String>), String> {
    let (content, from_path) = if op == "new" {
        // A New patch IS a load, of an empty patch from nowhere, so the two cannot drift.
        (Graph::new().serialize(), None)
    } else if op == "load" {
        // Expand `~` exactly as the browser does — the two must agree on what a path means.
        let path =
            fsbrowse::resolve(payload.get("path").and_then(|v| v.as_str()).ok_or("load: missing path")?);
        let manifest = goofi_engine::archive::read_gfi(std::path::Path::new(&path), mount)
            .map_err(|e| format!("load failed: {e}"))?;
        // Whether this file becomes the patch's home — the target a later silent Save overwrites.
        let adopt = payload.get("adopt").and_then(Value::as_bool).unwrap_or(true);
        (manifest, adopt.then_some(path))
    } else {
        let content =
            payload.get("content").and_then(|v| v.as_str()).ok_or("load_text: missing content")?;
        (content.to_string(), None)
    };
    // Only a workspace goofi minted empty is seeded: a `load` has just unpacked the patch's OWN
    // workspace into `mount`, and goofi does not write into someone's patch.
    if op != "load" {
        term::seed_orientation(mount);
    }
    Ok((content, from_path))
}

/// The API routes, unguarded. The ONLY caller is [`app`], because `Router::layer` wraps only what
/// is already on the router — a route added elsewhere would miss the [`origin`] guard.
fn routes(state: AppState) -> Router {
    Router::new()
        .route("/control", any(control_ws))
        // One stream per (node, slot): each connection sends its viewers' ViewSpecs inband.
        .route("/data/{node}/{slot}", any(data_ws))
        // The body limit is lifted: axum caps at 2 MB and a patch with a workspace is larger.
        .route(
            "/patch.gfi",
            get(patchfile::download)
                .post(patchfile::upload)
                .layer(axum::extract::DefaultBodyLimit::disable()),
        )
        .route("/mcp", post(mcp::endpoint))
        // One address per spawned harness: identity is the ROUTE, so there is nothing to validate.
        .route("/mcp/{instance}", post(mcp::instance_endpoint))
        // A spawned harness's terminal: binary frames are PTY bytes, text frames JSON control.
        .route("/term/{instance}", any(term_ws))
        .with_state(state)
}

/// The uids whose error state changed since `last`, which is updated in place. A node first seen
/// HEALTHY is not a change, and a removed node is forgotten so a re-created uid reports fresh.
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

/// How often the worker takes what the nodes reported — deliberately NOT the event rate, because
/// draining is the RUNTIME's clock: it advances every wire's three-phase sequence, one per ack.
const DRAIN_PERIOD: Duration = Duration::from_millis(1);

/// The status-drain worker: take every node's reports, apply them to the graph, and broadcast the
/// events that carry them at `hz`.
///
/// It must never `set_dirty(true)` — a node reporting its own state is not a user edit — and must
/// FORGET a uid on removal, so a stale error cannot outlive its node.
pub fn spawn_stats(graph: Arc<Mutex<Graph>>, events: broadcast::Sender<String>, hz: u64) {
    std::thread::spawn(move || {
        let period = Duration::from_secs_f64(1.0 / hz as f64);
        let mut last_errors: HashMap<String, Option<String>> = HashMap::new();
        // A node's stage changes on its own thread, with no RPC to ride on.
        let mut last_stages: HashMap<String, &'static str> = HashMap::new();
        let mut next_broadcast = Instant::now() + period;
        loop {
            std::thread::sleep(DRAIN_PERIOD);
            let due = Instant::now() >= next_broadcast;
            let collected = {
                let mut g = graph.lock().unwrap();
                g.drain_status();
                if !due {
                    None
                } else {
                    // Options are the one thing a node reports that the doc has no field for, so
                    // this echo is the only way they reach a client.
                    let refreshed = g.take_refreshed();
                    let g = &*g;
                    let mut rates: Vec<(String, f64)> = Vec::new();
                    let mut errs: Vec<(String, Option<String>)> = Vec::new();
                    let mut stages: Vec<(String, &'static str)> = Vec::new();
                    let mut expr_vals: Vec<(String, Value)> = Vec::new();
                    for u in g.node_uids() {
                        let hex = u.to_hex();
                        if let Some(f) = g.node_ufreq(u) {
                            rates.push((hex.clone(), f));
                        }
                        let vals = schemas::expression_value_map(g, u);
                        if vals.as_object().is_some_and(|o| !o.is_empty()) {
                            expr_vals.push((hex.clone(), vals));
                        }
                        stages.push((hex.clone(), g.node_stage(u)));
                        errs.push((hex, g.last_error(u).map(str::to_string)));
                    }
                    let refreshed: Vec<String> = refreshed
                        .into_iter()
                        .filter(|(uid, _)| g.node_uids().contains(uid))
                        .map(|(uid, key)| {
                            param_state_update_refreshed(g, uid, &[(&key.group, &key.name)])
                        })
                        .collect();
                    Some((rates, errs, expr_vals, stages, refreshed))
                }
            };
            let Some((rates, errs, expr_vals, stages, refreshed)) = collected else { continue };
            // From NOW, not the deadline just passed: a worker held off the lock owes no burst of
            // catch-up broadcasts.
            next_broadcast = Instant::now() + period;
            for ev in refreshed {
                let _ = events.send(ev);
            }
            let changed = error_transitions(&errs, &mut last_errors);
            for (node, ufreq) in rates {
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

/// The API router, with no SPA.
pub fn router(state: AppState) -> Router {
    app(state, &[], false)
}

/// The full router, optionally serving the SPA on the fallback. `dev_routes` opens `/dev/*`, the
/// development surfaces. The [`origin`] guard goes on LAST, so it wraps every route — the WebSocket
/// upgrades included, which CORS would not cover.
pub fn app(state: AppState, spa: Spa, dev_routes: bool) -> Router {
    let base = routes(state);
    let served = if spa.is_empty() {
        base
    } else {
        base.fallback(move |uri| serve_spa_file(uri, dev_routes))
    };
    served.layer(axum::middleware::from_fn(origin::guard))
}

/// One embedded file, or the page itself for anything else: the client router owns every route
/// under `/`, so an unknown path is one of ITS routes and not a 404.
async fn serve_spa_file(uri: axum::http::Uri, dev_routes: bool) -> Response {
    // The client router owns unknown paths, so withholding a dev route means refusing it HERE —
    // handing back the page would let the router mount it anyway.
    if !dev_routes && uri.path().starts_with(DEV_ROUTE_PREFIX) {
        return axum::response::IntoResponse::into_response((
            axum::http::StatusCode::NOT_FOUND,
            "development routes are off — start with --debug (or GOOFI_DEBUG=1) to open them",
        ));
    }
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    let (name, body) = match SPA.iter().find(|(p, _)| *p == path) {
        Some(&(p, b)) => (p, b),
        None => match SPA.iter().find(|(p, _)| *p == "index.html") {
            Some(&(p, b)) => (p, b),
            None => return axum::response::IntoResponse::into_response(
                (axum::http::StatusCode::NOT_FOUND, "no frontend build")),
        },
    };
    axum::response::IntoResponse::into_response((
        [(axum::http::header::CONTENT_TYPE, content_type(name))],
        body,
    ))
}

/// The types the built bundle contains, plus what a `static/` asset may add; anything else is
/// served as bytes.
fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "txt" => "text/plain; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

/// The background workers a live server needs: the status-drain worker, and a primed harness
/// detection cache so the first tab already has its launch buttons.
pub fn spawn_workers(state: &AppState) {
    spawn_stats(state.graph.clone(), state.events.clone(), 2);
    state.harnesses.refresh_in_background(state.events.clone());
}

pub async fn serve_app(
    listener: tokio::net::TcpListener,
    state: AppState,
    spa: Spa,
    dev_routes: bool,
) -> std::io::Result<()> {
    axum::serve(listener, app(state, spa, dev_routes)).await
}

/// Native node type names visible in the catalog, `_`-prefixed test nodes hidden.
pub fn catalog_type_names() -> Vec<String> {
    goofi_node::catalog()
        .filter(|m| !m.type_name.starts_with('_'))
        .map(|m| m.type_name.to_string())
        .collect()
}

/// Which tier took a node file — and, when neither could, why. Reported rather than printed,
/// because only the caller can tell a boot scan from a rescan.
pub enum Tier {
    InProcess,
    Subprocess,
    /// Neither tier could load it, so the palette lists it greyed with this reason.
    Unavailable(String),
}

/// A file's size and mtime. `None` when it could not be stat'd, which compares equal to itself and
/// so reads as "unchanged".
pub type Stamp = (u64, std::time::SystemTime);

/// One node file's outcome from a scan of one directory.
pub struct ScannedType {
    pub type_name: String,
    pub tier: Tier,
    pub stamp: Option<Stamp>,
    pub registration: goofi_engine::Registration,
}

/// The node-discovery seam: scan ONE directory and report what it registered. Injected by the CLI,
/// so boot and [`rescan`] re-derive the registry through the same function.
pub type NodeScan = Arc<dyn Fn(&mut Graph, &std::path::Path) -> Vec<ScannedType> + Send + Sync>;

/// What a [`rescan`] changed, for the caller that asked.
#[derive(Default)]
pub struct ScanDiff {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
}

/// Re-derive the registry from the directories that exist RIGHT NOW — the shipped tree, then
/// `<patch>/nodes`, so a patch-local node of the same name wins. The previous scan's stamps are the
/// baseline, so this answers a DIFF and removes only what it registered.
pub fn rescan(
    state: &AppState,
    g: &mut Graph,
    patch: &std::path::Path,
) -> (ScanDiff, Vec<ScannedType>) {
    let mut found: std::collections::BTreeMap<String, Option<Stamp>> = Default::default();
    let mut patch_types: HashSet<String> = HashSet::new();
    let mut outcomes = Vec::new();
    // The scan order IS the precedence: patch LAST, so it shadows every shipped tree.
    let dirs = (state.system_nodes.iter().map(|d| (d.clone(), false)))
        .chain(std::iter::once((patch.join("nodes"), true)));
    for (dir, is_patch) in dirs {
        if !dir.is_dir() {
            continue;
        }
        for t in (state.scan_nodes)(g, &dir) {
            // A refused name never reaches the palette, so it must not enter the index either.
            if t.registration != goofi_engine::Registration::Refused {
                if is_patch {
                    patch_types.insert(t.type_name.clone());
                }
                found.insert(t.type_name.clone(), t.stamp);
            }
            outcomes.push(t);
        }
    }
    g.set_patch_types(patch_types);

    let mut prev = state.node_index.lock().unwrap();
    let mut diff = ScanDiff::default();
    for (name, stamp) in &found {
        match prev.get(name) {
            None => diff.added.push(name.clone()),
            Some(before) if before != stamp => diff.changed.push(name.clone()),
            Some(_) => {}
        }
    }
    diff.removed = prev.keys().filter(|n| !found.contains_key(*n)).cloned().collect();
    for name in &diff.removed {
        g.remove_dyn_type(name);
    }
    *prev = found;
    (diff, outcomes)
}

/// Restart every live instance of a type whose file changed, so an edit reaches the nodes already
/// on the canvas. NOT part of [`rescan`], whose graph may be about to be replaced by a load.
fn restart_changed(g: &mut Graph, diff: &ScanDiff) {
    for uid in g.node_uids() {
        if g.type_name(uid).is_some_and(|t| diff.changed.iter().any(|c| c == t)) {
            let _ = g.restart_node(uid);
        }
    }
}

async fn control_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_control(socket, state))
}

async fn handle_control(socket: WebSocket, state: AppState) {
    let (mut tx, mut rx) = socket.split();

    // Subscribe BEFORE snapshotting the document: in the other order a peer's edit lands in
    // neither, and the replica desyncs silently. A re-delivery is read as stale and skipped.
    let mut events = state.events.subscribe();

    // Answered BEFORE the graph lock is taken: it walks the mount, and no filesystem walk may run
    // while the status-drain worker waits on that lock.
    let unsaved = state.is_dirty();
    let saved_at = state.save_path();
    let hello = {
        let g = state.graph.lock().unwrap();
        event(
            "hello",
            schemas::snapshot(&g, &state.instance_id, true, unsaved, saved_at.as_deref(),
                              state.harnesses.roster()),
        )
    };
    if tx.send(Message::Text(hello.into())).await.is_err() {
        return;
    }

    if tx.send(Message::Text(doc_state(&state).into())).await.is_err() {
        return;
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
                // Lagged past the shared ring, so both halves are re-seeded exactly as a fresh
                // connection seeds them.
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let unsaved = state.is_dirty(); // off the graph lock, as above
                    let saved_at = state.save_path();
                    let hello = {
                        let g = state.graph.lock().unwrap();
                        event(
                            "hello",
                            schemas::snapshot(&g, &state.instance_id, true, unsaved,
                                              saved_at.as_deref(), state.harnesses.roster()),
                        )
                    };
                    if tx.send(Message::Text(hello.into())).await.is_err() {
                        break;
                    }
                    if tx.send(Message::Text(doc_state(&state).into())).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
        }
    }
}

impl AppState {
    /// Whether the patch differs from its last saved state. TWO sources, because a patch is a graph
    /// AND a workspace, and the workspace half is walked on ask rather than watched.
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Relaxed)
            || goofi_engine::archive::fingerprint(&self.mount()) != *self.workspace_baseline.lock().unwrap()
    }

    /// Set the dirty flag, returning an `unsaved_changes` event only when it actually changed.
    fn set_dirty(&self, dirty: bool) -> Option<String> {
        let was = self.dirty.swap(dirty, std::sync::atomic::Ordering::Relaxed);
        (was != dirty).then(|| event("unsaved_changes", json!({ "unsaved_changes": dirty })))
    }
}

pub(crate) fn event(name: &str, payload: Value) -> String {
    json!({ "event": name, "payload": payload }).to_string()
}

/// The palette catalog changed — how a client that is already connected learns what `hello` would
/// have told it.
fn node_types_event(g: &Graph) -> String {
    event("node_types", json!({ "types": schemas::catalog_types(g) }))
}

/// A per-node `state_update` event carrying a node's current params and error.
fn param_state_update(g: &Graph, peer: Uid) -> String {
    param_state_update_refreshed(g, peer, &[])
}

/// As [`param_state_update`], naming the params whose ⟳ refresh just completed. It must be sent on
/// EVERY outcome, a refresh that found nothing included, or the button spins on.
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

/// A required string field from an RPC payload.
fn parse_str<'a>(payload: &'a Value, key: &str) -> Result<&'a str, String> {
    payload.get(key).and_then(|v| v.as_str()).ok_or_else(|| format!("missing {key}"))
}

/// A boundary wire's inner target: both halves named is a wire, both absent is an UNWIRE. Parsed
/// as ONE value, so the half-specified third state is unconstructible.
fn parse_inner(payload: &Value) -> Result<goofi_engine::subpatch::StubInner, String> {
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

/// The `params` bag `add_node` and `edit_node` share — `{group: {param: value | {value,
/// expression, mode, triggers}}}` — as one `EditParam` per entry.
fn parse_params_bag(g: &Graph, uid: Uid, params: &Value) -> Result<Vec<goofi_engine::Command>, String> {
    let groups = params.as_object().ok_or("params is {group: {param: …}}")?;
    let mut cmds = Vec::new();
    for (group, entries) in groups {
        let entries =
            entries.as_object().ok_or_else(|| format!("params.{group} is {{param: …}}"))?;
        for (name, spec) in entries {
            let existing = g
                .params(uid)
                .and_then(|p| goofi_node::param(&p, group, name).cloned())
                .ok_or_else(|| format!("no param {group}.{name}"))?;
            let cur = g.param_expression(uid, group, name);
            let (value, expr) = parse_param_entry(&existing, cur, spec)
                .map_err(|e| format!("params.{group}.{name}: {e}"))?;
            if value.is_none() && expr.is_none() {
                return Err(format!("params.{group}.{name} sets neither a value nor an expression"));
            }
            cmds.push(goofi_engine::Command::EditParam {
                uid,
                group: group.clone(),
                name: name.clone(),
                value,
                expr,
            });
        }
    }
    Ok(cmds)
}

/// A JSON merge patch applied in place: objects merge key by key, `null` deletes, anything else
/// replaces.
fn merge_json(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(t), Value::Object(p)) => {
            for (k, v) in p {
                if v.is_null() {
                    t.remove(k);
                } else {
                    merge_json(t.entry(k.clone()).or_insert(Value::Null), v);
                }
            }
        }
        (t, p) => *t = p.clone(),
    }
}

/// One `params.<group>.<name>` entry: a bare literal, or `{value, expression, mode, triggers}`.
/// No param type is an object, so the two forms cannot be confused. An expression given without a
/// mode turns the binding on, and a mode or trigger given alone edits the binding already there.
fn parse_param_entry(
    existing: &goofi_core::Param,
    cur: Option<goofi_engine::ExprInfo>,
    spec: &Value,
) -> Result<(Option<goofi_core::Param>, Option<goofi_engine::ExprState>), String> {
    let Some(o) = spec.as_object() else {
        return Ok((Some(goofi_engine::param_from_json(existing, spec, true)), None));
    };
    if let Some(k) = o.keys().find(|k| !matches!(k.as_str(), "value" | "expression" | "mode" | "triggers")) {
        return Err(format!("unknown field `{k}` — value, expression, mode, triggers"));
    }
    let value = o
        .get("value")
        .filter(|v| !v.is_null())
        .map(|v| goofi_engine::param_from_json(existing, v, true));
    let mode = match o.get("mode").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => match v.as_str() {
            Some("expression") => Some(true),
            Some("constant") => Some(false),
            _ => return Err(format!("mode is `constant` or `expression`, not {v}")),
        },
    };
    let source = o.get("expression").filter(|v| !v.is_null()).map(|v| {
        v.as_str().map(str::to_string).ok_or_else(|| format!("expression is a string, not {v}"))
    });
    let triggers = match o.get("triggers").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_bool().ok_or_else(|| format!("triggers is a bool, not {v}"))?),
    };
    let expr = match (source, mode, triggers) {
        (None, None, None) => None,
        (source, mode, triggers) => {
            // An expression given is an expression MEANT, so it binds without being told to; a mode
            // or a trigger alone edits whatever binding is already there.
            let default_enabled = match &source {
                Some(_) => true,
                None => cur.as_ref().is_some_and(|c| c.enabled),
            };
            let source = match source {
                Some(s) => s?,
                None => cur.as_ref().map(|c| c.source.clone()).unwrap_or_default(),
            };
            Some(goofi_engine::ExprState {
                // An empty source is an UNBIND, so it cannot end up enabled.
                enabled: mode.unwrap_or(default_enabled) && !source.is_empty(),
                triggers: triggers.unwrap_or_else(|| cur.as_ref().is_some_and(|c| c.triggers_process)),
                source,
            })
        }
    };
    Ok((value, expr))
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

/// Translate a link endpoint that names a sub-patch boundary port into the flat inner leaf it
/// resolves to, so every runtime and persisted link is leaf→leaf.
fn resolve_link_endpoint(g: &goofi_engine::Graph, uid: Uid, slot: &str) -> (Uid, String) {
    if g.scope(uid).is_some() {
        if let Some(leaf) = g.resolve_stub(uid, slot) {
            return leaf;
        }
    }
    (uid, slot.to_string())
}

/// Resolve a link endpoint AND refuse one that names nothing wirable — the check a caller-initiated
/// `add_link` gets and a REPLAY does not, since a replay must converge rather than wedge the stack.
fn wirable_endpoint(g: &Graph, uid: Uid, slot: &str, which: &str) -> Result<(Uid, String), String> {
    let (node, slot) = resolve_link_endpoint(g, uid, slot);
    if g.contains(node) {
        return Ok((node, slot));
    }
    if g.scope(uid).is_some() {
        return Err(format!(
            "add_link: `{which}` names sub-patch {} port `{slot}`, which exposes no inner slot — \
             wire_boundary it to a member's slot first",
            uid.to_hex()
        ));
    }
    Err(format!("add_link: `{which}` names no node in this patch: {}", uid.to_hex()))
}

/// Is `node` something a panel could bind to? A UID, and only a uid: a display name stops resolving
/// the moment somebody renames the node.
fn bindable_node(g: &Graph, node: &str) -> bool {
    Uid::from_hex(node).is_some_and(|u| g.contains(u) || g.scope(u).is_some())
}

/// Route a layout planner's per-entry writes through the command history as ONE undo step, and
/// answer with the arrangement they produced, drawn as `inspect_layout` draws it.
fn apply_layout(
    state: &AppState,
    g: &mut Graph,
    session: &str,
    cmd: goofi_engine::Command,
) -> Result<Value, String> {
    state.history.lock().unwrap().apply(g, session, cmd)?;
    Ok(json!({ "text": inspect::layout_tree(g.arrangement(), None) }))
}

impl AppState {
    /// Run one control op — the single entry point every surface shares. `session` scopes the undo
    /// history the way a browser tab's id does.
    pub fn call(&self, op: &str, payload: Value, session: &str) -> Result<Value, String> {
        let state = self;
        let (op, session) = (op.to_string(), session.to_string());
        let spec = ops::find(&op);
        let mut events: Vec<String> = Vec::new();
        let result: Result<Value, String> = (|| {
            if spec.is_none() {
                return Err(format!("unknown op `{op}`"));
            }
            // Ops that read no graph state are served WITHOUT the graph mutex: these two walk the
            // filesystem, which under the lock would stall the status-drain worker.
            if op == "list_dir" {
                return Ok(fsbrowse::list_dir(payload.get("path").and_then(|v| v.as_str())));
            }
        if op == "get_state" {
            return Ok(state.doc.lock().unwrap().to_json());
        }
        if op == "get_patch" {
                return Ok(json!({
                    "save_path": state.save_path(),
                    "workspace": goofi_core::path::to_slash(&state.mount()),
                    "dirty": state.is_dirty(),
                }));
            }
            // The harness ops touch no graph state either: they fork and signal children, and the
            // roster converges through `harness_changed` rather than by making a caller wait.
            if op == "list_harnesses" {
                state.harnesses.refresh_in_background(state.events.clone());
                return Ok(state.harnesses.roster());
            }
            if op == "spawn_harness" {
                let h = payload.get("harness").and_then(|v| v.as_str())
                    .ok_or("spawn_harness: missing harness")?;
                let id = state.harnesses.spawn(h, &state.mount(), &state.mcp_url(),
                                               &term::parent_env(), state.events.clone())?;
                events.push(event("harness_changed", state.harnesses.roster()));
                return Ok(json!({ "instance_id": id }));
            }
            if op == "stop_harness" {
                state.harnesses.stop(payload.get("instance").and_then(|v| v.as_str())
                    .ok_or("stop_harness: missing instance")?)?;
                events.push(event("harness_changed", state.harnesses.roster()));
                return Ok(json!({ "ok": true }));
            }
            // Several writes as ONE undo step. Taken before the graph lock, because each step is a
            // whole call: it locks, mirrors the document and broadcasts exactly as it would alone.
            if op == "compound" {
                let steps = payload
                    .get("ops")
                    .and_then(|v| v.as_array())
                    .ok_or("compound: `ops` is a list of {op, payload}")?
                    .clone();
                for (i, step) in steps.iter().enumerate() {
                    let name = step.get("op").and_then(|v| v.as_str());
                    let ok = name.and_then(ops::find).is_some_and(|o| {
                        o.writes && !matches!(o.name, "compound" | "undo" | "redo" | "load" | "load_text" | "new")
                    });
                    if !ok {
                        return Err(format!(
                            "compound: step {i} `{}` is not a step — a step is one undoable write",
                            name.unwrap_or("")
                        ));
                    }
                }
                // The redo run is cleared UP FRONT so no step's own clearing can shift the mark.
                let from = {
                    let mut h = state.history.lock().unwrap();
                    h.clear_redo(&session);
                    h.len()
                };
                let mut results = Vec::with_capacity(steps.len());
                for (i, step) in steps.iter().enumerate() {
                    let name = step["op"].as_str().unwrap_or_default().to_string();
                    let arg = step.get("payload").cloned().unwrap_or_else(|| json!({}));
                    match state.call(&name, arg, &session) {
                        Ok(r) => results.push(r),
                        Err(e) => {
                            // A compound is a UNIT, so a refused step takes back the ones that landed.
                            let mut g = state.graph.lock().unwrap();
                            state.history.lock().unwrap().rollback(&mut g, &session, from);
                            drop(g);
                            resync_and_broadcast(state);
                            return Err(format!("compound: step {i} `{name}` was refused: {e}"));
                        }
                    }
                }
                state.history.lock().unwrap().coalesce(&session, from);
                return Ok(json!({ "results": results }));
            }
            // `inspect_patch`'s header carries the same walk, so it is taken before the lock too.
            let dirty = op == "inspect_patch" && state.is_dirty();
            let mut g = state.graph.lock().unwrap();
            match op.as_str() {
                "list_nodes" => Ok(json!({ "types": schemas::catalog_types(&g) })),
                // Explicit, never watched: an agent calls it after writing a node file.
                "rescan_nodes" => {
                    let (diff, _) = rescan(state, &mut g, &state.mount());
                    restart_changed(&mut g, &diff);
                    events.push(node_types_event(&g));
                    Ok(json!({ "added": diff.added, "changed": diff.changed, "removed": diff.removed }))
                }
                "add_node" => {
                    let ty = payload
                        .get("type")
                        .and_then(|v| v.as_str())
                        .ok_or("add_node: missing type")?
                        .to_string();
                    // A CHOSEN uid and name, so a caller reconstructing a known graph keeps its
                    // uid-keyed bindings. Not the undo path, which is manager-owned.
                    let restore = payload.get("member_uid").and_then(|v| v.as_str()).and_then(Uid::from_hex);
                    let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let pos = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
                    // Never silently rooted on a bad `inst_id`: the canvas draws only the entered
                    // scope, so a rooted node would be invisible exactly where the user placed it.
                    let scope = match payload.get("inst_id").filter(|v| !v.is_null()) {
                        Some(v) => {
                            Some(v.as_str().and_then(Uid::from_hex).ok_or("add_node: malformed inst_id")?)
                        }
                        None => None,
                    };
                    // Inline params are applied AFTER: RemoveNode's inverse captures the LIVE node,
                    // so an undo→redo restores them without threading them through the command.
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
                    // Applied UNDER THE GRAPH LOCK, so the node is born configured before the
                    // resync mirrors it into the doc.
                    if let Some(params) = payload.get("params").filter(|v| !v.is_null()) {
                        for cmd in
                            parse_params_bag(&g, uid, params).map_err(|e| format!("add_node: {e}"))?
                        {
                            cmd.execute(&mut g).map_err(|e| format!("add_node: {e}"))?;
                        }
                    }
                    // A bare uid: the node itself arrives via the doc mirror.
                    events.push(event("node_added", json!({ "uid": uid.to_hex() })));
                    // The REPLY answers a caller with no doc replica: the minted name, the slots to
                    // wire, and the params as BORN.
                    let m = g.manifest(uid);
                    Ok(json!({
                        "uid": uid.to_hex(),
                        "name": g.name(uid).unwrap_or_default(),
                        "input_slots": m.map(schemas::input_slots).unwrap_or_else(|| json!({})),
                        "output_slots": m.map(schemas::output_slots).unwrap_or_else(|| json!({})),
                        "params": g.params(uid).map(|p| schemas::param_value_map(&p)).unwrap_or_else(|| json!({})),
                    }))
                }
                "remove_node" => {
                    let uid = parse_uid(&payload, "node")?;
                    // The command is idempotent, so a uid naming nothing succeeds; the reply says
                    // which of the two happened.
                    let existed = bindable_node(&g, &uid.to_hex());
                    state
                        .history
                        .lock()
                        .unwrap()
                        .apply(&mut g, &session, goofi_engine::Command::RemoveNode { uid })?;
                    Ok(json!({ "removed": existed }))
                }
                // Recovery, not an edit, so it is NOT routed through the command history: the client
                // records no `graph_cmd` for a restart and the two stacks must stay 1:1.
                "restart_node" => {
                    let uid = parse_uid(&payload, "node")?;
                    g.restart_node(uid)?;
                    // Pushed at once, so the red border lifts on the click rather than on the sweep.
                    events.push(param_state_update(&g, uid));
                    Ok(json!({ "ok": true }))
                }
                "add_link" => {
                    let (a, so, b, si) = parse_link(&payload)?;
                    let (a, so) = wirable_endpoint(&g, a, &so, "node_out")?;
                    let (b, si) = wirable_endpoint(&g, b, &si, "node_in")?;
                    state.history.lock().unwrap().apply(
                        &mut g,
                        &session,
                        goofi_engine::Command::AddLink {
                            node_out: a,
                            slot_out: so.clone(),
                            node_in: b,
                            slot_in: si.clone(),
                        },
                    )?;
                    // The wire AS MADE, not as named: a boundary endpoint resolves to its inner leaf,
                    // and the agreed dtype gates the next link to this output.
                    let dtype = vocab::output_slots(&g, a)
                        .into_iter()
                        .find(|(name, _)| *name == so)
                        .map(|(_, dtype)| dtype);
                    Ok(json!({
                        "node_out": a.to_hex(), "slot_out": so,
                        "node_in": b.to_hex(), "slot_in": si,
                        "dtype": dtype,
                    }))
                }
                "remove_link" => {
                    let (a, so, b, si) = parse_link(&payload)?;
                    let (a, so) = resolve_link_endpoint(&g, a, &so);
                    let (b, si) = resolve_link_endpoint(&g, b, &si);
                    // Idempotent for the same reason `remove_node` is, and answered the same way.
                    let existed = g.has_link(a, &so, b, &si);
                    state.history.lock().unwrap().apply(
                        &mut g,
                        &session,
                        goofi_engine::Command::RemoveLink { node_out: a, slot_out: so, node_in: b, slot_in: si },
                    )?;
                    Ok(json!({ "removed": existed }))
                }
                // NOT a command: options are runtime-only, so there is nothing to undo. They do not
                // ride this reply either — the hook runs on the node's own thread.
                "refresh_param" => {
                    let uid = parse_uid(&payload, "node")?;
                    let group = parse_str(&payload, "group")?.to_string();
                    let name = parse_str(&payload, "name")?.to_string();
                    g.refresh_param(uid, &group, &name)?;
                    Ok(json!({ "ok": true }))
                }
                "edit_node" => {
                    let uid = parse_uid(&payload, "node")?;
                    let name = payload.get("name").and_then(|v| v.as_str()).map(str::to_string);
                    // The rename command tolerates a collision as a no-op so a stale replay
                    // converges; the user-facing error therefore belongs here, at the forward RPC.
                    if let Some(n) = &name {
                        if g.name_taken(n, uid) {
                            return Err(format!("edit_node: the name `{n}` is taken"));
                        }
                    }
                    // A display name is spliced into expression SOURCE, so a quote or backslash
                    // would yield invalid Python in every referring node.
                    if name.as_deref().is_some_and(|n| n.contains(['\'', '"', '\\'])) {
                        return Err("edit_node: a name cannot contain a quote or backslash — it is \
                                    spliced into nd() expression source"
                            .into());
                    }
                    let pos = payload
                        .get("pos")
                        .filter(|v| !v.is_null())
                        .map(|v| parse_pos(v).ok_or("edit_node: pos is [x, y]"))
                        .transpose()?;
                    // Viewers MERGE key by key, so only the slots named move; the command then sets
                    // the whole blob, which is what makes its inverse exact. The PATCH is what is
                    // checked — a stale slot already stored is inert, and refusing it would block
                    // every later edit on a node whose file changed its slots.
                    let viewers = match payload.get("viewers").filter(|v| !v.is_null()) {
                        Some(patch) => {
                            vocab::check_viewers(&g, uid, patch)?;
                            let mut whole =
                                g.viewers(uid).cloned().ok_or("edit_node: no such node")?;
                            merge_json(&mut whole, patch);
                            Some(whole)
                        }
                        None => None,
                    };
                    let params = payload.get("params").filter(|v| !v.is_null());
                    if name.is_none() && pos.is_none() && viewers.is_none() && params.is_none() {
                        return Err("edit_node: give a name, pos, params or viewers".into());
                    }

                    // ONE command, so one undo step covers whatever the call carried: the node's own
                    // fields, then a param edit each.
                    let mut cmds = Vec::new();
                    if name.is_some() || pos.is_some() || viewers.is_some() {
                        cmds.push(goofi_engine::Command::EditNode { uid, name, pos, viewers });
                    }
                    let mut touched: Vec<(String, String)> = Vec::new();
                    if let Some(params) = params {
                        for cmd in parse_params_bag(&g, uid, params)
                            .map_err(|e| format!("edit_node: {e}"))?
                        {
                            if let goofi_engine::Command::EditParam { group, name, .. } = &cmd {
                                touched.push((group.clone(), name.clone()));
                            }
                            cmds.push(cmd);
                        }
                    }
                    let out = state.history.lock().unwrap().apply(
                        &mut g,
                        &session,
                        if cmds.len() == 1 { cmds.pop().unwrap() } else { goofi_engine::Command::Compound(cmds) },
                    )?;
                    // The runtime `expression_error` is doc-invisible, so echo the descriptors —
                    // for this node, and for every referrer a rename rewrote.
                    if !touched.is_empty() {
                        events.push(param_state_update(&g, uid));
                    }
                    if let goofi_engine::Outcome::Nodes(referrers) = out {
                        for r in referrers {
                            events.push(param_state_update(&g, r));
                        }
                    }
                    // Every param touched AS STORED: a literal is coerced to its declared type, and a
                    // binding that does not compile is stored WITH its error.
                    let mut out = serde_json::Map::new();
                    for (group, name) in touched {
                        let entry = json!({
                            "value": g.params(uid)
                                .and_then(|p| goofi_node::param(&p, &group, &name).cloned())
                                .map(|p| goofi_engine::param_value_json(&p, true)),
                            "error": g.param_expression(uid, &group, &name).and_then(|e| e.error),
                        });
                        out.entry(group).or_insert_with(|| json!({}))
                            .as_object_mut().unwrap()
                            .insert(name, entry);
                    }
                    Ok(json!({ "params": Value::Object(out) }))
                }
                // Where THIS client is looking: not a doc root, so it neither drags a peer nor raises
                // the unsaved dot, but it still rides the `.gfi` and `hello`.
                "set_viewpoint" => {
                    g.set_viewpoint(payload.get("viewpoint").cloned().unwrap_or(Value::Null));
                    Ok(json!({ "ok": true }))
                }

                "inspect_layout" => {
                    let tab = payload.get("tab").and_then(|v| v.as_str()).map(str::to_string);
                    Ok(json!({ "text": inspect::layout_tree(g.arrangement(), tab.as_deref()) }))
                }
                "add_tab" => {
                    let name = parse_str(&payload, "name")?.to_string();
                    let index = payload.get("index").and_then(|v| v.as_u64()).map(|i| i as usize);
                    let subtree = payload.get("subtree").and_then(|v| v.as_str()).map(str::to_string);
                    let (plan, tab) = g.arrangement().add_tab(&name, index, subtree.as_deref())?;
                    // A tab built AROUND an existing subtree is a MOVE, so its undo puts the subtree
                    // back; one born with a fresh panel has nothing to give back and inverts by closing.
                    match subtree.as_deref() {
                        Some(s) => {
                            let root = s.to_string();
                            let cmd = goofi_engine::Command::LayoutMove { plan: Some(plan), root, home: None };
                            apply_layout(state, &mut g, &session, cmd)?
                        }
                        None => {
                            let cmd = goofi_engine::Command::LayoutBirth { plan, born: tab.clone() };
                            apply_layout(state, &mut g, &session, cmd)?
                        }
                    };
                    // The root panel's id, which a caller cannot otherwise know.
                    let panel = g.arrangement().root_of(&tab).unwrap_or_default();
                    Ok(json!({ "tab": tab, "panel": panel }))
                }
                "remove_tab" => {
                    let tab = parse_str(&payload, "tab")?.to_string();
                    // Planned here only so a bad id answers teachably: `LayoutClose` re-plans it under
                    // this same lock, and DEGRADES rather than errors.
                    g.arrangement().remove_tab(&tab)?;
                    apply_layout(state, &mut g, &session, goofi_engine::Command::LayoutClose { born: tab })
                }
                "rename_tab" => {
                    let (tab, name) = (parse_str(&payload, "tab")?, parse_str(&payload, "name")?);
                    let writes = g.arrangement().rename_tab(tab, name)?;
                    // A name is CONTENTS: the strip index is the slot, and a peer may now hold it.
                    apply_layout(state, &mut g, &session, goofi_engine::Command::LayoutContents { writes })
                }
                "reorder_tab" => {
                    let tab = parse_str(&payload, "tab")?;
                    let to = payload.get("to_index").and_then(|v| v.as_u64()).ok_or("missing to_index")?;
                    // Planned here only so a bad id answers teachably; the command re-plans it under
                    // this same lock.
                    g.arrangement().reorder_tab(tab, to as usize)?;
                    let cmd = goofi_engine::Command::LayoutReorderTab {
                        tab: tab.to_string(),
                        to_index: to as usize,
                    };
                    apply_layout(state, &mut g, &session, cmd)
                }
                "split_panel" => {
                    let panel = parse_str(&payload, "panel")?.to_string();
                    let dir = payload.get("direction").and_then(|v| v.as_str()).unwrap_or("row");
                    let axis = goofi_engine::layout::Axis::parse(dir)
                        .ok_or("split_panel: direction is `row` or `column`")?;
                    let before = payload.get("place_before").and_then(|v| v.as_bool()).unwrap_or(false);
                    let ratio = payload.get("ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
                    let (plan, fresh) = g.arrangement().split_panel(&panel, axis, before, ratio)?;
                    let cmd = goofi_engine::Command::LayoutBirth { plan, born: fresh.clone() };
                    apply_layout(state, &mut g, &session, cmd)?;
                    // The uid, because a split births an EMPTY panel the caller must then fill.
                    Ok(json!(fresh))
                }
                "set_panel" => {
                    let panel = parse_str(&payload, "panel")?.to_string();
                    let ty = payload.get("type").and_then(|v| v.as_str()).map(str::to_string);
                    let panel_state = payload.get("state").cloned();
                    // A panel bound to a node that is not there renders empty and explains nothing.
                    let named = panel_state
                        .as_ref()
                        .and_then(|s| s.get("node"))
                        .and_then(|v| v.as_str())
                        .filter(|n| !n.is_empty());
                    if let Some(node) = named {
                        if !bindable_node(&g, node) {
                            return Err(format!("set_panel: no node `{node}` in this patch"));
                        }
                    }
                    // The slot is checked against the node this write LEAVES the panel bound to: its
                    // own, or the one already stored, since a state write merges.
                    let bound = named
                        .or_else(|| {
                            g.arrangement()
                                .panel_state(&panel)
                                .and_then(|s| s.get("node"))
                                .and_then(|v| v.as_str())
                        })
                        .and_then(Uid::from_hex);
                    vocab::check_panel(&g, ty.as_deref(), panel_state.as_ref(), bound)?;
                    let writes = g.arrangement().set_panel(&panel, ty.as_deref(), panel_state)?;
                    apply_layout(state, &mut g, &session, goofi_engine::Command::LayoutContents { writes })
                }
                "move_panel" => {
                    let panel = parse_str(&payload, "panel")?.to_string();
                    let dest = parse_str(&payload, "new_parent")?.to_string();
                    let at = payload.get("order_index").and_then(|v| v.as_u64()).unwrap_or(0);
                    let plan = g.arrangement().move_subtree(&panel, &dest, at as usize)?;
                    let cmd = goofi_engine::Command::LayoutMove { plan: Some(plan), root: panel.clone(), home: None };
                    apply_layout(state, &mut g, &session, cmd)
                }
                // ONE op per drag gesture: a drop is one undo step, and peers never see an
                // arrangement that was not on somebody's screen.
                "insert_at_panel" => {
                    let subtree = parse_str(&payload, "subtree")?.to_string();
                    let target = parse_str(&payload, "target")?.to_string();
                    let dir = payload.get("direction").and_then(|v| v.as_str()).unwrap_or("row");
                    let axis = goofi_engine::layout::Axis::parse(dir)
                        .ok_or("insert_at_panel: direction is `row` or `column`")?;
                    let before = payload.get("place_before").and_then(|v| v.as_bool()).unwrap_or(false);
                    let ratio = payload.get("ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
                    let plan =
                        g.arrangement().insert_at_panel(&subtree, &target, axis, before, ratio)?;
                    let cmd = goofi_engine::Command::LayoutMove { plan: Some(plan), root: subtree.clone(), home: None };
                    apply_layout(state, &mut g, &session, cmd)
                }
                "resize_split" => {
                    let split = parse_str(&payload, "split")?.to_string();
                    // A non-numeric entry becomes NaN, which the planner refuses beside a zero or a
                    // negative one — so "is this a fraction" is answered in one place.
                    let fractions: Vec<f64> = payload
                        .get("fractions")
                        .and_then(|v| v.as_array())
                        .ok_or("resize_split: missing fractions")?
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(f64::NAN))
                        .collect();
                    // Planned here only so a bad split or a wrong fraction count answers teachably;
                    // the command re-plans it under this same lock.
                    g.arrangement().resize_split(&split, &fractions)?;
                    let cmd = goofi_engine::Command::LayoutResizeSplit { split, fractions };
                    apply_layout(state, &mut g, &session, cmd)
                }
                "remove_panel" => {
                    let panel = parse_str(&payload, "panel")?.to_string();
                    // Planned only for its teachable refusal — see `remove_tab` above.
                    g.arrangement().remove_subtree(&panel)?;
                    apply_layout(state, &mut g, &session, goofi_engine::Command::LayoutClose { born: panel })
                }
                "set_global" => {
                    let name = parse_str(&payload, "name")?.to_string();
                    let held = g.globals().get(&name).map(goofi_engine::global_to_json);
                    // NO value is a delete, so removing a global is the absence of one rather than
                    // an op of its own.
                    let Some(val) = payload.get("value").filter(|v| !v.is_null()) else {
                        if held.is_none() {
                            return Err(format!("set_global: no such global `{name}`"));
                        }
                        state.history.lock().unwrap().apply(
                            &mut g,
                            &session,
                            goofi_engine::Command::EditGlobal { name, value: None, at: None },
                        )?;
                        return Ok(json!({ "removed": true }));
                    };
                    // Every expression reading a global depends on its TYPE, so re-typing one
                    // through a value edit would break the reference rather than the call.
                    let held_ty = held.as_ref().map(|h| h["type"].as_str().unwrap_or_default().to_string());
                    let ty = match (payload.get("type").and_then(|v| v.as_str()), &held_ty) {
                        (Some(t), Some(h)) if t != h => {
                            return Err(format!(
                                "set_global: `{name}` is a {h} — remove it and set it again to re-type it"
                            ))
                        }
                        (Some(t), _) => t.to_string(),
                        (None, Some(h)) => h.clone(),
                        (None, None) => {
                            return Err(format!("set_global: `{name}` is new — give its `type`"))
                        }
                    };
                    let value = goofi_engine::global_from_json(&json!({ "value": val, "type": ty }))
                        .ok_or_else(|| format!("set_global: `{val}` is not a {ty}"))?;
                    state.history.lock().unwrap().apply(
                        &mut g,
                        &session,
                        goofi_engine::Command::EditGlobal { name, value: Some(value.clone()), at: None },
                    )?;
                    // As STORED: the conversion is type-directed, so a fraction into an int rounds.
                    Ok(json!({ "value": goofi_engine::global_to_json(&value)["value"] }))
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
                "inspect_patch" => {
                    let scope = match payload.get("scope").filter(|v| !v.is_null()) {
                        Some(v) => {
                            Some(v.as_str().and_then(Uid::from_hex).ok_or("inspect_patch: malformed scope")?)
                        }
                        None => None,
                    };
                    let workspace = state.mount();
                    let text =
                        inspect::patch(&g, scope, state.save_path().as_deref(), &goofi_core::path::to_slash(&workspace), dirty)?;
                    Ok(json!({ "text": text }))
                }
                "inspect_node" => {
                    let uid = parse_uid(&payload, "node")?;
                    let want = |k: &str| payload.get(k).and_then(|v| v.as_bool()).unwrap_or(true);
                    let slot = payload.get("slot").and_then(|v| v.as_str());
                    let text = inspect::node(&g, uid, slot, want("params"), want("error"))?;
                    Ok(json!({ "text": text }))
                }
                "list_globals" => Ok(inspect::globals(&g)),
                "read_node_source" => {
                    // `.rev()` is load-bearing: `rescan` scans the shipped list forwards and lets each
                    // directory overwrite the last, so a first-match search must walk it backwards.
                    let dirs: Vec<(PathBuf, &str)> = [(state.mount().join("nodes"), "patch")]
                        .into_iter()
                        .chain(state.system_nodes.iter().rev().map(|d| (d.clone(), "shipped")))
                        .collect();
                    inspect::node_source(&g, parse_str(&payload, "type")?, &dirs)
                }
                "serialize" => Ok(json!({ "yaml": g.serialize() })),
                // The mount is a per-run temp directory under a random name, so asking is the only
                // way a client or a harness can find it.
                "open_workspace" => Ok(json!({ "path": goofi_core::path::to_slash(&state.mount()) })),
                "save" => {
                    // Expand `~` exactly as the browser does — the two must agree on what a path
                    // means. A save writes a file or it is malformed.
                    let path = payload
                        .get("path")
                        .and_then(|v| v.as_str())
                        .map(fsbrowse::resolve)
                        .ok_or("save: missing path")?;
                    let mount = state.mount();
                    // Sampled BEFORE the pack: baselining after would call a file written during the
                    // zip packed either way, which is the direction that LOSES an edit.
                    let packed = goofi_engine::archive::fingerprint(&mount);
                    save_archive(std::path::Path::new(&path), &g.serialize(), &mount)?;
                    // Announced UNCONDITIONALLY, not on the flag's transition: a patch dirtied solely
                    // by a file in the mount leaves the flag already false, so no transition comes.
                    *state.workspace_baseline.lock().unwrap() = packed;
                    state.set_dirty(false);
                    events.push(event("unsaved_changes", json!({ "unsaved_changes": false })));
                    // The patch's home, stored ONLY on success and announced as well as stored: an
                    // already-connected peer gets no new snapshot to read it from.
                    *state.save_path.lock().unwrap() = Some(path.clone());
                    events.push(event("save_path_changed", json!({ "save_path": &path })));
                    Ok(json!({ "path": path }))
                }
                // One arm for every source, so nothing after the read can drift between them.
                "load_text" | "load" | "new" => {
                    // Every source mounts FRESH, and the live mount is swapped only once the manifest
                    // has parsed, so a refused load leaves the open patch untouched on both planes.
                    let fresh = new_mount();
                    let (content, from_path) =
                        stage_load(&fresh, &op, &payload).inspect_err(|_| remove_mount(&fresh))?;
                    // ORDER is load-bearing: the types the patch SHIPS are registered before the
                    // manifest resolves, or the unknown-type gate fires on the nodes the archive brought.
                    rescan(state, &mut g, &fresh);
                    // Parse BEFORE anything is announced or committed.
                    if let Err(e) = g.load_doc(&content) {
                        // Refused, so the registry the scan above swapped is re-derived from the mount
                        // that is still live.
                        rescan(state, &mut g, &state.mount());
                        remove_mount(&fresh);
                        return Err(e);
                    }
                    // Commit, now that nothing left can fail: the loaded patch's workspace becomes the
                    // live one, and the replaced mount goes with the harnesses spawned into it.
                    let replaced = std::mem::replace(&mut *state.mount.lock().unwrap(), fresh);
                    state.retire_mount(&replaced);
                    events.push(event("harness_changed", state.harnesses.roster()));
                    // `read_gfi` restores no mtimes, so without a baseline taken HERE a patch would be
                    // dirty from the moment it finished loading.
                    *state.workspace_baseline.lock().unwrap() = goofi_engine::archive::fingerprint(&state.mount());
                    // A load fully resets the session: there is nothing to undo across it.
                    state.history.lock().unwrap().clear();
                    events.extend(state.set_dirty(false));
                    // NONE for `load_text` and `new`, neither of which has a file behind it: an
                    // inherited path would aim the next silent Save at an unrelated `.gfi`.
                    *state.save_path.lock().unwrap() = from_path.clone();
                    events.push(event(
                        "graph_replaced",
                        schemas::snapshot(&g, &state.instance_id, false, false, from_path.as_deref(),
                                          state.harnesses.roster()),
                    ));
                    // The patch brought its own node types, which `graph_replaced` does not carry.
                    events.push(node_types_event(&g));
                    if let Some(path) = from_path {
                        events.push(event("save_path_changed", json!({ "save_path": path })));
                    }
                    // A stored arrangement this model admits but cannot render falls back to the
                    // default, so the reply says so rather than leaving the change unexplained.
                    Ok(json!({ "ok": true, "layout_warning": g.arrangement_warning() }))
                }
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

        // Gated on whether the op COULD have mutated the graph, not on `events`: a link or boundary
        // write mutates the doc-read graph while emitting no client event.
        let read_only = spec.is_some_and(|o| !o.writes);
        if result.is_ok() && !read_only {
            resync_and_broadcast(state);
            // These need the re-mirror but are not EDITS, so they must not raise the unsaved dot:
            // a load clears the flag in its own arm, and the other three are recovery or runtime.
            match op.as_str() {
                "load" | "load_text" | "new" => events.extend(state.set_dirty(false)),
                "set_viewpoint" => {}
                "restart_node" | "refresh_param" | "rescan_nodes" => {}
                _ => events.extend(state.set_dirty(true)),
            }
        }

        for e in events {
            let _ = state.events.send(e);
        }

        result
    }
}

/// The `/control` envelope over [`AppState::call`]: `{id, op, payload, session}` in, `{id, result}`
/// or `{id, error}` out. A request with no numeric `id` wants no reply.
fn dispatch(state: &AppState, text: &str) -> Option<String> {
    let req: Value = serde_json::from_str(text).ok()?;
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let op = req.get("op")?.as_str()?.to_string();
    let payload = req.get("payload").cloned().unwrap_or_else(|| json!({}));
    // Absent ⇒ one shared "default" session, so a client that presents none still works.
    let session = req.get("session").and_then(|v| v.as_str()).unwrap_or("default").to_string();

    let result = state.call(&op, payload, &session);
    match id {
        Value::Number(_) => Some(match result {
            Ok(r) => json!({ "id": id, "result": r }).to_string(),
            Err(e) => json!({ "id": id, "error": e }).to_string(),
        }),
        _ => None,
    }
}

/// Re-project the already-locked graph into the already-locked document and broadcast the delta.
/// The caller holds `graph` then `doc` — the canonical order — which keeps apply→re-project atomic.
fn remirror_and_broadcast_locked(state: &AppState, g: &Graph, doc: &mut crate::doc::GraphDoc) {
    let from = doc.version();
    let Some(patch) = doc.reconcile_root(&projection::of(g)) else { return };
    let _ = state
        .events
        .send(event("doc_patch", json!({ "from": from, "v": doc.version(), "patch": patch })));
}

/// The whole document as an event — what seeds a fresh connection, and what recovers a lagged one.
fn doc_state(state: &AppState) -> String {
    let doc = state.doc.lock().unwrap();
    event("doc_state", json!({ "v": doc.version(), "doc": doc.to_json() }))
}

/// Re-project the authoritative graph into the document and broadcast the delta, after an RPC
/// mutates the graph. The projection is built WHOLE, so a stale leaf converges too.
fn resync_and_broadcast(state: &AppState) {
    let g = state.graph.lock().unwrap();
    let mut doc = state.doc.lock().unwrap();
    remirror_and_broadcast_locked(state, &g, &mut doc);
}

async fn term_ws(
    Path(instance): Path<String>,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_term(socket, state, instance))
}

/// The inband control a `/term` client sends; resize is the only one there is.
#[derive(serde::Deserialize)]
struct TermControl {
    op: String,
    cols: u16,
    rows: u16,
}

/// One `/term` socket: binary frames are PTY bytes in both directions, text frames are JSON control
/// — `{op:"resize", cols, rows}` inbound, `{op:"size", cols, rows}` and `{exit_code}` outbound.
///
/// A resize is a PROPOSAL: [`term::Sizes`] arbitrates and the answer is broadcast to every view,
/// the one that asked included. A view that says `0` retracts; closing the socket leaves the seat.
async fn handle_term(socket: WebSocket, state: AppState, instance: String) {
    let (mut tx, mut rx) = socket.split();
    let Some(inst) = state.harnesses.get(&instance) else {
        let _ = tx.send(close(4004, "unknown harness instance")).await;
        return;
    };
    // Taken before the first await: a subscription made later would miss what the child wrote.
    let (mut output, mut exit, mut eof) = inst.attach();
    let (seat, mut size) = inst.join();
    // Sent up front: a view arriving on a settled size has no change event coming to tell it.
    let settled = *size.borrow_and_update();
    if let Some((cols, rows)) = settled {
        let _ = tx.send(size_frame(cols, rows)).await;
    }
    let mut reaped_at: Option<tokio::time::Instant> = None;
    loop {
        // The exit frame is the LAST thing this socket sends: `child.wait()` returns while a dying
        // harness's final words are still in flight. Copied out of the guards, which are not `Send`.
        let (ended, drained) = (*exit.borrow_and_update(), *eof.borrow_and_update());
        if ended.is_some() && reaped_at.is_none() {
            reaped_at = Some(tokio::time::Instant::now());
        }
        // The wait is BOUNDED because ConPTY keeps the pseudoconsole open after the child exits, so
        // `drained` never turns true on Windows. Where EOF is real it wins and the settle never runs.
        let may_announce = drained || reaped_at.is_some_and(|t| t.elapsed() >= EXIT_SETTLE);
        if let (Some(code), true) = (ended, may_announce && output.is_empty()) {
            let _ = tx.send(Message::Text(json!({ "exit_code": code }).to_string().into())).await;
            break;
        }
        tokio::select! {
            incoming = rx.next() => match incoming {
                Some(Ok(Message::Binary(b))) => inst.write(&b),
                Some(Ok(Message::Text(t))) => {
                    if let Ok(c) = serde_json::from_str::<TermControl>(t.as_str()) {
                        if c.op == "resize" {
                            let some = c.cols > 0 && c.rows > 0;
                            inst.propose(seat, some.then_some((c.cols, c.rows)));
                        }
                    }
                }
                Some(Ok(_)) => {}
                Some(Err(_)) | None => break,
            },
            bytes = output.recv() => match bytes {
                Ok(b) => {
                    if tx.send(Message::Binary(b.into())).await.is_err() {
                        break;
                    }
                }
                // A viewer that fell behind loses bytes rather than the CHILD stalling behind it.
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            },
            _ = size.changed() => {
                // Copied out of the guard before the await: a `watch` borrow is not `Send`.
                let now = *size.borrow_and_update();
                if let Some((cols, rows)) = now {
                    if tx.send(size_frame(cols, rows)).await.is_err() {
                        break;
                    }
                }
            }
            _ = exit.changed() => {}
            _ = eof.changed() => {}
            // Wake to re-check the settle above, armed only once the child has been reaped.
            _ = tokio::time::sleep_until(reaped_at.unwrap_or_else(tokio::time::Instant::now) + EXIT_SETTLE),
                if reaped_at.is_some() && !drained => {}
        }
    }
    inst.leave(seat);
}

fn size_frame(cols: u16, rows: u16) -> Message {
    Message::Text(json!({ "op": "size", "cols": cols, "rows": rows }).to_string().into())
}

async fn data_ws(
    Path((node, slot)): Path<(String, String)>,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_data(socket, state, node, slot))
}

/// The inband `{op:"view", specs:[…]}` a viewer sends to declare what it can draw. Latest-wins.
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

/// Write one message to a `/data` socket, giving up after `bound`. The bound is not a policy about
/// slow viewers: an `.await` in a select branch body runs to completion, parking the keepalive beat.
async fn send_bounded<S>(tx: &mut S, msg: Message, bound: Duration) -> SendOutcome
where
    S: futures_util::Sink<Message> + Unpin,
{
    // At most one message stays buffered: `poll_ready` gates the next write on the same flush.
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
    /// A ping is outstanding but still inside the deadline.
    Wait,
    /// The deadline lapsed unanswered: the peer is gone.
    Dead,
}

/// Whether the peer on one `/data` socket has shown, within the deadline, that its receive path is
/// moving. A stalled WRITE is not evidence of death — a slow phone stalls writes too.
struct PeerLiveness {
    cfg: DataLiveness,
    /// When the oldest un-answered probe was sent; `None` while the peer is known to be moving.
    awaiting_pong_since: Option<std::time::Instant>,
}

impl PeerLiveness {
    fn new(cfg: DataLiveness) -> PeerLiveness {
        PeerLiveness { cfg, awaiting_pong_since: None }
    }

    /// The verdict for this beat. A probe is marked outstanding on the ATTEMPT, not on a successful
    /// write: a jammed peer cannot be pinged at all, which is what the deadline exists to catch.
    fn beat(&mut self, now: std::time::Instant) -> Beat {
        match self.awaiting_pong_since {
            // From the OLDEST unanswered probe, so beating faster than the deadline — the normal
            // case — cannot keep postponing the verdict.
            Some(sent) if now.duration_since(sent) >= self.cfg.pong_deadline => Beat::Dead,
            Some(_) => Beat::Wait,
            None => {
                self.awaiting_pong_since = Some(now);
                Beat::Ping
            }
        }
    }

    /// The peer answered. The ONLY thing that keeps a connection alive: neither a sent probe nor a
    /// flushed frame is proof of life, because a socket buffer swallows frames until it is full.
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
    // Exactly one physical leaf slot is streamed, so a stub viewer and an inner-scope viewer
    // coalesce onto the same reducer.
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

    // The SHARED per-slot reducer: this connection forwards its frames and pushes its own
    // ViewSpecs into the union.
    let key: reducer::SlotKey = (stream_uid, stream_slot);
    let conn = state.reducers.new_conn();
    let mut frames = state.reducers.subscribe(key.clone(), conn);

    // A dead-but-not-closed peer produces NO socket error, so without an active probe this
    // connection would live forever.
    let cfg = state.data_liveness;
    let mut live = PeerLiveness::new(cfg);
    let mut keepalive = tokio::time::interval(cfg.ping_interval);

    loop {
        tokio::select! {
            frame = frames.recv() => match frame {
                Ok(bytes) => {
                    // Giving up on a frame is NOT a liveness signal in either direction: only the
                    // pong decides.
                    match send_bounded(&mut tx, Message::Binary(bytes), cfg.send_timeout).await {
                        SendOutcome::Sent => {}
                        // The sweep will not resend an unchanged frame, so a dropped one must be
                        // asked for again.
                        SendOutcome::Dropped => state.reducers.reoffer(&key),
                        SendOutcome::Gone => break, // the socket really is gone
                    }
                }
                // A lagged viewer drops frames rather than stalling the shared reducer, and
                // re-offers because the missed frame may have been the last.
                Err(broadcast::error::RecvError::Lagged(_)) => state.reducers.reoffer(&key),
                Err(broadcast::error::RecvError::Closed) => break,
            },
            incoming = rx.next() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                Some(Ok(Message::Text(t))) => {
                    if let Ok(m) = serde_json::from_str::<ViewMsg>(t.as_str()) {
                        if m.op == "view" {
                            state.reducers.set_specs(&key, conn, m.specs);
                        }
                    }
                }
                Some(Ok(Message::Pong(_))) => live.pong(),
                _ => {}
            },
            // The bounded send above stops the loop parking on a BACKED-UP peer; this catches an
            // IDLE dead one, where no frames means no send and a write timeout never fires.
            _ = keepalive.tick() => match live.beat(std::time::Instant::now()) {
                Beat::Ping => {
                    if send_bounded(&mut tx, Message::Ping(Default::default()), cfg.send_timeout).await
                        == SendOutcome::Gone
                    {
                        break;
                    }
                }
                Beat::Wait => {}
                Beat::Dead => break,
            },
        }
    }
    state.reducers.unsubscribe(&key, conn);
}

// The two blocks below are the deliberate exception to "the suite lives in goofi-tests": each
// drives a PRIVATE state machine no external test can name.
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
        let t0 = Instant::now();
        let mut live = PeerLiveness::new(cfg());
        assert_eq!(live.beat(t0), Beat::Ping, "an unprobed peer is pinged");
        assert_eq!(live.beat(t0 + Duration::from_millis(100)), Beat::Wait, "the probe stands");
        assert_eq!(live.beat(t0 + Duration::from_millis(299)), Beat::Wait, "still inside deadline");
    }

    #[test]
    fn a_probe_unanswered_past_the_deadline_declares_the_peer_dead() {
        // The dead-but-not-closed case: nothing errored and nothing closed.
        let t0 = Instant::now();
        let mut live = PeerLiveness::new(cfg());
        assert_eq!(live.beat(t0), Beat::Ping);
        assert_eq!(live.beat(t0 + Duration::from_millis(300)), Beat::Dead, "deadline lapsed");
    }

    #[test]
    fn a_pong_clears_the_deadline_so_an_alive_peer_is_never_declared_dead() {
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
        // A late answer is credited in full: the clock restarts from the next probe.
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
        // A `Wait` beat must not refresh the clock, or a fast-beaten peer never expires.
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

    /// A sink that never becomes ready — a peer whose TCP window stopped draining.
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
