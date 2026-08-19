//! The axum server: `/control` (JSON RPC + broadcast events, doc state and doc deltas among them),
//! `/data/<node>/<slot>` (ONE reduced GOOF stream per slot, whatever the viewer count — the kind
//! is not in the path, since viewers publish their ViewSpec inband), `/term`, `/mcp`, and the SPA
//! compiled into the binary.

/// The control-plane document and its deltas — shape-agnostic. `pub` because the transport tests
/// drive a real replica through the real wire.
pub mod doc;
mod projection;
mod fsbrowse;
mod inspect;
mod mcp;
pub mod ops;
mod origin;
mod patchfile;
mod proc;
// Public because `AppState::reducers` is: the field's type has to be nameable by anything that
// reads it.
pub mod reducer;
mod schemas;
pub mod term;
pub mod vocab;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a `/term` socket waits for the PTY's end-of-stream after the child is reaped. ConPTY
/// keeps its pseudoconsole open past the child's death, so on Windows that end never comes; this
/// bounds the wait without branching on the platform.
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

/// The built SPA as it ships: a URL path and its bytes, compiled into the binary. Empty when the
/// crate was built without a frontend tree beside it.
pub type Spa = &'static [(&'static str, &'static [u8])];
include!(concat!(env!("OUT_DIR"), "/spa.rs"));

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<Mutex<Graph>>,
    pub events: broadcast::Sender<String>,
    pub instance_id: Arc<str>,
    /// The control-plane document every client replicates, re-projected from the graph after each
    /// successful op. Its deltas ride the `events` channel, so a replica sees them in the same
    /// order as everything else the manager says.
    pub doc: Arc<Mutex<crate::doc::GraphDoc>>,
    /// Whether the patch has been mutated since it was last saved or loaded — the title-bar dot
    /// and the unload guard. DERIVED, not stored: nothing persists it, so a fresh session starts
    /// clean and every successful mutating op sets it.
    dirty: Arc<std::sync::atomic::AtomicBool>,
    /// Shared per-slot data reducers (thalamus G1/G2): one reduction per active (node, slot),
    /// fanned out to every viewer, so N tabs on one slot cost one reduce+encode, not N.
    pub reducers: reducer::SlotReducers,
    /// The single central per-session command history (unified-command API). A command-backed op
    /// applies through here (recording its inverse tagged with the caller's session); `undo`/`redo`
    /// replay the inverse/forward for that session. Locked AFTER `graph`, BEFORE `doc`.
    pub history: Arc<Mutex<goofi_engine::CommandHistory>>,
    /// Liveness policy for `/data` sockets. Injectable so a test need not sit through a
    /// production-length deadline.
    pub data_liveness: DataLiveness,
    /// How a directory of node files becomes registered node types. Injected by the CLI at boot
    /// (see [`NodeScan`]); the default discovers nothing.
    pub scan_nodes: NodeScan,
    /// The shipped node directories — `nodes/`, then every `--extra-nodes`. Empty when the binary
    /// found neither. Boot-time config, set alongside the seam.
    pub system_nodes: Vec<PathBuf>,
    /// What the last scan found, by type name → the file's stamp. The baseline the next [`rescan`]
    /// diffs against, and the list it removes from — so a type registered some other way (a
    /// test's direct registration) is never swept up by a rescan of these two trees.
    node_index: Arc<Mutex<std::collections::BTreeMap<String, Option<Stamp>>>>,
    /// The tree a `.gfi` packs and unpacks. Dropped on a graceful exit; after a crash it stays,
    /// because a reboot clears the system temp directory. Shared and private because a LOAD
    /// replaces it while every handler holds its own clone of the state.
    mount: Arc<Mutex<PathBuf>>,
    /// What the workspace looked like when it was last packed into a `.gfi` or unpacked from one —
    /// the fingerprint [`AppState::is_dirty`] compares the live mount against. Re-taken at BOTH
    /// ends; the load end is the one that is easy to miss, and the `load` arm says why.
    workspace_baseline: Arc<Mutex<std::collections::BTreeMap<PathBuf, (u64, std::time::SystemTime)>>>,
    /// Where the open patch lives on disk — `None` until it is saved somewhere or loaded from
    /// somewhere. MANAGER-owned (C38) rather than remembered per tab: it rides the snapshot every
    /// client connects with, so a tab that opens later and a tab that never pressed Save name the
    /// same file as the one that did.
    save_path: Arc<Mutex<Option<String>>>,
    /// The port a spawned harness reaches this server's MCP surface on. Only the PORT: a harness
    /// is a CHILD of this process, so `127.0.0.1` is right whatever `--bind` says — and
    /// `--bind 0.0.0.0` would mint a URL no client can connect to.
    mcp_port: Arc<std::sync::atomic::AtomicU16>,
    /// The spawned agent harnesses, their PTYs, and the detection cache. See [`term`].
    pub harnesses: Arc<term::Harnesses>,
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
        // Project the INITIAL graph (no nodes, but the seeded system globals) so a client that
        // connects to a fresh backend has the current state at once — `default_ufreq` among it —
        // rather than a blank document that stays blank until the first mutation.
        let graph_val = Graph::new();
        let mut doc = crate::doc::GraphDoc::new();
        doc.reconcile_root(&projection::of(&graph_val));
        let graph = Arc::new(Mutex::new(graph_val));
        let reducers = reducer::SlotReducers::new(graph.clone());
        // The baseline is the fingerprint of whatever mount the patch owns — stated that way even
        // at boot, so the invariant has one spelling everywhere. Seeded BEFORE it is taken, or the
        // patch would be dirty from the moment it booted, having written the seed itself.
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

    /// Point a spawned harness's MCP config at the port this server actually bound. Called by the
    /// CLI after the bind, since the port is only known there (and may be 0-assigned).
    pub fn set_mcp_port(&self, port: u16) {
        self.mcp_port.store(port, std::sync::atomic::Ordering::Relaxed);
    }

    /// The base URL a spawned harness reaches this server's MCP surface at — see [`mcp_port`].
    ///
    /// [`mcp_port`]: AppState::mcp_port
    fn mcp_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.mcp_port.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Where the open patch lives on disk, if anywhere. Copied out for the same reason as
    /// [`AppState::mount`]: the lock guards the swap, not the reader.
    fn save_path(&self) -> Option<String> {
        self.save_path.lock().unwrap().clone()
    }

    /// Where the open patch's workspace files live *right now*. Copied out rather than borrowed:
    /// the lock guards only the swap, and no filesystem walk may run while holding it.
    pub fn mount(&self) -> PathBuf {
        self.mount.lock().unwrap().clone()
    }

    /// Drop the workspace mount, nonce directory and all. Safe to delete a PARENT because the
    /// field is private and both its writers store a `new_mount()` result. Best-effort: a failure
    /// leaves one directory in the system temp dir, which a reboot clears.
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
pub(crate) fn nonce_hex() -> String {
    let mut nonce = [0u8; 16];
    getrandom::fill(&mut nonce).expect("the OS random source");
    format!("{:032x}", u128::from_be_bytes(nonce))
}

/// Pack the patch to `target`: `manifest` beside the live workspace `mount`.
///
/// To a temp sibling and RENAMED onto the target, so a write that dies part-way leaves the previous
/// `.gfi` standing — a multi-entry zip truncates the old file first and has many chances to fail.
///
/// Under the graph lock the caller already holds, so every edit stalls for the duration. Accepted:
/// taking it off-lock means guarding a race that only exists once it is off-lock.
pub fn save_archive(target: &std::path::Path, manifest: &str, mount: &std::path::Path) -> Result<(), String> {
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

/// The front half of a load, against a mount that is not yet live. It stops AT the manifest rather
/// than applying it, because the patch's own node types live in the tree it just unpacked and have
/// to be registered before `load_doc` resolves the graph.
fn stage_load(
    mount: &std::path::Path,
    op: &str,
    payload: &Value,
) -> Result<(String, Option<String>), String> {
    let (content, from_path) = if op == "new" {
        // A New patch IS a load, of an empty patch from nowhere — which is what stops the two
        // drifting: `clear` keeps the editor layout, and that is how New used to inherit the
        // previous patch's panels. The live `g` is reused, so the catalog survives.
        (Graph::new().serialize(), None)
    } else if op == "load" {
        // Expand `~` exactly as the browser does — the two must agree on what a path means.
        let path =
            fsbrowse::resolve(payload.get("path").and_then(|v| v.as_str()).ok_or("load: missing path")?);
        let manifest = goofi_engine::archive::read_gfi(std::path::Path::new(&path), mount)
            .map_err(|e| format!("load failed: {e}"))?;
        // Whether this file becomes the patch's home — the target a later silent Save overwrites.
        // `/patch.gfi` declines: a browser upload was staged to a temp file that is deleted the
        // moment this returns, so adopting it would aim Ctrl-S at a path that no longer exists.
        // The patch's real home is on the USER's machine, which this process cannot name.
        let adopt = payload.get("adopt").and_then(Value::as_bool).unwrap_or(true);
        (manifest, adopt.then_some(path))
    } else {
        let content =
            payload.get("content").and_then(|v| v.as_str()).ok_or("load_text: missing content")?;
        (content.to_string(), None)
    };
    // A workspace goofi minted empty is a NEW one, and it is the only kind it initialises: `new`
    // and a browser upload leave `mount` exactly the empty directory the caller made, while `load`
    // has just unpacked the patch's OWN workspace into it — orientation included, edits and all,
    // or none at all if the patch predates seeding. goofi does not write into someone's patch.
    if op != "load" {
        term::seed_orientation(mount);
    }
    Ok((content, from_path))
}

/// The API routes, unguarded. Private, and the ONLY caller is [`app`] — the [`origin`] layer is
/// what a route must not be able to be added outside of, and `Router::layer` wraps only what is
/// already on the router when it is applied.
fn routes(state: AppState) -> Router {
    Router::new()
        .route("/control", any(control_ws))
        // One stream per (node, slot) — the kind segment is gone; a single reduced stream
        // serves every viewer kind. Each connection sends its viewers' ViewSpecs inband.
        .route("/data/{node}/{slot}", any(data_ws))
        // ONE MCP server per goofi instance, so it lives here rather than in a client-spawned
        // sidecar; `post` alone, so axum answers the retired GET and DELETE with their 405.
        // `/patch.gfi` is the patch as a FILE, the one door onto locations no mount reaches.
        // Its body limit is lifted: axum caps at 2 MB and a patch with a workspace is routinely
        // larger, and a `.gfi` is the user's own file on a single-user local app.
        .route(
            "/patch.gfi",
            get(patchfile::download)
                .post(patchfile::upload)
                .layer(axum::extract::DefaultBodyLimit::disable()),
        )
        .route("/mcp", post(mcp::endpoint))
        // …and one address per spawned harness, minted by `spawn_harness` and written into that
        // harness's own config. Identity is the ROUTE, so there is no id to spoof and none to
        // validate: `stop_harness` drops the instance and the address stops serving with it.
        .route("/mcp/{instance}", post(mcp::instance_endpoint))
        // A spawned harness's terminal: binary frames are PTY bytes, text frames JSON control.
        .route("/term/{instance}", any(term_ws))
        .with_state(state)
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

/// How often the worker takes what the nodes reported — deliberately NOT the event rate. Draining
/// is the RUNTIME's clock: it makes a node addressable and advances every wire's three-phase
/// sequence, one phase per ack. At `hz`, a link would be three broadcast periods from its first
/// frame.
const DRAIN_PERIOD: Duration = Duration::from_millis(1);

/// **The status-drain worker**: take every node's reports, apply them to the graph, and broadcast
/// the events that carry them — `node_stats`, `param_values`, `error`, `node_stage` — at `hz`.
///
/// Two obligations bind it. Never `set_dirty(true)`: a node reporting its own state is not a user
/// edit. And always FORGET a uid on removal, so a stale error cannot outlive its node or suppress
/// a re-created uid's first report.
///
/// The transition push is the identity-only `error` event, never a full-params snapshot: a late one
/// would overwrite a fresher `expression_error`, and every param of every node twice a second is
/// bandwidth nothing reads. `param_values` is safe where that is not — it carries only EVALUATED
/// values, which are never user-editable literals, so there is no concurrent edit to clobber.
pub fn spawn_stats(graph: Arc<Mutex<Graph>>, events: broadcast::Sender<String>, hz: u64) {
    std::thread::spawn(move || {
        let period = Duration::from_secs_f64(1.0 / hz as f64);
        let mut last_errors: HashMap<String, Option<String>> = HashMap::new();
        // A node's stage now changes on its own thread, with no RPC to ride on — the same reason
        // the error transition is pushed from here.
        let mut last_stages: HashMap<String, &'static str> = HashMap::new();
        let mut next_broadcast = Instant::now() + period;
        loop {
            std::thread::sleep(DRAIN_PERIOD);
            let due = Instant::now() >= next_broadcast;
            let collected = {
                let mut g = graph.lock().unwrap();
                // THE SOURCE (§6.2). Every field read below is filed by `apply_status` from a
                // report the node published; nothing here polls the node itself. A drain that
                // applied nothing still costs one non-blocking receive per node.
                g.drain_status();
                if !due {
                    None
                } else {
                    // Options are the one thing a node reports that the CRDT doc has no field for,
                    // so they cannot be recovered from a re-mirror — the node has to name the
                    // params it re-enumerated, and this is the only echo that reaches the client.
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
            // From NOW, not from the deadline just passed: a worker held off the lock owes no
            // burst of catch-up broadcasts over state it has only just read.
            next_broadcast = Instant::now() + period;
            for ev in refreshed {
                let _ = events.send(ev);
            }
            // Diff + build payloads after releasing the lock (both inputs are owned).
            let changed = error_transitions(&errs, &mut last_errors);
            for (node, ufreq) in rates {
                // Only send what we actually measure — no fabricated process-time or
                // run-count placeholders (the frontend treats those as optional).
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

/// The API router, with no SPA — `app(state, None)`, kept as a name because that is what it reads
/// as at a call site.
pub fn router(state: AppState) -> Router {
    app(state, &[])
}

/// The full router, optionally serving the SPA on the fallback.
///
/// The [`origin`] guard goes on LAST, so it wraps the API routes, the SPA and the fallback alike —
/// including the three WebSocket upgrades, which a CORS-based defence would miss entirely.
pub fn app(state: AppState, spa: Spa) -> Router {
    let base = routes(state);
    let served = if spa.is_empty() { base } else { base.fallback(serve_spa_file) };
    served.layer(axum::middleware::from_fn(origin::guard))
}

/// One embedded file, or the page itself for anything else — the client router owns every route
/// under `/`, so an unknown path is one of ITS routes and not a 404.
async fn serve_spa_file(uri: axum::http::Uri) -> Response {
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

/// The types the built bundle actually contains, plus the image and font formats a `static/` asset
/// may add. Anything else is served as bytes, which every browser downloads rather than mis-renders.
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
) -> std::io::Result<()> {
    axum::serve(listener, app(state, spa)).await
}

/// Native node type names visible in the catalog (`--list-nodes`).
/// Hides `_`-prefixed test nodes, exactly as the palette projection does.
pub fn catalog_type_names() -> Vec<String> {
    goofi_node::catalog()
        .filter(|m| !m.type_name.starts_with('_'))
        .map(|m| m.type_name.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Node discovery — one seam, called at boot and on every rescan
// ---------------------------------------------------------------------------

/// Which tier took a node file — and, when neither could, why. Reported rather than printed,
/// because the seam below is shared: only the caller can tell a boot scan (whose registry starts
/// empty, so any collision is two files claiming one name) from a rescan (which re-registers every
/// type it finds on purpose, and must not spew to stderr for doing its job).
pub enum Tier {
    InProcess,
    Subprocess,
    /// Neither tier could load it. It is recorded as unloadable, so the palette lists it greyed
    /// with this reason instead of letting the file silently not exist.
    Unavailable(String),
}

/// A file's size and mtime — the "did this node's code change since the last scan?" test, the same
/// one the workspace dirty check uses. `None` when the file could not be stat'd, which compares
/// equal to itself and so reads as "unchanged".
pub type Stamp = (u64, std::time::SystemTime);

/// One node file's outcome from a scan of one directory.
pub struct ScannedType {
    pub type_name: String,
    pub tier: Tier,
    pub stamp: Option<Stamp>,
    /// What the registry did with it. An unloadable file reports `Added`/`Refused` (it entered the
    /// unavailable registry, or a built-in owns the name); `Replaced` is the boot-only warning.
    pub registration: goofi_engine::Registration,
}

/// The node-discovery seam: scan ONE directory and report what it registered. Injected by the CLI,
/// which owns the interpreters and the probe, so boot and [`rescan`] re-derive the registry through
/// the same function. The default is a no-op.
pub type NodeScan = Arc<dyn Fn(&mut Graph, &std::path::Path) -> Vec<ScannedType> + Send + Sync>;

/// What a [`rescan`] changed, for the caller that asked — an agent that just wrote a node file, or
/// the palette's refresh button.
#[derive(Default)]
pub struct ScanDiff {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
}

/// Re-derive the registry from the directories that exist RIGHT NOW — the shipped tree, then
/// `<patch>/nodes`, in that order, so a patch-local node of the same name wins.
///
/// The previous scan's stamps are the baseline, so this answers a DIFF rather than a listing — and
/// removal is driven by that baseline too, which keeps a type registered some other way out of the
/// blast radius. The CLI's boot scan runs this rather than the seam, so the first refresh diffs
/// against the boot scan rather than re-announcing the whole tree.
pub fn rescan(
    state: &AppState,
    g: &mut Graph,
    patch: &std::path::Path,
) -> (ScanDiff, Vec<ScannedType>) {
    let mut found: std::collections::BTreeMap<String, Option<Stamp>> = Default::default();
    let mut patch_types: HashSet<String> = HashSet::new();
    let mut outcomes = Vec::new();
    // Shipped trees in the order they were named, patch LAST — the scan order IS the precedence,
    // so a later `--extra-nodes` shadows an earlier one exactly as the patch shadows them all.
    let dirs = (state.system_nodes.iter().map(|d| (d.clone(), false)))
        .chain(std::iter::once((patch.join("nodes"), true)));
    for (dir, is_patch) in dirs {
        if !dir.is_dir() {
            continue;
        }
        for t in (state.scan_nodes)(g, &dir) {
            // A refused name never reaches the palette (a built-in owns it), so it must not enter
            // the index either — it would report as `added` and, later, as `removed`.
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
/// on the canvas. `setup()` re-runs — a buffer empties, a device reopens — which is the price.
/// NOT part of [`rescan`]: a load re-derives the registry for a graph about to be replaced.
fn restart_changed(g: &mut Graph, diff: &ScanDiff) {
    for uid in g.node_uids() {
        if g.type_name(uid).is_some_and(|t| diff.changed.iter().any(|c| c == t)) {
            // Can only fail for a type that does not resolve, and every name in `changed` was just
            // registered by the scan that produced it.
            let _ = g.restart_node(uid);
        }
    }
}

// ---------------------------------------------------------------------------
// Control plane
// ---------------------------------------------------------------------------

async fn control_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_control(socket, state))
}

async fn handle_control(socket: WebSocket, state: AppState) {
    let (mut tx, mut rx) = socket.split();

    // Subscribe BEFORE snapshotting the document: a peer's edit landing in the other order is in
    // neither, and the replica desyncs silently with nothing to notice it. This way the worst case
    // is a RE-delivery — a patch whose result the snapshot already carries — which a replica reads
    // as stale by its version and skips.
    let mut events = state.events.subscribe();

    // Answered BEFORE the graph lock is taken: it walks the workspace mount (see `is_dirty`), and
    // no filesystem walk may run while the status-drain worker is waiting on that lock.
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

    // …then the document, whole. It rides the SAME socket as `hello` and as every later delta, so
    // a replica never has to reconcile two orderings.
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
                // Lagged past the shared ring, so a structural event AND any number of document
                // deltas are gone. Both halves are re-seeded exactly as a fresh connection seeds
                // them — `hello` for what is client-local, the whole document for the rest — which
                // is why the two travel together here as they do above.
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
    /// Whether the patch differs from its last saved state — the title-bar dot and the unload
    /// guard. TWO sources, because a patch is a graph AND a workspace: the flag every mutating op
    /// sets, and a directory the agent or the user's editor writes into with no RPC to ride on.
    ///
    /// No watcher: the workspace half is answered by WALKING the mount when a client asks, OFF the
    /// graph lock — that walk is a stat per file, and the drain worker takes the lock every ms.
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Relaxed)
            || goofi_engine::archive::fingerprint(&self.mount()) != *self.workspace_baseline.lock().unwrap()
    }

    /// Set the dirty flag, returning an `unsaved_changes` event ONLY when it actually changed —
    /// every mutation would otherwise re-broadcast the same value.
    fn set_dirty(&self, dirty: bool) -> Option<String> {
        let was = self.dirty.swap(dirty, std::sync::atomic::Ordering::Relaxed);
        (was != dirty).then(|| event("unsaved_changes", json!({ "unsaved_changes": dirty })))
    }
}

pub(crate) fn event(name: &str, payload: Value) -> String {
    json!({ "event": name, "payload": payload }).to_string()
}

/// The palette catalog changed — a rescan re-derived it, or a load brought a patch's own node types
/// with it. `hello` carries the catalog to a client that is CONNECTING; this is how one that is
/// already connected learns the same thing, and it is what keeps a second tab from offering a node
/// that no longer exists.
fn node_types_event(g: &Graph) -> String {
    event("node_types", json!({ "types": schemas::catalog_types(g) }))
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

/// Resolve a link endpoint AND refuse one that names nothing wirable — the check a caller-initiated
/// `add_link` gets and a REPLAY does not.
///
/// The command itself tolerates an endpoint naming no node, because it is the inverse of every
/// `remove_link` and a toggle replayed against a peer's delete must converge rather than wedge the
/// stack. A request is not a replay, so it is checked HERE, where an `Err` short-circuits before
/// the history records anything and the two stacks stay 1:1.
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

/// Resolve the `page` argument — a unique human name — to its stable id. A name is the ONLY way a
/// caller addresses a page, so an unknown one has to say which ones exist rather than just refusing.
fn resolve_page(g: &Graph, payload: &Value) -> Result<String, String> {
    let name = parse_str(payload, "page")?;
    g.arrangement().page_named(name).ok_or_else(|| {
        let have: Vec<&str> = g.arrangement().pages().iter().filter_map(|p| g.arrangement().name_of(p)).collect();
        // A leaked borrow would outlive the closure, so the names are collected before formatting.
        format!("no page named `{name}` — this patch has: {}", have.join(", "))
    })
}

/// Is `node` something a viewer/parameters/metadata panel could actually bind to? A UID, and only a
/// uid: a display name resolves until somebody renames the node, at which point the panel is bound
/// to nothing and says nothing about why. The uid is the identity, and it is what the frontend
/// stores — which is also what lets `RemoveNode` clear the bindings it invalidates.
fn bindable_node(g: &Graph, node: &str) -> bool {
    Uid::from_hex(node).is_some_and(|u| g.contains(u) || g.scope(u).is_some())
}

/// Route a layout planner's per-entry writes through the command history as ONE undo step, and
/// answer with the arrangement they produced — drawn as `inspect_layout` draws it, so a caller with
/// no screen does not have to follow every write with a read.
///
/// `born` names the entry an op BRINGS INTO BEING, which changes what undo means: the slots the
/// writes displaced are no longer the inverse — closing `born` with promote is. An op that only
/// rearranges what already exists passes `None` and inverts slot by slot.
///
/// **The rule, stated once rather than four times:** no layout inverse restores raw state. Every
/// one re-plans through the forward planners. Pinning an entry back into the slot it held
/// resurrects the split the forward op promoted away, on top of whatever a peer has since built
/// there — stranding their panels and corrupting the next save.
fn apply_layout(
    state: &AppState,
    g: &mut Graph,
    session: &str,
    cmd: goofi_engine::Command,
) -> Result<Value, String> {
    state.history.lock().unwrap().apply(g, session, cmd)?;
    Ok(json!({ "text": inspect::layout_tree(g.arrangement(), None) }))
}

/// Dispatch one control RPC. Mutates the graph, queues broadcast events, and
/// returns the `{id,result}`/`{id,error}` reply (only when `id` is numeric).
impl AppState {
    /// Run one control op. **The single entry point every surface shares** — `/control` and `/mcp`
    /// are JSON transports over this, and an in-process caller (a script, an integration test) needs
    /// no transport at all. `session` scopes the undo history the way a browser tab's id does.
    pub fn call(&self, op: &str, payload: Value, session: &str) -> Result<Value, String> {
        let state = self;
        let (op, session) = (op.to_string(), session.to_string());
        // Every op is declared once, in `ops::REGISTRY`. Refusing an unregistered one HERE makes a
        // dispatch arm without a row unreachable rather than a second, invisible declaration of the
        // op set — and it is where `read_only` below comes from, so classifying a new op is part of
        // declaring it.
        let spec = ops::find(&op);
        let mut events: Vec<String> = Vec::new();
        let result: Result<Value, String> = (|| {
            if spec.is_none() {
                return Err(format!("unknown op `{op}`"));
            }
            // Ops that read no graph state are served WITHOUT the graph mutex. `list_dir` walks a
            // directory and stats every child, which can block for a long time on a huge or network
            // path — under the lock that would stall the status-drain worker for the whole walk. `get_patch`
            // is here for the same reason: `is_dirty` walks the workspace mount.
            if op == "list_dir" {
                return Ok(fsbrowse::list_dir(payload.get("path").and_then(|v| v.as_str())));
            }
            // Off the graph lock: the doc is its own mutex, and this reads only the doc.
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
            // The harness ops are here for the same reason and one more: they fork children and signal
            // them, and none of it reads or writes the graph. `dispatch` stays synchronous throughout —
            // detection and the stop grace both run on their own threads, and the roster converges
            // through `harness_changed` rather than by making a caller wait.
            if op == "list_harnesses" {
                state.harnesses.refresh_in_background(state.events.clone());
                return Ok(state.harnesses.roster());
            }
            if op == "spawn_harness" {
                let h = payload.get("harness").and_then(|v| v.as_str())
                    .ok_or("spawn_harness: missing harness")?;
                // A closed set the caller cannot see from here, so a refusal that does not name it
                // leaves nothing to try next. `list_harnesses` says which are actually INSTALLED; this
                // says which words exist at all.
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
            // …and `inspect_patch`'s header says the same thing, so its walk is taken here too, before
            // the lock — and only for that op, which is what the short circuit is for.
            let dirty = op == "inspect_patch" && state.is_dirty();
            let mut g = state.graph.lock().unwrap();
            match op.as_str() {
                "list_nodes" => Ok(json!({ "types": schemas::catalog_types(&g) })),
                // Re-derive the node registry from the directories that exist RIGHT NOW. Explicit, not
                // watched (decision, 2026-08-09): an agent calls it straight after writing a node file,
                // a human presses the palette's refresh button. The diff comes back so either can say
                // what happened, and the instances of a type whose file changed restart onto it.
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
                    // A CHOSEN uid and name rather than fresh ones — an automation door, not the
                    // undo path, which is manager-owned and never comes through this RPC. It is
                    // what lets a caller reconstructing a known graph keep its uid-keyed bindings.
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
                                    g.params(uid).and_then(|p| goofi_node::param(&p, group, name).cloned())
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
                    // The REPLY, though, answers a caller with no doc replica. Three of these it
                    // cannot derive from the type it just named: the display NAME the manager minted
                    // (which is how `nd()` addresses the node), the slots to wire, and the params as
                    // BORN — the inline ones above, and any seeded from a `default_expr`.
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
                    // A top-level leaf, a sub-patch member (leaf or nested instance), or a collapsed
                    // instance — RemoveNode dispatches internally and CAPTURES the whole subtree
                    // (members + params + links + stubs + membership) so its inverse restores it
                    // uid-stably (undoable; B3b closed the delete-undo gap). The result reaches clients
                    // via the post-dispatch re-mirror.
                    // Idempotent by design (a redo racing a peer's delete must converge, not wedge),
                    // which made `{ok: true}` on a uid naming nothing indistinguishable from a real
                    // delete — so the doc had to tell callers not to read `ok` as proof. Say it here
                    // instead, where it is a fact rather than a warning.
                    let existed = bindable_node(&g, &uid.to_hex());
                    state
                        .history
                        .lock()
                        .unwrap()
                        .apply(&mut g, &session, goofi_engine::Command::RemoveNode { uid })?;
                    Ok(json!({ "removed": existed }))
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
                    // Resolve either endpoint through a sub-patch boundary → flat leaf→leaf, REFUSING one
                    // that names nothing wirable, THEN route the resolved flat link through the history
                    // (undoable; inverse is a RemoveLink).
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
                    // The wire AS MADE. A boundary endpoint resolves to the flat inner leaf it exposes,
                    // so what got wired is not literally what was named; and the dtype the two slots
                    // agreed on is what decides whether the next link to the same output can be made
                    // at all. Neither is derivable from the request.
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
                // The leaf edits (param value / expression / node+instance pos / rename / globals) route
                // through the command history so each is undoable (B3a). The mutation reaches clients via
                // the post-dispatch re-mirror; only the runtime-derived, doc-invisible bits (a param's
                // `expression_error`, a rename's nd()-rewrite echo) are pushed as `state_update` events.
                // Re-enumerate a refreshable string param (a device/stream picker). NOT a command —
                // options are runtime-only, never persisted, so there is nothing to undo. They are
                // also invisible to the CRDT doc, so the status worker's echo is the ONLY way they
                // reach the client.
                //
                // The options do NOT ride this reply, and nothing here waits for them: the hook runs
                // on the node's own thread (§8.5), which is what stops a multi-second device scan
                // stalling anything. `Err` is still a real refusal — an unknown node or param, or one
                // the type never declared refreshable.
                "refresh_param" => {
                    let uid = parse_uid(&payload, "node")?;
                    let group = parse_str(&payload, "group")?.to_string();
                    let name = parse_str(&payload, "name")?.to_string();
                    g.refresh_param(uid, &group, &name)?;
                    Ok(json!({ "ok": true }))
                }
                "update_param" => {
                    let uid = parse_uid(&payload, "node")?;
                    let group = parse_str(&payload, "group")?.to_string();
                    let name = parse_str(&payload, "name")?.to_string();
                    let vjson = payload.get("value").ok_or("missing value")?;
                    let existing = g
                        .params(uid)
                        .and_then(|p| goofi_node::param(&p, &group, &name).cloned())
                        .ok_or("no such param")?;
                    let newp = goofi_engine::param_from_json(&existing, vjson, true);
                    state.history.lock().unwrap().apply(
                        &mut g,
                        &session,
                        goofi_engine::Command::EditParam {
                            uid,
                            group: group.clone(),
                            name: name.clone(),
                            value: Some(newp),
                            expr: None,
                        },
                    )?;
                    // The value AS STORED, which is not always the value asked for: a literal is
                    // coerced to the param's declared type and clamped to its range. A bare `ok` for a
                    // 500 that became 100 does not merely say nothing — it asserts a state the patch
                    // is not in, and every later decision the caller makes is taken against it.
                    Ok(json!({
                        "value": g
                            .params(uid)
                            .and_then(|p| goofi_node::param(&p, &group, &name).cloned())
                            .map(|p| goofi_engine::param_value_json(&p, true))
                    }))
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
                            group: group.clone(),
                            name: name.clone(),
                            value: None,
                            expr: Some(goofi_engine::ExprState { source, enabled, triggers }),
                        },
                    )?;
                    // The binding source rides the doc re-mirror; the runtime `expression_error` is
                    // doc-invisible, so echo the enriched descriptor (what the retired leaf path did).
                    events.push(param_state_update(&g, uid));
                    // A binding that does not compile is stored, not rejected — the source is kept so
                    // it can be fixed. So the REPLY has to carry the compile error, or a caller with no
                    // inspector open would read a plain `ok` and believe the binding took.
                    Ok(json!({ "error": g.param_expression(uid, &group, &name).and_then(|e| e.error) }))
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
                // Where THIS client is looking. Stored opaquely and NOT a doc root, so it can neither
                // drag a peer nor raise the unsaved dot; it rides the `.gfi` and `hello` all the same,
                // because persistence and dirtiness are separate axes.
                "set_viewpoint" => {
                    g.set_viewpoint(payload.get("viewpoint").cloned().unwrap_or(Value::Null));
                    Ok(json!({ "ok": true }))
                }

                // ── The flat arrangement (the fifth doc root) ────────────────────────────────────
                // Reads are served straight off the layout the manager holds. Writes are planned
                // against it and applied as ordinary commands, so every op below is undoable, persisted
                // and broadcast without a line of its own for any of the three.
                // The one layout read. `page` narrows it, exactly as `inspect_patch {scope}` narrows
                // the graph — the reason `page_list_panels` existed, without a second shape to keep
                // honest or a second name to choose between.
                "inspect_layout" => {
                    let page = match payload.get("page").filter(|v| !v.is_null()) {
                        Some(_) => Some(resolve_page(&g, &payload)?),
                        None => None,
                    };
                    Ok(json!({ "text": inspect::layout_tree(g.arrangement(), page.as_deref()) }))
                }
                "session_add_page" => {
                    let name = parse_str(&payload, "name")?.to_string();
                    let index = payload.get("index").and_then(|v| v.as_u64()).map(|i| i as usize);
                    let subtree = payload.get("subtree").and_then(|v| v.as_str()).map(str::to_string);
                    let (writes, page) = g.arrangement().add_page(&name, index, subtree.as_deref())?;
                    // A page built AROUND an existing subtree is a MOVE: its undo has to put the subtree
                    // back, where closing the page would delete it. A page born with its own fresh panel
                    // has nothing to give back, so it inverts by closing (see `Command::LayoutBirth`).
                    match subtree.as_deref() {
                        Some(s) => {
                            let root = s.to_string();
                            let cmd = goofi_engine::Command::LayoutMove { writes: Some(writes), root, home: None };
                            apply_layout(state, &mut g, &session, cmd)?
                        }
                        None => {
                            let cmd = goofi_engine::Command::LayoutBirth { writes, page: page.clone(), born: page.clone() };
                            apply_layout(state, &mut g, &session, cmd)?
                        }
                    };
                    // The page's id and its root panel's — a caller's next act is to give that panel
                    // content, which needs an id it cannot otherwise know (`page_split_panel`'s rule).
                    let panel = g.arrangement().children(&page).first().cloned().unwrap_or_default();
                    Ok(json!({ "page": page, "panel": panel }))
                }
                "session_remove_page" => {
                    let name = parse_str(&payload, "name")?.to_string();
                    // Planned here only so a bad name answers teachably: `LayoutClose` re-plans it under
                    // this same lock, and DEGRADES rather than errors, which a user's own op must not.
                    g.arrangement().remove_page(&name)?;
                    let page = g.arrangement().page_named(&name).unwrap_or_default();
                    let cmd = goofi_engine::Command::LayoutClose { page: page.clone(), born: page.clone() };
                    apply_layout(state, &mut g, &session, cmd)
                }
                "session_rename_page" => {
                    let (from, to) = (parse_str(&payload, "from")?, parse_str(&payload, "to")?);
                    let writes = g.arrangement().rename_page(from, to)?;
                    // A name is contents; the tab index is the slot, and a peer's new page may hold the
                    // one this page had when the rename was planned.
                    apply_layout(state, &mut g, &session, goofi_engine::Command::LayoutContents { writes })
                }
                "session_reorder_page" => {
                    let name = parse_str(&payload, "name")?;
                    let to = payload.get("to_index").and_then(|v| v.as_u64()).ok_or("missing to_index")?;
                    let writes = g.arrangement().reorder_page(name, to as usize)?;
                    let edits = writes
                        .into_iter()
                        .map(|(id, entry)| goofi_engine::Command::EditLayoutEntry { id, entry })
                        .collect();
                    apply_layout(state, &mut g, &session, goofi_engine::Command::Compound(edits))
                }
                "page_split_panel" => {
                    let page = resolve_page(&g, &payload)?;
                    let panel = parse_str(&payload, "panel")?.to_string();
                    let dir = payload.get("direction").and_then(|v| v.as_str()).unwrap_or("row");
                    let axis = goofi_engine::layout::Axis::parse(dir)
                        .ok_or("page_split_panel: direction is `row` or `column`")?;
                    let before = payload.get("place_before").and_then(|v| v.as_bool()).unwrap_or(false);
                    let ratio = payload.get("ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
                    let (writes, fresh) = g.arrangement().split_panel(&page, &panel, axis, before, ratio)?;
                    let cmd = goofi_engine::Command::LayoutBirth { writes, page: page.clone(), born: fresh.clone() };
                    apply_layout(state, &mut g, &session, cmd)?;
                    // The uid, because a split births an EMPTY panel and the caller's next act is to
                    // give it content — which needs the id it cannot otherwise know.
                    Ok(json!(fresh))
                }
                "page_set_panel" => {
                    let page = resolve_page(&g, &payload)?;
                    let panel = parse_str(&payload, "panel")?.to_string();
                    let ty = payload.get("type").and_then(|v| v.as_str()).map(str::to_string);
                    let panel_state = payload.get("state").cloned();
                    // A panel bound to a node that is not there renders empty and explains nothing, so
                    // the bind is checked HERE, where the answer can teach. Cheap: no graph mutation.
                    let named = panel_state
                        .as_ref()
                        .and_then(|s| s.get("node"))
                        .and_then(|v| v.as_str())
                        .filter(|n| !n.is_empty());
                    if let Some(node) = named {
                        if !bindable_node(&g, node) {
                            return Err(format!("page_set_panel: no node `{node}` in this patch"));
                        }
                    }
                    // …and the same argument, one word further in: the panel type and the viewer kind
                    // are vocabularies the manager stores as free strings, so a plausible GUESS at one
                    // used to be answered `{ok: true}`. The slot is checked against the node this write
                    // LEAVES the panel bound to — its own, or the one already stored, since state
                    // merges.
                    let bound = named
                        .or_else(|| match g.arrangement().get(&panel) {
                            Some(goofi_engine::layout::Entry::Panel { state, .. }) => {
                                state.get("node").and_then(|v| v.as_str())
                            }
                            _ => None,
                        })
                        .and_then(Uid::from_hex);
                    vocab::check_panel(&g, ty.as_deref(), panel_state.as_ref(), bound)?;
                    let writes = g.arrangement().set_panel(&page, &panel, ty.as_deref(), panel_state)?;
                    apply_layout(state, &mut g, &session, goofi_engine::Command::LayoutContents { writes })
                }
                "page_move_panel" => {
                    let page = resolve_page(&g, &payload)?;
                    let panel = parse_str(&payload, "panel")?.to_string();
                    let dest = parse_str(&payload, "new_parent")?.to_string();
                    let at = payload.get("order_index").and_then(|v| v.as_u64()).unwrap_or(0);
                    let writes = g.arrangement().move_subtree(&page, &panel, &dest, at as usize)?;
                    let cmd = goofi_engine::Command::LayoutMove { writes: Some(writes), root: panel.clone(), home: None };
                    apply_layout(state, &mut g, &session, cmd)
                }
                // The frozen drag gestures, each ONE op — a drop is one undo step and peers never see an
                // arrangement that was not on somebody's screen. Composed from the primitive ops they
                // would cost three to five of both.
                "page_insert_at_panel" => {
                    let page = resolve_page(&g, &payload)?;
                    let subtree = parse_str(&payload, "subtree")?.to_string();
                    let target = parse_str(&payload, "target")?.to_string();
                    let dir = payload.get("direction").and_then(|v| v.as_str()).unwrap_or("row");
                    let axis = goofi_engine::layout::Axis::parse(dir)
                        .ok_or("page_insert_at_panel: direction is `row` or `column`")?;
                    let before = payload.get("place_before").and_then(|v| v.as_bool()).unwrap_or(false);
                    let ratio = payload.get("ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
                    let writes =
                        g.arrangement().insert_at_panel(&page, &subtree, &target, axis, before, ratio)?;
                    let cmd = goofi_engine::Command::LayoutMove { writes: Some(writes), root: subtree.clone(), home: None };
                    apply_layout(state, &mut g, &session, cmd)
                }
                "page_resize_split" => {
                    let page = resolve_page(&g, &payload)?;
                    let split = parse_str(&payload, "split")?.to_string();
                    // A non-numeric entry becomes NaN and is refused by the planner alongside a zero or
                    // a negative one, so the whole "is this a fraction" answer is stated in one place.
                    let fractions: Vec<f64> = payload
                        .get("fractions")
                        .and_then(|v| v.as_array())
                        .ok_or("page_resize_split: missing fractions")?
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(f64::NAN))
                        .collect();
                    let writes = g.arrangement().resize_split(&page, &split, &fractions)?;
                    apply_layout(state, &mut g, &session, goofi_engine::Command::LayoutContents { writes })
                }
                "page_remove_panel" => {
                    let page = resolve_page(&g, &payload)?;
                    let panel = parse_str(&payload, "panel")?.to_string();
                    // Planned only for its teachable refusal — see `session_remove_page` above.
                    g.arrangement().remove_subtree(&page, &panel)?;
                    let cmd = goofi_engine::Command::LayoutClose { page: page.clone(), born: panel.clone() };
                    apply_layout(state, &mut g, &session, cmd)
                }
                "set_node_viewers" => {
                    // Soft per-slot view-state (kind/settings/collapse) persisted to `.gfi` — NOT a
                    // command (not undoable). Written to the graph; the re-mirror persists + broadcasts.
                    let uid = parse_uid(&payload, "node")?;
                    let viewers = payload.get("viewers").cloned().ok_or("set_node_viewers: missing viewers")?;
                    // Opaque to the ENGINE, which is why the words in it are checked here — the bag is
                    // keyed by output slot and each entry names a viewer kind, the same two
                    // vocabularies `page_set_panel` refuses a guess at.
                    vocab::check_viewers(&g, uid, &viewers)?;
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
                    // A display name is spliced into expression SOURCE by `rewrite_nd_refs`, which
                    // replaces the literal's content span in place — so a quote or backslash yields
                    // `nd('a'b')`, invalid Python that the REFERRING node then carries as a binding
                    // error while this rename reports success. Constraining the name is one line;
                    // making the rewriter quote-aware is a Python tokenizer.
                    if name.contains(['\'', '"', '\\']) {
                        return Err(format!(
                            "rename_node: `{name}` cannot contain a quote or backslash — a display \
                             name is spliced into nd() expression source"
                        ));
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
                    let Some(held) = g.globals().get(&name).map(goofi_engine::global_to_json) else {
                        return Err(format!("set_global: no such global `{name}`"));
                    };
                    // A global's TYPE is what every expression reading it depends on, so re-typing one
                    // through a value edit breaks the reference rather than the call. Choosing a type
                    // is `add_global`'s; this op edits what a global HOLDS.
                    let held_ty = held["type"].as_str().unwrap_or_default();
                    if held_ty != ty {
                        return Err(format!("set_global: `{name}` is a {held_ty} — set_global edits a \
                                            global's value, not its type"));
                    }
                    let value = goofi_engine::global_from_json(&json!({ "value": val, "type": ty }))
                        .ok_or_else(|| format!("set_global: `{val}` is not a {ty}"))?;
                    state.history.lock().unwrap().apply(
                        &mut g,
                        &session,
                        goofi_engine::Command::EditGlobal { name, value: Some(value.clone()), at: None },
                    )?;
                    // As STORED: `global_from_json` is type-directed, so a fraction into an int global
                    // rounds — the same reason `update_param` answers its value.
                    Ok(json!({ "value": goofi_engine::global_to_json(&value)["value"] }))
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
                // The inspect reads. Every one is `writes: false` in the registry, so none re-mirrors
                // and none dirties the patch — they answer questions, they do not edit.
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
                    // The three sections default ON — the op is the cheap peek, and a caller that
                    // wants less says so.
                    let want = |k: &str| payload.get(k).and_then(|v| v.as_bool()).unwrap_or(true);
                    let slot = payload.get("slot").and_then(|v| v.as_str());
                    let text = inspect::node(&g, uid, slot, want("params"), want("error"))?;
                    Ok(json!({ "text": text }))
                }
                "list_globals" => Ok(inspect::globals(&g)),
                "read_node_source" => {
                    // The two trees a scan registers from, patch first — the same precedence the
                    // palette's `source` badge reports, so provenance cannot disagree with it.
                    // `.rev()` is load-bearing, not tidiness: `rescan` scans the shipped list forwards
                    // and lets each directory overwrite the last, so the LAST one holds the name. This
                    // search runs first-match-wins, so it has to walk the same list backwards to land
                    // on the same file. Forwards here would hand back a shadowed copy nothing runs.
                    let dirs: Vec<(PathBuf, &str)> = [(state.mount().join("nodes"), "patch")]
                        .into_iter()
                        .chain(state.system_nodes.iter().rev().map(|d| (d.clone(), "shipped")))
                        .collect();
                    inspect::node_source(&g, parse_str(&payload, "type")?, &dirs)
                }
                "serialize" => Ok(json!({ "yaml": g.serialize() })),
                // Where this patch's workspace files live right now. The mount is a per-run temp
                // directory under a random name, so a client — and the agent harness after it — cannot
                // derive it; asking the manager is the only way to open a browser or a shell on it.
                "open_workspace" => Ok(json!({ "path": goofi_core::path::to_slash(&state.mount()) })),
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
                    let mount = state.mount();
                    // Sampled BEFORE the pack and committed only once it succeeded. A file written
                    // while the zip is being built may or may not have made it in; baselining
                    // afterwards would call it packed either way, and that is the one direction that
                    // loses an edit rather than merely reporting a spurious one.
                    let packed = goofi_engine::archive::fingerprint(&mount);
                    save_archive(std::path::Path::new(&path), &g.serialize(), &mount)?;
                    // Written to disk ⇒ clean, on both planes — and said so UNCONDITIONALLY, not on the
                    // flag's transition: a patch dirtied solely by a file written into the mount leaves
                    // the flag already false, so no transition comes and every tab would keep its dot
                    // on a patch that is entirely on disk. The duplicate event the common case now gets
                    // is free — a save is one user action, and every client apply branch is idempotent.
                    *state.workspace_baseline.lock().unwrap() = packed;
                    state.set_dirty(false);
                    events.push(event("unsaved_changes", json!({ "unsaved_changes": false })));
                    // …and the patch now has a home the MANAGER knows (C38), so a later plain Save
                    // overwrites this file from any tab, and a reload still names it. Announced as
                    // well as stored: an already-connected peer gets no new snapshot to read it from.
                    // Only on success — a failed save wrote nothing, so whatever home the patch had
                    // (including none) is still the true one, and claiming this one would point the
                    // next silent overwrite at a file this patch has never been written to.
                    *state.save_path.lock().unwrap() = Some(path.clone());
                    events.push(event("save_path_changed", json!({ "save_path": &path })));
                    Ok(json!({ "path": path }))
                }
                // One load path for every source: `load_text` carries the YAML inline (a browser
                // upload), `load` names a `.gfi` the BACKEND reads, and `new` brings an empty patch
                // from nowhere. Everything after the read — replace, reset history, announce — must
                // not drift between them, so they share an arm.
                "load_text" | "load" | "new" => {
                    // Every source mounts FRESH, and the live mount is swapped for it only once the
                    // manifest has parsed. So a refused load leaves the open patch untouched on both
                    // planes — its graph AND its workspace files — and a loaded patch never inherits
                    // the files of the patch it replaced.
                    let fresh = new_mount();
                    let (content, from_path) =
                        stage_load(&fresh, &op, &payload).inspect_err(|_| remove_mount(&fresh))?;
                    // ORDERING, load-bearing: the types the patch SHIPS are registered before the
                    // manifest is resolved, or `load_doc`'s unknown-type gate fires on exactly the
                    // nodes the archive brought. They live in the tree just unpacked, so the scan runs
                    // against `fresh` — which is not the live mount yet.
                    rescan(state, &mut g, &fresh);
                    // Parse BEFORE anything is announced or committed: a rejected patch must not leave
                    // the title bar naming a file the graph was never loaded from.
                    if let Err(e) = g.load_doc(&content) {
                        // Refused, so the open patch keeps its graph AND its workspace — and therefore
                        // its registry, which the scan above swapped for the refused patch's. Re-derive
                        // it from the mount that is still live.
                        rescan(state, &mut g, &state.mount());
                        remove_mount(&fresh);
                        return Err(e);
                    }
                    // Commit, now that nothing left can fail: the loaded patch's workspace becomes the
                    // live one and the mount it replaced is reclaimed — after the lock drops, since
                    // deleting a tree is a walk and the lock guards only the swap. The harnesses the
                    // replaced patch spawned go with it: each was launched INTO that workspace and
                    // edits that graph through an address goofi minted for it, so one surviving here
                    // would go on editing a patch it was never launched for out of a directory the
                    // next line deletes. Announced too — `graph_replaced` below carries the emptied
                    // roster, but a client tracking only the transitions must not have to infer it.
                    state.harnesses.stop_all();
                    events.push(event("harness_changed", state.harnesses.roster()));
                    let replaced = std::mem::replace(&mut *state.mount.lock().unwrap(), fresh);
                    remove_mount(&replaced);
                    // The unpacked tree IS what the archive holds — but every file in it was written
                    // seconds ago (`read_gfi` restores no mtimes), so this baseline has to be taken
                    // HERE. Without it a patch would be dirty from the moment it finished loading.
                    *state.workspace_baseline.lock().unwrap() = goofi_engine::archive::fingerprint(&state.mount());
                    // A load fully resets the session — there is nothing to undo across it (spec §3:
                    // no load command / no checkpoint), so drop every session's command history.
                    state.history.lock().unwrap().clear();
                    events.extend(state.set_dirty(false));
                    // The loaded patch's home is the archive it came from — or NONE for `load_text` (an
                    // upload) and `new`, neither of which has a file behind it. Inheriting a path there
                    // would aim the next silent Save at an unrelated `.gfi` and overwrite it with a
                    // patch that never came from it. Stored BEFORE the snapshot is built, so the
                    // snapshot carries it.
                    *state.save_path.lock().unwrap() = from_path.clone();
                    events.push(event(
                        "graph_replaced",
                        schemas::snapshot(&g, &state.instance_id, false, false, from_path.as_deref(),
                                          state.harnesses.roster()),
                    ));
                    // The patch brought its own node types (and dropped the last patch's), which
                    // `graph_replaced` does not carry — the snapshot's catalog rides `hello` alone.
                    events.push(node_types_event(&g));
                    if let Some(path) = from_path {
                        // The announcement the title bar reads. The ORDER no longer carries meaning:
                        // the snapshot the client applies wholesale now names the same file, so
                        // announcing first would be re-affirmed rather than clobbered. It is kept
                        // after `graph_replaced` because that is the order the two facts happen in,
                        // and kept at all because `save` — which ships no snapshot — needs the event
                        // to exist, and one event shape is easier to be right about than two.
                        events.push(event("save_path_changed", json!({ "save_path": path })));
                    }
                    // A stored arrangement the flat model admits but cannot render falls back to the
                    // default — the graph is the value, the arrangement is chrome. Say so here, or the
                    // patch would open on a layout the user did not save and nothing would explain it.
                    Ok(json!({ "ok": true, "layout_warning": g.arrangement_warning() }))
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
        // `open_workspace` joins them: it answers where the mount is and writes nothing on either
        // plane. Being here is also what keeps it out of the dirty tail below — the whole block is
        // skipped — which is the right door for it, since it is a question, not an op that "did not
        // happen to be an edit".
        // `new` is deliberately NOT here: it empties the graph, and the re-mirror is the only thing
        // that empties an already-open tab's canvas with it (`graph_replaced` carries no node list).
        // The classification lives on the op's registry row, so declaring an op is what classifies it
        // — there is no second list to forget. An unregistered op never reaches here (its result is
        // the `unknown op` Err above).
        let read_only = spec.is_some_and(|o| !o.writes);
        if result.is_ok() && !read_only {
            resync_and_broadcast(state);
            // "Could this have changed the graph?" is a good enough answer to "does the patch now
            // differ from disk?" for most ops that the two share a gate — but it is an INFERENCE, and
            // these four are where it is wrong:
            //   `load`/`load_text`/`new` clear the flag inside their arm, which runs first and is then
            //     re-set here; re-clear it. `new` is the one where the tail's default is most clearly
            //     wrong rather than merely conservative: an empty patch with nothing in it and no file
            //     behind it would be born unsaved, offering to be written over the last real patch.
            //   `restart_node` respawns an instance in place, replaying the node's own ParamGroups
            //     verbatim and touching neither name, position, bindings, viewers, links nor scopes, so
            //     `serialize()` is byte-identical. It is RECOVERY, not an edit, and it is reached by one
            //     click on the inspector's Restart button after a node raised — exactly where a spurious
            //     unsaved dot is least distinguishable from a real one.
            //   `rescan_nodes` re-derives the CATALOG, which is not patch content. It still re-mirrors,
            //     because restarting a node whose type gained a param changes that node's params — and
            //     it still must not dirty: pressing refresh with nothing edited would otherwise put the
            //     dot on an untouched patch, while a rescan that DID follow an edit is already dirty
            //     through the workspace fingerprint (`is_dirty`), which is where a file edit belongs.
            //   `refresh_param` re-enumerates a device/stream picker's options, which are runtime-only
            //     and never persisted. Latent today (no shipped node declares `refresh: true`, and the
            //     engine rejects the op for any param that does not, so the `Err` skips this gate
            //     entirely) — listed here because it is the same op-is-not-an-edit case, not a
            //     prediction that it currently misfires.
            //   `set_viewpoint` is persistence-without-dirtiness, and by CONSTRUCTION rather than by a
            //     classification the client has to get right: a viewpoint is where a client is LOOKING,
            //     so writing one is never authoring. It still rides the `.gfi`, which is exactly why it
            //     needs an arm here and not `writes: false`. Every op that edits the ARRANGEMENT is
            //     authoring by the same construction, and so needs no arm at all.
            // These stay OUT of `read_only`: none is an edit, but all still need the re-mirror.
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
    // The caller's session tag (a browser tab's stable id) scopes the command history's undo/redo.
    // Absent ⇒ a single shared "default" session, so a client that never presents one still works.
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

/// Re-project the (already-locked) graph into the (already-locked) document and broadcast the
/// delta. The caller holds `graph` then `doc` — the canonical order — and passing the guards in is
/// what keeps the apply→re-project critical section atomic, so no concurrent writer can observe a
/// document leaf the graph has not yet caught up to.
fn remirror_and_broadcast_locked(state: &AppState, g: &Graph, doc: &mut crate::doc::GraphDoc) {
    let from = doc.version();
    // `None` means the projection is what the document already holds — a read-only op, or a write
    // that landed on the value already there. Nothing goes on the wire and the version does not
    // move, so an idle patch is silent.
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

/// Re-project the authoritative graph into the document and broadcast the delta. Called after an
/// RPC dispatch mutates the graph. Because the projection is built WHOLE, this also converges any
/// stale document leaf back to the graph rather than trusting an op to describe its own change.
fn resync_and_broadcast(state: &AppState) {
    let g = state.graph.lock().unwrap();
    let mut doc = state.doc.lock().unwrap();
    remirror_and_broadcast_locked(state, &g, &mut doc);
}

// ---------------------------------------------------------------------------
// Harness plane
// ---------------------------------------------------------------------------

async fn term_ws(
    Path(instance): Path<String>,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_term(socket, state, instance))
}

/// The inband control a `/term` client sends. Resize is the only one there is: everything else a
/// terminal needs already rides the byte stream.
#[derive(serde::Deserialize)]
struct TermControl {
    op: String,
    cols: u16,
    rows: u16,
}

/// One `/term` socket. **Binary frames are PTY bytes** in both directions; **text frames are JSON
/// control**: `{op:"resize", cols, rows}` inbound, `{op:"size", cols, rows}` and `{exit_code}`
/// outbound. An already-exited instance answers its code at once and closes, so a tab that opens
/// the panel after the harness died sees why instead of an empty terminal.
///
/// A resize is a PROPOSAL, not an order: several views can watch one instance and a PTY has one
/// window, so [`term::Sizes`] arbitrates and the answer is broadcast to every view — including the
/// one that asked, which is what lets a client resize its terminal from the authoritative frame
/// alone and never from its own measurement. A view that says `0` retracts (its panel unmounted);
/// closing the socket leaves the seat, and both hand the terminal back to the survivors.
///
/// Nothing is buffered here — see [`term`]'s module note: the client keeps its own terminal alive,
/// so this socket is a pipe rather than a replayable log.
async fn handle_term(socket: WebSocket, state: AppState, instance: String) {
    let (mut tx, mut rx) = socket.split();
    let Some(inst) = state.harnesses.get(&instance) else {
        let _ = tx.send(close(4004, "unknown harness instance")).await;
        return;
    };
    // Both halves are taken before the first await: a subscription made later would miss whatever
    // the child wrote in between, and the exit is the only thing that ends this loop.
    let (mut output, mut exit, mut eof) = inst.attach();
    let (seat, mut size) = inst.join();
    // The current answer, sent up front: a view that arrives while the size is already settled has
    // no change event coming to tell it, and would otherwise draw at its own default forever.
    let settled = *size.borrow_and_update();
    if let Some((cols, rows)) = settled {
        let _ = tx.send(size_frame(cols, rows)).await;
    }
    // When the child was reaped, so the announcement below can bound its wait for the PTY.
    let mut reaped_at: Option<tokio::time::Instant> = None;
    loop {
        // The exit frame is the LAST thing this socket sends, not the first thing it reaches for.
        // A dying harness writes its stack trace, its auth failure, its rate-limit message in the
        // instant before it goes, and `child.wait()` returns while those bytes are still in flight
        // — so the announcement waits for the PTY's own end-of-stream (set after the drain's final
        // publish, which a `try_recv` sweep here would race) AND for this socket to have been
        // handed everything published. The senders live in the instance this task holds an `Arc`
        // of, so neither `changed()` can fail. Both are copied out of their guards rather than
        // matched through them: a `watch` borrow is not `Send`, and the send below is an await.
        let (ended, drained) = (*exit.borrow_and_update(), *eof.borrow_and_update());
        if ended.is_some() && reaped_at.is_none() {
            reaped_at = Some(tokio::time::Instant::now());
        }
        // …but the PTY's end-of-stream is not guaranteed to come. ConPTY keeps the pseudoconsole
        // open after the child exits, so on Windows `drained` never turns true and a viewer waiting
        // only for it would never be told the harness finished — on the very platform where a
        // headless agent has nobody watching the screen to notice. So the wait is BOUNDED: the
        // drain's own end when it arrives, or a short settle after the reap, which is ample for
        // bytes already sitting in the buffer to be published. No platform branch: where EOF is
        // real it wins the race every time and the settle never elapses.
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
                // A viewer that fell behind loses those bytes rather than the CHILD stalling behind
                // it. The screen is garbled only until the next repaint, which is the trade the
                // no-emulator decision already made.
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
            // Wake to re-check the settle above. Armed only once the child has been reaped and the
            // PTY has not ended on its own — where EOF is real this never runs.
            _ = tokio::time::sleep_until(reaped_at.unwrap_or_else(tokio::time::Instant::now) + EXIT_SETTLE),
                if reaped_at.is_some() && !drained => {}
        }
    }
    inst.leave(seat);
}

fn size_frame(cols: u16, rows: u16) -> Message {
    Message::Text(json!({ "op": "size", "cols": cols, "rows": rows }).to_string().into())
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

// The two blocks below are the deliberate exception to "the suite lives in goofi-tests". Each
// drives a PRIVATE state machine — the `/data` probe/pong deadline, and the bounded write that
// gives up rather than parking the loop — whose outcome the transport suite already asserts. What
// it cannot reach is the machine itself, and publishing one purely so a test could name it would
// be the tail wagging the dog.
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
