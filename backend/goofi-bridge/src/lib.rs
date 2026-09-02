//! The axum server: `/control` (JSON RPC + broadcast events, doc state and doc deltas among them),
//! `/data/<node>/<slot>` (ONE reduced GOOF stream per slot, whatever the viewer count — the kind
//! is not in the path, since viewers publish their ViewSpec inband), `/term`, `/mcp`, and the SPA
//! compiled into the binary.

mod arms;
pub mod phrase;
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

/// How long a closing socket waits for the peer's own close before it drops the connection.
const FAREWELL: Duration = Duration::from_millis(250);


use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::response::Response;
use axum::routing::{any, get, post};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use goofi_graph::{Graph, Uid};
use serde_json::{json, Value};
use tokio::sync::broadcast;

/// Where the development surfaces live. One literal, so the gate and the app agree on the prefix.
pub const DEV_ROUTE_PREFIX: &str = "/dev/";

/// The undo actor for a caller that names none — one shared stack, isolated from every named one.
pub const DEFAULT_ACTOR: &str = "default";

/// The built SPA as it ships: a URL path and its bytes, compiled into the binary. Empty when the
/// crate was built without a frontend, which [`HEADLESS_BUILD`] says whether anyone asked for.
pub type Spa = &'static [(&'static str, &'static [u8])];
include!(concat!(env!("OUT_DIR"), "/spa.rs"));

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<Mutex<Graph>>,
    pub events: broadcast::Sender<String>,
    pub instance_id: Arc<str>,
    /// The op rows THIS instance serves — headless leaves the layout group out.
    ops: Arc<Vec<&'static ops::Op>>,
    /// The control-plane document every client replicates, re-projected from the graph after each
    /// successful op; its deltas ride the `events` channel.
    pub doc: Arc<Mutex<crate::doc::GraphDoc>>,
    /// Whether the patch has been mutated since it was last saved or loaded. Nothing persists it,
    /// so a fresh session starts clean.
    dirty: Arc<std::sync::atomic::AtomicBool>,
    /// One reduction per active (node, slot), fanned out to every viewer.
    pub reducers: reducer::SlotReducers,
    /// The central per-session command history. Locked AFTER `graph`, BEFORE `doc`.
    pub history: Arc<Mutex<goofi_graph::CommandHistory>>,
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
    /// The bound address — ONE owner for every local base url: the session file's `/exec`
    /// url derives from it.
    bound: Arc<Mutex<std::net::SocketAddr>>,
    /// The spawned agent harnesses and their PTYs.
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
        Self::new(false)
    }
}

impl AppState {
    pub fn new(headless: bool) -> AppState {
        let (events, _) = broadcast::channel(256);
        let iid = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        // Project the INITIAL graph — no nodes, but the seeded system globals — so a client that
        // connects to a fresh backend has the current state at once.
        let graph_val = fresh_graph();
        let mut doc = crate::doc::GraphDoc::new();
        doc.reconcile_root(&projection::of(&graph_val));
        let graph = Arc::new(Mutex::new(graph_val));
        let reducers = reducer::SlotReducers::new(graph.clone());
        // Seeded BEFORE the baseline is taken, or the patch is dirty from boot, having written
        // the seed itself.
        let mount = new_mount();
        term::seed_orientation(&mount);
        let workspace_baseline = goofi_graph::archive::fingerprint(&mount);
        AppState {
            graph,
            events,
            instance_id: Arc::from(format!("{iid:x}").as_str()),
            ops: Arc::new(ops::table(headless)),
            doc: Arc::new(Mutex::new(doc)),
            dirty: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            reducers,
            history: Arc::new(Mutex::new(goofi_graph::CommandHistory::new())),
            data_liveness: DataLiveness::DEFAULT,
            scan_nodes: Arc::new(|_, _| Vec::new()),
            system_nodes: Vec::new(),
            node_index: Arc::new(Mutex::new(Default::default())),
            mount: Arc::new(Mutex::new(mount)),
            workspace_baseline: Arc::new(Mutex::new(workspace_baseline)),
            save_path: Arc::new(Mutex::new(None)),
            bound: Arc::new(Mutex::new(([127, 0, 0, 1], 8000).into())),
            harnesses: Arc::new(term::Harnesses::default()),
        }
    }

    /// Record the address this server actually bound — what `local_url` derives from.
    pub fn set_bound(&self, addr: std::net::SocketAddr) {
        *self.bound.lock().unwrap() = addr;
    }

    /// The base URL a LOCAL client reaches this server at — a spawned harness, the session
    /// file's reader. Loopback whenever loopback listens (a wildcard or loopback bind); the
    /// bound address itself when `--bind` named one other interface, where loopback answers
    /// nothing.
    pub fn local_url(&self) -> String {
        let a = *self.bound.lock().unwrap();
        match a.ip().is_unspecified() || a.ip().is_loopback() {
            true => format!("http://127.0.0.1:{}", a.port()),
            false => format!("http://{a}"),
        }
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
    let packed = goofi_graph::archive::write_gfi(&tmp, manifest, mount)
        .and_then(|()| std::fs::rename(&tmp, target).map_err(|e| format!("{}: {e}", target.display())));
    if packed.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    packed.map_err(|e| format!("save failed: {e}"))
}

/// The front half of a load, against a mount that is not yet live. It stops AT the manifest,
/// because the patch's own node types must be registered before `load_doc` resolves the graph.
fn stage_load(mount: &std::path::Path, payload: &Value) -> Result<(String, Option<String>), String> {
    let from_file = payload.get("path").and_then(|v| v.as_str()).filter(|p| !p.is_empty());
    let inline = payload.get("content").and_then(|v| v.as_str());
    let (content, from_path, unpacked) = if let Some(p) = from_file {
        if inline.is_some() {
            return Err("session load: a `path` to an archive or a `content` manifest, never both".into());
        }
        // Expand `~` exactly as the browser does — the two must agree on what a path means.
        let path = fsbrowse::resolve(p);
        let manifest = goofi_graph::archive::read_gfi(std::path::Path::new(&path), mount)
            .map_err(|e| format!("session load failed: {e}"))?;
        // Whether this file becomes the patch's home — the target a later silent Save overwrites.
        let adopt = payload.get("adopt").and_then(Value::as_bool).unwrap_or(true);
        (manifest, adopt.then_some(path), true)
    } else if let Some(content) = inline {
        (content.to_string(), None, false)
    } else {
        // Naming no source IS the source: an empty patch, so New cannot drift from Load.
        (Graph::new().serialize(), None, false)
    };
    // Only a workspace goofi minted empty is seeded: an archive has just unpacked the patch's OWN
    // workspace into `mount`, and goofi does not write into someone's patch.
    if !unpacked {
        term::seed_orientation(mount);
    }
    Ok((content, from_path))
}

/// The API routes, unguarded. The ONLY caller is [`app`], because `Router::layer` wraps only what
/// is already on the router — a route added elsewhere would miss the [`origin`] guard.
fn routes(state: AppState) -> Router {
    Router::new()
        .route(
            "/control",
            any(|ws: WebSocketUpgrade, State(state): State<AppState>| async {
                ws.on_upgrade(move |socket| handle_control(socket, state))
            }),
        )
        // One stream per (node, slot): each connection sends its viewers' ViewSpecs inband.
        .route(
            "/data/{node}/{slot}",
            any(|Path((node, slot)): Path<(String, String)>,
                 ws: WebSocketUpgrade,
                 State(state): State<AppState>| async {
                ws.on_upgrade(move |socket| handle_data(socket, state, node, slot))
            }),
        )
        // The body limit is lifted: axum caps at 2 MB and a patch with a workspace is larger.
        .route(
            "/patch.gfi",
            get(patchfile::download)
                .post(patchfile::upload)
                .layer(axum::extract::DefaultBodyLimit::disable()),
        )
        // The CLI's door: the same lines, parse and batch semantics as `goofi_exec`.
        .route("/exec", post(exec_endpoint))
        .route("/mcp", post(mcp::endpoint))
        // A spawned harness's terminal: binary frames are PTY bytes, text frames JSON control.
        .route(
            "/term/{instance}",
            any(|Path(instance): Path<String>,
                 ws: WebSocketUpgrade,
                 State(state): State<AppState>| async {
                ws.on_upgrade(move |socket| handle_term(socket, state, instance))
            }),
        )
        .with_state(state)
}

/// The uids whose error state changed since `last`, which is updated in place. A node first seen
/// HEALTHY is not a change, and a memo answers for the node INSTANCE that reported it: a rebirth at
/// a uid — a load, a restart — is a transition whatever its predecessor last said.
fn error_transitions(
    current: &[(String, u64, Option<String>)],
    last: &mut HashMap<String, (u64, Option<String>)>,
) -> Vec<String> {
    let seen: HashSet<&String> = current.iter().map(|(u, ..)| u).collect();
    let mut changed = Vec::new();
    for (uid, generation, err) in current {
        let is_changed = match last.get(uid) {
            Some((g, e)) => g != generation || e != err,
            None => err.is_some(),
        };
        if is_changed {
            changed.push(uid.clone());
        }
        last.insert(uid.clone(), (*generation, err.clone()));
    }
    last.retain(|k, _| seen.contains(k));
    changed
}

/// How often the drained reports are broadcast — the event rate, distinct from the drain, which
/// is EVENT-WOKEN: a node's report notifies the waker, so nothing polls to discover one.
const BROADCAST_PERIOD: Duration = Duration::from_millis(500);

/// The background worker a live server needs — the status drain: take every node's reports, apply
/// them to the graph, and broadcast the events that carry them.
///
/// It must never `set_dirty(true)` — a node reporting its own state is not a user edit — and must
/// FORGET a uid on removal, so a stale error cannot outlive its node.
pub fn spawn_workers(state: &AppState) {
    let (graph, events) = (state.graph.clone(), state.events.clone());
    std::thread::spawn(move || {
        let waker = graph.lock().unwrap().drain_waker();
        let period = BROADCAST_PERIOD;
        let mut last_errors: HashMap<String, (u64, Option<String>)> = HashMap::new();
        // A node's stage changes on its own thread, with no RPC to ride on. It carries the error
        // too, because a facade HAS health and never reports one.
        let mut last_stages: HashMap<String, NodeState> = HashMap::new();
        let mut next_broadcast = Instant::now() + period;
        loop {
            // Parked until a report lands or the broadcast pace comes due — pacing, not polling.
            let wait = next_broadcast.saturating_duration_since(Instant::now()).min(period);
            waker.wait_timeout(wait);
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
                    let mut errs: Vec<(String, u64, Option<String>)> = Vec::new();
                    let mut stages: Vec<(String, NodeState)> = Vec::new();
                    let mut expr_vals: Vec<(String, Value)> = Vec::new();
                    let leaves = g.node_uids();
                    for u in g.all_uids() {
                        let hex = u.to_hex();
                        if let Some(f) = g.node_ufreq(u) {
                            rates.push((hex.clone(), f));
                        }
                        // The WHOLE map, for every node with a driven param — an empty one is how
                        // a client learns a value was withdrawn.
                        if g.driven(u) {
                            expr_vals.push((hex.clone(), schemas::expression_value_map(g, u)));
                        }
                        let generation = g.node_generation(u);
                        let err = g.last_error(u).map(str::to_string);
                        // The console is a transcript of REPORTS, so only a node that runs enters it.
                        if leaves.contains(&u) {
                            errs.push((hex.clone(), generation, err.clone()));
                        }
                        // The tier rides the TRANSITION channel, not the snapshot alone: a node
                        // added after connecting is in no snapshot, and a demotion changes it live.
                        let tier = g.node_tier(u).map(goofi_node::Isolation::wire);
                        stages.push((hex, (generation, g.node_stage(u), err, tier)));
                    }
                    let refreshed: Vec<String> = refreshed
                        .into_iter()
                        .filter(|(uid, _)| leaves.contains(uid))
                        .map(|(uid, key)| {
                            param_state_update(g, uid, &[(&key.group, &key.name)])
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
                let err = errs.iter().find(|(h, ..)| *h == hex).and_then(|(.., e)| e.clone());
                let _ = events.send(event("error", json!({ "node": hex, "error": err })));
            }
            last_stages.retain(|h, _| stages.iter().any(|(s, ..)| s == h));
            for (node, now) in stages {
                if last_stages.get(&node) == Some(&now) {
                    continue;
                }
                let ev =
                    json!({ "node": &node, "stage": now.1, "error": &now.2, "runtime": now.3 });
                let _ = events.send(event("node_stage", ev));
                last_stages.insert(node, now);
            }
        }
    });
}

/// What the state sweep diffs per node: generation, stage, last error, and the tier it runs on.
/// A change in any of them is one `node_stage` event.
type NodeState = (u64, &'static str, Option<String>, Option<&'static str>);

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

pub async fn serve_app(
    listener: tokio::net::TcpListener,
    state: AppState,
    spa: Spa,
    dev_routes: bool,
) -> std::io::Result<()> {
    axum::serve(listener, app(state, spa, dev_routes)).await
}

/// The composed graph the app boots: the model plus the signal engine, registered first. The
/// one anchor that makes the linker keep goofi-nodes' inventory registrations lives here, at the
/// composition root.
pub fn fresh_graph() -> Graph {
    let _ = goofi_nodes::native_node_count();
    let mut g = Graph::new();
    let signal = goofi_signal::SignalEngine::new(
        g.instance().to_string(),
        g.patch_start(),
        g.drain_waker(),
    );
    g.register_engine(Box::new(signal));
    g
}

/// The signal engine registered in `g` — the composition root's reach to its concrete doors.
pub fn signal_engine(g: &mut Graph) -> &mut goofi_signal::SignalEngine {
    goofi_signal::SignalEngine::of(g.engine_mut("signal").expect("the signal engine is registered"))
        .expect("the `signal` registration is the signal engine")
}

/// Register a runtime type AND clear its greyed row — the one owner of that pairing: a name must
/// never hold a live registration and an unavailable row at once.
pub fn register_dyn_type(
    g: &mut Graph,
    manifest: &'static goofi_node::NodeManifest,
    factory: goofi_signal::discover::NodeFactory,
    tier: &'static goofi_node::IsolationCell,
) -> goofi_signal::Registration {
    let r = signal_engine(g).register_dyn_type(manifest, factory, tier);
    if r != goofi_signal::Registration::Refused {
        g.forget_unavailable(manifest.type_name);
    }
    r
}

/// Forget a runtime type from BOTH registries — the registry and the greyed overlay.
pub fn remove_dyn_type(g: &mut Graph, type_name: &str) -> bool {
    let had = signal_engine(g).remove_dyn_type(type_name);
    g.forget_unavailable(type_name) || had
}

/// One output slot's data service name — the resolver over the graph's own birth facts. Also the
/// `/data` plane's subscribe address.
pub fn output_service_of(g: &Graph, uid: goofi_graph::Uid, slot: &str) -> String {
    goofi_transport::output_service(
        &goofi_transport::service_base(g.instance(), uid, g.node_generation(uid)),
        slot,
    )
}

/// Every node type name visible in the catalog — all engines' libraries plus the unavailable
/// overlay, `_`-prefixed test nodes hidden.
pub fn catalog_type_names(g: &Graph) -> Vec<String> {
    g.library_manifests()
        .into_iter()
        .filter(|m| !m.type_name.starts_with('_'))
        .map(|m| m.type_name.to_string())
        .chain(g.unavailable_types().map(|(name, _)| name.to_string()))
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
    pub registration: goofi_signal::Registration,
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
            if t.registration != goofi_signal::Registration::Refused {
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
        remove_dyn_type(g, name);
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

/// `POST /exec {commands, actor}` — the CLI's door, sharing the MCP tool's parse and batch
/// semantics verbatim. Each entry answers its JSON and its rendered text, so the client prints
/// without op knowledge; a refusal is one `{error}`, since a batch lands whole or not at all.
async fn exec_endpoint(State(state): State<AppState>, body: String) -> Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    let Ok(req) = serde_json::from_str::<Value>(&body) else {
        return (StatusCode::BAD_REQUEST, axum::Json(json!({ "error": "the body is JSON: {commands, actor}" })))
            .into_response();
    };
    let lines: Vec<String> = req["commands"]
        .as_array()
        .map(|c| c.iter().map(|l| l.as_str().unwrap_or_default().to_string()).collect())
        .unwrap_or_default();
    let actor = req["actor"].as_str().unwrap_or(DEFAULT_ACTOR).to_string();
    // Off the async workers: a batch can hold the graph lock for seconds (a `session load`
    // provisions nodes), and the sockets must keep being polled meanwhile.
    let ran = tokio::task::spawn_blocking(move || phrase::exec_lines(&state, &lines, &actor))
        .await
        .unwrap_or_else(|e| Err(format!("the exec task died: {e}")));
    match ran {
        Ok(results) => {
            let entries: Vec<Value> = results
                .iter()
                .map(|r| json!({ "result": r, "text": phrase::render(r) }))
                .collect();
            axum::Json(json!({ "results": entries })).into_response()
        }
        Err(e) => {
            (StatusCode::BAD_REQUEST, axum::Json(json!({ "error": e }))).into_response()
        }
    }
}

/// The two messages that seed (or re-seed) a control socket: the hello snapshot, then the whole
/// document. The filesystem reads — the mount walk, the agents config — happen BEFORE the graph
/// lock is taken, because no filesystem read may run while the status-drain worker waits on it.
fn control_seeds(state: &AppState) -> (String, String) {
    let unsaved = state.is_dirty();
    let saved_at = state.save_path();
    let roster = state.harnesses.roster(&goofi_core::home::agents());
    let hello = {
        let g = state.graph.lock().unwrap();
        event(
            "hello",
            schemas::snapshot(&g, &state.instance_id, true, unsaved, saved_at.as_deref(), roster),
        )
    };
    (hello, doc_state(state))
}

async fn handle_control(socket: WebSocket, state: AppState) {
    let (mut tx, mut rx) = socket.split();

    // Subscribe BEFORE snapshotting the document: in the other order a peer's edit lands in
    // neither, and the replica desyncs silently. A re-delivery is read as stale and skipped.
    let mut events = state.events.subscribe();

    let (hello, doc) = control_seeds(&state);
    if tx.send(Message::Text(hello.into())).await.is_err() {
        return;
    }
    if tx.send(Message::Text(doc.into())).await.is_err() {
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
                    let (hello, doc) = control_seeds(&state);
                    if tx.send(Message::Text(hello.into())).await.is_err() {
                        break;
                    }
                    if tx.send(Message::Text(doc.into())).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
        }
    }
    farewell(tx, rx, 1000, "").await;
}

impl AppState {
    /// Whether the patch differs from its last saved state. TWO sources, because a patch is a graph
    /// AND a workspace, and the workspace half is walked on ask rather than watched.
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Relaxed)
            || goofi_graph::archive::fingerprint(&self.mount()) != *self.workspace_baseline.lock().unwrap()
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

/// A per-node `state_update` event carrying a node's current params and error. `refreshed` names
/// the params whose ⟳ refresh just completed — it must be sent on EVERY outcome, a refresh that
/// found nothing included, or the button spins on.
fn param_state_update(g: &Graph, peer: Uid, refreshed: &[(&str, &str)]) -> String {
    let Value::Object(mut body) = schemas::runtime_json(g, peer) else {
        unreachable!("runtime_json builds an object")
    };
    body.insert("node".into(), json!(peer.to_hex()));
    body.insert("params".into(), schemas::describe_node_params(g, peer));
    body.insert(
        "refreshed_params".into(),
        refreshed.iter().map(|(g, n)| json!([g, n])).collect(),
    );
    event("state_update", Value::Object(body))
}

fn parse_uid(g: &goofi_graph::Graph, payload: &Value, key: &str) -> Result<Uid, String> {
    let raw = payload
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing/invalid uid `{key}`"))?;
    g.resolve_ref(raw)
        .ok_or_else(|| format!("`{raw}` names no node — `{key}` takes a uid or a node's name"))
}

/// An OPTIONAL node reference: absent is `None`, present must resolve.
fn parse_uid_opt(
    g: &goofi_graph::Graph,
    payload: &Value,
    key: &str,
    op: &str,
) -> Result<Option<Uid>, String> {
    match payload.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .and_then(|s| g.resolve_ref(s))
            .map(Some)
            .ok_or_else(|| format!("{op}: `{key}` names no node")),
    }
}

/// A required uid ARRAY, refused whole rather than silently short: a caller that named one bad
/// uid asked for a batch that is not the one it would get.
fn parse_uid_list(g: &goofi_graph::Graph, payload: &Value, key: &str) -> Result<Vec<Uid>, String> {
    let arr = payload.get(key).and_then(|v| v.as_array()).ok_or_else(|| format!("missing {key}"))?;
    let uids: Vec<Uid> =
        arr.iter().filter_map(|m| m.as_str().and_then(|s| g.resolve_ref(s))).collect();
    match uids.len() == arr.len() {
        true => Ok(uids),
        false => Err(format!("an entry in `{key}` names no node")),
    }
}

/// A required string field from an RPC payload.
fn parse_str<'a>(payload: &'a Value, key: &str) -> Result<&'a str, String> {
    payload.get(key).and_then(|v| v.as_str()).ok_or_else(|| format!("missing {key}"))
}

fn parse_pos(v: &Value) -> Option<[f64; 2]> {
    let a = v.as_array()?;
    if a.len() != 2 {
        return None;
    }
    Some([a[0].as_f64()?, a[1].as_f64()?])
}

/// Which side of a target a newcomer lands on. ONE argument, because an axis and a half are two
/// halves of one answer and two arguments can disagree; absent defaults right, and a present
/// value that is not a side word is refused rather than defaulted.
fn parse_side(p: &Value, op: &str) -> Result<goofi_graph::layout::Side, String> {
    match p.get("side").filter(|v| !v.is_null()) {
        None => Ok(goofi_graph::layout::Side::Right),
        Some(v) => v
            .as_str()
            .and_then(goofi_graph::layout::Side::parse)
            .ok_or_else(|| format!("{op}: side is `left`, `right`, `top` or `bottom`, not {v}")),
    }
}

/// An `endpoint` — `node/slot`, split on the FIRST `/`, the node half a uid or a name. The slot
/// half may itself be a port uid (wiring a facade from outside), so it is never validated here.
fn parse_endpoint(
    g: &goofi_graph::Graph,
    p: &Value,
    op: &str,
    key: &str,
) -> Result<(Uid, String), String> {
    let raw =
        p.get(key).and_then(|v| v.as_str()).ok_or_else(|| format!("{op}: missing {key}"))?;
    let (node, slot) =
        raw.split_once('/').ok_or_else(|| format!("{op}: `{key}` is `node/slot`, not `{raw}`"))?;
    let uid = g.resolve_ref(node)
        .ok_or_else(|| format!("{op}: `{node}` in `{key}` names no node"))?;
    Ok((uid, slot.to_string()))
}

fn parse_link(
    g: &goofi_graph::Graph,
    p: &Value,
    op: &str,
) -> Result<(Uid, String, Uid, String), String> {
    let (node_out, slot_out) = parse_endpoint(g, p, op, "from")?;
    let (node_in, slot_in) = parse_endpoint(g, p, op, "to")?;
    Ok((node_out, slot_out, node_in, slot_in))
}


/// Resolve a link endpoint AND refuse one that names nothing wirable — the check a caller-initiated
/// `add_link` gets and a REPLAY does not, since a replay must converge rather than wedge the stack.
fn wirable_endpoint(g: &Graph, uid: Uid, slot: &str, which: &str) -> Result<(Uid, String), String> {
    let (node, slot) = g.normalise(uid, slot);
    if g.wirable(node) {
        return Ok((node, slot));
    }
    // A FACADE is a node that exists and simply has no slot by that name — saying it names nothing
    // sends a caller looking for the wrong mistake.
    match g.is_facade(uid) {
        true => Err(format!("link add: `{which}` names sub-patch {} — name one of its ports as the slot", uid.to_hex())),
        false => Err(format!("link add: `{which}` names no node in this patch: {}", uid.to_hex())),
    }
}

/// Is `node` something a panel could bind to? A UID, and only a uid: a display name stops resolving
/// the moment somebody renames the node. A boundary port counts — it exposes a real stream.
fn bindable_node(g: &Graph, node: &str) -> bool {
    Uid::from_hex(node).is_some_and(|u| g.exists(u))
}

/// Route a layout planner's per-entry writes through the command history as ONE undo step, and
/// answer with the arrangement they produced, drawn as `layout inspect` draws it.
fn apply_layout(
    state: &AppState,
    g: &mut Graph,
    actor: &str,
    cmd: goofi_graph::Command,
) -> Result<Value, String> {
    state.history.lock().unwrap().apply(g, actor, cmd)?;
    Ok(json!({ "text": inspect::layout_tree(g.arrangement(), None) }))
}

impl AppState {
    /// The op rows this instance serves.
    pub fn ops(&self) -> &[&'static ops::Op] {
        &self.ops
    }

    /// The served row for `name` — absent rows (headless's layout group) answer `unknown op`.
    pub fn find_op(&self, name: &str) -> Option<&'static ops::Op> {
        self.ops.iter().find(|o| o.name == name).copied()
    }

    /// Run one control op — the single entry point every surface shares. `actor` scopes the undo
    /// history: whose undo, the way a browser tab's id does. The op's row does the work: its handler
    /// runs, and its KIND decides the tail — a Write mutated the graph through the history, so
    /// ONE re-mirror and one dirty decision happen here, where no write arm can forget either; a
    /// Read touches nothing; an Effect's arm owns its own consequences.
    pub fn call(&self, op: &str, payload: Value, actor: &str) -> Result<Value, String> {
        let Some(spec) = self.find_op(op) else {
            return Err(format!("unknown op `{op}`"));
        };
        let mut events: Vec<String> = Vec::new();
        let result = spec.handler.run(self, &payload, actor, &mut events);
        if result.is_ok() && spec.handler.is_write() {
            resync_and_broadcast(self);
            events.extend(self.set_dirty(true));
        }
        for e in events {
            let _ = self.events.send(e);
        }
        result
    }
}

/// The `/control` envelope over [`AppState::call`]: `{id, op, payload, actor}` in, `{id, result}`
/// or `{id, error}` out. A request with no numeric `id` wants no reply.
fn dispatch(state: &AppState, text: &str) -> Option<String> {
    let req: Value = serde_json::from_str(text).ok()?;
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let op = req.get("op")?.as_str()?.to_string();
    let payload = req.get("payload").cloned().unwrap_or_else(|| json!({}));
    // The ACTOR scopes the undo history — whose undo, where `GOOFI_SESSION` says which server.
    // Absent ⇒ the one shared actor, so a caller that presents none still works.
    let actor = req.get("actor").and_then(|v| v.as_str()).unwrap_or(DEFAULT_ACTOR).to_string();

    let result = state.call(&op, payload, &actor);
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
    let mut g = state.graph.lock().unwrap();
    // The settle point: one delivery per batch, before the projection, from settled state.
    g.settle();
    let mut doc = state.doc.lock().unwrap();
    remirror_and_broadcast_locked(state, &g, &mut doc);
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
        farewell(tx, rx, 4004, "unknown harness instance").await;
        return;
    };
    // Attach snapshots the tail and subscribes as one step, so what the child wrote before this
    // socket arrived replays first and the live stream follows with no byte lost or doubled.
    let term::Attached { tail, mut output, mut exit, mut eof } = inst.attach();
    let (seat, mut size) = inst.join();
    if !tail.is_empty() && tx.send(Message::Binary(tail.into())).await.is_err() {
        return;
    }
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
    farewell(tx, rx, 1000, "").await;
}

/// How EVERY socket here ends: by the handshake, never by dropping. A dropped connection is
/// RESET, and a reset discards what is still in flight — so a peer that was not reading at that
/// moment loses the last frames, the exit code and a refusal's own close code most of all.
async fn farewell(
    mut tx: futures_util::stream::SplitSink<WebSocket, Message>,
    mut rx: futures_util::stream::SplitStream<WebSocket>,
    code: u16,
    reason: &str,
) {
    if tx.send(close(code, reason)).await.is_err() {
        return;
    }
    // The peer's own close is what says it read everything; the bound is for one that never sends it.
    let _ = tokio::time::timeout(FAREWELL, async { while rx.next().await.is_some() {} }).await;
}

fn size_frame(cols: u16, rows: u16) -> Message {
    Message::Text(json!({ "op": "size", "cols": cols, "rows": rows }).to_string().into())
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
            farewell(tx, rx, 4004, "bad node uid").await;
            return;
        }
    };
    // The address must NAME an output slot — key or display label, resolved exactly as a
    // snapshot's is — of a leaf, a port or a facade alike. What is behind it is a separate
    // question, asked again below, because a port with nothing wired yet is a real node with no
    // data, exactly as a leaf nobody has connected is.
    let named = {
        let g = state.graph.lock().unwrap();
        vocab::resolve_slot(&g, "data", uid, &slot).ok()
    };
    let Some(slot) = named else {
        farewell(tx, rx, 4004, "unknown node/slot").await;
        return;
    };

    // The SHARED per-slot reducer, keyed on the PHYSICAL slot: a viewer on a facade port, one on the
    // port inside the sub-patch and one on the leaf itself all coalesce onto the same one. Which
    // physical slot a port stands in front of is graph state, so the socket re-asks rather than
    // freezing the answer at open — a port wired later starts drawing, and a re-wire is followed.
    let conn = state.reducers.new_conn();
    let mut key = stream_behind(&state.graph.lock().unwrap(), uid, &slot);
    let mut frames = key.clone().map(|k| state.reducers.subscribe(k, conn));
    let mut specs: Vec<goofi_view::ViewSpec> = Vec::new();
    let mut rehome = tokio::time::interval(reducer::REHOME_INTERVAL);

    // A dead-but-not-closed peer produces NO socket error, so without an active probe this
    // connection would live forever.
    let cfg = state.data_liveness;
    let mut live = PeerLiveness::new(cfg);
    let mut keepalive = tokio::time::interval(cfg.ping_interval);

    loop {
        let mut recheck = false;
        tokio::select! {
            frame = next_frame(&mut frames) => match frame {
                Ok(bytes) => {
                    // Giving up on a frame is NOT a liveness signal in either direction: only the
                    // pong decides.
                    match send_bounded(&mut tx, Message::Binary(bytes), cfg.send_timeout).await {
                        SendOutcome::Sent => {}
                        // The sweep will not resend an unchanged frame, so a dropped one must be
                        // asked for again.
                        SendOutcome::Dropped => reoffer(&state, &key),
                        SendOutcome::Gone => break, // the socket really is gone
                    }
                }
                // A lagged viewer drops frames rather than stalling the shared reducer, and
                // re-offers because the missed frame may have been the last.
                Err(broadcast::error::RecvError::Lagged(_)) => reoffer(&state, &key),
                Err(broadcast::error::RecvError::Closed) => break,
            },
            incoming = rx.next() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                Some(Ok(Message::Text(t))) => {
                    if let Ok(m) = serde_json::from_str::<ViewMsg>(t.as_str()) {
                        if m.op == "view" {
                            // Held, because a re-subscribe onto another physical slot has to carry
                            // this viewer's constraints with it or it asks for full resolution.
                            specs = m.specs;
                            if let Some(k) = &key {
                                state.reducers.set_specs(k, conn, specs.clone());
                            }
                        }
                    }
                }
                Some(Ok(Message::Pong(_))) => live.pong(),
                _ => {}
            },
            _ = rehome.tick() => recheck = true,
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
        if recheck {
            let want = stream_behind(&state.graph.lock().unwrap(), uid, &slot);
            if want != key {
                if let Some(old) = &key {
                    state.reducers.unsubscribe(old, conn);
                }
                frames = want.clone().map(|k| {
                    let rx = state.reducers.subscribe(k.clone(), conn);
                    state.reducers.set_specs(&k, conn, specs.clone());
                    rx
                });
                key = want;
            }
        }
    }
    if let Some(k) = &key {
        state.reducers.unsubscribe(k, conn);
    }
    farewell(tx, rx, 1000, "").await;
}

/// The physical `(node, slot)` a `/data` address stands for — the engine's ONE wiring resolution.
/// `None` while nothing is behind a port yet — a wait, never an error.
fn stream_behind(g: &Graph, uid: Uid, slot: &str) -> Option<reducer::SlotKey> {
    match g.stream(uid, slot) {
        Some(goofi_graph::Stream::At(leaf, s)) => Some((leaf, s.to_string())),
        _ => None,
    }
}

/// The next frame, or a future that never finishes while nothing is behind the address yet — which
/// is what keeps an unwired port's socket open and silent instead of closed.
async fn next_frame(
    frames: &mut Option<broadcast::Receiver<axum::body::Bytes>>,
) -> Result<axum::body::Bytes, broadcast::error::RecvError> {
    match frames {
        Some(f) => f.recv().await,
        None => std::future::pending().await,
    }
}

fn reoffer(state: &AppState, key: &Option<reducer::SlotKey>) {
    if let Some(k) = key {
        state.reducers.reoffer(k);
    }
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
