//! goofi-bridge — the axum HTTP/WebSocket server that exposes the engine to the
//! browser: the `/control` JSON RPC + broadcast-event plane and the
//! `/data/<node>/<slot>` binary GOOF plane — ONE reduced stream per (node, slot);
//! the viewer kind is NOT in the path (viewers send their ViewSpec inband via
//! `{op:"view"}`). Event-sourced: RPCs return thin acks; real state changes arrive
//! as broadcast events the client applies. The built SPA is served from disk
//! (`frontend/build`, or `GOOFI_FRONTEND_BUILD`) via `ServeDir`.

mod crdt_mirror;
mod fsbrowse;
mod inspect;
pub mod ops;
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
    /// How a directory of node files becomes registered node types. Injected by the CLI at boot
    /// (see [`NodeScan`]); the default discovers nothing.
    pub scan_nodes: NodeScan,
    /// The shipped node directory — `nodes/`, or whatever `--auto-nodes` named. `None` when the
    /// binary was launched with no auto-routed source. Boot-time config, set alongside the seam.
    pub system_nodes: Option<PathBuf>,
    /// What the last scan found, by type name → the file's stamp. The baseline the next [`rescan`]
    /// diffs against, and the list it removes from — so a type registered some other way (a
    /// `--subproc-nodes` directory, a test) is never swept up by a rescan of these two trees.
    node_index: Arc<Mutex<std::collections::BTreeMap<String, Option<Stamp>>>>,
    /// Where the open patch's workspace files live while it is open — the tree a `.gfi` packs and
    /// unpacks. Created at boot, dropped by [`AppState::release_mount`] on a graceful exit; after a
    /// crash it simply stays, because a reboot clears the system temp directory.
    ///
    /// Shared and private because a load REPLACES it (the loaded patch brings its own workspace)
    /// while every handler holds its own clone of the state — one stored path, read through
    /// [`AppState::mount`], is the single source of truth for where the workspace is right now.
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
        // The baseline is the fingerprint of whatever mount the patch owns — stated that way even
        // at boot, where the mount is empty, so the invariant has one spelling everywhere.
        let mount = new_mount();
        let workspace_baseline = goofi_engine::archive::fingerprint(&mount);
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
            scan_nodes: Arc::new(|_, _| Vec::new()),
            system_nodes: None,
            node_index: Arc::new(Mutex::new(Default::default())),
            mount: Arc::new(Mutex::new(mount)),
            workspace_baseline: Arc::new(Mutex::new(workspace_baseline)),
            save_path: Arc::new(Mutex::new(None)),
        }
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

/// The front half of a load, run against a mount that is not yet live: obtain the patch's manifest
/// and the file it came from. `load` unpacks the archive's workspace tree into `mount`; `load_text`
/// (a browser upload) and `new` cannot carry a workspace, so their `mount` stays the empty one the
/// caller made.
///
/// It stops at the manifest rather than applying it, because the patch's own node types live in the
/// tree it just unpacked and have to be registered BEFORE `load_doc` resolves the graph — see the
/// arm. Nothing here touches state the caller has not already agreed to lose: `mount` is not yet the
/// live one, and the graph is untouched.
fn stage_load(
    mount: &std::path::Path,
    op: &str,
    payload: &Value,
) -> Result<(String, Option<String>), String> {
    let (content, from_path) = if op == "new" {
        // A New patch IS a load — of an empty patch, from nowhere. Routing it through `load_doc`
        // rather than `Graph::clear` is what stops the two from drifting: `clear` deliberately
        // keeps the editor layout (it is not graph content, and only a load overwrites it), which
        // is exactly how New used to inherit the previous patch's panels. Whatever a load resets,
        // New resets, by construction. The live `g` is reused rather than replaced, so the
        // catalog — runtime-registered Python types, the expression evaluator — survives.
        (Graph::new().serialize(), None)
    } else if op == "load" {
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
    Ok((content, from_path))
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

/// The node-discovery seam: scan ONE directory, registering every node file in it into `g`
/// (replacing a type of the same name), and report what it did.
///
/// Injected by the CLI at boot — per-file TIER ROUTING lives in the binary, which owns the
/// interpreters and the probe, not here — so boot and [`rescan`] re-derive the registry through the
/// very same function rather than two implementations that can drift. The default is a no-op: a
/// bridge with no discovery behind it (every test that does not inject one) discovers nothing.
pub type NodeScan = Arc<dyn Fn(&mut Graph, &std::path::Path) -> Vec<ScannedType> + Send + Sync>;

/// What a [`rescan`] changed, for the caller that asked — an agent that just wrote a node file, or
/// the palette's refresh button.
#[derive(Default)]
pub struct ScanDiff {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
}

/// Re-derive the runtime node registry from the directories that exist RIGHT NOW: the shipped node
/// directory, then `<patch>/nodes` — in that order, so a patch-local node of the same name wins,
/// which is what "patch node" means (it falls out of replace-on-register).
///
/// The previous scan's stamps are the baseline, so what comes back is a diff rather than a listing.
/// Removal is driven by that baseline too, which is what keeps a type discovered some OTHER way —
/// a `--subproc-nodes` directory, a test's direct registration — out of the blast radius.
///
/// `pub`, and returning the raw per-file outcomes beside the diff, for ONE caller: the CLI's boot
/// scan runs this rather than the seam directly, so the baseline the first refresh diffs against is
/// the boot scan itself (otherwise the first press of refresh re-announces the whole shipped tree as
/// new) — and so boot reports outcomes the seam deliberately does not print.
pub fn rescan(
    state: &AppState,
    g: &mut Graph,
    patch: &std::path::Path,
) -> (ScanDiff, Vec<ScannedType>) {
    let mut found: std::collections::BTreeMap<String, Option<Stamp>> = Default::default();
    let mut patch_types: HashSet<String> = HashSet::new();
    let mut outcomes = Vec::new();
    let dirs = [(state.system_nodes.clone(), false), (Some(patch.join("nodes")), true)];
    for (dir, is_patch) in dirs {
        let Some(dir) = dir.filter(|d| d.is_dir()) else { continue };
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

/// Restart every live instance of a type whose file changed, so editing a node file makes the nodes
/// already on the canvas run the new code (decision, 2026-08-09). `setup()` re-runs — a buffer
/// empties, a device reopens — which is the accepted price of that.
///
/// Deliberately NOT part of [`rescan`]: a load re-derives the registry for a graph that is about to
/// be replaced wholesale, and restarting the outgoing patch's nodes there would be pure cost.
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

    // Subscribe to every broadcast plane BEFORE snapshotting. If we snapshotted first and
    // subscribed second, a mutation landing in that window would be neither in the snapshot nor
    // delivered — the client's mirror would silently desync. Subscribing first can at worst
    // re-deliver an event already reflected in the snapshot, which every apply branch absorbs
    // idempotently. The CRDT plane subscribes first for the same reason.
    let mut events = state.events.subscribe();
    let mut sync_updates = state.sync_updates.subscribe();

    // Answered BEFORE the graph lock is taken: it walks the workspace mount (see `is_dirty`), and
    // no filesystem walk may run while the tick thread is waiting on that lock.
    let unsaved = state.is_dirty();
    let saved_at = state.save_path();
    let hello = {
        let g = state.graph.lock().unwrap();
        event(
            "hello",
            schemas::snapshot(&g, &state.instance_id, true, unsaved, saved_at.as_deref()),
        )
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
                    let unsaved = state.is_dirty(); // off the graph lock, as above
                    let saved_at = state.save_path();
                    let hello = {
                        let g = state.graph.lock().unwrap();
                        event(
                            "hello",
                            schemas::snapshot(&g, &state.instance_id, true, unsaved, saved_at.as_deref()),
                        )
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
    /// Whether the patch differs from its last saved/loaded state — the title-bar dot and the
    /// unload guard.
    ///
    /// TWO independent sources, because a patch is a graph AND a workspace tree. The graph half is
    /// the flag every mutating op sets. The workspace half is a directory goofi does not own: the
    /// agent it will host, or the user's editor, writes into it with no RPC to ride on. There is no
    /// watcher (decision, 2026-08-09), so the question is answered by WALKING the mount at the
    /// moment a client asks it — an external edit surfaces on the asker's next `hello`, and no
    /// thread wakes up to hunt for one. Walk it OFF the graph lock: a workspace of code files is a
    /// stat per file, and the tick holds that lock.
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

fn event(name: &str, payload: Value) -> String {
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

/// Route a layout planner's per-entry writes through the command history as ONE undo step. This is
/// the whole of "layout reuses the graph machinery": persistence, the CRDT broadcast and per-session
/// undo all follow from the write being an ordinary command.
///
/// `born` names the entry an op BRINGS INTO BEING (`(page, id)`), and changes what undo means: the
/// slots the writes displaced are no longer the inverse — closing `born` with promote is. That is
/// what keeps a foreign undo from deleting a subtree a peer grew under the newcomer. See
/// [`goofi_engine::Command::LayoutBirth`]. An op that only rearranges what already exists passes
/// `None` and inverts slot by slot.
fn apply_layout(
    state: &AppState,
    g: &mut Graph,
    session: &str,
    writes: Vec<goofi_engine::layout::Write>,
    born: Option<(&str, &str)>,
) -> Result<Value, String> {
    let cmd = match born {
        Some((page, born)) => goofi_engine::Command::LayoutBirth {
            writes,
            page: page.to_string(),
            born: born.to_string(),
        },
        None => goofi_engine::Command::Compound(
            writes
                .into_iter()
                .map(|(id, entry)| goofi_engine::Command::EditLayoutEntry { id, entry })
                .collect(),
        ),
    };
    state.history.lock().unwrap().apply(g, session, cmd)?;
    Ok(json!({ "ok": true }))
}

/// Like [`apply_layout`], but for an op that CLOSES the subtree rooted at `born` (a page goes with
/// its own, like `session_remove_page`). Its inverse restores those dead entries and then re-homes
/// their root through the forward planners — pinning it back into the slot it held resurrects the
/// split the close promoted away, on top of whatever a peer built there. See
/// [`goofi_engine::Command::LayoutClose`].
fn apply_layout_close(
    state: &AppState,
    g: &mut Graph,
    session: &str,
    page: &str,
    born: &str,
) -> Result<Value, String> {
    let cmd = goofi_engine::Command::LayoutClose { page: page.to_string(), born: born.to_string() };
    state.history.lock().unwrap().apply(g, session, cmd)?;
    Ok(json!({ "ok": true }))
}

/// Like [`apply_layout`], but for an op that MOVES the subtree rooted at `root`. Its inverse is
/// another move, re-planned at undo time (see [`goofi_engine::Command::LayoutMove`]) — restoring the
/// slots a move displaced resurrects the split it promoted away, and strands whatever a peer built
/// where that split used to stand.
fn apply_layout_move(
    state: &AppState,
    g: &mut Graph,
    session: &str,
    writes: Vec<goofi_engine::layout::Write>,
    root: &str,
) -> Result<Value, String> {
    let cmd =
        goofi_engine::Command::LayoutMove { writes: Some(writes), root: root.to_string(), home: None };
    state.history.lock().unwrap().apply(g, session, cmd)?;
    Ok(json!({ "ok": true }))
}

/// Like [`apply_layout`], but for an op that edits what entries HOLD rather than where they sit (a
/// panel's type/state, a split's shares). Its inverse re-reads each slot at flip time instead of
/// restoring the whole entry — see [`goofi_engine::Command::LayoutContents`].
fn apply_layout_contents(
    state: &AppState,
    g: &mut Graph,
    session: &str,
    writes: Vec<goofi_engine::layout::Write>,
) -> Result<Value, String> {
    let cmd = goofi_engine::Command::LayoutContents { writes };
    state.history.lock().unwrap().apply(g, session, cmd)?;
    Ok(json!({ "ok": true }))
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

    // Every op is declared once, in `ops::REGISTRY`. Refusing an unregistered one HERE is what
    // makes a dispatch arm without a row unreachable rather than a second, invisible declaration
    // of the op set — and it is where `read_only` comes from below, so the classification a new op
    // needs lives beside the op instead of in a parallel list that can disagree with it.
    let spec = ops::find(&op);
    let mut events: Vec<String> = Vec::new();
    let result: Result<Value, String> = (|| {
        if spec.is_none() {
            return Err(format!("unknown op `{op}`"));
        }
        // Ops that read no graph state are served WITHOUT the graph mutex. `list_dir` walks a
        // directory and stats every child, which can block for a long time on a huge or network
        // path — under the lock that would stall the tick thread for the whole walk. `get_patch`
        // is here for the same reason: `is_dirty` walks the workspace mount.
        if op == "list_dir" {
            return Ok(fsbrowse::list_dir(payload.get("path").and_then(|v| v.as_str())));
        }
        if op == "get_patch" {
            return Ok(json!({
                "save_path": state.save_path(),
                "workspace": state.mount().to_string_lossy(),
                "dirty": state.is_dirty(),
            }));
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
                // The freshly-enumerated list rides the REPLY as well as the event: a caller that
                // is not the editor (an agent picking a device) would otherwise have to guess
                // which broadcast belongs to its request.
                let options = g.refresh_param(uid, &group, &name)?;
                events.push(param_state_update_refreshed(&g, uid, &[(&group, &name)]));
                Ok(json!({ "options": options }))
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
            "inspect_layout" => Ok(json!({ "text": inspect::layout_tree(g.arrangement()) })),
            "session_list_pages" => Ok(json!({ "pages": inspect::layout_pages(g.arrangement()) })),
            "page_list_panels" => {
                let page = resolve_page(&g, &payload)?;
                Ok(json!({ "text": inspect::panel_table(g.arrangement(), &page) }))
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
                    Some(s) => apply_layout_move(state, &mut g, &session, writes, s)?,
                    None => apply_layout(state, &mut g, &session, writes, Some((&page, &page)))?,
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
                apply_layout_close(state, &mut g, &session, &page, &page)
            }
            "session_rename_page" => {
                let (from, to) = (parse_str(&payload, "from")?, parse_str(&payload, "to")?);
                let writes = g.arrangement().rename_page(from, to)?;
                // A name is contents; the tab index is the slot, and a peer's new page may hold the
                // one this page had when the rename was planned.
                apply_layout_contents(state, &mut g, &session, writes)
            }
            "session_reorder_page" => {
                let name = parse_str(&payload, "name")?;
                let to = payload.get("to_index").and_then(|v| v.as_u64()).ok_or("missing to_index")?;
                let writes = g.arrangement().reorder_page(name, to as usize)?;
                apply_layout(state, &mut g, &session, writes, None)
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
                apply_layout(state, &mut g, &session, writes, Some((&page, &fresh)))?;
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
                if let Some(node) = panel_state
                    .as_ref()
                    .and_then(|s| s.get("node"))
                    .and_then(|v| v.as_str())
                    .filter(|n| !n.is_empty())
                {
                    if !bindable_node(&g, node) {
                        return Err(format!("page_set_panel: no node `{node}` in this patch"));
                    }
                }
                let writes = g.arrangement().set_panel(&page, &panel, ty.as_deref(), panel_state)?;
                apply_layout_contents(state, &mut g, &session, writes)
            }
            "page_move_panel" => {
                let page = resolve_page(&g, &payload)?;
                let panel = parse_str(&payload, "panel")?.to_string();
                let dest = parse_str(&payload, "new_parent")?.to_string();
                let at = payload.get("order_index").and_then(|v| v.as_u64()).unwrap_or(0);
                let writes = g.arrangement().move_subtree(&page, &panel, &dest, at as usize)?;
                apply_layout_move(state, &mut g, &session, writes, &panel)
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
                apply_layout_move(state, &mut g, &session, writes, &subtree)
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
                apply_layout_contents(state, &mut g, &session, writes)
            }
            "page_remove_panel" => {
                let page = resolve_page(&g, &payload)?;
                let panel = parse_str(&payload, "panel")?.to_string();
                // Planned only for its teachable refusal — see `session_remove_page` above.
                g.arrangement().remove_subtree(&page, &panel)?;
                apply_layout_close(state, &mut g, &session, &page, &panel)
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
                    inspect::patch(&g, scope, state.save_path().as_deref(), &workspace.to_string_lossy(), dirty)?;
                Ok(json!({ "text": text }))
            }
            "inspect_node" => {
                let uid = parse_uid(&payload, "node")?;
                // The three sections default ON — the op is the cheap peek, and a caller that
                // wants less says so.
                let want = |k: &str| payload.get(k).and_then(|v| v.as_bool()).unwrap_or(true);
                let slot = payload.get("slot").and_then(|v| v.as_str());
                let text = inspect::node(&g, uid, slot, want("params"), want("meta"), want("error"))?;
                Ok(json!({ "text": text }))
            }
            "list_globals" => Ok(inspect::globals(&g)),
            "read_node_source" => {
                // The two trees a scan registers from, patch first — the same precedence the
                // palette's `source` badge reports, so provenance cannot disagree with it.
                let dirs: Vec<(PathBuf, &str)> = [(state.mount().join("nodes"), "patch")]
                    .into_iter()
                    .chain(state.system_nodes.clone().map(|d| (d, "shipped")))
                    .collect();
                inspect::node_source(&g, parse_str(&payload, "type")?, &dirs)
            }
            "serialize" => Ok(json!({ "yaml": g.serialize() })),
            // Where this patch's workspace files live right now. The mount is a per-run temp
            // directory under a random name, so a client — and the agent harness after it — cannot
            // derive it; asking the manager is the only way to open a browser or a shell on it.
            "open_workspace" => Ok(json!({ "path": state.mount().to_string_lossy() })),
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
                // deleting a tree is a walk and the lock guards only the swap.
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
                    schemas::snapshot(&g, &state.instance_id, false, false, from_path.as_deref()),
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

/// The two ops whose old bare `{ok:true}` left the caller unable to see what actually happened.
/// Both answers exist only at runtime — neither reaches the CRDT doc — so without them in the
/// reply a caller with no inspector open (an agent, a script) is simply blind.
#[cfg(test)]
mod result_enrichment_tests {
    use super::*;

    fn call(state: &AppState, op: &str, payload: Value) -> Value {
        let req = json!({ "id": 1, "op": op, "payload": payload }).to_string();
        serde_json::from_str(&dispatch(state, &req).expect("a numeric id is answered")).unwrap()
    }

    #[test]
    fn set_expression_answers_with_the_binding_error_rather_than_a_bare_ok() {
        let state = AppState::new();
        let uid = state.graph.lock().unwrap().add_node("Oscillator", None).unwrap();
        let bind = |expr: &str| {
            call(
                &state,
                "set_expression",
                json!({ "node": uid.to_hex(), "group": "oscillator", "name": "amplitude",
                        "expression": expr, "enabled": true }),
            )
        };
        // A binding that cannot compile is STORED (the source is kept so it can be fixed), so the
        // refusal has to travel in the reply or it is invisible.
        let bad = bind("@@ not an expression @@");
        assert!(
            bad["result"]["error"].as_str().is_some_and(|e| !e.is_empty()),
            "the compile error must ride the reply: {bad}"
        );
        // An empty expression clears the binding — nothing left to report.
        assert_eq!(bind("")["result"]["error"], Value::Null);
        state.release_mount();
    }

    struct Picker;
    impl goofi_node::Node for Picker {
        fn process(
            &mut self,
            _i: &goofi_node::Inputs<'_>,
            _o: &mut goofi_node::Outputs<'_>,
            _c: &mut goofi_node::NodeCtx,
            _p: &goofi_node::Params<'_>,
        ) -> goofi_node::NodeResult {
            Ok(())
        }
        fn on_param_refreshed(
            &mut self,
            _k: &goofi_node::ParamKey,
            _p: &goofi_node::Params<'_>,
        ) -> Option<Vec<String>> {
            Some(vec!["mic-a".into(), "mic-b".into()])
        }
    }
    static PICKER_PARAMS: &[goofi_node::ParamDecl] = &[goofi_node::ParamDecl {
        group: "io",
        name: "device",
        spec: goofi_node::ParamSpec::Str { default: "", options: &[], refresh: true },
        default_expr: None,
        doc: None,
    }];
    static PICKER: goofi_node::NodeManifest = goofi_node::NodeManifest {
        type_name: "Picker",
        category: "test",
        doc: "a refreshable device picker",
        inputs: &[],
        outputs: &[],
        params: PICKER_PARAMS,
        isolation: goofi_node::Isolation::InProcess,
        factory: || Box::new(Picker),
    };

    #[test]
    fn refresh_param_answers_with_the_options_it_just_enumerated() {
        let state = AppState::new();
        let uid = {
            let mut g = state.graph.lock().unwrap();
            g.register_dyn_type(&PICKER, Box::new(|_| Box::new(Picker)));
            g.add_node("Picker", None).unwrap()
        };
        let r = call(
            &state,
            "refresh_param",
            json!({ "node": uid.to_hex(), "group": "io", "name": "device" }),
        );
        assert_eq!(r["result"]["options"], json!(["mic-a", "mic-b"]), "{r}");
        state.release_mount();
    }
}

/// The inspect ARMS, as distinct from the formatters `inspect.rs` pins. Everything between the
/// payload and the formatter's arguments lives only here: which scope was asked for, which
/// sections default on, and the dirtiness that has to be sampled before the lock is taken.
#[cfg(test)]
mod inspect_dispatch_tests {
    use super::*;

    fn call(state: &AppState, op: &str, payload: Value) -> Value {
        let req = json!({ "id": 1, "op": op, "payload": payload }).to_string();
        let reply: Value =
            serde_json::from_str(&dispatch(state, &req).expect("a numeric id is answered")).unwrap();
        reply["result"].clone()
    }

    fn text(state: &AppState, op: &str, payload: Value) -> String {
        call(state, op, payload)["text"].as_str().expect("the op answers with text").to_string()
    }

    #[test]
    fn inspect_patch_reads_the_scope_it_was_asked_for_and_the_dirtiness_it_sampled() {
        let state = AppState::new();
        let (uid, scope) = {
            let mut g = state.graph.lock().unwrap();
            let n = g.add_node("Oscillator", None).unwrap();
            let s = g.group_nodes(&[n], [0.0, 0.0]).unwrap();
            (n, s)
        };
        // No mutation has gone through dispatch, so the patch matches disk.
        let root = text(&state, "inspect_patch", json!({}));
        assert!(root.contains("scope: root") && root.contains("unsaved changes: no"), "{root}");
        // …and the scope argument reaches the formatter rather than being dropped.
        let inner = text(&state, "inspect_patch", json!({ "scope": scope.to_hex() }));
        assert!(inner.contains(&format!("({})", scope.to_hex())), "{inner}");
        assert!(!inner.contains("scope: root"), "{inner}");
        // A write through dispatch dirties the patch, and the header must follow it.
        call(&state, "rename_node", json!({ "node": uid.to_hex(), "name": "src" }));
        let after = text(&state, "inspect_patch", json!({}));
        assert!(after.contains("unsaved changes: yes"), "{after}");
        state.release_mount();
    }

    #[test]
    fn get_patch_answers_with_the_live_dirty_flag_not_a_constant() {
        let state = AppState::new();
        let uid = state.graph.lock().unwrap().add_node("Oscillator", None).unwrap();
        let before = call(&state, "get_patch", json!({}));
        assert_eq!(before["dirty"], json!(false), "{before}");
        assert_eq!(before["workspace"], json!(state.mount().to_string_lossy()), "{before}");
        call(&state, "rename_node", json!({ "node": uid.to_hex(), "name": "src" }));
        assert_eq!(call(&state, "get_patch", json!({}))["dirty"], json!(true));
        state.release_mount();
    }

    #[test]
    fn inspect_node_defaults_every_section_on_and_takes_a_no_for_each() {
        let state = AppState::new();
        let uid = state.graph.lock().unwrap().add_node("Oscillator", None).unwrap();
        // The oscillator emits by wall clock, so the meta line needs time to pass, not just a tick.
        for _ in 0..50 {
            let mut g = state.graph.lock().unwrap();
            g.tick();
            if g.latest_frame(uid, "out").is_some() {
                break;
            }
            drop(g);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let full = text(&state, "inspect_node", json!({ "node": uid.to_hex() }));
        assert!(full.contains("params:"), "params default on: {full}");
        assert!(full.contains("meta: "), "meta defaults on: {full}");
        assert!(full.contains("error:"), "error defaults on: {full}");
        let bare = text(
            &state,
            "inspect_node",
            json!({ "node": uid.to_hex(), "params": false, "meta": false, "error": false }),
        );
        assert!(!bare.contains("params:") && !bare.contains("meta: ") && !bare.contains("error:"), "{bare}");
        state.release_mount();
    }
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
mod workspace_dirty_tests {
    use super::*;

    fn save_to(state: &AppState, target: &std::path::Path) {
        let req = json!({ "id": 1, "op": "save", "payload": { "path": target.to_string_lossy() } });
        let reply = dispatch(state, &req.to_string()).expect("a numeric id gets a reply");
        assert!(reply.contains("result"), "the save is accepted; got {reply}");
    }

    /// A workspace file edited OUTSIDE goofi — by the agent the harness will run in it, or by the
    /// user's own editor — makes the patch differ from its `.gfi` exactly as a moved node does.
    /// There is no watcher (decision, 2026-08-09), so the manager notices by comparing the mount
    /// against the fingerprint it took when it last packed one.
    #[test]
    fn an_external_workspace_edit_makes_the_patch_differ_from_its_archive() {
        let state = AppState::new();
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("patch.gfi");
        std::fs::write(state.mount().join("agent.md"), b"notes").unwrap();
        save_to(&state, &target);
        assert!(!state.is_dirty(), "the patch was just written to disk, workspace and all");

        // A file the archive does not have — what an agent writing into the workspace does.
        std::fs::write(state.mount().join("scratch.txt"), b"written since the save").unwrap();
        assert!(state.is_dirty(), "a workspace file the archive lacks is an unsaved change");

        // A file that was packed but whose CONTENT has since changed: the fingerprint has to carry
        // more than the set of names, or the commonest edit of all goes unnoticed.
        save_to(&state, &target);
        assert!(!state.is_dirty(), "saving again re-baselines the workspace");
        std::fs::write(state.mount().join("agent.md"), b"notes, and then some more notes").unwrap();
        assert!(state.is_dirty(), "an edit to a packed file is an unsaved change too");

        // …including one that leaves the LENGTH alone — an editor rewriting a line in place. That
        // is the only edit the mtime half catches on its own, and the half a `(name, len)`
        // fingerprint would silently drop.
        save_to(&state, &target);
        std::fs::write(state.mount().join("agent.md"), b"NOTES, AND THEN SOME MORE NOTES").unwrap();
        assert!(state.is_dirty(), "a same-length in-place edit is an unsaved change");

        // And a save that FAILED re-baselines nothing: it packed no file, so those edits still live
        // only in the mount — a per-run temp tree that a graceful exit deletes. Calling them packed
        // is the one direction that loses them (the arm's comment states the other half, the sample
        // taken before the pack, which is a race against the zip write and cannot be pinned).
        let nowhere = tmp.path().join("no-such-dir").join("patch.gfi");
        let req = json!({ "id": 1, "op": "save", "payload": { "path": nowhere.to_string_lossy() } });
        let reply = dispatch(&state, &req.to_string()).expect("a numeric id gets a reply");
        assert!(reply.contains("error"), "the save fails; got {reply}");
        assert!(state.is_dirty(), "a save that wrote nothing cannot call the workspace packed");
    }

    /// What the `load` arm's re-baseline buys, and the trap it steps around: a freshly opened patch
    /// has the dot off and the unload guard down, on a graph and a workspace that are byte-for-byte
    /// the file's.
    #[test]
    fn a_freshly_loaded_patch_is_clean_though_every_file_in_it_was_just_written() {
        let state = AppState::new();
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("patch.gfi");
        std::fs::write(state.mount().join("agent.md"), b"notes").unwrap();
        save_to(&state, &target);

        // Loaded into a SECOND manager, which is the real case — the goofi that opens a patch is
        // rarely the one that wrote it, and it has no baseline of its own to fall back on.
        let opened = AppState::new();
        let req = json!({ "id": 1, "op": "load", "payload": { "path": target.to_string_lossy() } });
        let reply = dispatch(&opened, &req.to_string()).expect("a numeric id gets a reply");
        assert!(reply.contains("result"), "the archive loads; got {reply}");
        assert_eq!(std::fs::read(opened.mount().join("agent.md")).unwrap(), b"notes");
        assert!(!opened.is_dirty(), "a patch is not unsaved the moment it finishes loading");
    }
}

#[cfg(test)]
mod node_scan_tests {
    use super::*;
    use goofi_core::{Data, Meta};
    use goofi_node::{
        Inputs, Isolation, Node, NodeCtx, NodeError, NodeManifest, NodeResult, OutputDecl, Outputs,
        Params,
    };
    use serde_json::json;

    static OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: goofi_core::SlotType::Array }];
    fn never() -> Box<dyn Node> {
        unreachable!("a scanned type is built by its registered factory, not manifest.factory")
    }

    /// A node that emits the number its file held WHEN THE SCAN RAN — the stand-in for a Python
    /// node's source, which discovery likewise captures at scan time. That capture is what makes
    /// "the running node is the NEW code" observable at all.
    struct Emit(f32);
    impl Node for Emit {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            let d = Data::array_f32(vec![1], self.0.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| NodeError(e.to_string()))?;
            out.set("out", d);
            Ok(())
        }
    }

    /// A stand-in for the CLI's real tier-routing scan, faithful in the ways these tests turn on —
    /// it reads each file's content at scan time, names the type after the file stem by the shared
    /// rule, and reports one [`ScannedType`] per file — and it needs no Python interpreter, which is
    /// the whole point of the seam being injectable.
    fn stub_scan(g: &mut Graph, dir: &std::path::Path) -> Vec<ScannedType> {
        let mut paths: Vec<_> =
            std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
        paths.sort();
        let mut out = Vec::new();
        for path in paths {
            if path.extension().and_then(|e| e.to_str()) != Some("py") {
                continue;
            }
            let name = goofi_node::discover::camel(&path.file_stem().unwrap().to_string_lossy());
            let value: f32 =
                std::fs::read_to_string(&path).unwrap_or_default().trim().parse().unwrap_or(0.0);
            let manifest: &'static NodeManifest = Box::leak(Box::new(NodeManifest {
                type_name: Box::leak(name.clone().into_boxed_str()),
                category: "python",
                doc: "a scanned node",
                inputs: &[],
                outputs: OUT,
                params: &[],
                isolation: Isolation::InProcess,
                factory: never,
            }));
            out.push(ScannedType {
                type_name: name,
                tier: Tier::InProcess,
                stamp: std::fs::metadata(&path).ok().map(|m| (m.len(), m.modified().unwrap())),
                registration: g.register_dyn_type(manifest, Box::new(move |_| Box::new(Emit(value)))),
            });
        }
        out
    }

    fn emitted(g: &Graph, uid: goofi_engine::Uid) -> f32 {
        match g.latest_frame(uid, "out").expect("the node emitted").value() {
            goofi_core::Value::Array(s) => f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()),
            _ => panic!("not an array"),
        }
    }

    fn write_node(dir: &std::path::Path, file: &str, body: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(file), body).unwrap();
    }

    fn scanning(state: &mut AppState) {
        state.scan_nodes = Arc::new(stub_scan);
    }

    /// The whole of a patch node's life, in the order a user lives it: write the file, rescan, and
    /// it is addable; edit it, rescan, and the node ALREADY ON THE CANVAS runs the new code; delete
    /// it, rescan, and it stops being addable while the instance that exists keeps running.
    #[test]
    fn a_node_file_in_the_workspace_is_live_after_a_rescan_and_follows_its_edits() {
        let mut state = AppState::new();
        scanning(&mut state);
        let nodes = state.mount().join("nodes");
        write_node(&nodes, "my_thing.py", "1.0");

        let mut g = state.graph.lock().unwrap();
        let diff = rescan(&state, &mut g, &state.mount()).0;
        assert_eq!(diff.added, ["MyThing"], "the file becomes a type");
        // Twice over an unchanged tree is a no-op: the baseline is what the LAST scan found, so
        // pressing refresh with nothing edited says nothing changed. Boot seeds that baseline
        // through this very function, which is why the first refresh of a session is quiet too.
        let again = rescan(&state, &mut g, &state.mount()).0;
        assert!(
            again.added.is_empty() && again.changed.is_empty() && again.removed.is_empty(),
            "a rescan of an unchanged tree changes nothing"
        );
        let live = g.add_node("MyThing", None).expect("a patch node is addable");
        g.tick();
        assert_eq!(emitted(&g, live), 1.0);

        // Edited: the type is re-registered and the LIVE instance is restarted onto it.
        write_node(&nodes, "my_thing.py", "2.0");
        let diff = rescan(&state, &mut g, &state.mount()).0;
        assert_eq!(diff.changed, ["MyThing"], "an edited file reports as changed");
        assert!(diff.added.is_empty() && diff.removed.is_empty());
        restart_changed(&mut g, &diff);
        g.tick();
        assert_eq!(emitted(&g, live), 2.0, "the running node is the new code");

        // Deleted: unaddable, but the instance is left alone (removal closes the door, it does not
        // reach into the graph).
        std::fs::remove_file(nodes.join("my_thing.py")).unwrap();
        let diff = rescan(&state, &mut g, &state.mount()).0;
        assert_eq!(diff.removed, ["MyThing"]);
        assert!(g.add_node("MyThing", None).is_err(), "a vanished type is no longer addable");
        g.tick();
        assert_eq!(emitted(&g, live), 2.0, "its instance still runs");
    }

    /// The two directories are one registry, and the patch is scanned SECOND so its own file wins a
    /// name the shipped tree also uses — that is what "patch node" means. Provenance is recorded in
    /// the same pass, because this is the only place that knows which tree a type came from.
    #[test]
    fn a_patch_local_node_wins_the_name_and_is_marked_as_the_patchs_own() {
        let mut state = AppState::new();
        scanning(&mut state);
        let shipped = tempfile::tempdir().unwrap();
        write_node(shipped.path(), "my_thing.py", "1.0");
        write_node(shipped.path(), "only_shipped.py", "7.0");
        state.system_nodes = Some(shipped.path().to_path_buf());
        write_node(&state.mount().join("nodes"), "my_thing.py", "9.0");

        let mut g = state.graph.lock().unwrap();
        rescan(&state, &mut g, &state.mount());
        let uid = g.add_node("MyThing", None).unwrap();
        g.tick();
        assert_eq!(emitted(&g, uid), 9.0, "the patch's own file wins the name");
        assert!(g.is_patch_type("MyThing"), "…and says where it came from");
        assert!(!g.is_patch_type("OnlyShipped"), "the shipped tree's own node is not the patch's");
    }

    /// …and `read_node_source` reports the same precedence, because an agent that reads a type's
    /// source is about to edit it: handing back the shipped file while the patch's own copy is the
    /// one running would send the edit to a file nothing executes.
    #[test]
    fn read_node_source_hands_back_the_file_that_is_actually_running() {
        let mut state = AppState::new();
        scanning(&mut state);
        let shipped = tempfile::tempdir().unwrap();
        write_node(shipped.path(), "my_thing.py", "1.0");
        state.system_nodes = Some(shipped.path().to_path_buf());
        write_node(&state.mount().join("nodes"), "my_thing.py", "9.0");
        rescan(&state, &mut state.graph.lock().unwrap(), &state.mount());

        let req = json!({ "id": 1, "op": "read_node_source", "payload": { "type": "MyThing" } });
        let reply: Value =
            serde_json::from_str(&dispatch(&state, &req.to_string()).expect("answered")).unwrap();
        let r = &reply["result"];
        assert_eq!(r["provenance"], json!("patch"), "{r}");
        assert_eq!(r["source"], json!("9.0"), "{r}");
        assert_eq!(r["path"], json!(state.mount().join("nodes/my_thing.py").to_string_lossy()), "{r}");
        state.release_mount();
    }

    /// A load swaps the workspace, so the registry must follow it — and the ORDER is load-bearing:
    /// `load_doc` rejects a type it does not know, which is precisely the set of types the patch
    /// ships. Pinned through the real op, because the ordering only exists inside that arm.
    #[test]
    fn loading_a_patch_registers_the_nodes_it_ships_before_resolving_them() {
        let mut state = AppState::new();
        scanning(&mut state);
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("patch.gfi");
        write_node(&state.mount().join("nodes"), "my_thing.py", "5.0");
        {
            let mut g = state.graph.lock().unwrap();
            rescan(&state, &mut g, &state.mount());
            g.add_node("MyThing", None).unwrap();
        }
        save_to(&state, &target);

        // A SECOND manager, which is the real case: it has never seen this type.
        let mut opened = AppState::new();
        scanning(&mut opened);
        let req = json!({ "id": 1, "op": "load", "payload": { "path": target.to_string_lossy() } });
        let reply = dispatch(&opened, &req.to_string()).expect("a numeric id gets a reply");
        assert!(reply.contains("result"), "the patch's own node type resolves; got {reply}");
        let mut g = opened.graph.lock().unwrap();
        assert_eq!(g.node_count(), 1);
        let uid = g.node_uids()[0];
        g.tick();
        assert_eq!(emitted(&g, uid), 5.0, "the instance runs the patch's code");
        assert!(g.is_patch_type("MyThing"));
        drop(g);

        // …and the NEXT patch drops it again: `new` swaps in an empty workspace, so the type the
        // previous patch brought must stop being addable rather than linger from a patch that is
        // no longer open.
        let req = json!({ "id": 2, "op": "new", "payload": {} });
        dispatch(&opened, &req.to_string()).expect("a numeric id gets a reply");
        let mut g = opened.graph.lock().unwrap();
        assert!(g.add_node("MyThing", None).is_err(), "the previous patch's type is gone");
    }

    fn save_to(state: &AppState, target: &std::path::Path) {
        let req = json!({ "id": 1, "op": "save", "payload": { "path": target.to_string_lossy() } });
        let reply = dispatch(state, &req.to_string()).expect("a numeric id gets a reply");
        assert!(reply.contains("result"), "the save is accepted; got {reply}");
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
