//! `goofi-pipe` — the binary: arg parsing, tier routing, and the one exit path.
//!
//! Every Python node is routed by ONE probe — in-process when free-threading-safe, else a
//! subprocess on the repo-local GIL venv. There is no flag for either choice.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use goofi_bridge::{serve_app, spawn_workers, AppState, ScannedType, Tier, HEADLESS_BUILD, SPA};
use goofi_engine::{Graph, Registration};

/// The shipped node directory, scanned whenever it exists — no flag turns it on, and none turns
/// it off. `--extra-nodes` names directories scanned *after* it.
const DEFAULT_NODES_DIR: &str = "nodes";

/// The directories to scan, in precedence order: the shipped tree first, then every
/// `--extra-nodes` in the order given. Scan order IS precedence (`goofi_bridge::rescan`), so a
/// user's own directory shadows a shipped type name while the rest of the shipped tree survives —
/// which is the whole difference between *extra* nodes and *instead-of* nodes.
fn node_dirs(extra: &[String], default_exists: bool) -> Vec<PathBuf> {
    (default_exists.then(|| PathBuf::from(DEFAULT_NODES_DIR)).into_iter())
        .chain(extra.iter().map(PathBuf::from))
        .collect()
}

/// The parsed command line. `help` is a mode rather than a setting, so the caller decides what
/// to do with it — which is also what keeps the parse itself pure and testable.
#[derive(Debug)]
struct Cli {
    port: u16,
    bind: String,
    /// Repeatable, and the only flag that is: these are directories scanned *in addition to* the
    /// shipped `nodes/` tree, which is what the name promises. Later entries win a shared type
    /// name — see `goofi_bridge::rescan`.
    extra_nodes: Vec<String>,
    list_nodes: bool,
    /// Serve the API without the app: `/control`, `/data`, `/term` and `/mcp` stay, the SPA's
    /// routes are never mounted. For a goofi driven by an agent or a script, which has no page to
    /// show — and for which an unmounted route is one nothing can reach by accident.
    ///
    /// Three doors, one mode: this flag, `GOOFI_HEADLESS` in the environment, and a binary BUILT
    /// with `GOOFI_HEADLESS` — which has no app to serve and so is headless for life. The env and
    /// the build stamp are folded in by [`main`], not read here, so the parse stays pure.
    headless: bool,
    help: bool,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            port: 8000,
            bind: String::from("127.0.0.1"),
            extra_nodes: Vec::new(),
            list_nodes: false,
            headless: false,
            help: false,
        }
    }
}

const USAGE: &str = "usage: goofi-pipe [--port N] [--bind HOST] \
     [--extra-nodes DIR] [--list-nodes] [--headless]";

/// The environment's spelling of `--headless`, for a caller that sets a variable rather than an
/// argument — a container, a service unit, a shell that exports it once. Only a truthy value opts
/// in, matching the build script, which reads the same name to decide whether to compile an app at
/// all. Nothing else in goofi is configured by environment, so this is a door onto an EXISTING
/// mode, never a setting of its own.
fn headless_env() -> bool {
    matches!(std::env::var("GOOFI_HEADLESS").as_deref(), Ok("1") | Ok("true"))
}

/// Parse the argument list (already skipping argv[0]). `Err` is the message to print before
/// exiting 2 — every malformed invocation reports, none is silently ignored: a value-taking flag
/// with its value missing used to fall through to the default, so `--extra-nodes` with a typo'd
/// directory silently scanned nothing while reporting success.
fn parse_args<I: Iterator<Item = String>>(mut args: I) -> Result<Cli, String> {
    let mut cli = Cli::default();
    while let Some(arg) = args.next() {
        // One shape for every value-taking flag, so a missing value is the same reported error
        // everywhere instead of five independent silent fallbacks.
        let need = |v: Option<String>| v.ok_or_else(|| format!("{arg} requires a value (try --help)"));
        match arg.as_str() {
            "--port" => {
                let v = need(args.next())?;
                cli.port = v.parse().map_err(|_| format!("invalid --port `{v}`"))?;
            }
            "--bind" => cli.bind = need(args.next())?,
            "--extra-nodes" => cli.extra_nodes.push(need(args.next())?),
            "--list-nodes" => cli.list_nodes = true,
            "--headless" => cli.headless = true,
            "-h" | "--help" => cli.help = true,
            other => return Err(format!("unknown argument `{other}` (try --help)")),
        }
    }
    Ok(cli)
}

#[tokio::main]
async fn main() {
    let mut cli = match parse_args(std::env::args().skip(1)) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    // The three doors meet HERE, once, and everything downstream sees one flag. A binary built
    // headless has no app in it, so it is not merely allowed to run that way — it is the only way
    // it can run, and saying so here is what keeps `--headless` from being something the user has
    // to remember to repeat.
    cli.headless |= headless_env() || HEADLESS_BUILD;
    if cli.help {
        println!(
            "{USAGE}\n\
             \n  \
             Scans `{DEFAULT_NODES_DIR}/` when it exists, plus every --extra-nodes directory. \
             Each node is routed in-process if free-threading-safe, else to a subprocess on \
             `{}`, which `cargo run -p goofi-init` provisions.\n  \
             GOOFI_HEADLESS=1 in the environment is --headless; setting it for the BUILD leaves \
             the app out of the binary entirely.",
            goofi_init::GIL_VENV
        );
        return;
    }
    // Settled HERE, and handed to `run` as its own argument: resolving needs the filesystem, which
    // `parse_args` must not touch, and passing it separately is what makes "no interpreter" a state
    // `run` cannot be called in rather than one it has to invent a value for.
    let python = match default_subproc_python() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    std::process::exit(run(cli, python, AppState::new(), shutdown_signal()).await);
}

/// The interpreter the subprocess tier runs on: the venv `goofi-init` made, and only that one —
/// no flag names another. goofi does NOT provision here: setup is one explicit command
/// (`cargo run -p goofi-init`), so an absent venv is reported once, by name, rather than as a
/// dozen Python nodes each failing their probe with their own version of the same news.
fn default_subproc_python() -> Result<String, String> {
    goofi_init::venv_python(&goofi_init::repo_root().join(goofi_init::GIL_VENV))
        .map(|p| p.display().to_string())
        .ok_or_else(|| format!("no {} — {}", goofi_init::GIL_VENV, goofi_init::RUN_ME))
}

/// The warning a `--bind` beyond this machine earns, or `None` for the loopback default.
///
/// Said out loud because the consequence is not the one the flag looks like it has. goofi spawns
/// agent harnesses on a PTY (`/term`) with the user's own environment, so a reachable goofi is a
/// reachable **shell** — RCE-class rather than "someone could read my patch". The Origin guard
/// stops a drive-by page in the user's own browser; it is not authentication, and there is none.
/// A name that is not an address (`goofi.local`) warns too: it resolves to whatever DNS says, and
/// the only address this function can prove is local is one it can parse.
fn exposure_warning(bind: &str) -> Option<String> {
    let local = bind == "localhost"
        || bind.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    (!local).then(|| {
        format!(
            "WARNING: --bind {bind} serves goofi beyond this machine, and goofi runs agent \
             harnesses on a shell with your environment. Anyone who can reach this port can run \
             commands as you: there is no authentication, only a guard against a web page \
             reaching it through your browser."
        )
    })
}

/// Everything the process does once it has a state, RETURNING its exit code — so the workspace
/// mount is reclaimed on the one path every outcome takes. `std::process::exit` unwinds nothing, so
/// a destructor could not stand in for this.
async fn run(
    cli: Cli,
    subproc_python: String,
    mut state: AppState,
    shutdown: impl Future<Output = ()>,
) -> i32 {
    // Before ANY use of the embedded interpreter — the evaluator below, and every in-process
    // Python node after it.
    point_embedded_python_at_its_venv();

    // `subproc_python` arrives as its own argument (see `main`) — it is resolved from the
    // filesystem, never parsed, so it was never a field on `Cli` to begin with.
    let Cli { port, bind, extra_nodes, list_nodes, headless, help: _ } = cli;

    if !list_nodes {
        register_evaluator(&state);
    }
    // Install the discovery seam BEFORE anything scans, so the boot scan below and every later
    // `rescan_nodes` (the palette's refresh, an agent that just wrote a node file) route each file
    // through one function. The interpreter choice is boot-time config, so the closure carries it
    // and the seam itself takes only a directory.
    let python = subproc_python.clone();
    state.scan_nodes = Arc::new(move |g, dir| register_routed(g, dir, &python));
    state.system_nodes = node_dirs(&extra_nodes, std::path::Path::new(DEFAULT_NODES_DIR).is_dir());
    // `--list-nodes` runs the SAME registration the server does and reports its result, so the
    // listing is what actually registered — not a hand-kept mirror of the routing rule.
    let discovered = if state.system_nodes.is_empty() { Vec::new() } else { boot_scan(&state) };

    let code = if list_nodes {
        let mut names = goofi_bridge::catalog_type_names();
        names.extend(discovered);
        println!("{} node types: {}", names.len(), names.join(", "));
        0
    // No app compiled in, and nobody asked for that: refused rather than attempted. A server that
    // came up and answered every route with nothing is the failure being prevented, and it looked
    // exactly like a working start. A build that MEANT to have no app set `HEADLESS_BUILD`, which
    // is already folded into `headless` — so reaching here means the bundle is missing, not waived.
    //
    // An arm of this chain rather than an early `return`, because `AppState::new` has already taken
    // a workspace mount and only the tail of this function gives it back. `--list-nodes` reports
    // the palette and serves nothing, so it passes above this.
    } else if !headless && SPA.is_empty() {
        eprintln!("refusing to start: no app is compiled into this binary.");
        eprintln!(
            "  The app is compiled in, so building it is not enough — build it, then rebuild \
             goofi-pipe:"
        );
        eprintln!("    npm install && npm run build   (in frontend/)");
        eprintln!("    cargo build");
        eprintln!("  Or serve the API alone: --headless, or GOOFI_HEADLESS=1.");
        1
    } else {
        spawn_workers(&state); // the status-drain worker: 1 ms drain, 2 Hz broadcast
        match tokio::net::TcpListener::bind((bind.as_str(), port)).await {
            Err(e) => {
                eprintln!("failed to bind {bind}:{port}: {e}");
                1
            }
            Ok(listener) => {
                let addr = listener.local_addr().unwrap();
                // A spawned harness is a CHILD of this process, so it reaches the MCP surface on
                // loopback whatever `--bind` says — only the port, which `--port 0` makes knowable
                // nowhere else, has to be handed over.
                state.set_mcp_port(addr.port());
                let url = format!("http://{addr}");
                println!("goofi-pipe → {url}");
                // Printed beside the app URL because it is what a user pastes into an MCP client's
                // config, and what H's harness launcher passes to a spawned agent in its
                // environment. There is one server per goofi instance, so this URL is the address
                // of the whole agent surface — no client ever spawns one of its own.
                println!("  MCP endpoint → http://{addr}/mcp");
                // `--headless` withholds the page itself, not merely a browser: an agent or a
                // script driving `/control` and `/mcp` has no use for the SPA, and a route that is
                // not mounted is one nothing can reach by accident.
                let spa = if headless { &[][..] } else { SPA };
                if headless {
                    println!("  headless: the API only, no app served");
                } else {
                    println!("  open {url} to use it");
                }
                // Last, and on stderr, so it is the line still on screen and survives a `> log`.
                if let Some(warning) = exposure_warning(&bind) {
                    eprintln!("{warning}");
                }
                // The stop is HERE and not inside `serve_app`, whose other callers are tests that
                // want it to serve forever. Dropping the server and draining it behave alike:
                // axum's per-connection task ends at the WS upgrade, so a socket held open for the
                // life of a tab delays neither.
                tokio::select! {
                    served = serve_app(listener, state.clone(), spa) => match served {
                        Ok(()) => 0,
                        Err(e) => {
                            eprintln!("server error: {e}");
                            1
                        }
                    },
                    _ = shutdown => 0,
                }
            }
        }
    };
    // The graceful exit, in the order it has to happen: every node stops and its own thread drops
    // the ports holding its shared memory, and only THEN does the workspace go. The only path that
    // runs either — without it every ctrl-C is a crash exit.
    state.graph.lock().unwrap().shutdown();
    state.release_mount();
    code
}

/// Resolve on the first request to stop: ctrl-C, or whatever else this platform's service manager
/// knocks with. A door that cannot be installed must **never** resolve — an immediately-ready arm
/// here would shut the server down at startup rather than merely leaving that one door closed.
async fn shutdown_signal() {
    let interrupt = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        _ = interrupt => {}
        _ = managed_stop() => {}
    }
}

/// The stop a SERVICE MANAGER sends, which ctrl-C does not cover — `SIGTERM` where signals exist.
/// The whole reason it is worth a door of its own is [`AppState::release_mount`]: a run killed
/// through this one still gets to delete its workspace mount.
#[cfg(unix)]
async fn managed_stop() {
    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(mut sig) => {
            sig.recv().await;
        }
        Err(e) => {
            eprintln!("SIGTERM handler unavailable: {e}");
            std::future::pending::<()>().await;
        }
    }
}

/// Windows has no SIGTERM. What stands in for it is the console window closing and the machine
/// going down — two events rather than one, so both are doors and the first to open wins.
#[cfg(windows)]
async fn managed_stop() {
    let doors = (tokio::signal::windows::ctrl_close(), tokio::signal::windows::ctrl_shutdown());
    let (mut close, mut shutdown) = match doors {
        (Ok(close), Ok(shutdown)) => (close, shutdown),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("Windows shutdown handlers unavailable: {e}");
            return std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        _ = close.recv() => {}
        _ = shutdown.recv() => {}
    }
}

/// Install the pyo3 param-expression evaluator into the graph. Independent of any node
/// directory: expressions are a core feature, needing only the embedded free-threaded
/// interpreter the `python` feature links.
#[cfg(feature = "python")]
fn register_evaluator(state: &AppState) {
    match goofi_python::inproc::PyExprEvaluator::new() {
        Ok(ev) => {
            state.graph.lock().unwrap().set_evaluator(std::sync::Arc::new(ev));
            println!("  param-expression evaluator ready (free-threaded Python)");
        }
        Err(e) => eprintln!("param-expression evaluator unavailable: {e}"),
    }
}

/// Hand the EMBEDDED interpreter the venv pyo3 was linked against.
///
/// pyo3 links `libpython` out of that venv's BASE install, so the venv — where `goofi` and `numpy`
/// actually are — sits on no search path at all. The cargo config covers this for `cargo run` and
/// nothing else, so a bare `./goofi-pipe` came up with a dead evaluator and in-process nodes that
/// cannot import their own package.
///
/// `--list-nodes` does not reveal it: registration runs through the probe, which is a SUBPROCESS
/// and finds its own site-packages. Only execution breaks.
#[cfg(feature = "python")]
fn point_embedded_python_at_its_venv() {
    // An existing value is the documented override, and under `cargo run` it is already right.
    if std::env::var_os("PYTHONPATH").is_some() {
        return;
    }
    let Some(python) = goofi_python::inproc::interpreter_path() else { return };
    // `<venv>/bin/python` → `<venv>`.
    let Some(venv) = Path::new(&python).parent().and_then(Path::parent) else { return };
    if let Some(dir) = goofi_init::site_packages(venv) {
        std::env::set_var("PYTHONPATH", dir);
    }
}

#[cfg(not(feature = "python"))]
fn point_embedded_python_at_its_venv() {}

#[cfg(not(feature = "python"))]
fn register_evaluator(_state: &AppState) {
    // Without the `python` feature there is no embedded interpreter; expression bindings are
    // stored but not evaluated — the node reports "no expression evaluator available". Announce
    // it at startup so that error isn't a mystery (the pyo3 evaluator needs a free-threaded 3.14t
    // interpreter, hence the opt-in feature).
    println!("  param expressions DISABLED — rebuild with `--features python` to enable the evaluator");
}

/// One boot registration, reported. The boot registry starts EMPTY, so a `Replaced` here can only
/// mean a second node file claimed a name an earlier one already took — the collision report that
/// used to live in the engine, now stated where it is true: a *rescan* replaces types on purpose
/// and says nothing. Returns whether the type is now registered.
fn note_registration(name: &str, r: Registration) -> bool {
    if r == Registration::Replaced {
        eprintln!("warning: two node files claim the type name `{name}`; the later one wins");
    }
    r != Registration::Refused
}

/// The files in a node directory, in a deterministic order.
#[cfg(feature = "python")]
fn sorted_dir(dir: &Path) -> Vec<std::path::PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(rd) => {
            let mut v: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
            v.sort();
            v
        }
        Err(e) => {
            eprintln!("failed to read {}: {e}", dir.display());
            Vec::new()
        }
    }
}

/// A node file's size + mtime — the stamp a rescan diffs to notice that its code changed.
#[cfg(feature = "python")]
fn stamp(path: &Path) -> Option<goofi_bridge::Stamp> {
    let m = std::fs::metadata(path).ok()?;
    Some((m.len(), m.modified().ok()?))
}

/// What one file's probes decided, before anything is registered. Separated from the registration
/// because the probes RUN IN PARALLEL and registration needs the graph.
#[cfg(feature = "python")]
#[derive(Clone)]
enum Probed {
    InProcess(goofi_python::Discovered),
    Subprocess(goofi_python::Discovered),
    Unavailable { type_name: String, reason: String },
    Skip,
}

/// What the last probe of each file decided, and the stamp it decided it at.
///
/// A probe is a whole interpreter importing a whole module, and a rescan re-probes the DIRECTORY —
/// so without this every refresh pays for every file again, however little changed. The shipped
/// entropy nodes import antropy, which imports numba: four of them made the palette's own refresh
/// button a twenty-second wait and a patch load a forty-second one. A file whose size and mtime
/// have not moved is the same file, which is the same thing `rescan` already trusts to decide that
/// a type is unchanged.
#[cfg(feature = "python")]
static PROBED: std::sync::Mutex<
    Option<std::collections::HashMap<std::path::PathBuf, (goofi_bridge::Stamp, Probed)>>,
> = std::sync::Mutex::new(None);

/// Probe one file on both tiers, in the order that decides the routing, unless a probe of this
/// exact file at this exact stamp already answered. Spawns interpreters and reads files; touches
/// no graph.
#[cfg(feature = "python")]
fn probe_one(path: &Path, ft: Option<&str>, subproc_python: &str) -> Probed {
    let stamp = stamp(path);
    if let Some(s) = stamp {
        let cache = PROBED.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((seen, probed)) = cache.as_ref().and_then(|c| c.get(path)) {
            if *seen == s {
                return probed.clone();
            }
        }
    }
    let probed = probe_uncached(path, ft, subproc_python);
    if let Some(s) = stamp {
        let mut cache = PROBED.lock().unwrap_or_else(|e| e.into_inner());
        cache.get_or_insert_with(Default::default).insert(path.to_path_buf(), (s, probed.clone()));
    }
    probed
}

#[cfg(feature = "python")]
fn probe_uncached(path: &Path, ft: Option<&str>, subproc_python: &str) -> Probed {
    // ONE free-threaded probe answers both questions: it imports the module and constructs the
    // class (so a dep missing on the FT interpreter shows up as a failed probe), then reports
    // whether the GIL is still disabled — `gil_safe` IS the routing gate.
    let ft_probe =
        ft.map_or(goofi_python::Discovery::Skip, |ftp| goofi_python::inproc::probe(path, ftp));
    if let goofi_python::Discovery::Found(d) = ft_probe {
        if d.gil_safe {
            return Probed::InProcess(d);
        }
        // Loading it re-enabled the GIL — quarantine it to the subprocess tier below. (So does a
        // failed FT probe: typically a dep present on the subproc python but absent on the FT
        // interpreter, which must fall through rather than drop the node.)
    }
    // One probe, both outcomes: the subprocess tier is the last chance, so its result decides
    // between "registered" and "listed as unavailable".
    match goofi_python::subproc::probe(path, subproc_python) {
        goofi_python::Discovery::Found(d) => Probed::Subprocess(d),
        goofi_python::Discovery::Unavailable { type_name, reason } => {
            Probed::Unavailable { type_name, reason }
        }
        goofi_python::Discovery::Skip => Probed::Skip,
    }
}

/// The GIL-gate router: probe each file and register it in-process when its imports keep the GIL
/// disabled, else quarantine it to a subprocess. THE discovery seam — boot and every later rescan
/// route the same file the same way by construction, not by two implementations agreeing.
///
/// **The probes run at once, one interpreter each.** A probe is a whole interpreter importing a
/// whole module, and a node file is entitled to a heavy import — the entropy nodes pull antropy,
/// which pulls numba, which takes seconds. Serially that is the slowest thing a start does. They
/// cannot SHARE an interpreter, though: re-enabling the GIL is one-way, so one file's import would
/// decide the next file's tier.
///
/// It prints NOTHING: a rescan runs it whenever an agent writes a node file. `boot_scan` talks.
#[cfg(feature = "python")]
fn register_routed(g: &mut Graph, dir: &Path, subproc_python: &str) -> Vec<ScannedType> {
    let ft = goofi_python::inproc::interpreter_path(); // the embedded FT interpreter, for probing
    let paths = sorted_dir(dir);
    // One interpreter per file at once, but no more of them than the machine has cores: a
    // workspace of a hundred node files must not answer a rescan with a hundred pythons.
    let width = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 8);
    let mut probes = Vec::with_capacity(paths.len());
    for chunk in paths.chunks(width) {
        std::thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|p| s.spawn(|| probe_one(p, ft.as_deref(), subproc_python)))
                .collect();
            for h in handles {
                probes.push(h.join().unwrap_or(Probed::Skip));
            }
        });
    }

    let mut found = Vec::new();
    for (path, probed) in paths.iter().zip(probes) {
        let (type_name, tier, registration) = match probed {
            Probed::InProcess(d) => {
                let t = goofi_python::inproc::node_type_from(d);
                (t.manifest.type_name.to_string(), Tier::InProcess, g.register_dyn_type(t.manifest, t.factory))
            }
            Probed::Subprocess(d) => {
                let t = goofi_python::subproc::node_type_from(subproc_python, d);
                (t.manifest.type_name.to_string(), Tier::Subprocess, g.register_dyn_type(t.manifest, t.factory))
            }
            // Neither tier could load it. Register it as unavailable WITH the reason so the
            // palette explains itself — a node file that silently does not appear reads as
            // "goofi ignored my file" rather than "install this dependency".
            Probed::Unavailable { type_name, reason } => {
                let registration = if g.register_unavailable(type_name.clone(), reason.clone()) {
                    Registration::Added
                } else {
                    Registration::Refused // a built-in owns the name
                };
                (type_name, Tier::Unavailable(reason), registration)
            }
            Probed::Skip => continue,
        };
        found.push(ScannedType { type_name, tier, stamp: stamp(path), registration });
    }
    found
}

#[cfg(not(feature = "python"))]
fn register_routed(_g: &mut Graph, _dir: &Path, _subproc_python: &str) -> Vec<ScannedType> {
    Vec::new()
}

/// Said once, by the boot summary: without the `python` feature there is no embedded interpreter,
/// hence no probe, hence no tier routing — the seam above finds nothing at all.
#[cfg(feature = "python")]
const NO_PYTHON_NOTE: &str = "";
#[cfg(not(feature = "python"))]
const NO_PYTHON_NOTE: &str = " (built without the `python` feature — node discovery is off)";

/// The boot scan, reported. It runs the bridge's own `rescan` — not merely the same seam — so the
/// baseline the first refresh diffs against IS this scan. The collision warning is only TRUE here:
/// the boot registry starts empty, so a `Replaced` can only be two files claiming one name.
fn boot_scan(state: &AppState) -> Vec<String> {
    let (found, dirs) = {
        let mut g = state.graph.lock().unwrap();
        // The mount is empty at boot, so these are exactly the shipped directories — one call, and
        // a patch loaded later re-derives through the same function.
        let patch = state.mount();
        (goofi_bridge::rescan(state, &mut g, &patch).1, state.system_nodes.clone())
    };
    let (mut n_in, mut n_sub, mut n_bad) = (0u32, 0u32, 0u32);
    let mut names = Vec::new();
    for t in found {
        if !note_registration(&t.type_name, t.registration) {
            continue;
        }
        let tier = match &t.tier {
            Tier::InProcess => {
                n_in += 1;
                "in-proc"
            }
            Tier::Subprocess => {
                n_sub += 1;
                "subproc"
            }
            Tier::Unavailable(reason) => {
                eprintln!("  node `{}` unavailable: {reason}", t.type_name);
                n_bad += 1;
                "unavailable"
            }
        };
        names.push(format!("{} ({tier})", t.type_name));
    }
    let bad = if n_bad > 0 { format!(", {n_bad} unavailable") } else { String::new() };
    // Every directory, in scan order — with several, "from nodes" would name one of them and read
    // as if the others had not been looked at.
    let from = dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ");
    println!(
        "  auto-routed {n_in} in-process + {n_sub} subprocess node type(s) from {from}{bad}{NO_PYTHON_NOTE}"
    );
    names
}

// The suite lives in `goofi-tests`. This block stays because goofi-cli is a BINARY: it has no
// lib target, so nothing outside can name its argument parser at all.
#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults_with_no_arguments() {
        let cli = parse(&[]).expect("no arguments is a valid invocation");
        assert_eq!(cli.port, 8000);
        assert_eq!(cli.bind, "127.0.0.1");
        assert!(cli.extra_nodes.is_empty());
        assert!(!cli.list_nodes && !cli.help);
    }

    #[test]
    fn reads_every_value_taking_flag() {
        let cli = parse(&[
            "--port", "9001", "--bind", "0.0.0.0", "--extra-nodes", "b", "--list-nodes",
        ])
        .expect("a well-formed invocation");
        assert_eq!(cli.port, 9001);
        assert_eq!(cli.bind, "0.0.0.0");
        assert_eq!(cli.extra_nodes, ["b"]);
        assert!(cli.list_nodes);
    }

    /// `--extra-nodes` ACCUMULATES where every other value-taking flag replaces — naming two
    /// directories scans both. A last-wins flag would silently drop the first, which is the
    /// opposite of what someone extending the palette is asking for.
    #[test]
    fn extra_nodes_accumulates_where_the_other_flags_replace() {
        let cli = parse(&["--extra-nodes", "theirs", "--bind", "a", "--extra-nodes", "mine",
                          "--bind", "b"])
            .expect("a repeated flag is well-formed");
        assert_eq!(cli.extra_nodes, ["theirs", "mine"], "--extra-nodes adds");
        assert_eq!(cli.bind, "b", "…while --bind still replaces");
    }

    /// `--extra-nodes` ADDS to the shipped tree; it never replaces it. The flag's name is the
    /// contract — a user naming their own directory still gets `nodes/`. Theirs comes LAST so it
    /// can shadow a shipped type name, the same "more specific wins" order the patch tree rides.
    #[test]
    fn extra_nodes_are_scanned_after_the_shipped_tree_not_instead_of_it() {
        let mine = || vec![String::from("mine")];
        assert_eq!(
            node_dirs(&mine(), true),
            [PathBuf::from("nodes"), PathBuf::from("mine")],
            "the shipped tree first, the user's own after it"
        );
        assert_eq!(node_dirs(&[], true), [PathBuf::from("nodes")], "…and alone when none is named");
        assert_eq!(node_dirs(&mine(), false), [PathBuf::from("mine")], "no shipped tree to lead");
    }

    /// A missing value is the same class of user error as an unknown flag, and used to be the
    /// only one handled silently: `--bind` alone served on the default host, and a node-directory
    /// flag alone left its value unset, so the run scanned somewhere other than the place asked
    /// for while reporting success.
    #[test]
    fn a_value_taking_flag_without_its_value_is_an_error() {
        for flag in ["--port", "--bind", "--extra-nodes"] {
            let err = parse(&[flag]).expect_err(&format!("`{flag}` alone must not be ignored"));
            assert!(err.contains(flag), "the message names the flag: {err}");
        }
    }

    /// A reachable goofi is a reachable SHELL: `/term` runs an agent harness on a PTY with the
    /// user's own environment, so the exposure a non-loopback `--bind` opens is RCE-class and not
    /// the "someone could see my patch" it looks like. The Origin guard stops a drive-by page in
    /// the user's own browser and nothing else — it is not authentication, and there is none — so
    /// the honest answer to `--bind 0.0.0.0` is to say out loud what was just agreed to.
    #[test]
    fn a_bind_beyond_this_machine_says_what_it_exposes() {
        for safe in ["127.0.0.1", "localhost", "::1", "127.0.0.53"] {
            assert!(exposure_warning(safe).is_none(), "`{safe}` is this machine");
        }
        for open in ["0.0.0.0", "::", "192.168.7.5", "goofi.local"] {
            let warn = exposure_warning(open).unwrap_or_else(|| panic!("`{open}` warns"));
            assert!(warn.contains(open), "the warning names the address: {warn}");
            // The consequence, not the fact. "Listening on 0.0.0.0" is what a user already knows.
            assert!(warn.contains("shell"), "the warning names the exposure: {warn}");
            assert!(warn.contains("no authentication"), "…and that nothing else guards it: {warn}");
        }
    }

    #[test]
    fn rejects_an_unparseable_port_and_an_unknown_flag() {
        assert!(parse(&["--port", "nope"]).unwrap_err().contains("--port"));
        // `--python-nodes` never existed — the warning in build.rs used to name it.
        assert!(parse(&["--python-nodes", "x"]).unwrap_err().contains("unknown argument"));
    }

    /// A RETIRED flag is REJECTED, never ignored. Someone's launcher still says
    /// `--subproc-python /usr/bin/python3`; exiting 2 tells them it stopped meaning anything,
    /// where swallowing it would let them keep believing the override applied — and the whole
    /// point of retiring it is that there is now exactly one interpreter.
    #[test]
    fn a_retired_node_flag_is_rejected_rather_than_ignored() {
        for retired in [
            ["--subproc-python", "/usr/bin/python3"],
            ["--subproc-nodes", "dir"],
            ["--auto-nodes", "dir"],
        ] {
            let err = parse(&retired).expect_err(&format!("`{}` is retired", retired[0]));
            assert!(err.contains("unknown argument"), "and says so plainly: {err}");
            assert!(err.contains(retired[0]), "…naming the flag the user typed: {err}");
        }
    }

    /// A node that records its own destruction. `Drop` runs on the NODE's thread, at the moment
    /// its runtime — and with it every iceoryx2 port it owns — goes; that is the release a
    /// graceful exit exists to get, and nothing on the manager side can fake it.
    struct Tracked(Arc<std::sync::atomic::AtomicBool>);
    impl goofi_node::Node for Tracked {
        fn process(
            &mut self,
            _i: &goofi_node::Inputs<'_>,
            _o: &mut goofi_node::Outputs<'_>,
            _c: &mut goofi_node::NodeCtx,
            _p: &goofi_node::Params<'_>,
        ) -> goofi_node::NodeResult {
            Ok(())
        }
    }
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Release);
        }
    }
    static TRACKED: goofi_node::NodeManifest = goofi_node::NodeManifest {
        type_name: "_TestTracked",
        category: "test",
        doc: "records its own teardown",
        inputs: &[],
        outputs: &[],
        params: &[],
        isolation: goofi_node::Isolation::InProcess,
        producer: true,
        factory: || unreachable!("built by the registered factory"),
    };

    /// Ctrl-C and SIGTERM are the only exits a user takes, and each has to leave the machine as it
    /// found it. Counted through the node's own `Drop` rather than the graph's bookkeeping: the
    /// segments go when the node's runtime drops on its own thread, and `shutdown` waiting for
    /// exactly that is why it is not `clear()`.
    #[tokio::test]
    async fn a_signal_stops_every_node_before_the_run_returns() {
        let state = AppState::new();
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let graph = state.graph.clone();
        {
            let mut g = graph.lock().unwrap();
            let flag = released.clone();
            g.register_dyn_type(&TRACKED, Box::new(move |_| Box::new(Tracked(flag.clone()))));
            g.add_node("_TestTracked", None).expect("a test node");
        }
        // An already-resolved shutdown takes the same path ctrl-C does; port 0 binds ephemerally.
        let cli = Cli { port: 0, ..Cli::default() };
        assert_eq!(run(cli, "python3".into(), state, std::future::ready(())).await, 0);
        assert!(
            released.load(std::sync::atomic::Ordering::Acquire),
            "the node's runtime was dropped — its shared memory went with it — before the exit"
        );
        assert_eq!(graph.lock().unwrap().node_count(), 0, "…and the graph is holding nothing");
    }

    /// The workspace mount's lifetime is the run's: present while the server is up, gone once it
    /// stops. Both ends live in `run` because main's exits unwind nothing.
    #[tokio::test]
    async fn the_mount_lives_exactly_as_long_as_the_run() {
        let state = AppState::new();
        let mount = state.mount();
        assert!(mount.is_dir(), "the mount exists after boot: {}", mount.display());
        // Port 0 binds ephemerally; an already-resolved shutdown takes the same path ctrl-C does.
        let cli = Cli { port: 0, ..Cli::default() };
        // The interpreter is named explicitly: this test spawns no Python node, and `run` no longer
        // has a default to fall back on — which is the point, since the old one was `python3`.
        assert_eq!(run(cli, "python3".into(), state, std::future::ready(())).await, 0);
        // The NONCE directory is what goes, not just `workspace` — else every run leaves an empty
        // husk behind. Asserting on the parent covers the leaf too.
        let husk = mount.parent().expect("the mount is nested under a nonce dir");
        assert!(!husk.exists(), "the nonce directory goes too, not just workspace: {}", husk.display());

        // `--list-nodes` returns before the server ever binds; the same tail must still reclaim.
        let listed = AppState::new();
        let m2 = listed.mount();
        let cli = Cli { list_nodes: true, ..Cli::default() };
        assert_eq!(run(cli, "python3".into(), listed, std::future::pending()).await, 0);
        assert!(!m2.exists(), "--list-nodes reclaims too: {}", m2.display());
    }

    #[test]
    fn help_is_a_mode_the_caller_handles() {
        assert!(parse(&["--help"]).expect("help parses").help);
        assert!(parse(&["-h"]).expect("help parses").help);
    }
}
