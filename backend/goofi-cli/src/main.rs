//! `goofi-pipe` — launches the Rust engine + bridge, serving the SPA and the two
//! WebSocket planes. Flags: `--port N` (default 8000), `--bind HOST` (default
//! 127.0.0.1), `--subproc-nodes DIR` (discover isolated-GIL subprocess Python nodes,
//! run on `--subproc-python`), `--auto-nodes DIR` (gil-gate routed). With no
//! `--*-nodes` flag it auto-discovers the default `nodes/` directory;
//! `--subproc-python` defaults to the repo-local `.gfivenv`.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use goofi_bridge::{resolve_frontend_dir, serve_app, spawn_workers, AppState, ScannedType, Tier};
use goofi_engine::{Graph, Registration};

/// The default node directory, auto-discovered (gil-gate routed) when no `--*-nodes` flag is given.
const DEFAULT_NODES_DIR: &str = "nodes";


/// The parsed command line. `help` is a mode rather than a setting, so the caller decides what
/// to do with it — which is also what keeps the parse itself pure and testable.
#[derive(Debug)]
struct Cli {
    port: u16,
    bind: String,
    subproc_nodes: Option<String>,
    /// Repeatable, and the only flag that is: a packaged build bakes its builtin node directory
    /// into the launch command, so a user's own `--auto-nodes` has to be added to that rather than
    /// replace it. Later entries win a shared type name — see `goofi_bridge::rescan`.
    auto_nodes: Vec<String>,
    subproc_python: Option<String>,
    list_nodes: bool,
    help: bool,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            port: 8000,
            bind: String::from("127.0.0.1"),
            subproc_nodes: None,
            auto_nodes: Vec::new(),
            subproc_python: None,
            list_nodes: false,
            help: false,
        }
    }
}

const USAGE: &str = "usage: goofi-pipe [--port N] [--bind HOST] \
     [--subproc-nodes DIR] [--auto-nodes DIR] [--list-nodes] [--subproc-python BIN]";

/// Parse the argument list (already skipping argv[0]). `Err` is the message to print before
/// exiting 2 — every malformed invocation reports, none is silently ignored: a value-taking flag
/// with its value missing used to fall through to the default, so `--subproc-nodes` with a typo'd
/// directory silently switched the whole run to gil-gated auto-routing of `nodes/` instead.
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
            "--subproc-nodes" => cli.subproc_nodes = Some(need(args.next())?),
            "--auto-nodes" => cli.auto_nodes.push(need(args.next())?),
            "--subproc-python" => cli.subproc_python = Some(need(args.next())?),
            "--list-nodes" => cli.list_nodes = true,
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
    if cli.help {
        println!(
            "{USAGE}\n\
             \n  \
             With no --*-nodes flag, auto-discovers `{DEFAULT_NODES_DIR}/` (each node routed \
             in-process if free-threading-safe, else to a subprocess).\n  \
             --subproc-python defaults to `{}`, which `cargo run -p goofi-init` provisions.",
            goofi_init::GIL_VENV
        );
        return;
    }
    // Settled HERE, and handed to `run` as its own argument: resolving needs the filesystem, which
    // `parse_args` must not touch, and passing it separately is what makes "no interpreter" a state
    // `run` cannot be called in rather than one it has to invent a value for.
    let python = match cli.subproc_python.take().map(Ok).unwrap_or_else(default_subproc_python) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    std::process::exit(run(cli, python, AppState::new(), shutdown_signal()).await);
}

/// The interpreter the subprocess tier runs on — the venv `goofi-init` made, unless
/// `--subproc-python` names another. goofi does NOT provision here: setup is one explicit command
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

/// Everything the process does once it has a state, as a function that *returns* its exit code:
/// the workspace mount is reclaimed here, on the one path every outcome takes. The alternative is
/// three `std::process::exit` calls that each have to remember — and `exit` unwinds nothing, so a
/// destructor would not save them either. `shutdown` is only awaited once the server is up, so a
/// signal that lands during boot still takes the default disposition and leaves the mount behind:
/// one empty temp directory in a rare race, the same as any other crash.
async fn run(
    cli: Cli,
    subproc_python: String,
    mut state: AppState,
    shutdown: impl Future<Output = ()>,
) -> i32 {
    // Before ANY use of the embedded interpreter — the evaluator below, and every in-process
    // Python node after it.
    point_embedded_python_at_its_venv();

    // `subproc_python` arrives resolved (see `main`), so the parsed field has already been taken.
    let Cli { port, bind, subproc_nodes, mut auto_nodes, subproc_python: _, list_nodes, help: _ } =
        cli;

    // When no explicit node source was given, auto-route the default `nodes/` directory.
    if subproc_nodes.is_none() && auto_nodes.is_empty()
        && std::path::Path::new(DEFAULT_NODES_DIR).is_dir()
    {
        auto_nodes.push(DEFAULT_NODES_DIR.to_string());
    }

    if !list_nodes {
        register_evaluator(&state);
    }
    // Install the discovery seam BEFORE anything scans, so the boot scan below and every later
    // `rescan_nodes` (the palette's refresh, an agent that just wrote a node file) route each file
    // through one function. The interpreter choice is boot-time config, so the closure carries it
    // and the seam itself takes only a directory.
    let python = subproc_python.clone();
    state.scan_nodes = Arc::new(move |g, dir| register_auto(g, dir, &python));
    state.system_nodes = auto_nodes.iter().map(PathBuf::from).collect();
    // `--list-nodes` runs the SAME registration the server does and reports its result, so the
    // listing is what actually registered — not a hand-kept mirror of the routing rule.
    let mut discovered = register_subproc(&state, subproc_nodes.as_deref(), &subproc_python);
    if !state.system_nodes.is_empty() {
        discovered.extend(boot_scan(&state));
    }

    let code = if list_nodes {
        let mut names = goofi_bridge::catalog_type_names();
        names.extend(discovered);
        println!("{} node types: {}", names.len(), names.join(", "));
        0
    } else {
        spawn_workers(&state); // adaptive tick loop + 2 Hz node-stats (header ufreq + errors)
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
                let dir = resolve_frontend_dir();
                println!("goofi-pipe (rust backend) → http://{addr}");
                // Printed beside the app URL because it is what a user pastes into an MCP client's
                // config, and what H's harness launcher passes to a spawned agent in its
                // environment. There is one server per goofi instance, so this URL is the address
                // of the whole agent surface — no client ever spawns one of its own.
                println!("  MCP endpoint → http://{addr}/mcp");
                match &dir {
                    Some(d) => println!("  serving SPA from {}", d.display()),
                    None => println!("  API only — no SPA build found (set GOOFI_FRONTEND_BUILD or build frontend/)"),
                }
                // Last, and on stderr, so it is the line still on screen and survives a `> log`.
                if let Some(warning) = exposure_warning(&bind) {
                    eprintln!("{warning}");
                }
                // The stop lives here rather than inside `serve_app`: the mount reclaim below is
                // the CLI's alone, and `serve_app`'s eight other callers are tests that want it to
                // serve forever. Dropping the server and draining it behave alike here anyway —
                // axum's per-connection task ends at the WS upgrade, so a `/control` socket held
                // open for the life of a tab delays neither — and with a handler installed, a
                // second ctrl-C no longer reaches the default disposition that would have killed us.
                tokio::select! {
                    served = serve_app(listener, state.clone(), dir) => match served {
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
    match goofi_py::PyExprEvaluator::new() {
        Ok(ev) => {
            state.graph.lock().unwrap().set_evaluator(std::sync::Arc::new(ev));
            println!("  param-expression evaluator ready (free-threaded Python)");
        }
        Err(e) => eprintln!("param-expression evaluator unavailable: {e}"),
    }
}

/// Hand the EMBEDDED interpreter the venv pyo3 was linked against.
///
/// pyo3 links `libpython` out of that venv's BASE install, so `sys.prefix` is the base install and
/// the venv — which is where `goofi` and `numpy` actually are — sits on no search path at all.
/// `.cargo/config.toml` covers this with a `PYTHONPATH` in its `[env]` block, and cargo applies
/// that to `cargo run` and to nothing else. So the binary launched any OTHER way — a packaged
/// build, a bare `./goofi-pipe` — came up with a dead param-expression evaluator
/// (`No module named 'numpy'`) and in-process Python nodes that cannot import their own package.
///
/// The `--list-nodes` count does not reveal it: registration runs through the discovery probe,
/// which is a SUBPROCESS and finds its own site-packages via `pyvenv.cfg`. Only execution breaks.
///
/// Doing it here makes the binary self-sufficient instead of dependent on the build tool that
/// happened to launch it — a cargo config is for configuring a *build*.
#[cfg(feature = "python")]
fn point_embedded_python_at_its_venv() {
    // An existing value is the documented override, and under `cargo run` it is already right.
    if std::env::var_os("PYTHONPATH").is_some() {
        return;
    }
    let Some(python) = goofi_py::interpreter_path() else { return };
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

/// Discover and register isolated-GIL subprocess Python node types (no build-time
/// Python needed — only a `python` interpreter at run time). Always available.
/// Returns the type names it actually registered (what `--list-nodes` reports).
fn register_subproc(state: &AppState, dir: Option<&str>, python: &str) -> Vec<String> {
    let Some(dir) = dir else { return Vec::new() };
    let types = match goofi_subproc::discover(std::path::Path::new(dir), python) {
        Ok(types) => types,
        Err(e) => {
            eprintln!("failed to discover subprocess nodes in {dir}: {e}");
            return Vec::new();
        }
    };
    let mut g = state.graph.lock().unwrap();
    // Only registrations that succeeded (a name colliding with a built-in is refused).
    let names: Vec<String> = types
        .into_iter()
        .filter_map(|t| {
            let name = t.manifest.type_name;
            let r = g.register_dyn_type(t.manifest, t.factory);
            note_registration(name, r).then(|| name.to_string())
        })
        .collect();
    println!("  registered {} subprocess node type(s) from {dir} (python `{python}`)", names.len());
    names
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

/// GIL-gate auto-router: probe each node file and register it in-process when its imports keep the
/// free-threaded GIL disabled, else quarantine it to a subprocess. THE node-discovery seam — the
/// bridge holds it as `AppState::scan_nodes`, so the boot scan below and every later `rescan_nodes`
/// route the same file the same way, by construction rather than by two implementations agreeing.
///
/// It reports and prints NOTHING: a rescan runs this same function whenever an agent writes a node
/// file, and it must not spew to stderr for doing its job. `boot_scan` does the talking.
#[cfg(feature = "python")]
fn register_auto(g: &mut Graph, dir: &Path, subproc_python: &str) -> Vec<ScannedType> {
    let ft = goofi_py::interpreter_path(); // the embedded FT interpreter, for probing
    let mut found = Vec::new();
    for path in sorted_dir(dir) {
        // ONE free-threaded probe answers both questions: it imports the module and constructs the
        // class (so a dep missing on the FT interpreter shows up as a failed probe), then reports
        // whether the GIL is still disabled — `gil_safe` IS the routing gate.
        if let goofi_py::Discovery::Found(d) = ft.as_deref().map_or(goofi_py::Discovery::Skip, |ftp| goofi_py::probe(&path, ftp)) {
            if d.gil_safe {
                let t = goofi_py::node_type_from(d);
                found.push(ScannedType {
                    type_name: t.manifest.type_name.to_string(),
                    tier: Tier::InProcess,
                    stamp: stamp(&path),
                    registration: g.register_dyn_type(t.manifest, t.factory),
                });
                continue;
            }
            // Loading it re-enabled the GIL — quarantine it to the subprocess tier below. (So does a
            // failed FT probe: typically a dep present on the subproc python but absent on the FT
            // interpreter, which must fall through rather than drop the node.)
        }
        // One probe, both outcomes: the subprocess tier is the last chance, so its result decides
        // between "registered" and "listed as unavailable".
        match goofi_subproc::probe(&path, subproc_python) {
            goofi_subproc::Discovery::Found(d) => {
                let t = goofi_subproc::node_type_from(subproc_python, d);
                found.push(ScannedType {
                    type_name: t.manifest.type_name.to_string(),
                    tier: Tier::Subprocess,
                    stamp: stamp(&path),
                    registration: g.register_dyn_type(t.manifest, t.factory),
                });
            }
            // Neither tier could load it. Register it as unavailable WITH the reason so the
            // palette explains itself — a node file that silently does not appear reads as
            // "goofi ignored my file" rather than "install this dependency".
            goofi_subproc::Discovery::Unavailable { type_name, reason } => {
                let registration = if g.register_unavailable(type_name.clone(), reason.clone()) {
                    Registration::Added
                } else {
                    Registration::Refused // a built-in owns the name
                };
                found.push(ScannedType {
                    type_name,
                    tier: Tier::Unavailable(reason),
                    stamp: stamp(&path),
                    registration,
                });
            }
            goofi_subproc::Discovery::Skip => {}
        }
    }
    found
}

#[cfg(not(feature = "python"))]
fn register_auto(_g: &mut Graph, _dir: &Path, _subproc_python: &str) -> Vec<ScannedType> {
    Vec::new()
}

/// Said once, by the boot summary: without the `python` feature there is no embedded interpreter,
/// hence no probe, hence no tier routing — the seam above finds nothing at all.
#[cfg(feature = "python")]
const NO_PYTHON_NOTE: &str = "";
#[cfg(not(feature = "python"))]
const NO_PYTHON_NOTE: &str = " (built without the `python` feature — node discovery is off)";

/// The boot scan, reported. It runs the bridge's own `rescan` — not merely the same seam — so the
/// baseline the palette's first refresh diffs against IS this scan, and pressing refresh with
/// nothing edited says "no changes" instead of re-announcing the whole shipped tree. What it adds
/// is the talking the seam deliberately does not do, including the collision warning, which is only
/// TRUE here: the boot registry starts empty, so a `Replaced` can only be a second node file
/// claiming a name an earlier one took. Returns the type names `--list-nodes` prints.
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
        assert!(cli.subproc_nodes.is_none() && cli.auto_nodes.is_empty());
        assert!(!cli.list_nodes && !cli.help);
    }

    #[test]
    fn reads_every_value_taking_flag() {
        let cli = parse(&[
            "--port", "9001", "--bind", "0.0.0.0", "--subproc-nodes", "a", "--auto-nodes", "b",
            "--subproc-python", "py", "--list-nodes",
        ])
        .expect("a well-formed invocation");
        assert_eq!(cli.port, 9001);
        assert_eq!(cli.bind, "0.0.0.0");
        assert_eq!(cli.subproc_nodes.as_deref(), Some("a"));
        assert_eq!(cli.auto_nodes, ["b"]);
        assert_eq!(cli.subproc_python.as_deref(), Some("py"));
        assert!(cli.list_nodes);
    }

    /// `--auto-nodes` ACCUMULATES where every other value-taking flag replaces. A packaged build
    /// bakes one into its launch command, so a user naming their own directory has to be *added*
    /// to the builtin tree — a last-wins flag would silently drop the shipped nodes, which is the
    /// opposite of what someone extending the palette is asking for.
    #[test]
    fn auto_nodes_accumulates_where_the_other_flags_replace() {
        let cli = parse(&["--auto-nodes", "builtin", "--bind", "a", "--auto-nodes", "mine",
                          "--bind", "b"])
            .expect("a repeated flag is well-formed");
        assert_eq!(cli.auto_nodes, ["builtin", "mine"], "--auto-nodes adds");
        assert_eq!(cli.bind, "b", "…while --bind still replaces");
    }

    /// A missing value is the same class of user error as an unknown flag, and used to be the
    /// only one handled silently: `--bind` alone served on the default host, and
    /// `--subproc-nodes` alone left the option `None`, which then satisfied the "no explicit node
    /// source" guard and auto-routed `nodes/` — the opposite tier from the one asked for.
    #[test]
    fn a_value_taking_flag_without_its_value_is_an_error() {
        for flag in ["--port", "--bind", "--subproc-nodes", "--auto-nodes", "--subproc-python"] {
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
