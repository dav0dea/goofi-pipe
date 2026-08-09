//! `goofi-pipe` — launches the Rust engine + bridge, serving the SPA and the two
//! WebSocket planes. Flags: `--port N` (default 8000), `--bind HOST` (default
//! 127.0.0.1), `--subproc-nodes DIR` (discover isolated-GIL subprocess Python nodes,
//! run on `--subproc-python`), `--auto-nodes DIR` (gil-gate routed). With no
//! `--*-nodes` flag it auto-discovers the default `nodes/` directory;
//! `--subproc-python` defaults to the repo-local `.venv`.

use std::future::Future;

use goofi_bridge::{resolve_frontend_dir, serve_app, spawn_workers, AppState};

/// The default node directory, auto-discovered (gil-gate routed) when no `--*-nodes` flag is given.
const DEFAULT_NODES_DIR: &str = "nodes";
/// The default subprocess-node interpreter — the repo-local `.venv` (the project convention).
const DEFAULT_VENV_PYTHON: &str = ".venv/bin/python";

/// The default subprocess interpreter: the repo-local `.venv` if present, else `python3`.
fn default_subproc_python() -> String {
    if std::path::Path::new(DEFAULT_VENV_PYTHON).is_file() {
        DEFAULT_VENV_PYTHON.to_string()
    } else {
        "python3".to_string()
    }
}

/// The parsed command line. `help` is a mode rather than a setting, so the caller decides what
/// to do with it — which is also what keeps the parse itself pure and testable.
#[derive(Debug)]
struct Cli {
    port: u16,
    bind: String,
    subproc_nodes: Option<String>,
    auto_nodes: Option<String>,
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
            auto_nodes: None,
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
            "--auto-nodes" => cli.auto_nodes = Some(need(args.next())?),
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
    let cli = match parse_args(std::env::args().skip(1)) {
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
             --subproc-python defaults to `{DEFAULT_VENV_PYTHON}` when present, else `python3`."
        );
        return;
    }
    std::process::exit(run(cli, AppState::new(), shutdown_signal()).await);
}

/// Everything the process does once it has a state, as a function that *returns* its exit code:
/// the workspace mount is reclaimed here, on the one path every outcome takes. The alternative is
/// three `std::process::exit` calls that each have to remember — and `exit` unwinds nothing, so a
/// destructor would not save them either. `shutdown` is only awaited once the server is up, so a
/// signal that lands during boot still takes the default disposition and leaves the mount behind:
/// one empty temp directory in a rare race, the same as any other crash.
async fn run(cli: Cli, state: AppState, shutdown: impl Future<Output = ()>) -> i32 {
    let Cli { port, bind, subproc_nodes, mut auto_nodes, subproc_python, list_nodes, help: _ } = cli;

    // Resolve defaults: the repo-local `.venv` for the subprocess tier (the project convention),
    // and — when no explicit node source was given — auto-route the default `nodes/` directory.
    let subproc_python = subproc_python.unwrap_or_else(default_subproc_python);
    if subproc_nodes.is_none() && auto_nodes.is_none()
        && std::path::Path::new(DEFAULT_NODES_DIR).is_dir()
    {
        auto_nodes = Some(DEFAULT_NODES_DIR.to_string());
    }

    if !list_nodes {
        register_evaluator(&state);
    }
    // `--list-nodes` runs the SAME registration the server does and reports its result, so the
    // listing is what actually registered — not a hand-kept mirror of the routing rule.
    let mut discovered = register_subproc(&state, subproc_nodes.as_deref(), &subproc_python);
    discovered.extend(register_auto(&state, auto_nodes.as_deref(), &subproc_python));

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
                let dir = resolve_frontend_dir();
                println!("goofi-pipe (rust backend) → http://{addr}");
                match &dir {
                    Some(d) => println!("  serving SPA from {}", d.display()),
                    None => println!("  API only — no SPA build found (set GOOFI_FRONTEND_BUILD or build frontend/)"),
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

/// Resolve on the first request to stop: ctrl-C, or the SIGTERM a service manager sends. A door
/// that cannot be installed must **never** resolve — an immediately-ready arm here would shut the
/// server down at startup rather than merely leaving that one door closed.
async fn shutdown_signal() {
    let interrupt = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                eprintln!("SIGTERM handler unavailable: {e}");
                std::future::pending::<()>().await;
            }
        }
    };
    tokio::select! {
        _ = interrupt => {}
        _ = terminate => {}
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

#[cfg(not(feature = "python"))]
fn register_evaluator(_state: &AppState) {
    // Without the `python` feature there is no embedded interpreter; expression bindings are
    // stored but not evaluated — the node reports "no expression evaluator available". Announce
    // it at startup so that error isn't a mystery (the pyo3 evaluator needs a free-threaded 3.14t
    // interpreter, hence the opt-in feature).
    println!("  param expressions DISABLED — rebuild with `--features python` to enable the evaluator");
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
    // Only registrations that succeeded (a name colliding with a built-in or an earlier
    // type is refused).
    let names: Vec<String> = types
        .into_iter()
        .filter_map(|t| {
            let name = t.manifest.type_name;
            g.register_dyn_type(t.manifest, t.factory).then(|| name.to_string())
        })
        .collect();
    println!("  registered {} subprocess node type(s) from {dir} (python `{python}`)", names.len());
    names
}

/// Node files from `dir` that appear in a directory, newest-first, deterministic.
#[cfg(feature = "python")]
fn sorted_dir(dir: &str) -> Vec<std::path::PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(rd) => {
            let mut v: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
            v.sort();
            v
        }
        Err(e) => {
            eprintln!("failed to read {dir}: {e}");
            Vec::new()
        }
    }
}

/// GIL-gate auto-router: probe each node file and register it in-process when its
/// imports keep the free-threaded GIL disabled, else quarantine it to a subprocess.
#[cfg(feature = "python")]
fn register_auto(state: &AppState, dir: Option<&str>, subproc_python: &str) -> Vec<String> {
    let Some(dir) = dir else { return Vec::new() };
    let ft = goofi_py::interpreter_path(); // the embedded FT interpreter, for probing
    let mut g = state.graph.lock().unwrap();
    let (mut n_in, mut n_sub, mut n_bad) = (0u32, 0u32, 0u32);
    let mut names = Vec::new();
    for path in sorted_dir(dir) {
        // ONE free-threaded probe answers both questions: it imports the module and constructs the
        // class (so a dep missing on the FT interpreter shows up as a failed probe), then reports
        // whether the GIL is still disabled — `gil_safe` IS the routing gate.
        if let goofi_py::Discovery::Found(d) = ft.as_deref().map_or(goofi_py::Discovery::Skip, |ftp| goofi_py::probe(&path, ftp)) {
            if d.gil_safe {
                let t = goofi_py::node_type_from(d);
                if g.register_dyn_type(t.manifest, t.factory) {
                    n_in += 1;
                    names.push(format!("{} (in-proc)", t.manifest.type_name));
                }
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
                if g.register_dyn_type(t.manifest, t.factory) {
                    n_sub += 1;
                    names.push(format!("{} (subproc)", t.manifest.type_name));
                }
            }
            // Neither tier could load it. Register it as unavailable WITH the reason so the
            // palette explains itself — a node file that silently does not appear reads as
            // "goofi ignored my file" rather than "install this dependency".
            goofi_subproc::Discovery::Unavailable { type_name, reason } => {
                eprintln!("  node `{type_name}` unavailable: {reason}");
                names.push(format!("{type_name} (unavailable)"));
                if g.register_unavailable(type_name, reason) {
                    n_bad += 1;
                }
            }
            goofi_subproc::Discovery::Skip => {}
        }
    }
    let bad = if n_bad > 0 { format!(", {n_bad} unavailable") } else { String::new() };
    println!("  auto-routed {n_in} in-process + {n_sub} subprocess node type(s) from {dir}{bad}");
    names
}

#[cfg(not(feature = "python"))]
fn register_auto(_state: &AppState, dir: Option<&str>, _subproc_python: &str) -> Vec<String> {
    if dir.is_some() {
        eprintln!("--auto-nodes ignored: this binary was built without the `python` feature");
    }
    Vec::new()
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
        assert!(cli.subproc_nodes.is_none() && cli.auto_nodes.is_none());
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
        assert_eq!(cli.auto_nodes.as_deref(), Some("b"));
        assert_eq!(cli.subproc_python.as_deref(), Some("py"));
        assert!(cli.list_nodes);
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
        assert_eq!(run(cli, state, std::future::ready(())).await, 0);
        // The NONCE directory is what goes, not just `workspace` — else every run leaves an empty
        // husk behind. Asserting on the parent covers the leaf too.
        let husk = mount.parent().expect("the mount is nested under a nonce dir");
        assert!(!husk.exists(), "the nonce directory goes too, not just workspace: {}", husk.display());

        // `--list-nodes` returns before the server ever binds; the same tail must still reclaim.
        let listed = AppState::new();
        let m2 = listed.mount();
        let cli = Cli { list_nodes: true, ..Cli::default() };
        assert_eq!(run(cli, listed, std::future::pending()).await, 0);
        assert!(!m2.exists(), "--list-nodes reclaims too: {}", m2.display());
    }

    #[test]
    fn help_is_a_mode_the_caller_handles() {
        assert!(parse(&["--help"]).expect("help parses").help);
        assert!(parse(&["-h"]).expect("help parses").help);
    }
}
