//! `goofi-pipe` — launches the Rust engine + bridge, serving the SPA and the two
//! WebSocket planes. Flags: `--port N` (default 8000), `--bind HOST` (default
//! 127.0.0.1), `--subproc-nodes DIR` (discover isolated-GIL subprocess Python nodes,
//! run on `--subproc-python`), `--auto-nodes DIR` (gil-gate routed). With no
//! `--*-nodes` flag it auto-discovers the default `nodes/` directory;
//! `--subproc-python` defaults to the repo-local `.venv`.

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

#[tokio::main]
async fn main() {
    let mut port: u16 = 8000;
    let mut bind = String::from("127.0.0.1");
    let mut subproc_nodes: Option<String> = None;
    let mut auto_nodes: Option<String> = None;
    let mut subproc_python: Option<String> = None;
    let mut list_nodes = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                if let Some(v) = args.next() {
                    port = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --port `{v}`");
                        std::process::exit(2);
                    });
                }
            }
            "--bind" => {
                if let Some(v) = args.next() {
                    bind = v;
                }
            }
            "--subproc-nodes" => {
                subproc_nodes = args.next();
            }
            "--auto-nodes" => {
                auto_nodes = args.next();
            }
            "--subproc-python" => {
                subproc_python = args.next();
            }
            "--list-nodes" => list_nodes = true,
            "-h" | "--help" => {
                println!(
                    "usage: goofi-pipe [--port N] [--bind HOST] \
                     [--subproc-nodes DIR] [--auto-nodes DIR] [--list-nodes] \
                     [--subproc-python BIN]\n\
                     \n  \
                     With no --*-nodes flag, auto-discovers `{DEFAULT_NODES_DIR}/` (each node routed \
                     in-process if free-threading-safe, else to a subprocess).\n  \
                     --subproc-python defaults to `{DEFAULT_VENV_PYTHON}` when present, else `python3`."
                );
                return;
            }
            other => {
                eprintln!("unknown argument `{other}` (try --help)");
                std::process::exit(2);
            }
        }
    }

    // Resolve defaults: the repo-local `.venv` for the subprocess tier (the project convention),
    // and — when no explicit node source was given — auto-route the default `nodes/` directory.
    let subproc_python = subproc_python.unwrap_or_else(default_subproc_python);
    if subproc_nodes.is_none() && auto_nodes.is_none()
        && std::path::Path::new(DEFAULT_NODES_DIR).is_dir()
    {
        auto_nodes = Some(DEFAULT_NODES_DIR.to_string());
    }

    let state = AppState::new();
    if !list_nodes {
        register_evaluator(&state);
    }
    // `--list-nodes` runs the SAME registration the server does and reports its result, so the
    // listing is what actually registered — not a hand-kept mirror of the routing rule.
    let mut discovered = register_subproc(&state, subproc_nodes.as_deref(), &subproc_python);
    discovered.extend(register_auto(&state, auto_nodes.as_deref(), &subproc_python));
    if list_nodes {
        let mut names = goofi_bridge::catalog_type_names();
        names.extend(discovered);
        println!("{} node types: {}", names.len(), names.join(", "));
        return;
    }
    spawn_workers(&state); // adaptive tick loop + 2 Hz node-stats (header ufreq + error transitions)

    let listener = match tokio::net::TcpListener::bind((bind.as_str(), port)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {bind}:{port}: {e}");
            std::process::exit(1);
        }
    };
    let addr = listener.local_addr().unwrap();
    let dir = resolve_frontend_dir();

    println!("goofi-pipe (rust backend) → http://{addr}");
    match &dir {
        Some(d) => println!("  serving SPA from {}", d.display()),
        None => println!("  API only — no SPA build found (set GOOFI_FRONTEND_BUILD or build frontend/)"),
    }

    if let Err(e) = serve_app(listener, state, dir).await {
        eprintln!("server error: {e}");
        std::process::exit(1);
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
