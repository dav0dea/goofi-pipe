//! `goofi-pipe` — launches the Rust engine + bridge, serving the SPA and the two
//! WebSocket planes. Flags: `--port N` (default 8000), `--bind HOST` (default
//! 127.0.0.1), `--headless` (accepted; no UI difference yet), `--python-nodes DIR`
//! (discover in-process Python nodes from DIR; requires the `python` feature).

use goofi_bridge::{resolve_frontend_dir, serve_app, spawn_tick, AppState};

#[tokio::main]
async fn main() {
    let mut port: u16 = 8000;
    let mut bind = String::from("127.0.0.1");
    let mut python_nodes: Option<String> = None;
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
            "--python-nodes" => {
                python_nodes = args.next();
            }
            "--headless" => {}
            "--list-nodes" => list_nodes = true,
            "-h" | "--help" => {
                println!(
                    "usage: goofi-pipe [--port N] [--bind HOST] [--headless] [--python-nodes DIR]"
                );
                return;
            }
            other => {
                eprintln!("unknown argument `{other}` (try --help)");
                std::process::exit(2);
            }
        }
    }

    if list_nodes {
        let mut names = goofi_bridge::catalog_type_names();
        names.extend(python_type_names(python_nodes.as_deref()));
        println!("{} node types: {}", names.len(), names.join(", "));
        return;
    }

    let state = AppState::new();
    register_python(&state, python_nodes.as_deref());
    spawn_tick(state.graph.clone(), 60);

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

/// Discover and register in-process Python node types into the live graph.
#[cfg(feature = "python")]
fn register_python(state: &AppState, dir: Option<&str>) {
    let Some(dir) = dir else { return };
    match goofi_py::discover(std::path::Path::new(dir)) {
        Ok(types) => {
            let n = types.len();
            let mut g = state.graph.lock().unwrap();
            for t in types {
                g.register_dyn_type(t.manifest, t.factory);
            }
            println!("  registered {n} Python node type(s) from {dir}");
        }
        Err(e) => eprintln!("failed to discover python nodes in {dir}: {e}"),
    }
}

#[cfg(not(feature = "python"))]
fn register_python(_state: &AppState, dir: Option<&str>) {
    if dir.is_some() {
        eprintln!("--python-nodes ignored: this binary was built without the `python` feature");
    }
}

/// The type names of discoverable Python nodes in `dir` (for `--list-nodes`).
#[cfg(feature = "python")]
fn python_type_names(dir: Option<&str>) -> Vec<String> {
    let Some(dir) = dir else { return Vec::new() };
    match goofi_py::discover(std::path::Path::new(dir)) {
        Ok(types) => types.iter().map(|t| t.manifest.type_name.to_string()).collect(),
        Err(e) => {
            eprintln!("failed to discover python nodes in {dir}: {e}");
            Vec::new()
        }
    }
}

#[cfg(not(feature = "python"))]
fn python_type_names(dir: Option<&str>) -> Vec<String> {
    if dir.is_some() {
        eprintln!("--python-nodes ignored: this binary was built without the `python` feature");
    }
    Vec::new()
}
