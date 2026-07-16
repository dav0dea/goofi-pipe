//! `goofi-pipe` — launches the Rust engine + bridge, serving the SPA and the two
//! WebSocket planes. Flags: `--port N` (default 8000), `--bind HOST` (default
//! 127.0.0.1), `--headless` (accepted; no UI difference yet).

use goofi_bridge::{resolve_frontend_dir, serve_app, spawn_tick, AppState};

#[tokio::main]
async fn main() {
    let mut port: u16 = 8000;
    let mut bind = String::from("127.0.0.1");
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
            "--headless" => {}
            "-h" | "--help" => {
                println!("usage: goofi-pipe [--port N] [--bind HOST] [--headless]");
                return;
            }
            other => {
                eprintln!("unknown argument `{other}` (try --help)");
                std::process::exit(2);
            }
        }
    }

    let state = AppState::new();
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
