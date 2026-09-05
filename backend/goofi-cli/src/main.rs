//! `goofi` — the binary: serve by default, or be the CLIENT of a running server. Client mode
//! holds zero op knowledge: it resolves WHICH server, sends the line, prints the answer.

use std::future::Future;
use std::path::{Path, PathBuf};

use goofi_bridge::{serve_app, spawn_workers, AppState, HEADLESS_BUILD, SPA};
use goofi_node::{Isolation, Scanned};

#[derive(Debug)]
struct Cli {
    /// `None` until `--port` names one; [`DEFAULT_PORT`] otherwise.
    port: Option<u16>,
    bind: String,
    /// Node source roots scanned before the patch's own; a later entry wins a shared type name.
    extra_nodes: Vec<String>,
    list_nodes: bool,
    /// Serve the API alone: the SPA's routes are never mounted. Also set by `GOOFI_HEADLESS` in
    /// the environment and by a binary built with it, both folded in by [`main`].
    headless: bool,
    /// Open `/dev/*`, the development surfaces. Also set by `GOOFI_DEBUG` in the environment.
    debug: bool,
    /// A PUBLIC goofi: no terminal, no agents, no filesystem, no save or load, no audio. Also set
    /// by `GOOFI_DEMO` in the environment. Not a sandbox — see `roadmap/demo-mode.md`.
    demo: bool,
    help: bool,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            port: None,
            bind: String::from("127.0.0.1"),
            extra_nodes: Vec::new(),
            list_nodes: false,
            headless: false,
            debug: false,
            demo: false,
            help: false,
        }
    }
}

const USAGE: &str = "usage: goofi [serve] [--port N] [--bind HOST] \
     [--extra-nodes DIR] [--list-nodes] [--headless] [--debug] [--demo]";

fn headless_env() -> bool {
    matches!(std::env::var("GOOFI_HEADLESS").as_deref(), Ok("1") | Ok("true"))
}

fn debug_env() -> bool {
    matches!(std::env::var("GOOFI_DEBUG").as_deref(), Ok("1") | Ok("true"))
}

/// The port with no door naming one.
const DEFAULT_PORT: u16 = 8000;

fn demo_env() -> bool {
    matches!(std::env::var("GOOFI_DEMO").as_deref(), Ok("1") | Ok("true"))
}

/// Parse the argument list (already skipping argv[0]). `Err` is the message to print before
/// exiting 2.
fn parse_args<I: Iterator<Item = String>>(mut args: I) -> Result<Cli, String> {
    let mut cli = Cli::default();
    while let Some(arg) = args.next() {
        let need = |v: Option<String>| v.ok_or_else(|| format!("{arg} requires a value (try --help)"));
        match arg.as_str() {
            "--port" => {
                let v = need(args.next())?;
                cli.port = Some(v.parse().map_err(|_| format!("invalid --port `{v}`"))?);
            }
            "--bind" => cli.bind = need(args.next())?,
            "--extra-nodes" => cli.extra_nodes.push(need(args.next())?),
            "--list-nodes" => cli.list_nodes = true,
            "--headless" => cli.headless = true,
            "--debug" => cli.debug = true,
            "--demo" => cli.demo = true,
            "-h" | "--help" => cli.help = true,
            other => return Err(format!("unknown argument `{other}` (try --help)")),
        }
    }
    Ok(cli)
}

fn main() {
    // Bare or flag-first argv serves — what `cargo run` depends on. A bare WORD is a command for
    // a running server, except the few the client itself owns (`ops::RESERVED`'s doors). The
    // client path is three blocking syscalls, so only the serve arm builds a runtime.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let rest = match argv.first().map(String::as_str) {
        None => argv,
        Some("help") | Some("--help") | Some("-h") => std::process::exit(help_main(&argv[1..])),
        Some("-") => std::process::exit(client_stdin(&argv[1..])),
        Some("serve") => argv[1..].to_vec(),
        // The binary is its own plugin scanner: a child per bundle, so a crash there is a
        // refusal here.
        Some("vst3-scan") => std::process::exit(goofi_audio::vst3::scan_main(&argv[1..])),
        Some(first) if first.starts_with('-') => argv,
        Some(_) => std::process::exit(client_main(argv)),
    };
    let (windows, ui) = match goofi_audio::ui::Loop::open() {
        Ok((windows, ui)) => (Some(windows), Some(ui)),
        Err(_) => (None, None),
    };
    let serve = move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("the serve runtime")
            .block_on(serve_main(rest, ui))
    };
    // Where a display answers, the main thread is the window thread — a plugin's editor lives
    // there — and the server runs beside it; where none does, it serves as it always did.
    match windows {
        Some(windows) => {
            std::thread::Builder::new().name("goofi-serve".into()).spawn(serve).expect("the serve thread");
            windows.run();
        }
        None => serve(),
    }
}

async fn serve_main(rest: Vec<String>, ui: Option<goofi_audio::ui::Ui>) {
    let mut cli = match parse_args(rest.into_iter()) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    // The three doors meet here, once: a binary built headless has no app to serve at all.
    cli.headless |= headless_env() || HEADLESS_BUILD;
    cli.debug |= debug_env();
    cli.demo |= demo_env();
    if cli.help {
        // `goofi serve --help` / a flag mix that asked: the SERVE usage, not the op help door.
        println!(
            "{USAGE}\n\
             \n  \
             Scans every --extra-nodes ROOT — a folder of node files, `.py` and `.rs` — and then the \
             open patch's own workspace, which wins a shared type name. \
             Each node is routed in-process if free-threading-safe, else to a subprocess on \
             `{}`, which `cargo run -p goofi-init` provisions.\n  \
             GOOFI_HEADLESS=1 in the environment is --headless; setting it for the BUILD leaves \
             the app out of the binary entirely. GOOFI_DEBUG=1 is --debug, which opens `/dev/*` \
             — the UI primitive gallery and the other development surfaces.",
            goofi_init::GIL_VENV
        );
        return;
    }
    let python = match default_subproc_python() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let mode = goofi_bridge::Mode { headless: cli.headless, demo: cli.demo };
    let state = AppState::new(mode, goofi_bridge::Clock::Device);
    std::process::exit(run(cli, python, state, shutdown_signal(), ui).await);
}

/// Send lines to the resolved server and print each entry — decoded NPY bytes when the result
/// carries them, the rendered text otherwise, or the raw JSON under `--json`.
fn forward(lines: &[String], json: bool) -> i32 {
    let target = match goofi_client::resolve_target() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let actor = std::env::var("GOOFI_ACTOR").ok();
    match goofi_client::exec(&target.url, lines, actor.as_deref()) {
        Ok(entries) => {
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            for e in &entries {
                let bytes = match json {
                    true => {
                        let mut b = serde_json::to_string_pretty(&e["result"])
                            .unwrap_or_default()
                            .into_bytes();
                        b.push(b'\n');
                        b
                    }
                    false => goofi_client::rendered(e),
                };
                let _ = out.write_all(&bytes);
            }
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

/// Split off the client-consumed `--json`; everything else is the server's line, re-quoted by
/// the same word rules bash used to split it.
fn take_json(words: &mut Vec<String>) -> bool {
    let n = words.len();
    words.retain(|w| w != "--json");
    words.len() != n
}

fn client_main(mut words: Vec<String>) -> i32 {
    let json = take_json(&mut words);
    match (words.first().map(String::as_str), words.get(1).map(String::as_str)) {
        (Some("session"), Some("list")) => return print_sessions(json),
        (Some("completions"), shell) => return print_completions(shell),
        (Some("op"), Some("complete")) => return complete_line(&words[2..]),
        (Some("agent"), Some("term")) => {
            eprintln!("`agent term` is not built yet — the app's agent panel serves the terminal.");
            return 1;
        }
        (Some("plugin"), _) => {
            eprintln!("`plugin` is not built yet — the word is reserved for plugin ops.");
            return 1;
        }
        _ => {}
    }
    forward(&[shell_words::join(words.iter().map(String::as_str))], json)
}

/// The completion callback: a running server answers with its LIVE vocabulary (its node uids,
/// its types); with none, the compiled-in registry answers the static half — same fallback shape
/// as [`help_main`]. Quiet on every failure: a completion must never print an error into a
/// half-typed command line.
fn complete_line(rest: &[String]) -> i32 {
    let line = rest.first().map(String::as_str).unwrap_or_default();
    if let Ok(target) = goofi_client::resolve_target() {
        let cmd = shell_words::join(["op", "complete", line]);
        if let Ok(entries) = goofi_client::exec(&target.url, &[cmd], None) {
            if let Some(text) = entries.first().and_then(|e| e["text"].as_str()) {
                println!("{text}");
                return 0;
            }
        }
    }
    let ops = goofi_bridge::ops::table(goofi_bridge::Mode::default());
    for (word, doc) in goofi_bridge::phrase::complete(&ops, None, line) {
        println!("{word}\t{doc}");
    }
    0
}

/// `goofi completions zsh|bash` — the script that wires a shell's TAB to [`complete_line`]. The
/// script holds NO vocabulary: every keystroke asks `goofi op complete`, so completions are as
/// current as the server answering them.
fn print_completions(shell: Option<&str>) -> i32 {
    // zsh: `words` holds the current (partial) word last; joining keeps its emptiness, so the
    // callback can tell `node<TAB>` from `node <TAB>`. compinit is bootstrapped when the rc file
    // has not run it yet — `compdef` does not exist before it has.
    const ZSH: &str = r#"# goofi completion — add to ~/.zshrc:  eval "$(goofi completions zsh)"
_goofi() {
	local -a cands lines
	local line="${(j: :)${(@)words[2,$CURRENT]}}"
	lines=("${(@f)$("__GOOFI__" op complete "$line" 2>/dev/null)}")
	for l in "${lines[@]}"; do
		[[ -n "$l" ]] && cands+=("${l%%$'\t'*}:${l#*$'\t'}")
	done
	(( ${#cands} )) && _describe -V goofi cands
}
if ! typeset -f compdef >/dev/null; then
	autoload -Uz compinit
	compinit
fi
compdef _goofi goofi "__GOOFI__""#;
    const BASH: &str = r#"# goofi completion — add to ~/.bashrc:  eval "$(goofi completions bash)"
_goofi() {
	local line="${COMP_LINE#* }"
	[[ "$COMP_LINE" == *' '* ]] || line=""
	local IFS=$'\n'
	COMPREPLY=($("__GOOFI__" op complete "$line" 2>/dev/null | cut -f1))
}
complete -F _goofi goofi "__GOOFI__""#;
    // The script pins THIS binary's path: a dev shell has no `goofi` on PATH, and the eval
    // re-resolves on every shell start, so an installed binary pins its installed path.
    let me = std::env::current_exe()
        .ok()
        .and_then(|p| p.into_os_string().into_string().ok())
        .unwrap_or_else(|| "goofi".into());
    match shell {
        Some("zsh") => println!("{}", ZSH.replace("__GOOFI__", &me)),
        Some("bash") => println!("{}", BASH.replace("__GOOFI__", &me)),
        _ => {
            eprintln!(
                "usage: goofi completions zsh|bash — register with\n  eval \"$(goofi completions zsh)\"\nin the shell or its rc file"
            );
            return 2;
        }
    }
    0
}

/// `goofi -`: stdin lines as ONE batch — several ops, one undo step.
fn client_stdin(rest: &[String]) -> i32 {
    let mut rest = rest.to_vec();
    let json = take_json(&mut rest);
    if let Some(stray) = rest.first() {
        eprintln!("`goofi -` reads its commands from stdin — `{stray}` has no meaning here");
        return 2;
    }
    let lines: Vec<String> = std::io::stdin()
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .collect();
    forward(&lines, json)
}

fn print_sessions(json: bool) -> i32 {
    let rows = goofi_client::list();
    let current = std::env::var("GOOFI_SESSION").ok();
    let current = |s: &goofi_core::home::Session| current.as_deref() == Some(&s.id);
    if json {
        let rows: Vec<serde_json::Value> = rows
            .iter()
            .map(|(s, p)| {
                serde_json::json!({
                    "id": s.id,
                    "url": s.url,
                    "state": state_word(p),
                    "current": current(s),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows).unwrap_or_default());
        return 0;
    }
    if rows.is_empty() {
        println!("no running goofi — start one with `goofi`");
        return 0;
    }
    for (s, p) in rows {
        let mark = if current(&s) { "  ← GOOFI_SESSION" } else { "" };
        println!("{}  {}  {}{mark}", s.id, s.url, state_word(&p));
    }
    0
}

fn state_word(p: &goofi_client::Probed) -> &'static str {
    match p {
        goofi_client::Probed::Live => "live",
        goofi_client::Probed::Unresponsive => "unresponsive",
    }
}

/// `goofi help [words…]`: any LIVE session answers — help does not depend on which — and with
/// none, the COMPILED-IN registry answers through the same renderer, so there is one help text.
/// An unresponsive record must not stall the one command a stuck user reaches for.
fn help_main(rest: &[String]) -> i32 {
    let mut rest = rest.to_vec();
    take_json(&mut rest); // help is text; the flag is not a word to look up
    let rows = goofi_client::list();
    let Some((live, _)) = rows.iter().find(|(_, p)| *p == goofi_client::Probed::Live) else {
        let words: Vec<String> = std::iter::once("help".to_string()).chain(rest.clone()).collect();
        match goofi_bridge::phrase::help(&goofi_bridge::ops::table(goofi_bridge::Mode::default()), &words) {
            Some(h) => {
                println!("no running server — the built-in index answers; `goofi serve` starts one.");
                println!("{h}");
                return 0;
            }
            None => {
                eprintln!("nothing under `{}`", rest.join(" "));
                return 1;
            }
        }
    };
    let words: Vec<&str> =
        std::iter::once("help").chain(rest.iter().map(String::as_str)).collect();
    match goofi_client::exec(&live.url, &[shell_words::join(words)], None) {
        Ok(entries) => {
            for e in &entries {
                println!("{}", e["text"].as_str().unwrap_or_default());
            }
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

/// The interpreter the subprocess tier runs on: the venv `goofi-init` made, and only that one.
fn default_subproc_python() -> Result<String, String> {
    goofi_init::venv_python(&goofi_init::repo_root().join(goofi_init::GIL_VENV))
        .map(|p| p.display().to_string())
        .ok_or_else(|| format!("no {} — {}", goofi_init::GIL_VENV, goofi_init::RUN_ME))
}

/// The warning a `--bind` beyond this machine earns, or `None` for the loopback default. A name
/// that is not an address warns too: only a parseable address can be proven local.
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

/// Everything the process does once it has a state, returning its exit code: `std::process::exit`
/// unwinds nothing, so the workspace mount is reclaimed here rather than by a destructor.
async fn run(
    cli: Cli,
    subproc_python: String,
    mut state: AppState,
    shutdown: impl Future<Output = ()>,
    ui: Option<goofi_audio::ui::Ui>,
) -> i32 {
    // Before ANY use of the embedded interpreter.
    point_embedded_python_at_its_venv();

    let Cli { port, bind, extra_nodes, list_nodes, headless, debug, demo, help: _ } = cli;
    let port = port.unwrap_or(DEFAULT_PORT);

    if !list_nodes {
        register_evaluator(&state);
    }
    state.roots.extend(extra_nodes.iter().map(PathBuf::from));
    ensure_packages(&state.roots, &subproc_python);
    // Handed to the engine before anything scans, so the boot scan and every rescan share it.
    goofi_bridge::signal_engine(&mut state.graph.lock().unwrap())
        .set_python(goofi_signal::Python::new(subproc_python.clone()));
    if !demo {
        let mut g = state.graph.lock().unwrap();
        let audio = goofi_bridge::audio_engine(&mut g);
        if let Ok(own) = std::env::current_exe() {
            audio.set_vst3(own, goofi_audio::vst3::platform_dirs());
        }
        audio.set_ui(ui);
    }
    boot_scan(&state);

    let code = if list_nodes {
        let names = goofi_bridge::catalog_type_names(&state.graph.lock().unwrap());
        println!("{} node types: {}", names.len(), names.join(", "));
        0
    // An arm of this chain rather than an early `return`: only the tail of this function gives
    // the workspace mount back.
    } else if !headless && SPA.is_empty() {
        eprintln!("refusing to start: no app is compiled into this binary.");
        eprintln!(
            "  The app is compiled in, so building it is not enough — build it, then rebuild \
             goofi:"
        );
        eprintln!("    npm install && npm run build   (in frontend/)");
        eprintln!("    cargo build");
        eprintln!("  Or serve the API alone: --headless, or GOOFI_HEADLESS=1.");
        1
    } else {
        spawn_workers(&state);
        match tokio::net::TcpListener::bind((bind.as_str(), port)).await {
            Err(e) => {
                eprintln!("failed to bind {bind}:{port}: {e}");
                1
            }
            Ok(listener) => {
                let addr = listener.local_addr().unwrap();
                // The session file needs the REAL address: `--port 0` makes it knowable
                // nowhere else.
                state.set_bound(addr);
                // Only a real server writes into the home: its record, and the config seed.
                goofi_core::home::seed_config();
                let _session = SessionFile::write(&state.instance_id, &state.local_url());
                // The OPENABLE spelling, as the session file records it — `http://0.0.0.0` is
                // not an address a browser can visit.
                let url = state.local_url();
                println!("goofi → {url}");
                if !demo {
                    println!("  MCP endpoint → {url}/mcp");
                }
                let spa = if headless { &[][..] } else { SPA };
                if headless {
                    println!("  headless: the API only, no app served");
                } else {
                    println!("  open {url} to use it");
                }
                if demo {
                    println!("  demo: no terminal, no agents, no filesystem, no audio");
                }
                if debug && !headless {
                    println!("  debug: {url}/dev/ui is open — the UI primitive gallery");
                }
                // Last, and on stderr, so it is the line still on screen and survives a `> log`.
                if let Some(warning) = exposure_warning(&bind).filter(|_| !demo) {
                    eprintln!("{warning}");
                }
                // The stop is here, not in `serve_app`, whose other callers serve forever.
                tokio::select! {
                    served = serve_app(listener, state.clone(), spa, debug) => match served {
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
    // The order is load-bearing: the agents leave before their workspace goes, and a node's
    // thread releases its shared memory before the mount goes.
    state.harnesses.reap_all(std::time::Duration::from_secs(5));
    state.graph.lock().unwrap().shutdown();
    state.release_mount();
    code
}

/// The `$GOOFI_HOME/.goofi/sessions/<id>.json` record of THIS server, removed when serving ends.
/// Only the binary's serve path writes one — an in-process test server records nothing — and what
/// a kill leaves behind, the next reader's probe sweeps.
struct SessionFile(String);

impl SessionFile {
    fn write(id: &str, url: &str) -> SessionFile {
        goofi_core::home::write_session(id, url);
        SessionFile(id.to_string())
    }
}

impl Drop for SessionFile {
    fn drop(&mut self) {
        goofi_core::home::remove_session(&self.0);
    }
}

/// Resolve on the first request to stop. A door that cannot be installed must **never** resolve —
/// an immediately-ready arm would shut the server down at startup.
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

/// The stop a service manager sends, which ctrl-C does not cover — `SIGTERM` where signals exist.
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

/// Windows has no SIGTERM: the console closing and the machine going down stand in for it.
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

/// Install the pyo3 param-expression evaluator into the graph.
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

/// Hand the EMBEDDED interpreter the venv pyo3 was linked against: pyo3 links `libpython` from
/// that venv's BASE install, so the venv's own site-packages is on no search path.
#[cfg(feature = "python")]
fn point_embedded_python_at_its_venv() {
    // An existing value is the documented override.
    if std::env::var_os("PYTHONPATH").is_some() {
        return;
    }
    let Some(python) = goofi_python::inproc::interpreter_path() else { return };
    let Some(venv) = Path::new(&python).parent().and_then(Path::parent) else { return };
    if let Some(dir) = goofi_init::site_packages(venv) {
        std::env::set_var("PYTHONPATH", dir);
    }
}

#[cfg(not(feature = "python"))]
fn point_embedded_python_at_its_venv() {}

/// Every node directory's `requirements.txt`, checked against both interpreters before the scan
/// imports anything. Nothing is installed unasked: a terminal is asked once, and anything else is
/// told what will be unavailable and served through.
#[cfg(feature = "python")]
fn ensure_packages(dirs: &[PathBuf], subproc_python: &str) {
    use std::io::IsTerminal;
    let reqs = goofi_init::requirements_in(dirs);
    if reqs.is_empty() {
        return;
    }
    let root = goofi_init::repo_root();
    let interpreters =
        [goofi_init::venv_python(&root.join(goofi_init::FT_VENV)), Some(PathBuf::from(subproc_python))];
    let mut lacking = Vec::new();
    for py in interpreters.into_iter().flatten() {
        let shown = py.strip_prefix(&root).unwrap_or(&py).display().to_string();
        match goofi_init::missing_packages(&py, &reqs) {
            Ok(missing) if missing.is_empty() => {}
            Ok(missing) => {
                eprintln!("  {shown} lacks {}", missing.join(", "));
                lacking.push(py);
            }
            Err(e) => eprintln!("  could not check {shown}: {e}"),
        }
    }
    if lacking.is_empty() {
        return;
    }
    let from = reqs.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", ");
    if !std::io::stdin().is_terminal() {
        eprintln!("  named by {from}; no terminal to ask, so those nodes will be unavailable");
        return;
    }
    eprint!("  named by {from} — install now? [y/N] ");
    let mut answer = String::new();
    let _ = std::io::stdin().read_line(&mut answer);
    if !answer.trim().eq_ignore_ascii_case("y") {
        eprintln!("  not installed; those nodes will be unavailable");
        return;
    }
    for py in lacking {
        if let Err(e) = goofi_init::install_packages(&py, &reqs) {
            eprintln!("  {e}");
        }
    }
}

#[cfg(not(feature = "python"))]
fn ensure_packages(_dirs: &[PathBuf], _subproc_python: &str) {}

#[cfg(not(feature = "python"))]
fn register_evaluator(_state: &AppState) {
    println!("  param expressions DISABLED — rebuild with `--features python` to enable the evaluator");
}

/// One boot registration, reported — the boot registry starts empty, so a replacement here can
/// only be two files claiming one name.
fn note_replaced(name: &str, replaced: bool) {
    if replaced {
        eprintln!("warning: two node files claim the type name `{name}`; the later one wins");
    }
}

#[cfg(feature = "python")]
const NO_PYTHON_NOTE: &str = "";
#[cfg(not(feature = "python"))]
const NO_PYTHON_NOTE: &str = " (built without the `python` feature — node discovery is off)";

/// The boot scan, reported. It runs the bridge's own `rescan`, so the baseline the first refresh
/// diffs against IS this scan.
fn boot_scan(state: &AppState) {
    goofi_bridge::prebuild(state, &state.mount());
    let (found, dirs) = {
        let mut g = state.graph.lock().unwrap();
        let patch = state.mount();
        (goofi_bridge::rescan(state, &mut g, &patch).1, state.roots.clone())
    };
    let (mut n_native, mut n_in, mut n_sub, mut n_bad) = (0u32, 0u32, 0u32, 0u32);
    for t in found {
        match t.outcome {
            Scanned::Registered { isolation, replaced } => {
                note_replaced(&t.type_name, replaced);
                match isolation {
                    Isolation::Native => n_native += 1,
                    Isolation::InProcess => n_in += 1,
                    Isolation::Subprocess => n_sub += 1,
                }
            }
            Scanned::Unavailable(reason) => {
                eprintln!("  node `{}` unavailable: {reason}", t.type_name);
                n_bad += 1;
            }
        }
    }
    let bad = if n_bad > 0 { format!(", {n_bad} unavailable") } else { String::new() };
    let from = dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ");
    println!(
        "  {n_native} native + {n_in} in-process + {n_sub} subprocess node type(s) from {from}{bad}{NO_PYTHON_NOTE}"
    );
}

// The suite lives in `goofi-tests`; a binary has no lib target for it to reach into.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    /// A booted state under a throwaway home, so no test writes the user's own `.goofi`.
    fn walled() -> AppState {
        static WALL: std::sync::Once = std::sync::Once::new();
        WALL.call_once(|| {
            let dir = std::env::temp_dir().join(format!("goofi-cli-test-home-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::env::set_var("GOOFI_HOME", dir);
        });
        AppState::new(goofi_bridge::Mode::default(), goofi_bridge::Clock::External)
    }

    #[test]
    fn defaults_with_no_arguments() {
        let cli = parse(&[]).expect("no arguments is a valid invocation");
        assert_eq!(cli.port, None, "…and the port is decided by the doors, not by the parse");
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
        assert_eq!(cli.port, Some(9001));
        assert_eq!(cli.bind, "0.0.0.0");
        assert_eq!(cli.extra_nodes, ["b"]);
        assert!(cli.list_nodes);
    }

    #[test]
    fn extra_nodes_accumulates_where_the_other_flags_replace() {
        let cli = parse(&["--extra-nodes", "theirs", "--bind", "a", "--extra-nodes", "mine",
                          "--bind", "b"])
            .expect("a repeated flag is well-formed");
        assert_eq!(cli.extra_nodes, ["theirs", "mine"], "--extra-nodes adds");
        assert_eq!(cli.bind, "b", "…while --bind still replaces");
    }

    #[test]
    fn a_value_taking_flag_without_its_value_is_an_error() {
        for flag in ["--port", "--bind", "--extra-nodes"] {
            let err = parse(&[flag]).expect_err(&format!("`{flag}` alone must not be ignored"));
            assert!(err.contains(flag), "the message names the flag: {err}");
        }
    }

    #[test]
    fn a_bind_beyond_this_machine_says_what_it_exposes() {
        for safe in ["127.0.0.1", "localhost", "::1", "127.0.0.53"] {
            assert!(exposure_warning(safe).is_none(), "`{safe}` is this machine");
        }
        for open in ["0.0.0.0", "::", "192.168.7.5", "goofi.local"] {
            let warn = exposure_warning(open).unwrap_or_else(|| panic!("`{open}` warns"));
            assert!(warn.contains(open), "the warning names the address: {warn}");
            assert!(warn.contains("shell"), "the warning names the exposure: {warn}");
            assert!(warn.contains("no authentication"), "…and that nothing else guards it: {warn}");
        }
    }

    #[test]
    fn rejects_an_unparseable_port_and_an_unknown_flag() {
        assert!(parse(&["--port", "nope"]).unwrap_err().contains("--port"));
        assert!(parse(&["--python-nodes", "x"]).unwrap_err().contains("unknown argument"));
    }

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

    /// A node that records its own destruction, on the node's own thread.
    struct Tracked(Arc<std::sync::atomic::AtomicBool>);
    impl goofi_signal_sdk::Node for Tracked {
        fn process(
            &mut self,
            _i: &goofi_signal_sdk::Inputs<'_>,
            _o: &mut goofi_signal_sdk::Outputs<'_>,
            _c: &mut goofi_signal_sdk::NodeCtx,
            _p: &goofi_node::Params<'_>,
        ) -> goofi_signal_sdk::NodeResult {
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
        tags: &[],
        doc: "records its own teardown",
        inputs: &[],
        outputs: &[],
        params: &[],
        producer: true,
    };

    #[tokio::test]
    async fn a_signal_stops_every_node_before_the_run_returns() {
        let state = walled();
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let graph = state.graph.clone();
        {
            let mut g = graph.lock().unwrap();
            let flag = released.clone();
            goofi_bridge::register_dyn_type(&mut g, &TRACKED, Box::new(move |_| Box::new(Tracked(flag.clone()))), &goofi_node::NATIVE);
            g.add_node("_TestTracked", None).expect("a test node");
        }
        // An already-resolved shutdown takes the same path ctrl-C does; port 0 binds ephemerally.
        let cli = Cli { port: Some(0), ..Cli::default() };
        assert_eq!(run(cli, "python3".into(), state, std::future::ready(()), None).await, 0);
        assert!(
            released.load(std::sync::atomic::Ordering::Acquire),
            "the node's runtime was dropped — its shared memory went with it — before the exit"
        );
        assert_eq!(graph.lock().unwrap().node_count(), 0, "…and the graph is holding nothing");
    }

    #[tokio::test]
    async fn the_mount_lives_exactly_as_long_as_the_run() {
        let state = walled();
        let mount = state.mount();
        assert!(mount.is_dir(), "the mount exists after boot: {}", mount.display());
        let cli = Cli { port: Some(0), ..Cli::default() };
        assert_eq!(run(cli, "python3".into(), state, std::future::ready(()), None).await, 0);
        let husk = mount.parent().expect("the mount is nested under a nonce dir");
        assert!(!husk.exists(), "the nonce directory goes too, not just workspace: {}", husk.display());

        // `--list-nodes` returns before the server ever binds; the same tail must still reclaim.
        let listed = walled();
        let m2 = listed.mount();
        let cli = Cli { list_nodes: true, ..Cli::default() };
        assert_eq!(run(cli, "python3".into(), listed, std::future::pending(), None).await, 0);
        assert!(!m2.exists(), "--list-nodes reclaims too: {}", m2.display());
    }

    #[test]
    fn help_is_a_mode_the_caller_handles() {
        assert!(parse(&["--help"]).expect("help parses").help);
        assert!(parse(&["-h"]).expect("help parses").help);
    }
}
