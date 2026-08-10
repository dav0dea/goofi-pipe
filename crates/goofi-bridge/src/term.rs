//! The harness plane: one PTY per spawned agent harness, and the MCP address goofi minted for it.
//!
//! **The address is the design's centre.** A spawns one MCP server at `/mcp`; a spawn here mints
//! `/mcp/<instance_id>` and writes THAT url into the config it hands its harness. Nothing about the
//! identity travels through the agent, so there is no id to spoof and none to validate; the route
//! either exists or it does not, and [`Harnesses::stop`] is what makes it not. A path rather than a
//! port was deliberate — a port per harness buys network isolation this single-user local app has
//! no use for, at the cost of a listener, an accept loop and real OS lifecycle.
//!
//! **The environment is inherited whole**, so the harness's own login and auth work and its
//! sessions land where it expects. Only the terminal contract is overlaid, because a dumb `TERM` or
//! a non-UTF-8 locale makes a TUI refuse to start or render mojibake. Nothing redirects `HOME`:
//! the harness writes its state there rather than into the cwd, so credentials never land in the
//! workspace — secret hygiene by construction rather than by a rule.
//!
//! **Nothing here emulates a terminal** (user, 2026-08-10). There is no grid, no scrollback and no
//! replay: the client keeps its own `Terminal` object alive across a panel close, and a re-attach
//! nudges the size so a full-screen TUI repaints itself. History across a page reload is allowed to
//! be lost; that is what buys this module its size.

use std::ffi::OsStr;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use portable_pty::{native_pty_system, CommandBuilder, MasterPty, PtySize};
use serde_json::{json, Value};
use tokio::sync::{broadcast, watch};

/// How long a stopped harness has to leave on its own before it is killed outright.
const GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// How many PTY chunks a `/term` socket may fall behind before it starts losing bytes. Losing them
/// only garbles the screen until the next repaint, which is the trade the no-emulator decision
/// already made; blocking the reader instead would stall the child itself.
const OUTPUT_BACKLOG: usize = 1024;

/// One harness goofi knows how to launch, and how to point it at an MCP address. `file` — written
/// under `harness/<instance_id>/` — plus `args` and `env` are the whole adapter: `{url}` expands to
/// the minted address and `{file}` to that file's path, so a harness that changes its config
/// mechanism is a one-row edit. A `_`-prefixed name is a TEST adapter, hidden from detection
/// exactly as the node catalog hides its `_` types.
struct Adapter {
    name: &'static str,
    bin: &'static str,
    file: Option<(&'static str, &'static str)>,
    args: &'static [&'static str],
    env: &'static [(&'static str, &'static str)],
}

static ADAPTERS: &[Adapter] = &[
    // Claude Code loads MCP servers from the JSON config files named on its command line.
    Adapter { name: "claude", bin: "claude", args: &["--mcp-config", "{file}"], env: &[],
              file: Some(("mcp.json", r#"{"mcpServers":{"goofi":{"type":"http","url":"{url}"}}}"#)) },
    // Codex has no per-invocation config FILE — `CODEX_HOME` would move its credentials with it —
    // but `-c` overrides one dotted key whose value is parsed as TOML, which is exactly the shape
    // `codex mcp add goofi --url …` would otherwise have written into `config.toml`.
    Adapter { name: "codex", bin: "codex", args: &["-c", "mcp_servers.goofi.url=\"{url}\""],
              env: &[], file: None },
    // opencode reads the config file `$OPENCODE_CONFIG` names.
    Adapter { name: "opencode", bin: "opencode", args: &[], env: &[("OPENCODE_CONFIG", "{file}")],
              file: Some(("opencode.json",
                          r#"{"mcp":{"goofi":{"type":"remote","url":"{url}","enabled":true}}}"#)) },
    // The test adapter: a plain shell, so the PTY, the roster, the reaper and the minted address
    // are drivable on a machine with no harness installed. It writes the same config file a real
    // adapter does (and ignores it), so the minting path under test is the shipping one.
    Adapter { name: "_sh", bin: "sh", args: &[], env: &[],
              file: Some(("mcp.json", r#"{"mcpServers":{"goofi":{"type":"http","url":"{url}"}}}"#)) },
    // And one that REPORTS the SIGTERM and then refuses to leave, so a stop can be watched asking
    // before it insists. It says `armed` on EVERY pass, not once: nothing here replays what a
    // harness wrote before a socket attached, and a signal delivered before the trap was installed
    // would prove nothing. The loop matters too — a bare `sleep` is a child of the same group and
    // would die of the group signal, taking the shell with it when it returned.
    Adapter { name: "_deaf", bin: "sh", env: &[], file: None,
              args: &["-c", "trap 'echo GOT-TERM' TERM; while :; do echo armed; sleep 0.2; done"] },
];

/// One installed harness binary: where it is, and what it calls itself.
struct Detected {
    name: &'static str,
    path: PathBuf,
    /// The binary's size+mtime — what `version` is cached under, so a re-detect re-probes only a
    /// harness that was actually updated.
    stamp: Option<crate::Stamp>,
    version: Option<String>,
}

/// The spawned harnesses and the detection cache — the whole harness plane's state.
#[derive(Default)]
pub struct Harnesses {
    /// In spawn order, which is the order the panel's switcher reads.
    instances: Mutex<Vec<(String, Arc<Instance>)>>,
    detected: Mutex<Vec<Detected>>,
}

impl Harnesses {
    /// Re-detect off the request path, announcing the result when it lands. Detection runs a
    /// `--version` per binary and a login shell whenever a bare PATH lookup misses — seconds of
    /// process spawning that must not sit on a socket's request loop, and that the snapshot must
    /// never pay for on a reconnect. So the roster answers from the cache and converges through
    /// `harness_changed`, exactly as an exit does.
    pub fn refresh_in_background(self: &Arc<Self>, events: broadcast::Sender<String>) {
        let harnesses = self.clone();
        std::thread::spawn(move || {
            harnesses.refresh_detected();
            let _ = events.send(crate::event("harness_changed", harnesses.roster()));
        });
    }

    /// Re-resolve every harness binary, re-probing only the versions whose binary changed.
    fn refresh_detected(&self) {
        let path = std::env::var_os("PATH");
        let shell = login_shell();
        let mut found = Vec::new();
        for a in ADAPTERS.iter().filter(|a| !a.name.starts_with('_')) {
            let Some(bin) = resolve(a.bin, path.as_deref(), &shell) else { continue };
            let stamp = stamp(&bin);
            let cache = self.detected.lock().unwrap();
            let cached = cache
                .iter()
                .find(|d| d.name == a.name && d.path == bin && d.stamp == stamp)
                .and_then(|d| d.version.clone());
            drop(cache);
            let version = cached.or_else(|| probe_version(&bin));
            found.push(Detected { name: a.name, path: bin, stamp, version });
        }
        *self.detected.lock().unwrap() = found;
    }

    /// The roster the snapshot seeds and `harness_changed` broadcasts — ONE shape, for the reason
    /// the `runtime` overlay exists: the live stream carries only transitions, so a tab that joins
    /// after a spawn would otherwise draw an empty switcher over a running harness.
    pub fn roster(&self) -> Value {
        let instances: Vec<Value> = self.instances.lock().unwrap().iter()
            .map(|(id, i)| {
                let exit = i.exit_code();
                json!({
                    "id": id,
                    "harness": i.harness,
                    "state": if exit.is_some() { "exited" } else { "running" },
                    "exit_code": exit,
                })
            })
            .collect();
        let detected: Vec<Value> = self.detected.lock().unwrap().iter()
            .map(|d| json!({ "harness": d.name, "path": d.path.to_string_lossy(), "version": d.version }))
            .collect();
        json!({ "instances": instances, "detected": detected })
    }

    /// The instance behind a `/term` or `/mcp` path, if it is still on the roster.
    pub fn get(&self, id: &str) -> Option<Arc<Instance>> {
        self.instances.lock().unwrap().iter().find(|(k, _)| k == id).map(|(_, i)| i.clone())
    }

    /// Whether `/mcp/<id>` still serves. False for an instance that is unknown, stopping or gone —
    /// a stop closes the address BEFORE it signals the child, so the last window in which a tool
    /// call could land on a harness that is already leaving is closed by construction.
    pub fn serves_mcp(&self, id: &str) -> bool {
        self.get(id).is_some_and(|i| !i.stopping.load(Ordering::Relaxed) && i.exit_code().is_none())
    }

    /// Launch `harness` on a PTY with the patch workspace as its cwd, minting the MCP address it is
    /// handed. `events` is the broadcast the reaper announces the exit on; the caller announces the
    /// spawn, so one `harness_changed` follows each.
    pub fn spawn(
        self: &Arc<Self>,
        harness: &str,
        cwd: &Path,
        base_url: &str,
        events: broadcast::Sender<String>,
    ) -> Result<String, String> {
        let adapter = ADAPTERS.iter().find(|a| a.name == harness)
            .ok_or_else(|| format!("unknown harness `{harness}`"))?;
        let bin = resolve(adapter.bin, std::env::var_os("PATH").as_deref(), &login_shell())
            .ok_or_else(|| format!("`{}` is not installed", adapter.bin))?;
        let id = crate::nonce_hex()[..12].to_string();

        let dir = config_dir(cwd, &id);
        std::fs::create_dir_all(&dir).map_err(|e| format!("harness config directory: {e}"))?;
        let url = format!("{base_url}/mcp/{id}");
        let mut file = String::new();
        if let Some((name, body)) = adapter.file {
            let at = dir.join(name);
            std::fs::write(&at, body.replace("{url}", &url))
                .map_err(|e| format!("harness config: {e}"))?;
            file = at.to_string_lossy().into_owned();
        }
        let expand = |s: &str| s.replace("{url}", &url).replace("{file}", &file);

        let pty = native_pty_system().openpty(PtySize::default()).map_err(|e| e.to_string())?;
        let mut cmd = CommandBuilder::new(&bin);
        for a in adapter.args {
            cmd.arg(expand(a));
        }
        cmd.cwd(cwd);
        // `CommandBuilder` starts from the parent environment, so this OVERLAYS rather than
        // replaces — see the module note on why that is the contract and not an oversight.
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLORTERM", "truecolor");
        if !parent_speaks_utf8() {
            cmd.env("LC_ALL", "C.UTF-8");
        }
        for (k, v) in adapter.env {
            cmd.env(k, expand(v));
        }
        let child = pty.slave.spawn_command(cmd).map_err(|e| format!("spawn {harness}: {e}"))?;
        // The slave end is closed here, or the master would never see EOF when the child exits and
        // the drain below would block forever on a PTY nothing writes to.
        drop(pty.slave);
        let mut reader = pty.master.try_clone_reader().map_err(|e| e.to_string())?;
        let writer = pty.master.take_writer().map_err(|e| e.to_string())?;
        let (output, _) = broadcast::channel(OUTPUT_BACKLOG);
        let (exit, _) = watch::channel(None);
        let inst = Arc::new(Instance {
            harness: harness.to_string(),
            pid: child.process_id(),
            writer: Mutex::new(writer),
            master: Mutex::new(pty.master),
            output: output.clone(),
            exit,
            stopping: AtomicBool::new(false),
        });

        // Drain the PTY unconditionally: a child whose output nobody reads eventually blocks on a
        // full buffer, whether or not a panel is open. `send` failing means no socket is attached,
        // which is the normal state, so it is discarded rather than treated as an end.
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let _ = output.send(buf[..n].to_vec());
            }
        });

        // Registered BEFORE the reaper starts, or a child that dies instantly would announce a
        // roster this instance is not yet on.
        self.instances.lock().unwrap().push((id.clone(), inst.clone()));
        let harnesses = self.clone();
        std::thread::spawn(move || {
            let mut child = child;
            let code = child.wait().map(|s| s.exit_code()).unwrap_or(1);
            inst.exit.send_replace(Some(code));
            let _ = events.send(crate::event("harness_changed", harnesses.roster()));
        });
        Ok(id)
    }

    /// Stop a running instance — or DISMISS an exited one, which is the same button pressed twice
    /// and so needs no second op. Returns as soon as the child has been signalled: `dispatch` is
    /// synchronous, and a grace the caller waits out is not a grace. The reaper records the code.
    pub fn stop(&self, id: &str) -> Result<(), String> {
        let inst = self.get(id).ok_or_else(|| format!("no harness instance `{id}`"))?;
        if inst.exit_code().is_some() {
            self.instances.lock().unwrap().retain(|(k, _)| k != id);
            return Ok(());
        }
        // The address closes FIRST, so an in-flight tool call finishes while the next one is
        // refused — the child is only signalled after.
        inst.stopping.store(true, Ordering::Relaxed);
        signal(&inst, libc::SIGTERM)?;
        std::thread::spawn(move || {
            std::thread::sleep(GRACE);
            let _ = signal(&inst, libc::SIGKILL);
        });
        Ok(())
    }
}

/// One spawned harness: its PTY, its exit, and whether its MCP address is still open.
pub struct Instance {
    harness: String,
    pid: Option<u32>,
    /// Writing can block if the child has stopped reading and the PTY's input buffer fills. Bounded
    /// in practice by what a person types or pastes into one terminal, so it is taken inline rather
    /// than through the `/data` plane's non-parking send.
    writer: Mutex<Box<dyn Write + Send>>,
    master: Mutex<Box<dyn MasterPty + Send>>,
    output: broadcast::Sender<Vec<u8>>,
    /// The child's exit code once reaped — and the channel a `/term` socket waits on, so the exit
    /// frame is pushed to an attached viewer rather than polled for.
    exit: watch::Sender<Option<u32>>,
    /// Set the moment a stop begins, ahead of the signal.
    stopping: AtomicBool,
}

impl Instance {
    /// The PTY's output, and the exit that ends it.
    pub fn attach(&self) -> (broadcast::Receiver<Vec<u8>>, watch::Receiver<Option<u32>>) {
        (self.output.subscribe(), self.exit.subscribe())
    }

    /// Keystrokes (or a paste) from an attached `/term` socket.
    pub fn write(&self, bytes: &[u8]) {
        let mut w = self.writer.lock().unwrap();
        let _ = w.write_all(bytes).and_then(|()| w.flush());
    }

    /// Tell the kernel the window changed, which is what raises the child's SIGWINCH.
    pub fn resize(&self, cols: u16, rows: u16) {
        let size = PtySize { rows, cols, pixel_width: 0, pixel_height: 0 };
        let _ = self.master.lock().unwrap().resize(size);
    }

    pub fn exit_code(&self) -> Option<u32> {
        *self.exit.borrow()
    }
}

/// Where one instance's config is written: BESIDE the workspace, not inside it. Inside would pack a
/// stale URL into every `.gfi` the patch is ever saved to, and would DIRTY the patch merely for
/// launching an agent — the workspace fingerprint has no exclusion list, and growing one to buy this
/// would be worse than putting a file goofi owns where goofi already owns the ground. It is
/// reclaimed with the mount, since `release_mount` takes the nonce directory both live under.
pub fn config_dir(mount: &Path, id: &str) -> PathBuf {
    mount.parent().unwrap_or(mount).join("harness").join(id)
}

/// Signal the instance's process GROUP: portable-pty makes each child a session leader, so its pid
/// is its group id and one signal reaches everything the harness itself spawned. Skipped once the
/// child has been reaped, since a recycled pid would name a stranger's group.
fn signal(inst: &Instance, sig: i32) -> Result<(), String> {
    if inst.exit_code().is_some() {
        return Ok(());
    }
    let pid = inst.pid.ok_or("the harness reported no pid to signal")?;
    // SAFETY: a plain syscall whose only fallible argument is the pid, and that pid names a child
    // this process spawned and has not yet reaped.
    if unsafe { libc::kill(-(pid as i32), sig) } == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    // ESRCH means it left between the check above and the syscall — which is what was asked for.
    match err.raw_os_error() {
        Some(libc::ESRCH) => Ok(()),
        _ => Err(err.to_string()),
    }
}

/// Resolve `bin` to an executable path: a plain `PATH` walk first, then — only when that misses —
/// a LOGIN shell's own lookup. The fallback is not hypothetical: a desktop-launched process
/// inherits none of nvm's shims, and that already cost this repo's own build script an exit 127.
/// `path` and `shell` are parameters rather than reads of the ambient environment so a test can
/// drive the fallback without mutating it — cargo runs the suite as threads in ONE process.
fn resolve(bin: &str, path: Option<&OsStr>, shell: &str) -> Option<PathBuf> {
    let direct =
        path.into_iter().flat_map(std::env::split_paths).map(|d| d.join(bin)).find(|c| is_executable(c));
    if direct.is_some() {
        return direct;
    }
    // `bin` is a literal from `ADAPTERS`, never a caller's string, so there is nothing here for a
    // shell metacharacter to escape into.
    let out = std::process::Command::new(shell).args(["-lc", &format!("command -v {bin}")]).output();
    let found = PathBuf::from(String::from_utf8_lossy(&out.ok()?.stdout).trim());
    is_executable(&found).then_some(found)
}

fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

fn stamp(p: &Path) -> Option<crate::Stamp> {
    let m = std::fs::metadata(p).ok()?;
    Some((m.len(), m.modified().ok()?))
}

/// What a harness calls itself. Best-effort and untimed: it runs off the graph lock, and a binary
/// that answers nothing is still listed — being on PATH is the fact, the version is a nicety.
fn probe_version(bin: &Path) -> Option<String> {
    let out = std::process::Command::new(bin).arg("--version").output().ok()?;
    let line = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
    (!line.is_empty()).then_some(line)
}

fn login_shell() -> String {
    std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into())
}

/// Whether the parent already has a UTF-8 locale, in the precedence the C library resolves them in.
/// Only when it has none is one imposed — overriding a deliberate non-UTF-8 choice would be worse
/// than the mojibake it prevents.
fn parent_speaks_utf8() -> bool {
    ["LC_ALL", "LC_CTYPE", "LANG"].iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
        .is_some_and(|v| v.to_ascii_uppercase().replace('-', "").contains("UTF8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PATH fallback, both halves. A binary the bare walk misses is still found through a login
    /// shell — that is the nvm case the module note names — and a name no shell can resolve stays
    /// unresolved, so the fallback is a lookup rather than a rubber stamp.
    #[test]
    fn a_binary_the_bare_path_lookup_misses_is_found_through_a_login_shell() {
        let nothing = OsStr::new("/goofi-no-such-directory");
        let found = resolve("sh", Some(nothing), "/bin/sh").expect("a login shell resolves `sh`");
        assert!(is_executable(&found) && found.ends_with("sh"), "{found:?}");
        assert_eq!(resolve("goofi-not-a-real-binary", Some(nothing), "/bin/sh"), None);
        // …and the direct walk is what answers when PATH does contain it: the shell named here
        // cannot resolve anything, so a hit proves the first branch rather than the second.
        let dir = found.parent().expect("an absolute path").as_os_str().to_owned();
        assert_eq!(resolve("sh", Some(&dir), "/bin/false").as_deref(), Some(found.as_path()));
    }
}
