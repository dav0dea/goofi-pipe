//! The harness plane: one PTY per spawned agent harness, and the MCP address goofi minted for it.
//!
//! The environment is inherited WHOLE, so the harness's own login and auth work; only the terminal
//! contract is overlaid. Nothing here emulates a terminal, so history is lost on a page reload.

use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use portable_pty::{native_pty_system, CommandBuilder, MasterPty, PtySize};
use serde_json::{json, Value};
use tokio::sync::{broadcast, watch};

/// How long a stopped harness has to leave on its own before it is killed outright.
const GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// How many PTY chunks a `/term` socket may fall behind before it loses bytes; blocking the reader
/// instead would stall the child.
const OUTPUT_BACKLOG: usize = 1024;

/// One harness goofi knows how to launch: `{url}` expands to the minted address and `{file}` to
/// the written config's path. A `_`-prefixed name is a TEST adapter, hidden from detection.
struct Adapter {
    name: &'static str,
    bin: &'static str,
    file: Option<(&'static str, &'static str)>,
    args: &'static [&'static str],
    env: &'static [(&'static str, &'static str)],
}

static ADAPTERS: &[Adapter] = &[
    Adapter { name: "claude", bin: "claude", args: &["--mcp-config", "{file}"], env: &[],
              file: Some(("mcp.json", r#"{"mcpServers":{"goofi":{"type":"http","url":"{url}"}}}"#)) },
    // Codex has no per-invocation config FILE: `-c` overrides one dotted key, parsed as TOML.
    Adapter { name: "codex", bin: "codex", args: &["-c", "mcp_servers.goofi.url=\"{url}\""],
              env: &[], file: None },
    Adapter { name: "opencode", bin: "opencode", args: &[], env: &[("OPENCODE_CONFIG", "{file}")],
              file: Some(("opencode.json",
                          r#"{"mcp":{"goofi":{"type":"remote","url":"{url}","enabled":true}}}"#)) },
    // The test adapter: a plain shell, writing the same config a real adapter does.
    Adapter { name: "_sh", bin: "sh", args: &[], env: &[],
              file: Some(("mcp.json", r#"{"mcpServers":{"goofi":{"type":"http","url":"{url}"}}}"#)) },
    // One that reports the SIGTERM and refuses to leave. The loop matters: a bare `sleep` is a
    // child of the same group and would die of the group signal.
    Adapter { name: "_deaf", bin: "sh", env: &[], file: None,
              args: &["-c", "trap 'echo GOT-TERM' TERM; while :; do echo armed; sleep 0.2; done"] },
];

/// One installed harness binary: where it is, and what it calls itself.
struct Detected {
    name: &'static str,
    path: PathBuf,
    /// The binary's size+mtime, so a re-detect re-probes only a harness that was updated.
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
    /// Re-detect OFF the request path: detection can spawn for seconds. The roster answers from the
    /// cache and converges by broadcast.
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

    /// The roster the snapshot seeds and `harness_changed` broadcasts — one shape for both.
    pub fn roster(&self) -> Value {
        let instances: Vec<Value> = self.instances.lock().unwrap().iter()
            .map(|(id, i)| {
                let exit = i.exit_code();
                json!({
                    "id": id,
                    "harness": i.harness,
                    // A stop asked for but not yet obeyed is its OWN state, for the whole grace.
                    "state": match (exit, i.stopping.load(Ordering::Relaxed)) {
                        (Some(_), _) => "exited",
                        (None, true) => "stopping",
                        (None, false) => "running",
                    },
                    "exit_code": exit,
                })
            })
            .collect();
        let detected: Vec<Value> = self.detected.lock().unwrap().iter()
            .map(|d| {
                json!({
                    "harness": d.name,
                    "path": goofi_core::path::to_slash(&d.path),
                    "version": d.version,
                })
            })
            .collect();
        json!({ "instances": instances, "detected": detected })
    }

    /// The instance behind a `/term` or `/mcp` path, if it is still on the roster.
    pub fn get(&self, id: &str) -> Option<Arc<Instance>> {
        self.instances.lock().unwrap().iter().find(|(k, _)| k == id).map(|(_, i)| i.clone())
    }

    /// Whether `/mcp/<id>` still serves; false for an instance that is unknown, stopping or gone.
    pub fn serves_mcp(&self, id: &str) -> bool {
        self.get(id).is_some_and(|i| !i.stopping.load(Ordering::Relaxed) && i.exit_code().is_none())
    }

    /// Launch `harness` on a PTY with the patch workspace as its cwd, minting the MCP address it is
    /// handed. `env` is the parent environment it inherits; the reaper announces the exit on
    /// `events`, and the caller announces the spawn.
    pub fn spawn(
        self: &Arc<Self>,
        harness: &str,
        cwd: &Path,
        base_url: &str,
        env: &[(OsString, OsString)],
        events: broadcast::Sender<String>,
    ) -> Result<String, String> {
        let adapter = ADAPTERS.iter().find(|a| a.name == harness).ok_or_else(|| {
            // The `_`-prefixed test adapters are spawnable but never advertised.
            let have: Vec<&str> =
                ADAPTERS.iter().map(|a| a.name).filter(|n| !n.starts_with('_')).collect();
            format!("unknown harness `{harness}` — this build knows: {}", have.join(", "))
        })?;
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
        // The parent environment, then the terminal contract ON TOP of it.
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLORTERM", "truecolor");
        if !parent_speaks_utf8(env) {
            cmd.env("LC_ALL", "C.UTF-8");
        }
        for (k, v) in adapter.env {
            cmd.env(k, expand(v));
        }
        let child = pty.slave.spawn_command(cmd).map_err(|e| format!("spawn {harness}: {e}"))?;
        // Closed here, or the master never sees EOF when the child exits and the drain blocks.
        drop(pty.slave);
        let mut reader = pty.master.try_clone_reader().map_err(|e| e.to_string())?;
        let writer = pty.master.take_writer().map_err(|e| e.to_string())?;
        let (output, _) = broadcast::channel(OUTPUT_BACKLOG);
        let (exit, _) = watch::channel(None);
        let (eof, _) = watch::channel(false);
        let ended = eof.clone();
        let inst = Arc::new(Instance {
            harness: harness.to_string(),
            pid: child.process_id(),
            writer: Mutex::new(writer),
            master: Mutex::new(pty.master),
            output: output.clone(),
            exit,
            eof,
            stopping: AtomicBool::new(false),
            sizes: Mutex::default(),
            size: watch::channel(None).0,
            seats: AtomicU64::new(0),
        });

        // Drained unconditionally: a child whose output nobody reads blocks on a full buffer. A
        // failed `send` only means no socket is attached, which is the normal state.
        let answering = inst.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let _ = output.send(answer_cursor_query(&answering, &buf[..n]));
            }
            // AFTER the final send, so a socket that waits on this was offered every byte.
            ended.send_replace(true);
        });

        // BEFORE the reaper starts, or a child that dies instantly announces a roster this
        // instance is not yet on.
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

    /// Stop a running instance, or dismiss an exited one. It returns as soon as the child is
    /// signalled, because `dispatch` is synchronous and a grace the caller waits out is not one.
    pub fn stop(&self, id: &str) -> Result<(), String> {
        let inst = self.get(id).ok_or_else(|| format!("no harness instance `{id}`"))?;
        if inst.exit_code().is_some() {
            self.instances.lock().unwrap().retain(|(k, _)| k != id);
            return Ok(());
        }
        begin_stop(inst)
    }

    /// Stop every instance and clear the roster — what opening another patch does. The roster is
    /// cleared at once, so a doomed child lives briefly on a deleted cwd rather than holding it.
    pub fn stop_all(&self) {
        for (_, inst) in std::mem::take(&mut *self.instances.lock().unwrap()) {
            if inst.exit_code().is_none() {
                let _ = begin_stop(inst);
            }
        }
    }
}

/// Ask a running instance to leave, and insist after the grace. The address closes FIRST, so an
/// in-flight tool call finishes while the next is refused.
fn begin_stop(inst: Arc<Instance>) -> Result<(), String> {
    inst.stopping.store(true, Ordering::Relaxed);
    signal(&inst, crate::proc::request_stop)?;
    std::thread::spawn(move || {
        std::thread::sleep(GRACE);
        let _ = signal(&inst, crate::proc::force_kill);
    });
    Ok(())
}

/// The terminal size every view of one instance shares: a PTY has ONE window, so the last view to
/// speak wins, and a view that retracts or leaves hands it back to the newest survivor.
#[derive(Default)]
struct Sizes {
    seats: Vec<(u64, Option<(u16, u16)>)>,
}

impl Sizes {
    /// `size` is `None` for a retraction. The seat moves to the end, so "last writer" means the
    /// last to have SPOKEN rather than the last to have arrived.
    fn propose(&mut self, seat: u64, size: Option<(u16, u16)>) -> Option<(u16, u16)> {
        self.seats.retain(|(s, _)| *s != seat);
        self.seats.push((seat, size));
        self.current()
    }

    fn leave(&mut self, seat: u64) -> Option<(u16, u16)> {
        self.seats.retain(|(s, _)| *s != seat);
        self.current()
    }

    fn current(&self) -> Option<(u16, u16)> {
        self.seats.iter().rev().find_map(|(_, s)| *s)
    }
}

/// One spawned harness: its PTY, its exit, and whether its MCP address is still open.
pub struct Instance {
    harness: String,
    pid: Option<u32>,
    /// Writing can block when the child stops reading, but it is bounded by what a person types,
    /// so it is taken inline.
    writer: Mutex<Box<dyn Write + Send>>,
    master: Mutex<Box<dyn MasterPty + Send>>,
    output: broadcast::Sender<Vec<u8>>,
    /// The child's exit code once reaped, and the channel a `/term` socket waits on.
    exit: watch::Sender<Option<u32>>,
    /// End-of-stream, which is NOT the exit: `child.wait()` returns while the words the child wrote
    /// on its way out are still in flight.
    eof: watch::Sender<bool>,
    /// Set the moment a stop begins, ahead of the signal.
    stopping: AtomicBool,
    sizes: Mutex<Sizes>,
    size: watch::Sender<Option<(u16, u16)>>,
    seats: AtomicU64,
}

impl Instance {
    /// The PTY's output, the exit code, and the end-of-stream that says the output is complete.
    pub fn attach(
        &self,
    ) -> (broadcast::Receiver<Vec<u8>>, watch::Receiver<Option<u32>>, watch::Receiver<bool>) {
        (self.output.subscribe(), self.exit.subscribe(), self.eof.subscribe())
    }

    /// Take a seat number in the size arbitration, and read the answer as it changes; the seat is
    /// materialised by its first proposal.
    pub fn join(&self) -> (u64, watch::Receiver<Option<(u16, u16)>>) {
        (self.seats.fetch_add(1, Ordering::Relaxed), self.size.subscribe())
    }

    /// This view's word on the size, `None` when it has nothing on screen. The lock is held across
    /// the settle, so two views resizing at once cannot land out of order.
    pub fn propose(&self, seat: u64, size: Option<(u16, u16)>) {
        let mut sizes = self.sizes.lock().unwrap();
        self.settle(sizes.propose(seat, size));
    }

    /// This view is gone; the terminal goes back to whichever survivor spoke last.
    pub fn leave(&self, seat: u64) {
        let mut sizes = self.sizes.lock().unwrap();
        self.settle(sizes.leave(seat));
    }

    /// Apply an arbitrated answer once, the kernel first; a `None` leaves the window where it was.
    fn settle(&self, now: Option<(u16, u16)>) {
        let Some((cols, rows)) = now else { return };
        if *self.size.borrow() == now {
            return;
        }
        let size = PtySize { rows, cols, pixel_width: 0, pixel_height: 0 };
        let _ = self.master.lock().unwrap().resize(size);
        self.size.send_replace(now);
    }

    /// Keystrokes (or a paste) from an attached `/term` socket.
    pub fn write(&self, bytes: &[u8]) {
        let mut w = self.writer.lock().unwrap();
        let _ = w.write_all(bytes).and_then(|()| w.flush());
    }

    pub fn exit_code(&self) -> Option<u32> {
        *self.exit.borrow()
    }
}

/// The whole of an agent's orientation — the very bytes [`seed_orientation`] lays in a workspace.
pub(crate) const ORIENTATION: &str = include_str!("orientation.md");

/// Seed a NEW workspace with the orientation an agent reads, plus the packaging ignore list —
/// absent-only, and never into a workspace a `.gfi` was unpacked into, whose files are the patch's.
pub fn seed_orientation(mount: &Path) {
    for (name, body) in [
        ("AGENTS.md", ORIENTATION),
        ("CLAUDE.md", "@AGENTS.md\n"),
        (goofi_engine::archive::IGNORE_FILE, goofi_engine::archive::DEFAULT_IGNORE),
    ] {
        let at = mount.join(name);
        if !at.exists() {
            let _ = std::fs::write(at, body);
        }
    }
}

/// Where one instance's config is written: BESIDE the workspace, since inside would pack a stale
/// URL into every `.gfi` and dirty the patch merely for launching an agent.
pub fn config_dir(mount: &Path, id: &str) -> PathBuf {
    mount.parent().unwrap_or(mount).join("harness").join(id)
}

/// Answer ConPTY's cursor-position query, which BLOCKS the child until something replies — but only
/// while no viewer is attached, since xterm.js gives the real position and a second reply is typed
/// input.
fn answer_cursor_query(inst: &Instance, bytes: &[u8]) -> Vec<u8> {
    if inst.output.receiver_count() > 0 {
        return bytes.to_vec();
    }
    let Some(stripped) = take_cursor_queries(bytes) else { return bytes.to_vec() };
    // Row 1, column 1 — a lie, told only when there is no screen to contradict it.
    if let Ok(mut w) = inst.writer.lock() {
        let _ = w.write_all(b"\x1b[1;1R");
        let _ = w.flush();
    }
    stripped
}

/// The bytes ConPTY sends to ask where the cursor is.
const CURSOR_QUERY: &[u8] = b"\x1b[6n";

/// `bytes` with every [`CURSOR_QUERY`] taken out, or `None` when there was none.
fn take_cursor_queries(bytes: &[u8]) -> Option<Vec<u8>> {
    if !bytes.windows(CURSOR_QUERY.len()).any(|w| w == CURSOR_QUERY) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len());
    let mut rest = bytes;
    while let Some(at) = rest.windows(CURSOR_QUERY.len()).position(|w| w == CURSOR_QUERY) {
        out.extend_from_slice(&rest[..at]);
        rest = &rest[at + CURSOR_QUERY.len()..];
    }
    out.extend_from_slice(rest);
    Some(out)
}

/// Reach the instance with one of [`crate::proc`]'s two asks, skipped once the child has been
/// reaped, since a recycled pid would name a stranger.
fn signal(inst: &Instance, how: fn(u32) -> Result<(), String>) -> Result<(), String> {
    if inst.exit_code().is_some() {
        return Ok(());
    }
    how(inst.pid.ok_or("the harness reported no pid to signal")?)
}

/// Resolve `bin`: a plain `PATH` walk, then — only on a miss — a LOGIN shell's lookup, because a
/// desktop-launched process inherits none of nvm's shims. `which_in_global` searches `path` ALONE,
/// so nothing in the patch workspace can shadow the harness.
fn resolve(bin: &str, path: Option<&OsStr>, shell: &str) -> Option<PathBuf> {
    let found = |name: &OsStr| which::which_in_global(name, path).ok()?.next();
    if let Some(direct) = found(OsStr::new(bin)) {
        return Some(direct);
    }
    // `bin` is an `ADAPTERS` literal, never a caller's string, so nothing can escape the shell.
    let out = std::process::Command::new(shell).args(["-lc", &format!("command -v {bin}")]).output();
    found(OsStr::new(answer(&String::from_utf8_lossy(&out.ok()?.stdout))?))
}

/// What a login shell actually answered, out of everything its profile said first: the LAST
/// non-empty line.
fn answer(stdout: &str) -> Option<&str> {
    stdout.lines().rev().find(|l| !l.trim().is_empty()).map(str::trim)
}

fn stamp(p: &Path) -> Option<crate::Stamp> {
    let m = std::fs::metadata(p).ok()?;
    Some((m.len(), m.modified().ok()?))
}

/// What a harness calls itself; best-effort, and a binary that answers nothing is still listed.
fn probe_version(bin: &Path) -> Option<String> {
    let out = std::process::Command::new(bin).arg("--version").output().ok()?;
    let line = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
    (!line.is_empty()).then_some(line)
}

fn login_shell() -> String {
    std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into())
}

/// The environment a spawned harness inherits: goofi's own, whole.
pub fn parent_env() -> Vec<(OsString, OsString)> {
    std::env::vars_os().collect()
}

/// Whether what the child will SEE names a UTF-8 locale, in the C library's precedence.
fn parent_speaks_utf8(env: &[(OsString, OsString)]) -> bool {
    ["LC_ALL", "LC_CTYPE", "LANG"].iter()
        .find_map(|k| {
            env.iter()
                .find(|(n, v)| n.as_os_str() == OsStr::new(k) && !v.is_empty())
                .map(|(_, v)| v.to_string_lossy().into_owned())
                .or_else(|| std::env::var(k).ok().filter(|v| !v.is_empty()))
        })
        .is_some_and(|v| v.to_ascii_uppercase().replace('-', "").contains("UTF8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new workspace is seeded with the WHOLE of [`ORIENTATION`], under both names.
    #[test]
    fn a_new_workspace_is_seeded_with_the_whole_orientation() {
        let tmp = tempfile::tempdir().expect("a temp dir");
        seed_orientation(tmp.path());
        assert_eq!(std::fs::read_to_string(tmp.path().join("AGENTS.md")).unwrap(), ORIENTATION);
        // Claude Code reads CLAUDE.md, and its `@` import is what points it at the other file.
        assert_eq!(std::fs::read_to_string(tmp.path().join("CLAUDE.md")).unwrap(), "@AGENTS.md\n");
        for section in ["\n## Seeing", "\n## Building", "\n## Custom Python nodes"] {
            assert!(ORIENTATION.contains(section), "the orientation has no `{section}` section");
        }
        // Read in EVERY turn of every agent in this patch, so the aim is ~6 KB.
        assert!(ORIENTATION.len() < 8192, "the orientation is {} bytes", ORIENTATION.len());
    }

    /// …and an orientation the agent has already edited is ITS OWN.
    #[test]
    fn an_orientation_the_agent_has_edited_is_never_seeded_over() {
        let tmp = tempfile::tempdir().expect("a temp dir");
        let learned = "goofi-pipe patch notes: the EEG source is on channel 3.\n";
        std::fs::write(tmp.path().join("AGENTS.md"), learned).unwrap();
        std::fs::write(tmp.path().join("CLAUDE.md"), "@AGENTS.md\nand a note of its own\n").unwrap();

        seed_orientation(tmp.path());

        assert_eq!(std::fs::read_to_string(tmp.path().join("AGENTS.md")).unwrap(), learned);
        assert!(std::fs::read_to_string(tmp.path().join("CLAUDE.md")).unwrap().contains("its own"));
    }

    /// The ignore list is seeded on the same terms as the orientation: absent-only.
    #[test]
    fn a_new_workspace_is_seeded_with_the_packaging_ignore_list() {
        let tmp = tempfile::tempdir().expect("a temp dir");
        let at = tmp.path().join(goofi_engine::archive::IGNORE_FILE);
        seed_orientation(tmp.path());
        assert_eq!(std::fs::read_to_string(&at).unwrap(), goofi_engine::archive::DEFAULT_IGNORE);

        // …and a list its author has made their own is never seeded over.
        std::fs::write(&at, "*.wav\n").unwrap();
        seed_orientation(tmp.path());
        assert_eq!(std::fs::read_to_string(&at).unwrap(), "*.wav\n");
    }

    /// The half of [`answer_cursor_query`] a test can reach anywhere: only ConPTY sends `\x1b[6n`,
    /// so on unix nothing else exercises the stripping.
    #[test]
    fn a_cursor_query_is_taken_out_of_the_stream_and_everything_else_survives() {
        assert_eq!(take_cursor_queries(b"plain output"), None, "nothing asked, nothing to strip");
        assert_eq!(take_cursor_queries(b"\x1b[6"), None, "half a query is not one");
        assert_eq!(take_cursor_queries(b"\x1b[6n").as_deref(), Some(&b""[..]));
        assert_eq!(take_cursor_queries(b"before\x1b[6nafter").as_deref(), Some(&b"beforeafter"[..]));
        // A `while` downgraded to an `if` passes every case above and fails this one.
        assert_eq!(
            take_cursor_queries(b"\x1b[6nmiddle\x1b[6nend").as_deref(),
            Some(&b"middleend"[..]),
            "every query, not just the first",
        );
    }

    /// A "shell" no platform has, so a resolution that still succeeds proves the direct walk rather
    /// than the fallback.
    const NO_SHELL: &str = "/goofi-no-such-shell";

    /// The direct walk, driven against this test binary by its STEM, as an npm-installed
    /// `claude.cmd` is typed.
    #[test]
    fn a_binary_is_found_by_the_name_a_caller_types_not_the_one_it_has_on_disk() {
        let exe = std::env::current_exe().expect("this test binary's path");
        let dir = exe.parent().expect("an absolute path").as_os_str().to_owned();
        let stem = exe.file_stem().expect("a file name").to_string_lossy().into_owned();

        let found = resolve(&stem, Some(&dir), NO_SHELL).expect("the test binary resolves by stem");

        assert_eq!(
            std::fs::canonicalize(&found).unwrap(),
            std::fs::canonicalize(&exe).unwrap(),
            "resolved {found:?}, wanted {exe:?}"
        );
        assert_eq!(resolve("goofi-not-a-real-binary", Some(&dir), NO_SHELL), None);
    }

    /// The other half of [`resolve`]: a binary the bare walk misses is found through a LOGIN shell.
    #[cfg(unix)]
    #[test]
    fn a_binary_the_bare_path_lookup_misses_is_found_through_a_login_shell() {
        let nothing = OsStr::new("/goofi-no-such-directory");
        let found = resolve("sh", Some(nothing), "/bin/sh").expect("a login shell resolves `sh`");
        assert!(found.ends_with("sh"), "{found:?}");
        assert_eq!(resolve("goofi-not-a-real-binary", Some(nothing), "/bin/sh"), None);
    }

    /// The arbitration, without a socket in the way: last writer wins, and a retraction or a
    /// departure falls back to the newest SURVIVING proposal.
    #[test]
    fn the_last_view_to_speak_owns_the_size_and_hands_it_back_when_it_stops() {
        let mut s = Sizes::default();
        assert_eq!(s.current(), None, "a view that has not measured yet says nothing");
        assert_eq!(s.propose(1, Some((100, 30))), Some((100, 30)));
        assert_eq!(s.propose(2, Some((80, 24))), Some((80, 24)), "the last writer wins");
        assert_eq!(s.propose(2, None), Some((100, 30)), "a retraction hands it to the survivor");
        assert_eq!(s.propose(2, Some((80, 24))), Some((80, 24)), "…and it can speak again");
        assert_eq!(s.leave(2), Some((100, 30)), "so does leaving");
        // A view that speaks twice is not seated twice: the second word REPLACES the first.
        s.propose(1, Some((90, 20)));
        s.propose(1, Some((70, 15)));
        assert_eq!(s.leave(1), None, "the last view out leaves nobody speaking");
    }

    /// …and it understands a login shell whose profile had something to say first.
    #[test]
    fn a_login_shell_that_greets_before_it_answers_is_still_understood() {
        assert_eq!(answer("nvm: Now using node v22\n/usr/bin/claude\n"), Some("/usr/bin/claude"));
        assert_eq!(answer("  /usr/bin/claude  \n"), Some("/usr/bin/claude"));
        assert_eq!(answer("a banner\n\n"), Some("a banner"), "only EMPTY lines are skipped");
        assert_eq!(answer("\n  \n"), None, "a shell that answered nothing resolves nothing");
    }
}
