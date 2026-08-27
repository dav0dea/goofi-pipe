//! The harness plane: one PTY per spawned agent, launched from the config list as a bash
//! command line — no detection, no adapters. A command that cannot launch fails ON its PTY,
//! where every agent already shows its output.
//!
//! The environment is inherited WHOLE, so the agent's own login and auth work; the terminal
//! contract, `GOOFI_SESSION`/`GOOFI_ACTOR` and the `goofi` shim are overlaid. Nothing here
//! emulates a terminal; a bounded tail of output replays on attach, so a command that fails
//! before any viewer arrives — or a page reload — still shows its words.

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

/// How much output an attach replays — enough for a `command not found` and a stack trace.
const TAIL_BYTES: usize = 8192;

/// The spawned agent instances — the whole harness plane's state.
#[derive(Default)]
pub struct Harnesses {
    /// In spawn order, which is the order the panel's switcher reads.
    instances: Mutex<Vec<(String, Arc<Instance>)>>,
}

impl Harnesses {
    /// The roster the snapshot seeds and `harness_changed` broadcasts — one shape for both: the
    /// live instances, and the CONFIG's launchable list, `_`-test entries withheld. `config` is
    /// `home::agents()`, read by the CALLER so the disk read runs off whatever lock it holds.
    pub fn roster(&self, config: &(Vec<goofi_core::home::Agent>, Option<String>)) -> Value {
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
        let (agents, config_error) = config;
        let agents: Vec<Value> = agents
            .iter()
            .filter(|a| !a.name.starts_with('_'))
            .map(|a| json!({ "name": a.name, "command": a.command }))
            .collect();
        json!({ "instances": instances, "agents": agents, "config_error": config_error })
    }

    /// The instance behind a `/term` path, if it is still on the roster.
    pub fn get(&self, id: &str) -> Option<Arc<Instance>> {
        self.instances.lock().unwrap().iter().find(|(k, _)| k == id).map(|(_, i)| i.clone())
    }

    /// Launch the config entry named `agent` on a PTY with the patch workspace as its cwd. The
    /// command runs under a LOGIN shell, so it resolves as the user's own terminal would, and a
    /// command that cannot launch fails on the PTY itself. `env` is the parent environment it
    /// inherits; the reaper announces the exit on `events`, the caller announces the spawn.
    pub fn spawn(
        self: &Arc<Self>,
        agent: &str,
        cwd: &Path,
        session_id: &str,
        env: &[(OsString, OsString)],
        events: broadcast::Sender<String>,
        history: Arc<Mutex<goofi_engine::CommandHistory>>,
    ) -> Result<String, String> {
        let (agents, _) = goofi_core::home::agents();
        let command = agents
            .iter()
            .find(|a| a.name == agent)
            .map(|a| a.command.clone())
            .ok_or_else(|| {
                // The `_`-prefixed test entries are spawnable but never advertised.
                let have: Vec<&str> =
                    agents.iter().map(|a| a.name.as_str()).filter(|n| !n.starts_with('_')).collect();
                match have.is_empty() {
                    true => format!(
                        "unknown agent `{agent}` — the config lists none; add [[agents]] to {}",
                        goofi_core::home::config_file().display()
                    ),
                    false => format!("unknown agent `{agent}` — the config offers: {}", have.join(", ")),
                }
            })?;
        let id = crate::nonce_hex()[..12].to_string();

        let dir = config_dir(cwd, &id);
        std::fs::create_dir_all(&dir).map_err(|e| format!("agent config directory: {e}"))?;
        write_shim(&dir)?;

        let pty = native_pty_system().openpty(PtySize::default()).map_err(|e| e.to_string())?;
        let mut cmd = shell_command(&command);
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
        // How the shell finds ITS server: the id names the session file, the actor its undo
        // stack, and the shim makes `goofi` this server's own binary whatever is installed.
        cmd.env("GOOFI_SESSION", session_id);
        cmd.env("GOOFI_ACTOR", actor_of(&id));
        cmd.env("PATH", prepend_path(&dir, env));
        let child = pty.slave.spawn_command(cmd).map_err(|e| format!("spawn {agent}: {e}"))?;
        // Closed here, or the master never sees EOF when the child exits and the drain blocks.
        drop(pty.slave);
        let mut reader = pty.master.try_clone_reader().map_err(|e| e.to_string())?;
        let writer = pty.master.take_writer().map_err(|e| e.to_string())?;
        let (output, _) = broadcast::channel(OUTPUT_BACKLOG);
        let (exit, _) = watch::channel(None);
        let (eof, _) = watch::channel(false);
        let ended = eof.clone();
        let inst = Arc::new(Instance {
            harness: agent.to_string(),
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
            tail: Mutex::default(),
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
                let bytes = answer_cursor_query(&answering, &buf[..n]);
                // Appended and sent under ONE lock, against `attach`'s snapshot-then-subscribe.
                let mut tail = answering.tail.lock().unwrap();
                tail.extend_from_slice(&bytes);
                let over = tail.len().saturating_sub(TAIL_BYTES);
                if over > 0 {
                    tail.drain(..over);
                }
                let _ = output.send(bytes);
            }
            // AFTER the final send, so a socket that waits on this was offered every byte.
            ended.send_replace(true);
        });

        // BEFORE the reaper starts, or a child that dies instantly announces a roster this
        // instance is not yet on.
        self.instances.lock().unwrap().push((id.clone(), inst.clone()));
        let harnesses = self.clone();
        let reaped = id.clone();
        std::thread::spawn(move || {
            let mut child = child;
            let code = child.wait().map(|s| s.exit_code()).unwrap_or(1);
            // A stack's lifetime follows its actor: dropped where the actor DIES, and BEFORE the
            // exit shows, so an observer that sees `exited` sees the stack gone too.
            history.lock().unwrap().drop_actor(&actor_of(&reaped));
            inst.exit.send_replace(Some(code));
            let _ =
                events.send(crate::event("harness_changed", harnesses.roster(&goofi_core::home::agents())));
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

/// Ask a running instance to leave, and insist after the grace.
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

/// One spawned harness: its PTY, its exit, and the tail an attach replays.
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
    /// The last [`TAIL_BYTES`] the child wrote, replayed to an attach that arrived after them.
    tail: Mutex<Vec<u8>>,
}

/// What an attach hands a `/term` socket: the replayed tail, then the live channels.
pub struct Attached {
    pub tail: Vec<u8>,
    pub output: broadcast::Receiver<Vec<u8>>,
    pub exit: watch::Receiver<Option<u32>>,
    pub eof: watch::Receiver<bool>,
}

impl Instance {
    /// A replay of the tail, the live output, the exit code, and the end-of-stream that says the
    /// output is complete. Snapshot and subscribe share the lock the drain sends under, so replay
    /// meets live with no byte lost or doubled.
    pub fn attach(&self) -> Attached {
        let tail = self.tail.lock().unwrap();
        Attached {
            tail: tail.clone(),
            output: self.output.subscribe(),
            exit: self.exit.subscribe(),
            eof: self.eof.subscribe(),
        }
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

/// Where one instance's config is written: BESIDE the workspace, since inside would pack the
/// shim into every `.gfi` and dirty the patch merely for launching an agent.
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

/// The undo stack a spawned shell's ops land in — dropped with the instance, so a stopped
/// agent's history goes with it.
pub fn actor_of(id: &str) -> String {
    format!("agent-{id}")
}

/// The agent's command line under the user's own shell, as their terminal would run it: a LOGIN
/// shell on unix — nvm shims and profiles included — and `cmd /C` on Windows.
fn shell_command(command: &str) -> CommandBuilder {
    #[cfg(windows)]
    {
        let mut cmd = CommandBuilder::new("cmd");
        cmd.args(["/C", command]);
        cmd
    }
    #[cfg(not(windows))]
    {
        let mut cmd =
            CommandBuilder::new(std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into()));
        cmd.args(["-lc", command]);
        cmd
    }
}

/// Lay the `goofi` shim — this very binary — into the instance's own config dir, so two servers
/// of different builds cannot overwrite each other's.
fn write_shim(dir: &Path) -> Result<(), String> {
    let me = std::env::current_exe().map_err(|e| format!("the shim's target: {e}"))?;
    #[cfg(windows)]
    let done = std::fs::write(dir.join("goofi.cmd"), format!("@\"{}\" %*\r\n", me.display()));
    #[cfg(not(windows))]
    let done = {
        // Linked aside and renamed over, so a leftover always names the CURRENT binary.
        let tmp = dir.join(".goofi.part");
        let _ = std::fs::remove_file(&tmp);
        std::os::unix::fs::symlink(&me, &tmp)
            .and_then(|()| std::fs::rename(&tmp, dir.join("goofi")))
    };
    done.map_err(|e| format!("the goofi shim: {e}"))
}

/// The child's PATH with the shim dir FIRST. A login shell's profile may rebuild PATH over
/// this; `GOOFI_SESSION` still names the server, so a globally installed `goofi` also lands.
fn prepend_path(dir: &Path, env: &[(OsString, OsString)]) -> OsString {
    let tail = env
        .iter()
        .find(|(k, _)| k == "PATH")
        .map(|(_, v)| v.clone())
        .or_else(|| std::env::var_os("PATH"))
        .unwrap_or_default();
    std::env::join_paths(std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(&tail)))
        .unwrap_or(tail)
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

}
