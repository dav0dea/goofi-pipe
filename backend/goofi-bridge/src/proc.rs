//! The one thing about stopping a harness that nothing else here has an answer to borrow: **ask a
//! process tree to leave, then insist.**
//!
//! Everything else in [`crate::term`] is the same code on every platform — `which` answers what
//! counts as an executable, `portable-pty` answers what a PTY is. This does not have such an answer
//! to lean on. `portable_pty::ChildKiller` offers ONE blunt kill (SIGHUP on unix, `TerminateProcess`
//! on Windows) with neither a graceful phase nor any reach past the harness into what the harness
//! itself spawned — and a harness that is never asked first never gets to save its session.
//!
//! So this module is the deliberate exception to keeping goofi free of OS conditionals: two
//! functions, implemented once per platform, and the only `cfg` in this crate. Both are idempotent
//! by contract — a process that has already left is what was being asked for, not a failure — and
//! [`crate::term`] leans on that, because its grace timer fires whether or not the child is still
//! there to receive it.

/// Ask everything the harness started to leave, so it can save its session on the way out.
#[cfg(unix)]
pub fn request_stop(pid: u32) -> Result<(), String> {
    signal(pid, libc::SIGTERM)
}

/// Insist, once the grace has run out.
#[cfg(unix)]
pub fn force_kill(pid: u32) -> Result<(), String> {
    signal(pid, libc::SIGKILL)
}

/// Signal the process GROUP: portable-pty makes each child a session leader, so its pid is its
/// group id and one signal reaches everything the harness itself spawned.
#[cfg(unix)]
fn signal(pid: u32, sig: i32) -> Result<(), String> {
    // SAFETY: a plain syscall whose only fallible argument is the pid, and that pid names a child
    // this process spawned and has not yet reaped.
    if unsafe { libc::kill(-(pid as i32), sig) } == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    // ESRCH means it left between the caller's check and the syscall — which is what was asked for.
    match err.raw_os_error() {
        Some(libc::ESRCH) => Ok(()),
        _ => Err(err.to_string()),
    }
}

/// Windows has no graceful signal a console process can be relied on to receive. `CTRL_BREAK` is
/// the closest thing, and it needs the target to share THIS process's console and to have been
/// spawned into its own process group — but a ConPTY child is attached to the pseudoconsole
/// instead, so the event never reaches it. The ask is therefore `taskkill` without `/F`, which a
/// windowed process honours and a console one refuses outright.
///
/// That refusal is not a failed stop, and is deliberately not reported as one: the grace timer is
/// already on its way to [`force_kill`], so the harness leaves either way — a console harness just
/// leaves at the end of the grace rather than at the start. Only a `taskkill` that could not be run
/// at all says anything the caller can act on.
#[cfg(windows)]
pub fn request_stop(pid: u32) -> Result<(), String> {
    taskkill(pid, false).map(drop)
}

/// Insist, once the grace has run out.
#[cfg(windows)]
pub fn force_kill(pid: u32) -> Result<(), String> {
    let out = taskkill(pid, true)?;
    if out.status.success() || out.status.code() == Some(NOT_FOUND) {
        return Ok(());
    }
    Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
}

/// taskkill's exit code for "there is no such pid" — it left between the caller's check and here,
/// which is the unix half's `ESRCH` under another name.
#[cfg(windows)]
const NOT_FOUND: i32 = 128;

/// `/T` is the whole point of shelling out rather than calling `TerminateProcess` on the one
/// handle: it takes the tree the harness spawned, which is what the unix half gets for free by
/// signalling the process group.
#[cfg(windows)]
fn taskkill(pid: u32, force: bool) -> Result<std::process::Output, String> {
    let mut cmd = std::process::Command::new("taskkill");
    cmd.args(["/T", "/PID"]).arg(pid.to_string());
    if force {
        cmd.arg("/F");
    }
    cmd.output().map_err(|e| format!("taskkill: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Set on the child of the tests below; its presence is what makes that child sleep instead of
    /// returning at once. The child is THIS test binary re-entered with one `#[test]` filtered in,
    /// because no binary that merely stays alive is named the same thing on every platform — the
    /// same trick `goofi-subproc`'s liveness tests use, and the reason these tests are portable at
    /// all rather than being a unix suite wearing a cross-platform name.
    const SLEEPER_ENV: &str = "GOOFI_PROC_SLEEPER";

    /// The child. Unset — every ordinary suite run — it does nothing at all.
    #[test]
    fn proc_sleeper_process() {
        if std::env::var(SLEEPER_ENV).is_err() {
            return;
        }
        std::thread::sleep(Duration::from_secs(30));
    }

    fn spawn_sleeper() -> std::process::Child {
        let mut cmd =
            std::process::Command::new(std::env::current_exe().expect("this test binary's path"));
        cmd.args(["--exact", "proc::tests::proc_sleeper_process", "--test-threads=1"])
            .env(SLEEPER_ENV, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        // A group of its own, because [`request_stop`] signals a process GROUP and a child that
        // merely inherits the runner's is in no group named by its own pid: `kill(-pid)` then finds
        // nothing, returns ESRCH, and ESRCH is — correctly — read as "already gone". The fixture
        // would report every stop a success while the child slept on. `portable-pty` gives the real
        // harness this for free by making it a session leader; a fixture that skips it is testing a
        // path production never takes.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        cmd.spawn().expect("spawn the sleeping child")
    }

    /// The contract [`crate::term`]'s stop is built on, in the order that code performs it: ask,
    /// insist, and — once the child is gone — insist again. That last step is the load-bearing one,
    /// because the grace timer fires five seconds after the ask whether or not the harness took the
    /// hint; a `force_kill` that reported "no such process" as a failure would make every
    /// well-behaved harness log an error on its way out. What the ASK achieves differs by platform
    /// (a SIGTERM a unix harness can trap; a refusal from a Windows console one) — that neither
    /// reports a failure the caller has to interpret does not.
    #[test]
    fn a_stop_ends_a_live_child_and_insisting_on_a_departed_one_is_not_an_error() {
        let mut child = spawn_sleeper();
        let pid = child.id();

        request_stop(pid).expect("asking a live child to stop reports no failure");
        force_kill(pid).expect("a live child is killable");

        // Reap it, so the pid is genuinely retired before the second call rather than a zombie.
        let deadline = Instant::now() + Duration::from_secs(10);
        let ended = loop {
            match child.try_wait().expect("wait on the child") {
                Some(_) => break true,
                None if Instant::now() >= deadline => break false,
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        };
        if !ended {
            let _ = child.kill();
            let _ = child.wait();
        }
        assert!(ended, "the child was still alive 10s after a force_kill");

        force_kill(pid).expect("a child that has already left is what was being asked for");
    }
}
