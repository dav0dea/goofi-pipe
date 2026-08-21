//! Ask a process TREE to leave, then insist — the crate's one deliberate OS conditional, because
//! `portable_pty::ChildKiller` offers a single blunt kill of the harness alone.
//!
//! Both functions are idempotent by contract: a process that has already left is what was being
//! asked for, not a failure.

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
    match err.raw_os_error() {
        Some(libc::ESRCH) => Ok(()),
        _ => Err(err.to_string()),
    }
}

/// Ask everything the harness started to leave. A Windows console process refuses `taskkill`
/// without `/F`; that is not reported as a failure, because the grace timer reaches [`force_kill`].
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

/// taskkill's exit code for "there is no such pid" — the unix half's `ESRCH` under another name.
#[cfg(windows)]
const NOT_FOUND: i32 = 128;

/// `/T` is why this shells out rather than calling `TerminateProcess`: it takes the whole tree.
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

    /// Makes the child sleep: the child is THIS test binary re-entered with one `#[test]` filtered
    /// in, because no binary that merely stays alive is named the same on every platform.
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
        // A group of its own: without it `kill(-pid)` finds nothing, reads as "already gone", and
        // the fixture reports every stop a success while the child sleeps on.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        cmd.spawn().expect("spawn the sleeping child")
    }

    /// The contract [`crate::term`]'s stop is built on: ask, insist, and — once the child is gone —
    /// insist again.
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
