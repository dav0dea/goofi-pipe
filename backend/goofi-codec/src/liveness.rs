//! Parent-liveness pipe — the subprocess tier's orphan guard, shared by both ends.
//!
//! The parent (`goofi_python::subproc`'s `Running::spawn`) creates a pipe, keeps the **write**
//! end alive for the child's lifetime and **never writes to it**, and hands the **read**
//! end to the child through [`ENV_VAR`]. The child (`goofi.serve`) blocks reading that end
//! on a watcher thread: when the parent dies for ANY reason — Ctrl-C, a panic, `kill -9`, a
//! clean exit — the OS closes every copy of the write end, the read returns EOF, and the
//! watcher exits the process at once.
//!
//! Why a pipe rather than a `ctrl_c` handler or a heartbeat: it survives a **hard parent
//! crash**, which neither of those can, and it is conceptually identical on Linux, macOS and
//! Windows — deliberately NOT `PR_SET_PDEATHSIG`, which is Linux-only.
//!
//! It lives in `goofi-codec` because this crate already owns the parent↔child subprocess
//! protocol (`encode_request`/`decode_response`): one module, compiled into both the parent
//! (`goofi-python`) and the child (`goofi-pymod`'s wheel), so the two ends cannot drift.

use std::io::{self, PipeReader, PipeWriter, Read};
use std::process::Command;

/// Env var carrying the read end the child inherits: a unix fd number, or a Windows HANDLE
/// value. Set by [`arm`], consumed by [`watch_parent`].
pub const ENV_VAR: &str = "GOOFI_PARENT_PIPE";

/// The armed pipe, between [`arm`] and the spawn it guards. Holds the parent's own copy of
/// the read end open across `Command::spawn` (the child inherits it there); call
/// [`Armed::into_writer`] afterwards to close that copy and keep only the write end.
pub struct Armed {
    writer: PipeWriter,
    reader: PipeReader,
}

impl Armed {
    /// Call AFTER the guarded `cmd.spawn()`. Closes the parent's copy of the read end and
    /// yields the write end — hold that for the child's lifetime and never write to it.
    pub fn into_writer(self) -> PipeWriter {
        drop(self.reader);
        self.writer
    }
}

/// Create the liveness pipe, arrange for `cmd`'s child to inherit the read end, and set
/// [`ENV_VAR`] on `cmd` to the value the child reconstructs it from.
///
/// The returned [`Armed`] must outlive `cmd.spawn()`.
pub fn arm(cmd: &mut Command) -> io::Result<Armed> {
    let (reader, writer) = io::pipe()?;
    // Only the READ end is shared. The write end stays private to this process by
    // construction (std creates pipes CLOEXEC / non-inheritable) — that is what makes the
    // child's EOF mean "the parent died" and not "some cousin process is still holding it".
    let value = share_read_end(cmd, &reader)?;
    cmd.env(ENV_VAR, value);
    Ok(Armed { writer, reader })
}

// ---------------------------------------------------------------------------
// The cfg split — the ONE genuinely platform-specific part: handing the read end across
// the spawn, and reconstructing it in the child from the value in `ENV_VAR`. Everything
// above and below is shared.
//
// The unix half is exercised by this repo's suite on Linux (and is identical on macOS).
// The Windows half is structurally correct but NOT CI-verified here — no Windows host runs
// this suite. Treat it as reviewed-not-proven.
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn share_read_end(cmd: &mut Command, reader: &PipeReader) -> io::Result<String> {
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;

    let fd = reader.as_raw_fd();
    // SAFETY: `pre_exec` runs in the forked child between fork and exec, where only
    // async-signal-safe calls are legal — `fcntl` is one. Clearing FD_CLOEXEC there rather
    // than in the parent keeps the fd private to THIS child: a spawn racing on another
    // thread cannot inherit it.
    unsafe {
        cmd.pre_exec(move || {
            if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    // fork copies the descriptor table verbatim, so the child sees the same number.
    Ok(fd.to_string())
}

#[cfg(windows)]
fn share_read_end(_cmd: &mut Command, reader: &PipeReader) -> io::Result<String> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{SetHandleInformation, HANDLE_FLAG_INHERIT};

    let handle = reader.as_raw_handle();
    // SAFETY: `handle` is live and owned by `reader` for the whole call.
    // std's `CreateProcessW` passes bInheritHandles=TRUE, so an inheritable handle crosses
    // with the same numeric value. Inheritability is process-global, so a spawn racing on
    // another thread may also inherit this READ end — harmless: only a live WRITE end can
    // hold off a child's EOF, and those are never marked inheritable.
    if unsafe { SetHandleInformation(handle as _, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((handle as usize).to_string())
}

#[cfg(unix)]
fn reader_from_raw(raw: &str) -> io::Result<std::fs::File> {
    use std::os::fd::FromRawFd;
    let fd: std::os::fd::RawFd =
        raw.parse().map_err(|_| io::Error::other(format!("{ENV_VAR}=`{raw}` is not an fd")))?;
    // SAFETY: the parent handed us this fd across exec and closed its own copy; we own it.
    Ok(unsafe { std::fs::File::from_raw_fd(fd) })
}

#[cfg(windows)]
fn reader_from_raw(raw: &str) -> io::Result<std::fs::File> {
    use std::os::windows::io::FromRawHandle;
    let value: usize =
        raw.parse().map_err(|_| io::Error::other(format!("{ENV_VAR}=`{raw}` is not a handle")))?;
    // SAFETY: the parent handed us this handle across `CreateProcessW` and closed its own
    // copy; we own it.
    Ok(unsafe { std::fs::File::from_raw_handle(value as _) })
}

/// Start the child's watcher thread on the read end named by `raw` (the [`ENV_VAR`] value):
/// it blocks until the parent dies, then ends **this** process immediately.
///
/// There is deliberately no graceful shutdown through the iceoryx2 loop — the parent is
/// gone, so there is nobody left to talk to, and a child that keeps polling is the orphan
/// this whole mechanism exists to prevent.
pub fn watch_parent(raw: &str) -> io::Result<()> {
    let reader = reader_from_raw(raw)?;
    std::thread::Builder::new().name("goofi-parent-watch".into()).spawn(move || {
        wait_for_parent_exit(reader);
        // Outliving the parent is not a failure, so exit clean.
        std::process::exit(0);
    })?;
    Ok(())
}

/// Block until the parent's write end closes.
///
/// The one platform-free half of the mechanism, and the whole of the decision: EOF (or a
/// broken pipe) means the parent is gone. The parent never writes, so a byte that does
/// arrive is *not* death — ignore it rather than treat a future heartbeat as a funeral.
pub fn wait_for_parent_exit(mut reader: impl Read) {
    let mut scratch = [0u8; 64];
    loop {
        match reader.read(&mut scratch) {
            Ok(0) => return,                                            // EOF — the parent is gone
            Ok(_) => continue,                                          // a byte is not a death
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue, // a signal, not a death
            Err(_) => return, // a broken/closed pipe is the parent's death too
        }
    }
}
