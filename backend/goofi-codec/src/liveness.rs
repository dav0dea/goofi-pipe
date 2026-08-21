//! Parent-liveness pipe, the subprocess tier's orphan guard: the parent holds the write end and
//! never writes, and the child exits when the read end reaches EOF. Both ends compile this module.

use std::io::{self, PipeReader, PipeWriter, Read};
use std::process::Command;

/// Env var carrying the read end the child inherits: a unix fd number, or a Windows HANDLE.
pub const ENV_VAR: &str = "GOOFI_PARENT_PIPE";

/// The armed pipe, holding the read end open across the `Command::spawn` that inherits it.
pub struct Armed {
    writer: PipeWriter,
    reader: PipeReader,
}

impl Armed {
    /// Call AFTER the guarded `cmd.spawn()`: yields the write end, which is never written to.
    pub fn into_writer(self) -> PipeWriter {
        drop(self.reader);
        self.writer
    }
}

/// Create the liveness pipe, arrange for `cmd`'s child to inherit the read end, and set
/// [`ENV_VAR`]. The returned [`Armed`] must outlive `cmd.spawn()`.
pub fn arm(cmd: &mut Command) -> io::Result<Armed> {
    let (reader, writer) = io::pipe()?;
    // Only the READ end is shared: std pipes are CLOEXEC, so the child's EOF means this
    // process died, not that a cousin still holds a write end.
    let value = share_read_end(cmd, &reader)?;
    cmd.env(ENV_VAR, value);
    Ok(Armed { writer, reader })
}

#[cfg(unix)]
fn share_read_end(cmd: &mut Command, reader: &PipeReader) -> io::Result<String> {
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;

    let fd = reader.as_raw_fd();
    // SAFETY: `fcntl` is async-signal-safe, so it is legal in `pre_exec`; clearing FD_CLOEXEC
    // there rather than in the parent keeps the fd private to THIS child.
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

// The Windows half is not CI-verified: no Windows host runs this suite.
#[cfg(windows)]
fn share_read_end(_cmd: &mut Command, reader: &PipeReader) -> io::Result<String> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{SetHandleInformation, HANDLE_FLAG_INHERIT};

    let handle = reader.as_raw_handle();
    // SAFETY: `handle` is live and owned by `reader` for the whole call; std passes
    // bInheritHandles=TRUE, so an inheritable handle crosses with the same numeric value.
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
    // SAFETY: the parent handed us this fd across exec and closed its own copy.
    Ok(unsafe { std::fs::File::from_raw_fd(fd) })
}

#[cfg(windows)]
fn reader_from_raw(raw: &str) -> io::Result<std::fs::File> {
    use std::os::windows::io::FromRawHandle;
    let value: usize =
        raw.parse().map_err(|_| io::Error::other(format!("{ENV_VAR}=`{raw}` is not a handle")))?;
    // SAFETY: the parent handed us this handle across the spawn and closed its own copy.
    Ok(unsafe { std::fs::File::from_raw_handle(value as _) })
}

/// Start the watcher thread on the read end named by `raw`: it blocks until the parent dies,
/// then ends this process at once, with no graceful shutdown.
pub fn watch_parent(raw: &str) -> io::Result<()> {
    let reader = reader_from_raw(raw)?;
    std::thread::Builder::new().name("goofi-parent-watch".into()).spawn(move || {
        wait_for_parent_exit(reader);
        // Outliving the parent is not a failure, so exit clean.
        std::process::exit(0);
    })?;
    Ok(())
}

/// Block until the parent's write end closes. The parent never writes, so a byte is not a death.
pub fn wait_for_parent_exit(mut reader: impl Read) {
    let mut scratch = [0u8; 64];
    loop {
        match reader.read(&mut scratch) {
            Ok(0) => return,
            Ok(_) => continue,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return,
        }
    }
}
