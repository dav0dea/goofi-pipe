//! Paths, in goofi's own spelling: **`/` is the separator everywhere inside goofi**, on every
//! platform. Windows is the special case, and it is accommodated at the boundary rather than
//! carried inward.
//!
//! That works because Win32 itself accepts `/` for filesystem calls — a path goofi hands to
//! `std::fs` or to a spawned process needs no conversion back. So there is no outbound half to
//! this module, and its absence is the design rather than an omission.
//!
//! The one thing that does NOT survive is the extended-length `\\?\` prefix `fs::canonicalize`
//! returns on Windows. Those paths are passed to the kernel unparsed: they reject forward slashes
//! outright, so a single one leaking inward makes every path derived from it un-spellable in
//! goofi's own syntax — and puts `\\?\C:\Users\…` in front of the user in the file browser.
//! Removing it is therefore the whole job, and [`canonical`] is where it happens: the point a path
//! ENTERS goofi.
//!
//! One consumer would still want `\` back: `cmd.exe` reads a leading `/` as a switch, not a path.
//! Nothing here spawns it today — the note is for whoever wires up `.cmd` harness shims.

use std::path::{Path, PathBuf, MAIN_SEPARATOR};

/// The goofi spelling of `p` — the ONE way a path becomes a string that leaves this process,
/// whether that is the wire, a `.gfi`, or a comparison against another path.
///
/// Replacing [`MAIN_SEPARATOR`] rather than `'\\'` is load-bearing, and the reason this is a
/// function and not an inline `.replace()`: on unix a backslash is a legal **filename** character,
/// so a blind replace would quietly corrupt a real name. `MAIN_SEPARATOR` is `\` only where `\` is
/// in fact the separator, which makes this a no-op exactly where it must be one.
pub fn to_slash(p: &Path) -> String {
    p.to_string_lossy().replace(MAIN_SEPARATOR, "/")
}

/// `p` made absolute and symlink-free, without the verbatim prefix.
///
/// `dunce` rather than [`std::fs::canonicalize`], which on Windows *always* returns the
/// `\\?\` form. dunce hands back the ordinary path whenever it is expressible without the prefix
/// and **keeps the prefix when it is not** — a path beyond `MAX_PATH`, a device path — which is
/// precisely the case a hand-rolled `strip_prefix(r"\\?\")` gets wrong, trading a cosmetic win for
/// a path that no longer opens.
pub fn canonical(p: &Path) -> std::io::Result<PathBuf> {
    dunce::canonicalize(p)
}
