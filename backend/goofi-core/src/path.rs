//! Paths in goofi's own spelling: `/` is the separator inside goofi on every platform. Win32
//! accepts `/` back, so there is no outbound half to this module.

use std::path::{Path, PathBuf, MAIN_SEPARATOR};

/// The goofi spelling of `p` — the one way a path becomes a string that leaves this process.
/// Replacing [`MAIN_SEPARATOR`], not `'\\'`: on unix a backslash is a legal filename character.
pub fn to_slash(p: &Path) -> String {
    p.to_string_lossy().replace(MAIN_SEPARATOR, "/")
}

/// `p` made absolute and symlink-free, without Windows' verbatim `\\?\` prefix.
/// dunce rather than [`std::fs::canonicalize`]: it keeps the prefix where the path needs it.
pub fn canonical(p: &Path) -> std::io::Result<PathBuf> {
    dunce::canonicalize(p)
}
