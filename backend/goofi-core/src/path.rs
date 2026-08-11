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

#[cfg(test)]
mod tests {
    use super::*;

    /// A path built the platform's own way comes back in goofi's spelling. On Windows that means
    /// the separators are gone; on unix it was already spelled this way and must survive untouched.
    #[test]
    fn a_path_is_spelled_with_slashes_whatever_the_platform_builds() {
        let p = Path::new("one").join("two").join("three.txt");
        assert_eq!(to_slash(&p), "one/two/three.txt");
    }

    /// …and a name that merely CONTAINS the other platform's separator is not mangled by it. On
    /// unix `a\b` is one legal filename and must stay one; on Windows it cannot occur at all, so
    /// the same assertion reads as "two components", which is equally what the platform means.
    #[test]
    fn a_backslash_is_a_separator_only_where_the_platform_says_so() {
        let one = to_slash(Path::new("plain.txt"));
        assert_eq!(one, "plain.txt");
        let joined = Path::new("dir").join("a\\b");
        // Whatever this platform thinks `a\b` is, `to_slash` must agree with it rather than impose
        // an answer: the string round-trips back to the same `Path` either way.
        assert_eq!(Path::new(&to_slash(&joined)).components().count(), joined.components().count());
    }

    /// Every goofi path is born through [`canonical`], so the extended-length prefix has to die
    /// here — downstream is too late, because a `\\?\` path cannot hold the forward slashes the
    /// rest of goofi speaks in.
    #[test]
    fn a_canonical_path_carries_no_verbatim_prefix() {
        let real = canonical(&std::env::temp_dir()).expect("the temp dir canonicalizes");
        let spelled = to_slash(&real);
        assert!(!spelled.starts_with("//?/"), "verbatim prefix survived: {spelled}");
        assert!(!spelled.contains('\\'), "a separator survived: {spelled}");
    }
}
