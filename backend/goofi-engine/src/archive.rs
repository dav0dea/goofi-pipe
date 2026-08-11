//! The `.gfi` container: a zip holding `patch.yaml` beside a `workspace/` tree.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use walkdir::WalkDir;
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

const MANIFEST: &str = "patch.yaml";
const WORKSPACE: &str = "workspace";

/// The workspace's own list of what NOT to package, at its root. Named for goofi rather than the
/// bare `.ignore` it was asked for, because that name is already taken where it would sit:
/// ripgrep, fd and their kin read `.ignore` as a SEARCH ignore, and the workspace is exactly the
/// cwd goofi spawns an agent harness into. A line added here to keep a cache out of the archive
/// would have silently blinded the agent's own grep to it, and a line the agent added to be left
/// alone would have silently dropped it from the patch. A tool-prefixed name — `.dockerignore`,
/// `.npmignore` — says which tool reads it, and this one is read by goofi alone.
pub const IGNORE_FILE: &str = ".goofiignore";

/// What a new workspace's [`IGNORE_FILE`] says. Its header IS the syntax documentation, kept here
/// so it cannot drift from the [`Rule`] implementing it. Every rule earns its line by naming a file
/// that appears WITHOUT the author putting it there — which is what makes it both archive bloat
/// and, until this existed, an unsaved change nobody made.
pub const DEFAULT_IGNORE: &str = "\
# What goofi keeps out of this patch's `.gfi` archive. The same list decides whether the workspace
# has unsaved changes, so a rule here can never leave a patch dirty that a save cannot clean. Edit
# it freely: it is packaged with the patch and travels with it.
#
# One rule per line, in one of three forms. A line whose FIRST character is `#` is a comment; a `#`
# anywhere else is part of the name.
#
#   name     an entry called exactly that, at any depth
#   name/    the same, but only a directory — everything under it goes too
#   *.ext    any file whose name ends in `.ext`
#
# There is no `!`, no `**`, and no pattern with a `/` inside it. A line using one is SKIPPED rather
# than guessed at, so a pattern can never quietly mean something other than it reads as.

# CPython writes one beside every module it imports, and a node is imported with the workspace as
# the working directory
__pycache__/
# the same bytecode, wherever it lands outside a __pycache__
*.pyc
# macOS Finder drops one into any directory it displays
.DS_Store
# an editor's swap file exists only while a node is being edited, and would otherwise flip the
# patch to unsaved and back as it comes and goes
*.swp
";

/// One line of [`IGNORE_FILE`]. The three forms are the whole grammar — see [`DEFAULT_IGNORE`],
/// whose header is what a user reads — and a line none of them can spell parses to `None`, so an
/// unimplemented glob excludes nothing rather than approximately something.
enum Rule {
    /// `name` — an entry called exactly that, at any depth.
    Name(String),
    /// `name/` — the same, but only a directory; the walk prunes everything below it.
    Dir(String),
    /// `*.ext` — any file whose name ends with the stored `.ext`, dot included.
    Ext(String),
}

impl Rule {
    fn parse(line: &str) -> Option<Rule> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (body, dir_only) = line.strip_suffix('/').map_or((line, false), |b| (b, true));
        // Every metacharacter this deliberately does not implement, in one place: `*` outside the
        // leading `*.`, a `/` inside the pattern, gitignore's `!` negation, and glob's `?`/`[`.
        let unspellable = |s: &str| s.is_empty() || s.contains(['*', '/', '!', '?', '[']);
        match body.strip_prefix("*.") {
            // `*.ext/` would be an extension that is also a directory — no form means that.
            Some(ext) => (!dir_only && !unspellable(ext)).then(|| Rule::Ext(format!(".{ext}"))),
            None if unspellable(body) => None,
            None if dir_only => Some(Rule::Dir(body.to_string())),
            None => Some(Rule::Name(body.to_string())),
        }
    }

    fn matches(&self, entry: &walkdir::DirEntry) -> bool {
        let name = entry.file_name().to_string_lossy();
        match self {
            Rule::Name(n) => *name == **n,
            Rule::Dir(n) => entry.file_type().is_dir() && *name == **n,
            Rule::Ext(dot_ext) => !entry.file_type().is_dir() && name.ends_with(dot_ext),
        }
    }
}

/// The rules in force for the workspace at `dir` — read inside the shared walk rather than passed
/// in, because two callers that cannot name a list cannot be handed different ones. No ignore file
/// (every patch saved before this existed) means no rules, so such a patch packs as it always did.
fn rules(dir: &Path) -> Vec<Rule> {
    fs::read_to_string(dir.join(IGNORE_FILE))
        .map(|s| s.lines().filter_map(Rule::parse).collect())
        .unwrap_or_default()
}

/// Every regular file under `dir` that [`IGNORE_FILE`] does not exclude, sorted. ONE walk serves
/// both the pack and the [`fingerprint`] it is compared against — a second walk with its own idea
/// of what counts as a workspace file would let the two disagree, and the disagreement would read
/// as a patch that is dirty the instant it is saved — which is why the ignore rules are applied
/// HERE and at neither call site: a `__pycache__` skipped by only one of the two would dirty a
/// patch that no save could ever clean. Anything that is not a regular file (directory, symlink,
/// socket) is skipped; walk errors are kept, because only the pack can decide whether one is fatal.
fn files(dir: &Path) -> impl Iterator<Item = walkdir::Result<walkdir::DirEntry>> {
    let rules = rules(dir);
    WalkDir::new(dir)
        .sort_by_file_name()
        .into_iter()
        // Excluding a directory prunes its subtree, which is what makes `__pycache__/` one rule
        // rather than one per file. Two entries are never excluded: the root, since a rule that
        // matched the mount's own name would silently pack nothing at all, and the ignore file,
        // which has to ride the archive or a loaded patch comes back without its own rules.
        .filter_entry(move |e| {
            e.depth() == 0
                || e.file_name() == std::ffi::OsStr::new(IGNORE_FILE)
                || !rules.iter().any(|r| r.matches(e))
        })
        .filter(|e| e.as_ref().map_or(true, |e| e.file_type().is_file()))
}

/// What the workspace at `mount` looked like: relative path → (length, mtime), for every regular
/// file. Two fingerprints differing means a file was added, removed, resized or rewritten — which
/// is the manager's ONLY way to notice an edit made outside goofi, since there is no filesystem
/// watcher (decision, 2026-08-09) and the archive stores no mtimes to compare against.
///
/// A file whose metadata cannot be read is simply absent, so an unreadable file reads as a
/// difference — the safe direction: it marks the patch unsaved rather than silently losing it.
pub fn fingerprint(mount: &Path) -> BTreeMap<PathBuf, (u64, SystemTime)> {
    files(mount)
        .filter_map(|e| {
            let entry = e.ok()?;
            let rel = entry.path().strip_prefix(mount).ok()?.to_path_buf();
            let md = entry.metadata().ok()?;
            Some((rel, (md.len(), md.modified().ok()?)))
        })
        .collect()
}

/// Pack `manifest` plus every regular file under `workspace_dir` into a `.gfi` at `out`.
///
/// The walk is sorted so an unchanged tree packs byte-identically: with zip's `time` feature off
/// every entry is stamped 1980-01-01, which leaves entry order as the only varying field.
/// Anything that is not a regular file (directory, symlink, socket) is skipped.
pub fn write_gfi(out: &Path, manifest: &str, workspace_dir: &Path) -> Result<(), String> {
    // io and zip errors both land here, each named by the path it is actually about.
    let at = |p: &Path, e: &dyn std::fmt::Display| format!("{}: {e}", p.display());
    let mut zip = ZipWriter::new(File::create(out).map_err(|e| at(out, &e))?);
    let opts = SimpleFileOptions::default();
    zip.start_file(MANIFEST, opts).map_err(|e| at(out, &e))?;
    zip.write_all(manifest.as_bytes()).map_err(|e| at(out, &e))?;

    for entry in files(workspace_dir) {
        let entry = entry.map_err(|e| at(workspace_dir, &e))?;
        let rel = entry.path().strip_prefix(workspace_dir).map_err(|e| e.to_string())?;
        // A zip entry name is bytes-as-UTF-8; refuse rather than mangle a name we cannot spell.
        let rel = rel.to_str().ok_or_else(|| format!("{}: name is not UTF-8", rel.display()))?;
        zip.start_file(format!("{WORKSPACE}/{rel}"), opts).map_err(|e| at(entry.path(), &e))?;
        let mut src = File::open(entry.path()).map_err(|e| at(entry.path(), &e))?;
        std::io::copy(&mut src, &mut zip).map_err(|e| at(entry.path(), &e))?;
    }
    zip.finish().map_err(|e| at(out, &e))?;
    Ok(())
}

/// Unpack a `.gfi`: the workspace tree lands at `dest` (which must not already hold files), and the
/// manifest text is returned. Both structural refusals — not a zip, no manifest — happen before any
/// extraction; a failure part-way through extraction may leave the scratch sibling on disk.
///
/// Extraction goes through `ZipArchive::extract`, whose `safe_prepare_path` is zip's own zip-slip
/// containment — this must not hand-roll path sanitization. It creates symlinks verbatim and
/// enforces no size cap; both are accepted (a `.gfi` is a user's own file, not hostile input).
pub fn read_gfi(archive: &Path, dest: &Path) -> Result<String, String> {
    let named = |e: String| format!("{}: {e}", archive.display());
    let file = File::open(archive).map_err(|e| named(e.to_string()))?;
    let mut zip = ZipArchive::new(file).map_err(|e| named(format!("not a zip archive ({e})")))?;
    let mut manifest = String::new();
    zip.by_name(MANIFEST)
        .map_err(|_| named(format!("archive has no {MANIFEST}")))?
        .read_to_string(&mut manifest)
        .map_err(|e| named(e.to_string()))?;

    // Scratch sits beside `dest` (suffix appended, not substituted) so the move below is a same-fs rename.
    let scratch = PathBuf::from({ let mut s = dest.as_os_str().to_owned(); s.push(".unpack"); s });
    let _ = fs::remove_dir_all(&scratch);
    zip.extract(&scratch).map_err(|e| named(e.to_string()))?;
    let packed = scratch.join(WORKSPACE);
    let moved = if packed.is_dir() {
        fs::rename(&packed, dest)
    } else {
        fs::create_dir_all(dest) // an archive may legitimately carry no workspace files
    };
    let _ = fs::remove_dir_all(&scratch);
    moved.map_err(|e| format!("{}: {e}", dest.display()))?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A workspace with a file at the root and one in a sub-directory, plus the manifest.
    fn sample(root: &Path) -> PathBuf {
        let ws = root.join("ws");
        fs::create_dir_all(ws.join("sub")).unwrap();
        fs::write(ws.join("a.txt"), b"alpha").unwrap();
        fs::write(ws.join("sub/b.txt"), b"beta").unwrap();
        ws
    }

    /// Every entry name a `.gfi` actually holds, in the order it holds them.
    fn entry_names(gfi: &Path) -> Vec<String> {
        let mut zip = ZipArchive::new(fs::File::open(gfi).unwrap()).unwrap();
        (0..zip.len()).map(|i| zip.by_index(i).unwrap().name().to_owned()).collect()
    }

    /// What a workspace packs to, given `ignore` as its [`IGNORE_FILE`] — the workspace-relative
    /// half of the names, since `patch.yaml` and the `workspace/` prefix are the same either way.
    fn packed_with(ws: &Path, tmp: &Path, ignore: &str) -> Vec<String> {
        fs::write(ws.join(IGNORE_FILE), ignore).unwrap();
        let gfi = tmp.join("packed.gfi");
        write_gfi(&gfi, "version: 7\n", ws).unwrap();
        entry_names(&gfi).iter().filter_map(|n| n.strip_prefix("workspace/").map(str::to_owned)).collect()
    }

    #[test]
    fn round_trips_a_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = sample(tmp.path());
        let gfi = tmp.path().join("patch.gfi");
        write_gfi(&gfi, "version: 7\n", &ws).unwrap();

        let dest = tmp.path().join("unpacked");
        let manifest = read_gfi(&gfi, &dest).unwrap();
        assert_eq!(manifest, "version: 7\n");
        assert_eq!(fs::read(dest.join("a.txt")).unwrap(), b"alpha");
        assert_eq!(fs::read(dest.join("sub/b.txt")).unwrap(), b"beta");
        assert!(!tmp.path().join("unpacked.unpack").exists(), "the scratch sibling is cleaned up");

        // A workspace with no regular files packs to the manifest alone, so `dest` is created rather
        // than renamed into place — the path every new patch takes.
        let empty = tmp.path().join("empty");
        fs::create_dir(&empty).unwrap();
        let bare = tmp.path().join("bare.gfi");
        write_gfi(&bare, "version: 7\n", &empty).unwrap();
        let dest = tmp.path().join("mounted");
        assert_eq!(read_gfi(&bare, &dest).unwrap(), "version: 7\n");
        assert!(dest.is_dir());
    }

    /// Entry order is the only field that varies between two packs of an unchanged tree (every
    /// entry is stamped 1980-01-01), so sorted order IS the byte-identity guarantee.
    #[test]
    fn packs_entries_in_sorted_order() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join("ws");
        fs::create_dir_all(&ws).unwrap();
        // names whose readdir order differs from their sorted order, so an unsorted walk fails here
        for name in ["beta.txt", "a.txt", "gamma.md"] {
            fs::write(ws.join(name), name).unwrap();
        }
        let gfi = tmp.path().join("patch.gfi");
        write_gfi(&gfi, "version: 7\n", &ws).unwrap();

        let mut zip = zip::ZipArchive::new(fs::File::open(&gfi).unwrap()).unwrap();
        let names: Vec<String> = (0..zip.len()).map(|i| zip.by_index(i).unwrap().name().to_owned()).collect();
        assert_eq!(
            names,
            ["patch.yaml", "workspace/a.txt", "workspace/beta.txt", "workspace/gamma.md"]
        );
    }

    /// The load-bearing pair, and the whole reason the ignore list is read INSIDE the shared walk:
    /// a `__pycache__` appears unasked the first time a Python node is imported (nodes import with
    /// the workspace as cwd), and it must reach NEITHER the archive nor the fingerprint. Were it
    /// filtered from only one of them the patch would be dirty the instant it was saved, and no
    /// save could ever clean it.
    #[test]
    fn what_the_pack_skips_the_fingerprint_skips_too() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = sample(tmp.path());
        fs::write(ws.join(IGNORE_FILE), DEFAULT_IGNORE).unwrap();
        let saved = fingerprint(&ws);

        // What importing a node does to the workspace, unasked and in two places.
        fs::create_dir_all(ws.join("sub/__pycache__")).unwrap();
        fs::write(ws.join("sub/__pycache__/b.cpython-314.pyc"), b"bytecode").unwrap();
        fs::write(ws.join("stray.pyc"), b"bytecode").unwrap();
        assert_eq!(fingerprint(&ws), saved, "importing a node left the patch permanently dirty");

        let packed = packed_with(&ws, tmp.path(), DEFAULT_IGNORE);
        assert!(
            !packed.iter().any(|n| n.contains("__pycache__") || n.ends_with(".pyc")),
            "bytecode rode the archive: {packed:?}"
        );
        // …and the rules themselves ride it, or the patch loses them on the way back off disk.
        assert!(packed.contains(&IGNORE_FILE.to_string()), "{packed:?}");
        assert!(packed.contains(&"a.txt".to_string()), "{packed:?}");
    }

    /// The grammar's three forms and its refusal, stated as behaviour: `*.ext` is a suffix, a bare
    /// name is the WHOLE name rather than a substring, a trailing `/` needs a directory — and a
    /// line none of the three can spell (a path, a negation, a `**`) is skipped rather than
    /// guessed at, so what it names is packed as if it had never been written.
    #[test]
    fn a_line_the_grammar_cannot_spell_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = sample(tmp.path());
        fs::write(ws.join("notes.md"), b"md").unwrap();
        fs::write(ws.join("keep.pycache"), b"named for it, but not it").unwrap();
        fs::write(ws.join("build"), b"a file, not a directory").unwrap();

        let packed = packed_with(
            &ws,
            tmp.path(),
            "# a comment is not a pattern\n\n  *.md  \nbuild/\npycache\nsub/b.txt\n!a.txt\n**/keep*\n",
        );

        assert!(!packed.contains(&"notes.md".to_string()), "*.md matched nothing: {packed:?}");
        for kept in ["a.txt", "sub/b.txt", "keep.pycache", "build", IGNORE_FILE] {
            assert!(packed.contains(&kept.to_string()), "{kept} was dropped: {packed:?}");
        }
    }

    /// The mount's own directory is never what a rule excludes. A workspace whose root happens to
    /// share a name with a rule — `ws/`, here — would otherwise be pruned at depth 0 and pack
    /// NOTHING: the one failure mode of an ignore list that loses a patch rather than bloating one.
    #[test]
    fn a_rule_matching_the_mount_itself_does_not_empty_the_patch() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = sample(tmp.path()); // …which `sample` names `ws`
        let packed = packed_with(&ws, tmp.path(), "ws/\n");
        assert!(packed.contains(&"a.txt".to_string()), "the mount pruned itself: {packed:?}");
        assert!(!fingerprint(&ws).is_empty(), "…and so did the fingerprint");
    }

    /// The ignore file is never ignored, not even by itself. It is packaged like any other
    /// workspace file — that is how a loaded patch arrives still knowing what to leave out — so a
    /// line naming it would otherwise quietly delete the patch's own rules at the next save.
    #[test]
    fn the_ignore_file_is_packaged_even_when_it_names_itself() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = sample(tmp.path());
        let packed = packed_with(&ws, tmp.path(), &format!("{IGNORE_FILE}\n"));
        assert!(packed.contains(&IGNORE_FILE.to_string()), "{packed:?}");
        assert!(fingerprint(&ws).contains_key(Path::new(IGNORE_FILE)), "…and the fingerprint has it");
    }

    #[test]
    fn refuses_an_archive_without_a_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let gfi = tmp.path().join("patch.gfi");
        let mut zip = zip::ZipWriter::new(fs::File::create(&gfi).unwrap());
        zip.start_file("workspace/a.txt", zip::write::SimpleFileOptions::default()).unwrap();
        zip.write_all(b"alpha").unwrap();
        zip.finish().unwrap();

        let dest = tmp.path().join("unpacked");
        let err = read_gfi(&gfi, &dest).unwrap_err();
        assert!(err.contains("patch.yaml"), "{err}");
        // refused before anything was written to disk
        assert!(!dest.exists());
    }

    #[test]
    fn refuses_a_file_that_is_not_a_zip() {
        let tmp = tempfile::tempdir().unwrap();
        let gfi = tmp.path().join("patch.gfi");
        fs::write(&gfi, b"version: 7\n").unwrap();

        let err = read_gfi(&gfi, &tmp.path().join("unpacked")).unwrap_err();
        assert!(err.contains("not a zip archive"), "{err}");
    }
}
