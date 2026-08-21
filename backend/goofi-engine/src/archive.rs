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

/// The workspace's own list of what NOT to package. Not named `.ignore`: ripgrep and its kin read
/// that as a SEARCH ignore, and the workspace is the cwd goofi spawns an agent harness into.
pub const IGNORE_FILE: &str = ".goofiignore";

/// What a new workspace's [`IGNORE_FILE`] says. Its header IS the syntax documentation.
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

/// One line of [`IGNORE_FILE`]. A line no form can spell parses to `None`, so an unimplemented
/// glob excludes nothing rather than approximately something.
enum Rule {
    Name(String),
    Dir(String),
    Ext(String),
}

impl Rule {
    fn parse(line: &str) -> Option<Rule> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (body, dir_only) = line.strip_suffix('/').map_or((line, false), |b| (b, true));
        // Every metacharacter this deliberately does not implement, in one place.
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

/// The rules in force for the workspace at `dir`; no ignore file means no rules.
fn rules(dir: &Path) -> Vec<Rule> {
    fs::read_to_string(dir.join(IGNORE_FILE))
        .map(|s| s.lines().filter_map(Rule::parse).collect())
        .unwrap_or_default()
}

/// Every regular file under `dir` that [`IGNORE_FILE`] does not exclude, sorted. ONE walk serves
/// both the pack and [`fingerprint`], or a file skipped by only one would dirty a patch no save can
/// clean.
fn files(dir: &Path) -> impl Iterator<Item = walkdir::Result<walkdir::DirEntry>> {
    let rules = rules(dir);
    WalkDir::new(dir)
        .sort_by_file_name()
        .into_iter()
        // The root and the ignore file are never excluded: a rule matching the mount's own name
        // would pack nothing, and a loaded patch must come back with its own rules.
        .filter_entry(move |e| {
            e.depth() == 0
                || e.file_name() == std::ffi::OsStr::new(IGNORE_FILE)
                || !rules.iter().any(|r| r.matches(e))
        })
        .filter(|e| e.as_ref().map_or(true, |e| e.file_type().is_file()))
}

/// What the workspace at `mount` looked like: relative path → (length, mtime), per regular file.
/// A file whose metadata cannot be read is absent, so it reads as unsaved rather than lost.
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

/// Pack `manifest` plus every regular file under `workspace_dir` into a `.gfi` at `out`. The walk
/// is sorted so an unchanged tree packs byte-identically.
pub fn write_gfi(out: &Path, manifest: &str, workspace_dir: &Path) -> Result<(), String> {
    let at = |p: &Path, e: &dyn std::fmt::Display| format!("{}: {e}", p.display());
    let mut zip = ZipWriter::new(File::create(out).map_err(|e| at(out, &e))?);
    let opts = SimpleFileOptions::default();
    zip.start_file(MANIFEST, opts).map_err(|e| at(out, &e))?;
    zip.write_all(manifest.as_bytes()).map_err(|e| at(out, &e))?;

    for entry in files(workspace_dir) {
        let entry = entry.map_err(|e| at(workspace_dir, &e))?;
        let rel = entry.path().strip_prefix(workspace_dir).map_err(|e| e.to_string())?;
        let rel = rel.to_str().ok_or_else(|| format!("{}: name is not UTF-8", rel.display()))?;
        // A zip entry name is `/`-separated BY SPEC; replacing `MAIN_SEPARATOR` rather than `\`
        // keeps a unix file genuinely called `a\b` intact.
        let rel = rel.replace(std::path::MAIN_SEPARATOR, "/");
        zip.start_file(format!("{WORKSPACE}/{rel}"), opts).map_err(|e| at(entry.path(), &e))?;
        let mut src = File::open(entry.path()).map_err(|e| at(entry.path(), &e))?;
        std::io::copy(&mut src, &mut zip).map_err(|e| at(entry.path(), &e))?;
    }
    zip.finish().map_err(|e| at(out, &e))?;
    Ok(())
}

/// Unpack a `.gfi`: the workspace tree lands at `dest`, and the manifest text is returned.
/// `ZipArchive::extract` carries zip's own zip-slip containment — do not hand-roll sanitization.
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

