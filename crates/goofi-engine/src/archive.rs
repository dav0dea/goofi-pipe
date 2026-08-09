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

/// Every regular file under `dir`, sorted. ONE walk serves both the pack and the [`fingerprint`]
/// it is compared against — a second walk with its own idea of what counts as a workspace file
/// would let the two disagree, and the disagreement would read as a patch that is dirty the
/// instant it is saved. Anything that is not a regular file (directory, symlink, socket) is
/// skipped; walk errors are kept, because only the pack can decide whether one is fatal.
fn files(dir: &Path) -> impl Iterator<Item = walkdir::Result<walkdir::DirEntry>> {
    WalkDir::new(dir)
        .sort_by_file_name()
        .into_iter()
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
