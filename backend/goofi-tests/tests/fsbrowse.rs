//! The filesystem browser behind Save and Load — `list_dir`, and the path spelling it promises.
//!
//! Expectations are built from `goofi_core::path`'s own primitives rather than from the browser's
//! answer: asserting with the function under test on both sides would accept, say, a normalizer
//! that reversed the segments it re-attaches.

use std::path::{Path, PathBuf};

use goofi_core::path::{canonical, to_slash};
use goofi_tests::{j, Goofi};
use serde_json::Value;

/// A unique directory removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = tempfile::Builder::new().prefix(&format!("goofi-fsbrowse-{tag}-")).tempdir().unwrap();
        Self(dir.keep())
    }
    fn file(&self, name: &str) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, b"x").unwrap();
        p
    }
    fn dir(&self, name: &str) {
        std::fs::create_dir_all(self.0.join(name)).unwrap();
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn list(g: &Goofi, path: Option<&str>) -> Value {
    match path {
        Some(p) => g.call("list_dir", j!({ "path": p })),
        None => g.call("list_dir", j!({})),
    }
}

fn names(listing: &Value) -> Vec<String> {
    listing["entries"].as_array().unwrap().iter()
        .map(|e| e["name"].as_str().unwrap().to_string()).collect()
}

/// The filesystem root, whatever this platform calls it. Derived from a real path rather than
/// written as `/`, because on Windows `/` is not a root at all: it absolutizes against the current
/// drive and comes back as `C:/`, so a hardcoded `/` tests nothing there.
fn root() -> String {
    let real = canonical(&std::env::temp_dir()).expect("the temp dir canonicalizes");
    to_slash(real.ancestors().last().expect("every path has a topmost ancestor"))
}

fn home() -> String {
    let h = std::env::var("HOME").expect("HOME is set in the test environment");
    to_slash(&canonical(Path::new(&h)).unwrap_or_else(|_| PathBuf::from(h)))
}

#[test]
fn directories_come_before_files_then_case_insensitively_by_name() {
    let g = Goofi::new();
    let tmp = TempDir::new("order");
    tmp.file("Beta.txt");
    tmp.file("alpha.txt");
    tmp.dir("Zeta");
    tmp.dir("apples");

    assert_eq!(names(&list(&g, Some(&tmp.path().to_string_lossy()))),
               ["apples", "Zeta", "alpha.txt", "Beta.txt"]);
}

#[test]
fn a_file_path_lists_its_parent_directory() {
    let g = Goofi::new();
    let tmp = TempDir::new("file");
    let patch = tmp.file("patch.gfi");

    let listing = list(&g, Some(&patch.to_string_lossy()));
    assert_eq!(listing["path"].as_str().unwrap(), to_slash(&canonical(tmp.path()).unwrap()));
    assert_eq!(names(&listing), ["patch.gfi"]);
}

#[test]
fn home_is_where_the_browser_opens_and_the_root_has_no_parent() {
    let g = Goofi::new();
    // The frontend omits `path` on the first open; a cleared path input sends "".
    assert_eq!(list(&g, None)["path"].as_str().unwrap(), home());
    assert_eq!(list(&g, Some("")), list(&g, None));
    assert_eq!(list(&g, Some("~"))["path"].as_str().unwrap(), home(), "a leading tilde expands");

    let root = root();
    let listing = list(&g, Some(&root));
    assert_eq!(listing["path"].as_str().unwrap(), root);
    assert!(listing["parent"].is_null(), "root has no parent to climb to");
}

#[test]
fn every_entry_carries_the_full_shape_and_nothing_unrendered() {
    let g = Goofi::new();
    let tmp = TempDir::new("shape");
    tmp.file("a.txt");
    tmp.file(".hidden");
    tmp.file("patch.gfi");

    let listing = list(&g, Some(&tmp.path().to_string_lossy()));
    let by_name = |n: &str| listing["entries"].as_array().unwrap().iter()
        .find(|e| e["name"] == n).unwrap_or_else(|| panic!("{n} is listed")).clone();

    let entry = by_name("a.txt");
    for key in ["name", "path", "kind", "is_gfi", "hidden"] {
        assert!(entry.get(key).is_some(), "an entry is missing `{key}`");
    }
    assert_eq!(entry["kind"], "file");
    assert_eq!(entry["path"], to_slash(&canonical(tmp.path()).unwrap().join("a.txt")));
    // The browser renders neither a size nor a date column, so the row carries neither. Re-add them
    // together with the column that shows them.
    for key in ["size", "mtime"] {
        assert!(entry.get(key).is_none(), "an entry carries an unrendered `{key}`");
    }

    // Hidden entries are EMITTED, not filtered — the browser owns that toggle.
    assert_eq!(by_name(".hidden")["hidden"], true);
    assert_eq!(by_name("patch.gfi")["hidden"], false);
    assert_eq!(by_name("patch.gfi")["is_gfi"], true);
    assert_eq!(by_name("a.txt")["is_gfi"], false);
}

#[test]
fn a_path_that_is_not_on_disk_echoes_back_exactly_rather_than_failing() {
    // Navigation must not error: the browser would keep showing the previous directory.
    let g = Goofi::new();
    let missing = format!("{}definitely/not/a/directory", root());
    let listing = list(&g, Some(&missing));
    assert_eq!(names(&listing), Vec::<String>::new());
    assert_eq!(listing["path"].as_str().unwrap(), missing, "the segments come back in order");

    // The Save-As case: the parent exists, the rest does not — the longest existing ancestor is
    // canonicalized and the tail re-attached.
    let tmp = TempDir::new("newpath");
    let real = canonical(tmp.path()).unwrap();
    let target = tmp.path().join("new/deeper");
    assert_eq!(list(&g, Some(&target.to_string_lossy()))["path"].as_str().unwrap(),
               to_slash(&real.join("new/deeper")));

    // `..` has no `file_name()`, so an ancestor walk would silently drop it and land somewhere
    // other than the directory the user typed.
    let target = tmp.path().join("missing/../also-missing");
    assert_eq!(list(&g, Some(&target.to_string_lossy()))["path"].as_str().unwrap(),
               to_slash(&real.join("also-missing")));
}

#[test]
fn a_non_utf8_entry_name_is_skipped_rather_than_mangled() {
    // A name this platform accepts but UTF-8 cannot express. Lossy-encoding one would emit an entry
    // that cannot be opened, and whose `path` key can collide with another undecodable one.
    let undecodable = {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            std::ffi::OsString::from_vec(b"bad\xff".to_vec()) // 0xFF starts no valid UTF-8 sequence
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            std::ffi::OsString::from_wide(&[0x62, 0x61, 0x64, 0xD800]) // an unpaired high surrogate
        }
    };
    let g = Goofi::new();
    let tmp = TempDir::new("badname");
    tmp.file("fine.txt");
    std::fs::write(tmp.path().join(undecodable), b"x").unwrap();

    assert_eq!(names(&list(&g, Some(&tmp.path().to_string_lossy()))), ["fine.txt"]);
}

#[test]
fn every_root_is_navigable_to_the_exact_same_path_string() {
    // The sidebar marks a root active by raw string equality against `path`, so a root whose
    // spelling differs from what `list_dir` echoes back never highlights.
    let g = Goofi::new();
    let listing = list(&g, None);
    let roots = listing["roots"].as_array().unwrap().clone();

    assert!(roots.iter().any(|r| r["label"] == "Home"), "expected a Home root");
    for root in roots {
        let path = root["path"].as_str().unwrap();
        assert_eq!(list(&g, Some(path))["path"].as_str().unwrap(), path, "root {path} is not stable");
    }
}

#[test]
fn save_and_load_expand_a_tilde_the_way_the_browser_does() {
    // They share the browser's path resolution, so a path a user can navigate to is one they can
    // write to. Proven through a REFUSAL that names the expanded path — the alternative, actually
    // saving into `$HOME`, is not something a test may do.
    let g = Goofi::new();
    let why = g.refuse("load", j!({ "path": "~/definitely-not-a-patch-goofi-wrote.gfi" }));
    assert!(why.contains(&home()), "the refusal names the expanded path, not the tilde: {why}");
}
