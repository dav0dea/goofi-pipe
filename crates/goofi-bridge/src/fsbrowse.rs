//! Filesystem browsing for the Save/Load modal.
//!
//! Deliberately **unjailed** — this is a single-user, local/trusted-LAN app and the user
//! saves patches wherever they like. Nothing here touches graph state, so `dispatch` serves
//! it without the graph mutex (a slow or huge directory would otherwise stall the tick).
//!
//! Navigation must never fail: an unreadable or not-yet-existing directory lists *empty*
//! rather than erroring, because the browser keeps the previous directory's entries on an
//! error, which reads as "the click did nothing".

use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};

/// One directory level, shaped as the frontend's `DirListing`.
pub fn list_dir(path: Option<&str>) -> Value {
    let base = base_dir(path);
    let parent = base.parent().filter(|p| *p != base).map(display);
    json!({
        "path": display(&base),
        "parent": parent,
        "entries": entries(&base),
        "roots": roots(),
    })
}

/// Interpret a user-supplied path the same way the browser does — `~` expanded, made absolute
/// and symlink-free — so a path that can be navigated to is also one that can be saved to or
/// loaded from. Shared with the `save` / `load` arms; without it those take `~/patches` literally
/// and create (or fail to find) a directory named `~`.
pub fn resolve(path: &str) -> String {
    display(&normalize(&expand_tilde(path)))
}

/// The directory a request lands in: `~` expanded, made absolute and symlink-free, and
/// stepped up to the parent when the path names a file (the browser navigates directories,
/// but a remembered save path points at the `.gfi` itself).
fn base_dir(path: Option<&str>) -> PathBuf {
    let base = match path.map(str::trim).filter(|p| !p.is_empty()) {
        Some(p) => normalize(&expand_tilde(p)),
        None => home(),
    };
    if base.is_file() {
        if let Some(parent) = base.parent() {
            return parent.to_path_buf();
        }
    }
    base
}

fn expand_tilde(path: &str) -> PathBuf {
    match path.strip_prefix('~') {
        Some(rest) => home().join(rest.trim_start_matches('/')),
        None => PathBuf::from(path),
    }
}

/// Absolute and symlink-free. A path that is not on disk yet (a fresh Save-As target) still
/// normalizes: canonicalize its longest existing ancestor and re-attach the rest, so the
/// result stays comparable with the roots — the sidebar highlights by string equality.
fn normalize(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")).join(path)
    };
    if let Ok(real) = abs.canonicalize() {
        return real;
    }
    // Not on disk, so resolve `.`/`..` ourselves before walking ancestors: `Path::file_name()`
    // is None for `..`, which would drop the component silently and land somewhere else entirely.
    let abs = lexical(&abs);
    let mut tail = Vec::new();
    let mut cur = abs.as_path();
    while let Some(parent) = cur.parent() {
        tail.push(cur.file_name().unwrap_or_default().to_os_string());
        if let Ok(mut real) = parent.canonicalize() {
            real.extend(tail.iter().rev());
            return real;
        }
        cur = parent;
    }
    abs
}

/// Resolve `.` and `..` textually. Only used for paths that are NOT on disk — an existing path
/// goes through `canonicalize`, which resolves them symlink-aware.
fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn home() -> PathBuf {
    // No HOME is survivable — fall back to the working directory so the modal still opens.
    std::env::home_dir()
        .map(|h| normalize(&h))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")))
}

fn display(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// The sidebar shortcuts. Normalized through the same function as `path`, or the "active"
/// highlight (raw string equality) would never fire.
fn roots() -> Value {
    let mut out = vec![json!({ "label": "Home", "path": display(&home()) })];
    if let Ok(cwd) = std::env::current_dir() {
        let cwd = normalize(&cwd);
        if cwd != home() {
            out.push(json!({ "label": "Working dir", "path": display(&cwd) }));
        }
    }
    Value::Array(out)
}

/// Directories first, then case-insensitive by name — the ordering is the server's job, the
/// browser renders the array as given.
fn entries(base: &Path) -> Value {
    let Ok(read) = std::fs::read_dir(base) else {
        return Value::Array(Vec::new());
    };
    let mut rows: Vec<(bool, String, Value)> = Vec::new();
    for entry in read.flatten() {
        let path = entry.path();
        // `metadata()` follows symlinks (unlike `entry.file_type()`), so a link to a directory
        // browses as one. A broken link or an unreadable child errors here and is skipped.
        let Ok(meta) = path.metadata() else { continue };
        // A name that is not valid UTF-8 cannot survive the JSON round trip: lossy-encoding it
        // yields an entry the browser can neither open nor key uniquely (two different names can
        // collapse to the same replacement string). Skip it rather than show something broken.
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else { continue };
        let is_dir = meta.is_dir();
        let row = json!({
            "name": name,
            "path": display(&path),
            "kind": if is_dir { "dir" } else { "file" },
            "is_gfi": path.extension().is_some_and(|e| e == "gfi"),
            "hidden": name.starts_with('.'),
        });
        rows.push((!is_dir, name.to_lowercase(), row));
    }
    rows.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    Value::Array(rows.into_iter().map(|(_, _, row)| row).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique directory removed on drop. The workspace carries no `tempfile` dependency
    /// and one test-only fixture does not justify adding it.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("goofi-fsbrowse-{}-{tag}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn file(&self, name: &str) -> PathBuf {
            let p = self.0.join(name);
            std::fs::write(&p, b"x").unwrap();
            p
        }
        fn dir(&self, name: &str) -> PathBuf {
            let p = self.0.join(name);
            std::fs::create_dir_all(&p).unwrap();
            p
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

    fn names(listing: &Value) -> Vec<String> {
        listing["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn lists_directories_before_files_then_case_insensitively_by_name() {
        let tmp = TempDir::new("order");
        tmp.file("Beta.txt");
        tmp.file("alpha.txt");
        tmp.dir("Zeta");
        tmp.dir("apples");

        let listing = list_dir(Some(&tmp.path().to_string_lossy()));

        assert_eq!(names(&listing), ["apples", "Zeta", "alpha.txt", "Beta.txt"]);
    }

    #[test]
    fn a_file_path_lists_its_parent_directory() {
        let tmp = TempDir::new("file");
        let patch = tmp.file("patch.gfi");

        let listing = list_dir(Some(&patch.to_string_lossy()));

        // Compare against a literal built from the temp dir itself, NOT against `normalize(..)`
        // — asserting with the function under test on both sides would accept, for example, a
        // normalize that reversed the path segments it re-attaches.
        let expected = std::fs::canonicalize(tmp.path()).unwrap().to_string_lossy().into_owned();
        assert_eq!(listing["path"].as_str().unwrap(), expected);
        assert_eq!(names(&listing), ["patch.gfi"]);
    }

    #[test]
    fn parent_is_null_at_the_filesystem_root() {
        let listing = list_dir(Some("/"));

        assert_eq!(listing["path"].as_str().unwrap(), "/");
        assert!(listing["parent"].is_null(), "root has no parent to climb to");
    }

    #[test]
    fn no_path_lists_the_home_directory() {
        let listing = list_dir(None);

        assert_eq!(listing["path"].as_str().unwrap(), display(&home()));
    }

    #[test]
    fn an_empty_path_lists_home_rather_than_the_working_directory() {
        // The frontend omits `path` on the first open, but a cleared path input sends "".
        assert_eq!(list_dir(Some("")), list_dir(None));
    }

    #[test]
    fn expands_a_leading_tilde() {
        let listing = list_dir(Some("~"));

        assert_eq!(listing["path"].as_str().unwrap(), display(&home()));
    }

    #[test]
    fn flags_hidden_files_and_gfi_patches() {
        let tmp = TempDir::new("flags");
        tmp.file(".hidden");
        tmp.file("patch.gfi");
        tmp.file("notes.txt");

        let listing = list_dir(Some(&tmp.path().to_string_lossy()));
        let by_name = |n: &str| {
            listing["entries"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["name"] == n)
                .unwrap()
                .clone()
        };

        // Hidden entries are EMITTED, not filtered — the browser owns that toggle.
        assert_eq!(by_name(".hidden")["hidden"], json!(true));
        assert_eq!(by_name("patch.gfi")["hidden"], json!(false));
        assert_eq!(by_name("patch.gfi")["is_gfi"], json!(true));
        assert_eq!(by_name("notes.txt")["is_gfi"], json!(false));
    }

    #[test]
    fn every_entry_carries_the_full_shape() {
        let tmp = TempDir::new("shape");
        tmp.file("a.txt");

        let listing = list_dir(Some(&tmp.path().to_string_lossy()));
        let entry = &listing["entries"][0];

        for key in ["name", "path", "kind", "is_gfi", "hidden"] {
            assert!(entry.get(key).is_some(), "entry is missing `{key}`");
        }
        assert_eq!(entry["kind"], json!("file"));
        assert_eq!(entry["path"], json!(display(&normalize(&tmp.path().join("a.txt")))));
        // The browser renders neither a size nor a date column, so the row carries neither.
        // Re-add them together with the column that shows them.
        for key in ["size", "mtime"] {
            assert!(entry.get(key).is_none(), "entry carries an unrendered `{key}`");
        }
    }

    #[test]
    fn an_unreadable_directory_lists_empty_instead_of_failing() {
        // Navigation must not error: the browser would keep showing the previous directory.
        let listing = list_dir(Some("/definitely/not/a/directory"));

        assert_eq!(names(&listing), Vec::<String>::new());
        // The path echoes back EXACTLY, in order — this is the only test that reaches
        // `normalize`'s not-on-disk branch, so it has to pin the reassembled segments rather
        // than settle for "looks absolute".
        assert_eq!(listing["path"].as_str().unwrap(), "/definitely/not/a/directory");
    }

    #[test]
    fn a_not_yet_existing_path_under_a_real_directory_keeps_its_segments() {
        // The Save-As case: the parent exists, the rest does not. `normalize` canonicalizes the
        // longest existing ancestor and re-attaches the tail — in the right order.
        let tmp = TempDir::new("newpath");
        let real = std::fs::canonicalize(tmp.path()).unwrap();
        let target = tmp.path().join("new/deeper");

        let listing = list_dir(Some(&target.to_string_lossy()));

        assert_eq!(listing["path"].as_str().unwrap(), real.join("new/deeper").to_string_lossy());
    }

    #[test]
    fn a_parent_dir_component_resolves_instead_of_vanishing() {
        // `..` has no `file_name()`, so the ancestor walk would silently drop it and land in a
        // different directory than the user typed.
        let tmp = TempDir::new("dotdot");
        let real = std::fs::canonicalize(tmp.path()).unwrap();
        let target = tmp.path().join("missing/../also-missing");

        let listing = list_dir(Some(&target.to_string_lossy()));

        assert_eq!(listing["path"].as_str().unwrap(), real.join("also-missing").to_string_lossy());
    }

    #[test]
    fn a_non_utf8_entry_name_is_skipped_rather_than_mangled() {
        use std::os::unix::ffi::OsStrExt;
        let tmp = TempDir::new("badname");
        tmp.file("fine.txt");
        // 0xFF is not valid UTF-8; lossy-encoding it would emit an entry that cannot be opened
        // and whose `path` key can collide with another undecodable name.
        let bad = tmp.path().join(std::ffi::OsStr::from_bytes(b"bad\xff"));
        std::fs::write(&bad, b"x").unwrap();

        let listing = list_dir(Some(&tmp.path().to_string_lossy()));

        assert_eq!(names(&listing), ["fine.txt"]);
    }

    #[test]
    fn resolve_expands_a_tilde_for_the_save_and_load_arms() {
        // `save`/`load` share this with the browser, so a path the user can navigate to is a
        // path they can write to. Compared against the raw HOME string, not against `home()`.
        let home = std::env::var("HOME").expect("HOME is set in the test environment");
        assert_eq!(resolve("~/patches/x.gfi"), format!("{home}/patches/x.gfi"));
        assert_eq!(resolve("/tmp"), "/tmp", "an absolute path is untouched");
    }

    #[test]
    fn roots_are_navigable_to_the_exact_same_path_string() {
        // The sidebar marks a root active by raw string equality against `path`, so a root
        // whose spelling differs from what `list_dir` echoes back never highlights.
        let listing = list_dir(None);
        let roots = listing["roots"].as_array().unwrap().clone();

        assert!(roots.iter().any(|r| r["label"] == "Home"), "expected a Home root");
        for root in roots {
            let path = root["path"].as_str().unwrap();
            assert_eq!(list_dir(Some(path))["path"].as_str().unwrap(), path, "root {path} is not stable");
        }
    }
}
