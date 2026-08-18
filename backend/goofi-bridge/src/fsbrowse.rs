//! Filesystem browsing for the Save/Load modal.
//!
//! Deliberately **unjailed** — this is a single-user, local/trusted-LAN app and the user
//! saves patches wherever they like. Nothing here touches graph state, so `dispatch` serves
//! it without the graph mutex — a slow or huge directory would otherwise hold that mutex, and
//! every graph EDIT waits on it.
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
    if let Ok(real) = goofi_core::path::canonical(&abs) {
        return real;
    }
    // Not on disk, so resolve `.`/`..` ourselves before walking ancestors: `Path::file_name()`
    // is None for `..`, which would drop the component silently and land somewhere else entirely.
    let abs = lexical(&abs);
    let mut tail = Vec::new();
    let mut cur = abs.as_path();
    while let Some(parent) = cur.parent() {
        tail.push(cur.file_name().unwrap_or_default().to_os_string());
        if let Ok(mut real) = goofi_core::path::canonical(parent) {
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

/// The one place a browsed path becomes a string for the client. Every `path`, `parent` and root
/// in a listing goes through here, which is what makes goofi's spelling — `/`, on every platform —
/// something the sidebar's raw string-equality highlight can rely on.
fn display(path: &Path) -> String {
    goofi_core::path::to_slash(path)
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
