//! Keep the served SPA fresh. `goofi-pipe` serves `frontend/build/` from disk, so a stale build
//! is silently served against a newer backend (a real bug once seen: the data-plane URL shape
//! changed and the old bundle 404'd). This script makes `cargo run`/`cargo build` rebuild the
//! frontend when its sources are newer than the last build — the same "rebuild if dirty" contract
//! cargo gives the Rust binary.
//!
//! Mechanism: emit `cargo:rerun-if-changed` for every frontend source path (so cargo re-invokes
//! this script on any edit/add/remove), then run `npm run build` only when `frontend/build/` is
//! older than a source. Degrades gracefully — no frontend tree, no `npm`, or a failed build just
//! prints a `cargo:warning` and leaves the previous build in place; the Rust build never fails.
//!
//! Opt out with `GOOFI_SKIP_FRONTEND_BUILD=1` (CI, offline, or a hand-managed build). Set it in a
//! frontend dev workflow too: cargo only re-runs this script after a frontend edit, but when it
//! does it blocks the cargo command on a full production `npm run build`. And if you point
//! rust-analyzer at a SEPARATE target dir, its `cargo check` and a CLI `cargo run` can both detect
//! staleness and run npm at once (racing `frontend/build/`); the skip var in rust-analyzer's env
//! avoids that.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// Frontend inputs (relative to `frontend/`) whose change should trigger a rebuild. Deliberately
/// NOT `build/`, `.svelte-kit/`, or `node_modules/` — watching those would self-retrigger or be
/// enormous.
const INPUTS: &[&str] = &[
    "src",
    "static",
    "package.json",
    "package-lock.json",
    "svelte.config.js",
    "vite.config.ts",
    "tsconfig.json",
];

fn main() {
    let frontend = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../frontend");
    // No frontend source tree (e.g. the crate vendored alone) → nothing to manage.
    if !frontend.join("src").is_dir() {
        return;
    }
    println!("cargo:rerun-if-env-changed=GOOFI_SKIP_FRONTEND_BUILD");

    // Watch every source path + track the newest source mtime in one walk.
    let mut newest_src: Option<SystemTime> = None;
    for input in INPUTS {
        watch_and_max(&frontend.join(input), &mut newest_src);
    }

    // `build/` is the output — walk it for its newest mtime but do NOT watch it (avoid self-retrigger).
    let built = newest_mtime(&frontend.join("build"));

    let stale = match (newest_src, built) {
        (_, None) => true,                 // no build yet
        (Some(src), Some(out)) => src > out, // a source is newer than the build
        (None, Some(_)) => false,          // nothing to build from
    };
    if !stale {
        return;
    }

    // Only a truthy value opts out — `=0`/empty must NOT skip (a user setting `=0` wants a build).
    if matches!(std::env::var("GOOFI_SKIP_FRONTEND_BUILD").as_deref(), Ok("1") | Ok("true")) {
        println!("cargo:warning=frontend/build is stale but GOOFI_SKIP_FRONTEND_BUILD is set — not rebuilding");
        return;
    }

    // Present tense: cargo replays a build script's `cargo:warning` lines on every build until the
    // script next re-runs, so this can reappear on a no-op build even though npm did not run (a
    // sub-second `Finished` is the tell). Wording stays accurate whether or not the build succeeds.
    println!("cargo:warning=frontend sources changed — rebuilding the served SPA (frontend/build)");
    let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
    match Command::new(npm).args(["run", "build"]).current_dir(&frontend).status() {
        Ok(s) if s.success() => {}
        Ok(s) => println!("cargo:warning=`npm run build` failed ({s}); serving the previous frontend/build"),
        Err(e) => println!("cargo:warning=could not run `npm` ({e}); serving the previous frontend/build"),
    }
}

/// Emit `cargo:rerun-if-changed` for `path` and everything beneath it, folding the max mtime seen
/// into `newest`. A per-path emit (not a bare directory) is robust across cargo versions, which
/// differ in whether a watched directory is scanned recursively.
fn watch_and_max(path: &Path, newest: &mut Option<SystemTime>) {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return;
    };
    println!("cargo:rerun-if-changed={}", path.display());
    if let Ok(t) = meta.modified() {
        if newest.is_none_or(|n| t > n) {
            *newest = Some(t);
        }
    }
    if meta.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                watch_and_max(&entry.path(), newest);
            }
        }
    }
}

/// Newest modification time of `path` (a file) or anything under it (a dir), or `None` if absent.
fn newest_mtime(path: &Path) -> Option<SystemTime> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if !meta.is_dir() {
        return meta.modified().ok();
    }
    let mut newest = meta.modified().ok();
    for entry in std::fs::read_dir(path).ok()?.flatten() {
        if let Some(t) = newest_mtime(&entry.path()) {
            if newest.is_none_or(|n| t > n) {
                newest = Some(t);
            }
        }
    }
    newest
}
