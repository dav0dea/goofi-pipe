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
    require_python_env();
    sync_frontend();
}

/// Rebuild the served SPA when its sources are newer than the last build (see the module doc).
fn sync_frontend() {
    let frontend = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../frontend");
    // No frontend source tree (e.g. the crate vendored alone) → nothing to manage.
    if !frontend.join("src").is_dir() {
        return;
    }
    println!("cargo:rerun-if-env-changed=GOOFI_SKIP_FRONTEND_BUILD");

    // Watch every source path + track the newest source mtime in one walk.
    let mut newest_src: Option<SystemTime> = None;
    for input in INPUTS {
        if let Some(t) = newest_mtime(&frontend.join(input), true) {
            if newest_src.is_none_or(|n| t > n) {
                newest_src = Some(t);
            }
        }
    }

    // `build/` is the output — walk it for its newest mtime but do NOT watch it (avoid self-retrigger).
    let built = newest_mtime(&frontend.join("build"), false);

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

    // Report AFTER the build, in the PAST tense with the measured duration. Cargo caches a build
    // script's `cargo:warning` lines and REPLAYS them on every later build until the script next
    // re-runs — so a present-tense "rebuilding…" reads as a fresh claim on a no-op build where npm
    // never ran (the confusing case: the line appears, then cargo finishes in milliseconds). Past
    // tense + a duration describe a completed event, so the line stays true when it is replayed.
    // Nothing is printed before the build because cargo captures build-script output and only shows
    // it once the script finishes — a pre-announcement could not act as live progress anyway.
    let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
    let started = SystemTime::now();
    match Command::new(npm).args(["run", "build"]).current_dir(&frontend).status() {
        Ok(s) if s.success() => {
            let secs = started.elapsed().map(|d| d.as_secs_f32()).unwrap_or(0.0);
            println!(
                "cargo:warning=rebuilt the served SPA (frontend/build) from changed sources in \
                 {secs:.1}s — cargo REPLAYS this line on later no-op builds, where npm did not re-run"
            );
        }
        Ok(s) => println!("cargo:warning=`npm run build` failed ({s}); serving the previous frontend/build"),
        Err(e) => println!("cargo:warning=could not run `npm` ({e}); serving the previous frontend/build"),
    }
}

/// Demand that `cargo run -p goofi-init` has been run, and place the interpreter's DLLs where the
/// loader will find them. This script no longer PROVISIONS anything: cargo reads
/// `.cargo/config.toml` once at startup, so a config written from here could never reach the build
/// it is part of — which is what used to make a fresh clone need two `cargo run`s and, on a machine
/// with no Python on `PATH`, fail with a bare pyo3 error instead. Setup is one explicit command now,
/// and this is the check that says so.
///
/// Only when the `python` feature is on (cargo sets `CARGO_FEATURE_PYTHON`); a
/// `--no-default-features` build embeds no interpreter and needs none.
fn require_python_env() {
    if std::env::var_os("CARGO_FEATURE_PYTHON").is_none() {
        return;
    }
    let root = goofi_init::repo_root();
    println!("cargo:rerun-if-changed={}", goofi_init::config_path(&root).display());
    assert!(goofi_init::ready(&root), "{}", goofi_init::RUN_ME);
    // The interpreter is known-good now, so its DLLs can be staged beside the executable.
    if let Some(py) = goofi_init::venv_python(&root.join(goofi_init::FT_VENV)) {
        copy_interpreter_dlls(&py);
    }
}

/// Windows has no rpath: its loader searches the executable's own directory, the system
/// directories and `PATH` — and a uv-managed interpreter is on none of them, so the binary dies
/// with `STATUS_DLL_NOT_FOUND` before `main` ever runs, surfacing as a bare exit code and no
/// message at all. Beside the executable is the one place a build script can write that the loader
/// is guaranteed to search. A unix interpreter has no `python*.dll` to copy, so this is a no-op
/// there rather than a platform branch.
fn copy_interpreter_dlls(py: &Path) -> bool {
    let (Some(base), Some(out)) = (query(py, "import sys;print(sys.base_prefix)"), std::env::var_os("OUT_DIR"))
    else {
        return false;
    };
    // OUT_DIR is `<target>/<profile>/build/<pkg>-<hash>/out`; three levels up is
    // `<target>/<profile>`, where cargo will place the executable this DLL has to sit beside.
    let Some(profile_dir) = Path::new(&out).ancestors().nth(3) else { return false };
    let Ok(entries) = std::fs::read_dir(&base) else { return false };
    let mut relocated = false;
    for dll in entries.flatten().map(|e| e.path()).filter(|p| {
        p.extension().is_some_and(|x| x.eq_ignore_ascii_case("dll"))
            && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("python"))
    }) {
        let Some(dest) = dll.file_name().map(|n| profile_dir.join(n)) else { continue };
        // Same name and size already there ⇒ leave it; this runs on every build.
        let same = std::fs::metadata(&dest)
            .ok()
            .zip(std::fs::metadata(&dll).ok())
            .is_some_and(|(a, b)| a.len() == b.len());
        if same || std::fs::copy(&dll, &dest).is_ok() {
            relocated = true;
        }
    }
    relocated
}

fn query(py: &Path, code: &str) -> Option<String> {
    let out = Command::new(py).args(["-c", code]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Newest modification time of `path` (a file) or anything under it (a dir), or `None` if absent.
/// When `watch`, also emits `cargo:rerun-if-changed` for every path visited — a per-path emit (not a
/// bare directory) is robust across cargo versions, which differ on whether a watched directory is
/// scanned recursively. Sources pass `watch = true`; the `build/` output passes `false` (watching it
/// would self-retrigger).
fn newest_mtime(path: &Path, watch: bool) -> Option<SystemTime> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if watch {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let mut newest = meta.modified().ok();
    if meta.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Some(t) = newest_mtime(&entry.path(), watch) {
                    if newest.is_none_or(|n| t > n) {
                        newest = Some(t);
                    }
                }
            }
        }
    }
    newest
}
