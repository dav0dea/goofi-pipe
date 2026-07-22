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
    provision_python();
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

/// Provision the free-threaded Python the `python` feature needs, so `cargo run` works with NO env
/// vars. Runs only when the feature is on (cargo sets `CARGO_FEATURE_PYTHON`). Idempotent: a present
/// `.ftvenv` + `.cargo/config.toml` is a no-op. Otherwise it creates a uv venv on a free-threaded
/// interpreter + numpy and writes `.cargo/config.toml` pointing pyo3 at it — `PYO3_PYTHON` (link
/// target), an rpath to the interpreter's libdir (replaces `LD_LIBRARY_PATH`), and the venv
/// site-packages on `PYTHONPATH` (numpy). Both are machine-specific ⇒ gitignored. Cargo reads config
/// at startup, so a freshly-written one applies on the NEXT cargo command (the warning says so).
/// Degrades gracefully: no `uv` / no free-threaded interpreter only warns — the build still succeeds
/// (pyo3 links whatever it auto-detected; expressions just stay unavailable).
fn provision_python() {
    if std::env::var_os("CARGO_FEATURE_PYTHON").is_none() {
        return; // built with --no-default-features (pure native, no interpreter needed)
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let venv = root.join(".ftvenv");
    let config = root.join(".cargo").join("config.toml");
    println!("cargo:rerun-if-changed={}", config.display());
    println!("cargo:rerun-if-env-changed=GOOFI_SKIP_PY_SETUP");
    let py = venv.join("bin").join("python");
    if py.exists() && config.is_file() {
        warn_if_goofi_missing(&py); // provisioned, but goofi may still be uninstalled
        return;
    }
    if matches!(std::env::var("GOOFI_SKIP_PY_SETUP").as_deref(), Ok("1") | Ok("true")) {
        return;
    }

    // Create the venv (from PYO3_PYTHON if set, else `python3.14t` on PATH) + numpy.
    if !py.exists() {
        let ft = std::env::var("PYO3_PYTHON").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| "python3.14t".into());
        let made = Command::new("uv").args(["venv", "--python", &ft]).arg(&venv).status().map(|s| s.success()).unwrap_or(false);
        if !made {
            println!(
                "cargo:warning=`python` feature is ON but couldn't provision {} — need `uv` + a \
                 free-threaded interpreter (e.g. `uv python install 3.14t`). Expressions + Python \
                 nodes stay unavailable; build a native-only binary with --no-default-features.",
                venv.display()
            );
            return;
        }
        let _ = Command::new("uv").args(["pip", "install", "--python"]).arg(&py).arg("numpy").status();
    }

    // Query the interpreter for the paths the config needs.
    let (Some(libdir), Some(purelib)) = (
        query(&py, "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"),
        query(&py, "import sysconfig;print(sysconfig.get_path('purelib'))"),
    ) else {
        println!("cargo:warning=couldn't query the .ftvenv interpreter; skipping .cargo/config.toml generation");
        return;
    };
    // Keep the venv `python` SYMLINK (do NOT `canonicalize` it — that would deref to the base
    // uv interpreter). A spawned probe/subprocess child must run the VENV so it self-detects the
    // venv's site-packages (goofi/numpy); the base interpreter doesn't see them. Resolve `..` via
    // the venv dir, but preserve the `python` symlink leaf. pyo3 still links fine (the symlink
    // python reports the shared base libdir, which the rpath below points at).
    let py_str = std::fs::canonicalize(&venv)
        .map(|v| v.join("bin").join("python"))
        .unwrap_or_else(|_| py.clone())
        .display()
        .to_string();
    let target = std::env::var("TARGET").unwrap_or_default();
    // `{x:?}` debug-quotes each path into a valid TOML string literal.
    let contents = format!(
        "# Generated by goofi-cli/build.rs — machine-specific, gitignored. Points pyo3 at the\n\
         # repo-local free-threaded 3.14t venv so `cargo run`/`test` need no env vars. Delete this\n\
         # file (and .ftvenv) to reprovision; edit freely if you manage the interpreter yourself.\n\
         [env]\n\
         PYO3_PYTHON = {py_str:?}\n\
         PYTHONPATH = {purelib:?}\n\
         \n\
         [target.{target}]\n\
         rustflags = [\"-C\", \"link-arg=-Wl,-rpath,{libdir}\"]\n",
    );
    if let Some(parent) = config.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&config, contents).is_ok() {
        println!(
            "cargo:warning=provisioned .ftvenv + .cargo/config.toml for the free-threaded \
             interpreter — RE-RUN your cargo command so it links against it (cargo reads \
             .cargo/config.toml only at startup)"
        );
    }
    warn_if_goofi_missing(&py); // a freshly-made venv has numpy but not yet goofi
}

/// The `goofi` package (the goofi-pymod wheel) must be importable in `.ftvenv` for the
/// introspection PROBE to discover Python nodes (`--python-nodes`/`--auto-nodes`). build.rs
/// can't build it — maturin invokes cargo, which would deadlock nested under this build
/// script — so on a missing package it points at the standalone provisioning script instead
/// of letting node discovery silently grey out.
fn warn_if_goofi_missing(py: &Path) {
    if query(py, "import goofi; print('ok')").as_deref() != Some("ok") {
        println!(
            "cargo:warning=the `goofi` Python package is not importable in .ftvenv — Python \
             node discovery (--python-nodes/--auto-nodes) will grey out. Run \
             ./scripts/provision-goofi-py.sh to build + install it into the venvs."
        );
    }
}

/// Run `py -c code`, returning trimmed stdout, or `None` on any failure.
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
