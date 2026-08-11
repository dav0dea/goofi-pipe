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
    let venv = root.join(".gfivenv-ft");
    let config = root.join(".cargo").join("config.toml");
    println!("cargo:rerun-if-changed={}", config.display());
    println!("cargo:rerun-if-env-changed=GOOFI_SKIP_PY_SETUP");
    // A venv keeps its interpreter under `bin/` on unix and `Scripts/` on Windows. Ask which is
    // actually there rather than which platform this is: the old `bin/python` guess made every
    // step below silently no-op on Windows, `.cargo/config.toml` generation included.
    fn interpreter(venv: &Path) -> Option<PathBuf> {
        [venv.join("bin").join("python"), venv.join("Scripts").join("python.exe")]
            .into_iter()
            .find(|p| p.is_file())
    }
    // The PYO3_PYTHON the config MUST carry: the venv symlink (dir-resolved, symlink leaf kept).
    // Computed here so the early-return can be content-aware — an existing config left by an older
    // generator (e.g. a canonicalized BASE-interpreter path) would, combined with the probe's
    // PYTHONPATH strip, silently grey out all Python-node discovery, and an existence-only guard
    // would never rewrite it.
    let expected = |py: &Path| {
        let full = std::fs::canonicalize(&venv)
            .ok()
            .and_then(|real| py.strip_prefix(&venv).ok().map(|tail| real.join(tail)))
            .unwrap_or_else(|| py.to_path_buf())
            .display()
            .to_string();
        // Drop Windows' extended-length prefix: `canonicalize` adds it, everything downstream
        // (pyo3, a spawned probe, a human reading the config) wants the ordinary path, and on unix
        // the prefix never appears so the strip is simply a no-op.
        full.strip_prefix(r"\\?\").unwrap_or(&full).to_string()
    };
    if let Some(py) = interpreter(&venv) {
        // `cargo clean` empties the profile dir but leaves the config, so the DLLs are restored
        // before the guard decides — and the guard needs the answer anyway.
        let relocated = copy_interpreter_dlls(&py);
        if config_is_current(&config, &expected(&py), &home_line(&py, relocated)) {
            return;
        }
    }
    if matches!(std::env::var("GOOFI_SKIP_PY_SETUP").as_deref(), Ok("1") | Ok("true")) {
        return;
    }

    // Create the venv (from PYO3_PYTHON if set, else `python3.14t` on PATH) + numpy. `uv` is a
    // HARD dependency, so a missing one fails the build here rather than warning: the alternative
    // is a binary that starts happily, greys out every Python node, and blames Python for it.
    if interpreter(&venv).is_none() {
        // `PYO3_PYTHON` seeds the venv only when it names an interpreter that still EXISTS. After a
        // successful provision that variable points at `.gfivenv-ft`'s own python — written into
        // `.cargo/config.toml` by this very function — so once the venv is deleted, feeding it back
        // asks uv to build the venv out of the venv it is here to recreate. Deleting the venv to
        // watch it come back is exactly when that happens, and it fails claiming uv is missing.
        let ft = std::env::var("PYO3_PYTHON")
            .ok()
            .filter(|p| !p.is_empty() && Path::new(p).exists())
            .unwrap_or_else(|| "python3.14t".into());
        let made = Command::new("uv").args(["venv", "--python", &ft]).arg(&venv).status().map(|s| s.success()).unwrap_or(false);
        assert!(
            made,
            "goofi needs `uv` and a free-threaded interpreter to provision {} (try `uv python \
             install 3.14t`). Install uv from https://docs.astral.sh/uv/, or build a native-only \
             binary with --no-default-features.",
            venv.display()
        );
        let py = interpreter(&venv)
            .unwrap_or_else(|| panic!("`uv venv` left no interpreter in {}", venv.display()));
        let _ = Command::new("uv").args(["pip", "install", "--python"]).arg(&py).arg("numpy").status();
    }
    let py = interpreter(&venv)
        .unwrap_or_else(|| panic!("`uv venv` left no interpreter in {}", venv.display()));
    let expected_py = expected(&py);

    // Query the interpreter for the paths the config needs.
    let Some(purelib) = query(&py, "import sysconfig;print(sysconfig.get_path('purelib'))") else {
        println!("cargo:warning=couldn't query the .gfivenv-ft interpreter; skipping .cargo/config.toml generation");
        return;
    };
    // An rpath is a GNU/Clang *driver* flag, and `link.exe` rejects it — so this is keyed on the
    // target's linker flavour, not on whether the interpreter has a libdir to name. A Windows
    // CPython reports one (`…\libs`) and still cannot use it, which is exactly the trap: keying on
    // the libdir looks equivalent and is not.
    let target = std::env::var("TARGET").unwrap_or_default();
    let rpath = query(&py, "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
        .filter(|d| d != "None" && !target.ends_with("windows-msvc"))
        .map(|libdir| {
            // Debug-quoted as ONE flag string. A raw interpolation of a Windows libdir makes `\U`
            // an invalid TOML escape, and cargo reports that as "could not load Cargo
            // configuration" — a broken build with nothing pointing at the line that broke it.
            let flag = format!("link-arg=-Wl,-rpath,{libdir}");
            format!("\n[target.{target}]\nrustflags = [\"-C\", {flag:?}]\n")
        })
        .unwrap_or_default();
    // Relocating the DLL is precisely what makes this necessary, so the two are decided together.
    // Windows CPython derives `sys.prefix` from the directory it loaded `python3XX.dll` FROM, and
    // that directory is now `target/<profile>/`, which holds no `Lib/`. The interpreter then comes
    // up far enough to report `Failed to import encodings module` and dies. Naming the home it can
    // no longer infer is the other half of the copy. Where nothing was relocated — unix — the
    // interpreter still knows its own prefix and this stays absent rather than overriding it.
    let home = home_line(&py, copy_interpreter_dlls(&py));
    // The venv `python` SYMLINK (NOT the canonicalized base — see `expected` above): a spawned
    // probe/subprocess child must run the VENV so it self-detects the venv site-packages.
    let py_str = expected_py;
    // `{x:?}` debug-quotes each path into a valid TOML string literal.
    let contents = format!(
        "# Generated by goofi-cli/build.rs — machine-specific, gitignored. Points pyo3 at the\n\
         # repo-local free-threaded 3.14t venv so `cargo run`/`test` need no env vars. Delete this\n\
         # file (and .gfivenv-ft) to reprovision. Editing PYO3_PYTHON does NOT stick: build.rs\n\
         # rewrites this file whenever that line diverges from the .gfivenv-ft it manages. To own\n\
         # the interpreter yourself, build with GOOFI_SKIP_PY_SETUP=1 — that is the escape hatch.\n\
         [env]\n\
         PYO3_PYTHON = {py_str:?}\n\
         PYTHONPATH = {purelib:?}\n\
         {home}\
         {rpath}",
    );
    if let Some(parent) = config.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&config, contents).is_ok() {
        println!(
            "cargo:warning=provisioned .gfivenv-ft + .cargo/config.toml for the free-threaded \
             interpreter — RE-RUN your cargo command so it links against it (cargo reads \
             .cargo/config.toml only at startup)"
        );
    }
}

/// The Windows half of the rpath above, and the reason it is not written as a platform branch:
/// Windows has no rpath at all. Its loader searches the executable's own directory, the system
/// directories and `PATH` — and a uv-managed interpreter is on none of them, so the binary dies
/// with `STATUS_DLL_NOT_FOUND` before `main` ever runs, which surfaces as a bare exit code and no
/// message whatsoever. Beside the executable is the one place a build script can put the DLL that
/// the loader is guaranteed to look. A unix interpreter simply has no `python*.dll` to copy, so
/// this is a no-op there rather than a conditional.
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

/// The `PYTHONHOME` line the config must carry, or empty where nothing was relocated and the
/// interpreter can still work out its own prefix. Shared by the generator and the guard below, so
/// the line that gets written is by construction the line that gets checked for.
fn home_line(py: &Path, relocated: bool) -> String {
    relocated
        .then(|| query(py, "import sys;print(sys.base_prefix)"))
        .flatten()
        .map(|h| format!("PYTHONHOME = {h:?}\n"))
        .unwrap_or_default()
}

/// Whether `config` already says everything this generator would say — the self-heal guard, so
/// build.rs rewrites a config left by an OLDER generator instead of trusting mere existence.
/// Checking only `PYO3_PYTHON` was not enough: a config written before `PYTHONHOME` existed named
/// the right interpreter and still left it unable to find its own stdlib, and an existence-only
/// guard would have kept that forever.
fn config_is_current(config: &Path, expected_py: &str, home: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(config) else { return false };
    let has = |want: &str| text.lines().any(|l| l.trim() == want.trim());
    has(&format!("PYO3_PYTHON = {expected_py:?}")) && (home.is_empty() || has(home))
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
