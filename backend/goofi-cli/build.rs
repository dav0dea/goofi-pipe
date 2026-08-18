//! Demand that `cargo run -p goofi-init` has been run. The SPA is built and embedded by
//! `goofi-bridge`, which serves it.

use std::path::Path;
use std::process::Command;

fn main() {
    require_python_env();
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
    // The variable, not the file that sets it: cargo has already expanded `[env]` by now, so this
    // re-runs when the interpreter actually changes and not when the config is merely rewritten.
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    let Some(py) = goofi_init::interpreter() else {
        // Not `assert!`/`panic!`: those wrap the one line a developer needs to read in a backtrace
        // preamble naming this file and line, neither of which is where the problem is.
        eprintln!("\n{}\n", goofi_init::RUN_ME);
        std::process::exit(1);
    };
    // The interpreter is known-good now, so its DLLs can be staged beside the executable.
    copy_interpreter_dlls(&py);
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

