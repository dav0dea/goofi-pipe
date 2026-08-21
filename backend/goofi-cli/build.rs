//! Demand that `cargo run -p goofi-init` has been run.

use std::path::Path;
use std::process::Command;

fn main() {
    require_python_env();
}

/// Demand that `cargo run -p goofi-init` has been run, and stage the interpreter's DLLs. Only
/// under the `python` feature; a `--no-default-features` build embeds no interpreter.
fn require_python_env() {
    if std::env::var_os("CARGO_FEATURE_PYTHON").is_none() {
        return;
    }
    // The variable, not the file that sets it: cargo has already expanded `[env]` by now.
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    let Some(py) = goofi_init::interpreter() else {
        // Not `panic!`: its backtrace preamble buries the one line a developer needs to read.
        eprintln!("\n{}\n", goofi_init::RUN_ME);
        std::process::exit(1);
    };
    copy_interpreter_dlls(&py);
}

/// Stage the interpreter's `python*.dll` beside the executable: Windows' loader searches there and
/// never a uv-managed venv. A no-op on unix, which has no such DLL.
fn copy_interpreter_dlls(py: &Path) -> bool {
    let (Some(base), Some(out)) = (query(py, "import sys;print(sys.base_prefix)"), std::env::var_os("OUT_DIR"))
    else {
        return false;
    };
    // OUT_DIR is `<target>/<profile>/build/<pkg>-<hash>/out`; three levels up is `<target>/<profile>`.
    let Some(profile_dir) = Path::new(&out).ancestors().nth(3) else { return false };
    let Ok(entries) = std::fs::read_dir(&base) else { return false };
    let mut relocated = false;
    for dll in entries.flatten().map(|e| e.path()).filter(|p| {
        p.extension().is_some_and(|x| x.eq_ignore_ascii_case("dll"))
            && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("python"))
    }) {
        let Some(dest) = dll.file_name().map(|n| profile_dir.join(n)) else { continue };
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

