//! Setting up the Python goofi runs its nodes on — the ONE place that does it.
//!
//! **Why this is a crate and not a build script.** pyo3 has to be told which interpreter to link
//! against *before cargo starts*, because it reads `PYO3_PYTHON` from the environment and cargo
//! reads `.cargo/config.toml` exactly once, at startup. A build script writing that file cannot
//! reach the build it is part of, so provisioning from `build.rs` means the first `cargo build`
//! links against whatever interpreter happened to be on `PATH` — or fails outright on a machine
//! with none. Running it as an ordinary binary first is the only ordering that works.
//!
//! **Why Rust and not a shell script.** `cargo run -p goofi-init` is one command in PowerShell,
//! cmd, bash, zsh and fish alike. A `.sh` would need Git Bash on Windows, and a `.sh`/`.ps1` pair
//! would be two sources of truth for one procedure.
//!
//! `-p goofi-init` builds this crate alone — it depends on no goofi crate and no pyo3, so it can
//! never trigger the very build it exists to configure.
//!
//! After it runs, `cargo build`, `cargo test` and `cargo run` all work first time, every time.
//! `uv` is the one thing that must already be installed.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The GIL venv the subprocess tier runs on, and the interpreter it is built from. Pinned, and
/// pinned to a *non*-free-threaded version on purpose: that tier exists precisely for packages
/// that are not free-threading-safe, and `uv`'s own pick on a machine whose only installed
/// interpreter is 3.14t would hand back the very thing the tier exists to avoid.
pub const GIL_VENV: &str = ".gfivenv";
const GIL_PYTHON: &str = "3.12";

/// The free-threaded venv pyo3 LINKS against, and which the introspection probe runs on.
pub const FT_VENV: &str = ".gfivenv-ft";
const FT_PYTHON: &str = "3.14t";

/// The repo root, canonical and without Windows' extended-length prefix — so what this crate
/// prints, and what it hands to uv and maturin, reads like a path a person would type rather than
/// `backend\goofi-init\../..\.gfivenv-ft`.
pub fn repo_root() -> PathBuf {
    let raw = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    PathBuf::from(slashless(&raw))
}

/// A venv's interpreter, or `None` when the venv is not there. Both layouts are candidates —
/// `bin/` where venvs put it, `Scripts/` where Windows does — so this asks which is actually
/// present rather than which platform this is.
pub fn venv_python(venv: &Path) -> Option<PathBuf> {
    [venv.join("bin").join("python"), venv.join("Scripts").join("python.exe")]
        .into_iter()
        .find(|p| p.is_file())
}

/// The generated cargo config. Machine-specific and gitignored — never committed, because it
/// names absolute paths on one developer's disk.
pub fn config_path(root: &Path) -> PathBuf {
    root.join(".cargo").join("config.toml")
}

/// Whether a build can proceed: the free-threaded venv exists and the config names it. This is
/// what `goofi-cli/build.rs` asks, so a fresh clone fails with one actionable line instead of a
/// pyo3 error about an interpreter it could not find.
pub fn ready(root: &Path) -> bool {
    let Some(py) = venv_python(&root.join(FT_VENV)) else { return false };
    std::fs::read_to_string(config_path(root))
        .map(|text| text.contains(&pyo3_python_line(&py)))
        .unwrap_or(false)
}

/// The instruction printed wherever readiness is demanded, so the wording exists once.
pub const RUN_ME: &str = "run `cargo run -p goofi-init` first — it provisions the Python \
                          interpreters goofi links against (needs `uv` on PATH)";

/// Provision everything, from nothing, idempotently.
pub fn init(root: &Path) -> Result<(), String> {
    require_uv()?;

    let ft = ensure_venv(root, FT_VENV, FT_PYTHON)?;
    let gil = ensure_venv(root, GIL_VENV, GIL_PYTHON)?;

    // The config BEFORE the wheels: building a wheel runs cargo, and a cargo that reads a config
    // naming the interpreter is the one that produces a correctly linked artefact.
    write_config(root, &ft)?;

    for (venv, py) in [(FT_VENV, &ft), (GIL_VENV, &gil)] {
        install_wheel(root, venv, py)?;
    }
    Ok(())
}

fn require_uv() -> Result<(), String> {
    uv(["--version"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|_| {
            "goofi needs `uv` on PATH — it owns the Python interpreters the node tiers run on. \
             Install it from https://docs.astral.sh/uv/ and re-run."
                .to_string()
        })
        .and_then(|s| s.success().then_some(()).ok_or_else(|| "`uv --version` failed".into()))
}

fn ensure_venv(root: &Path, name: &str, python: &str) -> Result<PathBuf, String> {
    let venv = root.join(name);
    if let Some(py) = venv_python(&venv) {
        return Ok(py);
    }
    println!("  creating {name} (python {python})");
    run(uv(["venv", "--python", python]).arg(&venv), &format!("create {name}"))?;
    venv_python(&venv).ok_or_else(|| format!("`uv venv` left no interpreter in {}", venv.display()))
}

/// Build the wheel for THIS interpreter and install it. Skipped when the interpreter already has
/// a matching one — the version comes from the workspace `Cargo.toml`, so a bump re-provisions.
fn install_wheel(root: &Path, venv: &str, py: &Path) -> Result<(), String> {
    if has_goofi(py) {
        return Ok(());
    }
    println!("  building the goofi wheel for {venv}");
    // One output directory per venv, emptied first, so "the wheel just built" is simply the only
    // file in it: the FT interpreter gets a native `cp3XXt` wheel and the GIL one an abi3 wheel,
    // and choosing between them by name or mtime is a guess this does not have to make.
    let out = root.join("target").join("wheels").join(venv);
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).map_err(|e| format!("wheel output directory: {e}"))?;

    // maturin through `uv tool run`, so `uv` stays the ONE thing a machine needs.
    run(
        uv(["tool", "run", "maturin", "build", "--release", "-i"])
            .arg(py)
            .arg("-o")
            .arg(&out)
            .arg("-m")
            .arg(root.join("backend").join("goofi-pymod").join("Cargo.toml"))
            // Run from OUTSIDE the repo. maturin shells out to a nested cargo, and cargo discovers
            // `.cargo/config.toml` by walking up from its working directory — so from in here that
            // nested build re-injects the `[env]` block naming the FREE-THREADED interpreter's home
            // while compiling for the 3.12 one. Stripping the vars in the parent cannot help: cargo
            // puts them back from the file. It fails as `AssertionError: SRE module mismatch` deep
            // in `import re`, naming neither Python nor the variable at fault. Every path passed
            // above is absolute precisely so this can be done.
            .current_dir(std::env::temp_dir()),
        "build the goofi wheel",
    )?;

    let wheel = std::fs::read_dir(&out)
        .map_err(|e| format!("read {}: {e}", out.display()))?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "whl"))
        .ok_or_else(|| format!("maturin wrote no wheel into {}", out.display()))?;

    // Deps included on purpose: numpy is declared by the wheel itself, so installing goofi is the
    // whole of provisioning rather than the first half of it.
    run(
        uv(["pip", "install", "--python"]).arg(py).arg("--force-reinstall").arg(&wheel),
        "install the goofi wheel",
    )
}

/// Three questions in one, because each alone lets a stale venv pass for a provisioned one: does
/// `goofi` import, is it *this* goofi (`introspect` is the Rust wheel's own surface — the old
/// Python package imports perfectly well and has no such attribute), and is it the version this
/// build expects? A `Cargo.toml` version bump therefore re-provisions. It does NOT catch an edit
/// that leaves the version alone; delete the venv for that.
fn has_goofi(py: &Path) -> bool {
    let probe = format!(
        "import goofi, importlib.metadata as m; goofi.introspect; \
         raise SystemExit(0 if m.version('goofi') == '{}' else 1)",
        env!("CARGO_PKG_VERSION")
    );
    Command::new(py)
        .args(["-c", &probe])
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn pyo3_python_line(py: &Path) -> String {
    format!("PYO3_PYTHON = {:?}", slashless(py))
}

/// A path without Windows' extended-length prefix. `canonicalize` adds it and pyo3 would carry it
/// into every path derived from the interpreter; on unix it never appears, so this is a no-op.
fn slashless(p: &Path) -> String {
    let full = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).display().to_string();
    full.strip_prefix(r"\\?\").unwrap_or(&full).to_string()
}

/// Point pyo3 at the free-threaded venv, for every cargo command from here on.
fn write_config(root: &Path, ft: &Path) -> Result<(), String> {
    let purelib = query(ft, "import sysconfig;print(sysconfig.get_path('purelib'))")
        .ok_or("could not ask the interpreter where its site-packages are")?;
    // Windows has no rpath, and its loader finds `python3XX.dll` beside the executable instead —
    // which is why `goofi-cli/build.rs` copies it there. `-Wl,-rpath` is a GNU/Clang driver flag
    // that `link.exe` rejects outright, so it is keyed on the TARGET's linker rather than on
    // whether the interpreter reports a libdir: a Windows CPython reports one and cannot use it.
    let host = host_triple()?;
    let rpath = query(ft, "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
        .filter(|d| d != "None" && !host.ends_with("windows-msvc"))
        .map(|libdir| {
            // Debug-quoted as ONE flag: a raw interpolation of a Windows libdir makes `\U` an
            // invalid TOML escape, which cargo reports as an unreadable config rather than a bad
            // flag — a broken build with nothing pointing at the line that broke it.
            let flag = format!("link-arg=-Wl,-rpath,{libdir}");
            format!("\n[target.{host}]\nrustflags = [\"-C\", {flag:?}]\n")
        })
        .unwrap_or_default();
    let home = query(ft, "import sys;print(sys.base_prefix)")
        .map(|h| format!("PYTHONHOME = {h:?}\n"))
        .unwrap_or_default();

    let contents = format!(
        "# Generated by `cargo run -p goofi-init` — machine-specific, gitignored, never committed.\n\
         # Points pyo3 at the repo-local free-threaded venv so cargo needs no env vars. Delete this\n\
         # file (and {FT_VENV}) and re-run goofi-init to reprovision.\n\
         [env]\n\
         {}\n\
         PYTHONPATH = {purelib:?}\n\
         {home}{rpath}",
        pyo3_python_line(ft),
    );
    let config = config_path(root);
    if let Some(parent) = config.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(&config, contents).map_err(|e| format!("{}: {e}", config.display()))
}

/// The triple cargo will build for, asked of `rustc` rather than assumed. It names the
/// `[target.…]` section the rpath goes under, and — read as data rather than as a `cfg!` — says
/// whether this toolchain's linker is `link.exe`, which takes no `-Wl,` flags at all.
fn host_triple() -> Result<String, String> {
    let out = Command::new("rustc").arg("-vV").output().map_err(|e| format!("run rustc: {e}"))?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_string))
        .ok_or_else(|| "`rustc -vV` reported no host triple".to_string())
}

fn query(py: &Path, code: &str) -> Option<String> {
    let out = Command::new(py).args(["-c", code]).env_remove("PYTHONPATH").env_remove("PYTHONHOME").output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn uv<'a>(args: impl IntoIterator<Item = &'a str>) -> Command {
    let mut cmd = Command::new("uv");
    // goofi-init runs with whatever Python environment the caller had; uv is here to drive a
    // DIFFERENT interpreter, and handing one interpreter's stdlib to another's binary dies inside
    // `import json` with "SRE module mismatch", naming neither Python nor the variable at fault.
    cmd.args(args).env_remove("PYTHONHOME").env_remove("PYTHONPATH");
    cmd
}

fn run(cmd: &mut Command, what: &str) -> Result<(), String> {
    match cmd.status() {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("could not {what} ({s})")),
        Err(e) => Err(format!("could not {what}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A venv keeps its interpreter under `bin/` on unix and `Scripts/` on Windows, and both
    /// [`ready`] and the subprocess tier's default find it through here. An absent venv must answer
    /// `None` rather than a path that isn't there: a confident wrong answer would let [`ready`] pass
    /// on a clone with no interpreter at all, and the build would fail deep inside pyo3 instead of
    /// on the one line telling the user to run this crate.
    #[test]
    fn a_venv_is_found_under_either_layout_and_an_empty_one_is_not() {
        // No `tempfile`: this crate deliberately carries no dependencies, and one fixture does not
        // justify the first.
        let dir = std::env::temp_dir().join(format!("goofi-init-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(venv_python(&dir), None, "a directory with no interpreter is not a venv");
        for (host, exe) in [("bin", "python"), ("Scripts", "python.exe")] {
            let at = dir.join(host);
            std::fs::create_dir_all(&at).unwrap();
            let py = at.join(exe);
            std::fs::write(&py, b"").unwrap();
            assert_eq!(venv_python(&dir), Some(py.clone()), "{host}/{exe} layout");
            std::fs::remove_file(&py).unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
