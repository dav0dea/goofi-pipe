//! The interpreters goofi runs Python nodes on — and goofi's own job to produce them.
//!
//! Two venvs, because the two tiers want opposite interpreters and no one venv can be both:
//!
//! - **`.gfivenv-ft`** — free-threaded. The in-process host pyo3 LINKS against it, and the
//!   introspection probe runs on it. Created by `build.rs`, because `PYO3_PYTHON` has to name it
//!   before this crate can compile; all that is left here is putting `goofi` inside.
//! - **`.gfivenv`** — a GIL interpreter, for the subprocess child. That tier exists precisely for
//!   packages that are *not* free-threading-safe, so its Python must not be free-threaded. Hence a
//!   pinned version rather than `uv`'s own pick, which on a machine whose only installed
//!   interpreter is 3.14t would hand back the very thing the tier exists to avoid.
//!
//! Both are **goofi's**, which is what the names buy. A generic `.venv` is claimed by editors, by
//! `uv` itself, and by whatever else the directory was used for — and a stale one is not inert:
//! this repo's own `.venv` held an editable install of the OLD Python goofi, which answered
//! `import goofi` perfectly well and then had no `introspect`.
//!
//! **`uv` is a hard dependency.** Every step here is one `uv` invocation or another, so a missing
//! one is reported as itself, once, instead of resurfacing later wearing a Python-shaped disguise.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The GIL venv, and the interpreter it is built on. Pinned because "not free-threaded" is a
/// requirement of the subprocess tier, not a preference — see the module note.
pub const GIL_VENV: &str = ".gfivenv";
const GIL_PYTHON: &str = "3.12";

/// The free-threaded venv `build.rs` creates and pyo3 links against.
const FT_VENV: &str = ".gfivenv-ft";

/// A venv's interpreter, or `None` when the venv does not exist yet. Both layouts are candidates —
/// `bin/` where venvs put it, `Scripts/` where Windows does — so this answers "which is actually
/// here" rather than "which platform is this".
fn venv_python(venv: &Path) -> Option<PathBuf> {
    [venv.join("bin").join("python"), venv.join("Scripts").join("python.exe")]
        .into_iter()
        .find(|p| p.is_file())
}

/// Put `goofi` in the free-threaded venv, if `build.rs` made one. A native-only build
/// (`--no-default-features`) has none and wants none, which is why an absent one is not an error.
pub fn ensure_ft(root: &Path) -> Result<(), String> {
    let ft = root.join(FT_VENV);
    match venv_python(&ft) {
        Some(py) => ensure_goofi(root, &ft, &py),
        None => Ok(()),
    }
}

/// Make the GIL venv real and current, and hand back the interpreter the subprocess tier runs on.
pub fn ensure_gil(root: &Path) -> Result<PathBuf, String> {
    let venv = root.join(GIL_VENV);
    if venv_python(&venv).is_none() {
        uv(["venv", "--python", GIL_PYTHON]).arg(&venv).status_ok("create the GIL venv")?;
    }
    let py = venv_python(&venv)
        .ok_or_else(|| format!("`uv venv` left no interpreter in {}", venv.display()))?;
    ensure_goofi(root, &venv, &py)?;
    Ok(py)
}

/// `uv` must be there before anything else can be. Named as itself so the failure is one line
/// about a missing tool rather than a cascade of interpreters that "have no goofi".
pub fn require_uv() -> Result<(), String> {
    // Silenced: this runs on every start, and a version banner is not news. The steps that DO
    // change the machine — creating a venv, building and installing the wheel — stay loud.
    uv(["--version"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status_ok("run `uv`")
        .map_err(|_| {
            "goofi needs `uv` on PATH — it owns the Python interpreters the node tiers run on. \
             Install it from https://docs.astral.sh/uv/ and re-run."
                .to_string()
        })
}

/// Build the wheel for THIS interpreter and install it, unless it is already there and current.
fn ensure_goofi(root: &Path, venv: &Path, py: &Path) -> Result<(), String> {
    if has_goofi(py) {
        return Ok(());
    }
    // One output directory per venv, emptied first, so "the wheel we just built" is simply the only
    // file in it — the FT interpreter gets a native `cp3XXt` wheel and the GIL one an abi3 wheel,
    // and picking between them by name or by mtime is a guess this does not have to make.
    let out = root.join("target").join("wheels").join(venv.file_name().unwrap_or_default());
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).map_err(|e| format!("wheel output directory: {e}"))?;

    // maturin through `uv tool run`, so `uv` stays the ONE thing a machine needs. It shells out to
    // cargo — which is exactly why this lives at runtime and not in `build.rs`, where a nested
    // cargo would deadlock on the target-directory lock. Nothing holds that lock here.
    uv(["tool", "run", "maturin", "build", "--release", "-i"])
        .arg(py)
        .arg("-o")
        .arg(&out)
        .arg("-m")
        .arg(root.join("crates").join("goofi-pymod").join("Cargo.toml"))
        // Run from OUTSIDE the repo. maturin shells out to a nested cargo, and cargo discovers
        // `.cargo/config.toml` by walking up from its working directory — so from in here that
        // nested build re-injects the `[env]` block naming the FREE-THREADED interpreter's home,
        // while it is compiling for the 3.12 one. Stripping those vars in the parent cannot help:
        // cargo puts them back from the file. Every path passed above is absolute for this reason.
        .current_dir(std::env::temp_dir())
        .status_ok("build the goofi wheel")?;

    let wheel = std::fs::read_dir(&out)
        .map_err(|e| format!("read {}: {e}", out.display()))?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "whl"))
        .ok_or_else(|| format!("maturin wrote no wheel into {}", out.display()))?;

    // Deps included on purpose: numpy is declared by the wheel itself, so installing goofi is the
    // whole of provisioning rather than the first half of it.
    uv(["pip", "install", "--python"])
        .arg(py)
        .arg("--force-reinstall")
        .arg(&wheel)
        .status_ok("install the goofi wheel")
}

/// Three questions in one, because each alone lets a stale venv pass for a provisioned one:
/// does `goofi` import, is it *this* goofi (`introspect` is the Rust wheel's own surface — the old
/// Python package imports perfectly well and has no such attribute), and is it the version this
/// build expects? The last is what makes a bump in the workspace `Cargo.toml` propagate: the wheel
/// takes its version from there too, so a mismatch here means the venv is running an older build
/// and gets rebuilt. It does NOT catch an edit that leaves the version alone — for that, still
/// delete the venv.
fn has_goofi(py: &Path) -> bool {
    let probe = format!(
        "import goofi, importlib.metadata as m; goofi.introspect; \
         raise SystemExit(0 if m.version('goofi') == '{}' else 1)",
        env!("CARGO_PKG_VERSION")
    );
    Command::new(py)
        .args(["-c", &probe])
        .env_remove("PYTHONPATH") // the host's FT paths must not vouch for the child's venv
        .env_remove("PYTHONHOME")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Every `uv` call, with the host's own Python environment stripped off it. goofi runs with
/// `PYTHONHOME`/`PYTHONPATH` pointing at the FREE-THREADED venv it embeds (`.cargo/config.toml`
/// sets them so the linked interpreter can find its stdlib), while uv is here to drive a
/// *different* interpreter — the 3.12 one. Handing 3.14t's stdlib to a 3.12 binary does not fail
/// politely: it dies inside `import json` with `AssertionError: SRE module mismatch`, which names
/// neither Python nor the variable that caused it.
fn uv<'a>(args: impl IntoIterator<Item = &'a str>) -> Command {
    let mut cmd = Command::new("uv");
    cmd.args(args).env_remove("PYTHONHOME").env_remove("PYTHONPATH");
    cmd
}

/// Run a command to completion, turning a non-zero exit into an error that says which step failed.
trait StatusOk {
    fn status_ok(&mut self, what: &str) -> Result<(), String>;
}

impl StatusOk for Command {
    fn status_ok(&mut self, what: &str) -> Result<(), String> {
        match self.status() {
            Ok(s) if s.success() => Ok(()),
            Ok(s) => Err(format!("could not {what} ({s})")),
            Err(e) => Err(format!("could not {what}: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A venv keeps its interpreter under `bin/` on unix and `Scripts/` on Windows, and goofi has
    /// to find either from the same code. An absent venv answers `None` rather than a path that
    /// isn't there — the caller uses that to decide whether to create one, so a confident wrong
    /// answer here would skip provisioning entirely and fail much later as "no goofi".
    #[test]
    fn a_venv_is_found_under_either_layout_and_an_empty_one_is_not() {
        let tmp = tempfile::tempdir().expect("a temp dir");
        let venv = tmp.path().join("v");
        std::fs::create_dir_all(&venv).unwrap();
        assert_eq!(venv_python(&venv), None, "a directory with no interpreter is not a venv");

        for (dir, exe) in [("bin", "python"), ("Scripts", "python.exe")] {
            let host = venv.join(dir);
            std::fs::create_dir_all(&host).unwrap();
            let py = host.join(exe);
            std::fs::write(&py, b"").unwrap();
            assert_eq!(venv_python(&venv), Some(py.clone()), "{dir}/{exe} layout");
            std::fs::remove_file(&py).unwrap();
        }
    }
}
