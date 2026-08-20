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
//! It provisions the FRONTEND's dependencies too, for the same reason: `goofi-bridge`'s build
//! script compiles the SPA into the binary, so `npm install` is a precondition of `cargo build` in
//! exactly the way the interpreters are. A fresh clone that ran this and then had `cargo run` stop
//! to name a second setup command would make "setup is one command" false.
//!
//! After it runs, `cargo build`, `cargo test` and `cargo run` all work first time, every time.
//! `uv` and `npm` are the two things that must already be installed.

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

/// The repo root. Reached by popping `backend/goofi-init` off this crate's compile-time manifest
/// directory rather than joining `../..` onto it: the answer is absolute and has no `..` to print,
/// and — unlike `canonicalize` — it resolves nothing, so a checkout reached through a symlink stays
/// spelled the way the caller reached it.
pub fn repo_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.ancestors().nth(2).unwrap_or(manifest).to_path_buf()
}

/// Where a venv keeps its interpreter, relative to the venv itself — `bin/` where venvs put it,
/// `Scripts/` where Windows does. This asks which is actually present rather than which platform
/// this is, and answers `None` for a directory that is no venv at all.
fn python_in(venv: &Path) -> Option<&'static str> {
    ["bin/python", "Scripts/python.exe"].into_iter().find(|rel| venv.join(rel).is_file())
}

/// A venv's interpreter as an absolute path, or `None` when the venv is not there.
pub fn venv_python(venv: &Path) -> Option<PathBuf> {
    python_in(venv).map(|rel| venv.join(rel))
}

/// A venv's `site-packages`, discovered rather than spelled out.
///
/// The EMBEDDED interpreter needs this handed to it: pyo3 links `libpython` from the venv's BASE
/// install, so `sys.prefix` is that install and the venv — holding `goofi` and `numpy` — is on no
/// search path at all. `.cargo/config.toml` sets a `PYTHONPATH` for `cargo run`; a binary invoked
/// any other way gets nothing.
///
/// The version is *found*, never named. Unix nests site-packages under `lib/python<X.Y>[t]/`, and
/// hardcoding that segment would go stale the next time [`FT_PYTHON`] moves — the failure being a
/// silent `ModuleNotFoundError` at startup rather than anything that points at this file.
pub fn site_packages(venv: &Path) -> Option<PathBuf> {
    // Windows keeps it flat and version-free; check that first because it needs no scan.
    let flat = venv.join("Lib").join("site-packages");
    if flat.is_dir() {
        return Some(flat);
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(venv.join("lib"))
        .ok()?
        .flatten()
        .map(|e| e.path().join("site-packages"))
        .filter(|p| p.is_dir())
        .collect();
    // Sorted so a venv that somehow holds two answers gives a stable one rather than whatever
    // order the filesystem happened to hand back.
    found.sort();
    found.pop()
}

/// The interpreter of `venv`, spelled RELATIVE to the repo root — the value [`write_config`] hands
/// cargo, which expands it back to an absolute path against the config's own repo.
///
/// Relative, and above all **unresolved**. On unix a venv's `python` is a symlink into the base
/// install, so canonicalizing it names the base interpreter — which has no `goofi` wheel, because
/// the wheel was installed into the venv. pyo3 then links that one, the discovery probe spawns it,
/// `import goofi` fails, and every Python node silently drops to the subprocess tier with no error
/// anywhere. Windows venvs hold a real `python.exe`, which is the whole reason resolving looked
/// harmless. Spelling the path relative retires the question: there is nothing left to resolve.
fn interpreter_rel(root: &Path, venv: &str) -> Option<String> {
    python_in(&root.join(venv)).map(|rel| format!("{venv}/{rel}"))
}

/// The generated cargo config. Machine-specific and gitignored — never committed, because the
/// interpreter's own home and libdir are absolute paths on one developer's disk.
fn config_path(root: &Path) -> PathBuf {
    root.join(".cargo").join("config.toml")
}

/// The interpreter this build links against, as cargo resolved it — `None` until [`init`] has run.
/// `goofi-cli/build.rs` asks this, so a fresh clone fails with one actionable line instead of a
/// pyo3 error about an interpreter it could not find.
///
/// It reads the ENVIRONMENT rather than the config file's text. Cargo has already read
/// `.cargo/config.toml` and expanded its `[env]` block by the time any build script runs, so this
/// sees the answer instead of one spelling of it — which is what lets the file say
/// `.gfivenv-ft/bin/python` while pyo3 still receives an absolute path. Matching on text could not:
/// it would have to agree with the writer character for character, and would call an interpreter a
/// developer manages by hand unprovisioned.
pub fn interpreter() -> Option<PathBuf> {
    std::env::var_os("PYO3_PYTHON").map(PathBuf::from).filter(|p| p.is_file())
}

/// The instruction printed wherever readiness is demanded, so the wording exists once.
pub const RUN_ME: &str = "run `cargo run -p goofi-init` first — it provisions the Python \
                          interpreters goofi links against and the frontend's dependencies \
                          (needs `uv` and `npm` on PATH)";

/// Provision everything, from nothing, idempotently.
pub fn init(root: &Path) -> Result<(), String> {
    require_uv()?;

    // Both tools asked for BEFORE either is used: a run that provisions Python for two minutes and
    // then stops for want of npm has spent the time to report what it could have said first.
    let frontend = root.join("frontend");
    let needs_npm = frontend.join("package.json").is_file();
    if needs_npm {
        require_npm()?;
    }

    let ft = ensure_venv(root, FT_VENV, FT_PYTHON)?;
    let gil = ensure_venv(root, GIL_VENV, GIL_PYTHON)?;

    // The config BEFORE the wheels, so a failed wheel build still leaves a config a re-run can
    // use. It is NOT that the wheel build reads it — that build deliberately runs from outside the
    // repo precisely so it cannot (see `install_wheel`).
    write_config(root, &ft)?;

    for (venv, py) in [(FT_VENV, &ft), (GIL_VENV, &gil)] {
        install_wheel(root, venv, py)?;
    }

    // Run every time rather than skipped when `node_modules` is there, and that is the difference
    // between this and the venvs above: this repo ships NO lockfile deliberately, so `npm install`
    // IS the resolve step, and a `package.json` that gained a dependency is exactly the case a
    // presence check would sail past. It is a no-op when nothing moved.
    if needs_npm {
        println!("  installing the frontend's dependencies");
        run(npm(["install"]).current_dir(&frontend), "install the frontend's dependencies")?;
    }
    Ok(())
}

/// `npm`, spelled the way this platform spells it — Windows resolves it through a `.cmd` shim that
/// `CreateProcess` does not find under the bare name.
fn npm<'a>(args: impl IntoIterator<Item = &'a str>) -> Command {
    let mut cmd = Command::new(if cfg!(windows) { "npm.cmd" } else { "npm" });
    cmd.args(args);
    cmd
}

fn require_npm() -> Result<(), String> {
    npm(["--version"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|_| {
            "goofi needs `npm` on PATH — the app is compiled into the binary, so building it is \
             part of building goofi. Install Node.js from https://nodejs.org and re-run."
                .to_string()
        })
        .and_then(|s| s.success().then_some(()).ok_or_else(|| "`npm --version` failed".into()))
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

    // Deps included on purpose, and the `nodes` extra with them: numpy is declared by the wheel
    // itself and antropy by that extra, so installing goofi is the whole of provisioning rather
    // than the first half of it.
    let spec = format!("{}[nodes]", wheel.display());
    run(
        uv(["pip", "install", "--python"]).arg(py).arg("--force-reinstall").arg(&spec),
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

/// Point pyo3 at the free-threaded venv, for every cargo command from here on.
///
/// The two repo-local values are written **relative** (`relative = true`, which cargo expands
/// against the directory holding `.cargo/`), so renaming or moving the checkout does not strand
/// them. The other two cannot be: the interpreter's home and its libdir live in uv's own install,
/// outside this repo entirely, and there is no relative spelling of somewhere else.
fn write_config(root: &Path, ft: &Path) -> Result<(), String> {
    let py = interpreter_rel(root, FT_VENV)
        .ok_or_else(|| format!("{FT_VENV} holds no interpreter to point cargo at"))?;
    // The interpreter is asked for its site-packages RELATIVE to the repo, spelled with `/`: Win32
    // takes either separator, and it keeps the value clear of backslashes TOML would need escaped.
    // `{root:?}` quotes and escapes the path into a Python string literal the same way `{h:?}`
    // below quotes one into TOML.
    let purelib = query(
        ft,
        &format!(
            "import os,sysconfig;print(os.path.relpath(sysconfig.get_path('purelib'), {root:?}).replace(os.sep,'/'))"
        ),
    )
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
    // Windows' loader finds `python3XX.dll` beside the executable, where `goofi-cli/build.rs` puts
    // it — and an interpreter loaded from there can no longer infer its own home, so it has to be
    // told. Stated on every platform because it is the interpreter's true base prefix everywhere,
    // and the unix build is unaffected by being told what it would have worked out for itself.
    let home = query(ft, "import sys;print(sys.base_prefix)")
        .map(|h| format!("PYTHONHOME = {h:?}\n"))
        .unwrap_or_default();

    let contents = format!(
        "# Generated by `cargo run -p goofi-init` — machine-specific, gitignored, never committed.\n\
         # Points pyo3 at the repo-local free-threaded venv so cargo needs no env vars. The two\n\
         # repo-local paths are relative to this checkout, so moving or renaming it keeps them\n\
         # true; cargo expands both to absolute paths. Delete this file (and {FT_VENV}) and re-run\n\
         # goofi-init to reprovision.\n\
         [env]\n\
         PYO3_PYTHON = {{ value = {py:?}, relative = true }}\n\
         PYTHONPATH = {{ value = {purelib:?}, relative = true }}\n\
         {home}{rpath}",
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

// The suite lives in `goofi-tests`. This block stays because goofi-init depends on no goofi
// crate — that is what lets `-p goofi-init` run without triggering the build it configures —
// and a test crate that named it would put that edge back.
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

    /// The EMBEDDED interpreter cannot find a venv on its own, so the binary has to hand it the
    /// site-packages directory — and must find that directory without being told the Python
    /// version, or the answer goes stale the next time `FT_PYTHON` moves.
    ///
    /// Why this exists at all: pyo3 links `libpython` from the venv's BASE install, so `sys.prefix`
    /// is the base install and the venv — where `goofi` and `numpy` actually live — is nowhere on
    /// `sys.path`. `.cargo/config.toml` papered over that with a `PYTHONPATH` in its `[env]` block,
    /// which cargo applies to `cargo run` and to nothing else. The binary run directly therefore
    /// started with a working node discovery (that spawns the venv's python as a SUBPROCESS, which
    /// finds its own site-packages) and a dead param-expression evaluator: `No module named
    /// 'numpy'`. Every non-cargo invocation had it.
    #[test]
    fn site_packages_is_found_under_either_layout_without_naming_a_python_version() {
        let dir = std::env::temp_dir().join(format!("goofi-init-sp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(site_packages(&dir), None, "a directory with no site-packages is not a venv");

        // unix: `lib/pythonX.Yt/site-packages`, where X.Y is exactly what must NOT be hardcoded.
        let unix = dir.join("lib").join("python3.14t").join("site-packages");
        std::fs::create_dir_all(&unix).unwrap();
        assert_eq!(site_packages(&dir), Some(unix.clone()), "unix layout, version discovered");
        std::fs::remove_dir_all(dir.join("lib")).unwrap();

        // Windows: `Lib/site-packages`, with no version component at all.
        let win = dir.join("Lib").join("site-packages");
        std::fs::create_dir_all(&win).unwrap();
        assert_eq!(site_packages(&dir), Some(win), "windows layout");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The value cargo is given for `PYO3_PYTHON` names the interpreter **inside the venv**, never
    /// what that interpreter turns out to point at.
    ///
    /// On unix a venv's `python` is a symlink into the base install, and the base install is
    /// exactly where the `goofi` wheel is NOT — it was installed into the venv. Resolving the link
    /// therefore hands pyo3, and the discovery probe, an interpreter that cannot `import goofi`;
    /// nothing errors, every Python node just quietly drops to the subprocess tier. The fixture
    /// below is a symlink for that reason: against a plain file — which is what a Windows venv
    /// holds, and why this went unseen there — it cannot tell a resolving implementation from a
    /// correct one.
    #[test]
    fn the_interpreter_value_names_the_venv_not_what_it_points_at() {
        let dir = std::env::temp_dir().join(format!("goofi-init-rel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let venv_bin = dir.join("venv").join("bin");
        std::fs::create_dir_all(&venv_bin).unwrap();

        // The base interpreter the venv's `python` defers to, deliberately OUTSIDE the venv.
        let base = dir.join("base-python");
        std::fs::write(&base, b"").unwrap();
        let link = venv_bin.join("python");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&base, &link).unwrap();
        #[cfg(windows)]
        std::fs::copy(&base, &link).unwrap(); // a Windows venv holds a real copy, not a link

        assert_eq!(interpreter_rel(&dir, "venv").as_deref(), Some("venv/bin/python"));
        // …and a directory that is no venv has no interpreter to name.
        assert_eq!(interpreter_rel(&dir, "not-a-venv"), None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
