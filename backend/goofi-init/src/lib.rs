//! Setting up the Python goofi runs its nodes on, and the frontend's dependencies — the ONE place
//! that does it. It must depend on no goofi crate and no pyo3, or it triggers the build it configures.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The GIL venv the subprocess tier runs on. Pinned *non*-free-threaded on purpose: that tier
/// exists precisely for packages that are not free-threading-safe.
pub const GIL_VENV: &str = ".gfivenv";
const GIL_PYTHON: &str = "3.12";

/// The free-threaded venv pyo3 LINKS against, and which the introspection probe runs on.
pub const FT_VENV: &str = ".gfivenv-ft";
const FT_PYTHON: &str = "3.14t";

/// The repo root. Nothing is resolved, so a checkout reached through a symlink stays spelled the
/// way the caller reached it.
pub fn repo_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.ancestors().nth(2).unwrap_or(manifest).to_path_buf()
}

/// Where a venv keeps its interpreter, asked by presence rather than by platform.
fn python_in(venv: &Path) -> Option<&'static str> {
    ["bin/python", "Scripts/python.exe"].into_iter().find(|rel| venv.join(rel).is_file())
}

/// A venv's interpreter as an absolute path, or `None` when the venv is not there.
pub fn venv_python(venv: &Path) -> Option<PathBuf> {
    python_in(venv).map(|rel| venv.join(rel))
}

/// A venv's `site-packages`, which the embedded interpreter must be handed. The Python version is
/// *found*, never named, so it cannot go stale when [`FT_PYTHON`] moves.
pub fn site_packages(venv: &Path) -> Option<PathBuf> {
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
    // Sorted so a venv that somehow holds two answers gives a stable one.
    found.sort();
    found.pop()
}

/// The interpreter of `venv`, RELATIVE to the repo root and above all UNRESOLVED: on unix that
/// symlink points at the base install, which has no `goofi` wheel.
fn interpreter_rel(root: &Path, venv: &str) -> Option<String> {
    python_in(&root.join(venv)).map(|rel| format!("{venv}/{rel}"))
}

/// The generated cargo config. Machine-specific and gitignored.
fn config_path(root: &Path) -> PathBuf {
    root.join(".cargo").join("config.toml")
}

/// The interpreter this build links against, as cargo resolved it — `None` until [`init`] has run.
/// Read from the ENVIRONMENT, which cargo has already expanded, not from the config file's text.
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

    // Both tools asked for BEFORE either is used, so a long provision cannot stop for want of npm.
    let frontend = root.join("frontend");
    let needs_npm = frontend.join("package.json").is_file();
    if needs_npm {
        require_npm()?;
    }

    let ft = ensure_venv(root, FT_VENV, FT_PYTHON)?;
    let gil = ensure_venv(root, GIL_VENV, GIL_PYTHON)?;

    // The config BEFORE the wheels, so a failed wheel build still leaves a config a re-run can use.
    write_config(root, &ft)?;

    // The bundles' packages every time, never gated on presence: a bundle added since the last
    // run names new ones, and uv answers a satisfied list in milliseconds.
    let reqs = requirements_in(&bundle_dirs(root));
    for (venv, py) in [(FT_VENV, &ft), (GIL_VENV, &gil)] {
        install_wheel(root, venv, py)?;
        if !reqs.is_empty() {
            println!("  installing the bundles' packages into {venv}");
            install_packages(py, &reqs)?;
        }
    }

    // Run every time, never gated on `node_modules`: no lockfile ships, so `npm install` IS the
    // resolve step and a presence check would sail past a new dependency.
    if needs_npm {
        println!("  installing the frontend's dependencies");
        run(npm(["install"]).current_dir(&frontend), "install the frontend's dependencies")?;
    }
    Ok(())
}

/// `npm`, spelled the way this platform spells it: Windows needs the `.cmd` shim by name.
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

/// Build the wheel for THIS interpreter and install it, unless a matching one is already there.
fn install_wheel(root: &Path, venv: &str, py: &Path) -> Result<(), String> {
    if has_goofi(py) {
        return Ok(());
    }
    println!("  building the goofi wheel for {venv}");
    // One output directory per venv, emptied first, so the wheel just built is the only file in it.
    let out = root.join("target").join("wheels").join(venv);
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).map_err(|e| format!("wheel output directory: {e}"))?;

    run(
        uv(["tool", "run", "maturin", "build", "--release", "-i"])
            .arg(py)
            .arg("-o")
            .arg(&out)
            .arg("-m")
            .arg(root.join("backend").join("signal").join("goofi-pymod").join("Cargo.toml"))
            // Run from OUTSIDE the repo, or maturin's nested cargo picks `.cargo/config.toml` up
            // and builds against the free-threaded interpreter's home. Hence the absolute paths.
            .current_dir(std::env::temp_dir()),
        "build the goofi wheel",
    )?;

    let wheel = std::fs::read_dir(&out)
        .map_err(|e| format!("read {}: {e}", out.display()))?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "whl"))
        .ok_or_else(|| format!("maturin wrote no wheel into {}", out.display()))?;

    run(
        uv(["pip", "install", "--python"]).arg(py).arg("--force-reinstall").arg(&wheel),
        "install the goofi wheel",
    )
}

/// The bundles this repo ships: every directory under `node-bundles/`, sorted.
pub fn bundle_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root.join("node-bundles"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// The `requirements.txt` each of `dirs` carries — what its nodes import beyond goofi's own.
pub fn requirements_in(dirs: &[PathBuf]) -> Vec<PathBuf> {
    dirs.iter().map(|d| d.join("requirements.txt")).filter(|p| p.is_file()).collect()
}

fn pip_install(py: &Path, reqs: &[PathBuf], dry_run: bool) -> Command {
    let mut cmd = uv(["pip", "install", "--python"]);
    cmd.arg(py);
    if dry_run {
        cmd.arg("--dry-run");
    }
    for r in reqs {
        cmd.arg("-r").arg(r);
    }
    cmd
}

fn names(reqs: &[PathBuf]) -> String {
    reqs.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", ")
}

/// What `py` lacks to satisfy `reqs`, as uv would install it. uv audits site-packages before it
/// resolves anything, so a satisfied set answers in milliseconds and without the network.
pub fn missing_packages(py: &Path, reqs: &[PathBuf]) -> Result<Vec<String>, String> {
    if reqs.is_empty() {
        return Ok(Vec::new());
    }
    let out = pip_install(py, reqs, true).output().map_err(|e| format!("could not run uv: {e}"))?;
    let text = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        return Err(format!("uv could not resolve {}: {}", names(reqs), text.trim()));
    }
    // ponytail: reads uv's dry-run listing; a `--format json` on `uv pip install` replaces this.
    Ok(text.lines().filter_map(|l| l.strip_prefix(" + ")).map(str::to_string).collect())
}

/// Install `reqs` into `py`.
pub fn install_packages(py: &Path, reqs: &[PathBuf]) -> Result<(), String> {
    run(&mut pip_install(py, reqs, false), &format!("install {}", names(reqs)))
}

/// Does this interpreter hold THIS goofi? `introspect` separates the Rust wheel from the old Python
/// package, and the version makes a bump re-provision — an edit that keeps it needs the venv deleted.
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

/// Point pyo3 at the free-threaded venv, for every cargo command from here on. The repo-local
/// values are `relative = true` so moving the checkout does not strand them.
fn write_config(root: &Path, ft: &Path) -> Result<(), String> {
    let py = interpreter_rel(root, FT_VENV)
        .ok_or_else(|| format!("{FT_VENV} holds no interpreter to point cargo at"))?;
    // Spelled with `/`: Win32 takes either separator, and it keeps the value clear of backslashes
    // TOML would need escaped. `{root:?}` quotes the path into a Python string literal.
    let purelib = query(
        ft,
        &format!(
            "import os,sysconfig;print(os.path.relpath(sysconfig.get_path('purelib'), {root:?}).replace(os.sep,'/'))"
        ),
    )
    .ok_or("could not ask the interpreter where its site-packages are")?;
    // `-Wl,-rpath` is a GNU/Clang flag `link.exe` rejects, so this is keyed on the TARGET's linker
    // rather than on the reported libdir: a Windows CPython reports one and cannot use it.
    let host = host_triple()?;
    let rpath = query(ft, "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
        .filter(|d| d != "None" && !host.ends_with("windows-msvc"))
        .map(|libdir| {
            // Debug-quoted: a raw Windows libdir makes `\U` an invalid TOML escape.
            let flag = format!("link-arg=-Wl,-rpath,{libdir}");
            format!("\n[target.{host}]\nrustflags = [\"-C\", {flag:?}]\n")
        })
        .unwrap_or_default();
    // Stated on every platform: a Windows interpreter loaded from beside the executable cannot
    // infer its own home, and unix is unaffected by being told what it would have worked out.
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

/// The triple cargo will build for, asked of `rustc` rather than assumed.
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
    // uv drives a DIFFERENT interpreter: the caller's stdlib kills it with "SRE module mismatch".
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

// Here rather than in `goofi-tests`, which would give this crate the dependency edge it must not
// have.
#[cfg(test)]
mod tests {
    use super::*;

    /// Either venv layout is found, and an absent venv answers `None` rather than a path that is
    /// not there.
    #[test]
    fn a_venv_is_found_under_either_layout_and_an_empty_one_is_not() {
        // No `tempfile`: this crate carries no dependencies.
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

    /// The embedded interpreter is handed site-packages, found without naming a Python version.
    #[test]
    fn site_packages_is_found_under_either_layout_without_naming_a_python_version() {
        let dir = std::env::temp_dir().join(format!("goofi-init-sp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(site_packages(&dir), None, "a directory with no site-packages is not a venv");

        let unix = dir.join("lib").join("python3.14t").join("site-packages");
        std::fs::create_dir_all(&unix).unwrap();
        assert_eq!(site_packages(&dir), Some(unix.clone()), "unix layout, version discovered");
        std::fs::remove_dir_all(dir.join("lib")).unwrap();

        let win = dir.join("Lib").join("site-packages");
        std::fs::create_dir_all(&win).unwrap();
        assert_eq!(site_packages(&dir), Some(win), "windows layout");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `PYO3_PYTHON` names the interpreter inside the venv, never what the symlink points at — so
    /// the fixture is a symlink, which a plain file could not tell apart.
    #[test]
    fn the_interpreter_value_names_the_venv_not_what_it_points_at() {
        let dir = std::env::temp_dir().join(format!("goofi-init-rel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let venv_bin = dir.join("venv").join("bin");
        std::fs::create_dir_all(&venv_bin).unwrap();

        let base = dir.join("base-python");
        std::fs::write(&base, b"").unwrap();
        let link = venv_bin.join("python");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&base, &link).unwrap();
        #[cfg(windows)]
        std::fs::copy(&base, &link).unwrap(); // a Windows venv holds a real copy, not a link

        assert_eq!(interpreter_rel(&dir, "venv").as_deref(), Some("venv/bin/python"));
        assert_eq!(interpreter_rel(&dir, "not-a-venv"), None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
