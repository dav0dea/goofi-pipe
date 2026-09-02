//! The one pipeline that turns a node's `.rs` file into a library goofi loads: generate a crate
//! against the embedded SDK, build it with cargo, cache the artifact by content, load it behind a
//! version symbol. No engine lives here — the composition root's build script runs it over the
//! shipped folders, and a scan runs it over a root at run time.

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{LazyLock, Mutex, OnceLock};

use sha2::{Digest, Sha256};

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

/// The goofi version every artifact is stamped with and checked against.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// One engine's authoring crate: what a node file compiles against, the crates it may reach
/// beyond it, and the line that makes the generated crate a cdylib of that engine.
pub struct Sdk {
    pub name: &'static str,
    pub dir: &'static str,
    pub allow: &'static [(&'static str, &'static str)],
    pub glue: &'static str,
}

pub const SIGNAL: Sdk = Sdk {
    name: "goofi-signal-sdk",
    dir: "backend/signal/goofi-signal-sdk",
    allow: &[("rustfft", "6.4.1"), ("realfft", "3"), ("libm", "0.2")],
    glue: "goofi_signal_sdk::cdylib!(node);",
};

pub const AUDIO: Sdk = Sdk {
    name: "goofi-audio-sdk",
    dir: "backend/audio/goofi-audio-sdk",
    allow: &[("libm", "0.2")],
    glue: "goofi_audio_sdk::cdylib!(node);",
};

pub fn sdk(name: &str) -> Option<&'static Sdk> {
    [&SIGNAL, &AUDIO].into_iter().find(|s| s.name == name)
}

/// Where the extracted SDK, the generated crates, one shared cargo target and every artifact
/// live: `$GOOFI_BUILD_DIR`, else `<home>/build`.
pub fn base_dir(home: &Path) -> PathBuf {
    std::env::var_os("GOOFI_BUILD_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join("build"))
}

/// What one source builds to, keyed by everything that decides it: the goofi version, the SDK
/// sources, the crates the SDK lets a node reach, and the file's own bytes.
fn cache_key(sdk: &Sdk, source: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(VERSION.as_bytes());
    hash.update(SDK_HASH.as_bytes());
    hash.update(sdk.name.as_bytes());
    for (name, version) in sdk.allow {
        hash.update(name.as_bytes());
        hash.update(version.as_bytes());
    }
    hash.update(sdk.glue.as_bytes());
    hash.update(source);
    format!("{:x}", hash.finalize())[..32].to_string()
}

fn artifact_path(base: &Path, key: &str, stem: &str) -> PathBuf {
    base.join("out").join(key).join(format!("{stem}.{}", std::env::consts::DLL_EXTENSION))
}

/// Why the last `ensure` of a key failed in this process, for `built` to answer. Never on disk:
/// a signal, a full disk or a resolve with no network is retried at the next build, not kept.
static FAILED: LazyLock<Mutex<HashMap<String, String>>> = LazyLock::new(Default::default);

fn stem_of(source: &Path) -> &str {
    source.file_stem().and_then(|s| s.to_str()).unwrap_or("node")
}

fn locate(sdk: &Sdk, source: &Path, base: &Path) -> Result<(String, PathBuf), String> {
    let bytes = std::fs::read(source).map_err(|e| format!("could not read {}: {e}", source.display()))?;
    let key = cache_key(sdk, &bytes);
    let artifact = artifact_path(base, &key, stem_of(source));
    Ok((key, artifact))
}

/// The artifact for `source` if it is there, else why the last build of these bytes failed.
/// Never runs cargo: this is the half a scan may take under the graph lock.
pub fn built(sdk: &Sdk, source: &Path, base: &Path) -> Result<PathBuf, String> {
    let (key, artifact) = locate(sdk, source, base)?;
    if artifact.is_file() {
        return Ok(artifact);
    }
    let failed = FAILED.lock().unwrap_or_else(|e| e.into_inner());
    Err(failed.get(&key).cloned().unwrap_or_else(|| "not built yet — refresh the library".into()))
}

/// The artifact for `source`, built if it is not there — every failure is retried, and what it
/// said is kept for `built`.
pub fn ensure(sdk: &Sdk, source: &Path, base: &Path) -> Result<PathBuf, String> {
    let (key, artifact) = locate(sdk, source, base)?;
    if artifact.is_file() {
        return Ok(artifact);
    }
    let result = build(sdk, source, base, &key, &artifact);
    if let Err(why) = &result {
        FAILED.lock().unwrap_or_else(|e| e.into_inner()).insert(key, why.clone());
    }
    result
}

fn build(sdk: &Sdk, source: &Path, base: &Path, key: &str, artifact: &Path) -> Result<PathBuf, String> {
    let Some(cargo) = cargo() else {
        return Err("needs `cargo` to build — install a Rust toolchain, or use a shipped node".into());
    };
    static BUILDING: Mutex<()> = Mutex::new(());
    let _one_at_a_time = BUILDING.lock().unwrap_or_else(|e| e.into_inner());
    if artifact.is_file() {
        return Ok(artifact.to_path_buf());
    }
    let crate_dir = base.join("crates").join(key);
    let crate_name = format!("goofi_node_{}", stem_of(source).to_lowercase());
    generate(sdk, source, &sdk_root(base), &crate_dir, &crate_name)?;
    let mut cmd = Command::new(cargo);
    cmd.args(["build", "--release", "--message-format", "short", "--color", "never"]).current_dir(&crate_dir);
    // A nested cargo must not inherit the outer build's own knobs — a build script's `OUT_DIR`,
    // its encoded rustflags, its target triple — only the jobserver and the home.
    for (k, _) in std::env::vars_os() {
        let k = k.to_string_lossy();
        let outer = k.starts_with("CARGO_") && !matches!(&*k, "CARGO_HOME" | "CARGO_MAKEFLAGS");
        if outer || matches!(&*k, "OUT_DIR" | "TARGET" | "HOST" | "PROFILE" | "OPT_LEVEL" | "DEBUG" | "NUM_JOBS" | "RUSTC" | "RUSTDOC" | "RUSTC_LINKER" | "RUSTC_WORKSPACE_WRAPPER") {
            cmd.env_remove(&*k);
        }
    }
    cmd.env("CARGO_TARGET_DIR", base.join("target"));
    let output = cmd.output().map_err(|e| format!("could not run cargo: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let built = base.join("target").join("release").join(format!(
        "{}{crate_name}.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    ));
    std::fs::read(&built)
        .and_then(|bytes| place(artifact, &bytes))
        .map_err(|e| format!("built, but could not place {}: {e}", artifact.display()))?;
    Ok(artifact.to_path_buf())
}

fn cargo() -> Option<PathBuf> {
    let cargo = std::env::var_os("CARGO").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("cargo"));
    Command::new(&cargo).arg("--version").output().ok().filter(|o| o.status.success()).map(|_| cargo)
}

/// The embedded SDK, written out under `base` for this goofi version — each file only when its
/// bytes differ, so an unchanged tree never re-dirties cargo's view of it.
pub fn sdk_root(base: &Path) -> PathBuf {
    let root = base.join("sdk").join(VERSION);
    for (rel, bytes) in SOURCES {
        write_if_changed(&root.join(rel), bytes);
    }
    write_if_changed(&root.join("Cargo.toml"), workspace_manifest().as_bytes());
    root
}

/// The SDK workspace's manifest: the repo's own `[workspace.package]` table, so the version and
/// edition have one owner, over the embedded members.
fn workspace_manifest() -> String {
    let package: String = WORKSPACE_MANIFEST
        .split("[workspace.package]")
        .nth(1)
        .unwrap_or("")
        .lines()
        .take_while(|l| !l.starts_with('['))
        .filter(|l| !l.trim_start().starts_with('#') && l.contains('='))
        .map(|l| format!("{l}\n"))
        .collect();
    let members: Vec<String> = SOURCES
        .iter()
        .filter_map(|(rel, _)| rel.strip_suffix("/Cargo.toml"))
        .map(|dir| format!("{dir:?}"))
        .collect();
    format!(
        "[workspace]\nresolver = \"2\"\nmembers = [{}]\n\n[workspace.package]\n{package}",
        members.join(", ")
    )
}

fn generate(sdk: &Sdk, source: &Path, sdk_root: &Path, crate_dir: &Path, crate_name: &str) -> Result<(), String> {
    let slash = |p: &Path| p.to_string_lossy().replace('\\', "/");
    let deps: String = sdk
        .allow
        .iter()
        .map(|(name, version)| format!("{name} = {version:?}\n"))
        .collect();
    let manifest = format!(
        "[package]\nname = {crate_name:?}\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
         [lib]\ncrate-type = [\"cdylib\"]\n\n\
         [dependencies]\n{} = {{ path = {:?} }}\ngoofi-core = {{ path = {:?} }}\n{deps}\n\
         [profile.release]\nopt-level = 3\ndebug = false\nincremental = true\ncodegen-units = 16\npanic = \"unwind\"\n\n\
         [workspace]\n",
        sdk.name,
        slash(&sdk_root.join(sdk.dir)),
        slash(&sdk_root.join("backend/goofi-core")),
    );
    let lib = format!(
        "//! Generated by goofi-build around {}.\n\
         #[forbid(unsafe_code)]\n#[path = {:?}]\nmod node;\n{}\n",
        source.display(),
        slash(&std::path::absolute(source).map_err(|e| e.to_string())?),
        sdk.glue,
    );
    std::fs::create_dir_all(crate_dir.join("src")).map_err(|e| e.to_string())?;
    write_if_changed(&crate_dir.join("Cargo.toml"), manifest.as_bytes());
    write_if_changed(&crate_dir.join("src/lib.rs"), lib.as_bytes());
    // The repo's own lock seeds the resolution, so a node builds against the versions goofi did.
    if !crate_dir.join("Cargo.lock").exists() {
        std::fs::write(crate_dir.join("Cargo.lock"), LOCK).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Write `bytes` at `path` unless they are already there, so an unchanged file keeps its mtime.
pub fn write_if_changed(path: &Path, bytes: &[u8]) {
    if std::fs::read(path).is_ok_and(|have| have == bytes) {
        return;
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, bytes);
}

/// Put `bytes` at `path` unless something is already there. An artifact under its key is whole or
/// absent and never rewritten: a process may have the one that is there mapped.
pub fn place(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    let dir = path.parent().expect("an artifact has a directory");
    std::fs::create_dir_all(dir)?;
    let part = dir.join(format!(".{}.{}", path.file_name().unwrap().to_string_lossy(), std::process::id()));
    std::fs::write(&part, bytes)?;
    std::fs::rename(&part, path)
}

/// A loaded artifact: the library, kept for the process lifetime, and what it says it is.
pub struct Opened {
    pub library: &'static libloading::Library,
    pub describe: String,
}

/// Load an artifact, once per path: `goofi_version` first — a mismatch is a refusal naming both
/// versions, never a call into a stale ABI — then `goofi_describe`. Never unloaded: the vtables,
/// `'static` data and thread-locals it hands out pin it for the life of the process.
pub fn open(path: &Path) -> Result<Opened, String> {
    static OPENED: OnceLock<Mutex<HashMap<PathBuf, (&'static libloading::Library, String)>>> = OnceLock::new();
    let cache = OPENED.get_or_init(Default::default);
    let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((library, describe)) = cache.get(path) {
        return Ok(Opened { library, describe: describe.clone() });
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let library = load(path).map_err(|e| format!("{name}: could not load: {e}"))?;
    let version = unsafe { c_string(&library, b"goofi_version\0") }?;
    if version != VERSION {
        return Err(format!("{name}: built for goofi {version}, and this is {VERSION}"));
    }
    let describe = unsafe { c_string(&library, b"goofi_describe\0") }?;
    let library: &'static libloading::Library = Box::leak(Box::new(library));
    cache.insert(path.to_path_buf(), (library, describe.clone()));
    Ok(Opened { library, describe })
}

#[cfg(unix)]
fn load(path: &Path) -> Result<libloading::Library, libloading::Error> {
    use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
    // NOW, not cargo's default LAZY: the first call into a fresh node must not run the resolver.
    unsafe { Library::open(Some(path), RTLD_NOW | RTLD_LOCAL) }.map(Into::into)
}

#[cfg(not(unix))]
fn load(path: &Path) -> Result<libloading::Library, libloading::Error> {
    unsafe { libloading::Library::new(path) }
}

unsafe fn c_string(library: &libloading::Library, symbol: &[u8]) -> Result<String, String> {
    let name = String::from_utf8_lossy(&symbol[..symbol.len() - 1]).into_owned();
    let f: libloading::Symbol<unsafe extern "C" fn() -> *const c_char> =
        library.get(symbol).map_err(|e| format!("no `{name}` symbol: {e}"))?;
    let ptr = f();
    if ptr.is_null() {
        return Err(format!("`{name}` answered null"));
    }
    Ok(CStr::from_ptr(ptr).to_string_lossy().into_owned())
}
