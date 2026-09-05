//! Build the SPA this crate serves and the shipped node folders, and embed both.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

fn main() {
    prebuild_nodes();
    // Outside every early return below, or the headless verdict outlives the change that revoked it.
    println!("cargo:rerun-if-env-changed=GOOFI_HEADLESS");
    let frontend = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../frontend");
    if headless() {
        println!("cargo:warning=GOOFI_HEADLESS: built without the frontend — this binary serves the API alone");
        return embed_spa(&frontend.join("build"), true);
    }
    sync_frontend(&frontend);
    embed_spa(&frontend.join("build"), false);
}

/// Build every `node-bundles/<bundle>/*.rs` through the one pipeline a scan runs,
/// into a build dir the test harness shares, and emit `$OUT_DIR/shipped.rs`: every file of every
/// bundle, and every artifact under the cache key a scan will look it up by. A shipped node that
/// does not compile fails THIS build, so a binary never ships a node it cannot load.
fn prebuild_nodes() {
    let bundles = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../node-bundles");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    // OUT_DIR is `<target>/<profile>/build/<pkg>-<hash>/out`; four levels up is `<target>`.
    let base = out.ancestors().nth(4).expect("a cargo OUT_DIR").join("goofi-build");
    println!("cargo:rerun-if-changed={}", bundles.display());
    let (mut sources, mut artifacts) = (String::new(), String::new());
    for bundle in dirs_under(&bundles) {
        println!("cargo:rerun-if-changed={}", bundle.display());
        let name = bundle.file_name().unwrap().to_string_lossy().into_owned();
        for path in files_under(&bundle) {
            println!("cargo:rerun-if-changed={}", path.display());
            let within = path.strip_prefix(&bundle).unwrap().to_string_lossy().replace('\\', "/");
            sources += &format!("    ({:?}, include_bytes!({:?})),\n", format!("{name}/{within}"), path.display().to_string());
            let Some(sdk) = sdk_of(&path) else { continue };
            let artifact = goofi_build::ensure(sdk, &path, &base)
                .unwrap_or_else(|why| panic!("the shipped node {} does not build:\n{why}", path.display()));
            println!("cargo:rerun-if-changed={}", artifact.display());
            let key = artifact.parent().unwrap().file_name().unwrap().to_string_lossy().into_owned();
            let file = artifact.file_name().unwrap().to_string_lossy().into_owned();
            artifacts += &format!("    ({key:?}, {file:?}, include_bytes!({:?})),\n", artifact.display().to_string());
        }
    }
    std::fs::write(
        out.join("shipped.rs"),
        format!(
            "pub static SHIPPED_SOURCES: &[(&str, &[u8])] = &[\n{sources}];\n\
             pub static SHIPPED_ARTIFACTS: &[(&str, &str, &[u8])] = &[\n{artifacts}];\n"
        ),
    )
    .expect("write shipped.rs");
}

/// The SDK a `.rs` node file is built with: the one it names.
fn sdk_of(path: &Path) -> Option<&'static goofi_build::Sdk> {
    if path.extension().is_none_or(|e| e != "rs") {
        return None;
    }
    match goofi_node::engine_of(path)?.as_str() {
        "signal" => Some(&goofi_build::SIGNAL),
        "audio" => Some(&goofi_build::AUDIO),
        _ => None,
    }
}

/// The directories directly under `dir`, sorted.
fn dirs_under(dir: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> =
        std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    dirs
}

/// Every file under `dir` at any depth, sorted, skipping what git skips: `__pycache__` and dot-files.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(d) = pending.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "__pycache__" {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Whether this is a headless build; only a truthy value opts in, so `=0` asks for the app.
fn headless() -> bool {
    matches!(std::env::var("GOOFI_HEADLESS").as_deref(), Ok("1") | Ok("true"))
}

/// Frontend inputs that trigger a rebuild — never `build/`, `.svelte-kit/` or `node_modules/`,
/// which self-retrigger or are enormous.
const INPUTS: &[&str] = &[
    "src",
    "static",
    "package.json",
    "package-lock.json",
    "svelte.config.js",
    "vite.config.ts",
    "tsconfig.json",
];

/// Rebuild the served SPA when its sources are newer than the last build; a build that cannot be
/// made current panics, because by here the caller has asked for an app.
fn sync_frontend(frontend: &Path) {
    if !frontend.join("src").is_dir() {
        return;
    }

    let mut newest_src: Option<SystemTime> = None;
    for input in INPUTS {
        if let Some(t) = newest_mtime(&frontend.join(input), true) {
            if newest_src.is_none_or(|n| t > n) {
                newest_src = Some(t);
            }
        }
    }

    let built = newest_mtime(&frontend.join("build"), false);

    let stale = match (newest_src, built) {
        (_, None) => true,                 // no build yet
        (Some(src), Some(out)) => src > out, // a source is newer than the build
        (None, Some(_)) => false,          // nothing to build from
    };
    if !stale {
        return;
    }

    // Asked before npm runs: without it a fresh clone fails as `svelte-kit: not found`, exit 127.
    assert!(
        frontend.join("node_modules").is_dir(),
        "the frontend's dependencies are not installed — {}",
        goofi_init::RUN_ME
    );

    // Past tense, after the fact: cargo REPLAYS a build script's warnings on later builds where
    // npm never ran, so only a completed-event wording stays true.
    let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
    let started = SystemTime::now();
    match Command::new(npm).args(["run", "build"]).current_dir(frontend).status() {
        Ok(s) if s.success() => {
            let secs = started.elapsed().map(|d| d.as_secs_f32()).unwrap_or(0.0);
            println!(
                "cargo:warning=rebuilt the served SPA (frontend/build) from changed sources in \
                 {secs:.1}s — cargo REPLAYS this line on later no-op builds, where npm did not re-run"
            );
        }
        Ok(s) => panic!("`npm run build` failed in frontend/ ({s})"),
        Err(e) => panic!(
            "could not run `{npm}` ({e}) — it builds the SPA compiled into this binary. Install \
             Node.js, or set GOOFI_HEADLESS=1 to build the API-only binary, which needs none"
        ),
    }
}


/// Newest modification time of `path` or anything under it, or `None` if absent. `watch` emits
/// `cargo:rerun-if-changed` per path visited — cargo versions differ on recursing a watched dir.
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

/// Emit `$OUT_DIR/spa.rs`: every file under `build/`, keyed by the URL path it is served at, plus
/// the `HEADLESS_BUILD` stamp that tells an empty table that was asked for from a broken build.
///
/// Every embedded file is watched: `include_bytes!` names an absolute path, so a `build/` rewritten
/// behind cargo's back would leave this table naming hashed files that are gone.
fn embed_spa(build: &Path, headless: bool) {
    let mut files = Vec::new();
    if !headless {
        walk(build, build, &mut files);
        files.sort();
        println!("cargo:rerun-if-changed={}", build.display());
        for (_, abs) in &files {
            println!("cargo:rerun-if-changed={abs}");
        }
    }
    let rows: String = files
        .iter()
        .map(|(url, abs)| format!("    ({url:?}, include_bytes!({abs:?})),\n"))
        .collect();
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("spa.rs");
    std::fs::write(
        &out,
        format!(
            "pub static SPA: Spa = &[\n{rows}];\n\
             /// Built with `GOOFI_HEADLESS`: no app is compiled in, BY REQUEST. An empty [`SPA`] \
             without this is a broken build, not a mode.\n\
             pub static HEADLESS_BUILD: bool = {headless};\n"
        ),
    )
    .expect("write spa.rs");
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.filter_map(Result::ok) {
        let path = e.path();
        if path.is_dir() {
            // Watched so a file APPEARING in it re-runs this script; no file's own mtime says that.
            println!("cargo:rerun-if-changed={}", path.display());
            walk(root, &path, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            let url = rel.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/");
            out.push((url, path.to_string_lossy().into_owned()));
        }
    }
}
