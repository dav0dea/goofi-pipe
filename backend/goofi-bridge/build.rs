//! Build the SPA this crate serves, and embed it.
//!
//! `goofi-pipe` ships as ONE file, so the built bundle is compiled in rather than read from a
//! `frontend/build/` that has to travel beside the binary and can silently go stale against it.
//! Both halves live here, in the crate that SERVES the bundle: putting the npm build in
//! `goofi-cli` instead left cargo free to compile this crate first, against whatever `build/`
//! happened to be on disk.
//!
//! The npm build runs only when a frontend source is newer than the last build, and a failure of
//! it FAILS the Rust build. It used to warn and leave the previous bundle in place, which is the
//! one thing it must not do: the bundle is compiled in, so "the previous build" is a frontend that
//! does not match the binary around it — and on a fresh clone, where `npm run build` exits 127
//! because `node_modules` is absent, there is no previous build at all. That combination started a
//! server that answered every route with nothing.
//!
//! What is NOT a failure here: no frontend tree beside the crate (vendored alone), and
//! `GOOFI_SKIP_FRONTEND_BUILD=1` over a stale bundle. Both are answered at startup instead — the
//! script stamps WHY the embedded bundle must not be served into `SPA_DEFECT`, and the binary
//! refuses to serve the app rather than serving the wrong one. `--headless` needs no SPA and is
//! never blocked by it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

fn main() {
    let frontend = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../frontend");
    let defect = sync_frontend(&frontend);
    embed_spa(&frontend.join("build"), defect);
}

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

/// Rebuild the served SPA when its sources are newer than the last build (see the module doc).
///
/// Returns why the bundle that ends up embedded must not be served, or `None`. A build that CANNOT
/// be made current panics instead — the difference is whether the caller asked for this outcome.
fn sync_frontend(frontend: &Path) -> Option<String> {
    // No frontend source tree (e.g. the crate vendored alone) → nothing to manage. Whether that
    // leaves anything to serve is `embed_spa`'s question.
    if !frontend.join("src").is_dir() {
        return None;
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

    // `build/` is the output — walk it for its newest mtime here; `embed_spa` is what watches it.
    let built = newest_mtime(&frontend.join("build"), false);

    let stale = match (newest_src, built) {
        (_, None) => true,                 // no build yet
        (Some(src), Some(out)) => src > out, // a source is newer than the build
        (None, Some(_)) => false,          // nothing to build from
    };
    if !stale {
        return None;
    }

    // Only a truthy value opts out — `=0`/empty must NOT skip (a user setting `=0` wants a build).
    // The opt-out is honoured, and then carried: the caller asked for a build that skips npm, not
    // for a server that hands out a bundle it knows is behind its own sources.
    if matches!(std::env::var("GOOFI_SKIP_FRONTEND_BUILD").as_deref(), Ok("1") | Ok("true")) {
        println!("cargo:warning=frontend/build is stale but GOOFI_SKIP_FRONTEND_BUILD is set — not rebuilding");
        return Some(
            "frontend/build was stale when this binary was built, and GOOFI_SKIP_FRONTEND_BUILD \
             suppressed the rebuild"
                .to_string(),
        );
    }

    // Asked before npm runs, because npm cannot say this. A fresh clone fails as `svelte-kit: not
    // found`, exit 127 — which names neither the cause nor the fix, and was the report the user
    // actually got.
    assert!(
        frontend.join("node_modules").is_dir(),
        "the frontend's dependencies are not installed — run `npm install` in frontend/"
    );

    // Report AFTER the build, in the PAST tense with the measured duration. Cargo caches a build
    // script's `cargo:warning` lines and REPLAYS them on every later build until the script next
    // re-runs — so a present-tense "rebuilding…" reads as a fresh claim on a no-op build where npm
    // never ran (the confusing case: the line appears, then cargo finishes in milliseconds). Past
    // tense + a duration describe a completed event, so the line stays true when it is replayed.
    // Nothing is printed before the build because cargo captures build-script output and only shows
    // it once the script finishes — a pre-announcement could not act as live progress anyway.
    let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
    let started = SystemTime::now();
    match Command::new(npm).args(["run", "build"]).current_dir(frontend).status() {
        Ok(s) if s.success() => {
            let secs = started.elapsed().map(|d| d.as_secs_f32()).unwrap_or(0.0);
            println!(
                "cargo:warning=rebuilt the served SPA (frontend/build) from changed sources in \
                 {secs:.1}s — cargo REPLAYS this line on later no-op builds, where npm did not re-run"
            );
            None
        }
        Ok(s) => panic!("`npm run build` failed in frontend/ ({s})"),
        Err(e) => panic!(
            "could not run `{npm}` ({e}) — it builds the SPA compiled into this binary. Install \
             Node.js, or set GOOFI_SKIP_FRONTEND_BUILD=1 to build against the bundle already in \
             frontend/build"
        ),
    }
}


/// Newest modification time of `path` (a file) or anything under it (a dir), or `None` if absent.
/// When `watch`, also emits `cargo:rerun-if-changed` for every path visited — a per-path emit (not a
/// bare directory) is robust across cargo versions, which differ on whether a watched directory is
/// scanned recursively. Sources pass `watch = true`; the `build/` output passes `false`, because
/// `embed_spa` watches it per embedded file instead.
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

/// Emit `$OUT_DIR/spa.rs`: every file under `build/`, keyed by the URL path it is served at.
///
/// An absent or empty tree emits an empty table rather than failing — a crate vendored without the
/// frontend still builds. It emits the REASON with it, so the binary can refuse to serve an app it
/// does not have instead of answering every route with nothing.
///
/// Every embedded file is WATCHED, and the tree's directories with it. `include_bytes!` names an
/// absolute path, so a `build/` rewritten behind cargo's back — which is what `npm run build` does,
/// and what `tests/e2e` runs before it builds the binary — leaves this table naming hashed files that
/// are gone, and the crate stops compiling. Watching does not self-retrigger: `sync_frontend`'s npm
/// run dirties the tree once, the next build re-runs this script, finds no source newer than the
/// build, and settles.
fn embed_spa(build: &Path, defect: Option<String>) {
    let mut files = Vec::new();
    walk(build, build, &mut files);
    files.sort();
    // An empty table is stated as a defect rather than left to be discovered one 404 at a time.
    // This is the only place that can tell an absent bundle from a present one, so it is where the
    // question is answered — the binary reads the verdict, and never re-derives it.
    let defect = defect.or_else(|| {
        files.is_empty().then(|| "no SPA was built — frontend/build is empty or absent".to_string())
    });
    println!("cargo:rerun-if-changed={}", build.display());
    for (_, abs) in &files {
        println!("cargo:rerun-if-changed={abs}");
    }
    let rows: String = files
        .iter()
        .map(|(url, abs)| format!("    ({url:?}, include_bytes!({abs:?})),\n"))
        .collect();
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("spa.rs");
    let verdict = match &defect {
        Some(d) => format!("Some({d:?})"),
        None => "None".to_string(),
    };
    std::fs::write(
        &out,
        format!(
            "pub static SPA: Spa = &[\n{rows}];\n\
             /// Why the embedded bundle must not be served, or `None`. Decided at build time, \
             where the sources are.\n\
             pub static SPA_DEFECT: Option<&str> = {verdict};\n"
        ),
    )
    .expect("write spa.rs");
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.filter_map(Result::ok) {
        let path = e.path();
        if path.is_dir() {
            // Watched so a file APPEARING in it re-runs this script — no file's own mtime can say
            // that, and a fresh bundle is all-new hashed names.
            println!("cargo:rerun-if-changed={}", path.display());
            walk(root, &path, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            // `/` in the URL on every platform, and the absolute path for `include_bytes!`.
            let url = rel.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/");
            out.push((url, path.to_string_lossy().into_owned()));
        }
    }
}
