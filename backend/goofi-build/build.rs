//! Embed the SDK crates' sources — what an authored node compiles against — with the workspace
//! manifest and lock they resolve under, and hash the lot: a changed SDK is a new cache key.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The crates a node crate reaches by path, in the layout the repo keeps them in.
const CRATES: &[&str] = &[
    "backend/goofi-view",
    "backend/goofi-core",
    "backend/goofi-node",
    "backend/goofi-codec",
    "backend/signal/goofi-signal-sdk",
    "backend/audio/goofi-audio-sdk",
];

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for dir in CRATES {
        let crate_dir = root.join(dir);
        println!("cargo:rerun-if-changed={}", crate_dir.join("Cargo.toml").display());
        files.push((format!("{dir}/Cargo.toml"), crate_dir.join("Cargo.toml")));
        walk(&crate_dir.join("src"), &format!("{dir}/src"), &mut files);
    }
    for name in ["Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", root.join(name).display());
    }
    let mut hash = Sha256::new();
    let mut rows = String::new();
    for (rel, abs) in &files {
        hash.update(rel.as_bytes());
        hash.update(std::fs::read(abs).expect("an embedded source reads"));
        rows += &format!("    ({rel:?}, include_bytes!({:?})),\n", abs.display().to_string());
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("embedded.rs");
    std::fs::write(
        &out,
        format!(
            "pub static SOURCES: &[(&str, &[u8])] = &[\n{rows}];\n\
             pub const SDK_HASH: &str = {:?};\n\
             pub static WORKSPACE_MANIFEST: &str = include_str!({:?});\n\
             pub static LOCK: &[u8] = include_bytes!({:?});\n",
            format!("{:x}", hash.finalize()),
            root.join("Cargo.toml").display().to_string(),
            root.join("Cargo.lock").display().to_string(),
        ),
    )
    .expect("write embedded.rs");
}

fn walk(dir: &Path, rel: &str, out: &mut Vec<(String, PathBuf)>) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.is_dir() {
            walk(&path, &format!("{rel}/{name}"), out);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
            out.push((format!("{rel}/{name}"), path));
        }
    }
}
