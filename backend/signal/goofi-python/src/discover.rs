//! Python node discovery: one probe per file, in the interpreter that will run it, answering the
//! manifest the engine registers — or why it could not.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use goofi_core::probe;
use goofi_node::{illegal_slot, leak_manifest, parse_introspection, type_name_of, Isolation, IsolationCell, NodeManifest};
use sha2::{Digest, Sha256};

/// A discovered Python node type: its manifest, its tier cell — leaked per type, and written at
/// runtime when a node re-enables the GIL — plus the routing flag (`gil_safe` → in-process).
#[derive(Clone)]
pub struct Discovered {
    pub manifest: &'static NodeManifest,
    pub isolation: &'static IsolationCell,
    pub gil_safe: bool,
    pub source: PathBuf,
}

/// Why a node file could not be introspected, phrased for the palette tooltip.
fn probe_reason(stderr: &str) -> String {
    if let Some(rest) = stderr.split("No module named ").nth(1) {
        let name: String = rest.trim_start().trim_matches(['\'', '"']).chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
            .collect();
        if !name.is_empty() {
            return name;
        }
    }
    stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("probe failed").trim().to_string()
}

/// Run `goofi.introspect(path)` in `python` and parse the result; `Err` carries why it failed.
pub fn probe_introspect(path: &Path, python: &str) -> Result<probe::Introspection, String> {
    // The payload is a dup of fd 1 taken before fd 1 is rerouted to stderr, so anything an
    // import prints to stdout — even from a C extension — cannot corrupt the JSON.
    const PROBE: &str = "\
import goofi, os, sys
payload = os.fdopen(os.dup(1), 'wb')
os.dup2(2, 1)
sys.stdout = sys.stderr
payload.write(goofi.introspect(sys.argv[1]).encode())
payload.close()
";
    let out = Command::new(python)
        .arg("-c")
        .arg(PROBE)
        .arg(path)
        // A host `PYTHONPATH` must not shadow the probe interpreter's own goofi and deps.
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("could not run `{python}`: {e}"))?;
    if !out.status.success() {
        return Err(probe_reason(&String::from_utf8_lossy(&out.stderr)));
    }
    let json = String::from_utf8(out.stdout).map_err(|_| "probe emitted non-UTF-8".to_string())?;
    parse_introspection(&json).map_err(|e| format!("malformed introspection: {e}"))
}

/// What decides a probe's answer, hashed: goofi's version, the file's bytes, and each interpreter
/// with its site-packages — the directory an install moves, so an installed dependency re-probes
/// the file that lacked it. `None` only for a file that cannot be read.
pub fn probe_key(path: &Path, pythons: &[&str]) -> Option<String> {
    let mut hash = Sha256::new();
    hash.update(env!("CARGO_PKG_VERSION").as_bytes());
    hash.update(std::fs::read(path).ok()?);
    for python in pythons {
        hash.update(python.as_bytes());
        hash.update(mtime(Path::new(python)));
        let site = Path::new(python).parent().and_then(Path::parent).and_then(goofi_init::site_packages);
        hash.update(site.as_deref().map(mtime).unwrap_or_default());
    }
    Some(format!("{:x}", hash.finalize())[..32].to_string())
}

fn mtime(p: &Path) -> [u8; 16] {
    std::fs::metadata(p)
        .ok()
        .and_then(|m| m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or([0; 16], |t| t.as_nanos().to_le_bytes())
}

/// The probe under `memo`, where its answer outlives the process: an import-heavy node costs
/// seconds to probe, and no boot should pay that twice for one file under one interpreter.
fn introspect_memoised(path: &Path, python: &str, memo: &Path) -> Result<probe::Introspection, String> {
    let Some(key) = probe_key(path, &[python]) else { return probe_introspect(path, python) };
    let entry = memo.join(format!("{key}.json"));
    if let Some(hit) = std::fs::read_to_string(&entry).ok().and_then(|s| serde_json::from_str(&s).ok()) {
        return hit;
    }
    let answer = probe_introspect(path, python);
    if let Ok(json) = serde_json::to_string(&answer) {
        let tmp = memo.join(format!("{key}.{}.tmp", std::process::id()));
        let _ = std::fs::create_dir_all(memo)
            .and_then(|()| std::fs::write(&tmp, json))
            .and_then(|()| std::fs::rename(&tmp, &entry));
    }
    answer
}

/// What a node file turned out to be.
pub enum Discovery {
    /// Not a node file: not `.py`, or `_`-prefixed (hidden).
    Skip,
    /// A node file whose probe failed: `reason` is a missing module name, else the exception line.
    Unavailable { type_name: String, reason: String },
    Found(Discovered),
}

/// Discover one Python node file; the file names the type, and `memo` keeps the probe's answer.
pub fn discover_one(path: &Path, python: &str, isolation: Isolation, memo: &Path) -> Discovery {
    if path.extension().and_then(|e| e.to_str()) != Some("py") {
        return Discovery::Skip;
    }
    let Some(type_name) = type_name_of(path) else { return Discovery::Skip };
    match introspect_memoised(path, python, memo) {
        Ok(intro) => {
            if let Some(reason) = illegal_slot(&intro) {
                return Discovery::Unavailable { type_name, reason };
            }
            match leak_manifest(type_name.clone(), &intro) {
                Ok(manifest) => Discovery::Found(Discovered {
                    manifest,
                    isolation: IsolationCell::leak(isolation),
                    gil_safe: intro.gil_safe,
                    source: path.to_path_buf(),
                }),
                Err(reason) => Discovery::Unavailable { type_name, reason },
            }
        }
        Err(reason) => Discovery::Unavailable { type_name, reason },
    }
}
