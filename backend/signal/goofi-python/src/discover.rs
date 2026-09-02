//! Python node discovery: one probe per file, in the interpreter that will run it, answering the
//! manifest the engine registers — or why it could not.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use goofi_core::probe;
use goofi_node::{illegal_slot, leak_manifest, parse_introspection, type_name_of, Isolation, IsolationCell, NodeManifest};

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

/// What a node file turned out to be.
pub enum Discovery {
    /// Not a node file: not `.py`, or `_`-prefixed (hidden).
    Skip,
    /// A node file whose probe failed: `reason` is a missing module name, else the exception line.
    Unavailable { type_name: String, reason: String },
    Found(Discovered),
}

/// Discover one Python node file; the file names the type.
pub fn discover_one(
    path: &Path,
    python: &str,
    category: &'static str,
    isolation: Isolation,
) -> Discovery {
    if path.extension().and_then(|e| e.to_str()) != Some("py") {
        return Discovery::Skip;
    }
    let Some(type_name) = type_name_of(path) else { return Discovery::Skip };
    match probe_introspect(path, python) {
        Ok(intro) => {
            if let Some(reason) = illegal_slot(&intro) {
                return Discovery::Unavailable { type_name, reason };
            }
            let manifest = leak_manifest(type_name, &intro, category);
            Discovery::Found(Discovered {
                manifest,
                isolation: IsolationCell::leak(isolation),
                gil_safe: intro.gil_safe,
                source: path.to_path_buf(),
            })
        }
        Err(reason) => Discovery::Unavailable { type_name, reason },
    }
}
