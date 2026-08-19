//! Runtime node-discovery scaffolding shared by every discovery backend (in-process Python,
//! subprocess Python, …): the runtime factory type, the `snake_case`→`CamelCase` type-name rule, the
//! fixed `process(x)` I/O shape, and the `'static` manifest leak. A backend supplies only its own
//! seam — the validate predicate + the factory closure + its category/isolation — so the same file
//! yields the same palette type name whichever backend hosts it.

use goofi_core::probe;
use goofi_core::SlotType;

use crate::{Isolation, Node, NodeManifest, OutputDecl, ParamDecl, ParamGroups, ParamSpec, SlotDecl};

/// Builds a fresh boxed instance of a runtime-discovered node type from its params. A bare `fn`
/// pointer can't close over per-type state (a source string, a device handle), so this is a boxed
/// closure — shared by the engine's `register_dyn_type` and every discovery backend.
pub type NodeFactory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// `snake_case` file stem → `CamelCase` palette type name. One source of this rule, so the same file
/// yields the same type name whichever backend hosts it (in-process `PyNode` vs subprocess `RemoteNode`).
pub fn camel(stem: &str) -> String {
    stem.split('_')
        .filter(|s| !s.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Rich manifests from a `goofi.introspect` probe (the pymod discovery path) —
// the ONE manifest path both backends use: multi-slot + params read from a
// node's `config_*` hooks. See the unification spec.
// ---------------------------------------------------------------------------

/// Parse the introspection JSON (the shared [`Introspection`] schema). Any malformed field is
/// an error (→ grey-out).
pub fn parse_introspection(json: &str) -> Result<probe::Introspection, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

/// Leak a `'static &str` (catalog-lifetime; the discovered type set is bounded).
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Build a rich, multi-slot + param `'static NodeManifest` from an introspection. The `factory`
/// field is an unreachable stub — a discovered node is built by its registered [`NodeFactory`],
/// never `manifest.factory`. The leak is bounded (one manifest per discovered type, catalog-lifetime).
pub fn leak_manifest(
    type_name: String,
    intro: &probe::Introspection,
    category: &'static str,
    isolation: Isolation,
) -> &'static NodeManifest {
    fn stub() -> Box<dyn Node> {
        unreachable!("a discovered node is built by its registered factory, not manifest.factory")
    }
    let inputs: Vec<SlotDecl> = intro
        .inputs
        .iter()
        .map(|s| SlotDecl {
            name: leak_str(&s.name),
            kind: SlotType::from_name(&s.kind).unwrap_or(SlotType::Array),
            trigger_process: s.trigger,
            multi: s.multi,
            required: s.required,
        })
        .collect();
    let outputs: Vec<OutputDecl> = intro
        .outputs
        .iter()
        .map(|s| OutputDecl {
            name: leak_str(&s.name),
            kind: SlotType::from_name(&s.kind).unwrap_or(SlotType::Array),
        })
        .collect();
    let params: Vec<ParamDecl> = intro.params.iter().map(param_decl).collect();

    Box::leak(Box::new(NodeManifest {
        type_name: leak_str(&type_name),
        category,
        doc: leak_str(&intro.doc),
        inputs: Box::leak(inputs.into_boxed_slice()),
        outputs: Box::leak(outputs.into_boxed_slice()),
        params: Box::leak(params.into_boxed_slice()),
        isolation,
        producer: intro.producer,
        factory: stub,
    }))
}

fn param_decl(p: &probe::Param) -> ParamDecl {
    // Exhaustive over the tagged variants — each carries exactly its own fields, so there is
    // no defaulting/unknown-kind path (the old `serde_json::Value` juggling is gone).
    let spec = match &p.spec {
        probe::ParamSpec::Int { default, min, max } => {
            ParamSpec::Int { default: *default, min: *min, max: *max }
        }
        probe::ParamSpec::Float { default, min, max } => {
            ParamSpec::Float { default: *default, min: *min, max: *max }
        }
        probe::ParamSpec::Bool { default } => ParamSpec::Bool { default: *default },
        probe::ParamSpec::Str { default, options, refresh } => {
            let opts: Vec<&'static str> = options.iter().map(|s| leak_str(s)).collect();
            ParamSpec::Str {
                default: leak_str(default),
                options: Box::leak(opts.into_boxed_slice()),
                refresh: *refresh,
            }
        }
    };
    ParamDecl {
        group: leak_str(&p.group),
        name: leak_str(&p.name),
        spec,
        expression: None,
        doc: p.doc.as_deref().map(leak_str),
    }
}

// ---------------------------------------------------------------------------
// The unified probe-based discoverer: run `goofi.introspect` per node file, parse
// the JSON, and leak a manifest. Shared by the in-process + subprocess backends
// (which attach their own per-instance factory + route on `gil_safe` in M2/M3).
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// A discovered Python node type: its rich manifest + the routing flag (`gil_safe`
/// → in-process, else subprocess). The per-instance factory is attached by the
/// backend in M2/M3; M1 produces the manifest.
#[derive(Clone)]
pub struct Discovered {
    pub manifest: &'static NodeManifest,
    pub gil_safe: bool,
    pub source: PathBuf,
}

/// Why a node file could not be introspected, phrased for the palette tooltip. A
/// `ModuleNotFoundError` becomes just the module name; anything else keeps its LAST non-empty
/// line, which is where Python prints the exception line of a traceback.
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

/// Run `goofi.introspect(path)` in `python` and parse the result. `Err` carries WHY it failed —
/// a bad interpreter, a failed import (missing dep), no `Node` subclass, or malformed JSON — so
/// the palette can say what is wrong instead of the node silently not existing.
pub fn probe_introspect(path: &Path, python: &str) -> Result<probe::Introspection, String> {
    // The payload channel is a DUP of fd 1 taken before fd 1 is rerouted to stderr, so a node
    // whose dependency greets stdout on import (the pygame banner) cannot prepend itself to the
    // JSON — that would exit 0 with an unparseable payload and grey out a working node. The
    // reroute is at the fd level, not `redirect_stdout`, so a C extension writing straight to
    // fd 1 is caught too. Same answer the child loop already gives itself (`goofi.serve`).
    const PROBE: &str = "\
import goofi, os, sys
payload = os.fdopen(os.dup(1), 'wb')
os.dup2(2, 1)
sys.stdout = sys.stderr
payload.write(goofi.introspect(os.environ['GOOFI_INTROSPECT_PATH']).encode())
payload.close()
";
    let out = Command::new(python)
        .arg("-c")
        .arg(PROBE)
        .env("GOOFI_INTROSPECT_PATH", path)
        // The probe interpreter must import ITS OWN installed goofi + deps. A host `PYTHONPATH`
        // (e.g. `.cargo/config.toml` points the embedded FT interpreter at .gfivenv-ft) must not leak
        // in and shadow the probe python's packages with an incompatible cross-version build —
        // that would silently fail every import and grey out discovery. Mirrors the child spawn.
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

/// What a node file turned out to be. Three outcomes, not two: a file that is not a node at all
/// is different from a node that exists but cannot load, and the palette shows the latter.
pub enum Discovery {
    /// Not a node file: not `.py`, or `_`-prefixed (hidden).
    Skip,
    /// A node file whose probe failed — it is real, it just cannot run here. `reason` is a
    /// module name for a missing dependency, else the exception line.
    Unavailable { type_name: String, reason: String },
    Found(Discovered),
}

/// Discover one Python node file. The type name is the `CamelCase` file stem.
pub fn discover_one(
    path: &Path,
    python: &str,
    category: &'static str,
    isolation: Isolation,
) -> Discovery {
    if path.extension().and_then(|e| e.to_str()) != Some("py") {
        return Discovery::Skip;
    }
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { return Discovery::Skip };
    if stem.starts_with('_') {
        return Discovery::Skip;
    }
    match probe_introspect(path, python) {
        Ok(intro) => {
            let manifest = leak_manifest(camel(stem), &intro, category, isolation);
            Discovery::Found(Discovered {
                manifest,
                gil_safe: intro.gil_safe,
                source: path.to_path_buf(),
            })
        }
        Err(reason) => Discovery::Unavailable { type_name: camel(stem), reason },
    }
}
