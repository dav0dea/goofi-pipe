//! Runtime node-discovery scaffolding shared by every discovery backend (in-process Python,
//! subprocess Python, …): the runtime factory type, the `snake_case`→`CamelCase` type-name rule, the
//! fixed `process(x)` I/O shape, and the `'static` manifest leak. A backend supplies only its own
//! seam — the validate predicate + the factory closure + its category/isolation — so the same file
//! yields the same palette type name whichever backend hosts it.

use goofi_core::SlotType;
use serde::Deserialize;

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

/// The fixed I/O of a discovered `process(x)` node: one ARRAY `data` input that triggers a tick,
/// one ARRAY `out` output.
pub static PROCESS_IN: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
pub static PROCESS_OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

/// Leak a `'static` [`NodeManifest`] for a discovered process-node type. The I/O shape, empty
/// params, and an unreachable stub `factory` (a runtime type is built by its registered
/// [`NodeFactory`], never `manifest.factory`) are fixed; `category` / `isolation` / `doc` vary per
/// backend. The leak is bounded — one manifest per discovered type, catalog-lifetime.
pub fn leak_process_manifest(
    type_name: String,
    doc: String,
    category: &'static str,
    isolation: Isolation,
) -> &'static NodeManifest {
    fn stub() -> Box<dyn Node> {
        unreachable!("a discovered node is built by its registered factory, not manifest.factory")
    }
    Box::leak(Box::new(NodeManifest {
        type_name: Box::leak(type_name.into_boxed_str()),
        category,
        doc: Box::leak(doc.into_boxed_str()),
        inputs: PROCESS_IN,
        outputs: PROCESS_OUT,
        params: &[],
        isolation,
        factory: stub,
    }))
}

// ---------------------------------------------------------------------------
// Rich manifests from a `goofi.introspect` probe (the pymod discovery path).
// Generalizes `leak_process_manifest` (single-slot, no params) to multi-slot +
// params read from a node's `config_*` hooks. See the unification spec.
// ---------------------------------------------------------------------------

/// The JSON `goofi.introspect` emits for one node file.
#[derive(Debug, Deserialize)]
pub struct Introspection {
    pub gil_safe: bool,
    #[serde(default)]
    pub doc: String,
    pub inputs: Vec<SlotJson>,
    pub outputs: Vec<OutSlotJson>,
    pub params: Vec<ParamJson>,
}

#[derive(Debug, Deserialize)]
pub struct SlotJson {
    pub name: String,
    pub kind: String,
    pub trigger: bool,
    pub multi: bool,
}

#[derive(Debug, Deserialize)]
pub struct OutSlotJson {
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Deserialize)]
pub struct ParamJson {
    pub group: String,
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub default: serde_json::Value,
    #[serde(default)]
    pub min: serde_json::Value,
    #[serde(default)]
    pub max: serde_json::Value,
    #[serde(default)]
    pub options: Vec<String>,
    #[serde(default)]
    pub refresh: bool,
}

/// Parse the introspection JSON. Any malformed field is an error (→ grey-out).
pub fn parse_introspection(json: &str) -> Result<Introspection, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

/// Leak a `'static &str` (catalog-lifetime; the discovered type set is bounded).
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Build a rich, multi-slot + param `'static NodeManifest` from an introspection.
/// Generalizes [`leak_process_manifest`]; the `factory` field is the same
/// unreachable stub — a discovered node is built by its registered [`NodeFactory`],
/// never `manifest.factory`.
pub fn leak_manifest(
    type_name: String,
    intro: &Introspection,
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
        factory: stub,
    }))
}

fn param_decl(p: &ParamJson) -> ParamDecl {
    let spec = match p.kind.as_str() {
        "int" => ParamSpec::Int {
            default: p.default.as_i64().unwrap_or(0),
            min: p.min.as_i64().unwrap_or(i64::MIN),
            max: p.max.as_i64().unwrap_or(i64::MAX),
        },
        "float" => ParamSpec::Float {
            default: p.default.as_f64().unwrap_or(0.0),
            min: p.min.as_f64().unwrap_or(f64::MIN),
            max: p.max.as_f64().unwrap_or(f64::MAX),
        },
        "bool" => ParamSpec::Bool { default: p.default.as_bool().unwrap_or(false) },
        _ => {
            // str (and any unknown kind degrades to a free string).
            let opts: Vec<&'static str> = p.options.iter().map(|s| leak_str(s)).collect();
            ParamSpec::Str {
                default: leak_str(p.default.as_str().unwrap_or("")),
                options: Box::leak(opts.into_boxed_slice()),
                refresh: p.refresh,
            }
        }
    };
    ParamDecl { group: leak_str(&p.group), name: leak_str(&p.name), spec, default_expr: None }
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
pub struct Discovered {
    pub manifest: &'static NodeManifest,
    pub gil_safe: bool,
    pub source: PathBuf,
}

/// Run `goofi.introspect(path)` in `python` and parse the result. `None` on ANY
/// failure — a bad interpreter, a failed import (missing dep), no `Node` subclass,
/// or malformed JSON — so a broken node greys out instead of crashing the catalog.
pub fn probe_introspect(path: &Path, python: &str) -> Option<Introspection> {
    const PROBE: &str =
        "import goofi, os, sys; sys.stdout.write(goofi.introspect(os.environ['GOOFI_INTROSPECT_PATH']))";
    let out = Command::new(python)
        .arg("-c")
        .arg(PROBE)
        .env("GOOFI_INTROSPECT_PATH", path)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let json = String::from_utf8(out.stdout).ok()?;
    parse_introspection(&json).ok()
}

/// Discover one Python node file: non-`.py`, `_`-prefixed (hidden), or a probe
/// failure → `None`. The type name is the `CamelCase` file stem.
pub fn discover_one(
    path: &Path,
    python: &str,
    category: &'static str,
    isolation: Isolation,
) -> Option<Discovered> {
    if path.extension().and_then(|e| e.to_str()) != Some("py") {
        return None;
    }
    let stem = path.file_stem().and_then(|s| s.to_str())?;
    if stem.starts_with('_') {
        return None;
    }
    let intro = probe_introspect(path, python)?;
    let manifest = leak_manifest(camel(stem), &intro, category, isolation);
    Some(Discovered { manifest, gil_safe: intro.gil_safe, source: path.to_path_buf() })
}

/// Scan `dir` for node files (skipping `_`-prefixed / non-`.py` / probe failures),
/// deterministic order so type names are stable.
pub fn discover(
    dir: &Path,
    python: &str,
    category: &'static str,
    isolation: Isolation,
) -> std::io::Result<Vec<Discovered>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    Ok(entries
        .iter()
        .filter_map(|e| discover_one(&e.path(), python, category, isolation))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::SlotType;

    #[test]
    fn camel_case_conversion() {
        assert_eq!(camel("double"), "Double");
        assert_eq!(camel("my_band_filter"), "MyBandFilter");
        assert_eq!(camel(""), "");
        assert_eq!(camel("__weird__name"), "WeirdName");
    }

    const SAMPLE: &str = r#"{"gil_safe":true,"doc":"PSD",
        "inputs":[{"name":"data","kind":"ARRAY","trigger":true,"multi":false}],
        "outputs":[{"name":"psd","kind":"ARRAY"}],
        "params":[{"group":"welch","name":"nperseg","kind":"int","default":256,"min":16,"max":4096},
                  {"group":"welch","name":"tag","kind":"str","default":"a","options":["a","b"],"refresh":false}]}"#;

    #[test]
    fn parse_and_leak_builds_a_rich_manifest() {
        let intro = parse_introspection(SAMPLE).expect("parse");
        assert!(intro.gil_safe);
        let m = leak_manifest("Psd".into(), &intro, "python", Isolation::Subprocess);
        assert_eq!(m.type_name, "Psd");
        assert_eq!(m.doc, "PSD");
        assert_eq!(m.inputs.len(), 1);
        assert_eq!(m.inputs[0].name, "data");
        assert_eq!(m.inputs[0].kind, SlotType::Array);
        assert!(m.inputs[0].trigger_process);
        assert_eq!(m.outputs[0].name, "psd");
        assert_eq!(m.params.len(), 2);
        assert_eq!(m.params[0].group, "welch");
        assert_eq!(m.params[0].name, "nperseg");
        assert_eq!(m.params[1].name, "tag");
        // The int param carries its bounds; the str param carries its options.
        assert!(matches!(m.params[0].spec, crate::ParamSpec::Int { default: 256, min: 16, max: 4096 }));
        assert!(matches!(
            m.params[1].spec,
            crate::ParamSpec::Str { default: "a", options: [_, _], refresh: false }
        ));
    }
}
