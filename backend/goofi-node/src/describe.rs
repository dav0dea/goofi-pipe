//! A manifest as plain data: what a probe or a `describe()` symbol answers, leaked back into the
//! `'static` manifest every engine reads — and the rule that names a type after its file.

use std::path::{Path, PathBuf};

use goofi_core::probe;
use goofi_core::SlotType;

use crate::{NodeManifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl};

/// The type a node file names: a `.py` stem CamelCased, an `.rs` stem as written. `None` for a
/// file that is not a node file, one hidden by a `_` prefix, or a stem outside the name rule.
pub fn type_name_of(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    if stem.starts_with('_') {
        return None;
    }
    let name = match path.extension()?.to_str()? {
        "py" => camel(stem),
        "rs" => stem.to_string(),
        _ => return None,
    };
    goofi_core::globals::is_valid_name(&name).then_some(name)
}

/// The folder under a node source root that holds `engine`'s files.
pub fn folder_of(engine: &str) -> String {
    format!("nodes_{engine}")
}

/// Every node file in `dir`, sorted: its path, the type it names, and the stamp a rescan diffs.
pub fn node_files(dir: &Path) -> Vec<(PathBuf, String, Option<crate::Stamp>)> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(e) => {
            eprintln!("failed to read {}: {e}", dir.display());
            return Vec::new();
        }
    };
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| {
            let name = type_name_of(&p)?;
            let stamp = std::fs::metadata(&p).ok().and_then(|m| Some((m.len(), m.modified().ok()?)));
            Some((p, name, stamp))
        })
        .collect()
}

/// `snake_case` file stem → `CamelCase` palette type name.
fn camel(stem: &str) -> String {
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

/// Parse the introspection JSON.
pub fn parse_introspection(json: &str) -> Result<probe::Introspection, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

/// A manifest as the probe schema — the one description every out-of-crate node answers, as JSON.
pub fn describe(
    category: &str,
    doc: &str,
    inputs: &[SlotDecl],
    outputs: &[OutputDecl],
    params: &[ParamDecl],
    producer: bool,
) -> String {
    let intro = probe::Introspection {
        gil_safe: true,
        doc: doc.to_string(),
        category: Some(category.to_string()),
        producer,
        inputs: inputs
            .iter()
            .map(|s| probe::Slot {
                name: s.name.to_string(),
                kind: s.kind.name().to_string(),
                trigger: s.trigger_process,
                multi: s.multi,
                required: s.required,
            })
            .collect(),
        outputs: outputs
            .iter()
            .map(|o| probe::OutSlot { name: o.name.to_string(), kind: o.kind.name().to_string() })
            .collect(),
        params: params
            .iter()
            .map(|p| probe::Param {
                group: p.group.to_string(),
                name: p.name.to_string(),
                doc: p.doc.map(str::to_string),
                expression: p.expression.map(|e| e.source.to_string()),
                spec: match p.spec {
                    ParamSpec::Int { default, min, max } => probe::ParamSpec::Int { default, min, max },
                    ParamSpec::Float { default, min, max } => probe::ParamSpec::Float { default, min, max },
                    ParamSpec::Bool { default } => probe::ParamSpec::Bool { default },
                    ParamSpec::Str { default, options, refresh } => probe::ParamSpec::Str {
                        default: default.to_string(),
                        options: options.iter().map(|s| s.to_string()).collect(),
                        refresh,
                    },
                },
            })
            .collect(),
    };
    serde_json::to_string(&intro).expect("a manifest serializes")
}

/// The first slot name the name rule refuses, phrased for the palette — a reference spells
/// `node.slot` and an expression reads a slot as an attribute, so a bad name never registers.
pub fn illegal_slot(intro: &probe::Introspection) -> Option<String> {
    intro
        .inputs
        .iter()
        .map(|s| &s.name)
        .chain(intro.outputs.iter().map(|s| &s.name))
        .find(|n| !goofi_core::globals::is_valid_name(n))
        .map(|bad| format!("slot `{bad}` is not a legal name: {}", goofi_core::globals::NAME_RULE))
}

/// Leak a `'static &str` for the catalog's lifetime.
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Build a `'static NodeManifest` from an introspection.
pub fn leak_manifest(
    type_name: String,
    intro: &probe::Introspection,
    category: &'static str,
) -> &'static NodeManifest {
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
        category: intro.category.as_deref().map(leak_str).unwrap_or(category),
        doc: leak_str(&intro.doc),
        inputs: Box::leak(inputs.into_boxed_slice()),
        outputs: Box::leak(outputs.into_boxed_slice()),
        params: Box::leak(params.into_boxed_slice()),
        producer: intro.producer,
    }))
}

fn param_decl(p: &probe::Param) -> ParamDecl {
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
        expression: p.expression.as_deref().map(|src| crate::ExprDecl {
            source: leak_str(src),
            mode: crate::ExprMode::On,
            trigger: false,
        }),
        doc: p.doc.as_deref().map(leak_str),
    }
}
