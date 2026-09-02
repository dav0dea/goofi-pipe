//! A manifest as plain data: what a probe or a `describe()` symbol answers, leaked back into the
//! `'static` manifest every engine reads — and the rule that names a type after its file.

use std::path::Path;

use goofi_core::probe;
use goofi_core::SlotType;

use crate::{NodeManifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl};

/// The type a node file names: a `.py` stem CamelCased, an `.rs` stem as written. `None` for a
/// file that is not a node file, or one hidden by a `_` prefix.
pub fn type_name_of(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    if stem.starts_with('_') {
        return None;
    }
    match path.extension()?.to_str()? {
        "py" => Some(camel(stem)),
        "rs" => Some(stem.to_string()),
        _ => None,
    }
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
