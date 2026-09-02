//! Runtime node discovery shared by every backend: the factory type, the type-name rule, and the
//! `'static` manifest leak.

use goofi_core::probe;
use goofi_core::SlotType;

use goofi_node::{
    Isolation, IsolationCell, NodeManifest, OutputDecl, ParamDecl, ParamGroups, ParamSpec,
    SlotDecl,
};

use crate::Node;

/// Builds a fresh boxed instance of a runtime-discovered node type from its params.
pub type NodeFactory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// `snake_case` file stem → `CamelCase` palette type name.
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

/// Parse the introspection JSON.
pub fn parse_introspection(json: &str) -> Result<probe::Introspection, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
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
        category,
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
        expression: p.expression.as_deref().map(|src| goofi_node::ExprDecl {
            source: leak_str(src),
            mode: goofi_node::ExprMode::On,
            trigger: false,
        }),
        doc: p.doc.as_deref().map(leak_str),
    }
}

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
            let slots = intro.inputs.iter().map(|s| &s.name).chain(intro.outputs.iter().map(|s| &s.name));
            if let Some(bad) = slots.into_iter().find(|n| !goofi_core::globals::is_valid_name(n)) {
                let reason = format!("slot `{bad}` is not a legal name: {}", goofi_core::globals::NAME_RULE);
                return Discovery::Unavailable { type_name: camel(stem), reason };
            }
            let manifest = leak_manifest(camel(stem), &intro, category);
            Discovery::Found(Discovered {
                manifest,
                isolation: IsolationCell::leak(isolation),
                gil_safe: intro.gil_safe,
                source: path.to_path_buf(),
            })
        }
        Err(reason) => Discovery::Unavailable { type_name: camel(stem), reason },
    }
}
