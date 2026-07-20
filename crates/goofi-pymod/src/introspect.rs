//! `goofi.introspect(path)` — the discovery probe. Import a node module in THIS
//! interpreter, find its `Node` subclass, call the `config_*` hooks (real imports
//! available — that is the point), read the GIL state, and return the declarations
//! as JSON. Raises on any failure so the Rust discoverer greys the node out.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyType};

use crate::node::Node;

#[pyfunction]
pub fn introspect(py: Python<'_>, path: &str) -> PyResult<String> {
    // Load the module from an arbitrary file path via importlib.util.
    let module = load_module(py, path)?;

    // Find the single goofi.Node subclass defined in the module, instantiate it.
    let cls = find_node_subclass(py, &module)?;
    let instance = cls.call0()?;

    let inputs = slots_json(&instance.call_method0("config_input_slots")?, true)?;
    let outputs = slots_json(&instance.call_method0("config_output_slots")?, false)?;
    let params = params_json(&instance.call_method0("config_params")?)?;
    let doc: String = cls
        .getattr("__doc__")
        .ok()
        .and_then(|d| d.extract::<String>().ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let gil_safe = !py
        .import("sys")?
        .getattr("_is_gil_enabled")
        .and_then(|f| f.call0())
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(true);

    // Hand-assemble JSON (no serde on the Python side); values are simple + trusted.
    Ok(format!(
        r#"{{"gil_safe":{gil_safe},"doc":{doc},"inputs":{inputs},"outputs":{outputs},"params":{params}}}"#,
        doc = json_str(&doc),
    ))
}

fn load_module<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyModule>> {
    let util = py.import("importlib.util")?;
    let spec = util.call_method1("spec_from_file_location", ("goofi_probe_mod", path))?;
    if spec.is_none() {
        return Err(pyo3::exceptions::PyImportError::new_err(format!("cannot load {path}")));
    }
    let module = util.call_method1("module_from_spec", (&spec,))?;
    spec.getattr("loader")?.call_method1("exec_module", (&module,))?;
    module.cast_into::<PyModule>().map_err(Into::into)
}

/// The first strict `goofi.Node` subclass defined in the module, else raise.
fn find_node_subclass<'py>(py: Python<'py>, module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyType>> {
    let node_ty = py.get_type::<Node>();
    for (_name, val) in module.dict().iter() {
        let Ok(ty) = val.cast_into::<PyType>() else {
            continue;
        };
        if !ty.is(&node_ty) && ty.is_subclass(&node_ty)? {
            return Ok(ty);
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err("no goofi.Node subclass in module"))
}

/// `{name: DataType}` → a JSON array of slot objects. Inputs get `trigger`/`multi`
/// (defaulting true/false in M1); outputs carry only `name`/`kind`.
fn slots_json(d: &Bound<'_, PyAny>, is_input: bool) -> PyResult<String> {
    let dict = d.cast::<PyDict>()?;
    let mut items = Vec::new();
    for (k, v) in dict.iter() {
        let name: String = k.extract()?;
        let kind: String = v.getattr("value")?.extract()?;
        if is_input {
            items.push(format!(
                r#"{{"name":{},"kind":{},"trigger":true,"multi":false}}"#,
                json_str(&name),
                json_str(&kind),
            ));
        } else {
            items.push(format!(r#"{{"name":{},"kind":{}}}"#, json_str(&name), json_str(&kind)));
        }
    }
    Ok(format!("[{}]", items.join(",")))
}

/// `{group: {name: <Param descriptor>}}` → a flat JSON array of param objects.
fn params_json(d: &Bound<'_, PyAny>) -> PyResult<String> {
    let groups = d.cast::<PyDict>()?;
    let mut items = Vec::new();
    for (group, names) in groups.iter() {
        let group: String = group.extract()?;
        let names = names.cast::<PyDict>()?;
        for (name, descr) in names.iter() {
            let name: String = name.extract()?;
            items.push(param_json(&group, &name, &descr)?);
        }
    }
    Ok(format!("[{}]", items.join(",")))
}

fn param_json(group: &str, name: &str, d: &Bound<'_, PyAny>) -> PyResult<String> {
    let ty = d.get_type().name()?.to_string();
    let head = format!(r#""group":{},"name":{}"#, json_str(group), json_str(name));
    Ok(match ty.as_str() {
        "IntParam" => format!(
            r#"{{{head},"kind":"int","default":{},"min":{},"max":{}}}"#,
            d.getattr("default")?.extract::<i64>()?,
            d.getattr("min")?.extract::<i64>()?,
            d.getattr("max")?.extract::<i64>()?,
        ),
        "FloatParam" => format!(
            r#"{{{head},"kind":"float","default":{},"min":{},"max":{}}}"#,
            d.getattr("default")?.extract::<f64>()?,
            d.getattr("min")?.extract::<f64>()?,
            d.getattr("max")?.extract::<f64>()?,
        ),
        "BoolParam" => format!(
            r#"{{{head},"kind":"bool","default":{}}}"#,
            if d.getattr("default")?.extract::<bool>()? { "true" } else { "false" },
        ),
        "StringParam" => {
            let options: Vec<String> = d.getattr("options")?.extract()?;
            let opts = options.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(",");
            format!(
                r#"{{{head},"kind":"str","default":{},"options":[{opts}],"refresh":{}}}"#,
                json_str(&d.getattr("default")?.extract::<String>()?),
                if d.getattr("refresh")?.extract::<bool>()? { "true" } else { "false" },
            )
        }
        other => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!("unknown param type {other}")))
        }
    })
}

/// Minimal JSON string escaper (the values here are node/slot/param identifiers +
/// docstrings — escape the characters JSON requires).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
