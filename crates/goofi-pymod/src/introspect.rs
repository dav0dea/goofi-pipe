//! `goofi.introspect(path)` — the discovery probe. Import a node module in THIS
//! interpreter, find its `Node` subclass, call the `config_*` hooks (real imports
//! available — that is the point), read the GIL state, and return the declarations as
//! JSON. Raises on any failure so the Rust discoverer greys the node out.
//!
//! The JSON is the shared [`goofi_core::probe`] schema, `serde_json`-serialized — so it
//! can't drift from the discoverer that parses it, and there is no hand-rolled escaper.
//! Param descriptors are read by TYPED extraction into a closed `ParamDescr` enum, so
//! "which kind is this param" is answered by the type system, not a string match with an
//! error arm.

use goofi_core::probe::{Introspection, OutSlot, Param, ParamSpec, Slot};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::loader::{find_node_class, module_from_path};
use crate::params::{BoolParam, FloatParam, IntParam, StringParam};

#[pyfunction]
pub fn introspect(py: Python<'_>, path: &str) -> PyResult<String> {
    let module = module_from_path(py, path)?;
    let cls = find_node_class(py, &module)?;
    let instance = cls.call0()?;

    // Run the hooks FIRST, then sample the GIL. A hook is where a node does its
    // declaration-time imports, and importing a C extension built without free-threading
    // support re-enables the GIL process-wide — the routing gate has to see that. (Struct
    // literal fields evaluate in written order, so this cannot be inlined below.)
    let inputs = slots(&instance.call_method0("config_input_slots")?)?;
    let outputs = out_slots(&instance.call_method0("config_output_slots")?)?;
    let params = params(&instance.call_method0("config_params")?)?;

    let intro = Introspection {
        gil_safe: !py
            .import("sys")?
            .getattr("_is_gil_enabled")
            .and_then(|f| f.call0())
            .and_then(|v| v.extract::<bool>())
            .unwrap_or(true),
        doc: cls
            .getattr("__doc__")
            .ok()
            .and_then(|d| d.extract::<String>().ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default(),
        inputs,
        outputs,
        params,
    };
    serde_json::to_string(&intro)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// `{name: DataType}` → the input slots (M1: every input triggers, none is `multi`).
fn slots(d: &Bound<'_, PyAny>) -> PyResult<Vec<Slot>> {
    d.cast::<PyDict>()?
        .iter()
        .map(|(k, v)| {
            Ok(Slot { name: k.extract()?, kind: slot_kind(&v)?, trigger: true, multi: false })
        })
        .collect()
}

/// `{name: DataType}` → the output slots.
fn out_slots(d: &Bound<'_, PyAny>) -> PyResult<Vec<OutSlot>> {
    d.cast::<PyDict>()?
        .iter()
        .map(|(k, v)| Ok(OutSlot { name: k.extract()?, kind: slot_kind(&v)? }))
        .collect()
}

/// A slot value is a `goofi.DataType`; its `.value` is the wire kind (`"ARRAY"`/…).
fn slot_kind(v: &Bound<'_, PyAny>) -> PyResult<String> {
    v.getattr("value")?.extract()
}

/// `{group: {name: <Param descriptor>}}` → a flat list of typed `Param`.
fn params(d: &Bound<'_, PyAny>) -> PyResult<Vec<Param>> {
    let mut out = Vec::new();
    for (group, names) in d.cast::<PyDict>()?.iter() {
        let group: String = group.extract()?;
        for (name, descr) in names.cast::<PyDict>()?.iter() {
            let (spec, doc) = param_spec(&descr)?;
            out.push(Param { group: group.clone(), name: name.extract()?, doc, spec });
        }
    }
    Ok(out)
}

/// A param descriptor is exactly one of our pyclasses — extract it typed, so the "which
/// kind" decision is exhaustive at the type level (a non-descriptor is a clean extract error,
/// not an `else` branch).
#[derive(FromPyObject)]
enum ParamDescr<'py> {
    Int(Bound<'py, IntParam>),
    Float(Bound<'py, FloatParam>),
    Bool(Bound<'py, BoolParam>),
    Str(Bound<'py, StringParam>),
}

/// The kind-specific spec plus the kind-independent `doc=` help text.
fn param_spec(descr: &Bound<'_, PyAny>) -> PyResult<(ParamSpec, Option<String>)> {
    Ok(match descr.extract::<ParamDescr>()? {
        ParamDescr::Int(p) => {
            let p = p.borrow();
            (ParamSpec::Int { default: p.default, min: p.min, max: p.max }, p.doc.clone())
        }
        ParamDescr::Float(p) => {
            let p = p.borrow();
            (ParamSpec::Float { default: p.default, min: p.min, max: p.max }, p.doc.clone())
        }
        ParamDescr::Bool(p) => {
            let p = p.borrow();
            (ParamSpec::Bool { default: p.default }, p.doc.clone())
        }
        ParamDescr::Str(p) => {
            let p = p.borrow();
            (
                ParamSpec::Str { default: p.default.clone(), options: p.options.clone(), refresh: p.refresh },
                p.doc.clone(),
            )
        }
    })
}
