//! The shared Rust↔`goofi.Node` call marshalling, used by BOTH in-process tiers:
//! goofi-py's `PyNode` (engine-tick-driven) and — in M3 — `goofi.serve`
//! (iceoryx2-driven). Writing it once is the point: the two tiers can't drift.
//! It operates on `goofi_core::Data`, so it is transport-agnostic; the caller owns
//! the cast-to-f32 warning dedup set.

use std::collections::HashSet;

use goofi_core::{warn_cast_once, Data as CoreData, Meta, Param, SrcDtype, Value};
use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::data::{array_to_f32, dict_to_meta, Data};

/// `group -> name -> Param`. Structurally identical to `goofi_node::ParamGroups` (a type
/// alias over the same `indexmap`/`goofi_core::Param`), so a caller passes its
/// `ParamGroups` directly — no pymod→goofi-node dependency.
pub type Groups = IndexMap<String, IndexMap<String, Param>>;

/// Apply the live params, then call `node.setup()`.
pub fn run_setup(py: Python<'_>, instance: &Bound<'_, PyAny>, params: &Groups) -> PyResult<()> {
    apply_params(py, instance, params)?;
    instance.call_method0("setup")?;
    Ok(())
}

/// Re-enumerate a refreshable string param's options by calling the node's
/// `refresh_{group}_{name}()` method — the Python side of the UI's ⟳ button, and the analogue
/// of the Rust `Node::on_param_refreshed` hook. Method naming (rather than a declared callback)
/// keeps the introspection probe carrying a plain `refresh: bool`.
///
/// `None` for every non-answer — no such method, a raise, or a return that isn't a list of
/// strings. A picker that can't enumerate right now (no soundcard, no LSL stream) is an ordinary
/// state, not a node failure: the param simply keeps the options it had.
pub fn run_refresh(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    params: &Groups,
    group: &str,
    name: &str,
) -> Option<Vec<String>> {
    let method = instance.getattr(format!("refresh_{group}_{name}").as_str()).ok()?;
    // Enumerate against the node's CURRENT settings, exactly as `process` would see them — a
    // node whose input is unwired never ticks, so without this its `self.params` would be frozen
    // at construction time forever.
    if let Err(e) = apply_params(py, instance, params) {
        eprintln!("refresh_{group}_{name}: could not apply params: {e}");
        return None;
    }
    match method.call0() {
        Ok(v) => match v.extract::<Vec<String>>() {
            Ok(options) => Some(options),
            Err(_) => {
                eprintln!("refresh_{group}_{name}() must return a list of strings; ignoring");
                None
            }
        },
        Err(e) => {
            // No UI warn channel exists for this (the ⟳ button reports only "no new options"),
            // so surface the traceback on stderr rather than swallowing it whole.
            eprintln!("refresh_{group}_{name}() raised: {e}");
            None
        }
    }
}

/// Apply the live params, call `node.process(**present)`, and marshal the return into
/// per-slot `Data`. `present` is `(slot, &Data)` for each PRESENT input slot (built into
/// `goofi.Data` kwargs). `out_slots` are the node's declared output slot names (used when
/// the node returns a bare value instead of a `{slot: value}` dict). `warned` is the
/// caller's per-node dedup set for the cast-to-f32 warning.
///
/// Return coercion, per slot value:
///
/// - a `goofi.Data`    → its (already-f32) core, verbatim;
/// - a `(array, meta)` → `array` cast to f32 (+ warn) with the explicit meta dict;
/// - a bare array-like → cast to f32 (+ warn); meta carried from the first present
///   input iff the output shape matches it, else empty.
///
/// A `None` return emits nothing.
pub fn run_process(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    params: &Groups,
    present: &[(&str, &CoreData)],
    out_slots: &[&str],
    warned: &mut HashSet<SrcDtype>,
) -> PyResult<Vec<(String, CoreData)>> {
    apply_params(py, instance, params)?;

    let kwargs = PyDict::new(py);
    for (name, core) in present {
        let d = Py::new(py, Data::from_core((*core).clone()))?;
        kwargs.set_item(*name, d)?;
    }
    let ret = instance.call_method("process", (), Some(&kwargs))?;
    if ret.is_none() {
        return Ok(Vec::new());
    }

    let primary = present.first().map(|(_, c)| *c);
    if let Ok(dict) = ret.cast::<PyDict>() {
        let mut outs = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let slot: String = k.extract()?;
            let core = value_to_core(py, &v, &slot, primary, warned)?;
            outs.push((slot, core));
        }
        Ok(outs)
    } else {
        let [slot] = out_slots else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "process() returned a bare value but the node does not have exactly one output slot; return {slot: value}",
            ));
        };
        let core = value_to_core(py, &ret, slot, primary, warned)?;
        Ok(vec![(slot.to_string(), core)])
    }
}

/// Build `types.SimpleNamespace(group=SimpleNamespace(name=value, …), …)` from the live
/// params and set it as `instance.params`, so a node reads `self.params.<group>.<name>`.
fn apply_params(py: Python<'_>, instance: &Bound<'_, PyAny>, params: &Groups) -> PyResult<()> {
    let types = py.import("types")?;
    let simple_ns = types.getattr("SimpleNamespace")?;
    let outer = PyDict::new(py);
    for (group, entries) in params {
        let inner = PyDict::new(py);
        for (name, p) in entries {
            inner.set_item(name, param_to_py(py, p)?)?;
        }
        outer.set_item(group, simple_ns.call((), Some(&inner))?)?;
    }
    instance.setattr("params", simple_ns.call((), Some(&outer))?)?;
    Ok(())
}

fn param_to_py<'py>(py: Python<'py>, p: &Param) -> PyResult<Bound<'py, PyAny>> {
    Ok(match p {
        Param::Float { value, .. } => value.into_bound_py_any(py)?,
        Param::Int { value, .. } => value.into_bound_py_any(py)?,
        Param::Bool { value } => value.into_bound_py_any(py)?,
        Param::Str { value, .. } => value.into_bound_py_any(py)?,
        Param::Trigger { fired } => fired.into_bound_py_any(py)?,
    })
}

/// Coerce one returned slot value (a `goofi.Data`, a `(array, meta)` tuple, or a bare
/// array-like) into a core `Data`. `slot` names it in the cast warning.
fn value_to_core(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    slot: &str,
    primary: Option<&CoreData>,
    warned: &mut HashSet<SrcDtype>,
) -> PyResult<CoreData> {
    if let Ok(d) = v.cast::<Data>() {
        return Ok(d.borrow().core().clone());
    }
    if let Ok(tup) = v.cast::<PyTuple>() {
        if tup.len() == 2 {
            let arr = tup.get_item(0)?;
            let meta_obj = tup.get_item(1)?;
            let meta = if meta_obj.is_none() {
                Meta::empty()
            } else {
                dict_to_meta(meta_obj.cast::<PyDict>()?)?
            };
            let (shape, bytes) = array_f32_bytes(py, &arr, slot, warned)?;
            return CoreData::array_f32(shape, bytes, meta)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
        }
    }
    // Bare array-like: carry the primary input's meta iff the output shape matches it.
    let (shape, bytes) = array_f32_bytes(py, v, slot, warned)?;
    let meta = match primary {
        Some(p) if primary_shape(p) == Some(shape.as_slice()) => p.meta().clone(),
        _ => Meta::empty(),
    };
    CoreData::array_f32(shape, bytes, meta)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// `np.ascontiguousarray(arr)` → `(shape, f32 bytes)` via the shared [`array_to_f32`]
/// funnel, adding the deduped cast warning (Rust-owned, so both tiers share the one
/// guard — replaces the retired `WRAP_SRC`). The `Data::new` path uses the same funnel
/// but elides the warn.
fn array_f32_bytes(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    slot: &str,
    warned: &mut HashSet<SrcDtype>,
) -> PyResult<(Vec<usize>, Vec<u8>)> {
    let (src, shape, bytes) = array_to_f32(py, arr)?;
    warn_cast_once(warned, slot, src);
    Ok((shape, bytes))
}

fn primary_shape(d: &CoreData) -> Option<&[usize]> {
    match d.value() {
        Value::Array(s) => Some(s.shape()),
        _ => None,
    }
}
