//! The shared Rust↔`goofi.Node` call marshalling both Python tiers run. It operates on
//! `goofi_core::Data`, so it is transport-agnostic; the caller owns the cast-warn dedup set.

use std::collections::HashSet;

use goofi_core::{warn_cast_once, Data as CoreData, Meta, Param, SrcDtype, Value};
use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::data::{array_to_f32, dict_to_meta, Data};

/// `group -> name -> Param` — the same shape as `goofi_node::ParamGroups`, so a caller passes
/// its own directly and pymod needs no dependency on goofi-node.
pub type Groups = IndexMap<String, IndexMap<String, Param>>;

/// Apply the live params, then call `node.setup()`.
pub fn run_setup(py: Python<'_>, instance: &Bound<'_, PyAny>, params: &Groups) -> PyResult<()> {
    apply_params(py, instance, params)?;
    instance.call_method0("setup")?;
    Ok(())
}

/// Re-enumerate a refreshable string param's options via the node's `refresh_{group}_{name}()`.
/// `None` for every non-answer: the param keeps the options it had.
pub fn run_refresh(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    params: &Groups,
    group: &str,
    name: &str,
) -> Option<Vec<String>> {
    let method = instance.getattr(format!("refresh_{group}_{name}").as_str()).ok()?;
    // Against the node's CURRENT settings: an unwired node never ticks, so its `self.params`
    // would otherwise stay frozen at construction time.
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
            eprintln!("refresh_{group}_{name}() raised: {e}");
            None
        }
    }
}

/// Apply the live params, call `node.process(**inputs)`, and marshal the return into per-slot
/// `Data`. `inputs` names every DECLARED slot in order, `None` where no frame arrived;
/// `out_slots` names the slot a bare (non-dict) return goes to.
pub fn run_process(
    py: Python<'_>,
    instance: &Bound<'_, PyAny>,
    params: &Groups,
    inputs: &[(&str, Option<&CoreData>)],
    out_slots: &[&str],
    warned: &mut HashSet<SrcDtype>,
) -> PyResult<Vec<(String, CoreData)>> {
    apply_params(py, instance, params)?;

    let kwargs = PyDict::new(py);
    for (name, core) in inputs {
        match core {
            Some(c) => kwargs.set_item(*name, Py::new(py, Data::from_core((*c).clone()))?)?,
            None => kwargs.set_item(*name, py.None())?,
        }
    }
    let ret = instance.call_method("process", (), Some(&kwargs))?;
    if ret.is_none() {
        return Ok(Vec::new());
    }

    // The first PRESENT frame, not the first declared slot: a node whose leading slot is unwired
    // must still carry meta from the input it did get.
    let primary = inputs.iter().find_map(|(_, c)| *c);
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

/// Set `instance.params` to nested `SimpleNamespace`, so a node reads `self.params.<group>.<name>`.
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
    })
}

/// Coerce one returned slot value into a core `Data`. `slot` names it in the cast warning.
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
    let (shape, bytes) = array_f32_bytes(py, v, slot, warned)?;
    let meta = match primary {
        Some(p) if primary_shape(p) == Some(shape.as_slice()) => p.meta().clone(),
        _ => Meta::empty(),
    };
    CoreData::array_f32(shape, bytes, meta)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// [`array_to_f32`] plus the deduped cast warning, which the `Data::new` path elides.
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
