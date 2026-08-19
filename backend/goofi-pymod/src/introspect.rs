//! `goofi.introspect(path)` — the discovery probe. Import a node module in THIS interpreter (real
//! imports available — that is the point), find its `Node` subclass, read its
//! `goofi.Manifest` and the GIL state, and return the declarations as JSON. Raises on any failure
//! so the Rust discoverer greys the node out.
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
use crate::params::{BoolParam, DataType, FloatParam, InputSlot, IntParam, StringParam};

#[pyfunction]
pub fn introspect(py: Python<'_>, path: &str) -> PyResult<String> {
    // The IMPORT is what has to happen before the GIL is sampled: a node's declaration-time
    // imports run in its module body and in its class body, and importing a C extension built
    // without free-threading support re-enables the GIL process-wide — which the routing gate has
    // to see. Reading the manifest afterwards cannot change that answer, because a class attribute
    // was already evaluated by the import that produced it.
    let module = module_from_path(py, path)?;
    let cls = find_node_class(py, &module)?;
    let m = cls.getattr("manifest")?;

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
        producer: m.getattr("producer")?.extract()?,
        inputs: slots(&m.getattr("inputs")?)?,
        outputs: out_slots(&m.getattr("outputs")?)?,
        params: params(&m.getattr("params")?)?,
    };
    serde_json::to_string(&intro)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// An input slot is declared either as a bare `goofi.DataType` — still the whole of it for a
/// node with nothing to say beyond the type — or as a `goofi.InputSlot` carrying the per-slot
/// options. Extracted typed, like [`ParamDescr`], so "which form is this" is a type-level
/// question and anything else is a clean extract error, not an `else` arm.
#[derive(FromPyObject)]
enum SlotDescr<'py> {
    Bare(Bound<'py, DataType>),
    Full(Bound<'py, InputSlot>),
}

/// `{name: DataType | InputSlot}` → the input slots. `multi` stays false: it is not authorable
/// from Python, because this tier has no variadic plumbing to honour it.
fn slots(d: &Bound<'_, PyAny>) -> PyResult<Vec<Slot>> {
    d.cast::<PyDict>()?
        .iter()
        .map(|(k, v)| {
            let (kind, required, trigger) = match v.extract::<SlotDescr>()? {
                // The bare form's answers are exactly `InputSlot`'s defaults, so a node written
                // before `InputSlot` existed declares the same slot it always did.
                SlotDescr::Bare(t) => (slot_kind(t.as_any())?, false, true),
                SlotDescr::Full(s) => {
                    let s = s.borrow();
                    (slot_kind(s.dtype.bind(v.py()).as_any())?, s.required, s.trigger)
                }
            };
            Ok(Slot { name: k.extract()?, kind, trigger, multi: false, required })
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

// The interpreter these run in comes from the `host` feature (pyo3 auto-initialize); the
// `extension-module` build links no libpython and cannot host one.
// The suite lives in `goofi-tests`. This block stays because goofi-pymod is the `goofi` PYTHON
// package: reaching it from Rust needs the `host` feature, which links it as an rlib, and
// `cargo test -p goofi-pymod --features host` is the documented way to run exactly this.
#[cfg(all(test, feature = "host"))]
mod tests {
    use super::*;

    /// The one `Slot` that `config_input_slots()` returning `{"data": v}` yields.
    fn slot_of(py: Python<'_>, v: Bound<'_, PyAny>) -> Slot {
        let d = PyDict::new(py);
        d.set_item("data", v).unwrap();
        let mut got = slots(d.as_any()).expect("slots");
        assert_eq!(got.len(), 1, "one declared slot in, one out");
        got.pop().unwrap()
    }

    /// `goofi.InputSlot(DataType.ARRAY, **kwargs)` built through the real constructor, so the
    /// signature's own defaults are what the tests read back.
    fn input_slot<'py>(py: Python<'py>, kwargs: &Bound<'py, PyDict>) -> Bound<'py, PyAny> {
        let dtype = Bound::new(py, DataType::ARRAY).unwrap();
        py.get_type::<InputSlot>().call((dtype,), Some(kwargs)).expect("InputSlot(…)")
    }

    #[test]
    fn a_bare_datatype_keeps_the_pre_inputslot_behaviour() {
        Python::attach(|py| {
            let s = slot_of(py, Bound::new(py, DataType::ARRAY).unwrap().into_any());
            assert_eq!(s.kind, "ARRAY");
            assert!(!s.required, "a bare DataType declares nothing, so it requires nothing");
            assert!(s.trigger, "and it triggers, as every Python slot did before InputSlot existed");
            assert!(!s.multi);
        });
    }

    #[test]
    fn an_input_slot_can_declare_itself_required() {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("required", true).unwrap();
            let s = slot_of(py, input_slot(py, &kwargs));
            assert_eq!(s.kind, "ARRAY");
            assert!(s.required);
            assert!(s.trigger, "required does not touch trigger (D2)");
        });
    }

    #[test]
    fn an_input_slot_can_opt_out_of_triggering() {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("trigger", false).unwrap();
            let s = slot_of(py, input_slot(py, &kwargs));
            assert!(!s.trigger, "trigger was hard-coded true before; it is authorable now");
            assert!(!s.required, "and trigger does not touch required either (D2)");
        });
    }
}
