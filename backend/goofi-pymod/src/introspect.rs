//! `goofi.introspect(path)` — the discovery probe: import a node module in THIS interpreter,
//! and return its declaration constants and GIL state as [`goofi_core::probe`] JSON.

use goofi_core::probe::{Introspection, OutSlot, Param, ParamSpec, Slot};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::loader::{find_node_class, module_from_path};
use crate::params::{BoolParam, DataType, FloatParam, InputSlot, IntParam, StringParam};

#[pyfunction]
pub fn introspect(py: Python<'_>, path: &str) -> PyResult<String> {
    // The import must precede the GIL sample: a node's declaration-time imports can re-enable
    // the GIL process-wide, and the routing gate has to see that.
    let module = module_from_path(py, path)?;
    let cls = find_node_class(py, &module)?;

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
        producer: cls.getattr("PRODUCER")?.extract()?,
        inputs: slots(&cls.getattr("INPUTS")?)?,
        outputs: out_slots(&cls.getattr("OUTPUTS")?)?,
        params: params(&cls.getattr("PARAMS")?)?,
    };
    serde_json::to_string(&intro)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// An input slot is declared as a bare `goofi.DataType` or as a `goofi.InputSlot`.
#[derive(FromPyObject)]
enum SlotDescr<'py> {
    Bare(Bound<'py, DataType>),
    Full(Bound<'py, InputSlot>),
}

/// `{name: DataType | InputSlot}` → the input slots. `multi` stays false: this tier has no
/// variadic plumbing to honour it.
fn slots(d: &Bound<'_, PyAny>) -> PyResult<Vec<Slot>> {
    d.cast::<PyDict>()?
        .iter()
        .map(|(k, v)| {
            let (kind, required, trigger) = match v.extract::<SlotDescr>()? {
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

/// A param descriptor is exactly one of our pyclasses, extracted typed.
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

// Here rather than in `goofi-tests`: reaching this needs the `host` feature, which links
// goofi-pymod as an rlib and brings the interpreter with it.
#[cfg(all(test, feature = "host"))]
mod tests {
    use super::*;

    /// The one `Slot` that `INPUTS = {"data": v}` yields.
    fn slot_of(py: Python<'_>, v: Bound<'_, PyAny>) -> Slot {
        let d = PyDict::new(py);
        d.set_item("data", v).unwrap();
        let mut got = slots(d.as_any()).expect("slots");
        assert_eq!(got.len(), 1, "one declared slot in, one out");
        got.pop().unwrap()
    }

    /// `goofi.InputSlot(DataType.ARRAY, **kwargs)` through the real constructor, so its own
    /// defaults are what the tests read back.
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
