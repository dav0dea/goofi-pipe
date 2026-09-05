//! The Python-facing param + slot-type descriptors a node states in its `PARAMS` and
//! `INPUTS` constants.

use pyo3::prelude::*;

/// `goofi.DataType.ARRAY|STRING|TABLE` — the slot element type; `.value` is the wire name.
/// The variants are the Python-facing API, so they stay screaming-case despite the acronym lint.
#[pyclass(eq, eq_int)]
#[derive(PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum DataType {
    ARRAY,
    STRING,
    TABLE,
}

#[pymethods]
impl DataType {
    #[getter]
    fn value(&self) -> &'static str {
        match self {
            DataType::ARRAY => "ARRAY",
            DataType::STRING => "STRING",
            DataType::TABLE => "TABLE",
        }
    }
}

/// `goofi.InputSlot(dtype, required=False, trigger=True)` — the per-slot options a bare
/// `goofi.DataType` has nowhere to put; the defaults are the bare form's behaviour.
#[pyclass]
pub struct InputSlot {
    #[pyo3(get)]
    pub dtype: Py<DataType>,
    #[pyo3(get)]
    pub required: bool,
    #[pyo3(get)]
    pub trigger: bool,
}

#[pymethods]
impl InputSlot {
    #[new]
    #[pyo3(signature = (dtype, required=false, trigger=true))]
    fn new(dtype: Py<DataType>, required: bool, trigger: bool) -> InputSlot {
        InputSlot { dtype, required, trigger }
    }
}

#[pyclass]
pub struct IntParam {
    #[pyo3(get)]
    pub default: i64,
    #[pyo3(get)]
    pub min: i64,
    #[pyo3(get)]
    pub max: i64,
    #[pyo3(get)]
    pub doc: Option<String>,
    #[pyo3(get)]
    pub expression: Option<String>,
}

#[pymethods]
impl IntParam {
    #[new]
    #[pyo3(signature = (default, min, max, doc=None, expression=None))]
    fn new(default: i64, min: i64, max: i64, doc: Option<String>, expression: Option<String>) -> IntParam {
        IntParam { default, min, max, doc, expression }
    }
}

#[pyclass]
pub struct FloatParam {
    #[pyo3(get)]
    pub default: f64,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub doc: Option<String>,
    #[pyo3(get)]
    pub expression: Option<String>,
}

#[pymethods]
impl FloatParam {
    #[new]
    #[pyo3(signature = (default, min, max, doc=None, expression=None))]
    fn new(default: f64, min: f64, max: f64, doc: Option<String>, expression: Option<String>) -> FloatParam {
        FloatParam { default, min, max, doc, expression }
    }
}

#[pyclass]
pub struct BoolParam {
    #[pyo3(get)]
    pub default: bool,
    #[pyo3(get)]
    pub doc: Option<String>,
    #[pyo3(get)]
    pub expression: Option<String>,
}

#[pymethods]
impl BoolParam {
    #[new]
    #[pyo3(signature = (default, doc=None, expression=None))]
    fn new(default: bool, doc: Option<String>, expression: Option<String>) -> BoolParam {
        BoolParam { default, doc, expression }
    }
}

#[pyclass]
pub struct StringParam {
    #[pyo3(get)]
    pub default: String,
    #[pyo3(get)]
    pub options: Vec<String>,
    #[pyo3(get)]
    pub refresh: bool,
    #[pyo3(get)]
    pub doc: Option<String>,
    #[pyo3(get)]
    pub expression: Option<String>,
}

#[pymethods]
impl StringParam {
    #[new]
    #[pyo3(signature = (default, options=None, refresh=false, doc=None, expression=None))]
    fn new(default: String, options: Option<Vec<String>>, refresh: bool, doc: Option<String>, expression: Option<String>) -> StringParam {
        StringParam { default, options: options.unwrap_or_default(), refresh, doc, expression }
    }
}

/// `goofi.PulseParam(doc=None)` — a request with no value; `pulse_<group>_<name>(self)` answers it.
#[pyclass]
pub struct PulseParam {
    #[pyo3(get)]
    pub doc: Option<String>,
}

#[pymethods]
impl PulseParam {
    #[new]
    #[pyo3(signature = (doc=None))]
    fn new(doc: Option<String>) -> PulseParam {
        PulseParam { doc }
    }
}
