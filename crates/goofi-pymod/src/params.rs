//! The Python-facing param + slot-type descriptors a node returns from its
//! `config_params` / `config_input_slots` hooks. Plain data holders — `introspect`
//! reads their attributes to build the Rust manifest.

use pyo3::prelude::*;

/// `goofi.DataType.ARRAY|STRING|TABLE` — the slot element type. `.value` is the
/// wire name (`"ARRAY"`/…) so introspect serializes it directly. The variant names
/// are the Python-facing API (`goofi.DataType.ARRAY`) + wire names, so they stay
/// screaming-case despite the acronym lint.
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

#[pyclass]
pub struct IntParam {
    #[pyo3(get)]
    pub default: i64,
    #[pyo3(get)]
    pub min: i64,
    #[pyo3(get)]
    pub max: i64,
}

#[pymethods]
impl IntParam {
    #[new]
    fn new(default: i64, min: i64, max: i64) -> IntParam {
        IntParam { default, min, max }
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
}

#[pymethods]
impl FloatParam {
    #[new]
    fn new(default: f64, min: f64, max: f64) -> FloatParam {
        FloatParam { default, min, max }
    }
}

#[pyclass]
pub struct BoolParam {
    #[pyo3(get)]
    pub default: bool,
}

#[pymethods]
impl BoolParam {
    #[new]
    fn new(default: bool) -> BoolParam {
        BoolParam { default }
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
}

#[pymethods]
impl StringParam {
    #[new]
    #[pyo3(signature = (default, options=None, refresh=false))]
    fn new(default: String, options: Option<Vec<String>>, refresh: bool) -> StringParam {
        StringParam { default, options: options.unwrap_or_default(), refresh }
    }
}
