//! `PyExprEvaluator` — the pyo3 param-expression evaluator the engine injects.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use goofi_core::{Data, Param, Value};
use goofi_node::{BindingId, Compiled, EvalCtx, ExprError, ExprEvaluator, Local};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule, PyString};

/// The Python harness. The graph has already rewritten every `nd(..)` and `globals.*` term into a
/// generated variable, so the expression is plain math over ordinary locals — with `np`, `math`'s
/// whole namespace and `time()` simply there. ONE dict as eval's globals, deliberately: a split
/// globals/locals pair breaks name lookup inside comprehensions. Locals land last, so a node
/// named `sin` shadows math's.
const EVAL_SRC: &str = r#"
import numpy as np
from math import *
from time import time

__goofi_scope = {k: v for k, v in globals().items() if not k.startswith("__")}

def __goofi_compile(source):
    return compile(source, "<goofi-expr>", "eval")

def __goofi_eval(code, locals_, t):
    ns = dict(__goofi_scope)
    ns["t"] = t
    ns.update(locals_)
    return eval(code, ns)
"#;

/// The pyo3 evaluator: the harness functions plus the code objects keyed by [`BindingId`].
pub struct PyExprEvaluator {
    compile_fn: Py<PyAny>,
    eval_fn: Py<PyAny>,
    codes: Mutex<HashMap<u64, Py<PyAny>>>,
    next: AtomicU64,
}

impl PyExprEvaluator {
    pub fn new() -> PyResult<PyExprEvaluator> {
        crate::attach(|py| {
            let m = PyModule::from_code(
                py,
                CString::new(EVAL_SRC)?.as_c_str(),
                c"goofi_expr.py",
                c"goofi_expr",
            )?;
            Ok(PyExprEvaluator {
                compile_fn: m.getattr("__goofi_compile")?.unbind(),
                eval_fn: m.getattr("__goofi_eval")?.unbind(),
                codes: Mutex::new(HashMap::new()),
                next: AtomicU64::new(1),
            })
        })
    }
}

/// Convert a resolved `Data` to a Python object; a table is unsupported and reads as `None`.
fn data_to_py(py: Python<'_>, d: &Data) -> PyResult<Py<PyAny>> {
    match d.value() {
        Value::Array(s) => {
            let np = PyModule::import(py, "numpy")?;
            let raw = PyBytes::new(py, s.as_bytes());
            let arr = np.getattr("frombuffer")?.call1((raw, "<f4"))?; // arrays are always f32
            let shape: Vec<usize> = s.shape().to_vec();
            Ok(arr.call_method1("reshape", (shape,))?.unbind())
        }
        Value::Str(st) => Ok(PyString::new(py, st.as_ref()).into_any().unbind()),
        Value::Table(_) => Ok(py.None()),
    }
}

/// Convert a resolved `Param` to a native Python scalar.
fn param_to_py(py: Python<'_>, p: &Param) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObject;
    Ok(match p {
        Param::Float { value, .. } => value.into_pyobject(py)?.into_any().unbind(),
        Param::Int { value, .. } => value.into_pyobject(py)?.into_any().unbind(),
        Param::Bool { value } => value.into_pyobject(py)?.to_owned().into_any().unbind(),
        Param::Str { value, .. } => PyString::new(py, value).into_any().unbind(),
        Param::Pulse => py.None(),
    })
}

/// Extract a scalar `T`, falling back to `.item()` on a size-1 array because numpy 2.x rejects
/// `float(np.array([x]))` and goofi promotes every scalar to a shape-[1] array.
fn to_scalar<'py, T>(result: &Bound<'py, PyAny>, noun: &str) -> Result<T, String>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    if let Ok(v) = result.extract::<T>() {
        return Ok(v);
    }
    let not_a = || format!("expression result is not a {noun}");
    let np = PyModule::import(result.py(), "numpy").map_err(|e| e.to_string())?;
    let a = np.getattr("asarray").and_then(|f| f.call1((result,))).map_err(|_| not_a())?;
    let size: usize = a.getattr("size").and_then(|s| s.extract()).map_err(|_| not_a())?;
    if size != 1 {
        return Err(format!("expression result is not a scalar {noun} (size {size})"));
    }
    a.call_method0("item").and_then(|it| it.extract::<T>()).map_err(|_| not_a())
}

/// Coerce the Python result to the target param's type.
fn coerce(result: &Bound<'_, PyAny>, target: &Param) -> Result<Param, String> {
    match target {
        Param::Float { vmin, vmax, .. } => {
            Ok(Param::Float { value: to_scalar::<f64>(result, "number")?, vmin: *vmin, vmax: *vmax })
        }
        Param::Int { vmin, vmax, .. } => {
            let v = to_scalar::<f64>(result, "number")?;
            // `as i64` silently saturates NaN and ±inf; error instead.
            if !v.is_finite() {
                return Err("expression result is not a finite number".to_string());
            }
            Ok(Param::Int { value: v.round() as i64, vmin: *vmin, vmax: *vmax })
        }
        // A pulse is a GATE to an expression: the runtime fires on the rise of this bool.
        Param::Bool { .. } | Param::Pulse => Ok(Param::Bool { value: to_scalar::<bool>(result, "bool")? }),
        Param::Str { options, refresh, .. } => {
            let v: String = result.extract().map_err(|_| "expression result is not a string".to_string())?;
            Ok(Param::Str { value: v, options: options.clone(), refresh: *refresh })
        }
    }
}

impl ExprEvaluator for PyExprEvaluator {
    fn compile(&self, source: &str) -> Result<Compiled, ExprError> {
        crate::attach(|py| -> Result<Compiled, ExprError> {
            let code = self
                .compile_fn
                .bind(py)
                .call1((source,))
                .map_err(|e| ExprError(e.to_string()))?;
            let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
            self.codes.lock().unwrap().insert(id, code.unbind());
            Ok(Compiled { id })
        })
    }

    fn eval(&self, id: BindingId, ctx: &EvalCtx<'_>) -> Result<Param, ExprError> {
        crate::attach(|py| -> Result<Param, ExprError> {
            let code = self
                .codes
                .lock()
                .unwrap()
                .get(&id)
                .map(|c| c.clone_ref(py))
                .ok_or_else(|| ExprError("expression not compiled".into()))?;
            let locals = PyDict::new(py);
            for (name, local) in ctx.locals {
                let val: Py<PyAny> = match local {
                    Some(Local::Frame(d)) => data_to_py(py, d).map_err(|e| ExprError(e.to_string()))?,
                    Some(Local::Value(p)) => param_to_py(py, p).map_err(|e| ExprError(e.to_string()))?,
                    None => py.None(),
                };
                locals.set_item(name.as_str(), val).map_err(|e| ExprError(e.to_string()))?;
            }
            let result = self
                .eval_fn
                .bind(py)
                .call1((code.bind(py), &locals, ctx.t))
                .map_err(|e| ExprError(e.to_string()))?;
            coerce(&result, ctx.target).map_err(ExprError)
        })
    }

    fn release(&self, id: BindingId) {
        self.codes.lock().unwrap().remove(&id);
    }
}
