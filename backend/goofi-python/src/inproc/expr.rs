//! `PyExprEvaluator` — the pyo3 param-expression evaluator (the `goofi_node::
//! ExprEvaluator` the engine injects). Runs each expression in the free-threaded
//! interpreter, so an eval does not serialize against node processing (no GIL global
//! lock). Expressions are lightweight numpy math over the graph's resolved variables + `t`;
//! anything needing a GIL-bound import belongs in a Python *node*, not an expression.
//!
//! The evaluator resolves NOTHING. It is handed a source the graph rewrote (spec §5.3) and one
//! local per variable, so `nd('lfo')` and `globals.gain` never reach it — which is why there is no
//! name lookup, no proxy and no `globals` namespace here any more.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use goofi_core::{Data, Param, Value};
use goofi_node::{BindingId, Compiled, EvalCtx, ExprError, ExprEvaluator, Local};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule, PyString};

/// The Python harness. The source it is given has already had every `nd(..)` and `globals.*` term
/// replaced by a generated variable (spec §5.3), so there is no `nd()` function, no proxy and no
/// `globals` namespace here — the variables arrive as ordinary locals and the expression is plain
/// numpy math over them. A variable that has not arrived is `None`, and using it raises naturally.
const EVAL_SRC: &str = r#"
import numpy as np

def __goofi_compile(source):
    return compile(source, "<goofi-expr>", "eval")

def __goofi_eval(code, locals_, t):
    ns = {"t": t, "np": np}
    ns.update(locals_)
    return eval(code, ns)
"#;

/// The pyo3 evaluator. Holds the harness functions + a registry of compiled code
/// objects keyed by [`BindingId`].
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

/// Convert a resolved `Data` to a Python object for the eval namespace (array → numpy,
/// string → str; tables are unsupported in v1 → None).
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

/// Convert a resolved `Param` to a native Python scalar — what a `globals.*` variable carries, and
/// what a param-valued arrival will. `Trigger` reads as its `fired` bool, matching the `Bool`
/// coercion on the way back out.
fn param_to_py(py: Python<'_>, p: &Param) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObject;
    Ok(match p {
        Param::Float { value, .. } => value.into_pyobject(py)?.into_any().unbind(),
        Param::Int { value, .. } => value.into_pyobject(py)?.into_any().unbind(),
        Param::Bool { value } => value.into_pyobject(py)?.to_owned().into_any().unbind(),
        Param::Trigger { fired } => fired.into_pyobject(py)?.to_owned().into_any().unbind(),
        Param::Str { value, .. } => PyString::new(py, value).into_any().unbind(),
    })
}

/// Extract a scalar `T` from an expression result. goofi Data force-promotes every scalar to a
/// shape-[1] array, and numpy 2.x rejects `float(np.array([x]))` (only 0-d arrays convert), so the
/// direct extract fails for the most natural expressions (bare `nd('x')`, `nd('a')*nd('b')`, a
/// comparison over a bare `nd()`). Fall back to `np.asarray(x).item()` for any size-1 array; a
/// genuinely multi-element result is a real error. `noun` names the target type in error messages.
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
            // `as i64` saturates NaN→0 and ±inf→±i64::MAX/MIN silently; error instead,
            // consistent with the other type-mismatch arms.
            if !v.is_finite() {
                return Err("expression result is not a finite number".to_string());
            }
            Ok(Param::Int { value: v.round() as i64, vmin: *vmin, vmax: *vmax })
        }
        Param::Bool { .. } => Ok(Param::Bool { value: to_scalar::<bool>(result, "bool")? }),
        // A Trigger errors on a non-bool result (like the other arms) rather than
        // silently swallowing it into `fired: false`.
        Param::Trigger { .. } => Ok(Param::Trigger { fired: to_scalar::<bool>(result, "bool")? }),
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
            // One dict, keyed by the generated variable names the graph minted. There is nothing
            // for the harness to resolve: a name the graph did not hand over is simply not defined,
            // which is the natural Python error and the only one an expression can now get wrong.
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
