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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testlock::interp;

    // These drive the real embedded interpreter (numpy required), matching the crate's existing
    // `host.rs` embed-test posture.
    use goofi_core::Meta;

    type Locals = HashMap<String, Option<Local>>;

    fn f32_1d(vals: &[f32]) -> Data {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::array_f32(vec![vals.len()], bytes, Meta::empty()).unwrap()
    }
    fn fparam() -> Param {
        Param::Float { value: 0.0, vmin: -1e9, vmax: 1e9 }
    }
    /// One local, by the generated name the graph's rewrite would have minted.
    fn frame(name: &str, vals: &[f32]) -> (String, Option<Local>) {
        (name.to_string(), Some(Local::Frame(f32_1d(vals))))
    }
    fn value(name: &str, p: Param) -> (String, Option<Local>) {
        (name.to_string(), Some(Local::Value(p)))
    }
    fn eval_once(src: &str, t: f64, locals: Locals, target: &Param) -> Result<Param, ExprError> {
        let ev = PyExprEvaluator::new().expect("interpreter");
        let c = ev.compile(src)?;
        let out = ev.eval(c.id, &EvalCtx { locals: &locals, t, target });
        ev.release(c.id);
        out
    }
    fn f(src: &str, locals: Locals) -> f64 {
        match eval_once(src, 0.0, locals, &fparam()).unwrap() {
            Param::Float { value, .. } => value,
            other => panic!("expected a float, got {other:?}"),
        }
    }

    #[test]
    fn the_evaluator_takes_locals_keyed_by_generated_variable() {
        let _interp = interp();
        // §5.3's locals channel — the whole point of the rewrite. `EvalCtx` carried `refs` keyed by
        // `(name, slot)` and a `globals` snapshot; now it carries one dict the graph filled, and the
        // evaluator resolves nothing.
        let ev = PyExprEvaluator::new().expect("interpreter");
        let c = ev.compile("__v0 * 2").unwrap();
        let mut locals = Locals::new();
        locals.insert("__v0".to_string(), Some(Local::Value(Param::float(21.0, 0.0, 100.0))));
        let ctx = EvalCtx { locals: &locals, t: 0.0, target: &Param::float(0.0, 0.0, 100.0) };
        assert_eq!(ev.eval(c.id, &ctx).unwrap(), Param::float(42.0, 0.0, 100.0));
        ev.release(c.id);
    }

    #[test]
    fn a_frame_local_is_a_numpy_array_the_expression_can_reduce() {
        let _interp = interp();
        // What the `nd('psd').out.mean()` half of the rewrite produces: `__v0.mean()`, with the
        // producer's frame in `__v0`. A local that arrived as a scalar could not answer `.mean()`,
        // so this is what pins that a stream variable stays an ARRAY across the seam.
        assert!((f("__v0.mean()", Locals::from([frame("__v0", &[3.0, 5.0])])) - 4.0).abs() < 1e-6);
        // …and the canonical shape-[1] producer still drives a Float, which numpy 2.x refuses to
        // convert directly (`to_scalar`'s whole reason for existing).
        assert!((f("__v0", Locals::from([frame("__v0", &[3.5])])) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn several_variables_and_t_share_one_namespace() {
        let _interp = interp();
        // The rewritten form of `nd('a') * nd('b') + globals.gain + t`: mixed frame and value
        // locals beside the clock, all read by their generated names.
        let locals = Locals::from([
            frame("__v0", &[2.0]),
            frame("__v1", &[3.0]),
            value("__v2", Param::int(4, 0, 10)),
        ]);
        let r = eval_once("__v0 * __v1 + __v2 + t", 5.0, locals, &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 15.0).abs() < 1e-6), "{r:?}");
    }

    #[test]
    fn a_variable_that_has_not_arrived_is_none_and_using_it_raises() {
        let _interp = interp();
        // The graph ships a variable it could not resolve as `Missing` and the node reports that
        // without evaluating — but a STREAM variable simply has not arrived yet, and an expression
        // that uses one anyway must fail visibly rather than compute with a placeholder.
        let locals = Locals::from([("__v0".to_string(), None)]);
        let err = eval_once("__v0 + 1", 0.0, locals, &fparam()).unwrap_err();
        assert!(err.0.contains("NoneType"), "got: {}", err.0);
    }

    #[test]
    fn a_name_the_graph_did_not_hand_over_is_not_defined() {
        let _interp = interp();
        // There is no `nd` and no `globals` namespace any more. A source that still names one — a
        // call the rewrite could not span, a `globals.` read inside a string the scan skipped —
        // fails visibly rather than resolving anything.
        let err = eval_once("nd('a', 2) + 1", 0.0, Locals::new(), &fparam()).unwrap_err();
        assert!(err.0.contains("not defined"), "got: {}", err.0);
        // `globals` alone is still Python's own builtin, so the failure there is the ATTRIBUTE, not
        // the name — which is why this asserts the error and not its wording.
        assert!(eval_once("globals.gain + 1", 0.0, Locals::new(), &fparam()).is_err());
    }

    #[test]
    fn constant_and_time_expressions_need_no_locals_at_all() {
        let _interp = interp();
        assert!((f("1 + 2", Locals::new()) - 3.0).abs() < 1e-9);
        let r = eval_once("t * 2", 4.0, Locals::new(), &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 8.0).abs() < 1e-9));
    }

    #[test]
    fn result_coerces_to_int_bool_and_str() {
        let _interp = interp();
        let ir = eval_once("2.7", 0.0, Locals::new(), &Param::Int { value: 0, vmin: -100, vmax: 100 }).unwrap();
        assert!(matches!(ir, Param::Int { value: 3, .. }), "float result rounds to nearest int");
        let br = eval_once("__v0 > 1", 0.0, Locals::from([frame("__v0", &[2.0])]), &Param::Bool { value: false })
            .unwrap();
        assert!(matches!(br, Param::Bool { value: true }), "a comparison over a frame drives a Bool");
        let sp = Param::Str { value: String::new(), options: None, refresh: false };
        let sr = eval_once("__v0", 0.0, Locals::from([value("__v0", Param::str_free("P07"))]), &sp).unwrap();
        assert!(matches!(sr, Param::Str { value, .. } if value == "P07"), "a str global reads as a str");
    }

    #[test]
    fn nonfinite_int_and_nonbool_trigger_error() {
        let _interp = interp();
        let ip = Param::Int { value: 0, vmin: -100, vmax: 100 };
        assert!(eval_once("float('inf')", 0.0, Locals::new(), &ip).is_err(), "inf into Int errors, not i64::MAX");
        let tp = Param::Trigger { fired: false };
        assert!(eval_once("1.5", 0.0, Locals::new(), &tp).is_err(), "non-bool into Trigger errors, not silent false");
    }

    #[test]
    fn compile_error_surfaces() {
        let _interp = interp();
        let ev = PyExprEvaluator::new().expect("interpreter");
        assert!(ev.compile("1 +").is_err(), "a syntax error must fail compile");
    }
}
