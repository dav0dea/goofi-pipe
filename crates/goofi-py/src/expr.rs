//! `PyExprEvaluator` — the pyo3 param-expression evaluator (the `goofi_node::
//! ExprEvaluator` the engine injects). Runs each expression in the free-threaded
//! interpreter, so an eval does not serialize against node processing (no GIL global
//! lock). Expressions are lightweight numpy math over `nd()` references + `t`; anything
//! needing a GIL-bound import belongs in a Python *node*, not an expression.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use goofi_core::{Data, Param, Value};
use goofi_node::{BindingId, Compiled, EvalCtx, ExprError, ExprEvaluator, ExprRef};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule, PyString};

/// The Python harness. `nd(name)` returns a proxy over the resolved refs: `.slot`
/// selects an output slot, any other attribute (`.mean`, …) delegates to the bare
/// single-output value (numpy methods), and using the proxy directly is the bare
/// single-output value. A bare nd() on a multi-output node raises.
const EVAL_SRC: &str = r#"
import numpy as np

class _NdProxy:
    __slots__ = ("_name", "_refs")
    def __init__(self, name, refs):
        self._name = name
        self._refs = refs
    def _bare(self):
        key = (self._name, None)
        if key in self._refs:
            v = self._refs[key]
            if v is not None:
                return v
            raise ValueError("nd('%s') is unavailable (no value yet)" % self._name)
        if any(k[0] == self._name for k in self._refs):
            raise ValueError("nd('%s') is ambiguous: it has multiple outputs; use nd('%s').slot" % (self._name, self._name))
        raise ValueError("nd('%s') is not a known node reference" % self._name)
    def __getattr__(self, attr):
        v = self._refs.get((self._name, attr))
        if v is not None:
            return v
        return getattr(self._bare(), attr)
    def __array__(self, dtype=None): return np.asarray(self._bare(), dtype=dtype)
    def __float__(self): return float(self._bare())
    def __int__(self): return int(self._bare())
    def __len__(self): return len(self._bare())
    def __getitem__(self, i): return self._bare()[i]
    def __add__(self, o): return self._bare() + o
    def __radd__(self, o): return o + self._bare()
    def __sub__(self, o): return self._bare() - o
    def __rsub__(self, o): return o - self._bare()
    def __mul__(self, o): return self._bare() * o
    def __rmul__(self, o): return o * self._bare()
    def __truediv__(self, o): return self._bare() / o
    def __rtruediv__(self, o): return o / self._bare()
    def __neg__(self): return -self._bare()

def __goofi_compile(source):
    return compile(source, "<goofi-expr>", "eval")

def __goofi_eval(code, refs, t):
    def nd(name):
        return _NdProxy(name, refs)
    return eval(code, {"nd": nd, "t": t, "np": np})
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
        Python::attach(|py| {
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

/// Extract the distinct node names referenced by `nd('name')` / `nd("name")`. A plain
/// scan (not a full parse): the engine resolves ALL of each referenced node's output
/// slots, so only the node NAME matters here — `.slot` vs `.method` is resolved at eval.
fn extract_refs(source: &str) -> Vec<ExprRef> {
    let b = source.as_bytes();
    let mut names: Vec<String> = Vec::new();
    let mut i = 0;
    while let Some(p) = source[i..].find("nd(") {
        let mut j = i + p + 3;
        while j < b.len() && (b[j] as char).is_whitespace() {
            j += 1;
        }
        if j < b.len() && (b[j] == b'\'' || b[j] == b'"') {
            let q = b[j];
            j += 1;
            let start = j;
            while j < b.len() && b[j] != q {
                j += 1;
            }
            if j < b.len() {
                let name = &source[start..j];
                if !name.is_empty() && !names.iter().any(|n| n == name) {
                    names.push(name.to_string());
                }
            }
        }
        i += p + 3;
    }
    names.into_iter().map(|node| ExprRef { node, slot: None }).collect()
}

/// Convert a resolved `Data` to a Python object for the eval namespace (array → numpy,
/// string → str; tables are unsupported in v1 → None).
fn data_to_py(py: Python<'_>, d: &Data) -> PyResult<Py<PyAny>> {
    match d.value() {
        Value::Array(s) => {
            let np = PyModule::import(py, "numpy")?;
            let raw = PyBytes::new(py, s.as_bytes());
            let arr = np.getattr("frombuffer")?.call1((raw, s.dtype().numpy_typestr()))?;
            let shape: Vec<usize> = s.shape().to_vec();
            Ok(arr.call_method1("reshape", (shape,))?.unbind())
        }
        Value::Str(st) => Ok(PyString::new(py, st.as_ref()).into_any().unbind()),
        Value::Table(_) => Ok(py.None()),
    }
}

/// Coerce the Python result to the target param's type.
fn coerce(result: &Bound<'_, PyAny>, target: &Param) -> Result<Param, String> {
    match target {
        Param::Float { vmin, vmax, .. } => {
            let v: f64 = result.extract().map_err(|_| "expression result is not a number".to_string())?;
            Ok(Param::Float { value: v, vmin: *vmin, vmax: *vmax })
        }
        Param::Int { vmin, vmax, .. } => {
            let v: f64 = result.extract().map_err(|_| "expression result is not a number".to_string())?;
            Ok(Param::Int { value: v.round() as i64, vmin: *vmin, vmax: *vmax })
        }
        Param::Bool { .. } => {
            let v: bool = result.extract().map_err(|_| "expression result is not a bool".to_string())?;
            Ok(Param::Bool { value: v })
        }
        Param::Trigger { .. } => Ok(Param::Trigger { fired: result.extract().unwrap_or(false) }),
        Param::Str { options, refresh, .. } => {
            let v: String = result.extract().map_err(|_| "expression result is not a string".to_string())?;
            Ok(Param::Str { value: v, options: options.clone(), refresh: *refresh })
        }
    }
}

impl ExprEvaluator for PyExprEvaluator {
    fn compile(&self, source: &str) -> Result<Compiled, ExprError> {
        let refs = extract_refs(source);
        Python::attach(|py| -> Result<Compiled, ExprError> {
            let code = self
                .compile_fn
                .bind(py)
                .call1((source,))
                .map_err(|e| ExprError(e.to_string()))?;
            let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
            self.codes.lock().unwrap().insert(id, code.unbind());
            Ok(Compiled { id, refs })
        })
    }

    fn eval(&self, id: BindingId, ctx: &EvalCtx<'_>) -> Result<Param, ExprError> {
        Python::attach(|py| -> Result<Param, ExprError> {
            let code = self
                .codes
                .lock()
                .unwrap()
                .get(&id)
                .map(|c| c.clone_ref(py))
                .ok_or_else(|| ExprError("expression not compiled".into()))?;
            let refs = PyDict::new(py);
            for ((name, slot), data) in ctx.refs {
                let val: Py<PyAny> = match data {
                    Some(d) => data_to_py(py, d).map_err(|e| ExprError(e.to_string()))?,
                    None => py.None(),
                };
                refs.set_item((name.as_str(), slot.as_deref()), val)
                    .map_err(|e| ExprError(e.to_string()))?;
            }
            let result = self
                .eval_fn
                .bind(py)
                .call1((code.bind(py), &refs, ctx.t))
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

    #[test]
    fn extract_refs_finds_distinct_node_names() {
        let refs = extract_refs("nd('lfo') * 2 + nd(\"psd\").out.mean() + nd('lfo')[0]");
        let names: Vec<&str> = refs.iter().map(|r| r.node.as_str()).collect();
        assert_eq!(names, vec!["lfo", "psd"], "distinct node names, deduped");
        assert!(refs.iter().all(|r| r.slot.is_none()), "slot resolution is engine-side");
    }

    #[test]
    fn extract_refs_empty_for_refless() {
        assert!(extract_refs("t * 2").is_empty());
    }

    // The tests below drive the real embedded interpreter (numpy required), matching
    // the crate's existing `host.rs` embed-test posture.
    use goofi_core::{DType, Meta};
    use std::collections::HashMap;

    type Refs = HashMap<(String, Option<String>), Option<Data>>;

    fn f32_1d(vals: &[f32]) -> Data {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, vec![vals.len()], bytes, Meta::empty()).unwrap()
    }
    fn fparam() -> Param {
        Param::Float { value: 0.0, vmin: -1e9, vmax: 1e9 }
    }
    fn eval_once(src: &str, t: f64, refs: Refs, target: &Param) -> Result<Param, ExprError> {
        let ev = PyExprEvaluator::new().expect("interpreter");
        let c = ev.compile(src)?;
        let out = ev.eval(c.id, &EvalCtx { refs: &refs, t, target });
        ev.release(c.id);
        out
    }

    #[test]
    fn constant_expression_coerces_to_float() {
        let r = eval_once("1 + 2", 0.0, Refs::new(), &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 3.0).abs() < 1e-9));
    }

    #[test]
    fn time_expression_reads_t() {
        let r = eval_once("t * 2", 4.0, Refs::new(), &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 8.0).abs() < 1e-9));
    }

    #[test]
    fn bare_nd_single_output_delegates_numpy_methods() {
        let mut refs = Refs::new();
        refs.insert(("osc".into(), None), Some(f32_1d(&[3.0, 5.0])));
        let r = eval_once("nd('osc').mean()", 0.0, refs, &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 4.0).abs() < 1e-6));
    }

    #[test]
    fn nd_slot_access_selects_the_output() {
        let mut refs = Refs::new();
        refs.insert(("psd".into(), Some("out".into())), Some(f32_1d(&[10.0, 20.0])));
        refs.insert(("psd".into(), None), Some(f32_1d(&[10.0, 20.0])));
        let r = eval_once("float(nd('psd').out.sum())", 0.0, refs, &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 30.0).abs() < 1e-6));
    }

    #[test]
    fn bare_nd_on_multi_output_is_error() {
        // Only slot keys present, no bare (name, None) -> bare use must raise.
        let mut refs = Refs::new();
        refs.insert(("m".into(), Some("a".into())), Some(f32_1d(&[1.0])));
        refs.insert(("m".into(), Some("b".into())), Some(f32_1d(&[2.0])));
        assert!(eval_once("nd('m').mean()", 0.0, refs, &fparam()).is_err());
    }

    #[test]
    fn missing_ref_is_error() {
        let mut refs = Refs::new();
        refs.insert(("ghost".into(), None), None);
        assert!(eval_once("nd('ghost') + 1", 0.0, refs, &fparam()).is_err());
    }

    #[test]
    fn result_coerces_to_int_and_bool() {
        let ir = eval_once("2.7", 0.0, Refs::new(), &Param::Int { value: 0, vmin: -100, vmax: 100 })
            .unwrap();
        assert!(matches!(ir, Param::Int { value: 3, .. }), "float result rounds to nearest int");
        let br = eval_once("3 > 1", 0.0, Refs::new(), &Param::Bool { value: false }).unwrap();
        assert!(matches!(br, Param::Bool { value: true }));
    }

    #[test]
    fn compile_error_surfaces() {
        let ev = PyExprEvaluator::new().expect("interpreter");
        assert!(ev.compile("1 +").is_err(), "a syntax error must fail compile");
    }
}
