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
    def __floordiv__(self, o): return self._bare() // o
    def __rfloordiv__(self, o): return o // self._bare()
    def __mod__(self, o): return self._bare() % o
    def __rmod__(self, o): return o % self._bare()
    def __pow__(self, o): return self._bare() ** o
    def __rpow__(self, o): return o ** self._bare()
    def __neg__(self): return -self._bare()
    def __abs__(self): return abs(self._bare())
    # Rich comparisons — Python resolves operator dunders on the TYPE, bypassing
    # __getattr__, so these must be defined explicitly or `nd('x') == v` compares by
    # object identity (silently always False) and `nd('x') > v` raises.
    def __lt__(self, o): return self._bare() < o
    def __le__(self, o): return self._bare() <= o
    def __gt__(self, o): return self._bare() > o
    def __ge__(self, o): return self._bare() >= o
    def __eq__(self, o): return self._bare() == o
    def __ne__(self, o): return self._bare() != o
    __hash__ = None  # defining __eq__ makes it unhashable; proxies are never dict keys

class _Globals:
    # The `globals` namespace an expression reads as `globals.<name>`. A missing name raises
    # NameError — the natural "not defined" error, per the spec: no rename cascade, a stale
    # reference just throws at eval time.
    #
    # Overrides __getattribute__ (not __getattr__), so EVERY name routes through the dict — a global
    # legally named like the internal slot (`_d`) or an object dunder (`__class__`) would otherwise be
    # served by normal attribute lookup and never reach the dict. `is_valid_global_name` permits
    # leading-underscore names, so this is reachable. The slot is read via the base impl to avoid
    # recursing back through __getattribute__.
    __slots__ = ("_d",)
    def __init__(self, d):
        object.__setattr__(self, "_d", d)
    def __getattribute__(self, name):
        d = object.__getattribute__(self, "_d")
        if name in d:
            return d[name]
        raise NameError("global '%s' is not defined" % name)

def __goofi_compile(source):
    return compile(source, "<goofi-expr>", "eval")

def __goofi_eval(code, refs, t, gvals):
    def nd(name):
        return _NdProxy(name, refs)
    return eval(code, {"nd": nd, "t": t, "np": np, "globals": _Globals(gvals)})
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

/// Extract the distinct node names referenced by `nd('name')` / `nd("name")`, as
/// `ExprRef`s with an unresolved `slot` (the engine resolves ALL of each referenced
/// node's output slots — only the node NAME matters here; `.slot` vs `.method` is
/// resolved at eval). Scanning lives in [`goofi_node::nd_ref_names`], the one source of
/// truth shared with the rename rewriter, so extraction and rewriting can't disagree.
fn extract_refs(source: &str) -> Vec<ExprRef> {
    goofi_node::nd_ref_names(source)
        .into_iter()
        .map(|node| ExprRef { node, slot: None })
        .collect()
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
        let refs = extract_refs(source);
        // The `globals.<name>` reads, so the engine re-evaluates this binding exactly when one of
        // those globals changes. Same scan as the eval-namespace injection, so they can't disagree.
        let global_refs = goofi_node::global_ref_names(source);
        Python::attach(|py| -> Result<Compiled, ExprError> {
            let code = self
                .compile_fn
                .bind(py)
                .call1((source,))
                .map_err(|e| ExprError(e.to_string()))?;
            let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
            self.codes.lock().unwrap().insert(id, code.unbind());
            Ok(Compiled { id, refs, global_refs })
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
            // The `globals` namespace, as native Python scalars (float/int/bool/str). A missing
            // `globals.<name>` raises NameError inside the harness — the natural "not defined" error.
            let gvals = PyDict::new(py);
            for (name, value) in ctx.globals.iter() {
                use goofi_core::globals::GlobalValue as G;
                let set = match value {
                    G::Float(f) => gvals.set_item(name.as_str(), *f),
                    G::Int(i) => gvals.set_item(name.as_str(), *i),
                    G::Bool(b) => gvals.set_item(name.as_str(), *b),
                    G::Str(s) => gvals.set_item(name.as_str(), s.as_str()),
                };
                set.map_err(|e| ExprError(e.to_string()))?;
            }
            let result = self
                .eval_fn
                .bind(py)
                .call1((code.bind(py), &refs, ctx.t, &gvals))
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
    fn extract_refs_maps_scanned_names_to_slotless_refs() {
        // The goofi-py-specific contract: extract_refs delegates the scan to
        // goofi_node::nd_ref_names (whose word-boundary/quote behavior is tested there)
        // and maps each name to an ExprRef with an unresolved slot.
        let refs = extract_refs("nd('lfo') * 2 + nd(\"psd\").out.mean() + nd('lfo')[0]");
        let names: Vec<&str> = refs.iter().map(|r| r.node.as_str()).collect();
        assert_eq!(names, vec!["lfo", "psd"], "distinct node names, deduped");
        assert!(refs.iter().all(|r| r.slot.is_none()), "slot resolution is engine-side");
    }

    // The tests below drive the real embedded interpreter (numpy required), matching
    // the crate's existing `host.rs` embed-test posture.
    use goofi_core::globals::{GlobalValue, GlobalsSnapshot};
    use goofi_core::{DType, Meta};
    use indexmap::IndexMap;
    use std::collections::HashMap;

    fn snap(pairs: &[(&str, GlobalValue)]) -> GlobalsSnapshot {
        let mut m = IndexMap::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        GlobalsSnapshot::new(m)
    }

    type Refs = HashMap<(String, Option<String>), Option<Data>>;

    fn f32_1d(vals: &[f32]) -> Data {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, vec![vals.len()], bytes, Meta::empty()).unwrap()
    }
    fn fparam() -> Param {
        Param::Float { value: 0.0, vmin: -1e9, vmax: 1e9 }
    }
    fn eval_once(src: &str, t: f64, refs: Refs, target: &Param) -> Result<Param, ExprError> {
        eval_with_globals(src, t, refs, target, &GlobalsSnapshot::default())
    }
    fn eval_with_globals(
        src: &str,
        t: f64,
        refs: Refs,
        target: &Param,
        globals: &GlobalsSnapshot,
    ) -> Result<Param, ExprError> {
        let ev = PyExprEvaluator::new().expect("interpreter");
        let c = ev.compile(src)?;
        let out = ev.eval(c.id, &EvalCtx { refs: &refs, t, target, globals });
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

    #[test]
    fn bare_nd_size1_array_coerces_to_scalar_float() {
        // The canonical case: a producer emits shape [1]; bare nd('x') drives a Float.
        let mut refs = Refs::new();
        refs.insert(("x".into(), None), Some(f32_1d(&[3.5])));
        let r = eval_once("nd('x')", 0.0, refs, &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 3.5).abs() < 1e-6));
    }

    #[test]
    fn arithmetic_over_size1_refs_coerces_to_float() {
        let mut refs = Refs::new();
        refs.insert(("a".into(), None), Some(f32_1d(&[2.0])));
        refs.insert(("b".into(), None), Some(f32_1d(&[3.0])));
        let r = eval_once("nd('a') * nd('b')", 0.0, refs, &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 6.0).abs() < 1e-6));
    }

    #[test]
    fn comparison_operators_on_bare_nd_drive_a_bool() {
        let bp = Param::Bool { value: false };
        let refs = || {
            let mut r = Refs::new();
            r.insert(("a".into(), None), Some(f32_1d(&[2.0])));
            r
        };
        // `>` used to raise; `==` used to be silently identity-False. Both must be correct.
        assert!(matches!(eval_once("nd('a') > 1", 0.0, refs(), &bp).unwrap(), Param::Bool { value: true }));
        assert!(matches!(eval_once("nd('a') == 2", 0.0, refs(), &bp).unwrap(), Param::Bool { value: true }));
        assert!(matches!(eval_once("nd('a') < 1", 0.0, refs(), &bp).unwrap(), Param::Bool { value: false }));
    }

    #[test]
    fn pow_mod_floordiv_abs_delegate() {
        let mut refs = Refs::new();
        refs.insert(("a".into(), None), Some(f32_1d(&[-3.0])));
        let f = |src: &str, r: Refs| match eval_once(src, 0.0, r, &fparam()).unwrap() {
            Param::Float { value, .. } => value,
            _ => f64::NAN,
        };
        assert!((f("abs(nd('a'))", refs.clone()) - 3.0).abs() < 1e-6);
        assert!((f("nd('a') ** 2", refs.clone()) - 9.0).abs() < 1e-6);
        assert!((f("nd('a') % 2", refs) - 1.0).abs() < 1e-6, "-3 % 2 == 1 (python)");
    }

    #[test]
    fn nonfinite_int_and_nonbool_trigger_error() {
        let ip = Param::Int { value: 0, vmin: -100, vmax: 100 };
        assert!(eval_once("float('inf')", 0.0, Refs::new(), &ip).is_err(), "inf into Int errors, not i64::MAX");
        let tp = Param::Trigger { fired: false };
        assert!(eval_once("1.5", 0.0, Refs::new(), &tp).is_err(), "non-bool into Trigger errors, not silent false");
    }

    #[test]
    fn globals_are_readable_as_a_namespace() {
        // The spec's headline: an expression reads a global as `globals.<name>`, typed natively.
        let g = snap(&[("default_ufreq", GlobalValue::Float(30.0)), ("gain", GlobalValue::Int(4))]);
        let r = eval_with_globals("globals.default_ufreq * globals.gain", 0.0, Refs::new(), &fparam(), &g).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 120.0).abs() < 1e-9));
        // A string global reads as a Python str.
        let gs = snap(&[("subject", GlobalValue::Str("P07".into()))]);
        let sp = Param::Str { value: String::new(), options: None, refresh: false };
        let rs = eval_with_globals("globals.subject", 0.0, Refs::new(), &sp, &gs).unwrap();
        assert!(matches!(rs, Param::Str { value, .. } if value == "P07"));
    }

    #[test]
    fn a_global_named_like_a_slot_or_dunder_is_still_readable() {
        // `is_valid_global_name` permits leading-underscore names, so `_d` (the proxy's internal slot
        // name) and dunder-ish names must still read from the dict, not the proxy's own attributes.
        let g = snap(&[("_d", GlobalValue::Float(5.0)), ("__class__", GlobalValue::Int(9))]);
        let r = eval_with_globals("globals._d + globals.__class__", 0.0, Refs::new(), &fparam(), &g).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 14.0).abs() < 1e-9));
    }

    #[test]
    fn missing_global_raises_a_not_defined_error() {
        // Per the spec: no rename cascade — a stale `globals.<old>` just throws at eval time.
        let g = snap(&[("default_ufreq", GlobalValue::Float(30.0))]);
        let err = eval_with_globals("globals.gone + 1", 0.0, Refs::new(), &fparam(), &g).unwrap_err();
        assert!(err.0.contains("not defined"), "got: {}", err.0);
    }

    #[test]
    fn compile_extracts_global_refs() {
        let ev = PyExprEvaluator::new().expect("interpreter");
        let c = ev.compile("globals.a + nd('x') * globals.b + globals.a").unwrap();
        assert_eq!(c.global_refs, vec!["a", "b"], "distinct global names, in order");
        ev.release(c.id);
    }
}
