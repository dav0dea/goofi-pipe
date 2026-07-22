//! `goofi.Data` — the Python view of `goofi_core::Data`. Construction copies a numpy
//! array into an `Arc<[u8]>` f32 buffer (keeping the core pyo3-free); `.data` builds a
//! fresh numpy `<f4` array via `numpy.frombuffer` (the expr.rs pattern, no rust-numpy
//! crate); `.meta` mirrors the core `Meta` map as a Python dict. `channels` is the one
//! structured key: a nested `{dimN: [coords...]}` dict.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::IntoPyObjectExt;

use goofi_core::{cast_to_f32, Axes, Axis, Coord, Data as CoreData, Meta, MetaValue, SrcDtype, Value};

#[pyclass]
pub struct Data {
    inner: CoreData,
}

#[pymethods]
impl Data {
    /// `Data(array, meta=None)` — copy a numpy array (cast to f32) + optional meta dict.
    #[new]
    #[pyo3(signature = (array, meta=None))]
    fn new(py: Python<'_>, array: &Bound<'_, PyAny>, meta: Option<&Bound<'_, PyDict>>) -> PyResult<Data> {
        // np.ascontiguousarray(array) then read its dtype string + bytes.
        let np = py.import("numpy")?;
        let arr = np.getattr("ascontiguousarray")?.call1((array,))?;
        let dtype_str: String = arr.getattr("dtype")?.getattr("str")?.extract()?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let raw: Vec<u8> = arr.call_method0("tobytes")?.extract()?;
        let src = SrcDtype::from_numpy_typestr(&dtype_str)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unsupported dtype {dtype_str}")))?;
        let (f32_bytes, _cast) = cast_to_f32(src, &raw)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let m = meta.map(dict_to_meta).transpose()?.unwrap_or_else(Meta::new);
        let inner = CoreData::array_f32(shape, f32_bytes, m)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Data { inner })
    }

    /// The array as a fresh numpy `<f4` array of the stored shape.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Value::Array(store) = self.inner.value() else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Data is not an array"));
        };
        let np = py.import("numpy")?;
        let bytes = PyBytes::new(py, store.as_bytes());
        let flat = np.getattr("frombuffer")?.call1((bytes, "<f4"))?;
        let shape: Vec<usize> = store.shape().to_vec();
        flat.call_method1("reshape", (shape,))
    }

    /// The meta as a Python dict (builtins present; channels nested).
    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        meta_to_dict(py, self.inner.meta())
    }
}

impl Data {
    /// Wrap an existing core `Data` (no numpy round-trip) — the seam the in-process
    /// executor + the subprocess serve loop use to hand a node its inputs and to read
    /// its `goofi.Data` outputs back out.
    pub fn from_core(inner: CoreData) -> Data {
        Data { inner }
    }
    /// The wrapped core `Data`.
    pub fn core(&self) -> &CoreData {
        &self.inner
    }
}

/// Build a `Meta` from a Python dict. `channels` (a `{dimN: [...]}` dict) → `Axes`;
/// everything else → a scalar/list `MetaValue`.
pub(crate) fn dict_to_meta(d: &Bound<'_, PyDict>) -> PyResult<Meta> {
    let mut m = Meta::new();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        if key == goofi_core::META_CHANNELS {
            m.set_channels(dict_to_axes(&v)?);
        } else {
            m.set(key, py_to_mv(&v)?);
        }
    }
    Ok(m)
}

fn dict_to_axes(v: &Bound<'_, PyAny>) -> PyResult<Axes> {
    let mut axes = Axes::new();
    let d = v.cast::<PyDict>()?;
    for (k, list) in d.iter() {
        let key: String = k.extract()?;
        let Some(dim) = key.strip_prefix("dim").and_then(|s| s.parse::<usize>().ok()) else {
            continue;
        };
        let items = list.cast::<PyList>()?;
        let mut coords = Vec::with_capacity(items.len());
        for it in items.iter() {
            if let Ok(s) = it.extract::<String>() {
                coords.push(Coord::Str(s.into()));
            } else {
                coords.push(Coord::Num(it.extract::<f64>()?));
            }
        }
        axes = axes.with(dim, Axis::coords(coords));
    }
    Ok(axes)
}

fn py_to_mv(v: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    if v.is_none() {
        Ok(MetaValue::Null)
    } else if let Ok(b) = v.extract::<bool>() {
        Ok(MetaValue::Bool(b))
    } else if let Ok(i) = v.extract::<i64>() {
        Ok(MetaValue::Int(i))
    } else if let Ok(f) = v.extract::<f64>() {
        Ok(MetaValue::Float(f))
    } else if let Ok(s) = v.extract::<String>() {
        Ok(MetaValue::Str(s))
    } else if let Ok(list) = v.cast::<PyList>() {
        Ok(MetaValue::List(list.iter().map(|it| py_to_mv(&it)).collect::<PyResult<_>>()?))
    } else {
        Ok(MetaValue::Null)
    }
}

fn meta_to_dict<'py>(py: Python<'py>, m: &Meta) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in m.iter() {
        if k == goofi_core::META_CHANNELS {
            d.set_item(k, axes_to_dict(py, m.channels())?)?;
        } else if !matches!(v, MetaValue::Null) {
            d.set_item(k, mv_to_py(py, v)?)?;
        }
    }
    Ok(d)
}

fn axes_to_dict<'py>(py: Python<'py>, axes: &Axes) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (dim, axis) in axes.0.iter().enumerate() {
        if let Some(coords) = &axis.coords {
            let list = PyList::empty(py);
            for c in coords.iter() {
                match c {
                    Coord::Num(n) => list.append(*n)?,
                    Coord::Str(s) => list.append(s.as_ref())?,
                }
            }
            d.set_item(format!("dim{dim}"), list)?;
        }
    }
    Ok(d)
}

fn mv_to_py<'py>(py: Python<'py>, v: &MetaValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        MetaValue::Null => py.None().into_bound(py),
        MetaValue::Bool(b) => b.into_bound_py_any(py)?,
        MetaValue::Int(i) => i.into_bound_py_any(py)?,
        MetaValue::Uint(u) => u.into_bound_py_any(py)?,
        MetaValue::Float(f) => f.into_bound_py_any(py)?,
        MetaValue::Str(s) => s.into_bound_py_any(py)?,
        MetaValue::Bytes(b) => PyBytes::new(py, b).into_any(),
        MetaValue::List(l) => {
            let list = PyList::empty(py);
            for it in l {
                list.append(mv_to_py(py, it)?)?;
            }
            list.into_any()
        }
        MetaValue::Map(mp) => {
            let d = PyDict::new(py);
            for (k, val) in mp {
                d.set_item(k, mv_to_py(py, val)?)?;
            }
            d.into_any()
        }
        // Only the top-level `channels` is Axes; handled by meta_to_dict directly.
        MetaValue::Axes(_) => py.None().into_bound(py),
    })
}
