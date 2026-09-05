//! `goofi.Data` — the Python view of `goofi_core::Data`.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use goofi_core::{cast_to_f32, resolve_axis, Axes, Axis, Coord, Data as CoreData, Meta, MetaValue, SrcDtype, Value};

#[pyclass]
pub struct Data {
    inner: CoreData,
}

/// A rank waiting to be compared: every comparison dunder raises unless the claim holds.
#[pyclass]
pub struct Ndims {
    shape: Vec<usize>,
}

impl Ndims {
    fn check(&self, holds: bool, op: &str, n: usize) -> PyResult<bool> {
        if holds {
            Ok(true)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "needs ndim {op} {n}, got {:?}",
                self.shape
            )))
        }
    }
}

#[pymethods]
impl Ndims {
    fn __gt__(&self, n: usize) -> PyResult<bool> {
        self.check(self.shape.len() > n, ">", n)
    }
    fn __ge__(&self, n: usize) -> PyResult<bool> {
        self.check(self.shape.len() >= n, ">=", n)
    }
    fn __lt__(&self, n: usize) -> PyResult<bool> {
        self.check(self.shape.len() < n, "<", n)
    }
    fn __le__(&self, n: usize) -> PyResult<bool> {
        self.check(self.shape.len() <= n, "<=", n)
    }
    fn __eq__(&self, n: usize) -> PyResult<bool> {
        self.check(self.shape.len() == n, "==", n)
    }
    fn __ne__(&self, n: usize) -> PyResult<bool> {
        self.check(self.shape.len() != n, "!=", n)
    }
    fn __int__(&self) -> usize {
        self.shape.len()
    }
    fn __repr__(&self) -> String {
        format!("ndim {} of shape {:?}", self.shape.len(), self.shape)
    }
}

impl Data {
    /// The array's shape; empty (rank 0) for a string/table frame.
    fn shape_or_empty(&self) -> Vec<usize> {
        match self.inner.value() {
            Value::Array(a) => a.shape().to_vec(),
            _ => Vec::new(),
        }
    }
}

#[pymethods]
impl Data {
    /// `Data(array, meta=None)` — copy a numpy array (cast to f32) + optional meta dict.
    #[new]
    #[pyo3(signature = (array, meta=None))]
    fn new(py: Python<'_>, array: &Bound<'_, PyAny>, meta: Option<&Bound<'_, PyDict>>) -> PyResult<Data> {
        let (_src, shape, f32_bytes) = array_to_f32(py, array)?;
        let m = meta.map(dict_to_meta).transpose()?.unwrap_or_else(Meta::new);
        let inner = CoreData::array_f32(shape, f32_bytes, m)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Data { inner })
    }

    /// The frame's rank, ready to be compared: `data.assert_ndims > 1` raises unless it holds.
    #[getter]
    fn assert_ndims(&self) -> Ndims {
        Ndims { shape: self.shape_or_empty() }
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

    /// The meta for a frame with `axis` removed: its labels go, the higher axes move down, and
    /// the rate goes with the last axis.
    fn drop_axis<'py>(&self, py: Python<'py>, axis: i64) -> PyResult<Bound<'py, PyDict>> {
        let ndim = self.shape_or_empty().len();
        let dim = resolve_axis(axis, ndim).map_err(pyo3::exceptions::PyValueError::new_err)?;
        meta_to_dict(py, &self.inner.meta().drop_axis(dim, ndim))
    }

    /// The meta for a frame subset to `indices` along `axis`, in the given order.
    fn keep<'py>(&self, py: Python<'py>, axis: i64, indices: Vec<usize>) -> PyResult<Bound<'py, PyDict>> {
        let ndim = self.shape_or_empty().len();
        let dim = resolve_axis(axis, ndim).map_err(pyo3::exceptions::PyValueError::new_err)?;
        meta_to_dict(py, &self.inner.meta().keep(dim, &indices))
    }

    /// The meta for this frame joined with `others` along `axis`; one unlabeled input leaves the
    /// axis unlabeled.
    fn concat<'py>(
        &self,
        py: Python<'py>,
        others: Vec<PyRef<'_, Data>>,
        axis: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ndim = self.shape_or_empty().len();
        let dim = resolve_axis(axis, ndim).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let metas: Vec<&Meta> = others.iter().map(|d| d.inner.meta()).collect();
        meta_to_dict(py, &self.inner.meta().concat(&metas, dim))
    }

    /// The meta for a frame with a new axis at `axis`, labeled by `coords` when the caller can
    /// name every entry; a new last axis takes the rate away.
    #[pyo3(signature = (axis, coords=None))]
    fn insert_axis<'py>(
        &self,
        py: Python<'py>,
        axis: i64,
        coords: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ndim = self.shape_or_empty().len();
        let dim = resolve_axis(axis, ndim + 1).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let ax = match coords {
            Some(c) => Axis::coords(list_to_coords(c)?),
            None => Axis::default(),
        };
        meta_to_dict(py, &self.inner.meta().insert_axis(dim, ax, ndim))
    }
}

impl Data {
    /// Wrap an existing core `Data` (no numpy round-trip).
    pub fn from_core(inner: CoreData) -> Data {
        Data { inner }
    }
    /// The wrapped core `Data`.
    pub fn core(&self) -> &CoreData {
        &self.inner
    }
}

/// `np.ascontiguousarray(array)` → `(source dtype, shape, f32 bytes)`, the one numpy→f32 funnel.
pub(crate) fn array_to_f32(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
) -> PyResult<(SrcDtype, Vec<usize>, Vec<u8>)> {
    let np = py.import("numpy")?;
    let arr = np.getattr("ascontiguousarray")?.call1((array,))?;
    let dtype_str: String = arr.getattr("dtype")?.getattr("str")?.extract()?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let raw: Vec<u8> = arr.call_method0("tobytes")?.extract()?;
    let src = SrcDtype::from_numpy_typestr(&dtype_str)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unsupported dtype {dtype_str}")))?;
    let (bytes, _did_cast) = cast_to_f32(src, &raw)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((src, shape, bytes))
}

/// Build a `Meta` from a Python dict; `channels` (a `{dimN: [...]}` dict) maps to `Axes`.
pub(crate) fn dict_to_meta(d: &Bound<'_, PyDict>) -> PyResult<Meta> {
    let mut m = Meta::new();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        if key == goofi_core::META_CHANNELS {
            m.set_channels(dict_to_axes(&v)?);
        } else {
            m.set(key, pythonize::depythonize(&v)?);
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
        axes = axes.with(dim, Axis::coords(list_to_coords(&list)?));
    }
    Ok(axes)
}

/// A Python sequence of names or numbers as axis coords.
fn list_to_coords(v: &Bound<'_, PyAny>) -> PyResult<Vec<Coord>> {
    let items = v.cast::<PyList>()?;
    let mut coords = Vec::with_capacity(items.len());
    for it in items.iter() {
        if let Ok(s) = it.extract::<String>() {
            coords.push(Coord::Str(s.into()));
        } else {
            coords.push(Coord::Num(it.extract::<f64>()?));
        }
    }
    Ok(coords)
}

fn meta_to_dict<'py>(py: Python<'py>, m: &Meta) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in m.iter() {
        if k == goofi_core::META_CHANNELS {
            d.set_item(k, axes_to_dict(py, m.channels())?)?;
        } else if !matches!(v, MetaValue::Null) {
            d.set_item(k, pythonize::pythonize(py, v)?)?;
        }
    }
    Ok(d)
}

fn axes_to_dict<'py>(py: Python<'py>, axes: &Axes) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (dim, coords) in axes.dims() {
        let list = PyList::empty(py);
        for c in coords {
            match c {
                Coord::Num(n) => list.append(*n)?,
                Coord::Str(s) => list.append(s.as_ref())?,
            }
        }
        d.set_item(dim, list)?;
    }
    Ok(d)
}

