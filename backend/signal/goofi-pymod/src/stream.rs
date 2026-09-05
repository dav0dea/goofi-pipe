//! `goofi.Stream` — the Python view of `goofi_core::Stream`.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use goofi_core::resolve_axis;

use crate::data::array_to_f32;

/// One input's recent past: a stateful transform keeps this instead of its own state, so any
/// chunking of one signal gives one answer.
#[pyclass]
#[derive(Default)]
pub struct Stream {
    inner: goofi_core::Stream,
}

#[pymethods]
impl Stream {
    #[new]
    fn new() -> Stream {
        Stream::default()
    }

    /// Fold one frame in along `axis`, over at least `history` steps of past. Answers the
    /// stitched array and the index the frame starts at along `axis`.
    #[pyo3(signature = (array, axis=-1, history=0))]
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        array: &Bound<'py, PyAny>,
        axis: i64,
        history: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (_src, shape, bytes) = array_to_f32(py, array)?;
        let dim = resolve_axis(axis, shape.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let (out_shape, out, offset) = self.inner.push(&shape, dim, &bytes, history);
        let np = py.import("numpy")?;
        let flat = np.getattr("frombuffer")?.call1((PyBytes::new(py, &out), "<f4"))?;
        Ok((flat.call_method1("reshape", (out_shape,))?, offset))
    }

    /// Forget the past — what a `reset` pulse clears.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// The steps held right now.
    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps()
    }
}
