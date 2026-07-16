use std::ffi::CString;

use goofi_core::{Data, DType, Meta, Value};
use goofi_node::{Inputs, Node, NodeCtx, NodeResult, Outputs};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

/// A Python harness that presents the raw float32 bytes to the user node's
/// `process` as a numpy array and returns the result as bytes. Copies once each
/// way (the zero-copy rust-numpy path replaces this later).
const WRAP_SRC: &str = r#"
import numpy as np
def __goofi_wrap(process, raw, n):
    x = np.frombuffer(raw, dtype=np.float32, count=n).copy()
    y = np.ascontiguousarray(np.asarray(process(x), dtype=np.float32)).ravel()
    return y.tobytes()
"#;

/// A Python node running in-process on the free-threaded interpreter.
pub struct PyNode {
    process: Py<PyAny>,
    wrap: Py<PyAny>,
}

impl PyNode {
    /// Compile a Python node from source defining `func_name(x) -> array-like`.
    pub fn from_source(source: &str, func_name: &str) -> PyResult<PyNode> {
        Python::attach(|py| {
            let user = PyModule::from_code(
                py,
                CString::new(source)?.as_c_str(),
                c"goofi_user.py",
                c"goofi_user",
            )?;
            let process = user.getattr(func_name)?.unbind();
            let wrapmod = PyModule::from_code(
                py,
                CString::new(WRAP_SRC)?.as_c_str(),
                c"goofi_wrap.py",
                c"goofi_wrap",
            )?;
            let wrap = wrapmod.getattr("__goofi_wrap")?.unbind();
            Ok(PyNode { process, wrap })
        })
    }

    /// Whether the embedded interpreter currently has the GIL enabled (should be
    /// `false` on a free-threaded build — the whole point).
    pub fn gil_enabled() -> PyResult<bool> {
        Python::attach(|py| {
            PyModule::import(py, "sys")?
                .getattr("_is_gil_enabled")?
                .call0()?
                .extract()
        })
    }
}

/// Path to the free-threaded interpreter the in-process host runs on — the binary
/// the GIL-gate spawns to probe a node's free-threading safety without touching
/// the shared host interpreter.
///
/// For an EMBEDDED interpreter `sys.executable` is the host program (`goofi-pipe`),
/// not a python binary, so it's unusable for spawning. Prefer the build-time
/// `PYO3_PYTHON` (the exact interpreter pyo3 linked against); fall back to
/// `sys.executable` only if it wasn't set.
pub fn interpreter_path() -> Option<String> {
    if let Some(p) = option_env!("PYO3_PYTHON") {
        if !p.is_empty() {
            return Some(p.to_string());
        }
    }
    Python::attach(|py| {
        PyModule::import(py, "sys")
            .ok()?
            .getattr("executable")
            .ok()?
            .extract::<String>()
            .ok()
    })
}

impl Node for PyNode {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }
        let n = store.shape().iter().product::<usize>();
        let bytes = store.as_bytes().to_vec();

        let out_bytes: Vec<u8> = Python::attach(|py| -> Result<Vec<u8>, String> {
            let raw = PyBytes::new(py, &bytes);
            let ret = self
                .wrap
                .call1(py, (&self.process, raw, n))
                .map_err(|e| e.to_string())?;
            let b = ret.bind(py).cast::<PyBytes>().map_err(|e| e.to_string())?;
            Ok(b.as_bytes().to_vec())
        })?;

        let len = out_bytes.len() / 4;
        let data = Data::from_array_bytes(DType::F32, vec![len], out_bytes, Meta::empty())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use std::sync::Arc;

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn run(node: &mut PyNode, input: &[f32]) -> Vec<f32> {
        let frame = Data::from_array_bytes(DType::F32, vec![input.len()], f32s(input), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outbuf.insert("out", None);
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        if let Value::Array(s) = d.value() {
            s.as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        } else {
            panic!("expected array")
        }
    }

    #[test]
    fn python_numpy_node_runs_in_process_gil_free() {
        assert!(!PyNode::gil_enabled().unwrap(), "interpreter must be free-threaded");
        let src = "import numpy as np\ndef process(x):\n    return x * 2.0 + 1.0\n";
        let mut node = PyNode::from_source(src, "process").expect("compile python node");
        assert_eq!(run(&mut node, &[1.0, 2.0, 3.0]), vec![3.0, 5.0, 7.0]);
        assert!(!PyNode::gil_enabled().unwrap(), "GIL must stay disabled after running");
    }

    #[test]
    fn python_nodes_run_concurrently_on_native_threads() {
        // Two Python nodes ticking on separate OS threads simultaneously — only
        // possible with the GIL disabled. Each does real numpy work.
        let src = "import numpy as np\ndef process(x):\n    return np.cumsum(x)\n";
        let mk = || PyNode::from_source(src, "process").unwrap();
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let mut node = mk();
                std::thread::spawn(move || run(&mut node, &vec![1.0f32; 3 + i]))
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            let out = h.join().unwrap();
            // cumsum of ones -> [1,2,3,...]
            assert_eq!(out.len(), 3 + i);
            assert_eq!(out[0], 1.0);
            assert_eq!(*out.last().unwrap(), (3 + i) as f32);
        }
        let _ = Arc::new(());
    }
}
