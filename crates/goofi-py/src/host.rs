use std::ffi::CString;

use goofi_core::{Data, Meta, SrcDtype, Value};
use goofi_node::{Inputs, Node, NodeCtx, NodeResult, Outputs, Params};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

/// A Python harness that presents the raw float32 bytes to the user node's
/// `process` as a numpy array and returns the result as bytes. Copies once each
/// way (the zero-copy rust-numpy path replaces this later).
const WRAP_SRC: &str = r#"
import numpy as np
def __goofi_wrap(process, raw, shape):
    # Feed the node its input at its REAL shape (not flattened) and preserve the
    # output shape (no ravel) — mirroring the subprocess worker so both backends
    # produce identical output for the same node source. Return the NATIVE dtype
    # (no forced float32) so Rust owns the cast-to-f32 guard, like the subprocess
    # tier. Returns (bytes, shape, dtype_str).
    x = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    y = np.ascontiguousarray(process(x))
    return (y.tobytes(), list(y.shape), y.dtype.str)
"#;

/// A Python node running in-process on the free-threaded interpreter.
pub struct PyNode {
    process: Py<PyAny>,
    wrap: Py<PyAny>,
    /// Whether the runtime GIL tripwire has run yet (it checks once, on the first
    /// `process`, so the steady-state hot path pays nothing).
    gil_checked: bool,
    /// Source dtypes already warned about (dedup for the ingest cast warning).
    cast_warned: std::collections::HashSet<SrcDtype>,
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
            Ok(PyNode {
                process,
                wrap,
                gil_checked: false,
                cast_warned: std::collections::HashSet::new(),
            })
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
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        let in_shape: Vec<usize> = store.shape().to_vec();

        // Check the GIL once (first tick): if running this node re-enabled it (an
        // FT-unsafe import at call time), the shared interpreter is now serialized
        // for ALL in-process nodes — a whole-host hazard the discovery probe can't
        // see. Steady-state ticks skip the check, so the hot path pays nothing.
        let check_gil = !self.gil_checked;
        let (native_bytes, out_shape, dtype_str, tripped): (Vec<u8>, Vec<usize>, String, bool) =
            Python::attach(|py| -> Result<(Vec<u8>, Vec<usize>, String, bool), String> {
                // Copy the Rust buffer straight into Python bytes (no intermediate
                // Vec) — `store` is borrowed from the live input for this call.
                let raw = PyBytes::new(py, store.as_bytes());
                let ret = self
                    .wrap
                    .call1(py, (&self.process, raw, in_shape.clone()))
                    .map_err(|e| e.to_string())?;
                // WRAP returns (bytes, shape, dtype_str): shape preserved, native dtype so
                // Rust owns the cast-to-f32 guard (like the subprocess tier).
                let (bytes, shape, dstr) = ret
                    .bind(py)
                    .extract::<(Vec<u8>, Vec<usize>, String)>()
                    .map_err(|e| e.to_string())?;
                let tripped = check_gil
                    && PyModule::import(py, "sys")
                        .and_then(|m| m.getattr("_is_gil_enabled"))
                        .and_then(|f| f.call0())
                        .and_then(|v| v.extract::<bool>())
                        .unwrap_or(false);
                Ok((bytes, shape, dstr, tripped))
            })?;
        self.gil_checked = true;
        if tripped {
            return Err(
                "node re-enabled the GIL at runtime; quarantine it to the subprocess tier".into(),
            );
        }

        // The ingest cast guard: a foreign output dtype is cast to f32 here (deduped warning).
        let src = SrcDtype::from_numpy_typestr(&dtype_str)
            .ok_or_else(|| format!("node output has unsupported dtype `{dtype_str}`"))?;
        let (out_bytes, _did_cast) = goofi_core::cast_to_f32(src, &native_bytes).map_err(|e| e.to_string())?;
        goofi_core::warn_cast_once(&mut self.cast_warned, "out", src);

        // Mirror the subprocess backend: carry the input meta through a
        // length-preserving node (same shape → sfreq/channels/index stay valid), and
        // drop it when the node changed the shape (stale channel coords would fail
        // Data validation). In-process we clone the meta directly — no re-serialization.
        let out_meta = if out_shape == in_shape { d.meta().clone() } else { Meta::empty() };
        let data = Data::array_f32(out_shape, out_bytes, out_meta)
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
        let frame = Data::array_f32(vec![input.len()], f32s(input), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outbuf.insert("out", None);
        let mut ctx = NodeCtx::new();
        let params = goofi_node::ParamGroups::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx, &Params::new(&params)).unwrap();
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

    fn try_run(node: &mut PyNode, input: &[f32]) -> Result<(), String> {
        let frame = Data::array_f32(vec![input.len()], f32s(input), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outbuf.insert("out", None);
        let mut ctx = NodeCtx::new();
        let params = goofi_node::ParamGroups::new();
        let mut out = Outputs::new(&mut outbuf);
        node.process(&inp, &mut out, &mut ctx, &Params::new(&params)).map_err(|e| e.0)
    }

    #[test]
    fn preserves_output_shape_and_length_preserving_meta() {
        // A length-preserving node (x*2) on a [2,3] input must return [2,3] — NOT
        // ravel to [6] — and carry the input meta (sfreq) through, matching the
        // subprocess backend. Regression for the in-process meta/shape-drop divergence.
        let src = "def process(x):\n    return x * 2.0\n";
        let mut node = PyNode::from_source(src, "process").expect("compile python node");
        let mut meta = Meta::empty();
        meta.sfreq = Some(250.0);
        let d = Data::array_f32(vec![2, 3], f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let params = goofi_node::ParamGroups::new();
        {
            let mut o = Outputs::new(&mut outmap);
            node.process(&inp, &mut o, &mut NodeCtx::new(), &Params::new(&params)).unwrap();
        }
        let outd = outmap.get("out").unwrap().as_ref().unwrap();
        match outd.value() {
            Value::Array(s) => {
                assert_eq!(s.shape(), &[2, 3], "shape preserved, not raveled to [6]");
                let v: Vec<f32> = s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                assert_eq!(v, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
            }
            _ => panic!("expected array"),
        }
        assert_eq!(outd.meta().sfreq, Some(250.0), "length-preserving node carries input meta");
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
    fn casts_non_f32_node_output_to_f32() {
        // A node returning float64 must have its output cast to f32 at the Rust ingest
        // boundary — values preserved. (The dedup warning is unit-tested in goofi-core.)
        let src = "import numpy as np\ndef process(x):\n    return x.astype(np.float64) * 2.0\n";
        let mut node = PyNode::from_source(src, "process").unwrap();
        assert_eq!(run(&mut node, &[1.0, 2.0, 3.0]), vec![2.0, 4.0, 6.0]);
        // An int-returning node is likewise cast to f32.
        let src2 = "import numpy as np\ndef process(x):\n    return (x * 10).astype(np.int32)\n";
        let mut node2 = PyNode::from_source(src2, "process").unwrap();
        assert_eq!(run(&mut node2, &[1.0, 2.0, 3.0]), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn gil_tripwire_fires_only_when_the_gil_is_enabled() {
        // The runtime tripwire errors a node whose execution left the GIL enabled.
        // We can't synthesize an FT-unsafe C-extension here, so drive both states
        // via the interpreter's own GIL: normally disabled (clean run); with
        // PYTHON_GIL=1 the interpreter starts GIL-on, which the tripwire must catch.
        let src = "import numpy as np\ndef process(x):\n    return x\n";
        let mut node = PyNode::from_source(src, "process").unwrap();
        let r = try_run(&mut node, &[1.0, 2.0]);
        if PyNode::gil_enabled().unwrap() {
            let e = r.expect_err("tripwire must fire when the GIL is enabled");
            assert!(e.contains("GIL"), "error should name the GIL: {e}");
        } else {
            assert!(r.is_ok(), "a clean node in free-threaded mode must not trip: {r:?}");
        }
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
