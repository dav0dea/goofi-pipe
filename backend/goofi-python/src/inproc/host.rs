use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use goofi_core::{Data, SrcDtype};
use goofi_node::{Inputs, Node, NodeCtx, NodeError, NodeResult, Outputs, Params};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::attach;

/// A Python node running in-process on the free-threaded interpreter: a live
/// `goofi.Node` subclass instance the engine tick drives. Multi-slot inputs + params +
/// `setup` are marshalled by `goofi_pymod::exec` (shared with the subprocess serve loop).
pub struct PyNode {
    /// The instantiated `goofi.Node` subclass.
    instance: Py<PyAny>,
    /// This node's declared input / output slot names (from its manifest) — the keys the
    /// engine's `Inputs`/`Outputs` use, so `process` knows which slots to gather/emit.
    in_slots: Vec<&'static str>,
    out_slots: Vec<&'static str>,
    /// Whether the runtime GIL tripwire has run and come back CLEAN — set only then, so a
    /// steady-state tick pays nothing while a serialized interpreter keeps being re-checked
    /// (and keeps being reported) for as long as it is serialized.
    gil_checked: bool,
    /// Source dtypes already warned about (dedup for the ingest cast warning).
    cast_warned: HashSet<SrcDtype>,
}

impl PyNode {
    /// Compile a node module from `source` (defining a `goofi.Node` subclass) and
    /// instantiate it. `in_slots`/`out_slots` are the engine-facing slot names (from the
    /// node's manifest), which must match the subclass's `config_*` declarations.
    pub fn from_source(
        source: &str,
        in_slots: Vec<&'static str>,
        out_slots: Vec<&'static str>,
    ) -> PyResult<PyNode> {
        // A unique module name per instance: `PyModule::from_code` registers the module
        // under this name, so a shared name would let concurrently-built nodes (and repeat
        // builds) clobber each other's module object in the one interpreter.
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let name = format!("goofi_user_{}", SEQ.fetch_add(1, Ordering::Relaxed));
        attach(|py| {
            let module = goofi_pymod::loader::module_from_source(py, &name, source)?;
            let instance = goofi_pymod::loader::instantiate(py, &module)?;
            // `from_code` inserts the module into `sys.modules` under `name`; the instance
            // keeps its own module alive via the class `__globals__`, so evict the
            // `sys.modules` entry to avoid unbounded growth as nodes are (re)built.
            py.import("sys")?.getattr("modules")?.call_method1("pop", (&name, py.None()))?;
            Ok(PyNode {
                instance: instance.unbind(),
                in_slots,
                out_slots,
                gil_checked: false,
                cast_warned: HashSet::new(),
            })
        })
    }

    /// Whether the embedded interpreter currently has the GIL enabled (should be `false`
    /// on a free-threaded build — the whole point).
    pub fn gil_enabled() -> PyResult<bool> {
        attach(|py| {
            PyModule::import(py, "sys")?.getattr("_is_gil_enabled")?.call0()?.extract()
        })
    }
}

/// Path to the free-threaded interpreter the in-process host runs on — the binary the
/// GIL-gate + the discovery probe spawn. For an EMBEDDED interpreter `sys.executable` is
/// the host program (`goofi-pipe`), not a python binary; prefer the build-time
/// `PYO3_PYTHON` (the exact interpreter pyo3 linked), falling back to `sys.executable`.
pub fn interpreter_path() -> Option<String> {
    if let Some(p) = option_env!("PYO3_PYTHON") {
        if !p.is_empty() {
            return Some(p.to_string());
        }
    }
    attach(|py| {
        PyModule::import(py, "sys").ok()?.getattr("executable").ok()?.extract::<String>().ok()
    })
}

impl Node for PyNode {
    fn setup(&mut self, _ctx: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        attach(|py| {
            goofi_pymod::exec::run_setup(py, self.instance.bind(py), p.groups())
                .map_err(|e: PyErr| NodeError(e.to_string()))
        })
    }

    fn on_param_refreshed(&mut self, key: &goofi_node::ParamKey, p: &Params<'_>) -> Option<Vec<String>> {
        attach(|py| {
            goofi_pymod::exec::run_refresh(py, self.instance.bind(py), p.groups(), &key.group, &key.name)
        })
    }

    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        // Every DECLARED input slot, with the frame it holds or `None` — the kwarg set
        // `process()` is authored against (single-source; M2's manifests are all single).
        let inputs: Vec<(&str, Option<&Data>)> =
            self.in_slots.iter().map(|name| (*name, inp.get(name))).collect();

        let check_gil = !self.gil_checked;
        let (outs, tripped): (Vec<(String, Data)>, bool) = attach(|py| -> Result<_, String> {
            let outs = goofi_pymod::exec::run_process(
                py,
                self.instance.bind(py),
                p.groups(),
                &inputs,
                &self.out_slots,
                &mut self.cast_warned,
            )
            .map_err(|e| e.to_string())?;
            // Tripwire: if running this node re-enabled the GIL (an FT-unsafe import at
            // call time), the shared interpreter is now serialized for ALL in-process
            // nodes. Checked once (first tick); steady-state ticks skip it.
            let tripped = check_gil
                && PyModule::import(py, "sys")
                    .and_then(|m| m.getattr("_is_gil_enabled"))
                    .and_then(|f| f.call0())
                    .and_then(|v| v.extract::<bool>())
                    .unwrap_or(false);
            Ok((outs, tripped))
        })?;
        if tripped {
            // Deliberately do NOT latch: the condition is permanent (the interpreter stays
            // serialized for every in-process node), so the error has to be permanent too.
            // Latching here would leave one one-tick error the 2 Hz stats sweep — which diffs
            // SAMPLED state, not tick edges — almost never observes. Re-checking keeps
            // `last_error` set for as long as the GIL is on, and self-clears if it goes off.
            return Err("node re-enabled the GIL at runtime; quarantine it to the subprocess tier".into());
        }
        self.gil_checked = true;
        for (slot, data) in outs {
            out.set(&slot, data);
        }
        Ok(())
    }
}
