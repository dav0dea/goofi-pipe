use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use goofi_core::{Data, SrcDtype};
use goofi_node::{Isolation, IsolationCell, Params};
use goofi_signal::{Inputs, Node, NodeCtx, NodeError, NodeResult, Outputs};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::attach;

/// A live `goofi.Node` subclass instance running in-process on the free-threaded interpreter.
pub struct PyNode {
    instance: Py<PyAny>,
    in_slots: Vec<&'static str>,
    out_slots: Vec<&'static str>,
    /// Set only once the GIL tripwire has come back CLEAN, so a serialized interpreter keeps
    /// being re-checked, and re-reported, for as long as it is serialized.
    gil_checked: bool,
    /// Source dtypes already warned about (dedup for the ingest cast warning).
    cast_warned: HashSet<SrcDtype>,
    /// This node TYPE's tier. The tripwire writes it, and the next build reads it. `None` for a
    /// node built from a source string rather than discovered, which no registry routes.
    tier: Option<&'static IsolationCell>,
}

impl PyNode {
    /// Compile a node module from `source` and instantiate its `goofi.Node` subclass.
    pub fn from_source(
        source: &str,
        in_slots: Vec<&'static str>,
        out_slots: Vec<&'static str>,
    ) -> PyResult<PyNode> {
        // Unique per instance: a shared name lets concurrent builds clobber each other's module.
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let name = format!("goofi_user_{}", SEQ.fetch_add(1, Ordering::Relaxed));
        attach(|py| {
            let module = goofi_pymod::loader::module_from_source(py, &name, source)?;
            let instance = goofi_pymod::loader::instantiate(py, &module)?;
            // The instance keeps its module alive through `__globals__`, so evicting the
            // `sys.modules` entry `from_code` inserted only bounds that map's growth.
            py.import("sys")?.getattr("modules")?.call_method1("pop", (&name, py.None()))?;
            Ok(PyNode {
                instance: instance.unbind(),
                in_slots,
                out_slots,
                gil_checked: false,
                cast_warned: HashSet::new(),
                tier: None,
            })
        })
    }

    /// Let this node demote its own TYPE when the runtime GIL tripwire fires.
    pub fn routed_by(mut self, tier: &'static IsolationCell) -> PyNode {
        self.tier = Some(tier);
        self
    }

    /// Whether the embedded interpreter currently has the GIL enabled.
    pub fn gil_enabled() -> PyResult<bool> {
        attach(|py| {
            PyModule::import(py, "sys")?.getattr("_is_gil_enabled")?.call0()?.extract()
        })
    }
}

/// Path to the free-threaded interpreter. `PYO3_PYTHON` comes first because, embedded,
/// `sys.executable` is the host binary rather than a python.
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
            // Tripwire: an FT-unsafe import at call time serializes the one interpreter for
            // every in-process node.
            let tripped = check_gil
                && PyModule::import(py, "sys")
                    .and_then(|m| m.getattr("_is_gil_enabled"))
                    .and_then(|f| f.call0())
                    .and_then(|v| v.extract::<bool>())
                    .unwrap_or(false);
            Ok((outs, tripped))
        })?;
        if tripped {
            // The probe cleared this type on its IMPORT; only running it revealed otherwise. Writing
            // the type's tier is the whole re-route: the next `restart_node` builds from this.
            let demoted = self.tier.is_some_and(|t| t.set(Isolation::Subprocess));
            if demoted {
                eprintln!("note: this node re-enabled the GIL; restart it to move it to a subprocess");
            }
            // Deliberately not latched: the condition is permanent, so the error must keep being
            // re-reported — the stats sweep samples state, and would miss a one-tick error.
            return Err("node re-enabled the GIL at runtime; restart it to move it to a subprocess".into());
        }
        self.gil_checked = true;
        for (slot, data) in outs {
            out.set(&slot, data);
        }
        Ok(())
    }
}
