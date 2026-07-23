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
    /// Whether the runtime GIL tripwire has run yet (checked once, on the first
    /// `process`, so steady-state ticks pay nothing).
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
        // Gather the present input slots (single-source; M2's manifests are all single).
        let present: Vec<(&str, &Data)> =
            self.in_slots.iter().filter_map(|name| inp.get(name).map(|d| (*name, d))).collect();

        let check_gil = !self.gil_checked;
        let (outs, tripped): (Vec<(String, Data)>, bool) = attach(|py| -> Result<_, String> {
            let outs = goofi_pymod::exec::run_process(
                py,
                self.instance.bind(py),
                p.groups(),
                &present,
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
        self.gil_checked = true;
        if tripped {
            return Err("node re-enabled the GIL at runtime; quarantine it to the subprocess tier".into());
        }
        for (slot, data) in outs {
            out.set(&slot, data);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Data, Meta, Param, Value};
    use goofi_node::{ParamGroups, ParamKey};
    use indexmap::IndexMap;

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    // A minimal single-input/single-output node: doubles its `data` input into `out`.
    // (Sources use concat! so every Python indent is explicit — a `\`-continuation would
    // strip the source-line leading whitespace and break Python's indentation.)
    const DOUBLE: &str = concat!(
        "import goofi\n",
        "import numpy as np\n",
        "class Double(goofi.Node):\n",
        "    def config_input_slots(self):\n",
        "        return {'data': goofi.DataType.ARRAY}\n",
        "    def config_output_slots(self):\n",
        "        return {'out': goofi.DataType.ARRAY}\n",
        "    def process(self, data):\n",
        "        return {'out': data.data * 2.0}\n",
    );

    fn run(node: &mut PyNode, input: &[f32]) -> Vec<f32> {
        let frame = Data::array_f32(vec![input.len()], f32s(input), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outbuf.insert("out", None);
        let mut ctx = NodeCtx::new();
        let params = ParamGroups::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx, &Params::new(&params)).unwrap();
        }
        match outbuf.get("out").unwrap().as_ref().unwrap().value() {
            Value::Array(s) => s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            _ => panic!("expected array"),
        }
    }

    fn mk(src: &str) -> PyNode {
        PyNode::from_source(src, vec!["data"], vec!["out"]).expect("compile node")
    }

    use crate::testlock::interp;

    // A device picker: the node supplies fresh options through the `refresh_<group>_<name>`
    // method convention, which is how a Python node answers the UI's re-enumerate button.
    const PICKER: &str = concat!(
        "import goofi\n",
        "class Picker(goofi.Node):\n",
        "    def config_input_slots(self):\n",
        "        return {'data': goofi.DataType.ARRAY}\n",
        "    def config_output_slots(self):\n",
        "        return {'out': goofi.DataType.ARRAY}\n",
        "    def config_params(self):\n",
        "        return {'audio': {'device': goofi.StringParam('none', refresh=True)}}\n",
        "    def refresh_audio_device(self):\n",
        "        return ['mic', 'line-in']\n",
        "    def process(self, data):\n",
        "        return {'out': data.data}\n",
    );

    #[test]
    fn a_refresh_hook_sees_the_params_as_they_are_now() {
        // The hook is how a picker enumerates against its CURRENT settings (a host, a driver, a
        // directory). Without applying the live params first it reads whatever they were at the
        // last setup/process — permanently stale for a node whose input is unwired, since such a
        // node never ticks.
        const SCOPED: &str = concat!(
            "import goofi\n",
            "class Scoped(goofi.Node):\n",
            "    def config_input_slots(self):\n",
            "        return {'data': goofi.DataType.ARRAY}\n",
            "    def config_output_slots(self):\n",
            "        return {'out': goofi.DataType.ARRAY}\n",
            "    def config_params(self):\n",
            "        return {'audio': {'host': goofi.StringParam('alsa'),\n",
            "                          'device': goofi.StringParam('none', refresh=True)}}\n",
            "    def refresh_audio_device(self):\n",
            "        return [self.params.audio.host + ':0']\n",
            "    def process(self, data):\n",
            "        return {'out': data.data}\n",
        );
        let _interp = interp();
        let mut node = mk(SCOPED);
        let mut params = ParamGroups::new();
        let mut audio = IndexMap::new();
        audio.insert("host".to_string(), Param::Str { value: "jack".into(), options: None, refresh: false });
        params.insert("audio".to_string(), audio);

        assert_eq!(
            node.on_param_refreshed(&ParamKey::new("audio", "device"), &Params::new(&params)),
            Some(vec!["jack:0".to_string()]),
            "the hook enumerated against the live `host`, not the value it was constructed with"
        );
    }

    #[test]
    fn python_node_supplies_fresh_options_by_method_convention() {
        let _interp = interp();
        let mut node = mk(PICKER);
        assert_eq!(
            node.on_param_refreshed(&ParamKey::new("audio", "device"), &Params::new(&ParamGroups::new())),
            Some(vec!["mic".to_string(), "line-in".to_string()])
        );
    }

    #[test]
    fn a_python_node_without_the_refresh_method_offers_nothing() {
        let _interp = interp();
        // Reported as "no options", never as an error: the button must not break a node that
        // simply does not implement the convention.
        let mut node = mk(DOUBLE);
        assert_eq!(node.on_param_refreshed(&ParamKey::new("audio", "device"), &Params::new(&ParamGroups::new())), None);
    }

    #[test]
    fn a_refresh_method_returning_junk_is_ignored_rather_than_crashing_the_node() {
        let _interp = interp();
        const BAD: &str = concat!(
            "import goofi\n",
            "class Bad(goofi.Node):\n",
            "    def config_input_slots(self):\n",
            "        return {'data': goofi.DataType.ARRAY}\n",
            "    def config_output_slots(self):\n",
            "        return {'out': goofi.DataType.ARRAY}\n",
            "    def refresh_audio_device(self):\n",
            "        return 17\n",
            "    def process(self, data):\n",
            "        return {'out': data.data}\n",
        );
        let mut node = mk(BAD);
        assert_eq!(node.on_param_refreshed(&ParamKey::new("audio", "device"), &Params::new(&ParamGroups::new())), None);
    }

    #[test]
    fn a_raising_refresh_method_offers_nothing_and_the_node_keeps_running() {
        let _interp = interp();
        const RAISER: &str = concat!(
            "import goofi\n",
            "class Raiser(goofi.Node):\n",
            "    def config_input_slots(self):\n",
            "        return {'data': goofi.DataType.ARRAY}\n",
            "    def config_output_slots(self):\n",
            "        return {'out': goofi.DataType.ARRAY}\n",
            "    def refresh_audio_device(self):\n",
            "        raise RuntimeError('no soundcard')\n",
            "    def process(self, data):\n",
            "        return {'out': data.data * 2.0}\n",
        );
        let mut node = mk(RAISER);
        assert_eq!(node.on_param_refreshed(&ParamKey::new("audio", "device"), &Params::new(&ParamGroups::new())), None);
        assert_eq!(run(&mut node, &[1.0, 2.0]), vec![2.0, 4.0], "the node still ticks");
    }

    #[test]
    fn class_node_runs_in_process_gil_free() {
        let _interp = interp();
        assert!(!PyNode::gil_enabled().unwrap(), "interpreter must be free-threaded");
        let mut node = mk(DOUBLE);
        assert_eq!(run(&mut node, &[1.0, 2.0, 3.0]), vec![2.0, 4.0, 6.0]);
        assert!(!PyNode::gil_enabled().unwrap(), "GIL must stay disabled after running");
    }

    #[test]
    fn length_preserving_node_carries_input_meta() {
        let _interp = interp();
        // A [2,3] input with sfreq through a shape-preserving node returns [2,3] and
        // carries the input meta (sfreq) — matching the subprocess backend.
        let mut node = mk(DOUBLE);
        let mut meta = Meta::empty();
        meta.set_sfreq(Some(250.0));
        let d = Data::array_f32(vec![2, 3], f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let params = ParamGroups::new();
        {
            let mut o = Outputs::new(&mut outmap);
            node.process(&inp, &mut o, &mut NodeCtx::new(), &Params::new(&params)).unwrap();
        }
        let outd = outmap.get("out").unwrap().as_ref().unwrap();
        match outd.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[2, 3], "shape preserved"),
            _ => panic!("expected array"),
        }
        assert_eq!(outd.meta().sfreq(), Some(250.0), "length-preserving node carries meta");
    }

    #[test]
    fn casts_non_f32_node_output_to_f32() {
        let _interp = interp();
        let f64_src = concat!(
            "import goofi\n",
            "import numpy as np\n",
            "class F(goofi.Node):\n",
            "    def config_input_slots(self):\n",
            "        return {'data': goofi.DataType.ARRAY}\n",
            "    def config_output_slots(self):\n",
            "        return {'out': goofi.DataType.ARRAY}\n",
            "    def process(self, data):\n",
            "        return {'out': data.data.astype(np.float64) * 2.0}\n",
        );
        assert_eq!(run(&mut mk(f64_src), &[1.0, 2.0, 3.0]), vec![2.0, 4.0, 6.0]);
        let i32_src = concat!(
            "import goofi\n",
            "import numpy as np\n",
            "class I(goofi.Node):\n",
            "    def config_input_slots(self):\n",
            "        return {'data': goofi.DataType.ARRAY}\n",
            "    def config_output_slots(self):\n",
            "        return {'out': goofi.DataType.ARRAY}\n",
            "    def process(self, data):\n",
            "        return {'out': (data.data * 10).astype(np.int32)}\n",
        );
        assert_eq!(run(&mut mk(i32_src), &[1.0, 2.0, 3.0]), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn setup_runs_once_and_a_param_reaches_process() {
        let _interp = interp();
        // `setup` seeds `self._base`; `process` reads a live param `gain.factor`. Proves
        // setup ran (its value shows up) AND that a param edit reaches process.
        let src = concat!(
            "import goofi\n",
            "import numpy as np\n",
            "class Scale(goofi.Node):\n",
            "    def config_input_slots(self):\n",
            "        return {'data': goofi.DataType.ARRAY}\n",
            "    def config_output_slots(self):\n",
            "        return {'out': goofi.DataType.ARRAY}\n",
            "    def config_params(self):\n",
            "        return {'gain': {'factor': goofi.IntParam(1, 0, 100)}}\n",
            "    def setup(self):\n",
            "        self._base = 100.0\n",
            "    def process(self, data):\n",
            "        return {'out': data.data * self.params.gain.factor + self._base}\n",
        );
        let mut node = PyNode::from_source(src, vec!["data"], vec!["out"]).unwrap();
        // Seed like the engine: replay on_param_changed (no-op for PyNode) then setup.
        let mut params = ParamGroups::new();
        let mut gain = IndexMap::new();
        gain.insert("factor".to_string(), Param::int(3, 0, 100));
        params.insert("gain".to_string(), gain);
        node.on_param_changed(&ParamKey::new("gain", "factor"), &Param::int(3, 0, 100)).unwrap();
        node.setup(&mut NodeCtx::new(), &Params::new(&params)).unwrap();

        let frame = Data::array_f32(vec![2], f32s(&[1.0, 2.0]), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        {
            let mut o = Outputs::new(&mut outmap);
            node.process(&inp, &mut o, &mut NodeCtx::new(), &Params::new(&params)).unwrap();
        }
        // 1*3 + 100 = 103 ; 2*3 + 100 = 106.
        match outmap.get("out").unwrap().as_ref().unwrap().value() {
            Value::Array(s) => {
                let v: Vec<f32> = s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                assert_eq!(v, vec![103.0, 106.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn python_nodes_run_concurrently_on_native_threads() {
        let _interp = interp();
        let src = concat!(
            "import goofi\n",
            "import numpy as np\n",
            "class Cs(goofi.Node):\n",
            "    def config_input_slots(self):\n",
            "        return {'data': goofi.DataType.ARRAY}\n",
            "    def config_output_slots(self):\n",
            "        return {'out': goofi.DataType.ARRAY}\n",
            "    def process(self, data):\n",
            "        return {'out': np.cumsum(data.data)}\n",
        );
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let mut node = mk(src);
                std::thread::spawn(move || run(&mut node, &vec![1.0f32; 3 + i]))
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            let out = h.join().unwrap();
            assert_eq!(out.len(), 3 + i);
            assert_eq!(out[0], 1.0);
            assert_eq!(*out.last().unwrap(), (3 + i) as f32);
        }
    }
}
