//! The in-process Python tier — a live `goofi.Node` hosted on a free-threaded interpreter,
//! through the same marshalling seam the subprocess tier uses.
//!
//! Behind `embed`, which LINKS libpython:
//!   cargo test -p goofi-tests --features embed
#![cfg(feature = "embed")]

/// This binary's tests share ONE embedded interpreter while cargo runs them on parallel threads, so
/// anything reading process-global interpreter state (`sys._is_gil_enabled`) can otherwise observe a
/// sibling mid-import and fail spuriously. Every test that drives the interpreter holds this.
fn interp() -> std::sync::MutexGuard<'static, ()> {
    static INTERP: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // Recover from a poisoned lock: one failing test must not cascade into all the others.
    INTERP.lock().unwrap_or_else(|e| e.into_inner())
}

use goofi_core::{Data, Meta, Param, Value};
use std::collections::HashMap;

use goofi_node::{EvalCtx, ExprError, ExprEvaluator, Inputs, Local, Node, NodeCtx, Outputs};
use goofi_python::inproc::PyExprEvaluator;
use goofi_python::inproc::PyNode;
use goofi_node::{ParamGroups, ParamKey, Params};
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

// Two declared slots, only one of which the tests wire. The `a is None` branch returns a
// BARE array (not a `{slot: value}` dict), so one node pins both halves of the contract:
// that the absent slot arrives at all, and which input the bare return's meta comes from.
const PAIR: &str = concat!(
    "import goofi\n",
    "class Pair(goofi.Node):\n",
    "    def config_input_slots(self):\n",
    "        return {'a': goofi.DataType.ARRAY, 'b': goofi.DataType.ARRAY}\n",
    "    def config_output_slots(self):\n",
    "        return {'out': goofi.DataType.ARRAY}\n",
    "    def process(self, a, b):\n",
    "        if a is None:\n",
    "            return b.data * 2.0\n",
    "        return a.data + b.data\n",
);

/// Tick a `PAIR` node with `a` unwired and `b` carrying `frame` — the shape a node sees when
/// only some of its declared inputs are wired.
fn tick_a_absent(frame: Data) -> Data {
    let mut node = PyNode::from_source(PAIR, vec!["a", "b"], vec!["out"]).expect("compile node");
    let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    inmap.insert("a", None);
    inmap.insert("b", Some(frame));
    let inp = Inputs::new(&inmap);
    let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    outmap.insert("out", None);
    let params = ParamGroups::new();
    {
        let mut o = Outputs::new(&mut outmap);
        node.process(&inp, &mut o, &mut NodeCtx::new(), &Params::new(&params)).expect("process");
    }
    outmap.get("out").unwrap().clone().expect("output frame")
}

#[test]
fn an_absent_declared_slot_arrives_as_none_rather_than_being_omitted() {
    let _interp = interp();
    // Omitting it is not a no-op: `def process(self, a, b)` then raises TypeError every tick,
    // so a partially-wired node could never run. `[2,4,6]` is reachable ONLY through the
    // node's own `a is None` branch — which is the point: what an absent non-required input
    // means is the node's call to make, exactly as a native node's `inp.get(...)` lets it.
    let frame = Data::array_f32(vec![3], f32s(&[1.0, 2.0, 3.0]), Meta::empty()).unwrap();
    match tick_a_absent(frame).value() {
        Value::Array(s) => {
            let v: Vec<f32> = s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            assert_eq!(v, vec![2.0, 4.0, 6.0]);
        }
        _ => panic!("expected array"),
    }
}

#[test]
fn a_bare_return_carries_the_first_present_inputs_meta_not_the_first_declared_slots() {
    let _interp = interp();
    // The meta source for a bare array return is a PRESENT frame. Keyed off the first
    // DECLARED slot instead, a node whose first input is unwired silently loses its output
    // meta — sfreq gone, so every downstream node reading a rate gets nothing.
    let mut meta = Meta::empty();
    meta.set_sfreq(Some(250.0));
    let frame = Data::array_f32(vec![3], f32s(&[1.0, 2.0, 3.0]), meta).unwrap();
    assert_eq!(tick_a_absent(frame).meta().sfreq(), Some(250.0));
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

// ---------------------------------------------------------------------------
// The pyo3 expression evaluator
// ---------------------------------------------------------------------------



type Locals = HashMap<String, Option<Local>>;

fn f32_1d(vals: &[f32]) -> Data {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    Data::array_f32(vec![vals.len()], bytes, Meta::empty()).unwrap()
}
fn fparam() -> Param {
    Param::Float { value: 0.0, vmin: -1e9, vmax: 1e9 }
}
/// One local, by the generated name the graph's rewrite would have minted.
fn frame(name: &str, vals: &[f32]) -> (String, Option<Local>) {
    (name.to_string(), Some(Local::Frame(f32_1d(vals))))
}
fn value(name: &str, p: Param) -> (String, Option<Local>) {
    (name.to_string(), Some(Local::Value(p)))
}
fn eval_once(src: &str, t: f64, locals: Locals, target: &Param) -> Result<Param, ExprError> {
    let ev = PyExprEvaluator::new().expect("interpreter");
        let c = ev.compile(src)?;
        let out = ev.eval(c.id, &EvalCtx { locals: &locals, t, target });
        ev.release(c.id);
        out
    }
    fn f(src: &str, locals: Locals) -> f64 {
        match eval_once(src, 0.0, locals, &fparam()).unwrap() {
            Param::Float { value, .. } => value,
            other => panic!("expected a float, got {other:?}"),
        }
    }

    #[test]
    fn the_evaluator_takes_locals_keyed_by_generated_variable() {
        let _interp = interp();
        // §5.3's locals channel — the whole point of the rewrite. `EvalCtx` carried `refs` keyed by
        // `(name, slot)` and a `globals` snapshot; now it carries one dict the graph filled, and the
        // evaluator resolves nothing.
        let ev = PyExprEvaluator::new().expect("interpreter");
        let c = ev.compile("__v0 * 2").unwrap();
        let mut locals = Locals::new();
        locals.insert("__v0".to_string(), Some(Local::Value(Param::float(21.0, 0.0, 100.0))));
        let ctx = EvalCtx { locals: &locals, t: 0.0, target: &Param::float(0.0, 0.0, 100.0) };
        assert_eq!(ev.eval(c.id, &ctx).unwrap(), Param::float(42.0, 0.0, 100.0));
        ev.release(c.id);
    }

    #[test]
    fn a_frame_local_is_a_numpy_array_the_expression_can_reduce() {
        let _interp = interp();
        // What the `nd('psd').out.mean()` half of the rewrite produces: `__v0.mean()`, with the
        // producer's frame in `__v0`. A local that arrived as a scalar could not answer `.mean()`,
        // so this is what pins that a stream variable stays an ARRAY across the seam.
        assert!((f("__v0.mean()", Locals::from([frame("__v0", &[3.0, 5.0])])) - 4.0).abs() < 1e-6);
        // …and the canonical shape-[1] producer still drives a Float, which numpy 2.x refuses to
        // convert directly (`to_scalar`'s whole reason for existing).
        assert!((f("__v0", Locals::from([frame("__v0", &[3.5])])) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn several_variables_and_t_share_one_namespace() {
        let _interp = interp();
        // The rewritten form of `nd('a') * nd('b') + globals.gain + t`: mixed frame and value
        // locals beside the clock, all read by their generated names.
        let locals = Locals::from([
            frame("__v0", &[2.0]),
            frame("__v1", &[3.0]),
            value("__v2", Param::int(4, 0, 10)),
        ]);
        let r = eval_once("__v0 * __v1 + __v2 + t", 5.0, locals, &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 15.0).abs() < 1e-6), "{r:?}");
    }

    #[test]
    fn a_variable_that_has_not_arrived_is_none_and_using_it_raises() {
        let _interp = interp();
        // The graph ships a variable it could not resolve as `Missing` and the node reports that
        // without evaluating — but a STREAM variable simply has not arrived yet, and an expression
        // that uses one anyway must fail visibly rather than compute with a placeholder.
        let locals = Locals::from([("__v0".to_string(), None)]);
        let err = eval_once("__v0 + 1", 0.0, locals, &fparam()).unwrap_err();
        assert!(err.0.contains("NoneType"), "got: {}", err.0);
    }

    #[test]
    fn a_name_the_graph_did_not_hand_over_is_not_defined() {
        let _interp = interp();
        // There is no `nd` and no `globals` namespace any more. A source that still names one — a
        // call the rewrite could not span, a `globals.` read inside a string the scan skipped —
        // fails visibly rather than resolving anything.
        let err = eval_once("nd('a', 2) + 1", 0.0, Locals::new(), &fparam()).unwrap_err();
        assert!(err.0.contains("not defined"), "got: {}", err.0);
        // `globals` alone is still Python's own builtin, so the failure there is the ATTRIBUTE, not
        // the name — which is why this asserts the error and not its wording.
        assert!(eval_once("globals.gain + 1", 0.0, Locals::new(), &fparam()).is_err());
    }

    #[test]
    fn constant_and_time_expressions_need_no_locals_at_all() {
        let _interp = interp();
        assert!((f("1 + 2", Locals::new()) - 3.0).abs() < 1e-9);
        let r = eval_once("t * 2", 4.0, Locals::new(), &fparam()).unwrap();
        assert!(matches!(r, Param::Float { value, .. } if (value - 8.0).abs() < 1e-9));
    }

    #[test]
    fn result_coerces_to_int_bool_and_str() {
        let _interp = interp();
        let ir = eval_once("2.7", 0.0, Locals::new(), &Param::Int { value: 0, vmin: -100, vmax: 100 }).unwrap();
        assert!(matches!(ir, Param::Int { value: 3, .. }), "float result rounds to nearest int");
        let br = eval_once("__v0 > 1", 0.0, Locals::from([frame("__v0", &[2.0])]), &Param::Bool { value: false })
            .unwrap();
        assert!(matches!(br, Param::Bool { value: true }), "a comparison over a frame drives a Bool");
        let sp = Param::Str { value: String::new(), options: None, refresh: false };
        let sr = eval_once("__v0", 0.0, Locals::from([value("__v0", Param::str_free("P07"))]), &sp).unwrap();
        assert!(matches!(sr, Param::Str { value, .. } if value == "P07"), "a str global reads as a str");
    }

    #[test]
    fn nonfinite_int_and_nonbool_trigger_error() {
        let _interp = interp();
        let ip = Param::Int { value: 0, vmin: -100, vmax: 100 };
        assert!(eval_once("float('inf')", 0.0, Locals::new(), &ip).is_err(), "inf into Int errors, not i64::MAX");
        let tp = Param::Trigger { fired: false };
        assert!(eval_once("1.5", 0.0, Locals::new(), &tp).is_err(), "non-bool into Trigger errors, not silent false");
    }

    #[test]
    fn compile_error_surfaces() {
        let _interp = interp();
        let ev = PyExprEvaluator::new().expect("interpreter");
        assert!(ev.compile("1 +").is_err(), "a syntax error must fail compile");
    }
