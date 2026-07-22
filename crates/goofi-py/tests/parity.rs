//! Backend parity: ONE `goofi.Node` source, run in-process (`PyNode`, free-threaded host)
//! and in a subprocess (`RemoteNode`, `goofi.serve`), must produce identical output. This is
//! the M4 proof that the two tiers share one marshalling seam (`goofi_pymod::exec`) and one
//! transport codec — they can't drift. The node returns a NON-f32 (int32) array, so it also
//! exercises the shared cast-to-f32 (+ deduped `warn_cast_once`) on BOTH tiers by construction.
//!
//! Runs only with `embed` + a free-threaded interpreter (the in-process host). The subprocess
//! interpreter defaults to the same one (it has goofi+numpy); override with
//! GOOFI_SUBPROC_TEST_PYTHON. Example:
//!   cargo test -p goofi-py --features embed --test parity
#![cfg(feature = "embed")]

use goofi_core::{Data, Meta, Param, Value};
use goofi_node::{Inputs, Node, NodeCtx, Outputs, ParamGroups, Params};
use goofi_py::PyNode;
use goofi_subproc::RemoteNode;
use indexmap::IndexMap;

/// One authored node, exercised by both tiers. Reads a param + input meta, and returns an
/// int32 array (length-preserving, bare return) — so the output cast + the carry-input-meta
/// path both run identically on each tier.
const SRC: &str = r#"
import goofi
import numpy as np
class Parity(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    @staticmethod
    def config_params():
        return {"gain": {"factor": goofi.IntParam(1, 0, 100)}}
    def setup(self):
        self._base = 10
    def process(self, data):
        return {"out": (data.data * self.params.gain.factor + self._base).astype(np.int32)}
"#;

fn params() -> ParamGroups {
    let mut p = ParamGroups::new();
    let mut g = IndexMap::new();
    g.insert("factor".to_string(), Param::int(3, 0, 100));
    p.insert("gain".to_string(), g);
    p
}

fn input() -> Data {
    let bytes: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    Data::array_f32(vec![3], bytes, Meta::new().with_sfreq(Some(250.0))).unwrap()
}

fn floats(d: &Data) -> Vec<f32> {
    match d.value() {
        Value::Array(s) => s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        _ => panic!("expected array"),
    }
}

/// Tick a node once with the `data` input + params, reading back `out`.
fn tick(node: &mut dyn Node, input: Data, params: &ParamGroups) -> Data {
    let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    inmap.insert("data", Some(input));
    let inp = Inputs::new(&inmap);
    let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    outmap.insert("out", None);
    let mut ctx = NodeCtx::new();
    {
        let mut out = Outputs::new(&mut outmap);
        node.process(&inp, &mut out, &mut ctx, &Params::new(params)).expect("process");
    }
    outmap.get("out").unwrap().clone().expect("output frame")
}

#[test]
fn in_process_and_subprocess_produce_identical_output() {
    let ft = goofi_py::interpreter_path().expect("no FT interpreter (PYO3_PYTHON) for the subprocess tier");
    let subpy = std::env::var("GOOFI_SUBPROC_TEST_PYTHON").unwrap_or(ft);
    let p = params();

    // In-process: seed params + run setup (like the engine), then tick.
    let mut inproc = PyNode::from_source(SRC, vec!["data"], vec!["out"]).expect("PyNode");
    inproc.setup(&mut NodeCtx::new(), &Params::new(&p)).expect("in-process setup");
    let a = tick(&mut inproc, input(), &p);

    // Subprocess: the child runs setup lazily on the first request.
    let mut remote = RemoteNode::new(&subpy, SRC, vec!["data"], vec!["out"]);
    let b = tick(&mut remote, input(), &p);

    // Identical values — int32 cast to f32, with the param (3) + setup base (10) applied:
    // 1*3+10=13, 2*3+10=16, 3*3+10=19.
    assert_eq!(floats(&a), floats(&b), "the two tiers must produce identical values");
    assert_eq!(floats(&a), vec![13.0, 16.0, 19.0], "shared cast + param + setup base");

    // Identical meta — sfreq carried on both (length-preserving bare return).
    assert_eq!(a.meta().sfreq(), b.meta().sfreq(), "identical carried meta");
    assert_eq!(a.meta().sfreq(), Some(250.0));

    // Both are f32 arrays of the same shape (the one cast seam).
    match (a.value(), b.value()) {
        (Value::Array(sa), Value::Array(sb)) => assert_eq!(sa.shape(), sb.shape()),
        _ => panic!("both tiers must return arrays"),
    }
}
