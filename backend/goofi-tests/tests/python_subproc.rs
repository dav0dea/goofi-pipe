//! The subprocess Python tier: a child interpreter running `goofi.serve`, reached over
//! iceoryx2 shared memory with seq-framed request/response.
//!
//! Unconditional — this tier only ever SPAWNS an interpreter, so nothing here links libpython.
//! (The liveness mechanism stays inside `goofi-python`: it reaches the child handle directly.)

use std::process::{Command, Stdio};
use std::time::Duration;

use goofi_core::{Data, Meta, Param, Value};
use goofi_node::{Inputs, Isolation, Node, NodeCtx, Outputs, ParamGroups, Params};
use goofi_python::subproc::{node_type_from, probe, RemoteNode, DEFAULT_TIMEOUT};
use goofi_python::*;
use indexmap::IndexMap;

// Node sources are authored to the `goofi.Node` class contract (the one authoring shape;
// the bare `def process(x)` function is gone). Raw strings keep Python indentation verbatim.

/// Doubles its `data` input into `out`.
const DOUBLE: &str = r#"
import goofi
class Double(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": data.data * 2.0}
"#;

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn arr(shape: Vec<usize>, v: &[f32], meta: Meta) -> Data {
    Data::array_f32(shape, f32s(v), meta).unwrap()
}
fn floats(d: &Data) -> Vec<f32> {
    match d.value() {
        Value::Array(s) => s
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        _ => panic!("not array"),
    }
}

/// Tick a node once: feed the named inputs + params, allocate the named output slots,
/// return the output slot map (or the node error).
fn tick(
    node: &mut RemoteNode,
    inputs: Vec<(&'static str, Data)>,
    out_names: &[&'static str],
    params: &ParamGroups,
) -> Result<IndexMap<&'static str, Option<Data>>, String> {
    let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    for (n, d) in inputs {
        inmap.insert(n, Some(d));
    }
    let inp = Inputs::new(&inmap);
    let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    for n in out_names {
        outmap.insert(n, None);
    }
    let mut ctx = NodeCtx::new();
    let r = {
        let mut out = Outputs::new(&mut outmap);
        node.process(&inp, &mut out, &mut ctx, &Params::new(params))
    };
    r.map_err(|e| e.0)?;
    Ok(outmap)
}

/// The common `data -> out` single-slot, no-param tick; returns the `out` frame.
fn run(node: &mut RemoteNode, d: Data) -> Data {
    try_run(node, d).expect("remote process")
}
fn try_run(node: &mut RemoteNode, d: Data) -> Result<Data, String> {
    let m = tick(node, vec![("data", d)], &["out"], &ParamGroups::new())?;
    Ok(m.get("out").unwrap().clone().expect("output frame"))
}

/// A python with BOTH goofi (the abi3 wheel) and numpy (the subprocess child needs both),
/// or None. Prefers `$GOOFI_SUBPROC_TEST_PYTHON`, then the repo's `.gfivenv`, then a PATH python.
/// The probe strips `PYTHONPATH` exactly like the real child spawn ([`Running::spawn`]), so a
/// host/pyo3 `PYTHONPATH` can't produce a false negative (it once masked real bugs by making
/// the venv python import an incompatible numpy → every tier test silently SKIPPED).
fn usable_python() -> Option<String> {
    let mut cands: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("GOOFI_SUBPROC_TEST_PYTHON") {
        cands.push(p);
    }
    // Both venv layouts — `bin/` on unix, `Scripts/` on Windows — because the fallbacks below
    // are worse than a miss on Windows: `python3` there is an App Execution Alias that answers
    // every probe with a Microsoft Store advert instead of failing.
    cands.push(format!("{}/../../.gfivenv/bin/python", env!("CARGO_MANIFEST_DIR")));
    cands.push(format!("{}/../../.gfivenv/Scripts/python.exe", env!("CARGO_MANIFEST_DIR")));
    cands.push("python3".to_string());
    cands.push("python".to_string());
    for cand in cands {
        if let Ok(out) = Command::new(&cand)
            .arg("-c")
            .arg("import goofi, numpy")
            .env_remove("PYTHONPATH")
            .env_remove("PYTHONHOME")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
        {
            if out.success() {
                return Some(cand);
            }
        }
    }
    None
}

/// Serializes the subprocess-tier tests. Cargo runs a crate's tests on parallel threads and
/// every one of these spawns a Python interpreter, so without this the latency measurement
/// below is taken while a dozen siblings are booting numpy on the same cores — it measured
/// the test harness, not the transport, and failed its budget at ~7 ms median.
static TIER: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The interpreter to spawn children with, plus the tier lock — held until the value drops,
/// i.e. for the rest of the test. Derefs to `&str`, so call sites read as a plain path.
struct Tier {
    py: String,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl std::ops::Deref for Tier {
    type Target = str;
    fn deref(&self) -> &str {
        &self.py
    }
}

/// Like [`usable_python`] but PANICS with an actionable message when none is found — the
/// subprocess tier tests HARD-REQUIRE a python (goofi + numpy) rather than silently
/// skipping, so a missing/misconfigured interpreter fails loudly instead of hiding bugs.
fn require_python() -> Tier {
    // A panicking test poisons the mutex; recover rather than cascade its failure onto
    // every sibling, which would bury the one real error.
    let _lock = TIER.lock().unwrap_or_else(|e| e.into_inner());
    let py = usable_python().unwrap_or_else(|| {
        panic!(
            "no python with goofi + numpy found (checked $GOOFI_SUBPROC_TEST_PYTHON, \
             ./.gfivenv/bin/python, python3, python). Run `cargo run -p goofi-init`, which \
             creates the venvs and installs the goofi wheel into them. The subprocess-tier \
             tests require one."
        )
    });
    Tier { py, _lock }
}

/// Counts in a background Python thread started from `setup()` — the shape of every device
/// input node (an OSC/LSL/serial receiver `serve_forever`-ing off the tick). `process` just
/// reports the count, so the parent can see whether the thread ran between two ticks.
const TICKER: &str = r#"
import threading, time
import numpy as np
import goofi
class Ticker(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def setup(self):
        self.count = 0
        def spin():
            while True:
                self.count += 1
                time.sleep(0.001)
        threading.Thread(target=spin, daemon=True).start()
    def process(self, data):
        return {"out": np.array([float(self.count)], dtype="float32")}
"#;

#[test]
fn a_nodes_own_python_thread_runs_while_the_child_is_idle() {
    // The subprocess tier exists to host GIL-bound libraries, and the canonical shape of a
    // device input is a receiver thread started in `setup()`. The child's serve loop is pure
    // Rust between requests, so if it holds the GIL across its idle sleep that thread is
    // starved for exactly as long as the node is not being ticked — which for an unwired or
    // slowly-paced node is forever.
    let py = require_python();
    let mut node = RemoteNode::new(&*py, TICKER, vec!["data"]);
    let d = arr(vec![1], &[0.0], Meta::empty());

    let first = floats(&run(&mut node, d.clone()))[0]; // cold start + setup + one tick
    std::thread::sleep(Duration::from_millis(300)); // an idle gap: no request in flight
    let second = floats(&run(&mut node, d))[0];

    // 300 ms at a 1 ms cadence is ~300 increments; a starved thread manages a couple at most,
    // stolen from the eval loop while `process` itself is running.
    assert!(
        second - first > 50.0,
        "the node's thread must run while the child idles: {first} -> {second}"
    );
}

/// A `RemoteNode` holds its iceoryx2 ports directly (via `ipc_threadsafe::Service`), so it must
/// stay `Send` for the scheduler. This compile-time guard fails loudly if the ports ever revert to
/// the `!Send` `ipc::Service` — the whole reason the port-owning io thread could be removed.
#[test]
fn remote_node_stays_send() {
    fn _assert_send<T: Send>() {}
    _assert_send::<RemoteNode>();
}

#[test]
fn remote_node_runs_a_class_node_in_a_subprocess() {
    let py = require_python();
    let src = r#"
import goofi
class Affine(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": data.data * 2.0 + 1.0}
"#;
    let mut node = RemoteNode::new(&*py, src, vec!["data"]);

    // Input carries sfreq/index; a length-preserving node carries the input meta back.
    let mut meta = Meta::empty();
    meta.set_sfreq(Some(128.0));
    meta.set_index(Some(7));
    let d = arr(vec![3], &[1.0, 2.0, 3.0], meta);

    // Two ticks on the same long-lived subprocess (proves it stays alive).
    let out1 = run(&mut node, d.clone());
    assert_eq!(floats(&out1), vec![3.0, 5.0, 7.0]);
    let out2 = run(&mut node, d);
    assert_eq!(floats(&out2), vec![3.0, 5.0, 7.0]);

    assert_eq!(out1.meta().sfreq(), Some(128.0));
    assert_eq!(out1.meta().index(), Some(7));
    match out1.value() {
        Value::Array(s) => assert_eq!(s.shape(), &[3]),
        _ => panic!("expected array"),
    }
}

#[test]
fn a_param_reaches_the_subprocess_node() {
    // Mirrors the in-process host test: `setup` seeds `self._base`; `process` reads a live
    // param `gain.factor`. Proves setup ran (its value appears) AND a param crossed the wire.
    let py = require_python();
    let src = r#"
import goofi
class Scale(goofi.Node):
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
        self._base = 100.0
    def process(self, data):
        return {"out": data.data * self.params.gain.factor + self._base}
"#;
    let mut node = RemoteNode::new(&*py, src, vec!["data"]);

    let mut params = ParamGroups::new();
    let mut gain = IndexMap::new();
    gain.insert("factor".to_string(), Param::int(3, 0, 100));
        params.insert("gain".to_string(), gain);

        let out = tick(&mut node, vec![("data", arr(vec![2], &[1.0, 2.0], Meta::empty()))], &["out"], &params)
            .expect("param tick");
        // 1*3 + 100 = 103 ; 2*3 + 100 = 106.
        assert_eq!(floats(out.get("out").unwrap().as_ref().unwrap()), vec![103.0, 106.0]);
    }

    #[test]
    fn psd_runs_over_the_transport_reading_sfreq() {
        let py = require_python();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../nodes/psd.py");
        // Discovery: CamelCase "Psd" on the subprocess tier, with its declared welch.nperseg param.
        let Discovery::Found(d) = probe(&path, &py) else { panic!("psd.py probes as a node") };
        let ty = node_type_from(&py, d);
        assert_eq!(ty.manifest.type_name, "Psd");
        assert_eq!(ty.manifest.isolation, Isolation::Subprocess);
        assert_eq!(ty.manifest.outputs[0].name, "psd");
        let nperseg = ty
            .manifest
            .params
            .iter()
            .find(|p| p.group == "welch" && p.name == "nperseg")
            .expect("the welch.nperseg param is discovered into the manifest");
        // End-to-end for `doc=`: authored in the .py, emitted by the installed wheel's probe,
        // parsed here into the manifest the UI renders its tooltip from.
        assert!(
            nperseg.doc.is_some_and(|d| d.contains("Window length")),
            "the param's doc= crossed the probe; got {:?}",
            nperseg.doc
        );

        // A 1x64 unit sine at exactly 8 cycles (bin 8, independent of sfreq); sfreq=1000 in
        // meta only scales the PSD magnitude, so a small peak proves the child read sfreq.
        let n = 64usize;
        let sfreq = 1000.0f64;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / n as f64).sin() as f32)
            .collect();
        let d = arr(vec![1, n], &samples, Meta::new().with_sfreq(Some(sfreq)));

        let src = std::fs::read_to_string(&path).unwrap();
        let mut node = RemoteNode::new(&*py, &src, vec!["data"]);
        let m = tick(&mut node, vec![("data", d)], &["psd"], &ParamGroups::new()).expect("psd tick");
        let out = m.get("psd").unwrap().as_ref().unwrap();

        match out.value() {
            Value::Array(s) => {
                assert_eq!(s.shape(), &[1, n / 2 + 1], "channels preserved; rfft bins");
                let psd = floats(out);
                let peak = (0..psd.len()).max_by(|a, b| psd[*a].total_cmp(&psd[*b])).unwrap();
                assert_eq!(peak, 8, "spectral peak at the input frequency bin");
                // sfreq=1000 normalization -> a small peak; the fallback sfreq=1 would be ~1000x larger.
                assert!(psd[peak] < 1.0, "peak {} implies sfreq=1000 reached the child, not the fallback", psd[peak]);
            }
            _ => panic!("expected array"),
        }
        // The node explicitly emits the frequency axis (a (array, meta) tuple return); that
        // node-authored meta crosses the transport intact — proving explicit output meta works.
        match out.meta().get("freqs") {
            Some(goofi_core::MetaValue::List(v)) => assert_eq!(v.len(), n / 2 + 1, "one freq per rfft bin"),
            other => panic!("expected a freqs list in the output meta, got {other:?}"),
        }
    }

    #[test]
    fn channels_preserved_on_2d_length_preserving_node() {
        // The canonical EEG case: a [2,3] array with dim0 channel labels through a length-preserving
        // node must come back [2,3] with channels intact (the shared full-meta codec carries them).
        let py = require_python();
        let mut node = RemoteNode::new(&*py, DOUBLE, vec!["data"]);

        let mut meta = Meta::empty();
        meta.set_channels(goofi_core::Axes::new().with(
            0,
            goofi_core::Axis::coords(vec![
                goofi_core::Coord::Str("Fz".into()),
                goofi_core::Coord::Str("Cz".into()),
            ]),
        ));
        let d = arr(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], meta);

        let out = try_run(&mut node, d).expect("2-D channel frame must round-trip");
        match out.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[2, 3], "shape preserved"),
            _ => panic!("expected array"),
        }
        assert_eq!(floats(&out), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let ch = out.meta().channels().get(0).and_then(|a| a.coords.clone()).expect("dim0 channels preserved");
        assert_eq!(ch.len(), 2);
    }

    #[test]
    fn subprocess_roundtrip_latency_and_stability() {
        // Concrete latency/stability read for the iceoryx2 subprocess tier: after cold start,
        // run many round-trips of a realistic EEG-sized frame and report the latency distribution.
        let py = require_python();
        let ident = r#"
import goofi
class Ident(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": data.data * 1.0}
"#;
    let mut node = RemoteNode::new(&*py, ident, vec!["data"]);
    // A 32-channel × 256-sample float32 frame (~32 KB) — a typical EEG buffer.
    let (c, t) = (32usize, 256usize);
    let vals: Vec<f32> = (0..c * t).map(|i| i as f32).collect();
    let make = || arr(vec![c, t], &vals, Meta::empty());

    // Warm up: the first tick pays cold start (spawn + goofi/numpy import + module compile).
    let cold = std::time::Instant::now();
    let _ = run(&mut node, make());
    let cold_ms = cold.elapsed().as_secs_f64() * 1e3;

    let iters = 300usize;
    let mut lat: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let out = run(&mut node, make());
        lat.push(t0.elapsed().as_secs_f64() * 1e3);
        assert_eq!(floats(&out).len(), c * t, "every tick round-trips intact (stability)");
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = lat.iter().sum::<f64>() / iters as f64;
    let p = |q: f64| lat[((iters as f64 * q) as usize).min(iters - 1)];
    eprintln!(
        "subproc iceoryx2 latency (32x256 f32, {iters} ticks): cold={cold_ms:.1}ms  \
         min={:.3}ms  p50={:.3}ms  p99={:.3}ms  max={:.3}ms  mean={mean:.3}ms",
        lat[0], p(0.50), p(0.99), lat[iters - 1]
    );
    // Steady-state p99 must be a small fraction of a 60 Hz tick (16.6 ms) — a generous
    // ceiling that still catches a regression to blocking/second-scale latency.
    assert!(p(0.99) < 10.0, "p99 round-trip {:.3}ms exceeds the budget", p(0.99));
}

#[test]
fn node_stdout_does_not_corrupt_the_transport() {
    // The child routes fd 1 to stderr before importing the node. A node that writes to stdout
    // (here a flushed print, but equally a C-extension's printf) must NOT inject bytes into the
    // SHM frame plane.
    let py = require_python();
    let src = r#"
import goofi
class Chatty(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        import sys
        print("debug from the node", flush=True)
        sys.stdout.flush()
        return {"out": data.data * 2.0}
"#;
    let mut node = RemoteNode::new(&*py, src, vec!["data"]);
    let out = try_run(&mut node, arr(vec![3], &[0.0, 1.0, 2.0], Meta::empty())).expect("a printing node still round-trips");
    assert_eq!(floats(&out), vec![0.0, 2.0, 4.0]);
    // A second tick proves the stream stayed in sync (not just the first frame).
    let out2 = try_run(&mut node, arr(vec![2], &[5.0, 6.0], Meta::empty())).expect("second tick in sync");
    assert_eq!(floats(&out2), vec![10.0, 12.0]);
}

#[test]
fn node_error_surfaces_fast_without_killing_the_child() {
    // A per-tick node RAISE must surface as an error FAST (not via a 10s respawn-timeout) and
    // must NOT kill the child — the SAME subprocess handles the next tick with state intact,
    // matching the in-process tier's Ok(Err). The Python exception text rides back.
    let py = require_python();
    let src = r#"
import goofi
class Boom(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def setup(self):
        self._ticks = 0
    def process(self, data):
        self._ticks += 1
        if data.data[0] < 0:
            raise ValueError("boom")
        return {"out": data.data + self._ticks}
"#;
    let mut node = RemoteNode::new(&*py, src, vec!["data"]);

    // A good tick: setup ran (ticks 0 -> 1), so 10 + 1 = 11.
    assert_eq!(floats(&run(&mut node, arr(vec![1], &[10.0], Meta::empty()))), vec![11.0]);

    // A raise: fast error carrying the exception text; the child stays alive.
    let t = std::time::Instant::now();
    let err = try_run(&mut node, arr(vec![1], &[-1.0], Meta::empty())).expect_err("a raise surfaces as an error");
        assert!(t.elapsed() < Duration::from_secs(2), "the error must surface fast, not via a respawn timeout");
        assert!(err.contains("boom"), "the Python exception text rides back: {err}");

        // The SAME child continues with state intact: ticks went 1 -> 2 (raised, still counted)
        // -> 3 here, so 10 + 3 = 13 proves NO respawn (a respawn would reset _ticks + re-run setup).
        assert_eq!(
            floats(&run(&mut node, arr(vec![1], &[10.0], Meta::empty()))),
            vec![13.0],
            "child survived the raise with its state (no respawn)"
        );
    }

    #[test]
    fn a_setup_that_raises_is_retried_on_the_next_tick() {
        // D3 across the tiers: the inline tier retries the whole initialization on any interaction,
        // so the child must too — it used to mark `did_setup` BEFORE running setup, deliberately,
        // "matching the in-process tier's run-once semantics". That parity argument now points the
        // other way. A device that was not ready at the first tick therefore comes back on the
        // next one, without a restart.
        let py = require_python();
        let src = r#"
import goofi
class LateBoot(goofi.Node):
    setups = 0
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def setup(self):
        LateBoot.setups += 1
        if LateBoot.setups < 2:
            raise RuntimeError("device is not open")
    def process(self, data):
        return {"out": data.data + LateBoot.setups}
"#;
    let mut node = RemoteNode::new(&*py, src, vec!["data"]);

    let err = try_run(&mut node, arr(vec![1], &[10.0], Meta::empty()))
        .expect_err("the first tick's setup raised");
    assert!(err.contains("device is not open"), "the Python exception text rides back: {err}");

    // The SAME child, one tick later: setup runs a second time and succeeds, so `process` runs
    // — 10 + 2 setups. A latched `did_setup` would either skip setup (and raise on the missing
    // attribute) or report the same failure forever.
    assert_eq!(
        floats(&run(&mut node, arr(vec![1], &[10.0], Meta::empty()))),
        vec![12.0],
        "the interaction retried the initialization and the node came up"
    );
}

#[test]
fn a_dead_child_is_reaped_and_respawns() {
    // If the child PROCESS actually dies (not a catchable raise — here os._exit), the tick
    // times out, the child is reaped, and a later tick respawns a fresh one.
    let py = require_python();
    let src = r#"
import os
import goofi
class Killer(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        if data.data[0] < 0:
            os._exit(1)
        return {"out": data.data * 2.0}
"#;
    let mut node = RemoteNode::new(&*py, src, vec!["data"]).with_timeout(Duration::from_millis(800));
    assert!(try_run(&mut node, arr(vec![1], &[-1.0], Meta::empty())).is_err(), "a dead child surfaces as an error");
        let out = try_run(&mut node, arr(vec![1], &[3.0], Meta::empty())).expect("must respawn on the next tick");
        assert_eq!(floats(&out), vec![6.0]);
    }

    #[test]
    fn an_exited_child_is_noticed_at_once_instead_of_waiting_out_the_timeout() {
        // A child that has ALREADY died — import crash, C-extension segfault, OOM kill — is not a
        // hang: there is nobody left to answer. Watching only the response port made every such
        // death cost the full production timeout (10 s of a parked worker) and then report
        // "did not respond in time", which names the wrong cause. Run at the DEFAULT timeout so
        // the assertion is about noticing the death, not about a short test timeout.
        let py = require_python();
        let src = r#"
import os
import goofi
class Killer(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        os._exit(3)
"#;
    // Built the only way a caller can, so it carries the production deadline.
    let mut node = RemoteNode::new(&*py, src, vec!["data"]);
    let t = std::time::Instant::now();
    let err = try_run(&mut node, arr(vec![1], &[1.0], Meta::empty())).expect_err("a dead child is an error");
        assert!(err.contains("subprocess exited"), "the error names the exit, not a timeout: {err}");
        assert!(
            t.elapsed() < Duration::from_secs(5),
            "noticed on the child's death, not after the {DEFAULT_TIMEOUT:?} deadline (took {:?})",
            t.elapsed()
        );
    }

    #[test]
    fn hung_subprocess_times_out_instead_of_hanging() {
        // A node that never responds must not block forever; with a short timeout the tick errors.
        let py = require_python();
        let src = r#"
import time
import goofi
class Hang(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        while True:
            time.sleep(1)
"#;
    let mut node = RemoteNode::new(&*py, src, vec!["data"]).with_timeout(Duration::from_millis(600));
    let t = std::time::Instant::now();
    let r = try_run(&mut node, arr(vec![1], &[1.0], Meta::empty()));
    assert!(r.is_err(), "a hung subprocess must error, not hang");
    assert!(t.elapsed() < Duration::from_secs(5), "must return near the timeout, not block");
}

#[test]
fn large_frame_into_a_stuck_child_times_out() {
    // The child blocks at module import (never enters its serve loop) AND the frame is large;
    // the publish is non-blocking SHM, so the tick must time out (not hang) with no response.
    let py = require_python();
    let src = r#"
import time
time.sleep(30)
import goofi
class Slow(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": data.data}
"#;
    let mut node = RemoteNode::new(&*py, src, vec!["data"]).with_timeout(Duration::from_millis(600));
    let n = 40_000usize; // 160 KB >> the 64 KiB initial slice — a big frame to a stuck child
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let t = std::time::Instant::now();
    let r = try_run(&mut node, arr(vec![n], &vals, Meta::empty()));
    assert!(r.is_err(), "a child stuck before the loop must error, not hang");
    assert!(t.elapsed() < Duration::from_secs(5), "must return near the timeout");
}

#[test]
fn large_frame_round_trips_over_shared_memory() {
    // A frame far larger than the 64 KiB initial slice must round-trip — iceoryx2 grows the
    // publisher's segment (PowerOfTwo), and the 4-byte sequence framing survives a big body.
    let py = require_python();
    let mut node = RemoteNode::new(&*py, DOUBLE, vec!["data"]);
    let n = 100_000usize; // 400 KB body
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let out = run(&mut node, arr(vec![n], &vals, Meta::empty()));
    let got = floats(&out);
    assert_eq!(got.len(), n, "shape preserved across the SHM round-trip");
    assert_eq!(got[1], 2.0, "1 * 2");
    assert_eq!(got[10], 20.0, "10 * 2");
    assert_eq!(got[n - 1], (n - 1) as f32 * 2.0, "last element doubled");
}


/// The per-file probe IS the discovery path: the CLI's router walks a directory itself and
/// asks this once per file, so what a scan yields is exactly what these three answers say.
/// Asserted over a directory rather than a lone file because the two files that must NOT
/// become nodes are the ones a real scan puts in its way.
#[test]
fn the_probe_yields_subprocess_types_that_run_and_passes_over_the_rest() {
    let py = require_python();
    let dir = std::env::temp_dir().join(format!("goofi_subdisc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let negate = r#"
import goofi
class Negate(goofi.Node):
    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}
    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}
    def process(self, data):
        return {"out": -data.data}
"#;
    std::fs::write(dir.join("negate.py"), negate).unwrap();
    std::fs::write(dir.join("_hidden.py"), negate).unwrap(); // underscore → skipped
    std::fs::write(dir.join("nope.py"), "x = 1\n").unwrap(); // no Node subclass → probe skips

    // Through the pair the CLI uses, so the test covers the path production takes.
    assert!(!matches!(probe(&dir.join("_hidden.py"), &py), Discovery::Found(_)), "`_`-prefixed is not a node");
    assert!(!matches!(probe(&dir.join("nope.py"), &py), Discovery::Found(_)), "no Node subclass → no type");
    let Discovery::Found(d) = probe(&dir.join("negate.py"), &py) else { panic!("a real node file") };
    let ty = node_type_from(&py, d);
    assert_eq!(ty.manifest.type_name, "Negate");
    assert_eq!(ty.manifest.category, "subprocess");
    assert_eq!(ty.manifest.isolation, Isolation::Subprocess);

    // The factory builds a working node.
    let mut node = (ty.factory)(&ParamGroups::new());
    let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    inmap.insert("data", Some(arr(vec![2], &[1.0, -2.0], Meta::empty())));
    let inp = Inputs::new(&inmap);
    let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    outmap.insert("out", None);
    let mut ctx = NodeCtx::new();
    let params = ParamGroups::new();
    {
        let mut out = Outputs::new(&mut outmap);
        node.process(&inp, &mut out, &mut ctx, &Params::new(&params)).unwrap();
    }
    assert_eq!(floats(outmap.get("out").unwrap().as_ref().unwrap()), vec![-1.0, 2.0]);
    drop(node);

    let _ = std::fs::remove_dir_all(&dir);
}
