//! The node library goofi SHIPS, run as a user gets it: the real `.py` files under `nodes/`,
//! installed through the same probe the CLI's scan uses, wired to real producers and read back.
//!
//! Its own test BINARY, and that is the point. Every node here imports antropy, which imports
//! numba — seconds of CPU per interpreter, and this scenario starts eight of them. Sharing a
//! process with `python.rs` made that file's own latency assertions fail one after another as the
//! load moved around: 38 s and green alone, 150 s and flaky together. A node whose dependency is
//! genuinely heavy is not a node whose neighbours should be re-timed around it.
//!
//! Subprocess tier throughout, and not by choice: antropy re-enables the GIL, so the routing probe
//! quarantines every one of these — which is exactly the tier a user gets for them. That is also
//! why the file is compiled out of the `embed` build rather than run by both: it reaches Python
//! only through `subproc`, so the second run is the same code path at the same tier — 79 s of the
//! gate list to re-decide what the first run decided. What `embed` adds is the OTHER tier, and the
//! suites that drive it are the ones that assert on it.
#![cfg(not(feature = "embed"))]

use std::process::{Command, Stdio};

use goofi_core::Data;
use goofi_tests::{hex, j, Goofi};

fn f32s(d: &Data) -> Vec<f32> {
    let goofi_core::Value::Array(a) = d.value() else { panic!("not an array: {d:?}") };
    a.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn shape(d: &Data) -> Vec<usize> {
    let goofi_core::Value::Array(a) = d.value() else { panic!("not an array: {d:?}") };
    a.shape().to_vec()
}

/// Serializes this binary's tier tests. Cargo runs a crate's tests on parallel threads and every
/// one of these spawns an interpreter, so without this a latency or liveness assertion is taken
/// while a dozen siblings boot numpy on the same cores.
static TIER: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The interpreter to spawn children with, plus the tier lock — held for the rest of the test.
struct Tier {
    py: String,
    _lock: std::sync::MutexGuard<'static, ()>,
}

/// A python with BOTH goofi (the abi3 wheel) and numpy, or a PANIC naming the fix. These tests
/// HARD-REQUIRE one rather than skipping: a silent skip once hid real bugs for days.
///
/// The probe strips `PYTHONPATH` exactly as the real child spawn does, so a host/pyo3 `PYTHONPATH`
/// cannot produce a false negative — it once made the venv python import an incompatible numpy.
fn require_python() -> Tier {
    // A panicking test poisons the mutex; recover rather than cascade its failure onto every
    // sibling, which would bury the one real error.
    let _lock = TIER.lock().unwrap_or_else(|e| e.into_inner());
    let mut cands: Vec<String> = std::env::var("GOOFI_SUBPROC_TEST_PYTHON").into_iter().collect();
    // Both venv layouts — `bin/` on unix, `Scripts/` on Windows — because the fallbacks are worse
    // than a miss there: `python3` on Windows is an App Execution Alias that answers every probe
    // with a Microsoft Store advert instead of failing.
    cands.push(format!("{}/../../.gfivenv/bin/python", env!("CARGO_MANIFEST_DIR")));
    cands.push(format!("{}/../../.gfivenv/Scripts/python.exe", env!("CARGO_MANIFEST_DIR")));
    cands.push("python3".into());
    cands.push("python".into());
    for py in cands {
        let ok = Command::new(&py).arg("-c").arg("import goofi, numpy")
            .env_remove("PYTHONPATH").env_remove("PYTHONHOME")
            .stdout(Stdio::null()).stderr(Stdio::null())
            .status().is_ok_and(|s| s.success());
        if ok {
            return Tier { py, _lock };
        }
    }
    panic!("no python with goofi + numpy found (checked $GOOFI_SUBPROC_TEST_PYTHON, \
            ./.gfivenv/bin/python, python3, python). Run `cargo run -p goofi-init`, which creates \
            the venvs and installs the goofi wheel into them.");
}

/// Write `source` into the patch's own node directory, probe it the way the CLI's scan does, and
/// register what comes back. Answers the type name the palette now offers.
///
/// This is the real seam: `probe` + `node_type_from` are the pair the node scan routes with, so a
/// node reaching the graph through them reaches it the way a user's file does.
fn install(g: &Goofi, py: &str, file: &str, source: &str) -> String {
    let dir = g.state.mount().join("nodes");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(file);
    std::fs::write(&path, source).unwrap();
    match goofi_python::subproc::probe(&path, py) {
        goofi_python::Discovery::Found(d) => {
            let t = goofi_python::subproc::node_type_from(py, d);
            let name = t.manifest.type_name.to_string();
            g.register_dyn(t.manifest, t.factory);
            name
        }
        goofi_python::Discovery::Unavailable { reason, .. } =>
            panic!("{file} probed as unavailable: {reason}"),
        goofi_python::Discovery::Skip => panic!("{file} was not taken for a node file at all"),
    }
}


/// One of the `.py` files goofi SHIPS, installed through the same seam a user's own file takes.
/// Reading the real file is the point: a node that ships broken — an API that moved under it, a
/// dependency provisioning forgot — is a node every user gets, and no hand-written fixture beside
/// it would notice.
fn install_shipped(g: &Goofi, py: &str, file: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../nodes").join(file);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read the shipped node {}: {e}", path.display()));
    install(g, py, file, &source)
}

#[test]
fn the_entropy_nodes_goofi_ships_reduce_the_time_axis_and_leave_the_channels_alone() {
    // Four measures over the same window, on a frame that is NOT a vector — because the mistake
    // they all invite is to flatten first, and against a single channel a flattening node and a
    // correct one are indistinguishable.
    let py = require_python();
    let g = Goofi::new();
    let src = g.add("_TestGrid");
    let buf = g.add("Buffer");
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 256 }));
    g.link(src, "out", buf, "data");

    // All four wired BEFORE any of them is judged: each is its own interpreter importing antropy,
    // and waiting one out before starting the next spends that import four times over.
    let nodes: Vec<_> = [
        ("lempel_ziv.py", "complexity"),
        ("permutation_entropy.py", "entropy"),
        ("spectral_entropy.py", "entropy"),
        ("detrended_fluctuation.py", "exponent"),
    ]
    .map(|(file, slot)| {
        let ty = install_shipped(&g, &py.py, file);
        let node = g.add(&ty);
        let probe = g.probe(node, slot);
        g.link(buf, "out", node, "data");
        (ty, node, probe)
    })
    .into_iter()
    .collect();

    for (ty, node, probe) in nodes {
        // [3, 256] in, [3] out: the measure consumes time and hands back one value per channel.
        let d = g.until(&format!("{ty} to answer"), |_| probe.latest().filter(|d| shape(d) == vec![3]));
        let v = f32s(&d);
        assert!(v.iter().all(|x| x.is_finite()), "{ty} answered {v:?}");
        // The three rows are the same signal at three offsets, and every one of these measures
        // ignores a constant offset — so three answers that DISAGREE mean the rows were mixed.
        assert!(
            v.iter().all(|x| (x - v[0]).abs() <= v[0].abs() * 1e-3 + 1e-4),
            "{ty} read the three channels as three different signals: {v:?}",
        );
        assert!(g.error(node).is_none(), "{ty} carries no error: {:?}", g.error(node));
    }
}

#[test]
fn a_shipped_entropy_node_reads_a_real_signal_rather_than_answering_a_constant() {
    // The scenario above is blind to a node that returns the same number whatever it is given: its
    // rows are one signal three times over. A sine says otherwise — permutation entropy of one is
    // solidly inside its range, where a flat or a saturated answer is not.
    let py = require_python();
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.call("update_param", j!({ "node": hex(osc), "group": "oscillator", "name": "sfreq", "value": 256.0 }));
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 256 }));
    let node = g.add(&install_shipped(&g, &py.py, "permutation_entropy.py"));
    let probe = g.probe(node, "entropy");
    g.link(osc, "out", buf, "data");
    g.link(buf, "out", node, "data");

    let d = g.until("a permutation entropy of a full window", |_| {
        probe.latest().filter(|d| shape(d) == vec![1] && f32s(d)[0] > 0.0)
    });
    let e = f32s(&d)[0];
    assert!((0.3..0.9).contains(&e), "a sine's permutation entropy is neither flat nor maximal: {e}");
}
