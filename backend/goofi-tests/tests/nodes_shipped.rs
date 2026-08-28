//! The node library goofi SHIPS, run as a user gets it: the real `.py` files under `nodes/`,
//! installed through the same probe the CLI's scan uses, wired to real producers and read back.
#![cfg(not(feature = "embed"))]


use goofi_tests::{f32s, install, require_python, shape, Goofi};

/// One of the `.py` files goofi SHIPS, installed through the same seam a user's own file takes.
fn install_shipped(g: &Goofi, py: &str, file: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../nodes").join(file);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read the shipped node {}: {e}", path.display()));
    install(g, py, file, &source)
}

#[test]
fn the_entropy_nodes_goofi_ships_reduce_the_time_axis_and_leave_the_channels_alone() {
    // A frame that is NOT a vector: against a single channel a flattening node reads as correct.
    let py = require_python();
    let g = Goofi::new();
    let src = g.add("_TestGrid");
    let buf = g.add("Buffer");
    g.set_param(buf, "buffer", "size", 256);
    g.link(src, "out", buf, "data");

    // All four PROBED AND WIRED AT ONCE, one interpreter per file, as a real scan does — so the
    // ceiling below covers the node boots rather than the probes.
    let (g, py) = (&g, &py.py);
    let nodes: Vec<_> = std::thread::scope(|s| {
        [
            ("lempel_ziv.py", "complexity"),
            ("permutation_entropy.py", "entropy"),
            ("spectral_entropy.py", "entropy"),
            ("detrended_fluctuation.py", "exponent"),
        ]
        .map(|(file, slot)| {
            s.spawn(move || {
                let ty = install_shipped(g, py, file);
                let node = g.add(&ty);
                let probe = g.probe(node, slot);
                g.link(buf, "out", node, "data");
                (ty, node, probe)
            })
        })
        .into_iter()
        .map(|h| h.join().expect("a probe thread panicked"))
        .collect()
    });

    for (ty, node, probe) in nodes {
        // The birth barrier splits "never started" from "started and failed", and the error channel
        // is read WHILE waiting, so a node that says why fails with its own words.
        g.until(&format!("{ty} to start"), |g| {
            if let Some(e) = g.error(node) {
                panic!("{ty} failed to start: {e}");
            }
            (g.stage(node) == "ready").then_some(())
        });
        let d = g.until(&format!("{ty} to answer once it is ready"), |g| {
            if let Some(e) = g.error(node) {
                panic!("{ty} failed instead of answering: {e}");
            }
            probe.latest().filter(|d| shape(d) == vec![3])
        });
        let v = f32s(&d);
        assert!(v.iter().all(|x| x.is_finite()), "{ty} answered {v:?}");
        // The three rows are one signal at three offsets, so answers that DISAGREE mean a mix.
        assert!(
            v.iter().all(|x| (x - v[0]).abs() <= v[0].abs() * 1e-3 + 1e-4),
            "{ty} read the three channels as three different signals: {v:?}",
        );
        assert!(g.error(node).is_none(), "{ty} carries no error: {:?}", g.error(node));
    }
}

#[test]
fn a_shipped_entropy_node_reads_a_real_signal_rather_than_answering_a_constant() {
    // A sine says otherwise: permutation entropy of one is solidly inside its range.
    let py = require_python();
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.set_param(osc, "oscillator", "sfreq", 256.0);
    g.set_param(buf, "buffer", "size", 256);
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
