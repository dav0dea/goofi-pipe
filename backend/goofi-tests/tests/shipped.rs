//! A machine with no toolchain: the shipped nodes load from the artifacts built into the binary,
//! and only authoring is absent — named as such, never a silent gap.

use goofi_tests::{drive, f32s, j, Goofi, OutputProbe};

#[test]
fn a_shipped_node_runs_with_no_cargo_and_an_authored_one_says_what_it_needs() {
    // Before the first boot: a build dir nothing pre-warmed, a cargo that does not exist, and a
    // home an EARLIER build left its tree in, whose files are not this build's and must not scan.
    let fresh = tempfile::tempdir().unwrap();
    std::env::set_var("GOOFI_BUILD_DIR", fresh.path());
    std::env::set_var("CARGO", fresh.path().join("no-cargo"));
    let home = tempfile::tempdir().unwrap();
    std::env::set_var("GOOFI_HOME", home.path());
    let shipped = goofi_core::home::dir().join("shipped").join(env!("CARGO_PKG_VERSION"));
    let stale = shipped.join("stale").join("signal");
    std::fs::create_dir_all(&stale).unwrap();
    std::fs::write(stale.join("Stale.rs"), "").unwrap();
    let g = Goofi::new();
    let trees: Vec<_> = g.state.roots.iter().map(|r| r.parent().unwrap().to_path_buf()).collect();
    assert!(trees.iter().all(|t| t == &trees[0] && t.starts_with(&shipped) && t != &shipped.join("stale")),
            "every shipped root lies under ONE tree, this build's own: {trees:?}");
    assert!(g.call("library list", j!({}))["types"].as_array().unwrap().iter().all(|v| v["type"] != "signal:Stale"),
            "a file an earlier build left in the home is not a node");
    let uid = g.add("LFO");
    let probe = OutputProbe::open(&g.state.graph.lock().unwrap(), uid, "out");
    g.until("the shipped LFO to emit", |g| probe.frame(&mut g.state.graph.lock().unwrap()));
    let osc = g.add("Osc");
    let tap = OutputProbe::open(&g.state.graph.lock().unwrap(), osc, "out");
    g.until("the shipped audio oscillator to sound", |g| {
        drive(g, 4800);
        tap.frame(&mut g.state.graph.lock().unwrap()).filter(|d| f32s(d).iter().any(|v| v.abs() > 0.5))
    });

    let dir = g.state.mount().join("nodes_signal");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("Twice.rs"), "fn never_built() {}\n").unwrap();
    g.call("library refresh", j!({}));
    let row = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == "signal:Twice").cloned().expect("the file is listed, greyed");
    assert_eq!(row["available"], false, "{row}");
    assert!(row["doc"].as_str().is_some_and(|d| d.contains("cargo")), "what is missing is named: {row}");
}
