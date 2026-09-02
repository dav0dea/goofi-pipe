//! A machine with no toolchain: the shipped nodes load from the artifacts built into the binary,
//! and only authoring is absent — named as such, never a silent gap.

use goofi_tests::{j, Goofi, OutputProbe};

#[test]
fn a_shipped_node_runs_with_no_cargo_and_an_authored_one_says_what_it_needs() {
    // Before the first boot: a build dir nothing pre-warmed, and a cargo that does not exist.
    let fresh = tempfile::tempdir().unwrap();
    std::env::set_var("GOOFI_BUILD_DIR", fresh.path());
    std::env::set_var("CARGO", fresh.path().join("no-cargo"));
    let g = Goofi::new();
    let uid = g.add("Oscillator");
    let probe = OutputProbe::open(&g.state.graph.lock().unwrap(), uid, "out");
    g.until("the shipped oscillator to emit", |g| probe.frame(&mut g.state.graph.lock().unwrap()));

    let dir = g.state.mount().join("nodes_signal");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("Twice.rs"), "fn never_built() {}\n").unwrap();
    g.call("library refresh", j!({}));
    let row = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == "Twice").cloned().expect("the file is listed, greyed");
    assert_eq!(row["available"], false, "{row}");
    assert!(row["missing_deps"].to_string().contains("cargo"), "what is missing is named: {row}");
}
