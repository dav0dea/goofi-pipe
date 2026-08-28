//! Headless operation: one mode with three doors, folded into one boolean before anything reads
//! it. The layout group is NOT REGISTERED — the one spelling of the mode — so every surface
//! shrinks with it, and the arrangement a patch carries rides through untouched.

use goofi_bridge::phrase;
use goofi_tests::{j, Goofi};

#[test]
fn a_headless_server_serves_no_layout_op_and_carries_an_arrangement_through() {
    // A patch AUTHORED with a layout, on a full instance.
    let full = Goofi::new();
    full.add("Oscillator");
    full.call("layout panel add", j!({ "name": "Scope" }));
    let yaml = full.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    let authored = full.doc()["arrangement"].clone();
    assert!(authored["tabs"].as_array().is_some_and(|t| t.len() == 2), "{authored}");

    let g = Goofi::headless();

    // The rows are ABSENT, not filtered on read: the index, dispatch and the resolver agree.
    let ops = g.call("op list", j!({}));
    let names: Vec<&str> =
        ops["ops"].as_array().unwrap().iter().filter_map(|o| o["op"].as_str()).collect();
    assert!(!names.iter().any(|n| n.starts_with("layout")), "no layout rows: {names:?}");
    assert!(names.contains(&"node add") && names.contains(&"node snapshot"), "{names:?}");
    let why = g.refuse("layout inspect", j!({}));
    assert!(why.contains("unknown op"), "{why}");
    // …and the phrase resolver adds the one teachable line for the group's first word.
    let words = vec!["layout".to_string(), "inspect".to_string()];
    let Err(why) = phrase::resolve(g.state.ops(), &words) else { panic!("resolved on headless") };
    assert!(why.contains("headless"), "the refusal names the mode: {why}");

    // The same line on a FULL server is an ordinary suggestion list instead.
    let bad = vec!["layout".to_string(), "frobnicate".to_string()];
    let Err(why) = phrase::resolve(full.state.ops(), &bad) else { panic!("frobnicate resolved") };
    assert!(why.contains("layout inspect") && !why.contains("headless"), "{why}");

    // Completion agrees: the group word is offered on the full server and absent headless.
    let offered = |g: &Goofi| phrase::complete(g.state.ops(), None, "").iter()
        .any(|(w, _)| w == "layout");
    assert!(offered(&full) && !offered(&g), "completion follows the served set");

    // A patch saved headless keeps the arrangement it arrived with, untouched — the whole
    // manifest survives BYTE-identical, so no layout machinery had to run to carry it.
    g.call("session load", j!({ "content": yaml }));
    assert_eq!(g.doc()["arrangement"], authored, "the arrangement rode through");
    let saved = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    assert_eq!(saved, yaml, "a headless save is the manifest it loaded, byte for byte");

    // Everything OUTSIDE the group still runs: the mode is one group absent, not a second track.
    let osc = g.add("Oscillator");
    g.set_param(osc, "oscillator", "sfreq", 32.0);
    assert_eq!(g.call("undo", j!({}))["changed"], true);
}
