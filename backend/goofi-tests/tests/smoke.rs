//! The harness proving itself: a node is born, wired, runs, and is observed — every door the rest
//! of the suite uses, exercised once so a harness fault reads as a harness failure.

use goofi_tests::{j, Goofi};
use goofi_view::Reducible; // shape()/ndim() on a decoded frame

#[test]
fn a_node_is_born_wired_and_runs() {
    let g = Goofi::new();
    let src = g.add("_TestCounter");
    let dst = g.add("_TestEcho");
    g.ready(src);
    g.ready(dst);

    let probe = g.probe(dst, "out");
    g.link(src, "out", dst, "in");

    let frame = g.until("a frame to cross the wire", |_| probe.latest());
    assert_eq!(frame.shape(), &[1], "the counter emits one number: {frame:?}");
}

#[test]
fn the_doc_and_the_event_agree_with_the_op() {
    let g = Goofi::new();
    let mut ev = g.events();
    let n = g.add("_TestEcho");

    assert_eq!(ev.next("node_added")["uid"], goofi_tests::hex(n));
    assert_eq!(g.doc()["nodes"][goofi_tests::hex(n)]["type"], "_TestEcho");
}

#[test]
fn a_failing_node_reports_why_and_a_healthy_one_stays_quiet() {
    let g = Goofi::new();
    let bad = g.add("_TestFail");
    let good = g.add("_TestEcho");

    let why = g.until("the fault to reach the graph", |g| g.error(bad));
    assert!(why.contains("unplugged"), "{why}");
    assert!(g.stays(|g| g.error(good).is_none()), "a healthy node has no standing error");
    // …and the same watcher answers false for the node that DOES carry one, or the line above is
    // decoration: a `stays` that could only ever say true would pass against any bug at all.
    assert!(!g.stays(|g| g.error(bad).is_none()), "the watcher can see a fault");
}

#[test]
fn a_refusal_says_what_was_wrong() {
    let g = Goofi::new();
    let n = g.add("_TestEcho");
    let why = g.refuse("add_link", j!({ "node_out": goofi_tests::hex(n), "slot_out": "nope",
                                        "node_in": goofi_tests::hex(n), "slot_in": "in" }));
    assert!(why.contains("nope"), "the refusal names the slot: {why}");
}
