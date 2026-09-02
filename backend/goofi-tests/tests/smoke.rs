//! The harness proving itself: a node is born, wired, runs, and is observed — every door the rest
//! of the suite uses, exercised once so a harness fault reads as a harness failure.

use goofi_tests::{Goofi, ep, j};
use goofi_view::Reducible; // shape()/ndim() on a decoded frame

#[test]
fn a_node_is_born_wired_and_runs() {
    let g = Goofi::new();
    let src = g.add("_TestCounter");
    let dst = g.add("_TestEcho");
    g.ready(src);
    g.ready(dst);

    let probe = g.probe(dst, "out");
    g.link(src, "out", dst, "input");

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
    // …and false for the node that DOES carry one, or the line above is decoration.
    assert!(!g.stays(|g| g.error(bad).is_none()), "the watcher can see a fault");
}

#[test]
fn a_refusal_says_what_was_wrong() {
    let g = Goofi::new();
    let n = g.add("_TestEcho");
    let why = g.refuse("link add", j!({ "from": ep(goofi_tests::hex(n), "nope"), "to": ep(goofi_tests::hex(n), "input") }));
    assert!(why.contains("nope"), "the refusal names the slot: {why}");
}

/// The replicated projection read as plain JSON through the ordinary op path.
#[test]
fn the_state_clients_replicate_is_readable_as_plain_json() {
    let g = Goofi::new();
    let n = g.add("_TestEcho");
    let doc = g.doc();
    assert_eq!(doc["nodes"][goofi_tests::hex(n)]["type"], "_TestEcho", "{doc}");
    assert!(doc["globals"]["default_ufreq"].is_object(), "the seeded system globals: {doc}");

    g.call("node remove", j!({ "node": goofi_tests::hex(n) }));
    assert!(g.doc()["nodes"].get(goofi_tests::hex(n)).is_none(), "and a removal leaves no tombstone");
}
