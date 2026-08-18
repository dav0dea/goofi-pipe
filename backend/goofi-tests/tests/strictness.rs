//! Tolerance belongs to replay; strictness belongs to the fresh caller.
//!
//! `Command::execute` is deliberately idempotent, because an `Err` inside `CommandHistory::flip`
//! wedges a session's undo stack forever. `apply` — the first-hand RPC path — called that same
//! `execute`, so eight ops answered success for work they had not done. The gate is
//! `Command::precondition`, checked in `apply` and never in `flip`. Both halves are pinned here:
//! a fresh call that names nothing is refused, and a stale toggle over the same ground still flips.

use goofi_tests::{hex, j, Goofi};

/// A canonical 12-hex uid that names nothing.
const GHOST: &str = "ffffffffffff";

#[test]
fn a_reply_says_what_the_write_actually_did() {
    let g = Goofi::new();

    // A node is born with a minted name, and with slots and params the caller cannot know from the
    // type alone — a param may be seeded from a `default_expr`, and `nd()` addresses the node by
    // that name. A bare uid made the next act an `inspect_node`.
    let born = g.call("add_node", j!({ "type": "Oscillator" }));
    let osc = born["uid"].as_str().unwrap().to_string();
    assert!(born["name"].as_str().is_some_and(|n| !n.is_empty()), "{born}");
    assert_eq!(born["output_slots"]["out"], "ARRAY", "{born}");
    assert_eq!(born["params"]["oscillator"]["frequency"], 1.0, "{born}");

    // A literal is COERCED to the param's declared type, so the value stored is not always the one
    // asked for, and a bare success asserted the caller's own number had landed.
    let buf = g.add("Buffer");
    let coerced = g.call("update_param", j!({ "node": hex(buf), "group": "buffer",
                                             "name": "size", "value": 512.6 }));
    assert_eq!(coerced["value"], 513, "an int param rounds: {coerced}");

    // A link's endpoints are RESOLVED and its dtype agreed between the two slots, so what got
    // wired is not literally what was asked for. Both come back.
    let wired = g.call("add_link", j!({ "node_out": osc, "slot_out": "out",
                                        "node_in": hex(buf), "slot_in": "data" }));
    assert_eq!(wired["node_out"], osc, "{wired}");
    assert_eq!(wired["dtype"], "ARRAY", "{wired}");
}

#[test]
fn an_idempotent_write_reports_that_it_did_nothing() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    assert_eq!(g.call("remove_node", j!({ "node": GHOST }))["removed"], false);
    assert_eq!(g.call("remove_node", j!({ "node": hex(osc) }))["removed"], true);
    assert_eq!(
        g.call("remove_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                   "node_in": hex(osc), "slot_in": "data" }))["removed"],
        false
    );
}

#[test]
fn a_refusal_names_what_the_caller_could_try_instead() {
    let g = Goofi::new();

    // A global's TYPE is what every expression reading it depends on. `set_global` took the type
    // off the wire, so re-typing `default_ufreq` as a string broke the reference, not the call.
    let why = g.refuse("set_global", j!({ "name": "default_ufreq", "value": "fast", "type": "string" }));
    assert!(why.contains("float") && why.contains("default_ufreq"), "{why}");
    assert_eq!(g.call("set_global", j!({ "name": "default_ufreq", "value": 12.5,
                                        "type": "float" }))["value"], 12.5);

    // A harness name is a closed set the caller cannot see from here, so a refusal that does not
    // name it leaves nothing to try next.
    let why = g.refuse("spawn_harness", j!({ "harness": "claude-code" }));
    assert!(why.contains("claude") && why.contains("codex"), "{why}");
}

#[test]
fn add_link_refuses_an_endpoint_that_names_nothing_wirable() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");

    let why = g.refuse("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                        "node_in": GHOST, "slot_in": "data" }));
    assert!(why.contains("node_in") && why.contains(GHOST), "{why}");

    // The same hole one step over: a REAL boundary port with no inner slot behind it. There is
    // nothing to wire, and the reply used to claim there was.
    let inst = g.call("group_nodes", j!({ "members": [hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let bnd = g.call("add_boundary", j!({ "inst_id": inst, "dir": "in", "dtype": "ARRAY",
                                          "pos": [0.0, 0.0] }))["bnd_id"].as_str().unwrap().to_string();
    let why = g.refuse("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                        "node_in": inst, "slot_in": bnd }));
    assert!(why.contains("wire_boundary"), "an unwired port names the op that fills it: {why}");

    // …and once the port IS wired the same call lands, so the refusal gates the impossible rather
    // than sub-patch wiring itself.
    g.call("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd,
                                 "inner_node": hex(buf), "inner_slot": "data" }));
    let made = g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                       "node_in": inst, "slot_in": bnd }));
    assert_eq!(made["node_in"], hex(buf), "the boundary resolves to its leaf: {made}");
}

#[test]
fn every_op_that_names_a_missing_target_refuses_it() {
    let g = Goofi::new();
    let buf = g.add("Buffer");
    let inst = g.call("group_nodes", j!({ "members": [hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();

    for (op, payload) in [
        ("wire_boundary", j!({ "inst_id": inst, "bnd_id": "in9", "inner_node": hex(buf), "inner_slot": "data" })),
        ("remove_boundary", j!({ "inst_id": inst, "bnd_id": "in9" })),
        ("rename_boundary", j!({ "inst_id": inst, "bnd_id": "in9", "name": "left" })),
        ("set_boundary_pos", j!({ "inst_id": inst, "bnd_id": "in9", "pos": [1.0, 2.0] })),
        ("expand_instance", j!({ "inst_id": GHOST })),
        ("set_node_pos", j!({ "node": GHOST, "pos": [1.0, 2.0] })),
        ("rename_node", j!({ "node": GHOST, "name": "renamed" })),
        ("set_expression", j!({ "node": GHOST, "group": "buffer", "name": "size",
                                "expression": "1", "enabled": true })),
    ] {
        g.refuse(op, payload);
    }

    // The other half of `wire_boundary`: a stub that DOES exist, aimed at an inner target that
    // cannot take the wire. `set_stub_inner` already refused this; the command swallowed it.
    let bnd = g.call("add_boundary", j!({ "inst_id": inst, "dir": "in", "dtype": "ARRAY",
                                          "pos": [0.0, 0.0] }))["bnd_id"].as_str().unwrap().to_string();
    g.refuse("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd,
                                   "inner_node": hex(buf), "inner_slot": "nope" }));
}

#[test]
fn rename_node_refuses_a_name_that_cannot_survive_nd_rewriting() {
    // A display name is spliced into expression SOURCE by `rewrite_nd_refs`, which replaces the
    // literal's content span in place. A quote or backslash therefore yields `nd('a'b')` — invalid
    // Python the referring node carries as a binding error while the rename reports success.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    for bad in ["a'b", "a\\b", "a\"b"] {
        g.refuse("rename_node", j!({ "node": hex(osc), "name": bad }));
    }
    g.call("rename_node", j!({ "node": hex(osc), "name": "a b-2" }));
}

#[test]
fn a_stale_link_toggle_still_flips_after_a_peer_deleted_an_endpoint() {
    let one = Goofi::new();
    let two = one.client("s2");
    let osc = one.add("Oscillator");
    let buf = one.add("Buffer");
    let link = j!({ "node_out": hex(osc), "slot_out": "out", "node_in": hex(buf), "slot_in": "data" });
    one.call("add_link", link.clone());
    one.call("remove_link", link);
    two.call("remove_node", j!({ "node": hex(buf) }));

    // s1's newest toggle is now an AddLink onto a dead uid. Erroring here would wedge s1's stack
    // forever — undo keeps re-selecting the entry it cannot flip.
    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert_eq!(one.call("redo", j!({}))["changed"], true);
    for _ in 0..4 {
        assert_eq!(one.call("undo", j!({}))["changed"], true, "the stack stays walkable to empty");
    }
}

#[test]
fn a_stale_boundary_toggle_still_flips_after_a_peer_removed_the_stub() {
    let one = Goofi::new();
    let two = one.client("s2");
    let buf = one.add("Buffer");
    let inst = one.call("group_nodes", j!({ "members": [hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let bnd = one.call("add_boundary", j!({ "inst_id": inst, "dir": "in", "dtype": "ARRAY",
                                            "pos": [0.0, 0.0] }))["bnd_id"].as_str().unwrap().to_string();
    one.call("rename_boundary", j!({ "inst_id": inst, "bnd_id": bnd, "name": "left" }));
    two.call("remove_boundary", j!({ "inst_id": inst, "bnd_id": bnd }));

    one.call("undo", j!({}));
    one.call("redo", j!({}));
}
