//! Sub-patches: a flat tree of uids plus stub "symlinks", and no sharing.
//!
//! A boundary is a NAMING indirection — the runtime link is always flat, leaf to leaf. Everything
//! here reads the instance forest out of the replicated projection, which is where it lives; the
//! structural broadcast events were retired. The boundary ops carry their own half of the
//! strictness rule, because a stub is the one place where a target can be real and still unwirable.

use serde_json::Value;

use goofi_tests::{hex, j, Goofi};

fn group(g: &Goofi, members: &[String]) -> String {
    g.call("group_nodes", j!({ "members": members, "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().expect("group answers an inst_id").to_string()
}

fn boundary(g: &Goofi, inst: &str, dir: &str) -> String {
    g.call("add_boundary", j!({ "inst_id": inst, "dir": dir, "dtype": "ARRAY", "pos": [0.0, 0.0] }))
        ["bnd_id"].as_str().expect("a stub id").to_string()
}

fn members(g: &Goofi, scope: &str) -> Vec<String> {
    let mut v: Vec<String> = g.doc()["instances"][scope]["members"].as_object()
        .map(|m| m.keys().cloned().collect()).unwrap_or_default();
    v.sort();
    v
}

#[test]
fn grouping_re_tags_the_members_and_expanding_gives_them_back() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    let inst = group(&g, &[hex(osc), hex(buf)]);

    let rec = g.doc()["instances"][&inst].clone();
    // A top-level scope's parent must be ROOT, not null: the editor renders it on the root canvas
    // only when `instance.parent === ROOT_ID`.
    assert_eq!(rec["parent"], "__root__");
    assert!(rec.get("def_id").is_none(), "no sharing ⇒ no def_id");
    assert_eq!(members(&g, &inst).len(), 2, "both members in the scope");
    assert!(members(&g, &inst).contains(&hex(osc)));

    g.call("expand_instance", j!({ "inst_id": inst }));
    assert!(g.instances().is_empty(), "the instance dropped out of the forest");
    assert_eq!(g.nodes().len(), 2, "and both leaves came back to root");
}

#[test]
fn a_node_added_inside_an_entered_scope_stays_inside_it_through_undo_and_redo() {
    // The canvas renders only the entered scope's children, so a dropped `inst_id` makes the node
    // invisible exactly where it was placed. The placement rides on the COMMAND, so it has to
    // survive undo→redo too — a missing command field shows up there first.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let scope = group(&g, &[hex(osc), hex(buf)]);

    let inner = g.call("add_node", j!({ "type": "Buffer", "inst_id": scope, "pos": [10.0, 20.0] }))
        ["uid"].as_str().unwrap().to_string();
    assert!(members(&g, &scope).contains(&inner), "a DIRECT member of the entered scope");

    g.call("undo", j!({}));
    assert!(!g.nodes().contains(&inner), "undo removed the node");

    g.call("redo", j!({}));
    assert!(members(&g, &scope).contains(&inner), "redo put it back INSIDE the scope, not at root");
}

#[test]
fn add_node_refuses_an_inst_id_it_cannot_honour_and_creates_nothing() {
    // No partial mutation and no silent rooting. Dropping the scope would put the node where the
    // canvas cannot show it; creating it and THEN failing would leave the graph and its mirror
    // disagreeing.
    let g = Goofi::new();
    let osc = g.add("Oscillator");

    g.refuse("add_node", j!({ "type": "Buffer", "inst_id": "deadbeef" }));   // hex, but no scope
    g.refuse("add_node", j!({ "type": "Buffer", "inst_id": "not-a-uid" }));  // not hex at all
    g.refuse("add_node", j!({ "type": "Buffer", "inst_id": hex(osc) }));     // a leaf is not a scope
    assert_eq!(g.nodes(), vec![hex(osc)], "no refused add left a node behind");
}

#[test]
fn removing_a_grouped_member_leaves_no_dangling_entry() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let inst = group(&g, &[hex(osc), hex(buf)]);

    g.call("remove_node", j!({ "node": hex(osc) }));
    assert_eq!(members(&g, &inst), vec![hex(buf)], "osc dropped from the scope's members too");
    assert!(!g.nodes().contains(&hex(osc)), "and out of the graph");
    assert_eq!(g.instances(), vec![inst], "the instance survives its other member");
}

#[test]
fn a_cable_onto_a_boundary_resolves_to_the_inner_leaf() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let inst = group(&g, &[hex(buf)]); // no links yet, so no auto boundaries
    let bnd = boundary(&g, &inst, "in");
    g.call("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd,
                                "inner_node": hex(buf), "inner_slot": "data" }));

    g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                           "node_in": inst, "slot_in": bnd }));

    let links = g.doc()["links"].as_array().cloned().unwrap_or_default();
    assert_eq!(links.len(), 1, "one flat leaf→leaf link");
    assert_eq!(links[0]["node_in"], hex(buf), "resolved to the inner leaf, not the instance");
    assert_eq!(links[0]["slot_in"], "data");
    assert_eq!(links[0]["node_out"], hex(osc), "the plain endpoint passes through");
}

#[test]
fn a_boundary_is_authored_wired_and_renamed_without_changing_its_id() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    // Grouping only the Buffer cuts osc→buf, so an input boundary appears; buf.out has no
    // downstream, so an output one can be authored onto it.
    let inst = group(&g, &[hex(buf)]);

    let bnd = boundary(&g, &inst, "out");
    g.call("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd,
                                "inner_node": hex(buf), "inner_slot": "out" }));
    g.call("rename_boundary", j!({ "inst_id": inst, "bnd_id": bnd, "name": "wave" }));

    let port = g.doc()["instances"][&inst]["stubs"][&bnd].clone();
    assert_eq!(port["dir"], "out");
    assert_eq!(port["inner_node"], hex(buf));
    assert_eq!(port["inner_slot"], "out");
    assert_eq!(port["name"], "wave", "renamed, and the stub id is unchanged");
}

#[test]
fn a_boundary_wires_to_a_nested_scopes_own_port() {
    // Inside an entered scope a nested sub-patch shows its collapsed facade, whose handles ARE its
    // own stub ids — so dropping a cable from the parent's pill onto one sends exactly this.
    let g = Goofi::new();
    let buf = g.add("Buffer");
    let inner = group(&g, &[hex(buf)]);
    let outer = group(&g, std::slice::from_ref(&inner));

    let ib = boundary(&g, &inner, "out");
    g.call("wire_boundary", j!({ "inst_id": inner, "bnd_id": ib,
                                "inner_node": hex(buf), "inner_slot": "out" }));
    let ob = boundary(&g, &outer, "out");
    g.call("wire_boundary", j!({ "inst_id": outer, "bnd_id": ob,
                                "inner_node": inner, "inner_slot": ib }));

    let port = g.doc()["instances"][&outer]["stubs"][&ob].clone();
    assert_eq!(port["inner_node"], inner, "wired to the nested scope's facade, not dropped");
    assert_eq!(port["inner_slot"], ib, "…at its own boundary port");
}

#[test]
fn unwiring_a_boundary_prunes_its_target_and_keeps_the_pill() {
    // Deleting an In→member edge is an UNWIRE: the pill survives and its inner target clears.
    // `Command::WireStub` models exactly that, and the delete path is the only door to it.
    let g = Goofi::new();
    let buf = g.add("Buffer");
    let inst = group(&g, &[hex(buf)]);
    let bnd = boundary(&g, &inst, "in");
    g.call("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd,
                                "inner_node": hex(buf), "inner_slot": "data" }));

    // The frontend sends nulls for BOTH halves to clear the target.
    g.call("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd,
                                "inner_node": null, "inner_slot": null }));
    let port = g.doc()["instances"][&inst]["stubs"][&bnd].clone();
    assert_eq!(port["inner_node"], Value::Null, "the leaf is pruned, not left stale");
    assert_eq!(port["inner_slot"], Value::Null);
    assert_eq!(port["dir"], "in", "the pill itself survives the unwire");

    // Naming ONE half is the third state the pair must not admit: neither a wire nor an unwire.
    g.refuse("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd, "inner_node": hex(buf) }));
}

#[test]
fn a_boundary_op_refuses_a_port_or_a_target_it_cannot_honour() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let inst = group(&g, &[hex(buf)]);

    // Every op that names a port refuses one that is not there.
    for (op, payload) in [
        ("wire_boundary", j!({ "inst_id": inst, "bnd_id": "in9", "inner_node": hex(buf), "inner_slot": "data" })),
        ("remove_boundary", j!({ "inst_id": inst, "bnd_id": "in9" })),
        ("rename_boundary", j!({ "inst_id": inst, "bnd_id": "in9", "name": "left" })),
        ("set_boundary_pos", j!({ "inst_id": inst, "bnd_id": "in9", "pos": [1.0, 2.0] })),
    ] {
        g.refuse(op, payload);
    }

    // A port that DOES exist, aimed at an inner target that cannot take the wire. `set_stub_inner`
    // already refused this; the command used to swallow the refusal.
    let bnd = boundary(&g, &inst, "in");
    g.refuse("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd,
                                   "inner_node": hex(buf), "inner_slot": "nope" }));

    // …and a cable onto a real but UNWIRED port has nothing behind it to reach. The reply used to
    // claim there was; it now names the op that fills the port.
    let why = g.refuse("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                        "node_in": inst, "slot_in": bnd }));
    assert!(why.contains("wire_boundary"), "an unwired port names the op that fills it: {why}");
    // Once the port IS wired the same call lands, so the refusal gates the impossible rather than
    // sub-patch wiring itself.
    g.call("wire_boundary", j!({ "inst_id": inst, "bnd_id": bnd,
                                 "inner_node": hex(buf), "inner_slot": "data" }));
    let made = g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                       "node_in": inst, "slot_in": bnd }));
    assert_eq!(made["node_in"], hex(buf), "the boundary resolves to its leaf: {made}");
}

#[test]
fn a_stale_boundary_toggle_still_flips_after_a_peer_removed_the_port() {
    // The replay half of the same rule: an `Err` inside a flip wedges that session's undo stack.
    let one = Goofi::new();
    let two = one.client("s2");
    let buf = one.add("Buffer");
    let inst = group(&one, &[hex(buf)]);
    let bnd = boundary(&one, &inst, "in");
    one.call("rename_boundary", j!({ "inst_id": inst, "bnd_id": bnd, "name": "left" }));
    two.call("remove_boundary", j!({ "inst_id": inst, "bnd_id": bnd }));

    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert_eq!(one.call("redo", j!({}))["changed"], true);
}
