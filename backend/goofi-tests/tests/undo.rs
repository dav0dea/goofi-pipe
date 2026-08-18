//! Undo and redo — manager-owned, per-session, and uid-stable.
//!
//! Every mutating op records an exact inverse; the history stores one TOGGLE per entry, so a redo
//! restores the very uid the undo removed. A session is a browser tab: two of them share one graph
//! and one history, and each undoes only its own work.

use goofi_tests::{hex, j, Goofi};

#[test]
fn an_add_survives_undo_and_redo_at_the_same_uid() {
    let g = Goofi::new();
    let n = g.add("Oscillator");
    assert_eq!(g.nodes(), vec![hex(n)]);

    let u = g.call("undo", j!({}));
    assert_eq!(u["changed"], true);
    assert_eq!(u["can_undo"], false, "nothing left to undo");
    assert_eq!(u["can_redo"], true, "the undone add is redoable");
    assert!(g.nodes().is_empty());

    assert_eq!(g.call("redo", j!({}))["changed"], true);
    assert_eq!(g.nodes(), vec![hex(n)], "redo restored the SAME uid");
}

#[test]
fn a_session_undoes_only_its_own_work() {
    let one = Goofi::new();
    let two = one.client("s2");
    let a = one.add("Oscillator");
    let b = two.add("Buffer");

    one.call("undo", j!({}));
    assert_eq!(one.nodes(), vec![hex(b)], "s1's undo left s2's node standing");

    two.call("undo", j!({}));
    assert!(one.nodes().is_empty());
    let _ = a;
}

#[test]
fn a_fresh_command_clears_the_sessions_redo_run() {
    let g = Goofi::new();
    g.add("Oscillator");
    g.call("undo", j!({}));
    g.add("Buffer");

    let r = g.call("redo", j!({}));
    assert_eq!(r["changed"], false, "the redo run went with the new command");
    assert_eq!(r["can_redo"], false);
}

#[test]
fn a_link_undoes_without_taking_its_endpoints() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    assert_eq!(g.doc()["links"].as_array().map(Vec::len), Some(1));

    g.call("undo", j!({}));
    assert_eq!(g.doc()["links"].as_array().map(Vec::len), Some(0), "the wire went");
    assert_eq!(g.nodes().len(), 2, "and both endpoints stayed");
}

#[test]
fn a_param_edit_undoes_to_the_value_before_it() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let set = |v: f64| {
        g.call("update_param", j!({ "node": hex(osc), "group": "common",
                                    "name": "max_frequency", "value": v }));
    };
    let read = || g.doc()["nodes"][hex(osc)]["params"]["common"]["max_frequency"]["value"].as_f64();
    set(20.0);
    set(33.0);
    assert_eq!(read(), Some(33.0));

    g.call("undo", j!({}));
    assert_eq!(read(), Some(20.0), "the value before the last edit, not the default");
    g.call("redo", j!({}));
    assert_eq!(read(), Some(33.0));
}

#[test]
fn a_global_rename_is_one_undo_step() {
    let g = Goofi::new();
    g.call("add_global", j!({ "name": "subj", "value": "P01", "type": "string" }));
    g.call("rename_global", j!({ "old": "subj", "new": "participant" }));
    assert_eq!(g.doc()["globals"]["participant"]["value"], "P01");
    assert!(g.doc()["globals"]["subj"].is_null());

    // ONE undo, though the rename is an add plus a remove composed into a Compound.
    g.call("undo", j!({}));
    assert_eq!(g.doc()["globals"]["subj"]["value"], "P01", "name and value back together");
    assert!(g.doc()["globals"]["participant"].is_null());
}

#[test]
fn a_global_add_refuses_a_collision_rather_than_upserting() {
    let g = Goofi::new();
    g.call("add_global", j!({ "name": "subj", "value": "P01", "type": "string" }));
    g.refuse("add_global", j!({ "name": "subj", "value": "P02", "type": "string" }));
    g.refuse("add_global", j!({ "name": "default_ufreq", "value": 5, "type": "int" }));
    assert_eq!(g.doc()["globals"]["subj"]["value"], "P01", "the refused write changed nothing");
}

#[test]
fn a_refused_system_rename_leaves_no_half_applied_half() {
    let g = Goofi::new();
    g.refuse("rename_global", j!({ "old": "default_ufreq", "new": "foo" }));
    // The rename is a Compound whose first child would have created `foo`. The guard runs before
    // it, so neither half lands.
    assert!(g.doc()["globals"]["foo"].is_null(), "no phantom global leaked");
    assert!(!g.doc()["globals"]["default_ufreq"].is_null());
}

#[test]
fn deleting_a_sub_patch_restores_its_whole_subtree_on_undo() {
    let g = Goofi::new();
    let a = g.add("Oscillator");
    let b = g.add("Buffer");
    g.link(a, "out", b, "data");
    let inst = g.call("group_nodes", j!({ "members": [hex(a), hex(b)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();

    g.call("remove_node", j!({ "node": inst }));
    assert!(g.nodes().is_empty() && g.instances().is_empty(), "the subtree went with the scope");

    g.call("undo", j!({}));
    assert_eq!(g.instances(), vec![inst], "scope back at the same uid");
    assert_eq!(g.nodes(), {
        let mut v = vec![hex(a), hex(b)];
        v.sort();
        v
    }, "and both members with it");
}

#[test]
fn grouping_undoes_and_redoes_at_the_same_scope_uid_with_its_stub() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let sink = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    g.link(buf, "out", sink, "data");

    let scope = g.call("group_nodes", j!({ "members": [hex(osc), hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let stubs = |g: &Goofi| g.doc()["instances"][&scope]["stubs"].as_object().map(|m| m.len()).unwrap_or(0);
    assert!(stubs(&g) > 0, "the buf→sink cut is exposed as a stub");

    g.call("undo", j!({}));
    assert!(g.instances().is_empty() && g.nodes().len() == 3, "expanded, all three leaves at root");

    g.call("redo", j!({}));
    assert_eq!(g.instances(), vec![scope.clone()], "the SAME scope uid");
    assert!(stubs(&g) > 0, "with its stub back");
}

#[test]
fn a_boundary_undoes_without_taking_its_scope() {
    let g = Goofi::new();
    let buf = g.add("Buffer");
    let scope = g.call("group_nodes", j!({ "members": [hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let count = |g: &Goofi| g.doc()["instances"][&scope]["stubs"].as_object().map(|m| m.len()).unwrap_or(0);
    let before = count(&g);

    let bnd = g.call("add_boundary", j!({ "inst_id": scope, "dir": "in", "dtype": "ARRAY", "pos": [0.0, 0.0] }))
        ["bnd_id"].as_str().unwrap().to_string();
    assert_eq!(count(&g), before + 1);

    g.call("undo", j!({}));
    assert!(g.doc()["instances"][&scope]["stubs"][&bnd].is_null(), "the stub went");
    assert_eq!(g.instances(), vec![scope], "the scope stayed");
}
