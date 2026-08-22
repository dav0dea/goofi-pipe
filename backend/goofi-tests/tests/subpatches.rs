//! Sub-patches: a flat tree of uids plus stub "symlinks", and no sharing.
//!
//! A boundary is a NAMING indirection — the runtime link is always flat, leaf to leaf. A port is a
//! node to every op that can name one, and the two link acts that look like one crossing are two
//! ordinary links in two scopes: node→facade above, port→member inside.

use std::sync::Arc;

use serde_json::Value;

use goofi_core::Param;
use goofi_node::{BindingId, Compiled, EvalCtx, ExprError, ExprEvaluator};
use goofi_tests::{hex, j, Goofi};

/// Compiles anything and hands the target value back. With one installed, a binding error in the
/// reply is the GRAPH's own resolution talking rather than "no evaluator here".
struct Always;

impl ExprEvaluator for Always {
    fn compile(&self, _source: &str) -> Result<Compiled, ExprError> {
        Ok(Compiled { id: 1 })
    }
    fn eval(&self, _id: BindingId, ctx: &EvalCtx<'_>) -> Result<Param, ExprError> {
        Ok(ctx.target.clone())
    }
    fn release(&self, _id: BindingId) {}
}

fn group(g: &Goofi, members: &[String]) -> String {
    g.call("group_nodes", j!({ "members": members, "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().expect("group answers an inst_id").to_string()
}

fn boundary(g: &Goofi, inst: &str, dir: &str) -> String {
    let ty = if dir == "in" { "InArray" } else { "OutArray" };
    g.call("add_node", j!({ "type": ty, "inst_id": inst, "pos": [0.0, 0.0] }))
        ["uid"].as_str().expect("a port uid").to_string()
}

/// The inner wire of a port, as the op vocabulary spells it: a link inside the sub-patch, with the
/// port on whichever end its direction puts it.
fn wire(g: &Goofi, bnd: &str, dir: &str, node: &str, slot: &str) -> Value {
    let p = match dir {
        "in" => j!({ "node_out": bnd, "slot_out": "value", "node_in": node, "slot_in": slot }),
        _ => j!({ "node_out": node, "slot_out": slot, "node_in": bnd, "slot_in": "value" }),
    };
    g.call("add_link", p)
}

#[test]
fn grouping_re_tags_the_members_and_expanding_gives_them_back() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    let inst = group(&g, &[hex(osc), hex(buf)]);

    let rec = g.doc()["nodes"][&inst].clone();
    // A facade is a node record; a TOP-LEVEL one simply names no scope, exactly as a root node does.
    assert_eq!(rec["type"], "SubPatch", "the facade is a node record: {rec}");
    assert!(rec.get("scope").is_none(), "a top-level scope names no parent: {rec}");
    assert!(rec.get("def_id").is_none(), "no sharing ⇒ no def_id");
    assert_eq!(g.members(&inst).len(), 2, "both members in the scope");
    assert!(g.members(&inst).contains(&hex(osc)));

    g.call("expand_instance", j!({ "inst_id": inst }));
    assert!(g.instances().is_empty(), "the instance dropped out of the forest");
    assert_eq!(g.nodes().len(), 2, "and both leaves came back to root");
}

#[test]
fn a_node_added_inside_an_entered_scope_stays_inside_it_through_undo_and_redo() {
    // The placement rides on the COMMAND, so a missing field shows up at undo→redo first.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let scope = group(&g, &[hex(osc), hex(buf)]);

    let inner = g.call("add_node", j!({ "type": "Buffer", "inst_id": scope, "pos": [10.0, 20.0] }))
        ["uid"].as_str().unwrap().to_string();
    assert!(g.members(&scope).contains(&inner), "a DIRECT member of the entered scope");

    g.call("undo", j!({}));
    assert!(!g.nodes().contains(&inner), "undo removed the node");

    g.call("redo", j!({}));
    assert!(g.members(&scope).contains(&inner), "redo put it back INSIDE the scope, not at root");
}

#[test]
fn add_node_refuses_an_inst_id_it_cannot_honour_and_creates_nothing() {
    // No partial mutation and no silent rooting.
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
    assert_eq!(g.members(&inst), vec![hex(buf)], "osc dropped from the scope's members too");
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
    wire(&g, &bnd, "in", &hex(buf), "data");

    g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                           "node_in": inst, "slot_in": bnd }));

    // Two cables, one per scope: the port's inner wire, and the external one — which the runtime
    // holds flat leaf→leaf, so the facade endpoint the caller NAMED is not what it stored.
    let links = g.doc()["links"].as_array().cloned().unwrap_or_default();
    assert_eq!(links.len(), 2, "the external cable and the port's inner one: {links:?}");
    assert_eq!(g.inner(&bnd), Some((hex(buf), "data".into())), "the inner one, inside the scope");
    let flat = links.iter().find(|l| l["node_out"] == hex(osc)).expect("the external cable");
    assert_eq!(flat["node_in"], hex(buf), "resolved to the inner leaf, not the instance");
    assert_eq!(flat["slot_in"], "data");
}

#[test]
fn a_boundary_is_authored_wired_and_renamed_without_changing_its_id() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    let inst = group(&g, &[hex(buf)]);

    // The wire, the label and the pill are three node ops — `compound` is what keeps them one
    // undo step now that the boundary has no op of its own.
    let bnd = boundary(&g, &inst, "out");
    g.call("compound", j!({ "ops": [
        { "op": "edit_node", "payload": { "node": bnd, "name": "wave", "pos": [12.0, 34.0] } },
        { "op": "add_link", "payload": { "node_out": hex(buf), "slot_out": "out",
                                         "node_in": bnd, "slot_in": "value" } },
    ] }));

    let port = g.doc()["nodes"][&bnd].clone();
    assert_eq!(port["type"], "OutArray", "the direction and the dtype ARE the type");
    assert_eq!(port["scope"], inst, "…and the sub-patch it is a port of is its scope");
    assert_eq!(g.inner(&bnd), Some((hex(buf), "out".into())), "the inner wire is a link: {port}");
    assert_eq!(port["name"], "wave", "renamed, and the port's uid is unchanged");
    assert_eq!(port["pos"], j!({ "x": 12.0, "y": 34.0 }));

    assert_eq!(g.call("undo", j!({}))["changed"], true);
    let back = g.doc()["nodes"][&bnd].clone();
    assert!(back["name"] != "wave" && g.inner(&bnd).is_none(),
            "one ctrl-Z took the whole edit back: {back}");
}

#[test]
fn a_boundary_wires_to_a_nested_scopes_own_port() {
    // A nested sub-patch's collapsed facade handles ARE its own stub ids.
    let g = Goofi::new();
    let buf = g.add("Buffer");
    let inner = group(&g, &[hex(buf)]);
    let outer = group(&g, std::slice::from_ref(&inner));

    let ib = boundary(&g, &inner, "out");
    wire(&g, &ib, "out", &hex(buf), "out");
    let ob = boundary(&g, &outer, "out");
    // The nested scope's facade is wired at ITS port, which is a slot name like any other.
    wire(&g, &ob, "out", &inner, &ib);

    assert_eq!(g.inner(&ob), Some((inner.clone(), ib.clone())),
               "wired to the nested scope's facade at its own port, not dropped");
}

#[test]
fn unwiring_a_boundary_prunes_its_target_and_keeps_the_pill() {
    // Deleting an In→member edge is an UNWIRE, and `Command::WireStub` is the only door to it.
    let g = Goofi::new();
    let buf = g.add("Buffer");
    let inst = group(&g, &[hex(buf)]);
    let bnd = boundary(&g, &inst, "in");
    wire(&g, &bnd, "in", &hex(buf), "data");

    // Cutting the inner cable is `remove_link` — the same op that cuts any other.
    let cut = g.call("remove_link", j!({ "node_out": bnd, "slot_out": "value",
                                        "node_in": hex(buf), "slot_in": "data" }));
    assert_eq!(cut["removed"], true, "the cut says it found the wire");
    assert_eq!(g.inner(&bnd), None, "the leaf is pruned, not left stale");
    assert_eq!(g.doc()["nodes"][&bnd]["type"], "InArray", "the pill itself survives the unwire");

    // Idempotent like every other remove, so a second cut is a no-op that says so.
    assert_eq!(g.call("remove_link", j!({ "node_out": bnd, "slot_out": "value",
                                         "node_in": hex(buf), "slot_in": "data" }))["removed"], false);

    // A port carries ONE inner wire, so a second is refused rather than replacing the first.
    wire(&g, &bnd, "in", &hex(buf), "data");
    let second = g.add("Buffer");
    g.refuse("add_link", j!({ "node_out": bnd, "slot_out": "value",
                             "node_in": hex(second), "slot_in": "data" }));
}

#[test]
fn a_boundary_op_refuses_a_port_or_a_target_it_cannot_honour() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let inst = group(&g, &[hex(buf)]);

    // A port needs the sub-patch it is a port OF; at root it is not a thing that can exist.
    g.refuse("add_node", j!({ "type": "InArray", "pos": [0.0, 0.0] }));

    // An edit that names NEITHER field is a caller error, not a silent no-op.
    let bnd = boundary(&g, &inst, "in");
    g.refuse("edit_node", j!({ "node": bnd }));
    // …and a port carries no params and no viewers, so asking is refused rather than dropped.
    g.refuse("edit_node", j!({ "node": bnd, "params": { "common": { "autotrigger": true } } }));

    // A port that DOES exist, aimed at an inner target that cannot take the wire.
    g.refuse("add_link", j!({ "node_out": bnd, "slot_out": "value",
                             "node_in": hex(buf), "slot_in": "nope" }));
    // …and one aimed the wrong way round: an input port FEEDS the sub-patch.
    g.refuse("add_link", j!({ "node_out": hex(buf), "slot_out": "out",
                             "node_in": bnd, "slot_in": "value" }));

    // …and a cable onto a real but UNWIRED port names the op that fills the port.
    let why = g.refuse("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                        "node_in": inst, "slot_in": bnd }));
    assert!(why.contains("add_link it to a member"), "an unwired port names the op that fills it: {why}");
    // Once the port IS wired the same call lands, so the refusal gates the impossible.
    wire(&g, &bnd, "in", &hex(buf), "data");
    let made = g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                       "node_in": inst, "slot_in": bnd }));
    assert_eq!(made["node_in"], hex(buf), "the boundary resolves to its leaf: {made}");
}

#[test]
fn a_stale_boundary_toggle_still_flips_after_a_peer_removed_the_port() {
    let one = Goofi::new();
    let two = one.client("s2");
    let buf = one.add("Buffer");
    let inst = group(&one, &[hex(buf)]);
    let bnd = boundary(&one, &inst, "in");
    one.call("edit_node", j!({ "node": bnd, "name": "left" }));
    two.call("remove_node", j!({ "node": bnd }));

    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert_eq!(one.call("redo", j!({}))["changed"], true);
}

#[test]
fn an_expression_reads_a_port_and_follows_the_wire_behind_it() {
    // A port carries no frame of its own, so `nd('port')` binds to the stream BEHIND it — and
    // unlike a node's, that stream MOVES when somebody wires the sub-patch, so the binding has to
    // be re-resolved by the graph rather than re-written by the user.
    let g = Goofi::new();
    g.state.graph.lock().unwrap().set_evaluator(Arc::new(Always));
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let inst = group(&g, &[hex(buf)]);

    let inp = boundary(&g, &inst, "in");
    wire(&g, &inp, "in", &hex(buf), "data");
    g.call("edit_node", j!({ "node": inp, "name": "wall" }));

    // A member reads its own sub-patch's input port. Nothing feeds it yet, so the refusal names it.
    let bind = |expr: &str| {
        g.call("edit_node", j!({ "node": hex(buf), "params": { "common": { "max_frequency":
                                     { "expression": expr } } } }))
            ["params"]["common"]["max_frequency"]["error"].clone()
    };
    let why = bind("nd('wall')");
    assert!(why.as_str().is_some_and(|e| e.contains("wall") && e.contains("wired")),
            "an unwired port says so, and says which: {why}");

    // Wiring the OUTSIDE resolves the same binding — nobody re-writes the expression.
    g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                           "node_in": inst, "slot_in": inp }));
    let bound = g.call("inspect_node", j!({ "node": hex(buf) }))["text"].as_str().unwrap().to_string();
    assert!(bound.contains("common.max_frequency = expr: nd('wall')"), "{bound}");
    assert!(!bound.contains("[error:"), "the wire behind the port made it resolvable: {bound}");

    // …and cutting that wire takes it back, through the same door.
    g.call("remove_link", j!({ "node_out": hex(osc), "slot_out": "out",
                              "node_in": inst, "slot_in": inp }));
    let cut = g.call("inspect_node", j!({ "node": hex(buf) }))["text"].as_str().unwrap().to_string();
    assert!(cut.contains("[error:") && cut.contains("wired"),
            "the binding follows the wire away again: {cut}");

    // A rename follows into the expression, exactly as a node's does.
    g.call("edit_node", j!({ "node": inp, "name": "left" }));
    let renamed = g.call("inspect_node", j!({ "node": hex(buf) }))["text"].as_str().unwrap().to_string();
    assert!(renamed.contains("nd('left')"), "the reference followed the port's rename: {renamed}");

    // An OUT port drains the sub-patch, so there is nothing to read from it.
    let outp = boundary(&g, &inst, "out");
    g.call("edit_node", j!({ "node": outp, "name": "sink" }));
    let refused = bind("nd('sink')");
    assert!(refused.as_str().is_some_and(|e| e.contains("no stream")), "{refused}");

    // A port's label lives in the ONE display-name namespace `nd()` reads, so it cannot shadow a
    // node's — a second `left` would make the reference above ambiguous.
    g.call("edit_node", j!({ "node": hex(osc), "name": "source" }));
    g.refuse("edit_node", j!({ "node": inp, "name": "source" }));
    g.refuse("add_node", j!({ "type": "Buffer", "name": "left" }));
}

#[test]
fn a_port_wears_a_viewer_on_the_stream_it_exposes() {
    // A port never runs, so a viewer on one has to reach the stream BEHIND it — the same reducer
    // the source's own viewer uses, because there is one stream per (node, slot) whatever the
    // viewer count.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let inst = group(&g, &[hex(buf)]);
    let inp = boundary(&g, &inst, "in");
    wire(&g, &inp, "in", &hex(buf), "data");
    g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                           "node_in": inst, "slot_in": inp }));

    // An IN port wears an output slot, so it takes a viewer exactly as a node does.
    g.call("edit_node", j!({ "node": inp, "viewers": { "value": { "kind": "line" } } }));
    let doc = g.doc();
    let stored = doc["nodes"][&inp]["viewers"].as_str().expect("a viewer blob rides as a string");
    assert!(stored.contains("line"), "the port kept the view state: {stored}");

    // …and it is refused on a slot the port does not have, rather than stored and never drawn.
    g.refuse("edit_node", j!({ "node": inp, "viewers": { "out": { "kind": "line" } } }));

    // An OUT port drains the sub-patch: no output, so no viewer.
    let outp = boundary(&g, &inst, "out");
    g.refuse("edit_node", j!({ "node": outp, "viewers": { "value": { "kind": "line" } } }));

    // The blob survives a save and a load, as a node's does.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ports.gfi");
    g.call("save", j!({ "path": path.to_string_lossy() }));
    let other = Goofi::new();
    other.call("load", j!({ "path": path.to_string_lossy() }));
    assert_eq!(other.doc()["nodes"][&inp]["viewers"], g.doc()["nodes"][&inp]["viewers"]);
}
