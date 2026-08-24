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

/// The port of `inst` whose type is `ty`, and its inner wire.
fn port_of(g: &Goofi, inst: &str, ty: &str) -> (String, Option<(String, String)>) {
    let doc = g.doc();
    let id = g
        .ports(inst)
        .into_iter()
        .find(|p| doc["nodes"][p]["type"] == ty)
        .unwrap_or_else(|| panic!("no {ty} port on {inst}: {}", doc["nodes"]));
    let inner = g.inner(&id);
    (id, inner)
}

#[test]
fn grouping_mints_a_port_for_every_crossing_cable_and_expanding_gives_them_back() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let sink = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    g.link(buf, "out", sink, "data");

    // The SELECTION is what decides which side of the boundary a cable is on. Grouping the middle
    // node alone leaves both its cables crossing, so grouping has to mint a port for each — the
    // sub-patch would otherwise be born unreachable from the patch it was cut out of.
    let inst = group(&g, &[hex(buf)]);
    let rec = g.doc()["nodes"][&inst].clone();
    // A facade is a node record; a TOP-LEVEL one simply names no scope, exactly as a root node does.
    assert_eq!(rec["type"], "SubPatch", "the facade is a node record: {rec}");
    // Its display name is minted from the ONE namespace, so it counts sub-patches — never uids.
    assert_eq!(rec["name"], "subpatch0", "the first sub-patch in a patch is subpatch0: {rec}");
    assert!(rec.get("scope").is_none(), "a top-level scope names no parent: {rec}");
    assert!(rec.get("def_id").is_none(), "no sharing ⇒ no def_id");

    let ports = g.ports(&inst);
    assert_eq!(ports.len(), 2, "one port per crossing cable, and not one per cable: {ports:?}");
    let (inp, in_inner) = port_of(&g, &inst, "InArray");
    let (outp, out_inner) = port_of(&g, &inst, "OutArray");
    assert_eq!(in_inner, Some((hex(buf), "data".into())), "the incoming cable's port feeds the member slot it crossed at");
    assert_eq!(out_inner, Some((hex(buf), "out".into())), "and the outgoing one drains that one");
    assert_eq!(g.members(&inst), {
        let mut want = vec![hex(buf), inp.clone(), outp.clone()];
        want.sort();
        want
    }, "the member and its two ports are what the scope holds");

    // The crossing cables themselves are UNTOUCHED. A port is a naming indirection, so the runtime
    // link stays flat leaf→leaf and no frame takes a detour through the boundary.
    let links = g.doc()["links"].as_array().cloned().unwrap_or_default();
    let flat = |a: &str, b: &str| links.iter().any(|l| l["node_out"] == a && l["node_in"] == b);
    assert!(flat(&hex(osc), &hex(buf)), "the cable in still names the leaf: {links:?}");
    assert!(flat(&hex(buf), &hex(sink)), "and so does the cable out: {links:?}");
    assert_eq!(links.len(), 4, "…plus the two ports' inner wires, and nothing else: {links:?}");

    // Expanding is the exact inverse: the ports go, the members come back, the cables never moved.
    g.call("expand_instance", j!({ "inst_id": inst }));
    assert!(g.instances().is_empty(), "the instance dropped out of the forest");
    assert_eq!(g.nodes().len(), 3, "and every leaf came back to root");
    let after = g.doc()["links"].as_array().cloned().unwrap_or_default();
    assert_eq!(after.len(), 2, "the minted ports took their inner wires with them: {after:?}");

    // Widen the selection and the cable between the two stops crossing, so nothing is minted for it.
    let both = group(&g, &[hex(osc), hex(buf)]);
    let (drain, _) = port_of(&g, &both, "OutArray");
    assert_eq!(g.ports(&both), vec![drain.clone()], "only buf→sink still crosses");
    assert!(g.members(&both).contains(&hex(osc)), "both leaves are in the scope");
    assert!(g.members(&both).contains(&hex(buf)));

    // Two cables of the SAME direction cross at once. Each port is named from the patch's one
    // display-name namespace, so the second has to see the first — a batch that names every port
    // from the state it started in mints two `out0`s, which `nd()` cannot tell apart.
    let far = g.add("Buffer");
    g.link(osc, "out", far, "data");
    let pair = group(&g, &[hex(buf), hex(osc)]);
    let doc = g.doc();
    let mut minted: Vec<&str> =
        g.ports(&pair).iter().map(|p| doc["nodes"][p]["name"].as_str().unwrap_or("?")).collect();
    minted.sort();
    minted.dedup();
    assert_eq!(minted.len(), g.ports(&pair).len(), "every minted port has its own name: {minted:?}");
    g.call("expand_instance", j!({ "inst_id": pair }));
    g.call("remove_node", j!({ "node": hex(far) }));

    // Group that sub-patch in turn. The cable still crosses, but the scope it crosses out of ALREADY
    // exposes it, so the outer port lands on the inner one's port rather than minting a rival for
    // the same stream — the reuse is what keeps one leaf slot behind exactly one chain of ports.
    let outer = group(&g, std::slice::from_ref(&both));
    let (_, nested_inner) = port_of(&g, &outer, "OutArray");
    assert_eq!(nested_inner, Some((both.clone(), drain)), "the outer port names the inner one");
    assert_eq!(g.ports(&both).len(), 1, "and nothing new was minted inside");

    // A selection that MIXES a sub-patch with plain nodes nests the sub-patch like any other
    // member — a facade is a node, so what it is made of is no business of the grouping.
    let mixed = group(&g, &[outer.clone(), hex(sink)]);
    let members = g.members(&mixed);
    assert!(members.contains(&outer), "the sub-patch went inside: {members:?}");
    assert!(members.contains(&hex(sink)), "…beside the leaf it was selected with: {members:?}");
    assert_eq!(g.doc()["nodes"][&outer]["scope"], mixed, "and it names its new parent");
    // The LOWEST free name, so the two scopes expanded above gave theirs back — a count of what
    // exists, never of what has ever existed, and never a uid.
    assert_eq!(g.doc()["nodes"][&mixed]["name"], "subpatch2");
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
    assert_eq!(g.instances(), vec![inst.clone()], "the instance survives its other member");

    // A port exposing the LAST member outlives it, unwired — a leaf whose upstream is deleted
    // stays too, and a port gets no different treatment.
    let bnd = boundary(&g, &inst, "out");
    wire(&g, &bnd, "out", &hex(buf), "out");
    assert_eq!(g.inner(&bnd), Some((hex(buf), "out".into())), "wired to the member");

    g.call("remove_node", j!({ "node": hex(buf) }));
    assert!(g.ports(&inst).contains(&bnd), "the port stayed: {:?}", g.ports(&inst));
    assert_eq!(g.inner(&bnd), None, "…and went unwired, not away");
    assert_eq!(g.members(&inst), vec![bnd.clone()], "the port is all the scope still holds");

    // Standing means usable: a fresh member takes the port that is already there.
    let member = g.call("add_node", j!({ "type": "Oscillator", "inst_id": inst, "pos": [0.0, 0.0] }))
        ["uid"].as_str().expect("a uid").to_string();
    wire(&g, &bnd, "out", &member, "out");
    assert_eq!(g.inner(&bnd), Some((member, "out".into())), "the standing port re-wired");
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

    // Authoring ALONE is what gives the sub-patch its slot: the record is complete the moment the
    // port exists — the scope it is a port of, and a type naming both the direction and the dtype.
    // That record IS the facade's slot on the parent canvas, so gating it on an inner wire hid an
    // authored port from the scope above until somebody wired it.
    let feed = boundary(&g, &inst, "in");
    let fresh = g.doc()["nodes"][&feed].clone();
    assert_eq!(fresh["scope"], inst, "a port names the sub-patch it is a port of: {fresh}");
    assert_eq!(fresh["type"], "InArray", "…and its type says which way it faces, and at what dtype");
    assert!(g.inner(&feed).is_none(), "…while the inner wire is a separate act it has not had yet");
    assert!(g.members(&inst).contains(&feed), "an unwired port is a member like any other");

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

    // Removing the nested port UNWIRES the outer one and leaves it standing — the state a fresh
    // port is minted in, which is where a leaf lands when the node feeding it is deleted.
    g.call("remove_node", j!({ "node": ib }));
    assert!(g.ports(&outer).contains(&ob), "the outer port stayed: {:?}", g.ports(&outer));
    assert_eq!(g.inner(&ob), None, "…and went unwired rather than naming a slot nothing has");
    let links = g.doc()["links"].as_array().cloned().unwrap_or_default();
    assert!(links.iter().all(|l| l["node_in"] != ob && l["node_out"] != ob),
            "so no cable is drawn at it: {links:?}");

    // An unwired port is re-wirable, exactly as a leaf whose upstream went away is.
    let again = boundary(&g, &inner, "out");
    wire(&g, &again, "out", &hex(buf), "out");
    wire(&g, &ob, "out", &inner, &again);
    assert_eq!(g.inner(&ob), Some((inner.clone(), again)), "the standing port took a new target");

    // …and undo walks all of it back, ending with the outer port naming the port it first exposed.
    for _ in 0..3 {
        g.call("undo", j!({}));
    }
    assert_eq!(g.call("undo", j!({}))["changed"], true);
    assert_eq!(g.inner(&ob), Some((inner, ib)), "the outer port names the inner one again");
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

    // An OUT port DRAINS a member, and what it drains is a stream — so it reads exactly as an IN
    // port does, from the other side. Unwired, it says so in the same words.
    let outp = boundary(&g, &inst, "out");
    g.call("edit_node", j!({ "node": outp, "name": "sink" }));
    let dry = bind("nd('sink')");
    assert!(dry.as_str().is_some_and(|e| e.contains("sink") && e.contains("wired")),
            "an unwired out port says so, like an in port: {dry}");
    wire(&g, &outp, "out", &hex(buf), "out");
    let drained = g.call("inspect_node", j!({ "node": hex(buf) }))["text"].as_str().unwrap().to_string();
    assert!(!drained.contains("[error:"), "wiring the out port made the reference resolve: {drained}");

    // …and the SUB-PATCH itself is referenceable, because it has that port as an output slot. One
    // output, so a bare reference is unambiguous — the same rule a one-output node gets.
    let facade = bind("nd('subpatch0')");
    assert!(facade.is_null(), "a sub-patch with one output resolves bare: {facade}");

    // A second output makes it ambiguous, and the refusal names the way out — again the node rule.
    // It drains a second member, because one leaf slot sits behind exactly one chain of ports.
    let other_member = g.call("add_node", j!({ "type": "Oscillator", "inst_id": inst, "pos": [0.0, 0.0] }))
        ["uid"].as_str().unwrap().to_string();
    let second = boundary(&g, &inst, "out");
    g.call("edit_node", j!({ "node": second, "name": "other" }));
    wire(&g, &second, "out", &other_member, "out");
    let ambiguous = bind("nd('subpatch0')");
    assert!(ambiguous.as_str().is_some_and(|e| e.contains("ambiguous")),
            "two outputs, so name one: {ambiguous}");

    // Naming one is by its PORT NAME: the facade's slots are called what its ports are called, so
    // both resolve and neither needs the uid the document keys them by.
    assert!(bind("nd('subpatch0').sink").is_null(), "the first port names a slot");
    assert!(bind("nd('subpatch0').other").is_null(), "and so does the second");
    let unknown = bind("nd('subpatch0').nosuch");
    assert!(unknown.as_str().is_some_and(|e| e.contains("no output")),
            "a name no port wears is refused as a leaf's would be: {unknown}");

    // A port's label lives in the ONE display-name namespace `nd()` reads, so it cannot shadow a
    // node's — a second `left` would make the reference above ambiguous.
    g.call("edit_node", j!({ "node": hex(osc), "name": "source" }));
    g.refuse("edit_node", j!({ "node": inp, "name": "source" }));
    g.refuse("add_node", j!({ "type": "Buffer", "name": "left" }));

    // Dissolving the sub-patch deletes its ports, and a binding that named one has to go
    // unresolvable HERE. `remove_node` on a port does that; expanding took the same ports out by
    // another door, and a name nothing wears can never be re-resolved by a later edit.
    g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                           "node_in": inst, "slot_in": inp }));
    bind("nd('left')");
    let live = g.call("inspect_node", j!({ "node": hex(buf) }))["text"].as_str().unwrap().to_string();
    assert!(!live.contains("[error:"), "the binding is live going in: {live}");
    g.call("expand_instance", j!({ "inst_id": inst }));
    let gone = g.call("inspect_node", j!({ "node": hex(buf) }))["text"].as_str().unwrap().to_string();
    assert!(gone.contains("[error:"), "the port is gone, so the binding that named it is: {gone}");
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

    // An OUT port RELAYS what it drains, so it carries a stream and takes a viewer on the same
    // `value` slot an IN port wears — a port is a pass-through, never a sink, whichever way it faces.
    let outp = boundary(&g, &inst, "out");
    g.call("edit_node", j!({ "node": outp, "viewers": { "value": { "kind": "line" } } }));
    let drained = g.doc()["nodes"][&outp]["viewers"].as_str().unwrap_or("").to_string();
    assert!(drained.contains("line"), "the out port kept its view state: {drained}");

    // The FACADE is a node too, and it draws that OUT port as one of its output slots — so the
    // viewer that has no meaning INSIDE the sub-patch is exactly the one it wears outside.
    wire(&g, &outp, "out", &hex(buf), "out");
    g.call("edit_node", j!({ "node": inst, "viewers": { &outp: { "kind": "line" } } }));
    let facade = g.doc()["nodes"][&inst]["viewers"].as_str().expect("a blob, as a node's").to_string();
    assert!(facade.contains("line"), "the facade kept the view state: {facade}");
    g.refuse("edit_node", j!({ "node": inst, "viewers": { "nope": { "kind": "line" } } }));

    // A facade NAMES its slots the way a node does: the port's display name, never its uid. The doc
    // keys them by uid so a rename cannot break a wire, but nothing a human or an agent reads
    // should show hex where a leaf shows `out`.
    let read = g.call("inspect_node", j!({ "node": inst }))["text"].as_str().unwrap().to_string();
    let outp_name = g.doc()["nodes"][&outp]["name"].as_str().unwrap_or("?").to_string();
    assert!(read.contains(&outp_name), "the facade lists its slot by name: {read}");
    assert!(!read.contains(&outp), "…and not by uid: {read}");

    // Both blobs survive a save and a load, as a node's does.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ports.gfi");
    g.call("save", j!({ "path": path.to_string_lossy() }));
    let other = Goofi::new();
    other.call("load", j!({ "path": path.to_string_lossy() }));
    assert_eq!(other.doc()["nodes"][&inp]["viewers"], g.doc()["nodes"][&inp]["viewers"]);
    assert_eq!(other.doc()["nodes"][&inst]["viewers"], g.doc()["nodes"][&inst]["viewers"],
               "the facade's too — it is a node record in the archive like any other");
}

#[test]
fn a_sub_patch_is_copied_whole_and_the_copy_owes_the_original_nothing() {
    // Duplicating a sub-patch is duplicating everything it contains, to any depth — members, nested
    // scopes, ports and the wiring among them. `copy_of` is the same op for a leaf, so there is one
    // door rather than a clone path per node kind.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let sink = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    g.link(buf, "out", sink, "data");

    // An inner sub-patch around the Buffer, then an outer one around that: a copy has to recurse.
    let inner = group(&g, &[hex(buf)]);
    let outer = group(&g, std::slice::from_ref(&inner));
    assert_eq!(g.doc()["nodes"][&outer]["name"], "subpatch1", "the second sub-patch is subpatch1");

    let before_nodes = g.nodes().len();
    let copy = g.call("add_node", j!({ "copy_of": outer, "pos": [400.0, 0.0] }))
        ["uid"].as_str().expect("the copy's facade uid").to_string();
    assert_ne!(copy, outer, "a copy is a new node, at a new uid");

    // The whole shape came with it: a nested scope inside, holding a Buffer, behind its own ports.
    let copied_inner = g.members(&copy).into_iter()
        .find(|m| g.instances().contains(m))
        .expect("the nested sub-patch was copied too");
    assert_ne!(copied_inner, inner, "…as a fresh scope, not a second reference to the original");
    let doc = g.doc();
    let leaf = g.members(&copied_inner).into_iter()
        .find(|m| doc["nodes"][m]["type"] == "Buffer")
        .expect("the innermost member came along");
    assert_ne!(leaf, hex(buf), "at its own uid");
    assert!(g.nodes().len() > before_nodes, "the copy added nodes rather than re-pointing at them");

    // Every name is minted fresh out of the one namespace, so nothing collides and `nd()` can tell
    // the copy from the original.
    let names: Vec<&str> = doc["nodes"].as_object().unwrap().values()
        .filter_map(|n| n["name"].as_str()).collect();
    let mut uniq = names.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(uniq.len(), names.len(), "no two nodes wear one name after a copy: {names:?}");

    // Wiring INSIDE the copy is the original's; wiring to the patch AROUND it is not, because a copy
    // is a new node and a new node arrives unconnected.
    let links = doc["links"].as_array().cloned().unwrap_or_default();
    let touches = |uid: &str| links.iter().filter(|l| l["node_out"] == uid || l["node_in"] == uid).count();
    assert_eq!(touches(&copy), 0, "the copied facade is not wired into the patch: {links:?}");
    assert!(touches(&leaf) > 0, "but its innards are wired to each other: {links:?}");

    // It is INDEPENDENT: editing the copy leaves the original alone.
    g.call("edit_node", j!({ "node": leaf, "params": { "common": { "max_frequency": 3.0 } } }));
    let orig = g.call("inspect_node", j!({ "node": hex(buf), "params": true }))["text"]
        .as_str().unwrap().to_string();
    assert!(!orig.contains("max_frequency = 3"), "the original kept its own params: {orig}");

    // …and it is ONE undo step, however deep it went.
    assert_eq!(g.call("undo", j!({}))["changed"], true); // the param edit
    assert_eq!(g.call("undo", j!({}))["changed"], true); // the whole copy
    assert!(!g.instances().contains(&copy), "undo took the copied subtree whole");
    assert_eq!(g.nodes().len(), before_nodes, "leaving exactly what was there before: {:?}", g.nodes());
    let _ = (osc, sink);
}

#[test]
fn frames_cross_a_boundary_and_stop_when_the_cable_is_cut() {
    // The NODE-TO-NODE plane, which no other test reaches across a sub-patch wall: `running.rs`
    // proves the viewer/reducer plane and would stay green while every frame stopped flowing here.
    // A boundary is bookkeeping — grouping, nesting and expanding must not interrupt a stream.
    let g = Goofi::new();
    let src = g.add("_TestCounter");
    let dst = g.add("_TestEcho");
    g.ready(src);
    g.ready(dst);
    let at_dst = g.probe(dst, "out");
    g.link(src, "out", dst, "in");
    g.until("a frame before any sub-patch exists", |_| at_dst.latest());

    // Collapsing the consumer is pure bookkeeping: the wire is the same wire.
    let inst = group(&g, &[hex(dst)]);
    assert!(g.stays(|_| at_dst.latest().is_some()), "grouping did not interrupt the stream");

    // A fresh member behind a fresh IN port: nothing feeds it until BOTH sides are wired, and each
    // half alone must leave it quiet rather than half-connected.
    let mid = g.call("add_node", j!({ "type": "_TestEcho", "inst_id": inst, "pos": [0.0, 0.0] }))
        ["uid"].as_str().unwrap().to_string();
    let mid_uid = goofi_engine::Uid::from_hex(&mid).expect("a uid");
    g.ready(mid_uid);
    let at_mid = g.probe(mid_uid, "out");
    let port = boundary(&g, &inst, "in");
    wire(&g, &port, "in", &mid, "in");
    assert!(g.stays(|_| at_mid.latest().is_none()), "the inside alone feeds it nothing");

    // Wiring the OUTSIDE is what makes frames cross. This is the step a planner that drops a port
    // endpoint fails, and it fails nowhere else.
    g.call("add_link", j!({ "node_out": hex(src), "slot_out": "out",
                            "node_in": inst, "slot_in": port }));
    g.until("a frame to cross the boundary", |_| at_mid.latest());

    // Cutting the outer cable stops it, through the same door.
    g.call("remove_link", j!({ "node_out": hex(src), "slot_out": "out",
                              "node_in": inst, "slot_in": port }));
    let seen = at_mid.count();
    assert!(g.stays(|_| at_mid.count() == seen), "the cut stopped the stream: {seen}");

    // Re-wire, then nest the whole sub-patch one level deeper. The chain of ports lengthens and the
    // frames must still arrive — a relay walk that stops at the first hop fails here.
    g.call("add_link", j!({ "node_out": hex(src), "slot_out": "out",
                            "node_in": inst, "slot_in": port }));
    g.until("frames again after re-wiring", |_| at_mid.latest());
    let outer = group(&g, std::slice::from_ref(&inst));
    assert!(g.stays(|_| at_mid.latest().is_some()), "nesting kept the stream alive");

    // …and expanding splices the chain back to one hop without dropping a frame.
    g.call("expand_instance", j!({ "inst_id": outer }));
    let before = at_mid.count();
    assert!(g.until("frames after the expand", |_| (at_mid.count() > before).then_some(true)),
            "expanding spliced the cable rather than cutting it");
}
