//! One JSON projection of the engine `Graph` — exactly the shape of the control-plane document.

use goofi_engine::subpatch;
use goofi_engine::Graph;
use serde_json::{json, Map, Value};

/// `g`'s whole control-plane state, in the shape a `.gfi` holds it: one node map carrying leaves,
/// sub-patch facades and boundary ports alike, one link list, globals, and the panel arrangement.
/// The sub-patch forest is not a block of its own — a member names its scope, and that is the only
/// place membership lives.
pub fn of(g: &Graph) -> Value {
    let pos_json = |p: [f64; 2]| json!({ "x": p[0], "y": p[1] });

    let mut nodes = Map::new();
    for uid in g.node_uids() {
        let mut node = Map::new();
        node.insert("type".into(), json!(g.type_name(uid).unwrap_or("")));
        node.insert("name".into(), json!(g.name(uid).unwrap_or("")));
        node.insert("pos".into(), pos_json(g.pos(uid).unwrap_or([0.0, 0.0])));
        let mut params = Map::new();
        if let Some(ps) = g.params(uid) {
            for (group, pg) in &*ps {
                let mut gmap = Map::new();
                for (pname, p) in pg {
                    let mut entry = Map::new();
                    entry.insert("value".into(), goofi_engine::param_value_json(p, true));
                    if let Some(e) = g.param_expression(uid, group, pname) {
                        entry.insert(
                            "expr".into(),
                            json!({ "source": e.source, "enabled": e.enabled, "triggers": e.triggers_process }),
                        );
                    }
                    gmap.insert(pname.clone(), Value::Object(entry));
                }
                params.insert(group.clone(), Value::Object(gmap));
            }
        }
        node.insert("params".into(), Value::Object(params));
        // `g.viewers` is Some for EVERY node, so the emptiness gate — not the Option — is what
        // keeps a viewerless node's leaf out of the doc.
        if let Some(v) = g.viewers(uid).filter(|v| v.as_object().is_some_and(|m| !m.is_empty())) {
            node.insert("viewers".into(), json!(v.to_string()));
        }
        nodes.insert(uid.to_hex(), Value::Object(node));
    }
    // A facade and a boundary port are node records too. Neither runs, so neither carries params or
    // viewers — the key is simply absent, as it is on a node that has none.
    for uid in g.scope_uids() {
        let Some(scope) = g.scope(uid) else { continue };
        nodes.insert(
            uid.to_hex(),
            json!({ "type": subpatch::SCOPE_TYPE, "name": scope.name, "pos": pos_json(scope.pos),
                    "params": {} }),
        );
        for (id, st) in scope.stubs.iter() {
            let mut rec = json!({ "type": subpatch::boundary_type_name(st.dir, st.dtype),
                                  "name": st.name, "pos": pos_json(st.pos), "params": {},
                                  "scope": uid.to_hex() });
            // The same `json_string` shape a node's viewers ride in: a merge patch spends `null`
            // on a key delete, so a viewer blob must not reach the document as a tree of leaves.
            if st.viewers.as_object().is_some_and(|m| !m.is_empty()) {
                rec["viewers"] = json!(st.viewers.to_string());
            }
            nodes.insert(id.to_hex(), rec);
        }
    }
    // Membership rides the member. Absent means ROOT — never a null, which a merge patch spends on
    // "delete this key" and could not tell from a move out of a scope.
    for (uid, parent) in g
        .node_uids()
        .into_iter()
        .chain(g.scope_uids())
        .filter_map(|u| g.scope_of(u).map(|p| (u, p)))
    {
        if let Some(Value::Object(rec)) = nodes.get_mut(&uid.to_hex()) {
            rec.insert("scope".into(), json!(parent.to_hex()));
        }
    }

    let mut links: Vec<Value> = g
        .links_view()
        .into_iter()
        .map(|l| {
            json!({
                "node_out": l.node_out.to_hex(), "slot_out": l.slot_out.to_string(),
                "node_in": l.node_in.to_hex(), "slot_in": l.slot_in.to_string(),
            })
        })
        .collect();
    // A port's inner wire is a link, so the cable drawn inside a sub-patch is read the same way as
    // every other cable rather than reconstructed from the port's own record.
    for scope in g.scope_uids().into_iter().filter_map(|u| g.scope(u)) {
        for (id, st) in scope.stubs.iter() {
            let Some((inner, slot)) = &st.inner else { continue };
            let (a, so, b, si) = match st.dir {
                subpatch::Dir::In => (id, subpatch::BOUNDARY_SLOT, inner, slot.as_str()),
                subpatch::Dir::Out => (inner, slot.as_str(), id, subpatch::BOUNDARY_SLOT),
            };
            links.push(json!({ "node_out": a.to_hex(), "slot_out": so,
                               "node_in": b.to_hex(), "slot_in": si }));
        }
    }

    // Known limitation: this Map is a BTreeMap, so a full mirror shows globals alphabetized until
    // the next live edit. A stable doc order needs an ordered globals shape.
    let mut globals = Map::new();
    for (name, value, is_system) in g.globals().entries() {
        let mut entry = goofi_engine::global_to_json(value);
        if let Value::Object(m) = &mut entry {
            m.insert("system".into(), json!(is_system));
        }
        globals.insert(name.to_string(), entry);
    }

    json!({ "nodes": nodes, "links": links,
        "globals": globals, "arrangement": g.arrangement().to_json() })
}
