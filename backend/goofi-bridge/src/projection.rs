//! One JSON projection of the engine `Graph` — exactly the shape of the control-plane document.

use goofi_graph::Graph;
use serde_json::{json, Map, Value};

/// `g`'s whole control-plane state, in the shape a `.gfi` holds it: one node map carrying leaves,
/// sub-patch facades and boundary ports alike, one link list, globals, and the panel arrangement.
/// The sub-patch forest is not a block of its own — a member names its scope, and that is the only
/// place membership lives.
pub fn of(g: &Graph) -> Value {
    let pos_json = |p: [f64; 2]| json!({ "x": p[0], "y": p[1] });

    let mut nodes = Map::new();
    // ONE loop over ONE namespace: a leaf, a facade and a boundary port are all node records, and
    // what differs between them is only what they HAVE — a facade and a port run nothing, so the
    // params key is simply empty, as it is on a node with none.
    for uid in g.all_uids() {
        let mut node = Map::new();
        node.insert("type".into(), json!(g.node_type(uid).unwrap_or("")));
        node.insert("name".into(), json!(g.name(uid).unwrap_or("")));
        node.insert("pos".into(), pos_json(g.pos(uid).unwrap_or([0.0, 0.0])));
        let mut params = Map::new();
        if let Some(ps) = g.params(uid) {
            for (group, pg) in &*ps {
                let mut gmap = Map::new();
                for (pname, p) in pg {
                    let mut entry = Map::new();
                    entry.insert("value".into(), goofi_graph::param_value_json(p));
                    if let Some(s) = g.param_source(uid, group, pname) {
                        entry.insert("mode".into(), json!(s.state.mode));
                        if !s.state.expression.is_empty() {
                            entry.insert("expr".into(), json!(s.state.expression));
                        }
                        if !s.state.reference.is_empty() {
                            entry.insert("ref".into(), json!(s.state.reference));
                        }
                        if s.state.triggers {
                            entry.insert("triggers".into(), json!(true));
                        }
                    }
                    gmap.insert(pname.clone(), Value::Object(entry));
                }
                params.insert(group.clone(), Value::Object(gmap));
            }
        }
        node.insert("params".into(), Value::Object(params));
        // The same `json_string` shape every kind's viewers ride in: a merge patch spends `null`
        // on a key delete, so a viewer blob must not reach the document as a tree of leaves.
        if let Some(v) = g.viewers(uid).filter(|v| v.as_object().is_some_and(|m| !m.is_empty())) {
            node.insert("viewers".into(), json!(v.to_string()));
        }
        nodes.insert(uid.to_hex(), Value::Object(node));
    }
    // Membership rides the member. Absent means ROOT — never a null, which a merge patch spends on
    // "delete this key" and could not tell from a move out of a scope.
    for (uid, parent) in g.all_uids().into_iter().filter_map(|u| g.scope_of(u).map(|p| (u, p))) {
        if let Some(Value::Object(rec)) = nodes.get_mut(&uid.to_hex()) {
            rec.insert("scope".into(), json!(parent.to_hex()));
        }
    }

    let links: Vec<Value> = g
        .links_view()
        .into_iter()
        .map(|l| {
            json!({
                "node_out": l.node_out.to_hex(), "slot_out": l.slot_out.to_string(),
                "node_in": l.node_in.to_hex(), "slot_in": l.slot_in.to_string(),
            })
        })
        .collect();

    // Known limitation: this Map is a BTreeMap, so a full mirror shows globals alphabetized until
    // the next live edit. A stable doc order needs an ordered globals shape.
    let mut globals = Map::new();
    for (name, value, is_system, locked) in g.globals().entries() {
        let mut entry = goofi_graph::global_to_json(value);
        if let Value::Object(m) = &mut entry {
            m.insert("system".into(), json!(is_system));
            m.insert("locked".into(), json!(locked));
        }
        globals.insert(name.to_string(), entry);
    }

    json!({ "nodes": nodes, "links": links,
        "globals": globals, "arrangement": g.arrangement().to_json() })
}
