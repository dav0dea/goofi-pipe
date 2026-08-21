//! One JSON projection of the engine `Graph` — exactly the shape of the control-plane document.

use goofi_engine::subpatch::Dir;
use goofi_engine::Graph;
use serde_json::{json, Map, Value};

use crate::schemas::ROOT_ID;

/// `g`'s whole control-plane state: nodes with their params and viewers, links, the sub-patch
/// forest, globals, and the panel arrangement.
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

    // Membership is a MAP keyed by member uid: the reconciler handles nested maps, not
    // arrays-in-records.
    let mut instances = Map::new();
    for uid in g.scope_uids() {
        let Some(scope) = g.scope(uid) else { continue };
        let mut srec = Map::new();
        srec.insert("name".into(), json!(scope.name));
        let parent = g.scope_of(uid).map(|p| p.to_hex()).unwrap_or_else(|| ROOT_ID.to_string());
        srec.insert("parent".into(), json!(parent));
        srec.insert("pos".into(), pos_json(scope.pos));
        let mut members = Map::new();
        for m in g.scope_members(uid) {
            members.insert(m.to_hex(), json!({ "is_instance": g.scope(m).is_some() }));
        }
        srec.insert("members".into(), Value::Object(members));
        let mut stubs = Map::new();
        for (id, st) in scope.stubs.iter() {
            let mut sm = Map::new();
            sm.insert("dir".into(), json!(match st.dir { Dir::In => "in", Dir::Out => "out" }));
            sm.insert("dtype".into(), json!(st.dtype.name()));
            sm.insert("name".into(), json!(st.name));
            sm.insert("pos".into(), pos_json(st.pos));
            // The stub's DIRECT inner, not the chain-resolved deep leaf: the editor's per-level
            // reroute matches each stub against its direct child.
            if let Some((u, s)) = st.inner.as_ref() {
                sm.insert("inner_node".into(), json!(u.to_hex()));
                sm.insert("inner_slot".into(), json!(s));
            }
            stubs.insert(id.clone(), Value::Object(sm));
        }
        srec.insert("stubs".into(), Value::Object(stubs));
        instances.insert(uid.to_hex(), Value::Object(srec));
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

    json!({ "nodes": nodes, "links": links, "instances": instances,
        "globals": globals, "arrangement": g.arrangement().to_json() })
}
