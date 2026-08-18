//! Keep a `GraphDoc` in agreement with the engine `Graph`. The mirror is a full re-sync after
//! each mutating RPC: `reconcile_root` diffs the whole projection and writes only what differs,
//! so the doc's own delta is what reaches the wire. The client replica never writes the doc.

use crate::crdt::GraphDoc;
use goofi_engine::subpatch::Dir;
use goofi_engine::Graph;
use serde_json::{json, Map, Value};

use crate::schemas::ROOT_ID;

/// Rebuild `doc` to mirror `g`'s control-plane state (nodes, params, pos, viewers, links, forest).
/// Builds ONE JSON projection of the engine graph — exactly the doc's field shape — and hands it to
/// the generic, idempotent, in-place [`GraphDoc::reconcile_root`]. The manager stays the sole author
/// of every structural field (§4.2); merge-safe leaves a client writes directly are preserved by the
/// reconciler's recurse-in-place discipline.
pub fn sync_graph_to_doc(g: &Graph, doc: &mut GraphDoc) {
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
                    // A cleared binding omits `expr` entirely → the reconciler prunes any stale one.
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
        // Viewers ride as an opaque JSON string leaf (typed view-state is a later step). `g.viewers`
        // returns Some for EVERY node, so the emptiness gate (matching serialize) — not the Option —
        // is what keeps a viewerless node's leaf out of the doc; the reconciler prunes any stale one.
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

    // The sub-patch scopes (a flat organizational overlay) — the manager is the sole author of
    // structural fields (§4.2). Kept under the doc's `instances` key; each record is a scope:
    // {name, parent, pos, members:{uid:{is_instance}}, stubs:{id:{dir,dtype,name,pos,inner_node?,
    // inner_slot?}}}. Membership is a MAP keyed by member uid (the CRDT reconciler handles nested
    // maps, not arrays-in-records). A stub's parent side is not stored — the frontend derives
    // facade-port edges from the flat links + each stub's resolved inner leaf.
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
            // The stub's DIRECT inner (a member uid + its real slot, or a nested scope facade uid +
            // that scope's StubId) — NOT the chain-resolved deep leaf. The editor's per-level
            // `drawEndpoint` reroute matches each stub against its direct child, so it needs the
            // direct reference; the data plane / link authoring chain-resolve server-side.
            // Unwired stub → no inner_node/inner_slot → the reconciler prunes any stale pair.
            if let Some((u, s)) = st.inner.as_ref() {
                sm.insert("inner_node".into(), json!(u.to_hex()));
                sm.insert("inner_slot".into(), json!(s));
            }
            stubs.insert(id.clone(), Value::Object(sm));
        }
        srec.insert("stubs".into(), Value::Object(stubs));
        instances.insert(uid.to_hex(), Value::Object(srec));
    }

    // Globals (system + user) — `{name: {value, type, system}}`. `global_to_json` gives `{value,
    // type}` (the type tag preserves float↔int); the `system` flag lets the panel disable
    // delete/rename. Reconciled like any other root map (idempotent, in-place, prunes user deletes).
    // KNOWN LIMITATION (low): this keyed serde_json::Map is a BTreeMap, so a FULL mirror (startup /
    // load) inserts globals into the doc alphabetically — a loaded patch shows them alphabetized
    // until the next live edit re-appends in order. The `.gfi` persists the true order (an ordered
    // array); giving the DOC a stable order needs an ordered globals shape, deferred.
    let mut globals = Map::new();
    for (name, value, is_system) in g.globals().entries() {
        let mut entry = goofi_engine::global_to_json(value);
        if let Value::Object(m) = &mut entry {
            m.insert("system".into(), json!(is_system));
        }
        globals.insert(name.to_string(), entry);
    }

    // The arrangement projects through the very `to_json` the `.gfi` uses — ONE shape for the
    // persisted patch and the live replica, so a panel cannot mean two things.
    doc.reconcile_root(&json!({ "nodes": nodes, "links": links, "instances": instances,
        "globals": globals, "arrangement": g.arrangement().to_json() }));
}
