//! Keep a `GraphDoc` in agreement with the engine `Graph`. Phase 1 is a full re-sync after
//! each mutating RPC — correctness first; incremental/direct writes come in later phases.

use goofi_crdt::GraphDoc;
use goofi_engine::subpatch::Dir;
use goofi_engine::Graph;
use serde_json::{json, Map, Value};

use crate::schemas::{param_value_json, ROOT_ID};

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
            for (group, pg) in ps {
                let mut gmap = Map::new();
                for (pname, p) in pg {
                    let mut entry = Map::new();
                    entry.insert("value".into(), param_value_json(p));
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
        // Viewers ride as an opaque JSON string leaf (typed view-state is a later step). Omitted
        // when the node has none → pruned from the doc.
        if let Some(v) = g.viewers(uid) {
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

    // The sub-patch forest — the manager is the sole author of structural fields (§4.2).
    let mut instances = Map::new();
    for uid in g.instance_uids() {
        let Some(inst) = g.instance(uid) else { continue };
        let mut irec = Map::new();
        irec.insert("name".into(), json!(inst.name));
        // Shared iff >1 instance references the def (matches `describe_instance`); unique omits def_id.
        if g.def_refcount(inst.def_id) > 1 {
            irec.insert("def_id".into(), json!(inst.def_id.to_hex()));
        }
        let parent = g.scope_of(uid).map(|p| p.to_hex()).unwrap_or_else(|| ROOT_ID.to_string());
        irec.insert("parent".into(), json!(parent));
        irec.insert("pos".into(), pos_json(inst.pos));
        let mut members = Map::new();
        for (local, muid) in inst.members.iter() {
            members.insert(local.clone(), json!(muid.to_hex()));
        }
        irec.insert("members".into(), Value::Object(members));
        let mut iface = Map::new();
        if let Some(def) = g.def(inst.def_id) {
            for (bnd, b) in def.interface.iter() {
                let resolved = g.resolve_boundary(uid, bnd);
                let mut bm = Map::new();
                bm.insert("dir".into(), json!(match b.dir { Dir::In => "in", Dir::Out => "out" }));
                bm.insert("dtype".into(), json!(b.dtype.name()));
                bm.insert("name".into(), json!(b.name));
                bm.insert("pos".into(), pos_json(b.pos));
                // Unwired boundary → no inner_node/inner_slot → the reconciler prunes any stale pair.
                if let Some((u, s)) = resolved.as_ref() {
                    bm.insert("inner_node".into(), json!(u.to_hex()));
                    bm.insert("inner_slot".into(), json!(s));
                }
                iface.insert(bnd.clone(), Value::Object(bm));
            }
        }
        irec.insert("interface".into(), Value::Object(iface));
        instances.insert(uid.to_hex(), Value::Object(irec));
    }

    // Globals (system + user) — `{name: {value, type, system}}`. `global_to_json` gives `{value,
    // type}` (the type tag preserves float↔int); the `system` flag lets the panel disable
    // delete/rename. Reconciled like any other root map (idempotent, in-place, prunes user deletes).
    let mut globals = Map::new();
    for (name, value, is_system) in g.globals().entries() {
        let mut entry = goofi_engine::global_to_json(value);
        if let Value::Object(m) = &mut entry {
            m.insert("system".into(), json!(is_system));
        }
        globals.insert(name.to_string(), entry);
    }

    doc.reconcile_root(&json!({ "nodes": nodes, "links": links, "instances": instances, "globals": globals }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::Param;
    use goofi_engine::Graph;

    #[test]
    fn mirror_reflects_nodes_params_and_links() {
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        g.rename_node(a, "osc").unwrap();
        g.update_param(a, "common", "max_frequency", Param::float(30.0, 0.0, 100.0)).unwrap();
        g.add_link(a, "out", b, "data").unwrap();

        let mut doc = GraphDoc::new();
        sync_graph_to_doc(&g, &mut doc);

        // Read leaves through the generic reader (the typed getters were removed); whole-number
        // params come back as integers from `to_json`, so the value compares via `as_f64`.
        let ah = a.to_hex();
        let bh = b.to_hex();
        assert_eq!(doc.node_ids().len(), 2);
        assert_eq!(doc.read_at(&["nodes", ah.as_str(), "name"]).as_ref().and_then(|v| v.as_str()), Some("osc"));
        assert_eq!(
            doc.read_at(&["nodes", ah.as_str(), "type"]).as_ref().and_then(|v| v.as_str()),
            Some("Oscillator")
        );
        assert_eq!(
            doc.read_at(&["nodes", ah.as_str(), "params", "common", "max_frequency", "value"])
                .and_then(|v| v.as_f64()),
            Some(30.0)
        );
        let j = doc.to_json();
        assert_eq!(j["links"].as_array().unwrap().len(), 1);
        assert_eq!(j["links"][0]["node_in"].as_str(), Some(bh.as_str()));
    }

    #[test]
    fn mirror_reflects_globals_with_system_flag() {
        use goofi_core::globals::GlobalValue;
        let mut g = Graph::new();
        g.apply_global_change("default_ufreq", Some(GlobalValue::Float(45.0))).unwrap();
        g.apply_global_change("subject", Some(GlobalValue::Str("P07".into()))).unwrap();

        let mut doc = GraphDoc::new();
        sync_graph_to_doc(&g, &mut doc);

        // System global carries system:true + its edited value; type tag present.
        assert_eq!(doc.read_at(&["globals", "default_ufreq", "system"]), Some(json!(true)));
        assert_eq!(doc.read_at(&["globals", "default_ufreq", "value"]).and_then(|v| v.as_f64()), Some(45.0));
        assert_eq!(doc.read_at(&["globals", "default_ufreq", "type"]).as_ref().and_then(|v| v.as_str()), Some("float"));
        // User global carries system:false.
        assert_eq!(doc.read_at(&["globals", "subject", "system"]), Some(json!(false)));
        assert_eq!(doc.read_at(&["globals", "subject", "value"]).as_ref().and_then(|v| v.as_str()), Some("P07"));
    }

    #[test]
    fn mirror_reflects_the_sub_patch_forest() {
        // Grouping two linked nodes surfaces one instance (with an auto boundary on the cut
        // link). The mirror must carry it: identity, ROOT parent, both members, and the boundary
        // resolved to its inner leaf — the doc's forest coverage (§4.2).
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();
        let sink = g.add_node("Buffer", None).unwrap();
        g.add_link(osc, "out", buf, "data").unwrap();
        // buf.out → sink makes buf's output a CUT link when buf is grouped → an output boundary.
        g.add_link(buf, "out", sink, "data").unwrap();
        let inst = g.group_nodes(&[buf], [5.0, 6.0]).unwrap();

        let mut doc = GraphDoc::new();
        sync_graph_to_doc(&g, &mut doc);

        assert_eq!(doc.instance_ids(), vec![inst.to_hex()]);
        // Read the mirrored instance object through the generic reader.
        let ih = inst.to_hex();
        let bh = buf.to_hex();
        let j = doc.to_json();
        let rec = &j["instances"][ih.as_str()];
        assert_eq!(rec["parent"].as_str(), Some(ROOT_ID), "top-level instance parents to ROOT");
        assert_eq!((rec["pos"]["x"].as_f64(), rec["pos"]["y"].as_f64()), (Some(5.0), Some(6.0)));
        assert!(rec.get("def_id").is_none(), "single reference ⇒ unique");
        assert!(
            rec["members"].as_object().unwrap().values().any(|v| v.as_str() == Some(bh.as_str())),
            "buf is a member"
        );
        // The output boundary resolves to the inner buffer leaf.
        let out = rec["interface"]
            .as_object()
            .unwrap()
            .values()
            .find(|b| b["dir"].as_str() == Some("out"))
            .expect("output boundary");
        assert_eq!(out["inner_node"].as_str(), Some(bh.as_str()));
        assert_eq!(out["inner_slot"].as_str(), Some("out"));

        // Expanding the instance removes it from the doc forest.
        g.expand_instance(inst).unwrap();
        sync_graph_to_doc(&g, &mut doc);
        assert!(doc.instance_ids().is_empty(), "expanded instance dropped from the forest");
    }
}
