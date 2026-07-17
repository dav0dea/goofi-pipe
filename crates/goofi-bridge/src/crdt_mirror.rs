//! Keep a `GraphDoc` in agreement with the engine `Graph`. Phase 1 is a full re-sync after
//! each mutating RPC — correctness first; incremental/direct writes come in later phases.

use std::collections::HashSet;

use goofi_crdt::{BoundaryRecord, ExprRecord, GraphDoc, InstanceRecord, LinkRecord};
use goofi_engine::subpatch::Dir;
use goofi_engine::Graph;

use crate::schemas::{param_value_json, ROOT_ID};

/// Rebuild `doc` to mirror `g`'s control-plane state (nodes, params, pos, viewers, links).
pub fn sync_graph_to_doc(g: &Graph, doc: &mut GraphDoc) {
    let live: HashSet<String> = g.node_uids().iter().map(|u| u.to_hex()).collect();
    // Drop nodes no longer in the graph.
    for id in doc.node_ids() {
        if !live.contains(&id) {
            doc.remove_node(&id);
        }
    }
    for uid in g.node_uids() {
        let id = uid.to_hex();
        let ty = g.type_name(uid).unwrap_or("");
        let name = g.name(uid).unwrap_or("");
        let pos = g.pos(uid).unwrap_or([0.0, 0.0]);
        doc.upsert_node(&id, ty, name, pos);
        if let Some(params) = g.params(uid) {
            for (group, pg) in params {
                for (pname, p) in pg {
                    let value = param_value_json(p);
                    let expr = g.param_expression(uid, group, pname).map(|e| ExprRecord {
                        source: e.source,
                        enabled: e.enabled,
                        triggers: e.triggers_process,
                    });
                    doc.set_param(&id, group, pname, &value, expr);
                }
            }
        }
        if let Some(v) = g.viewers(uid) {
            doc.set_viewers(&id, v);
        }
    }
    doc.replace_links(
        g.links_view()
            .into_iter()
            .map(|l| LinkRecord {
                node_out: l.node_out.to_hex(),
                slot_out: l.slot_out.to_string(),
                node_in: l.node_in.to_hex(),
                slot_in: l.slot_in.to_string(),
            })
            .collect(),
    );

    // Mirror the sub-patch forest (§4.2 — the manager is the sole author of structural fields).
    let live_inst: HashSet<String> = g.instance_uids().iter().map(|u| u.to_hex()).collect();
    for id in doc.instance_ids() {
        if !live_inst.contains(&id) {
            doc.remove_instance(&id);
        }
    }
    for uid in g.instance_uids() {
        let Some(inst) = g.instance(uid) else { continue };
        // Shared iff more than one instance references the def (matches `describe_instance`).
        let shared = g.def_refcount(inst.def_id) > 1;
        let parent = g
            .scope_of(uid)
            .map(|p| p.to_hex())
            .unwrap_or_else(|| ROOT_ID.to_string());
        let members = inst
            .members
            .iter()
            .map(|(local, muid)| (local.clone(), muid.to_hex()))
            .collect();
        let mut interface = Vec::new();
        if let Some(def) = g.def(inst.def_id) {
            for (bnd, b) in def.interface.iter() {
                let resolved = g.resolve_boundary(uid, bnd);
                interface.push(BoundaryRecord {
                    bnd_id: bnd.clone(),
                    dir: match b.dir {
                        Dir::In => "in",
                        Dir::Out => "out",
                    }
                    .to_string(),
                    dtype: b.dtype.name().to_string(),
                    name: b.name.clone(),
                    pos: b.pos,
                    inner_node: resolved.as_ref().map(|(u, _)| u.to_hex()),
                    inner_slot: resolved.as_ref().map(|(_, s)| s.clone()),
                });
            }
        }
        doc.upsert_instance(
            &uid.to_hex(),
            &InstanceRecord {
                name: inst.name.clone(),
                def_id: if shared { Some(inst.def_id.to_hex()) } else { None },
                parent,
                pos: inst.pos,
                members,
                interface,
            },
        );
    }
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

        assert_eq!(doc.node_ids().len(), 2);
        assert_eq!(doc.node_name(&a.to_hex()).as_deref(), Some("osc"));
        assert_eq!(doc.node_type(&a.to_hex()).as_deref(), Some("Oscillator"));
        assert_eq!(
            doc.param_value(&a.to_hex(), "common", "max_frequency"),
            Some(serde_json::json!(30.0))
        );
        assert_eq!(doc.links().len(), 1);
        assert_eq!(doc.links()[0].node_in, b.to_hex());
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
        let rec = doc.instance_record(&inst.to_hex()).expect("instance mirrored");
        assert_eq!(rec.parent, ROOT_ID, "top-level instance parents to ROOT");
        assert_eq!(rec.pos, [5.0, 6.0]);
        assert_eq!(rec.def_id, None, "single reference ⇒ unique");
        assert!(rec.members.iter().any(|(_, u)| *u == buf.to_hex()), "buf is a member");
        // The output boundary resolves to the inner buffer leaf.
        let out = rec.interface.iter().find(|b| b.dir == "out").expect("output boundary");
        assert_eq!(out.inner_node.as_deref(), Some(buf.to_hex()).as_deref());
        assert_eq!(out.inner_slot.as_deref(), Some("out"));

        // Expanding the instance removes it from the doc forest.
        g.expand_instance(inst).unwrap();
        sync_graph_to_doc(&g, &mut doc);
        assert!(doc.instance_ids().is_empty(), "expanded instance dropped from the forest");
    }
}
