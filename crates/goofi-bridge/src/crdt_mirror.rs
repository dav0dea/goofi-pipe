//! Keep a `GraphDoc` in agreement with the engine `Graph`. Phase 1 is a full re-sync after
//! each mutating RPC — correctness first; incremental/direct writes come in later phases.

use std::collections::HashSet;

use goofi_crdt::{ExprRecord, GraphDoc, LinkRecord};
use goofi_engine::Graph;

use crate::schemas::param_value_json;

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
}
