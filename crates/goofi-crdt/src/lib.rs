//! `GraphDoc` — a typed façade over a `yrs::Doc` holding goofi's control-plane state
//! (nodes + nested params/viewers, links). The manager keeps this in agreement with the
//! engine `Graph`; it is the sync structure clients will later replicate. Pure: depends
//! only on `yrs` + `serde_json`, no engine/payload types.

use yrs::{Any, Doc, Map, MapPrelim, MapRef, Out, Transact};

/// The control-plane document. `nodes` is a Map<uid, {type, name, pos, params, viewers}>,
/// `links` an Array of {node_out, slot_out, node_in, slot_in} (added in a later task).
pub struct GraphDoc {
    doc: Doc,
    nodes: MapRef,
}

impl GraphDoc {
    pub fn new() -> GraphDoc {
        let doc = Doc::new();
        let nodes = doc.get_or_insert_map("nodes");
        GraphDoc { doc, nodes }
    }

    /// The uids of all nodes currently in the doc.
    pub fn node_ids(&self) -> Vec<String> {
        let txn = self.doc.transact();
        self.nodes.keys(&txn).map(|k| k.to_string()).collect()
    }

    /// Insert or update a node's identity fields. Creates the node map on first call;
    /// on later calls updates the scalar fields in place (keeps nested params/viewers).
    pub fn upsert_node(&mut self, uid: &str, ty: &str, name: &str, pos: [f64; 2]) {
        let mut txn = self.doc.transact_mut();
        let node = match self.nodes.get(&txn, uid).and_then(|v| v.cast::<MapRef>().ok()) {
            Some(n) => n,
            None => self.nodes.insert(&mut txn, uid, MapPrelim::default()),
        };
        node.insert(&mut txn, "type", ty);
        node.insert(&mut txn, "name", name);
        let posv: MapRef = node.insert(&mut txn, "pos", MapPrelim::default());
        posv.insert(&mut txn, "x", pos[0]);
        posv.insert(&mut txn, "y", pos[1]);
    }

    fn node_map(&self, txn: &yrs::Transaction, uid: &str) -> Option<MapRef> {
        self.nodes.get(txn, uid).and_then(|v| v.cast::<MapRef>().ok())
    }

    pub fn node_name(&self, uid: &str) -> Option<String> {
        let txn = self.doc.transact();
        match self.node_map(&txn, uid)?.get(&txn, "name") {
            Some(Out::Any(Any::String(s))) => Some(s.to_string()),
            _ => None,
        }
    }

    pub fn node_type(&self, uid: &str) -> Option<String> {
        let txn = self.doc.transact();
        match self.node_map(&txn, uid)?.get(&txn, "type") {
            Some(Out::Any(Any::String(s))) => Some(s.to_string()),
            _ => None,
        }
    }

    pub fn node_pos(&self, uid: &str) -> Option<[f64; 2]> {
        let txn = self.doc.transact();
        let p = self.node_map(&txn, uid)?.get(&txn, "pos").and_then(|v| v.cast::<MapRef>().ok())?;
        let f = |k| match p.get(&txn, k) {
            Some(Out::Any(Any::Number(n))) => Some(n),
            _ => None,
        };
        Some([f("x")?, f("y")?])
    }
}

impl Default for GraphDoc {
    fn default() -> GraphDoc {
        GraphDoc::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_doc_has_no_nodes() {
        let doc = GraphDoc::new();
        assert!(doc.node_ids().is_empty());
    }

    #[test]
    fn upsert_node_writes_and_reads_back() {
        let mut doc = GraphDoc::new();
        doc.upsert_node("000000000001", "Oscillator", "osc0", [10.0, 20.0]);
        assert_eq!(doc.node_ids(), vec!["000000000001"]);
        assert_eq!(doc.node_name("000000000001").as_deref(), Some("osc0"));
        assert_eq!(doc.node_type("000000000001").as_deref(), Some("Oscillator"));
        assert_eq!(doc.node_pos("000000000001"), Some([10.0, 20.0]));
        doc.upsert_node("000000000001", "Oscillator", "osc-renamed", [10.0, 20.0]);
        assert_eq!(doc.node_ids().len(), 1);
        assert_eq!(doc.node_name("000000000001").as_deref(), Some("osc-renamed"));
    }
}
