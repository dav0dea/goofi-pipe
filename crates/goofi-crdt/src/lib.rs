//! `GraphDoc` — a typed façade over a `yrs::Doc` holding goofi's control-plane state
//! (nodes + nested params/viewers, links). The manager keeps this in agreement with the
//! engine `Graph`; it is the sync structure clients will later replicate. Pure: depends
//! only on `yrs` + `serde_json`, no engine/payload types.

use yrs::{Any, Array, ArrayRef, Doc, Map, MapPrelim, MapRef, Out, Transact};

/// An expression binding as mirrored into the doc.
#[derive(Clone, Debug, PartialEq)]
pub struct ExprRecord {
    pub source: String,
    pub enabled: bool,
    pub triggers: bool,
}

/// A link as mirrored into the doc (hex uids + slot names).
#[derive(Clone, Debug, PartialEq)]
pub struct LinkRecord {
    pub node_out: String,
    pub slot_out: String,
    pub node_in: String,
    pub slot_in: String,
}

/// Insert a scalar json value (number/bool/string; anything else → Null) into a yrs map.
fn insert_scalar(map: &MapRef, txn: &mut yrs::TransactionMut, key: &str, v: &serde_json::Value) {
    match v {
        serde_json::Value::Number(n) => {
            map.insert(txn, key, n.as_f64().unwrap_or(0.0));
        }
        serde_json::Value::Bool(b) => {
            map.insert(txn, key, *b);
        }
        serde_json::Value::String(s) => {
            map.insert(txn, key, s.as_str());
        }
        _ => {
            map.insert(txn, key, Any::Null);
        }
    }
}

/// Get an existing nested map by key, or insert a fresh one.
fn get_or_insert_map(parent: &MapRef, txn: &mut yrs::TransactionMut, key: &str) -> MapRef {
    match parent.get(txn, key).and_then(|v| v.cast::<MapRef>().ok()) {
        Some(m) => m,
        None => parent.insert(txn, key, MapPrelim::default()),
    }
}

/// The control-plane document. `nodes` is a Map<uid, {type, name, pos, params, viewers}>,
/// `links` an Array of {node_out, slot_out, node_in, slot_in} (added in a later task).
pub struct GraphDoc {
    doc: Doc,
    nodes: MapRef,
    links: ArrayRef,
}

impl GraphDoc {
    pub fn new() -> GraphDoc {
        let doc = Doc::new();
        let nodes = doc.get_or_insert_map("nodes");
        let links = doc.get_or_insert_array("links");
        GraphDoc { doc, nodes, links }
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

    /// Set (or replace) a param's `{value, expr?}` under `nodes[uid].params[group][name]`.
    pub fn set_param(
        &mut self,
        uid: &str,
        group: &str,
        name: &str,
        value: &serde_json::Value,
        expr: Option<ExprRecord>,
    ) {
        let mut txn = self.doc.transact_mut();
        let Some(node) = self.nodes.get(&txn, uid).and_then(|v| v.cast::<MapRef>().ok()) else {
            return;
        };
        let params = get_or_insert_map(&node, &mut txn, "params");
        let g = get_or_insert_map(&params, &mut txn, group);
        let entry: MapRef = g.insert(&mut txn, name, MapPrelim::default()); // replace whole entry
        insert_scalar(&entry, &mut txn, "value", value);
        if let Some(e) = expr {
            let ex: MapRef = entry.insert(&mut txn, "expr", MapPrelim::default());
            ex.insert(&mut txn, "source", e.source.as_str());
            ex.insert(&mut txn, "enabled", e.enabled);
            ex.insert(&mut txn, "triggers", e.triggers);
        }
    }

    fn param_entry(
        &self,
        txn: &yrs::Transaction,
        uid: &str,
        group: &str,
        name: &str,
    ) -> Option<MapRef> {
        self.node_map(txn, uid)?
            .get(txn, "params")
            .and_then(|v| v.cast::<MapRef>().ok())?
            .get(txn, group)
            .and_then(|v| v.cast::<MapRef>().ok())?
            .get(txn, name)
            .and_then(|v| v.cast::<MapRef>().ok())
    }

    pub fn param_value(&self, uid: &str, group: &str, name: &str) -> Option<serde_json::Value> {
        let txn = self.doc.transact();
        match self.param_entry(&txn, uid, group, name)?.get(&txn, "value") {
            Some(Out::Any(Any::Number(n))) => Some(serde_json::json!(n)),
            Some(Out::Any(Any::Bool(b))) => Some(serde_json::json!(b)),
            Some(Out::Any(Any::String(s))) => Some(serde_json::json!(s.to_string())),
            _ => None,
        }
    }

    pub fn param_expr_source(&self, uid: &str, group: &str, name: &str) -> Option<String> {
        let txn = self.doc.transact();
        let ex = self
            .param_entry(&txn, uid, group, name)?
            .get(&txn, "expr")
            .and_then(|v| v.cast::<MapRef>().ok())?;
        match ex.get(&txn, "source") {
            Some(Out::Any(Any::String(s))) => Some(s.to_string()),
            _ => None,
        }
    }

    /// Store a node's opaque per-slot viewer blob as a JSON string (typed in a later phase).
    pub fn set_viewers(&mut self, uid: &str, viewers: &serde_json::Value) {
        let mut txn = self.doc.transact_mut();
        if let Some(node) = self.nodes.get(&txn, uid).and_then(|v| v.cast::<MapRef>().ok()) {
            node.insert(&mut txn, "viewers", viewers.to_string());
        }
    }

    pub fn viewers_json(&self, uid: &str) -> Option<serde_json::Value> {
        let txn = self.doc.transact();
        match self.node_map(&txn, uid)?.get(&txn, "viewers") {
            Some(Out::Any(Any::String(s))) => serde_json::from_str(&s).ok(),
            _ => None,
        }
    }

    /// Replace the whole link set (Phase 1 mirror is wholesale; incremental comes later).
    pub fn replace_links(&mut self, links: Vec<LinkRecord>) {
        let mut txn = self.doc.transact_mut();
        let len = self.links.len(&txn);
        self.links.remove_range(&mut txn, 0, len);
        for l in links {
            let m: MapRef = self.links.push_back(&mut txn, MapPrelim::default());
            m.insert(&mut txn, "node_out", l.node_out.as_str());
            m.insert(&mut txn, "slot_out", l.slot_out.as_str());
            m.insert(&mut txn, "node_in", l.node_in.as_str());
            m.insert(&mut txn, "slot_in", l.slot_in.as_str());
        }
    }

    pub fn links(&self) -> Vec<LinkRecord> {
        let txn = self.doc.transact();
        let s = |m: &MapRef, k| match m.get(&txn, k) {
            Some(Out::Any(Any::String(v))) => v.to_string(),
            _ => String::new(),
        };
        self.links
            .iter(&txn)
            .filter_map(|v| v.cast::<MapRef>().ok())
            .map(|m| LinkRecord {
                node_out: s(&m, "node_out"),
                slot_out: s(&m, "slot_out"),
                node_in: s(&m, "node_in"),
                slot_in: s(&m, "slot_in"),
            })
            .collect()
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

    #[test]
    fn set_param_writes_value_and_expression() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);
        doc.set_param("1", "common", "max_frequency", &json!(30.0), None);
        doc.set_param(
            "1", "oscillator", "waveform", &json!("sine"),
            Some(ExprRecord { source: "nd('lfo')".into(), enabled: true, triggers: false }),
        );
        assert_eq!(doc.param_value("1", "common", "max_frequency"), Some(json!(30.0)));
        assert_eq!(doc.param_value("1", "oscillator", "waveform"), Some(json!("sine")));
        assert_eq!(doc.param_expr_source("1", "oscillator", "waveform").as_deref(), Some("nd('lfo')"));
        assert_eq!(doc.param_expr_source("1", "common", "max_frequency"), None);
    }

    #[test]
    fn viewers_blob_and_links_round_trip() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);
        doc.set_viewers("1", &json!({"out": {"kind": "line"}}));
        assert_eq!(doc.viewers_json("1"), Some(json!({"out": {"kind": "line"}})));

        doc.replace_links(vec![LinkRecord {
            node_out: "1".into(),
            slot_out: "out".into(),
            node_in: "2".into(),
            slot_in: "data".into(),
        }]);
        assert_eq!(doc.links().len(), 1);
        assert_eq!(doc.links()[0].slot_in, "data");
        doc.replace_links(vec![]);
        assert!(doc.links().is_empty());
    }
}
