//! `GraphDoc` — a `yrs::Doc` holding goofi's control-plane state, reconciled from and read back
//! as plain JSON. The crate is deliberately SHAPE-AGNOSTIC: [`reconcile_map`] recurses over
//! arbitrary JSON, so what the doc's roots actually contain is owned in exactly one place —
//! `goofi-bridge`'s `crdt_mirror`, which builds the projection. Pure: depends only on `yrs` +
//! `serde_json`, no engine/payload types.

use yrs::updates::decoder::Decode;
use yrs::types::ToJson;
use yrs::{Any, Array, ArrayRef, Doc, Map, MapPrelim, MapRef, Out, ReadTxn, Transact};

/// A framed message on the `/control` binary channel, one leading tag byte: the minimal
/// equivalent of the Yjs sync protocol, both ends driving their doc by hand (no `y-protocols`
/// dependency).
#[derive(Clone, Debug, PartialEq)]
pub enum SyncMsg {
    /// A replica's state vector — "here is what I already have; send me the rest."
    StateVector(Vec<u8>),
    /// An incremental doc update — a diff reply, or a live change to apply.
    Update(Vec<u8>),
}

const SYNC_TAG_SV: u8 = 0;
const SYNC_TAG_UPDATE: u8 = 1;

impl SyncMsg {
    /// Frame as `[tag, payload…]`.
    pub fn encode(self) -> Vec<u8> {
        let (tag, mut body) = match self {
            SyncMsg::StateVector(b) => (SYNC_TAG_SV, b),
            SyncMsg::Update(b) => (SYNC_TAG_UPDATE, b),
        };
        let mut out = Vec::with_capacity(body.len() + 1);
        out.push(tag);
        out.append(&mut body);
        out
    }

    /// Parse a framed message; `None` on empty input or an unknown tag.
    pub fn decode(bytes: &[u8]) -> Option<SyncMsg> {
        let (tag, body) = bytes.split_first()?;
        match *tag {
            SYNC_TAG_SV => Some(SyncMsg::StateVector(body.to_vec())),
            SYNC_TAG_UPDATE => Some(SyncMsg::Update(body.to_vec())),
            _ => None,
        }
    }
}

/// True when the leading LEB128 var-uint entry count of a v1 state vector is one the remaining
/// bytes could actually back (every entry costs at least two bytes: a var-uint client id and a
/// var-uint clock).
///
/// This must be checked BEFORE handing the bytes to `yrs`: `StateVector::decode_v1` allocates a
/// `HashMap` from the DECLARED count before reading a single entry, so six bytes off the wire
/// declaring ~4e9 entries abort the whole process through `handle_alloc_error` — which is neither
/// an `Err` nor a catchable panic, and takes the engine, the tick thread and the unsaved patch
/// with it. An unterminated or over-wide var-uint is rejected for the same reason.
fn declared_len_is_backed(bytes: &[u8]) -> bool {
    let mut declared: u64 = 0;
    let mut shift = 0u32;
    for (i, byte) in bytes.iter().enumerate() {
        declared |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return declared <= (bytes.len() - i - 1) as u64 / 2;
        }
        shift += 7;
        if shift >= 64 {
            return false; // wider than the u64 it is being read into
        }
    }
    false // ran out of bytes mid-var-uint (or empty input)
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

/// Read a scalar map key as JSON, for change detection. Numbers are always stored as f64 (see
/// [`insert_scalar`]), so a caller comparing against an `i64`-typed value must compare by
/// `as_f64` — [`scalar_unchanged`] does.
fn read_scalar<T: ReadTxn>(map: &MapRef, txn: &T, key: &str) -> Option<serde_json::Value> {
    match map.get(txn, key) {
        Some(Out::Any(Any::Number(n))) => Some(serde_json::json!(n)),
        Some(Out::Any(Any::Bool(b))) => Some(serde_json::json!(b)),
        Some(Out::Any(Any::String(s))) => Some(serde_json::json!(s.to_string())),
        // The fourth kind [`insert_scalar`] can write. Without it a null leaf reads as absent, so
        // it compares unequal to the null being written and is rewritten on every re-mirror.
        Some(Out::Any(Any::Null)) => Some(serde_json::Value::Null),
        _ => None,
    }
}

/// True when the map's current scalar at `key` already equals `v` — comparing numbers by f64
/// (the doc stores every number as f64) so an incoming `i64` matches its stored `f64` form.
fn scalar_unchanged<T: ReadTxn>(map: &MapRef, txn: &T, key: &str, v: &serde_json::Value) -> bool {
    match (read_scalar(map, txn, key), v) {
        (Some(serde_json::Value::Number(a)), serde_json::Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Some(a), b) => &a == b,
        (None, _) => false,
    }
}

/// Get an existing nested map by key, or insert a fresh one.
fn get_or_insert_map(parent: &MapRef, txn: &mut yrs::TransactionMut, key: &str) -> MapRef {
    match parent.get(txn, key).and_then(|v| v.cast::<MapRef>().ok()) {
        Some(m) => m,
        None => parent.insert(txn, key, MapPrelim::default()),
    }
}

/// The single generic writer behind the graph→doc mirror: recursively reconcile a live Y.Map to
/// match `target` (a JSON object). For each target key — a nested object recurses INTO the existing
/// sub-map (get-or-insert; the sub-map is NEVER replaced, so a concurrent client leaf-write into a
/// sibling key survives — the "params lesson"); a scalar/string is written only when its Any-space
/// value differs (idempotent, with int/float normalized by [`scalar_unchanged`]). Finally every doc
/// key absent from `target` is pruned. An unchanged re-assert produces zero doc ops — which is what
/// keeps the re-mirror from churning tombstones or manufacturing a write that races a client's edit.
fn reconcile_map(
    txn: &mut yrs::TransactionMut,
    map: &MapRef,
    target: &serde_json::Map<String, serde_json::Value>,
) {
    for (key, val) in target {
        match val {
            serde_json::Value::Object(obj) => {
                let child = get_or_insert_map(map, txn, key);
                reconcile_map(txn, &child, obj);
            }
            scalar => {
                if !scalar_unchanged(map, txn, key, scalar) {
                    insert_scalar(map, txn, key, scalar);
                }
            }
        }
    }
    let keep: std::collections::HashSet<&str> = target.keys().map(String::as_str).collect();
    let stale: Vec<String> =
        map.keys(&*txn).filter(|k| !keep.contains(*k)).map(|k| k.to_string()).collect();
    for k in stale {
        map.remove(txn, k.as_str());
    }
}

/// Convert a yrs `Any` (what `ToJson` yields) into a `serde_json::Value` — the generic bridge
/// behind [`GraphDoc::to_json`]. Goes via `Any`'s own JSON serialization (numbers are f64).
fn any_to_json(a: Any) -> serde_json::Value {
    let mut s = String::new();
    a.to_json(&mut s);
    serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
}

/// The four string leaves a stored link has, in canonical form — dropping any projection entry
/// missing one. This canonical object is BOTH the equality key (vs `links.to_json`) and the source
/// for the yrs rebuild, so links never round-trip through a typed struct.
fn canonical_link(v: &serde_json::Value) -> Option<serde_json::Value> {
    let s = |k: &str| -> Option<&str> { v.get(k)?.as_str() };
    Some(serde_json::json!({
        "node_out": s("node_out")?, "slot_out": s("slot_out")?,
        "node_in": s("node_in")?, "slot_in": s("slot_in")?,
    }))
}

/// The control-plane document: four roots — three maps keyed by uid/name, plus the ordered
/// `links` array. What each root's values contain is the projection's business, not this crate's.
pub struct GraphDoc {
    doc: Doc,
    nodes: MapRef,
    links: ArrayRef,
    instances: MapRef,
    /// Patch globals — a Map<name, {value, type, system}>. System globals carry `system: true` (the
    /// panel disables their delete). Reconciled from the engine like `nodes`/`instances`.
    globals: MapRef,
}

impl GraphDoc {
    pub fn new() -> GraphDoc {
        let doc = Doc::new();
        let nodes = doc.get_or_insert_map("nodes");
        let links = doc.get_or_insert_array("links");
        let instances = doc.get_or_insert_map("instances");
        let globals = doc.get_or_insert_map("globals");
        GraphDoc { doc, nodes, links, instances, globals }
    }

    /// The uids of all nodes currently in the doc.
    pub fn node_ids(&self) -> Vec<String> {
        let txn = self.doc.transact();
        self.nodes.keys(&txn).map(|k| k.to_string()).collect()
    }

    /// Reconcile the ENTIRE control-plane doc from one JSON projection of the engine graph — the
    /// generic mirror that replaces the typed writer zoo. `target` is `{nodes, links, instances,
    /// globals}`, each root's contents opaque here. Idempotent and in-place (see
    /// [`reconcile_map`]); a key omitted from the projection (a cleared `expr`, an unwired
    /// boundary's `inner_node`) is pruned.
    pub fn reconcile_root(&mut self, target: &serde_json::Value) {
        let empty = serde_json::Map::new();
        let nodes = target.get("nodes").and_then(|v| v.as_object()).unwrap_or(&empty);
        let instances = target.get("instances").and_then(|v| v.as_object()).unwrap_or(&empty);
        let globals = target.get("globals").and_then(|v| v.as_object()).unwrap_or(&empty);
        {
            let mut txn = self.doc.transact_mut();
            reconcile_map(&mut txn, &self.nodes, nodes);
            reconcile_map(&mut txn, &self.instances, instances);
            reconcile_map(&mut txn, &self.globals, globals);
        }
        // Links are an ordered, manager-authoritative array (no client leaf-merge) → the idempotent
        // skip-if-equal wholesale replace, reused verbatim, straight from the projection's JSON array.
        self.replace_links(target.get("links").and_then(|v| v.as_array()).map(|a| a.as_slice()).unwrap_or(&[]));
    }

    /// The entire control-plane doc as plain JSON (`{nodes, links, instances, globals}`) — the
    /// generic reader, via yrs' own `ToJson`. The manager/tests navigate this instead of typed getters.
    pub fn to_json(&self) -> serde_json::Value {
        let txn = self.doc.transact();
        serde_json::json!({
            "nodes": any_to_json(self.nodes.to_json(&txn)),
            "links": any_to_json(self.links.to_json(&txn)),
            "instances": any_to_json(self.instances.to_json(&txn)),
            "globals": any_to_json(self.globals.to_json(&txn)),
        })
    }

    /// Read the JSON value at a `["nodes", uid, …]` path, or `None` if any segment is missing —
    /// the generic getter (serde-pointer navigation over [`Self::to_json`]).
    pub fn read_at(&self, path: &[&str]) -> Option<serde_json::Value> {
        let mut cur = self.to_json();
        for seg in path {
            cur = cur.get(seg)?.clone();
        }
        Some(cur)
    }

    /// Replace the whole link set (wholesale; a fine-grained incremental diff comes later). Guarded
    /// idempotent: the re-mirror re-asserts this after every op, so when the set is UNCHANGED (the
    /// common case — links change far less often than params/positions) it must produce no doc ops.
    /// An unguarded remove-all+re-push would churn the link array (new items + tombstones) on every
    /// unrelated edit, defeating the empty-diff broadcast-skip for any patch that has links.
    fn replace_links(&mut self, links: &[serde_json::Value]) {
        // Canonicalize the projection to exactly the four string leaves we store, in order, then
        // compare against the current array read through the generic `ToJson` bridge: order-sensitive
        // array equality, string leaves, no numeric normalization.
        let target: Vec<serde_json::Value> = links.iter().filter_map(canonical_link).collect();
        {
            let txn = self.doc.transact();
            if any_to_json(self.links.to_json(&txn)) == serde_json::Value::Array(target.clone()) {
                return; // unchanged — no wholesale rewrite (order-sensitive equality)
            }
        }
        let mut txn = self.doc.transact_mut();
        let len = self.links.len(&txn);
        self.links.remove_range(&mut txn, 0, len);
        for l in &target {
            let m: MapRef = self.links.push_back(&mut txn, MapPrelim::default());
            for k in ["node_out", "slot_out", "node_in", "slot_in"] {
                m.insert(&mut txn, k, l[k].as_str().unwrap_or_default());
            }
        }
    }

    /// The uids of all sub-patch instances currently in the doc.
    pub fn instance_ids(&self) -> Vec<String> {
        let txn = self.doc.transact();
        self.instances.keys(&txn).map(|k| k.to_string()).collect()
    }

    /// The full document state as a v1 update (what a joining client would receive).
    fn encode_state(&self) -> Vec<u8> {
        let txn = self.doc.transact();
        txn.encode_state_as_update_v1(&yrs::StateVector::default())
    }

    /// This replica's state vector (v1), which a peer advertises so the other side can
    /// compute the minimal diff. Opaque bytes — the sync relay just shuttles them.
    pub fn state_vector(&self) -> Vec<u8> {
        use yrs::updates::encoder::Encode;
        self.doc.transact().state_vector().encode_v1()
    }

    /// The minimal v1 update carrying everything this doc has that the peer (described by
    /// its `state_vector`) lacks. Malformed bytes degrade to the empty state vector — i.e. the
    /// peer is sent everything — which is always correct, just larger.
    pub fn diff(&self, peer_state_vector: &[u8]) -> Vec<u8> {
        let sv = if declared_len_is_backed(peer_state_vector) {
            yrs::StateVector::decode_v1(peer_state_vector).unwrap_or_default()
        } else {
            yrs::StateVector::default()
        };
        self.doc.transact().encode_state_as_update_v1(&sv)
    }

    /// Apply a peer's incremental v1 update into this replica. `Err` if it is malformed.
    fn apply_update(&mut self, update: &[u8]) -> Result<(), String> {
        let u = yrs::Update::decode_v1(update).map_err(|e| e.to_string())?;
        let mut txn = self.doc.transact_mut();
        txn.apply_update(u).map_err(|e| e.to_string())
    }

    /// The message to send a peer on connect: this replica's state vector, framed. The peer
    /// answers with the diff it owes (via [`Self::on_sync`]).
    pub fn sync_hello(&self) -> Vec<u8> {
        SyncMsg::StateVector(self.state_vector()).encode()
    }

    /// A framed `Update` carrying this replica's ENTIRE state — the recovery payload for a
    /// peer that has fallen behind (broadcast lag) or reconnected. Applying it is idempotent
    /// and resolves any gap, including updates a client buffered as pending because they
    /// depended on a dropped one. Recovery must use this, NOT [`Self::sync_hello`]: a reader
    /// answers a bare state vector with an empty diff and never pulls what it is missing.
    pub fn full_state_frame(&self) -> Vec<u8> {
        SyncMsg::Update(self.encode_state()).encode()
    }

    /// Drive the pairwise sync handshake for one inbound message, returning the messages to
    /// send back. Receiving a peer's `StateVector` yields the `Update` it lacks; receiving an
    /// `Update` applies it and replies with nothing. Symmetric — both ends run this.
    pub fn on_sync(&mut self, msg: SyncMsg) -> Vec<SyncMsg> {
        match msg {
            SyncMsg::StateVector(sv) => vec![SyncMsg::Update(self.diff(&sv))],
            SyncMsg::Update(u) => {
                let _ = self.apply_update(&u);
                Vec::new()
            }
        }
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

    // ---- test shims: read leaves through the generic reader (the typed getters were removed) ----
    // Numbers come back from `to_json` in shortest form (a whole f64 `30.0` reads as integer `30`),
    // so numeric assertions compare via `as_f64`, not exact `json!` equality.
    fn nstr(doc: &GraphDoc, uid: &str, key: &str) -> Option<String> {
        doc.read_at(&["nodes", uid, key]).and_then(|v| v.as_str().map(String::from))
    }
    fn npos(doc: &GraphDoc, root: &str, uid: &str) -> Option<[f64; 2]> {
        let x = doc.read_at(&[root, uid, "pos", "x"])?.as_f64()?;
        let y = doc.read_at(&[root, uid, "pos", "y"])?.as_f64()?;
        Some([x, y])
    }
    fn pval(doc: &GraphDoc, uid: &str, g: &str, n: &str) -> Option<serde_json::Value> {
        doc.read_at(&["nodes", uid, "params", g, n, "value"])
    }
    fn pnum(doc: &GraphDoc, uid: &str, g: &str, n: &str) -> Option<f64> {
        pval(doc, uid, g, n).and_then(|v| v.as_f64())
    }
    fn pexpr_src(doc: &GraphDoc, uid: &str, g: &str, n: &str) -> Option<String> {
        doc.read_at(&["nodes", uid, "params", g, n, "expr", "source"])
            .and_then(|v| v.as_str().map(String::from))
    }
    fn viewers(doc: &GraphDoc, uid: &str) -> Option<serde_json::Value> {
        doc.read_at(&["nodes", uid, "viewers"])
            .and_then(|v| v.as_str().and_then(|s| serde_json::from_str(s).ok()))
    }
    fn links(doc: &GraphDoc) -> Vec<serde_json::Value> {
        doc.read_at(&["links"]).and_then(|v| v.as_array().cloned()).unwrap_or_default()
    }

    #[test]
    fn a_fresh_doc_has_no_nodes() {
        let doc = GraphDoc::new();
        assert!(doc.node_ids().is_empty());
    }

    #[test]
    fn viewers_blob_and_links_round_trip() {
        use serde_json::json;
        // A node's viewers blob is a STRING leaf; build it via the generic reconciler and read it
        // back through the generic reader (parsing the string), then exercise the wholesale replace.
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&json!({
            "nodes": { "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0},
                "params": {}, "viewers": "{\"out\":{\"kind\":\"line\"}}" } },
            "links": [], "instances": {}
        }));
        assert_eq!(viewers(&doc, "1"), Some(json!({"out": {"kind": "line"}})));

        doc.replace_links(&[json!({
            "node_out": "1", "slot_out": "out", "node_in": "2", "slot_in": "data",
        })]);
        assert_eq!(links(&doc).len(), 1);
        assert_eq!(links(&doc)[0]["slot_in"], json!("data"));
        doc.replace_links(&[]);
        assert!(links(&doc).is_empty());
    }

    #[test]
    fn replace_links_is_idempotent() {
        // The re-mirror re-asserts the whole link set after every op. Re-asserting the SAME links
        // must produce NO doc ops — else the link array churns (new items + tombstones) on every
        // unrelated edit, defeating the empty-diff broadcast-skip for any patch that has links.
        let mut doc = GraphDoc::new();
        let l = |a: &str, b: &str| {
            serde_json::json!({ "node_out": a, "slot_out": "out", "node_in": b, "slot_in": "in" })
        };
        doc.replace_links(&[l("1", "2"), l("2", "3")]);

        let before = doc.to_json();
        doc.replace_links(&[l("1", "2"), l("2", "3")]);
        assert_eq!(doc.to_json(), before, "re-asserting the same link set must be a no-op");
        // A real change (an added link) still applies.
        doc.replace_links(&[l("1", "2"), l("2", "3"), l("3", "4")]);
        assert_ne!(doc.to_json(), before, "a real link change is a logical change");
        assert_eq!(links(&doc).len(), 3);
        // Order matters — a reordering is a real change.
        let before2 = doc.to_json();
        doc.replace_links(&[l("3", "4"), l("1", "2"), l("2", "3")]);
        assert_ne!(doc.to_json(), before2, "a reordering is a change");
    }

    #[test]
    fn remove_node_and_state_round_trip() {
        use serde_json::json;
        let node2 = || json!({ "type": "Buffer", "name": "buf", "pos": {"x": 1.0, "y": 2.0}, "params": {} });
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {} },
            "2": node2()
        }, "links": [], "instances": {} }));
        // Removal is wholesale: re-mirror the projection with node 1 omitted → it is pruned.
        doc.reconcile_root(&json!({ "nodes": { "2": node2() }, "links": [], "instances": {} }));
        assert_eq!(doc.node_ids(), vec!["2"]);

        let bytes = doc.encode_state();
        let mut copy = GraphDoc::new();
        copy.apply_update(&bytes).unwrap();
        assert_eq!(copy.node_ids(), vec!["2"]);
        assert_eq!(nstr(&copy, "2", "name").as_deref(), Some("buf"));
    }

    #[test]
    fn sync_diff_converges_two_replicas() {
        use serde_json::json;
        // The relay handshake: a peer advertises its state vector, the other returns a diff,
        // the peer applies it and converges — the primitive the /control sync relay uses.
        let node = |name: &str| json!({ "nodes": {
            "1": { "type": "Oscillator", "name": name, "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} });
        let mut server = GraphDoc::new();
        server.reconcile_root(&node("osc"));

        let client = GraphDoc::new(); // empty replica just joined
        let diff = server.diff(&client.state_vector());
        let mut client = client;
        client.apply_update(&diff).unwrap();
        assert_eq!(nstr(&client, "1", "name").as_deref(), Some("osc"), "client converged via diff");

        // A later incremental edit on the server produces a small diff the client applies.
        server.reconcile_root(&node("osc2"));
        let diff2 = server.diff(&client.state_vector());
        client.apply_update(&diff2).unwrap();
        assert_eq!(nstr(&client, "1", "name").as_deref(), Some("osc2"));
    }

    #[test]
    fn a_hostile_state_vector_degrades_instead_of_aborting() {
        use serde_json::json;
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));

        // Six bytes off the wire declaring ~4e9 entries. yrs pre-allocates the map from the
        // DECLARED count before reading a single entry, so an unvalidated decode aborts the
        // WHOLE PROCESS via `handle_alloc_error` — not a catchable panic, not an `Err`.
        let full = server.diff(&[]);
        assert_eq!(server.diff(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F]), full, "count exceeds the bytes backing it");
        assert_eq!(server.diff(&[0xFF; 12]), full, "var-uint that never terminates");
        assert_eq!(server.diff(&[0x04, 0x01, 0x00]), full, "4 entries in 2 bytes");

        // An honest state vector still computes a real (here: empty) diff.
        assert!(server.diff(&server.state_vector()).len() < full.len(), "an up-to-date peer is owed nothing");
    }

    #[test]
    fn sync_msg_encode_decode_round_trip() {
        for m in [
            SyncMsg::StateVector(vec![1, 2, 3]),
            SyncMsg::Update(vec![9, 8]),
        ] {
            let bytes = m.clone().encode();
            assert_eq!(SyncMsg::decode(&bytes), Some(m));
        }
        assert_eq!(SyncMsg::decode(&[]), None, "empty is not a message");
        assert_eq!(SyncMsg::decode(&[7, 0]), None, "unknown tag rejected");
    }

    #[test]
    fn on_sync_pairwise_handshake_converges() {
        use serde_json::json;
        // The symmetric handshake: each side sends its SV on connect; receiving a peer's SV
        // yields an Update carrying what the peer lacks; receiving an Update applies it.
        let node1 = || json!({ "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {} });
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": { "1": node1() }, "links": [], "instances": {} }));
        let mut client = GraphDoc::new();

        // Connect: both emit their SV.
        let server_hello = server.sync_hello();
        let client_hello = client.sync_hello();

        // Server receives client's SV → replies with the diff the client is missing.
        let to_client = server.on_sync(SyncMsg::decode(&client_hello).unwrap());
        // Client receives server's SV → replies with the diff the server is missing (none here).
        let _to_server = client.on_sync(SyncMsg::decode(&server_hello).unwrap());

        // Client applies the server's diff → converges.
        for m in to_client {
            client.on_sync(m);
        }
        assert_eq!(nstr(&client, "1", "name").as_deref(), Some("osc"), "client converged via on_sync");

        // A live server edit, relayed as one Update, lands on the client. reconcile_root is
        // wholesale, so add node 2 while KEEPING node 1 (omitting it would prune it).
        server.reconcile_root(&json!({ "nodes": {
            "1": node1(),
            "2": { "type": "Buffer", "name": "buf", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));
        let live = server.diff(&client.state_vector());
        client.on_sync(SyncMsg::Update(live));
        assert_eq!(nstr(&client, "2", "name").as_deref(), Some("buf"));
    }

    #[test]
    fn full_state_frame_recovers_a_gapped_replica() {
        // The recovery contract: when a client has missed deltas (lag/reconnect), the server
        // ships its FULL STATE as an Update; applying it converges the client regardless of
        // what it missed — including a change that DEPENDS on a missed one (which yrs would
        // otherwise buffer as an unresolvable pending update). This is why recovery must send
        // full state, not the server's state vector (which a reader answers with an empty diff).
        use serde_json::json;
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));

        let mut client = GraphDoc::new();
        client.apply_update(&server.diff(&client.state_vector())).unwrap();
        assert_eq!(client.node_ids(), vec!["1"], "client synced node 1");

        // The client now MISSES everything below (dropped deltas): a new node, a param edit, and a
        // rename of node 1 (a struct that chains off the earlier one). One wholesale re-mirror.
        server.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "name": "renamed", "pos": {"x": 0.0, "y": 0.0},
                "params": { "common": { "max_frequency": { "value": 50.0 } } } },
            "2": { "type": "Buffer", "name": "buf", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));

        // Recovery: apply the framed full state. Convergence, not divergence.
        let SyncMsg::Update(full) = SyncMsg::decode(&server.full_state_frame()).unwrap() else {
            panic!("full_state_frame is an Update");
        };
        client.apply_update(&full).unwrap();
        assert_eq!(client.node_ids().len(), 2, "gapped node arrived");
        assert_eq!(nstr(&client, "1", "name").as_deref(), Some("renamed"), "dependent change resolved");
        assert_eq!(nstr(&client, "2", "name").as_deref(), Some("buf"));
        assert_eq!(pnum(&client, "1", "common", "max_frequency"), Some(50.0));
    }

    // ---- generic reconcile_root: the single writer that subsumes the typed writer zoo ----

    /// A doc-projection covering every shape the reconciler must handle: a node with params (one
    /// plain, one expression-bound), a viewers blob, a link, and a sub-patch instance with a member
    /// and a wired output boundary. Exactly the doc's field set — no runtime fields.
    fn full_projection() -> serde_json::Value {
        serde_json::json!({
            "nodes": {
                "1": {
                    "type": "Oscillator", "name": "osc", "pos": {"x": 10.0, "y": 20.0},
                    "params": {
                        "common": { "max_frequency": { "value": 30.0 } },
                        "oscillator": { "waveform": { "value": "sine",
                            "expr": { "source": "nd('lfo')", "enabled": true, "triggers": false } } }
                    },
                    "viewers": "{\"out\":{\"kind\":\"line\"}}"
                },
                "2": { "type": "Buffer", "name": "buf", "pos": {"x": 0.0, "y": 0.0}, "params": {} }
            },
            "links": [ { "node_out": "1", "slot_out": "out", "node_in": "2", "slot_in": "data" } ],
            "instances": {
                "i1": {
                    "name": "subpatch0", "parent": ROOT_MARK, "pos": {"x": 5.0, "y": 6.0},
                    "members": { "buffer0": "2" },
                    "interface": { "out0": { "dir": "out", "dtype": "ARRAY", "name": "wave",
                        "pos": {"x": 1.0, "y": 2.0}, "inner_node": "2", "inner_slot": "out" } }
                }
            }
        })
    }
    const ROOT_MARK: &str = "__root__";

    #[test]
    fn reconcile_mirrors_globals_and_is_idempotent() {
        use serde_json::json;
        let proj = || json!({
            "nodes": {}, "links": [], "instances": {},
            "globals": {
                "default_ufreq": { "value": 30.0, "type": "float", "system": true },
                "subject": { "value": "P07", "type": "string", "system": false },
            }
        });
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&proj());
        assert_eq!(doc.read_at(&["globals", "default_ufreq", "value"]).and_then(|v| v.as_f64()), Some(30.0));
        assert_eq!(doc.read_at(&["globals", "default_ufreq", "system"]), Some(json!(true)));
        assert_eq!(doc.read_at(&["globals", "subject", "value"]).and_then(|v| v.as_str().map(str::to_string)), Some("P07".into()));
        // Idempotent: re-mirroring the same globals produces no logical change — the params lesson.
        let before = doc.to_json();
        doc.reconcile_root(&proj());
        assert_eq!(doc.to_json(), before, "re-mirroring identical globals is a no-op");
    }

    #[test]
    fn reconcile_prunes_a_removed_global() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&json!({ "nodes": {}, "links": [], "instances": {},
            "globals": { "g": { "value": 1, "type": "int", "system": false } } }));
        assert!(doc.read_at(&["globals", "g", "value"]).is_some());
        // A re-mirror without `g` prunes it (mirror of a user delete applied to the engine).
        doc.reconcile_root(&json!({ "nodes": {}, "links": [], "instances": {}, "globals": {} }));
        assert!(doc.read_at(&["globals", "g"]).is_none());
    }

    #[test]
    fn reconcile_root_builds_the_whole_graph() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());

        // Nodes + identity + params (value AND binding) + viewers, via the generic reader.
        assert_eq!(doc.node_ids().len(), 2);
        assert_eq!(nstr(&doc, "1", "type").as_deref(), Some("Oscillator"));
        assert_eq!(nstr(&doc, "1", "name").as_deref(), Some("osc"));
        assert_eq!(npos(&doc, "nodes", "1"), Some([10.0, 20.0]));
        assert_eq!(pnum(&doc, "1", "common", "max_frequency"), Some(30.0));
        assert_eq!(pval(&doc, "1", "oscillator", "waveform"), Some(json!("sine")));
        assert_eq!(pexpr_src(&doc, "1", "oscillator", "waveform").as_deref(), Some("nd('lfo')"));
        assert_eq!(viewers(&doc, "1"), Some(json!({"out": {"kind": "line"}})));
        // Links.
        assert_eq!(links(&doc).len(), 1);
        assert_eq!(links(&doc)[0]["node_in"], json!("2"));
        // The sub-patch forest — read the instance object from the generic reader.
        let j = doc.to_json();
        let rec = &j["instances"]["i1"];
        assert_eq!(rec["parent"], json!("__root__"));
        assert_eq!(npos(&doc, "instances", "i1"), Some([5.0, 6.0]));
        assert!(rec.get("def_id").is_none(), "a unique instance omits def_id");
        assert_eq!(rec["members"], json!({ "buffer0": "2" }));
        let out = rec["interface"]
            .as_object()
            .unwrap()
            .values()
            .find(|b| b["dir"] == json!("out"))
            .expect("output boundary");
        assert_eq!(out["inner_node"], json!("2"));
        assert_eq!(out["inner_slot"], json!("out"));
    }

    #[test]
    fn reconcile_root_is_idempotent() {
        // The load-bearing invariant: re-asserting an UNCHANGED projection produces ZERO doc ops
        // (else the re-mirror churns tombstones and manufactures competing writes that race a
        // client's leaf-edit — the "params lesson" the typed writers hand-rolled per field).
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());
        let before = doc.to_json();
        doc.reconcile_root(&full_projection());
        assert_eq!(doc.to_json(), before, "re-reconciling an unchanged graph is a no-op");
    }

    #[test]
    fn a_null_leaf_is_idempotent_like_every_other_scalar() {
        use serde_json::json;
        // `insert_scalar` stores a JSON null as `Any::Null`, so `read_scalar` must read it back as
        // a null — otherwise the leaf is forever "changed", `reconcile_map` rewrites it on every
        // re-mirror, and the idempotence invariant holds for three of the four scalar kinds only.
        // `to_json` cannot see this (a rewritten null looks identical), so assert on the doc's own
        // clock: a write bumps the state vector, a no-op does not.
        let mut doc = GraphDoc::new();
        let proj = json!({ "nodes": { "1": { "type": "Buffer", "name": "buf",
            "pos": {"x": 0.0, "y": 0.0},
            "params": { "buffer": { "size": { "value": null } } } } },
            "links": [], "instances": {} });
        doc.reconcile_root(&proj);
        let sv = doc.state_vector();
        doc.reconcile_root(&proj);
        assert_eq!(doc.state_vector(), sv, "re-asserting an unchanged null leaf writes nothing");
    }

    #[test]
    fn reconcile_normalizes_int_vs_float_numbers() {
        use serde_json::json;
        // Numbers are stored as f64. A projection carrying an INT param value (e.g. Buffer.size)
        // must not churn against its stored f64 form on the next re-mirror.
        let mut doc = GraphDoc::new();
        let mut proj = json!({ "nodes": { "1": { "type": "Buffer", "name": "buf",
            "pos": {"x": 0.0, "y": 0.0}, "params": { "buffer": { "size": { "value": 1000 } } } } },
            "links": [], "instances": {} });
        doc.reconcile_root(&proj);
        let before = doc.to_json();
        // Re-assert with the value as a float — the same number, different JSON repr.
        proj["nodes"]["1"]["params"]["buffer"]["size"]["value"] = json!(1000.0);
        doc.reconcile_root(&proj);
        assert_eq!(doc.to_json(), before, "int 1000 vs f64 1000.0 is not a change");
    }

    #[test]
    fn reconcile_prunes_removed_keys() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());
        assert!(pexpr_src(&doc, "1", "oscillator", "waveform").is_some());

        // A shrunk projection: node 2 gone, node 1's expr binding cleared, the instance's member
        // dropped, and the instance itself removed. Every stale key must be pruned.
        let shrunk = json!({
            "nodes": { "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 10.0, "y": 20.0},
                "params": { "oscillator": { "waveform": { "value": "sine" } } } } },
            "links": [],
            "instances": {}
        });
        doc.reconcile_root(&shrunk);
        assert_eq!(doc.node_ids(), vec!["1"], "node 2 pruned");
        assert_eq!(pexpr_src(&doc, "1", "oscillator", "waveform"), None, "cleared binding pruned");
        assert!(pval(&doc, "1", "common", "max_frequency").is_none(), "removed param group pruned");
        assert!(links(&doc).is_empty(), "links cleared");
        assert!(doc.instance_ids().is_empty(), "instance pruned");
    }

    // ---- generic read (to_json / read_at) ----

    #[test]
    fn to_json_and_read_at_expose_the_whole_doc_generically() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());

        // to_json yields the doc's three roots as plain JSON — the generic reader. Numbers come
        // back in shortest form (a whole f64 30.0 serializes as 30), so compare them numerically.
        let j = doc.to_json();
        assert_eq!(j["nodes"]["1"]["name"], json!("osc"));
        assert_eq!(j["nodes"]["1"]["params"]["common"]["max_frequency"]["value"].as_f64(), Some(30.0));
        assert_eq!(j["instances"]["i1"]["parent"], json!("__root__"));
        assert_eq!(j["links"][0]["node_in"], json!("2"));

        // read_at navigates by path (serde-pointer semantics), None when absent.
        assert_eq!(read_at_val(&doc, &["nodes", "1", "pos", "x"]).and_then(|v| v.as_f64()), Some(10.0));
        assert_eq!(
            read_at_val(&doc, &["nodes", "1", "params", "oscillator", "waveform", "expr", "source"]),
            Some(json!("nd('lfo')"))
        );
        assert_eq!(read_at_val(&doc, &["nodes", "nope"]), None);
    }
    // Small shim so the test reads a path without repeating the join.
    fn read_at_val(doc: &GraphDoc, path: &[&str]) -> Option<serde_json::Value> {
        doc.read_at(path)
    }
}
