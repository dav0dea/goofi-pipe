//! `GraphDoc` — a `yrs::Doc` holding goofi's control-plane state, reconciled from and read back
//! as plain JSON. Deliberately SHAPE-AGNOSTIC: [`reconcile_map`] recurses over arbitrary JSON, so
//! what the doc's roots actually contain is owned in exactly one place — [`crate::crdt_mirror`],
//! which builds the projection. Nothing here names an engine or payload type, and that is the
//! property to preserve: this module sees `serde_json::Value` and `yrs`, never a `Graph`.

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
/// an `Err` nor a catchable panic, and takes the whole process with it — every node thread and the
/// unsaved patch alike. An unterminated or over-wide var-uint is rejected for the same reason.
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

/// The control-plane document: five roots — four maps keyed by uid/name/id, plus the ordered
/// `links` array. What each root's values contain is the projection's business, not this crate's.
pub struct GraphDoc {
    doc: Doc,
    nodes: MapRef,
    links: ArrayRef,
    instances: MapRef,
    /// Patch globals — a Map<name, {value, type, system}>. System globals carry `system: true` (the
    /// panel disables their delete). Reconciled from the engine like `nodes`/`instances`.
    globals: MapRef,
    /// The editor's panel arrangement — a Map<id, {kind, parent, order, size, …}>. Flat for exactly
    /// the reason this crate erases nested arrays: an id-keyed map of scalars is what survives.
    arrangement: MapRef,
}

impl GraphDoc {
    pub fn new() -> GraphDoc {
        let doc = Doc::new();
        let nodes = doc.get_or_insert_map("nodes");
        let links = doc.get_or_insert_array("links");
        let instances = doc.get_or_insert_map("instances");
        let globals = doc.get_or_insert_map("globals");
        let arrangement = doc.get_or_insert_map("arrangement");
        GraphDoc { doc, nodes, links, instances, globals, arrangement }
    }

    /// The uids of all nodes currently in the doc.
    pub fn node_ids(&self) -> Vec<String> {
        let txn = self.doc.transact();
        self.nodes.keys(&txn).map(|k| k.to_string()).collect()
    }

    /// Reconcile the ENTIRE control-plane doc from one JSON projection of the engine graph — the
    /// generic mirror that replaces the typed writer zoo. `target` is `{nodes, links, instances,
    /// globals, arrangement}`, each root's contents opaque here. Idempotent and in-place (see
    /// [`reconcile_map`]); a key omitted from the projection (a cleared `expr`, an unwired
    /// boundary's `inner_node`) is pruned.
    pub fn reconcile_root(&mut self, target: &serde_json::Value) {
        let empty = serde_json::Map::new();
        let nodes = target.get("nodes").and_then(|v| v.as_object()).unwrap_or(&empty);
        let instances = target.get("instances").and_then(|v| v.as_object()).unwrap_or(&empty);
        let globals = target.get("globals").and_then(|v| v.as_object()).unwrap_or(&empty);
        let arrangement = target.get("arrangement").and_then(|v| v.as_object()).unwrap_or(&empty);
        {
            let mut txn = self.doc.transact_mut();
            reconcile_map(&mut txn, &self.nodes, nodes);
            reconcile_map(&mut txn, &self.instances, instances);
            reconcile_map(&mut txn, &self.globals, globals);
            reconcile_map(&mut txn, &self.arrangement, arrangement);
        }
        // Links are an ordered, manager-authoritative array (no client leaf-merge) → the idempotent
        // skip-if-equal wholesale replace, reused verbatim, straight from the projection's JSON array.
        self.replace_links(target.get("links").and_then(|v| v.as_array()).map(|a| a.as_slice()).unwrap_or(&[]));
    }

    /// The entire control-plane doc as plain JSON (`{nodes, links, instances, globals,
    /// arrangement}`) — the
    /// generic reader, via yrs' own `ToJson`. The manager/tests navigate this instead of typed getters.
    pub fn to_json(&self) -> serde_json::Value {
        let txn = self.doc.transact();
        serde_json::json!({
            "nodes": any_to_json(self.nodes.to_json(&txn)),
            "links": any_to_json(self.links.to_json(&txn)),
            "instances": any_to_json(self.instances.to_json(&txn)),
            "globals": any_to_json(self.globals.to_json(&txn)),
            "arrangement": any_to_json(self.arrangement.to_json(&txn)),
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
    /// The links root is an ARRAY, not a map, so it is written wholesale rather than reconciled
    /// key by key. Part of the doc's write API alongside [`GraphDoc::reconcile_root`].
    pub fn replace_links(&mut self, links: &[serde_json::Value]) {
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
