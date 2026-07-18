//! `GraphDoc` — a typed façade over a `yrs::Doc` holding goofi's control-plane state
//! (nodes + nested params/viewers, links). The manager keeps this in agreement with the
//! engine `Graph`; it is the sync structure clients will later replicate. Pure: depends
//! only on `yrs` + `serde_json`, no engine/payload types.

use std::collections::HashMap;

use yrs::updates::decoder::Decode;
use yrs::types::ToJson;
use yrs::{Any, Array, ArrayRef, Doc, Map, MapPrelim, MapRef, Out, ReadTxn, Transact};

/// An expression binding as mirrored into the doc.
#[derive(Clone, Debug, PartialEq)]
pub struct ExprRecord {
    pub source: String,
    pub enabled: bool,
    pub triggers: bool,
}

/// A framed message on the `/control` binary channel, one leading tag byte. Two families:
/// **doc sync** (`StateVector` / `Update`) — the minimal equivalent of the Yjs sync protocol,
/// both ends driving their doc by hand (no `y-protocols` dependency); and the **ephemeral /
/// awareness channel** (`Ephemeral`) — presence-style state (cursors, live-drag values,
/// expression previews, active viewer specs) that is NOT persisted in the doc and NOT
/// recovered on lag. The manager relays `Ephemeral` payloads verbatim to all clients; their
/// internal `{client, state}` structure is owned by the browser (each peer self-filters its
/// own client id), so the manager stays an opaque relay until it needs to read viewer specs.
#[derive(Clone, Debug, PartialEq)]
pub enum SyncMsg {
    /// A replica's state vector — "here is what I already have; send me the rest."
    StateVector(Vec<u8>),
    /// An incremental doc update — a diff reply, or a live change to apply.
    Update(Vec<u8>),
    /// An ephemeral/awareness update — relayed to peers, never applied to the doc.
    Ephemeral(Vec<u8>),
}

const SYNC_TAG_SV: u8 = 0;
const SYNC_TAG_UPDATE: u8 = 1;
const SYNC_TAG_EPHEMERAL: u8 = 2;

impl SyncMsg {
    /// Frame as `[tag, payload…]`.
    pub fn encode(self) -> Vec<u8> {
        let (tag, mut body) = match self {
            SyncMsg::StateVector(b) => (SYNC_TAG_SV, b),
            SyncMsg::Update(b) => (SYNC_TAG_UPDATE, b),
            SyncMsg::Ephemeral(b) => (SYNC_TAG_EPHEMERAL, b),
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
            SYNC_TAG_EPHEMERAL => Some(SyncMsg::Ephemeral(body.to_vec())),
            _ => None,
        }
    }
}

/// The merge-safe leaves a client's incremental update changed — what the manager pushes into the
/// engine `Graph` after applying a client doc write. Params carry `(uid, group, name, value)`;
/// positions carry `(uid, [x, y])` for each node or instance box that moved; viewers carry
/// `(uid, blob)` for each node whose per-slot viewer view-state changed; expressions carry
/// `(uid, group, name, binding)` — `Some` for a bound/edited expression, `None` when it was cleared.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClientChanges {
    pub params: Vec<(String, String, String, serde_json::Value)>,
    pub positions: Vec<(String, [f64; 2])>,
    pub viewers: Vec<(String, serde_json::Value)>,
    pub expressions: Vec<(String, String, String, Option<ExprRecord>)>,
    /// Globals: `(name, entry)` — `Some({value, type, ...})` for an added/edited global, `None` when
    /// it was deleted. The manager coerces the entry to a `GlobalValue` and applies it to the engine.
    pub globals: Vec<(String, Option<serde_json::Value>)>,
}

impl ClientChanges {
    /// No leaf changed — the manager has nothing to push.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
            && self.positions.is_empty()
            && self.viewers.is_empty()
            && self.expressions.is_empty()
            && self.globals.is_empty()
    }
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

/// The control-plane document. `nodes` is a Map<uid, {type, name, pos, params, viewers}>,
/// `links` an Array of {node_out, slot_out, node_in, slot_in}, and `instances` the sub-patch
/// forest — a Map<uid, {name, def_id?, parent, pos, members:Map<local,uid>, interface:Map<bnd,…>}>.
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
    /// generic mirror that replaces the typed writer zoo. `target` carries exactly the doc's shape:
    /// `{ nodes: {uid: {type, name, pos, params, viewers}}, links: [{node_out,…}],
    ///    instances: {uid: {name, def_id?, parent, pos, members, interface}} }`.
    /// Idempotent and in-place (see [`reconcile_map`]); optional keys omitted from the projection
    /// (a cleared `expr`, an unwired boundary's `inner_node`, a unique instance's `def_id`) are pruned.
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

    /// The entire control-plane doc as plain JSON (`{nodes, links, instances}`) — the generic
    /// reader, via yrs' own `ToJson`. The manager/tests navigate this instead of the typed getters.
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

    /// Write (or remove) one value at a `["nodes"|"instances", uid, …, leaf]` path, IN PLACE — the
    /// generic merge-safe leaf write, the single-path counterpart of [`Self::reconcile_root`] and the
    /// Rust twin of the browser's `graphDoc` setters. `Some(object)` reconciles that subtree,
    /// `Some(scalar)` set-if-changed, `None` removes the key. No-op when the addressed ENTITY
    /// (`root[uid]`) is absent — the manager owns entity existence, so a leaf write never mints a
    /// phantom node/instance. Idempotent; intermediate maps below the entity are created on demand.
    pub fn write_at(&mut self, path: &[&str], value: Option<&serde_json::Value>) {
        // Need at least [root, uid, leaf]; only the two nested maps carry client-writable leaves.
        if path.len() < 3 {
            return;
        }
        let root = match path[0] {
            "nodes" => &self.nodes,
            "instances" => &self.instances,
            _ => return,
        };
        let mut txn = self.doc.transact_mut();
        let Some(entity) = root.get(&txn, path[1]).and_then(|v| v.cast::<MapRef>().ok()) else {
            return; // absent entity → never mint a phantom
        };
        // Walk/create the intermediate maps between the entity and the leaf.
        let mut cur = entity;
        for seg in &path[2..path.len() - 1] {
            cur = get_or_insert_map(&cur, &mut txn, seg);
        }
        let leaf = path[path.len() - 1];
        match value {
            None => {
                if cur.get(&txn, leaf).is_some() {
                    cur.remove(&mut txn, leaf);
                }
            }
            Some(serde_json::Value::Object(obj)) => {
                let child = get_or_insert_map(&cur, &mut txn, leaf);
                reconcile_map(&mut txn, &child, obj);
            }
            Some(scalar) => {
                if !scalar_unchanged(&cur, &txn, leaf, scalar) {
                    insert_scalar(&cur, &mut txn, leaf, scalar);
                }
            }
        }
    }

    /// Write (or remove) a whole global entry at `globals[name]` — the CRDT twin of the panel's global
    /// add / edit / delete. `Some(obj)` reconciles the `{value, type, system}` entry IN PLACE (so a
    /// value-only edit is one leaf op that a concurrent re-mirror preserves); `None` deletes the entry.
    /// Unlike [`Self::write_at`], this MAY mint a top-level entry — a user adds a global by naming a new
    /// one. (System globals can't be deleted; the manager rejects that and the re-mirror re-asserts.)
    pub fn write_global(&mut self, name: &str, entry: Option<&serde_json::Value>) {
        let mut txn = self.doc.transact_mut();
        match entry {
            Some(serde_json::Value::Object(obj)) => {
                let child = get_or_insert_map(&self.globals, &mut txn, name);
                reconcile_map(&mut txn, &child, obj);
            }
            Some(_) => {} // a global entry is always an object; ignore a malformed scalar
            None => {
                if self.globals.get(&txn, name).is_some() {
                    self.globals.remove(&mut txn, name);
                }
            }
        }
    }

    /// The client-writable global entries `(name, {value, type, system})` for each global in the doc —
    /// the manager diffs these before/after a client update to detect add/edit (entry differs) and
    /// delete (name gone).
    fn client_globals(&self) -> Vec<(String, serde_json::Value)> {
        let doc = self.to_json();
        let Some(globals) = doc.get("globals").and_then(|v| v.as_object()) else {
            return Vec::new();
        };
        globals.iter().map(|(n, e)| (n.clone(), e.clone())).collect()
    }

    /// Replace the whole link set (wholesale; a fine-grained incremental diff comes later). Guarded
    /// idempotent: the re-mirror re-asserts this after every op, so when the set is UNCHANGED (the
    /// common case — links change far less often than params/positions) it must produce no doc ops.
    /// An unguarded remove-all+re-push would churn the link array (new items + tombstones) on every
    /// unrelated edit, defeating the empty-diff broadcast-skip for any patch that has links.
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
    pub fn encode_state(&self) -> Vec<u8> {
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
    /// its `state_vector`) lacks. `Err` if the state-vector bytes are malformed.
    pub fn diff(&self, peer_state_vector: &[u8]) -> Vec<u8> {
        let sv = yrs::StateVector::decode_v1(peer_state_vector).unwrap_or_default();
        self.doc.transact().encode_state_as_update_v1(&sv)
    }

    /// Whether `delta` (a v1 update) carries no changes — the canonical empty update this
    /// doc's own encoder produces for an up-to-date peer. Used to skip no-op broadcasts.
    pub fn is_empty_diff(&self, delta: &[u8]) -> bool {
        delta == self.diff(&self.state_vector()).as_slice()
    }

    /// Apply a peer's incremental v1 update into this replica. `Err` if it is malformed.
    pub fn apply_update(&mut self, update: &[u8]) -> Result<(), String> {
        let u = yrs::Update::decode_v1(update).map_err(|e| e.to_string())?;
        let mut txn = self.doc.transact_mut();
        txn.apply_update(u).map_err(|e| e.to_string())
    }

    /// Derive the four merge-safe leaf snapshots — params, node/instance positions, viewer blobs,
    /// and expression bindings — from a SINGLE [`Self::to_json`] walk. This is the before/after
    /// basis for [`Self::apply_client_update`]; each list reproduces what the retired typed
    /// snapshots yielded, except values now carry `to_json`'s shortest-number form (a whole `30.0`
    /// reads back as integer `30`). Positions cover ROOT nodes AND sub-patch instance boxes (both
    /// carry a top-level `pos`; a node needs both coords numeric, an instance box defaults to 0.0);
    /// viewers parse the opaque STRING leaf; an expression requires a string `source`.
    fn client_leaves(
        &self,
    ) -> (
        Vec<(String, String, String, serde_json::Value)>,
        Vec<(String, [f64; 2])>,
        Vec<(String, serde_json::Value)>,
        Vec<(String, String, String, ExprRecord)>,
    ) {
        let doc = self.to_json();
        let mut params = Vec::new();
        let mut positions = Vec::new();
        let mut viewers = Vec::new();
        let mut exprs = Vec::new();

        let coord = |m: &serde_json::Value, k| m.get("pos").and_then(|p| p.get(k)).and_then(|v| v.as_f64());

        if let Some(nodes) = doc.get("nodes").and_then(|v| v.as_object()) {
            for (uid, node) in nodes {
                // Position — only when both coords are numeric (matches the retired `node_pos`).
                if let (Some(x), Some(y)) = (coord(node, "x"), coord(node, "y")) {
                    positions.push((uid.clone(), [x, y]));
                }
                // Viewers — an opaque STRING leaf, parsed to JSON (skip a node without it).
                if let Some(v) =
                    node.get("viewers").and_then(|v| v.as_str()).and_then(|s| serde_json::from_str(s).ok())
                {
                    viewers.push((uid.clone(), v));
                }
                // Param values (scalar leaves) + expression bindings.
                if let Some(groups) = node.get("params").and_then(|v| v.as_object()) {
                    for (group, g) in groups {
                        let Some(names) = g.as_object() else { continue };
                        for (name, entry) in names {
                            if let Some(val) = entry.get("value") {
                                if val.is_number() || val.is_boolean() || val.is_string() {
                                    params.push((uid.clone(), group.clone(), name.clone(), val.clone()));
                                }
                            }
                            if let Some(expr) = entry.get("expr") {
                                if let Some(source) = expr.get("source").and_then(|v| v.as_str()) {
                                    let flag = |k| expr.get(k) == Some(&serde_json::Value::Bool(true));
                                    exprs.push((
                                        uid.clone(),
                                        group.clone(),
                                        name.clone(),
                                        ExprRecord {
                                            source: source.to_string(),
                                            enabled: flag("enabled"),
                                            triggers: flag("triggers"),
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(instances) = doc.get("instances").and_then(|v| v.as_object()) {
            for (uid, inst) in instances {
                positions.push((uid.clone(), [coord(inst, "x").unwrap_or(0.0), coord(inst, "y").unwrap_or(0.0)]));
            }
        }
        (params, positions, viewers, exprs)
    }

    /// Apply a client's incremental update to this replica and return the merge-safe leaves that
    /// changed — param values, node/instance positions, viewer blobs, and expression bindings — so
    /// the manager can push exactly those into the engine `Graph`. The diff is loop-safe: the
    /// manager's subsequent graph→doc re-mirror writes the same values, which yrs records as no
    /// change. `Err` only if the update bytes are malformed.
    pub fn apply_client_update(&mut self, update: &[u8]) -> Result<ClientChanges, String> {
        let (params_b, pos_b, viewers_b, expr_b) = self.client_leaves();
        let params_before: HashMap<(String, String, String), serde_json::Value> =
            params_b.into_iter().map(|(u, g, n, v)| ((u, g, n), v)).collect();
        let pos_before: HashMap<String, [f64; 2]> = pos_b.into_iter().collect();
        let viewers_before: HashMap<String, serde_json::Value> = viewers_b.into_iter().collect();
        let expr_before: HashMap<(String, String, String), ExprRecord> =
            expr_b.into_iter().map(|(u, g, n, e)| ((u, g, n), e)).collect();
        let globals_before: HashMap<String, serde_json::Value> = self.client_globals().into_iter().collect();

        self.apply_update(update)?;

        let (params_a, pos_a, viewers_a, expr_a) = self.client_leaves();
        let params = params_a
            .into_iter()
            .filter(|(u, g, n, v)| params_before.get(&(u.clone(), g.clone(), n.clone())) != Some(v))
            .collect();
        let positions = pos_a.into_iter().filter(|(u, p)| pos_before.get(u) != Some(p)).collect();
        let viewers = viewers_a.into_iter().filter(|(u, v)| viewers_before.get(u) != Some(v)).collect();
        // Expressions: an added/edited binding appears in `after` with a value differing from
        // `before`; a CLEARED binding is a key in `before` no longer in `after` → reported as None.
        let expr_after: HashMap<(String, String, String), ExprRecord> =
            expr_a.into_iter().map(|(u, g, n, e)| ((u, g, n), e)).collect();
        let mut expressions: Vec<(String, String, String, Option<ExprRecord>)> = Vec::new();
        for ((u, g, n), e) in &expr_after {
            if expr_before.get(&(u.clone(), g.clone(), n.clone())) != Some(e) {
                expressions.push((u.clone(), g.clone(), n.clone(), Some(e.clone())));
            }
        }
        for (k, _) in &expr_before {
            if !expr_after.contains_key(k) {
                expressions.push((k.0.clone(), k.1.clone(), k.2.clone(), None));
            }
        }
        // Globals: an added/edited entry differs from `before`; a deleted one is a `before` name gone
        // from `after` → reported as None.
        let globals_after: HashMap<String, serde_json::Value> = self.client_globals().into_iter().collect();
        let mut globals: Vec<(String, Option<serde_json::Value>)> = Vec::new();
        for (name, entry) in &globals_after {
            if globals_before.get(name) != Some(entry) {
                globals.push((name.clone(), Some(entry.clone())));
            }
        }
        for name in globals_before.keys() {
            if !globals_after.contains_key(name) {
                globals.push((name.clone(), None));
            }
        }
        Ok(ClientChanges { params, positions, viewers, expressions, globals })
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
            // Ephemeral frames are relayed to peers, never applied to the doc — they must not
            // reach the doc handshake. Ignore defensively (the relay routes them separately).
            SyncMsg::Ephemeral(_) => Vec::new(),
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
    fn pexpr(doc: &GraphDoc, uid: &str, g: &str, n: &str) -> Option<ExprRecord> {
        let e = doc.read_at(&["nodes", uid, "params", g, n, "expr"])?;
        let source = e.get("source")?.as_str()?.to_string();
        let flag = |k| e.get(k) == Some(&serde_json::Value::Bool(true));
        Some(ExprRecord { source, enabled: flag("enabled"), triggers: flag("triggers") })
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

        let sv = doc.state_vector();
        doc.replace_links(&[l("1", "2"), l("2", "3")]);
        assert!(
            doc.is_empty_diff(&doc.diff(&sv)),
            "re-asserting the same link set must be a no-op"
        );
        // A real change (an added link) still applies.
        doc.replace_links(&[l("1", "2"), l("2", "3"), l("3", "4")]);
        assert!(!doc.is_empty_diff(&doc.diff(&sv)), "a real link change produces a delta");
        assert_eq!(links(&doc).len(), 3);
        // Order matters — a reordering is a real change.
        let sv2 = doc.state_vector();
        doc.replace_links(&[l("3", "4"), l("1", "2"), l("2", "3")]);
        assert!(!doc.is_empty_diff(&doc.diff(&sv2)), "a reordering is a change");
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
    fn sync_msg_encode_decode_round_trip() {
        for m in [
            SyncMsg::StateVector(vec![1, 2, 3]),
            SyncMsg::Update(vec![9, 8]),
            SyncMsg::Ephemeral(vec![5, 5, 5]),
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
    fn is_empty_diff_detects_no_op_deltas() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));
        // Diff against the current SV = nothing new → empty.
        assert!(doc.is_empty_diff(&doc.diff(&doc.state_vector())));
        // Diff against an empty replica = the whole doc → NOT empty.
        let fresh = GraphDoc::new();
        assert!(!doc.is_empty_diff(&doc.diff(&fresh.state_vector())));
    }

    #[test]
    fn apply_client_update_reports_changed_param_leaves() {
        use serde_json::json;
        // The manager applies a client's leaf write to its replica and learns exactly which
        // params changed, so it can push them to the engine Graph. Diff-based: loop-safe,
        // because the subsequent graph->doc re-mirror writes the same values (idempotent).
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": { "1": {
            "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0},
            "params": { "common": { "max_frequency": { "value": 10.0 } },
                        "oscillator": { "amplitude": { "value": 1.0 } } } } },
            "links": [], "instances": {} }));

        // A client replica syncs, then edits ONE param leaf locally, producing an update.
        let mut client = GraphDoc::new();
        client.apply_update(&server.diff(&client.state_vector())).unwrap();
        client.write_at(&["nodes", "1", "params", "common", "max_frequency", "value"], Some(&json!(25.0)));
        let update = client.diff(&server.state_vector());

        // The manager applies it and is told precisely what changed.
        let changed = server.apply_client_update(&update).unwrap();
        // The changed value is reported in `to_json`'s shortest-number form (whole 25.0 → `25`), so
        // check the tuple identity and the value numerically rather than by exact `json!` equality.
        assert_eq!(changed.params.len(), 1);
        let (u, g, n, v) = &changed.params[0];
        assert_eq!((u.as_str(), g.as_str(), n.as_str()), ("1", "common", "max_frequency"));
        assert_eq!(v.as_f64(), Some(25.0));
        assert!(changed.positions.is_empty(), "no node moved");
        assert_eq!(pnum(&server, "1", "common", "max_frequency"), Some(25.0), "doc updated");
        assert_eq!(pnum(&server, "1", "oscillator", "amplitude"), Some(1.0), "untouched param unchanged");

        // Re-applying the SAME update reports no further changes (idempotent, no phantom loop).
        assert!(server.apply_client_update(&update).unwrap().is_empty());
    }

    #[test]
    fn apply_client_update_reports_changed_positions() {
        use serde_json::json;
        // Dragging a node or a sub-patch instance box commits its new position as a merge-safe
        // leaf write (§4). The manager applies it and learns exactly which uids moved, so it can
        // push each into the engine Graph (set_member_pos) — the same diff-based, loop-safe path
        // as params. A ROOT node and an instance box are both reported.
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({
            "nodes": {
                "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {} },
                "2": { "type": "Buffer", "name": "buf", "pos": {"x": 5.0, "y": 5.0}, "params": {} }
            },
            "links": [],
            "instances": { "00000000000000ff": { // box at [10, 20]
                "name": "subpatch0", "def_id": "00000000000000aa", "parent": "__root__",
                "pos": {"x": 10.0, "y": 20.0},
                "members": { "buffer0": "000000000001", "osc0": "000000000002" },
                "interface": { "out0": { "dir": "out", "dtype": "ARRAY", "name": "wave",
                    "pos": {"x": 1.0, "y": 2.0}, "inner_node": "000000000001", "inner_slot": "out" } }
            } }
        }));
        // A shared instance's def_id round-trips through the generic reconciler + reader.
        assert_eq!(
            server.read_at(&["instances", "00000000000000ff", "def_id"]).as_ref().and_then(|v| v.as_str()),
            Some("00000000000000aa")
        );

        let mut client = GraphDoc::new();
        client.apply_update(&server.diff(&client.state_vector())).unwrap();
        // Move node 1 and the instance box (in-place pos leaf writes); leave node 2 where it is.
        client.write_at(&["nodes", "1", "pos"], Some(&json!({"x": 100.0, "y": 200.0})));
        client.write_at(&["instances", "00000000000000ff", "pos"], Some(&json!({"x": 30.0, "y": 40.0})));
        let update = client.diff(&server.state_vector());

        let mut changed = server.apply_client_update(&update).unwrap();
        changed.positions.sort_by(|a, b| a.0.cmp(&b.0)); // Y.Map key order isn't guaranteed; uids unique
        assert_eq!(
            changed.positions,
            vec![("00000000000000ff".into(), [30.0, 40.0]), ("1".into(), [100.0, 200.0])]
        );
        assert!(changed.params.is_empty(), "no param changed");
        assert_eq!(npos(&server, "nodes", "1"), Some([100.0, 200.0]), "doc updated");
        assert_eq!(npos(&server, "nodes", "2"), Some([5.0, 5.0]), "untouched node unchanged");

        // Idempotent: re-applying reports nothing further.
        assert!(server.apply_client_update(&update).unwrap().is_empty());
    }

    #[test]
    fn apply_client_update_reports_changed_expressions() {
        use serde_json::json;
        // A client binds / edits / clears an nd() expression on a param — the binding is a merge-safe
        // leaf (§4). The manager applies each via set_member_expression and echoes the runtime-
        // enriched param descriptor (carrying expression_error). An added/edited binding is reported
        // as Some; a cleared one as None.
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": { "1": {
            "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0},
            "params": { "common": {
                "frequency": { "value": 10.0 },
                "amplitude": { "value": 1.0,
                    "expr": { "source": "nd('a')", "enabled": true, "triggers": false } }
            } } } },
            "links": [], "instances": {} }));

        let mut client = GraphDoc::new();
        client.apply_update(&server.diff(&client.state_vector())).unwrap();
        // Bind frequency (an expr leaf write; value unchanged), clear amplitude's binding.
        client.write_at(
            &["nodes", "1", "params", "common", "frequency", "expr"],
            Some(&json!({ "source": "nd('f')", "enabled": true, "triggers": true })),
        );
        client.write_at(&["nodes", "1", "params", "common", "amplitude", "expr"], None);
        let update = client.diff(&server.state_vector());

        let mut changed = server.apply_client_update(&update).unwrap();
        changed.expressions.sort_by(|a, b| a.2.cmp(&b.2)); // by param name — deterministic compare
        assert_eq!(
            changed.expressions,
            vec![
                ("1".into(), "common".into(), "amplitude".into(), None),
                (
                    "1".into(),
                    "common".into(),
                    "frequency".into(),
                    Some(ExprRecord { source: "nd('f')".into(), enabled: true, triggers: true })
                ),
            ]
        );
        assert!(changed.params.is_empty(), "values unchanged");
        assert_eq!(
            pexpr(&server, "1", "common", "frequency"),
            Some(ExprRecord { source: "nd('f')".into(), enabled: true, triggers: true })
        );
        assert_eq!(pexpr(&server, "1", "common", "amplitude"), None, "binding cleared");

        // Idempotent.
        assert!(server.apply_client_update(&update).unwrap().expressions.is_empty());
    }

    #[test]
    fn apply_client_update_reports_changed_viewers() {
        use serde_json::json;
        // A client picks a viewer kind / collapses a slot / edits settings — the per-node viewer
        // blob is a merge-safe leaf (§4). The manager applies it and learns which nodes' view-state
        // changed, so it can push each into the engine Graph (set_node_viewers → persists to .gfi).
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {},
                "viewers": json!({"out": {"kind": "line", "collapsed": false}}).to_string() },
            "2": { "type": "Buffer", "name": "buf", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));

        let mut client = GraphDoc::new();
        client.apply_update(&server.diff(&client.state_vector())).unwrap();
        // The viewer blob is a STRING leaf — the client writes its `.to_string()` form.
        let blob = json!({"out": {"kind": "spectrum", "collapsed": true}});
        client.write_at(&["nodes", "1", "viewers"], Some(&json!(blob.to_string())));
        let update = client.diff(&server.state_vector());

        let changed = server.apply_client_update(&update).unwrap();
        assert_eq!(
            changed.viewers,
            vec![("1".into(), json!({"out": {"kind": "spectrum", "collapsed": true}}))]
        );
        assert!(changed.params.is_empty() && changed.positions.is_empty(), "only viewers changed");
        assert_eq!(
            viewers(&server, "1"),
            Some(json!({"out": {"kind": "spectrum", "collapsed": true}}))
        );

        // Idempotent: re-applying reports nothing further.
        assert!(server.apply_client_update(&update).unwrap().is_empty());
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
        // Idempotent: re-mirroring the same globals produces no doc ops (empty diff) — the params lesson.
        let sv = doc.state_vector();
        doc.reconcile_root(&proj());
        assert!(doc.is_empty_diff(&doc.diff(&sv)), "re-mirroring identical globals is a no-op");
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
    fn apply_client_update_reports_global_add_edit_delete() {
        use serde_json::json;
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": {}, "links": [], "instances": {},
            "globals": { "default_ufreq": { "value": 30.0, "type": "float", "system": true } } }));
        let mut client = GraphDoc::new();
        client.apply_update(&server.encode_state()).unwrap();

        // Client ADDS a user global + EDITS the system value in one update.
        let before = client.state_vector();
        client.write_global("gain", Some(&json!({ "value": 2.0, "type": "float", "system": false })));
        client.write_global("default_ufreq", Some(&json!({ "value": 60.0, "type": "float", "system": true })));
        let changes = server.apply_client_update(&client.diff(&before)).unwrap();
        let g: std::collections::HashMap<_, _> = changes.globals.iter().cloned().collect();
        // `to_json` normalizes a whole f64 (2.0) to int 2; the entry's `type: "float"` tag is what the
        // manager reads (via global_from_json) to reconstruct a Float — so compare the value via as_f64.
        assert_eq!(g.get("gain").unwrap().as_ref().unwrap()["value"].as_f64(), Some(2.0));
        assert_eq!(g.get("gain").unwrap().as_ref().unwrap()["type"], json!("float"));
        assert_eq!(g.get("default_ufreq").unwrap().as_ref().unwrap()["value"].as_f64(), Some(60.0));

        // Client DELETES the user global → reported as None.
        let before2 = client.state_vector();
        client.write_global("gain", None);
        let changes2 = server.apply_client_update(&client.diff(&before2)).unwrap();
        assert_eq!(changes2.globals, vec![("gain".to_string(), None)]);
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
        let sv = doc.state_vector();
        doc.reconcile_root(&full_projection());
        assert!(doc.is_empty_diff(&doc.diff(&sv)), "re-reconciling an unchanged graph is a no-op");
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
        let sv = doc.state_vector();
        // Re-assert with the value as a float — the same number, different JSON repr.
        proj["nodes"]["1"]["params"]["buffer"]["size"]["value"] = json!(1000.0);
        doc.reconcile_root(&proj);
        assert!(doc.is_empty_diff(&doc.diff(&sv)), "int 1000 vs f64 1000.0 is not a change");
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

    #[test]
    fn reconcile_preserves_a_concurrent_leaf_write() {
        // The no-clobber invariant via the generic path: a client commits a param leaf-write; an
        // intervening re-mirror at the OLD value must not orphan it (recurse-in-place, never replace
        // the entry map). Then applying the client's delta lands the new value.
        use serde_json::json;
        let mut server = GraphDoc::new();
        let proj = |v: f64| json!({ "nodes": { "1": { "type": "Oscillator", "name": "osc",
            "pos": {"x": 0.0, "y": 0.0}, "params": { "common": { "max_frequency": { "value": v } } } } },
            "links": [], "instances": {} });
        server.reconcile_root(&proj(30.0));

        let mut client = GraphDoc::new();
        client.apply_update(&server.diff(&client.state_vector())).unwrap();
        // Client edits the value to 99 against the entry map it currently holds.
        let cbefore = client.state_vector();
        client.write_at(&["nodes", "1", "params", "common", "max_frequency", "value"], Some(&json!(99.0)));
        let edit = client.diff(&cbefore);

        // Interleaved re-mirror at the still-old 30, then the client's in-flight edit applies.
        server.reconcile_root(&proj(30.0));
        server.apply_update(&edit).unwrap();
        assert_eq!(
            pnum(&server, "1", "common", "max_frequency"),
            Some(99.0),
            "the concurrent param leaf-write survives the intervening re-mirror"
        );
    }

    // ---- generic read (to_json / read_at) + generic single-path write (write_at) ----

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

    #[test]
    fn write_at_sets_a_scalar_leaf_in_place() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());
        // A client edits one param value — the single-path counterpart of reconcile_root.
        doc.write_at(&["nodes", "1", "params", "common", "max_frequency", "value"], Some(&json!(42.0)));
        assert_eq!(pnum(&doc, "1", "common", "max_frequency"), Some(42.0));
        // Idempotent: re-writing the same value is a no-op.
        let sv = doc.state_vector();
        doc.write_at(&["nodes", "1", "params", "common", "max_frequency", "value"], Some(&json!(42.0)));
        assert!(doc.is_empty_diff(&doc.diff(&sv)), "re-writing the same leaf is a no-op");
    }

    #[test]
    fn write_at_sets_and_clears_a_nested_object() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());
        // Set an expression binding (a nested object) on the plain param.
        doc.write_at(
            &["nodes", "1", "params", "common", "max_frequency", "expr"],
            Some(&json!({ "source": "nd('x')", "enabled": true, "triggers": false })),
        );
        assert_eq!(pexpr_src(&doc, "1", "common", "max_frequency").as_deref(), Some("nd('x')"));
        // Clear it with None — the key is removed.
        doc.write_at(&["nodes", "1", "params", "common", "max_frequency", "expr"], None);
        assert_eq!(pexpr_src(&doc, "1", "common", "max_frequency"), None);
    }

    #[test]
    fn write_at_writes_a_node_position_and_an_instance_position() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());
        doc.write_at(&["nodes", "1", "pos"], Some(&json!({ "x": 111.0, "y": 222.0 })));
        assert_eq!(npos(&doc, "nodes", "1"), Some([111.0, 222.0]));
        doc.write_at(&["instances", "i1", "pos"], Some(&json!({ "x": 7.0, "y": 8.0 })));
        assert_eq!(npos(&doc, "instances", "i1"), Some([7.0, 8.0]));
    }

    #[test]
    fn write_at_never_mints_a_phantom_entity() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());
        // Writing under an ABSENT node/instance is a no-op (the manager owns entity existence).
        doc.write_at(&["nodes", "ghost", "params", "g", "p", "value"], Some(&json!(1.0)));
        assert!(!doc.node_ids().contains(&"ghost".to_string()), "no phantom node minted");
    }
}
