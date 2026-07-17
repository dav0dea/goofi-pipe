//! `GraphDoc` — a typed façade over a `yrs::Doc` holding goofi's control-plane state
//! (nodes + nested params/viewers, links). The manager keeps this in agreement with the
//! engine `Graph`; it is the sync structure clients will later replicate. Pure: depends
//! only on `yrs` + `serde_json`, no engine/payload types.

use std::collections::HashMap;

use yrs::updates::decoder::Decode;
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

/// A link as mirrored into the doc (hex uids + slot names).
#[derive(Clone, Debug, PartialEq)]
pub struct LinkRecord {
    pub node_out: String,
    pub slot_out: String,
    pub node_in: String,
    pub slot_in: String,
}

/// One exposed sub-patch boundary port, as mirrored into the doc.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryRecord {
    pub bnd_id: String,
    pub dir: String,   // "in" | "out"
    pub dtype: String, // "ARRAY" | "STRING" | "TABLE"
    pub name: String,
    pub pos: [f64; 2],
    /// The wired inner leaf (hex uid + slot), or `None` when the boundary is unwired.
    pub inner_node: Option<String>,
    pub inner_slot: Option<String>,
}

/// A sub-patch instance as mirrored into the doc — the manager-written forest (§4.2). Identity +
/// parentage + placement + the local→uid member map + the exposed interface. Runtime-derived
/// fields (error, siblings, resolved slot types) stay on the event/runtime plane.
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceRecord {
    pub name: String,
    /// The shared def id (hex) when the instance is shared; `None` when unique.
    pub def_id: Option<String>,
    /// Parent scope (hex uid, or "__root__" for a top-level instance).
    pub parent: String,
    pub pos: [f64; 2],
    /// local name → live member uid (hex). The SSOT for member identity.
    pub members: Vec<(String, String)>,
    pub interface: Vec<BoundaryRecord>,
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

/// Idempotent scalar write: only touch `key` when its stored value actually differs. Every
/// re-mirror (which re-asserts the whole graph after each op) then produces zero ops for an
/// unchanged field — no churn and, load-bearing, no concurrent competing write that could race a
/// client's edit (the same discipline as the in-place [`GraphDoc::set_param`]).
fn set_scalar_if_changed(map: &MapRef, txn: &mut yrs::TransactionMut, key: &str, v: &serde_json::Value) {
    if !scalar_unchanged(map, txn, key, v) {
        insert_scalar(map, txn, key, v);
    }
}

/// Idempotent `[x, y]` write onto a nested `pos` map.
fn set_pos_if_changed(parent: &MapRef, txn: &mut yrs::TransactionMut, pos: [f64; 2]) {
    let m = get_or_insert_map(parent, txn, "pos");
    set_scalar_if_changed(&m, txn, "x", &serde_json::json!(pos[0]));
    set_scalar_if_changed(&m, txn, "y", &serde_json::json!(pos[1]));
}

/// Remove every key of `map` not present in `keep` (used to prune stale members/boundaries when
/// a re-mirror shrinks a collection).
fn retain_keys<T: ReadTxn>(map: &MapRef, txn_keys: &T, keep: &std::collections::HashSet<String>) -> Vec<String> {
    map.keys(txn_keys).map(|k| k.to_string()).filter(|k| !keep.contains(k)).collect()
}

/// The control-plane document. `nodes` is a Map<uid, {type, name, pos, params, viewers}>,
/// `links` an Array of {node_out, slot_out, node_in, slot_in}, and `instances` the sub-patch
/// forest — a Map<uid, {name, def_id?, parent, pos, members:Map<local,uid>, interface:Map<bnd,…>}>.
pub struct GraphDoc {
    doc: Doc,
    nodes: MapRef,
    links: ArrayRef,
    instances: MapRef,
}

impl GraphDoc {
    pub fn new() -> GraphDoc {
        let doc = Doc::new();
        let nodes = doc.get_or_insert_map("nodes");
        let links = doc.get_or_insert_array("links");
        let instances = doc.get_or_insert_map("instances");
        GraphDoc { doc, nodes, links, instances }
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
        // Get-or-insert a STABLE entry map — never replace it. Replacing minted a fresh nested
        // map on every write, so the manager's blanket re-mirror and a client's direct leaf write
        // became competing same-key insertions in the CRDT; the winner was decided by map/client
        // identity, not recency, and could silently drop a concurrent client update. Writing the
        // scalar fields in place instead keeps one entry per param, and each writer's edits form a
        // single causal chain that converges.
        let entry = get_or_insert_map(&g, &mut txn, name);
        // Idempotent: only touch a field when it actually changed. A re-mirror of an unchanged
        // param then produces ZERO ops — so it never manufactures a concurrent write that could
        // race (and lose to) a client's in-flight edit, and the doc does not churn/grow per tick.
        if !scalar_unchanged(&entry, &txn, "value", value) {
            insert_scalar(&entry, &mut txn, "value", value);
        }
        match expr {
            Some(e) => {
                let ex = get_or_insert_map(&entry, &mut txn, "expr");
                if !scalar_unchanged(&ex, &txn, "source", &serde_json::json!(e.source)) {
                    ex.insert(&mut txn, "source", e.source.as_str());
                }
                if !scalar_unchanged(&ex, &txn, "enabled", &serde_json::json!(e.enabled)) {
                    ex.insert(&mut txn, "enabled", e.enabled);
                }
                if !scalar_unchanged(&ex, &txn, "triggers", &serde_json::json!(e.triggers)) {
                    ex.insert(&mut txn, "triggers", e.triggers);
                }
            }
            // Clearing an expression drops the whole sub-map (only when one is present).
            None => {
                if entry.get(&txn, "expr").is_some() {
                    entry.remove(&mut txn, "expr");
                }
            }
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

    pub fn remove_node(&mut self, uid: &str) {
        let mut txn = self.doc.transact_mut();
        self.nodes.remove(&mut txn, uid);
    }

    /// The uids of all sub-patch instances currently in the doc.
    pub fn instance_ids(&self) -> Vec<String> {
        let txn = self.doc.transact();
        self.instances.keys(&txn).map(|k| k.to_string()).collect()
    }

    /// Insert or update a sub-patch instance's mirror record, IN PLACE (stable maps, per-field
    /// skip-if-unchanged), so a re-mirror of an unchanged instance produces zero ops — same
    /// idempotency discipline as [`Self::set_param`], and required for the same reason (the
    /// mirror re-asserts the whole forest after every op).
    pub fn upsert_instance(&mut self, uid: &str, rec: &InstanceRecord) {
        use std::collections::HashSet;
        let mut txn = self.doc.transact_mut();
        let inst = match self.instances.get(&txn, uid).and_then(|v| v.cast::<MapRef>().ok()) {
            Some(m) => m,
            None => self.instances.insert(&mut txn, uid, MapPrelim::default()),
        };
        set_scalar_if_changed(&inst, &mut txn, "name", &serde_json::json!(rec.name));
        set_scalar_if_changed(&inst, &mut txn, "parent", &serde_json::json!(rec.parent));
        match &rec.def_id {
            Some(d) => set_scalar_if_changed(&inst, &mut txn, "def_id", &serde_json::json!(d)),
            None if inst.get(&txn, "def_id").is_some() => {
                inst.remove(&mut txn, "def_id");
            }
            None => {}
        }
        set_pos_if_changed(&inst, &mut txn, rec.pos);

        // members: Map<local, uid> — set-if-changed, prune stale locals.
        let members = get_or_insert_map(&inst, &mut txn, "members");
        let keep: HashSet<String> = rec.members.iter().map(|(l, _)| l.clone()).collect();
        for k in retain_keys(&members, &txn, &keep) {
            members.remove(&mut txn, &k);
        }
        for (local, muid) in &rec.members {
            set_scalar_if_changed(&members, &mut txn, local, &serde_json::json!(muid));
        }

        // interface: Map<bnd_id, {dir, dtype, name, pos, inner_node?, inner_slot?}>.
        let iface = get_or_insert_map(&inst, &mut txn, "interface");
        let keep_b: HashSet<String> = rec.interface.iter().map(|b| b.bnd_id.clone()).collect();
        for k in retain_keys(&iface, &txn, &keep_b) {
            iface.remove(&mut txn, &k);
        }
        for b in &rec.interface {
            let bm = get_or_insert_map(&iface, &mut txn, &b.bnd_id);
            set_scalar_if_changed(&bm, &mut txn, "dir", &serde_json::json!(b.dir));
            set_scalar_if_changed(&bm, &mut txn, "dtype", &serde_json::json!(b.dtype));
            set_scalar_if_changed(&bm, &mut txn, "name", &serde_json::json!(b.name));
            set_pos_if_changed(&bm, &mut txn, b.pos);
            match &b.inner_node {
                Some(n) => set_scalar_if_changed(&bm, &mut txn, "inner_node", &serde_json::json!(n)),
                None if bm.get(&txn, "inner_node").is_some() => {
                    bm.remove(&mut txn, "inner_node");
                }
                None => {}
            }
            match &b.inner_slot {
                Some(s) => set_scalar_if_changed(&bm, &mut txn, "inner_slot", &serde_json::json!(s)),
                None if bm.get(&txn, "inner_slot").is_some() => {
                    bm.remove(&mut txn, "inner_slot");
                }
                None => {}
            }
        }
    }

    pub fn remove_instance(&mut self, uid: &str) {
        let mut txn = self.doc.transact_mut();
        self.instances.remove(&mut txn, uid);
    }

    /// Read back a mirrored instance record (for tests / a manager-side reader).
    pub fn instance_record(&self, uid: &str) -> Option<InstanceRecord> {
        let txn = self.doc.transact();
        let inst = self.instances.get(&txn, uid).and_then(|v| v.cast::<MapRef>().ok())?;
        let s = |m: &MapRef, k: &str| match m.get(&txn, k) {
            Some(Out::Any(Any::String(v))) => Some(v.to_string()),
            _ => None,
        };
        let pos_of = |m: &MapRef| -> [f64; 2] {
            let p = m.get(&txn, "pos").and_then(|v| v.cast::<MapRef>().ok());
            let f = |k: &str| match p.as_ref().and_then(|pm| pm.get(&txn, k)) {
                Some(Out::Any(Any::Number(n))) => n,
                _ => 0.0,
            };
            [f("x"), f("y")]
        };
        let mut members = Vec::new();
        if let Some(mm) = inst.get(&txn, "members").and_then(|v| v.cast::<MapRef>().ok()) {
            for local in mm.keys(&txn) {
                if let Some(uid) = s(&mm, local) {
                    members.push((local.to_string(), uid));
                }
            }
        }
        let mut interface = Vec::new();
        if let Some(im) = inst.get(&txn, "interface").and_then(|v| v.cast::<MapRef>().ok()) {
            for bnd in im.keys(&txn) {
                if let Some(bm) = im.get(&txn, bnd).and_then(|v| v.cast::<MapRef>().ok()) {
                    interface.push(BoundaryRecord {
                        bnd_id: bnd.to_string(),
                        dir: s(&bm, "dir").unwrap_or_default(),
                        dtype: s(&bm, "dtype").unwrap_or_default(),
                        name: s(&bm, "name").unwrap_or_default(),
                        pos: pos_of(&bm),
                        inner_node: s(&bm, "inner_node"),
                        inner_slot: s(&bm, "inner_slot"),
                    });
                }
            }
        }
        Some(InstanceRecord {
            name: s(&inst, "name").unwrap_or_default(),
            def_id: s(&inst, "def_id"),
            parent: s(&inst, "parent").unwrap_or_default(),
            pos: pos_of(&inst),
            members,
            interface,
        })
    }

    /// The full document state as a v1 update (what a joining client would receive).
    pub fn encode_state(&self) -> Vec<u8> {
        let txn = self.doc.transact();
        txn.encode_state_as_update_v1(&yrs::StateVector::default())
    }

    /// Build a fresh doc and apply a v1 update (state) to it.
    pub fn from_update(update: &[u8]) -> GraphDoc {
        let this = GraphDoc::new();
        {
            let mut txn = this.doc.transact_mut();
            let u = yrs::Update::decode_v1(update).expect("valid v1 update");
            txn.apply_update(u).expect("apply update");
        }
        this
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

    /// Every param value in the doc, as `(uid, group, name, value)` — the snapshot the
    /// client-update diff is computed against. Order is doc key order (deterministic).
    pub fn param_snapshot(&self) -> Vec<(String, String, String, serde_json::Value)> {
        let txn = self.doc.transact();
        let mut out = Vec::new();
        for uid in self.nodes.keys(&txn) {
            let Some(node) = self.nodes.get(&txn, uid).and_then(|v| v.cast::<MapRef>().ok()) else {
                continue;
            };
            let Some(params) = node.get(&txn, "params").and_then(|v| v.cast::<MapRef>().ok()) else {
                continue;
            };
            for group in params.keys(&txn) {
                let Some(g) = params.get(&txn, group).and_then(|v| v.cast::<MapRef>().ok()) else {
                    continue;
                };
                for name in g.keys(&txn) {
                    if let Some(v) = self.param_value(uid, group, name) {
                        out.push((uid.to_string(), group.to_string(), name.to_string(), v));
                    }
                }
            }
        }
        out
    }

    /// Apply a client's incremental update to this replica and return the param leaves whose
    /// value changed — so the manager can push exactly those into the engine `Graph`. The
    /// diff is loop-safe: the manager's subsequent graph→doc re-mirror writes the same values,
    /// which yrs records as no change. `Err` only if the update bytes are malformed.
    pub fn apply_client_update(
        &mut self,
        update: &[u8],
    ) -> Result<Vec<(String, String, String, serde_json::Value)>, String> {
        let before: HashMap<(String, String, String), serde_json::Value> = self
            .param_snapshot()
            .into_iter()
            .map(|(u, g, n, v)| ((u, g, n), v))
            .collect();
        self.apply_update(update)?;
        let changed = self
            .param_snapshot()
            .into_iter()
            .filter(|(u, g, n, v)| before.get(&(u.clone(), g.clone(), n.clone())) != Some(v))
            .collect();
        Ok(changed)
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

    #[test]
    fn a_fresh_doc_has_no_nodes() {
        let doc = GraphDoc::new();
        assert!(doc.node_ids().is_empty());
    }

    #[test]
    fn set_param_is_idempotent_and_writes_in_place() {
        // The re-mirror re-asserts every param after every op. Re-setting a param to its CURRENT
        // value must produce NO doc ops (an empty diff) — otherwise the manager's blanket
        // re-mirror manufactures a concurrent write on the value key that can race, and lose to,
        // a client's in-flight edit (the concurrent-leaf-write lost-update bug). It must also not
        // grow the doc unboundedly per tick.
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.upsert_node("000000000001", "Oscillator", "osc", [0.0, 0.0]);
        doc.set_param("000000000001", "common", "max_frequency", &json!(30.0), None);

        let sv = doc.state_vector();
        // Re-assert the SAME value (what a re-mirror does): no change ⇒ empty diff.
        doc.set_param("000000000001", "common", "max_frequency", &json!(30.0), None);
        assert!(
            doc.is_empty_diff(&doc.diff(&sv)),
            "re-setting a param to its current value must be a no-op"
        );
        // An i64 that equals the stored f64 must also be treated as unchanged.
        doc.set_param("000000000001", "common", "max_frequency", &json!(30), None);
        assert!(
            doc.is_empty_diff(&doc.diff(&sv)),
            "an int equal to the stored float is unchanged"
        );
        // A genuine change is still applied.
        doc.set_param("000000000001", "common", "max_frequency", &json!(42.0), None);
        assert!(!doc.is_empty_diff(&doc.diff(&sv)), "a real change produces a delta");
        assert_eq!(
            doc.param_value("000000000001", "common", "max_frequency"),
            Some(json!(42.0))
        );
    }

    fn sample_instance() -> InstanceRecord {
        InstanceRecord {
            name: "subpatch0".into(),
            def_id: Some("00000000000000aa".into()),
            parent: "__root__".into(),
            pos: [10.0, 20.0],
            members: vec![("buffer0".into(), "000000000001".into()), ("osc0".into(), "000000000002".into())],
            interface: vec![BoundaryRecord {
                bnd_id: "out0".into(),
                dir: "out".into(),
                dtype: "ARRAY".into(),
                name: "wave".into(),
                pos: [1.0, 2.0],
                inner_node: Some("000000000001".into()),
                inner_slot: Some("out".into()),
            }],
        }
    }

    #[test]
    fn upsert_instance_round_trips_the_forest_record() {
        let mut doc = GraphDoc::new();
        let rec = sample_instance();
        doc.upsert_instance("00000000000000f0", &rec);
        assert_eq!(doc.instance_ids(), vec!["00000000000000f0"]);
        // members/interface are read back from Y.Maps (key order not guaranteed), so normalize by
        // sorting before comparing — consumers key them by name/bnd_id, not position.
        let mut got = doc.instance_record("00000000000000f0").expect("mirrored");
        got.members.sort();
        got.interface.sort_by(|a, b| a.bnd_id.cmp(&b.bnd_id));
        let mut want = rec;
        want.members.sort();
        want.interface.sort_by(|a, b| a.bnd_id.cmp(&b.bnd_id));
        assert_eq!(got, want);
        assert_eq!(doc.instance_record("nope"), None);
    }

    #[test]
    fn upsert_instance_is_idempotent_and_prunes_removed_members() {
        // The forest mirror re-asserts every instance after every op, so re-upserting an
        // unchanged record must produce NO doc ops (empty diff) — same discipline as set_param.
        let mut doc = GraphDoc::new();
        doc.upsert_instance("00000000000000f0", &sample_instance());
        let sv = doc.state_vector();
        doc.upsert_instance("00000000000000f0", &sample_instance());
        assert!(doc.is_empty_diff(&doc.diff(&sv)), "re-upserting an unchanged instance is a no-op");

        // Dropping a member + unwiring the boundary prunes stale keys and updates in place.
        let mut shrunk = sample_instance();
        shrunk.members.pop(); // remove osc0
        shrunk.interface[0].inner_node = None;
        shrunk.interface[0].inner_slot = None;
        doc.upsert_instance("00000000000000f0", &shrunk);
        let got = doc.instance_record("00000000000000f0").unwrap();
        assert_eq!(got.members.len(), 1, "stale member pruned");
        assert_eq!(got.members[0].0, "buffer0");
        assert_eq!(got.interface[0].inner_node, None, "boundary unwired in place");

        doc.remove_instance("00000000000000f0");
        assert!(doc.instance_ids().is_empty());
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
    fn re_setting_a_param_without_an_expr_prunes_the_old_binding() {
        // The mirror re-syncs a param by re-calling set_param. When a binding is cleared, the
        // whole entry is replaced, so the stale `expr` must NOT linger in the doc (else a
        // cleared expression would resurrect on a client). Guards the re-sync prune contract.
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);
        doc.set_param(
            "1", "g", "p", &json!(1.0),
            Some(ExprRecord { source: "nd('x')".into(), enabled: true, triggers: false }),
        );
        assert_eq!(doc.param_expr_source("1", "g", "p").as_deref(), Some("nd('x')"));
        doc.set_param("1", "g", "p", &json!(2.0), None);
        assert_eq!(doc.param_value("1", "g", "p"), Some(json!(2.0)), "value updated");
        assert_eq!(doc.param_expr_source("1", "g", "p"), None, "stale binding pruned");
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

    #[test]
    fn remove_node_and_state_round_trip() {
        let mut doc = GraphDoc::new();
        doc.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);
        doc.upsert_node("2", "Buffer", "buf", [1.0, 2.0]);
        doc.remove_node("1");
        assert_eq!(doc.node_ids(), vec!["2"]);

        let bytes = doc.encode_state();
        let copy = GraphDoc::from_update(&bytes);
        assert_eq!(copy.node_ids(), vec!["2"]);
        assert_eq!(copy.node_name("2").as_deref(), Some("buf"));
    }

    #[test]
    fn sync_diff_converges_two_replicas() {
        // The relay handshake: a peer advertises its state vector, the other returns a diff,
        // the peer applies it and converges — the primitive the /control sync relay uses.
        let mut server = GraphDoc::new();
        server.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);

        let client = GraphDoc::new(); // empty replica just joined
        let diff = server.diff(&client.state_vector());
        let mut client = client;
        client.apply_update(&diff).unwrap();
        assert_eq!(client.node_name("1").as_deref(), Some("osc"), "client converged via diff");

        // A later incremental edit on the server produces a small diff the client applies.
        server.upsert_node("1", "Oscillator", "osc2", [0.0, 0.0]);
        let diff2 = server.diff(&client.state_vector());
        client.apply_update(&diff2).unwrap();
        assert_eq!(client.node_name("1").as_deref(), Some("osc2"));
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
        // The symmetric handshake: each side sends its SV on connect; receiving a peer's SV
        // yields an Update carrying what the peer lacks; receiving an Update applies it.
        let mut server = GraphDoc::new();
        server.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);
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
        assert_eq!(client.node_name("1").as_deref(), Some("osc"), "client converged via on_sync");

        // A live server edit, relayed as one Update, lands on the client.
        server.upsert_node("2", "Buffer", "buf", [0.0, 0.0]);
        let live = server.diff(&client.state_vector());
        client.on_sync(SyncMsg::Update(live));
        assert_eq!(client.node_name("2").as_deref(), Some("buf"));
    }

    #[test]
    fn is_empty_diff_detects_no_op_deltas() {
        let mut doc = GraphDoc::new();
        doc.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);
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
        server.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);
        server.set_param("1", "common", "max_frequency", &json!(10.0), None);
        server.set_param("1", "oscillator", "amplitude", &json!(1.0), None);

        // A client replica syncs, then edits ONE param locally, producing an update.
        let mut client = GraphDoc::new();
        client.apply_update(&server.diff(&client.state_vector())).unwrap();
        client.set_param("1", "common", "max_frequency", &json!(25.0), None);
        let update = client.diff(&server.state_vector());

        // The manager applies it and is told precisely what changed.
        let changed = server.apply_client_update(&update).unwrap();
        assert_eq!(changed, vec![("1".into(), "common".into(), "max_frequency".into(), json!(25.0))]);
        assert_eq!(server.param_value("1", "common", "max_frequency"), Some(json!(25.0)), "doc updated");
        assert_eq!(server.param_value("1", "oscillator", "amplitude"), Some(json!(1.0)), "untouched param unchanged");

        // Re-applying the SAME update reports no further changes (idempotent, no phantom loop).
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
        server.upsert_node("1", "Oscillator", "osc", [0.0, 0.0]);

        let mut client = GraphDoc::new();
        client.apply_update(&server.diff(&client.state_vector())).unwrap();
        assert_eq!(client.node_ids(), vec!["1"], "client synced node 1");

        // The client now MISSES everything below (dropped deltas): a new node, a param edit,
        // and a rename of node 1 (a struct that chains off the earlier one).
        server.upsert_node("2", "Buffer", "buf", [0.0, 0.0]);
        server.set_param("1", "common", "max_frequency", &json!(50.0), None);
        server.upsert_node("1", "Oscillator", "renamed", [0.0, 0.0]);

        // Recovery: apply the framed full state. Convergence, not divergence.
        let SyncMsg::Update(full) = SyncMsg::decode(&server.full_state_frame()).unwrap() else {
            panic!("full_state_frame is an Update");
        };
        client.apply_update(&full).unwrap();
        assert_eq!(client.node_ids().len(), 2, "gapped node arrived");
        assert_eq!(client.node_name("1").as_deref(), Some("renamed"), "dependent change resolved");
        assert_eq!(client.node_name("2").as_deref(), Some("buf"));
        assert_eq!(client.param_value("1", "common", "max_frequency"), Some(json!(50.0)));
    }
}
