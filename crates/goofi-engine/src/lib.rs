//! goofi-engine — the graph + a minimal single-threaded tick scheduler (M1).
//!
//! Grows into the work-stealing compute pool + reserved RT sub-pool + timer-wheel
//! autotrigger in M2. For now: instantiate catalog nodes, wire one-wire-per-input
//! links, and `tick()` all nodes once in topological order, moving each node's
//! outputs into its consumers' inputs (latest-wins) so a single pass propagates
//! through an acyclic graph. Each node's latest output frame is exposed for the
//! data plane.

use std::collections::HashMap;
use std::time::Instant;

use goofi_core::{Data, Param};
use goofi_node::{Inputs, NodeCtx, NodeManifest, Outputs, ParamGroups, ParamKey, RunPolicy};
use indexmap::IndexMap;
use rayon::prelude::*;

/// A stable node identity. Encoded as a 12-hex string for the `.gfi` / frontend
/// (the same key those use), a `u64` internally.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Uid(pub u64);

impl Uid {
    pub fn to_hex(self) -> String {
        format!("{:012x}", self.0)
    }
    pub fn from_hex(s: &str) -> Option<Uid> {
        u64::from_str_radix(s, 16).ok().map(Uid)
    }
}

impl std::fmt::Display for Uid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// EMA weight for the measured update-frequency (`ufreq`). Smooths the inter-emit
/// interval: time-constant ≈ `1/α` emits, so a steady slot reads exact from its 2nd
/// emit and a jittery one settles within ~10–15. Tunable in this one place.
const UFREQ_EMA_ALPHA: f64 = 0.2;

/// Per-output-slot measured emit-rate state (see [`stamp_meta`]). Tracks the
/// wall-clock (`ctx.now`) of the previous emit and the smoothed inter-emit
/// interval; `ufreq = 1/ema`. `ema == None` until the second emit gives one
/// interval to seed it.
struct UfreqMeter {
    last_emit: f64,
    ema: Option<f64>,
}

struct NodeEntry {
    manifest: &'static NodeManifest,
    node: Box<dyn goofi_node::Node>,
    params: ParamGroups,
    inputs: IndexMap<&'static str, Option<Data>>,
    outputs: IndexMap<&'static str, Option<Data>>,
    ctx: NodeCtx,
    last_error: Option<String>,
    /// Globally-unique display name (type-numbered), for the frontend/`.gfi`.
    name: String,
    /// Editor position `[x, y]`.
    pos: [f64; 2],
    /// Whether this node has any triggering input (else it free-runs each tick).
    has_trigger_inputs: bool,
    /// Set when a triggering input received a fresh frame; cleared on process.
    trigger_pending: bool,
    /// Per-output-slot source-origin emit counter for `meta["index"]`. Advanced
    /// only when a slot's frame starts a *fresh* timeline (a generator, or a
    /// length-changing transform); a length-preserving emit mirrors its matching
    /// input's index instead. Engine-owned — the node never sees it.
    index_counters: HashMap<&'static str, u64>,
    /// Per-output-slot measured update-rate state for `meta["ufreq"]`. Engine-owned;
    /// advanced only when a slot actually emits, so it tracks that slot's true cadence.
    ufreq_meters: HashMap<&'static str, UfreqMeter>,
    /// The node's run gate (from its `common` params), consulted each tick.
    run_policy: RunPolicy,
    /// Wall-clock instant the node last ran, for rate-cap gating (`None` = never).
    last_run: Option<Instant>,
}

/// A resolved link (uids + `&'static` slot names), for snapshot projection.
#[derive(Clone, Copy, Debug)]
pub struct LinkView {
    pub node_out: Uid,
    pub slot_out: &'static str,
    pub node_in: Uid,
    pub slot_in: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Link {
    node_out: Uid,
    slot_out: &'static str,
    node_in: Uid,
    slot_in: &'static str,
}

/// Extract a readable message from a caught panic payload.
fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = p.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic in node".to_string()
    }
}

/// The persisted scalar value of a param (flat form; triggers persist `false`).
fn param_value_json(p: &Param) -> serde_json::Value {
    use serde_json::json;
    match p {
        Param::Float { value, .. } => json!(value),
        Param::Int { value, .. } => json!(value),
        Param::Bool { value } => json!(value),
        Param::Trigger { .. } => json!(false),
        Param::Str { value, .. } => json!(value),
    }
}

/// A node factory that can capture runtime state (a Python class handle, a device
/// descriptor). Used for node types discovered at runtime rather than compiled
/// into the `inventory` catalog — a bare `fn` pointer can't close over such state.
pub type NodeFactory = Box<dyn Fn(&ParamGroups) -> Box<dyn goofi_node::Node> + Send + Sync>;

/// A runtime-registered node type: its (leaked-`'static`) manifest plus the
/// factory that builds instances of it. Its `manifest.make` is never called.
struct DynType {
    manifest: &'static NodeManifest,
    factory: NodeFactory,
}

/// The authoritative graph + scheduler.
pub struct Graph {
    nodes: IndexMap<Uid, NodeEntry>,
    links: Vec<Link>,
    next_uid: u64,
    /// Node types registered at runtime (e.g. discovered Python nodes), keyed by
    /// type name. Survives `clear()`/`load_doc` — these are catalog, not content.
    dyn_types: HashMap<&'static str, DynType>,
    /// Wall-clock reference, anchored at the first tick, so `NodeCtx::now` is
    /// seconds-since-start (deterministic under an injected clock).
    start: Option<Instant>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Graph {
        // Reference goofi-nodes so the linker keeps its inventory registrations.
        let _ = goofi_nodes::native_node_count();
        Graph {
            nodes: IndexMap::new(),
            links: Vec::new(),
            next_uid: 1,
            dyn_types: HashMap::new(),
            start: None,
        }
    }

    /// Register a node type discovered at runtime. `manifest` must be `'static`
    /// (runtime types leak one manifest per type — bounded, catalog-lifetime); its
    /// `make` field is unused (instances come from `factory`).
    ///
    /// A name that collides with a built-in catalog type or an already-registered
    /// runtime type is refused (with a warning) rather than silently shadowed or
    /// overwritten — a built-in always wins `add_node`/`load_doc` resolution, and a
    /// blind overwrite would orphan the loser's leaked manifest and make its node
    /// unreachable. Returns whether the type was registered.
    pub fn register_dyn_type(
        &mut self,
        manifest: &'static NodeManifest,
        factory: NodeFactory,
    ) -> bool {
        let name = manifest.type_name;
        if goofi_node::find(name).is_some() {
            eprintln!("warning: runtime node type `{name}` collides with a built-in; ignoring it");
            return false;
        }
        if self.dyn_types.contains_key(name) {
            eprintln!("warning: runtime node type `{name}` already registered; ignoring the duplicate");
            return false;
        }
        self.dyn_types.insert(name, DynType { manifest, factory });
        true
    }

    /// Whether a type name resolves to either the compile-time catalog or a
    /// runtime-registered type.
    fn known_type(&self, type_name: &str) -> bool {
        goofi_node::find(type_name).is_some() || self.dyn_types.contains_key(type_name)
    }

    /// The manifests of all runtime-registered node types, sorted by type name
    /// (the compile-time catalog is enumerated separately via `goofi_node::catalog`).
    /// Used by the bridge to include runtime types in the editor palette.
    pub fn dyn_type_manifests(&self) -> Vec<&'static NodeManifest> {
        let mut ms: Vec<&'static NodeManifest> =
            self.dyn_types.values().map(|dt| dt.manifest).collect();
        ms.sort_by_key(|m| m.type_name);
        ms
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains(&self, uid: Uid) -> bool {
        self.nodes.contains_key(&uid)
    }

    /// Node uids in insertion order.
    pub fn node_uids(&self) -> Vec<Uid> {
        self.nodes.keys().copied().collect()
    }

    pub fn type_name(&self, uid: Uid) -> Option<&'static str> {
        self.nodes.get(&uid).map(|e| e.manifest.type_name)
    }

    pub fn manifest(&self, uid: Uid) -> Option<&'static NodeManifest> {
        self.nodes.get(&uid).map(|e| e.manifest)
    }

    pub fn last_error(&self, uid: Uid) -> Option<&str> {
        self.nodes.get(&uid).and_then(|e| e.last_error.as_deref())
    }

    fn mint(&mut self) -> Uid {
        let u = Uid(self.next_uid);
        self.next_uid += 1;
        u
    }

    /// Instantiate a node by type name (compile-time catalog or a
    /// runtime-registered type). `params` defaults to the type's defaults.
    pub fn add_node(
        &mut self,
        type_name: &str,
        params: Option<ParamGroups>,
    ) -> Result<Uid, String> {
        let (manifest, params, node): (&'static NodeManifest, ParamGroups, Box<dyn goofi_node::Node>) =
            if let Some(m) = goofi_node::find(type_name) {
                let p = goofi_node::with_common(params.unwrap_or_else(|| (m.default_params)()));
                let n = (m.make)(&p);
                (m, p, n)
            } else if let Some(dt) = self.dyn_types.get(type_name) {
                let p = goofi_node::with_common(params.unwrap_or_else(|| (dt.manifest.default_params)()));
                let n = (dt.factory)(&p);
                (dt.manifest, p, n)
            } else {
                return Err(format!("unknown node type `{type_name}`"));
            };
        Ok(self.insert_node(manifest, node, params))
    }

    /// Build a `NodeEntry` from a manifest + a constructed node, run its `setup`,
    /// seed its I/O buffers, assign a fresh name, and insert it. Shared by the
    /// catalog and runtime instantiation paths.
    fn insert_node(
        &mut self,
        manifest: &'static NodeManifest,
        mut node: Box<dyn goofi_node::Node>,
        params: ParamGroups,
    ) -> Uid {
        let mut ctx = NodeCtx::new();
        let last_error = node.setup(&mut ctx).err().map(|e| e.0);

        let inputs: IndexMap<&'static str, Option<Data>> =
            manifest.inputs.iter().map(|s| (s.name, None)).collect();
        let outputs = manifest.output_buffer();

        let name = self.fresh_name(&manifest.type_name.to_lowercase());
        let has_trigger_inputs = manifest.inputs.iter().any(|i| i.trigger_process);
        let run_policy = RunPolicy::from_params(&params);
        let uid = self.mint();
        self.nodes.insert(
            uid,
            NodeEntry {
                manifest,
                node,
                params,
                inputs,
                outputs,
                ctx,
                last_error,
                name,
                pos: [0.0, 0.0],
                has_trigger_inputs,
                trigger_pending: false,
                index_counters: HashMap::new(),
                ufreq_meters: HashMap::new(),
                run_policy,
                last_run: None,
            },
        );
        uid
    }

    /// Lowest `{base}{N}` display name not already in use (globally unique).
    fn fresh_name(&self, base: &str) -> String {
        for n in 0.. {
            let cand = format!("{base}{n}");
            if !self.nodes.values().any(|e| e.name == cand) {
                return cand;
            }
        }
        unreachable!()
    }

    pub fn name(&self, uid: Uid) -> Option<&str> {
        self.nodes.get(&uid).map(|e| e.name.as_str())
    }

    pub fn pos(&self, uid: Uid) -> Option<[f64; 2]> {
        self.nodes.get(&uid).map(|e| e.pos)
    }

    pub fn params(&self, uid: Uid) -> Option<&ParamGroups> {
        self.nodes.get(&uid).map(|e| &e.params)
    }

    pub fn rename_node(&mut self, uid: Uid, name: &str) -> Result<(), String> {
        if self.nodes.values().any(|e| e.name == name) {
            return Err(format!("display name `{name}` already in use"));
        }
        let e = self
            .nodes
            .get_mut(&uid)
            .ok_or_else(|| format!("no such node {uid}"))?;
        e.name = name.to_string();
        Ok(())
    }

    pub fn set_node_pos(&mut self, uid: Uid, pos: [f64; 2]) -> Result<(), String> {
        let e = self
            .nodes
            .get_mut(&uid)
            .ok_or_else(|| format!("no such node {uid}"))?;
        e.pos = pos;
        Ok(())
    }

    /// All links as resolved views (snapshot projection).
    pub fn links_view(&self) -> Vec<LinkView> {
        self.links
            .iter()
            .map(|l| LinkView {
                node_out: l.node_out,
                slot_out: l.slot_out,
                node_in: l.node_in,
                slot_in: l.slot_in,
            })
            .collect()
    }

    pub fn remove_node(&mut self, uid: Uid) -> Result<(), String> {
        if self.nodes.shift_remove(&uid).is_none() {
            return Err(format!("no such node {uid}"));
        }
        // Drop links touching the node; clear any downstream input it fed.
        let dropped: Vec<Link> = self
            .links
            .iter()
            .filter(|l| l.node_out == uid || l.node_in == uid)
            .cloned()
            .collect();
        self.links
            .retain(|l| l.node_out != uid && l.node_in != uid);
        for l in dropped {
            self.clear_input(l.node_in, l.slot_in);
        }
        Ok(())
    }

    pub fn update_param(
        &mut self,
        uid: Uid,
        group: &str,
        name: &str,
        value: Param,
    ) -> Result<(), String> {
        let entry = self
            .nodes
            .get_mut(&uid)
            .ok_or_else(|| format!("no such node {uid}"))?;
        if let Some(g) = entry.params.get_mut(group) {
            g.insert(name.to_string(), value.clone());
        } else {
            return Err(format!("no such param group `{group}`"));
        }
        // The `common` group is scheduler metadata, not a node param — re-derive
        // the cached run gate rather than dispatching it to the node.
        if group == "common" {
            entry.run_policy = RunPolicy::from_params(&entry.params);
            return Ok(());
        }
        entry
            .node
            .on_param_changed(&ParamKey::new(group, name), &value)
            .map_err(|e| e.0)
    }

    /// Resolve an output slot name to its `&'static` manifest name.
    fn resolve_output(&self, uid: Uid, slot: &str) -> Option<&'static str> {
        let e = self.nodes.get(&uid)?;
        e.manifest.outputs.iter().find(|o| o.name == slot).map(|o| o.name)
    }
    fn resolve_input(&self, uid: Uid, slot: &str) -> Option<&'static str> {
        let e = self.nodes.get(&uid)?;
        e.manifest.inputs.iter().find(|i| i.name == slot).map(|i| i.name)
    }

    pub fn add_link(
        &mut self,
        node_out: Uid,
        slot_out: &str,
        node_in: Uid,
        slot_in: &str,
    ) -> Result<(), String> {
        let slot_out = self
            .resolve_output(node_out, slot_out)
            .ok_or_else(|| format!("no output slot `{slot_out}` on {node_out}"))?;
        let slot_in = self
            .resolve_input(node_in, slot_in)
            .ok_or_else(|| format!("no input slot `{slot_in}` on {node_in}"))?;

        let new = Link {
            node_out,
            slot_out,
            node_in,
            slot_in,
        };
        if self.links.contains(&new) {
            return Ok(()); // idempotent
        }
        // One wire per input: evict any prior source of this (node_in, slot_in).
        self.links
            .retain(|l| !(l.node_in == node_in && l.slot_in == slot_in));
        self.clear_input(node_in, slot_in);
        self.links.push(new);
        Ok(())
    }

    pub fn remove_link(
        &mut self,
        node_out: Uid,
        slot_out: &str,
        node_in: Uid,
        slot_in: &str,
    ) -> Result<(), String> {
        let before = self.links.len();
        self.links.retain(|l| {
            !(l.node_out == node_out
                && l.slot_out == slot_out
                && l.node_in == node_in
                && l.slot_in == slot_in)
        });
        if self.links.len() == before {
            return Err("no such link".into());
        }
        self.clear_input(node_in, slot_in);
        Ok(())
    }

    fn clear_input(&mut self, uid: Uid, slot: &str) {
        if let Some(e) = self.nodes.get_mut(&uid) {
            if let Some(s) = e.inputs.get_mut(slot) {
                *s = None;
            }
        }
    }

    /// The latest output frame on `(uid, slot)`, if any (data plane read).
    pub fn latest_frame(&self, uid: Uid, slot: &str) -> Option<Data> {
        self.nodes
            .get(&uid)
            .and_then(|e| e.outputs.get(slot))
            .cloned()
            .flatten()
    }

    /// Remove all nodes and links.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.links.clear();
    }

    fn force_set_name(&mut self, uid: Uid, name: &str) {
        if let Some(e) = self.nodes.get_mut(&uid) {
            e.name = name.to_string();
        }
    }

    fn set_param_from_json(&mut self, uid: Uid, group: &str, name: &str, val: &serde_json::Value) {
        let existing = self
            .nodes
            .get(&uid)
            .and_then(|e| goofi_node::param(&e.params, group, name))
            .cloned();
        let Some(existing) = existing else {
            return;
        };
        let newp = match existing {
            Param::Float { vmin, vmax, .. } => Param::Float {
                value: val.as_f64().unwrap_or(0.0),
                vmin,
                vmax,
            },
            Param::Int { vmin, vmax, .. } => Param::Int {
                value: val.as_i64().unwrap_or(0),
                vmin,
                vmax,
            },
            Param::Bool { .. } => Param::Bool {
                value: val.as_bool().unwrap_or(false),
            },
            Param::Trigger { .. } => Param::Trigger { fired: false },
            Param::Str {
                options, refresh, ..
            } => Param::Str {
                value: val.as_str().unwrap_or("").to_string(),
                options,
                refresh,
            },
        };
        let _ = self.update_param(uid, group, name, newp);
    }

    /// Serialize the graph to a `.gfi` v3 document (YAML text).
    pub fn serialize(&self) -> String {
        use serde_json::{json, Map, Value};
        let mut nodes = Map::new();
        for uid in self.node_uids() {
            let e = &self.nodes[&uid];
            let mut params = Map::new();
            for (group, names) in &e.params {
                let mut gmap = Map::new();
                for (name, p) in names {
                    gmap.insert(name.clone(), param_value_json(p));
                }
                params.insert(group.clone(), Value::Object(gmap));
            }
            nodes.insert(
                uid.to_hex(),
                json!({ "type": e.manifest.type_name, "name": e.name, "pos": e.pos, "params": Value::Object(params) }),
            );
        }
        let links: Vec<Value> = self
            .links
            .iter()
            .map(|l| json!([l.node_out.to_hex(), l.slot_out, l.node_in.to_hex(), l.slot_in]))
            .collect();
        let doc = json!({ "version": 3, "nodes": Value::Object(nodes), "links": links });
        serde_yaml_ng::to_string(&doc).unwrap_or_default()
    }

    /// Replace the graph from a `.gfi` v3 document. Node types are validated
    /// before the current graph is torn down (a rejected load is a no-op).
    pub fn load_doc(&mut self, text: &str) -> Result<(), String> {
        let doc: serde_json::Value = serde_yaml_ng::from_str(text).map_err(|e| e.to_string())?;
        if doc.get("version").and_then(|v| v.as_i64()) != Some(3) {
            return Err("unsupported .gfi version (expected 3)".into());
        }
        let nodes = doc
            .get("nodes")
            .and_then(|v| v.as_object())
            .ok_or("missing `nodes`")?;
        for rec in nodes.values() {
            let ty = rec.get("type").and_then(|v| v.as_str()).ok_or("node missing `type`")?;
            if !self.known_type(ty) {
                return Err(format!("unknown node type `{ty}`"));
            }
        }

        self.clear();
        let mut idmap: HashMap<String, Uid> = HashMap::new();
        for (old, rec) in nodes {
            let ty = rec["type"].as_str().unwrap();
            let uid = self.add_node(ty, None)?;
            idmap.insert(old.clone(), uid);
            if let Some(name) = rec.get("name").and_then(|v| v.as_str()) {
                self.force_set_name(uid, name);
            }
            if let Some(p) = rec.get("pos").and_then(|v| v.as_array()) {
                if p.len() == 2 {
                    if let (Some(x), Some(y)) = (p[0].as_f64(), p[1].as_f64()) {
                        let _ = self.set_node_pos(uid, [x, y]);
                    }
                }
            }
            if let Some(groups) = rec.get("params").and_then(|v| v.as_object()) {
                for (group, names) in groups {
                    if let Some(nm) = names.as_object() {
                        for (name, val) in nm {
                            self.set_param_from_json(uid, group, name, val);
                        }
                    }
                }
            }
        }
        if let Some(links) = doc.get("links").and_then(|v| v.as_array()) {
            for l in links {
                if let Some(a) = l.as_array() {
                    if a.len() == 4 {
                        let no = a[0].as_str().and_then(|s| idmap.get(s)).copied();
                        let ni = a[2].as_str().and_then(|s| idmap.get(s)).copied();
                        if let (Some(no), Some(ni)) = (no, ni) {
                            let _ = self.add_link(
                                no,
                                a[1].as_str().unwrap_or(""),
                                ni,
                                a[3].as_str().unwrap_or(""),
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// BFS topological layering (producers before consumers). Each returned level
    /// is a set of mutually-independent nodes — no edges run between them — and
    /// every node's predecessors lie in strictly earlier levels. Nodes trapped in
    /// a cycle form a final level (latest-wins tolerates their back-edges). This
    /// is what lets a level's nodes run concurrently while the graph as a whole
    /// still propagates end-to-end in a single tick.
    fn topo_levels(&self) -> Vec<Vec<Uid>> {
        let mut indeg: HashMap<Uid, usize> = self.nodes.keys().map(|k| (*k, 0)).collect();
        for l in &self.links {
            if self.nodes.contains_key(&l.node_out) && indeg.contains_key(&l.node_in) {
                *indeg.get_mut(&l.node_in).unwrap() += 1;
            }
        }
        let mut levels: Vec<Vec<Uid>> = Vec::new();
        let mut placed: std::collections::HashSet<Uid> = std::collections::HashSet::new();
        // Level 0: insertion-order nodes with no incoming edges.
        let mut current: Vec<Uid> = self
            .nodes
            .keys()
            .copied()
            .filter(|u| indeg[u] == 0)
            .collect();
        while !current.is_empty() {
            for u in &current {
                placed.insert(*u);
            }
            // Relax edges out of this level; a successor whose indegree hits zero
            // joins the next level. Reorder by insertion order for determinism.
            let mut freed: std::collections::HashSet<Uid> = std::collections::HashSet::new();
            for u in &current {
                for l in &self.links {
                    if l.node_out == *u {
                        if let Some(d) = indeg.get_mut(&l.node_in) {
                            if *d > 0 {
                                *d -= 1;
                                if *d == 0 {
                                    freed.insert(l.node_in);
                                }
                            }
                        }
                    }
                }
            }
            levels.push(current);
            current = self
                .nodes
                .keys()
                .copied()
                .filter(|u| freed.contains(u))
                .collect();
        }
        // Any node never freed sits in a cycle; run them together, last.
        let remainder: Vec<Uid> = self
            .nodes
            .keys()
            .copied()
            .filter(|u| !placed.contains(u))
            .collect();
        if !remainder.is_empty() {
            levels.push(remainder);
        }
        levels
    }

    /// The set of nodes with at least one *wired* triggering input — a link feeds a
    /// `trigger_process` input slot. Mirrors Python's `_has_no_triggering_inputs`
    /// (negated): `autotrigger` free-runs a node only when this is empty for it, so
    /// a connected consumer runs on its producer's rate rather than every tick.
    fn wired_trigger_nodes(&self) -> std::collections::HashSet<Uid> {
        self.links
            .iter()
            .filter(|l| {
                self.nodes.get(&l.node_in).is_some_and(|e| {
                    e.manifest
                        .inputs
                        .iter()
                        .any(|i| i.name == l.slot_in && i.trigger_process)
                })
            })
            .map(|l| l.node_in)
            .collect()
    }

    /// Run one tick of the whole graph against the wall clock. See [`Self::tick_at`].
    pub fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    /// Run one tick as of instant `now` (injectable so rate gating is
    /// deterministically testable). Nodes are grouped into topological levels
    /// ([`Self::topo_levels`]); each level's mutually-independent nodes execute
    /// concurrently on the rayon work-stealing pool, then their fresh outputs are
    /// propagated to the next level's inputs before it runs — so an acyclic graph
    /// still propagates end-to-end within a single tick. A node runs iff it *wants*
    /// to run — it's a pure source (no triggering inputs), a triggering input
    /// received a fresh frame, or it autotriggers *and has no wired trigger* — AND
    /// its [`RunPolicy`] rate cap has elapsed since it last ran. A skipped node
    /// keeps its outputs. With the default policy (`max_frequency == 0`) the rate
    /// cap is unbounded, so this reduces to pure trigger arbitration.
    fn tick_at(&mut self, now: Instant) {
        // Seconds since the first-ever tick — the monotonic wall clock nodes read.
        let start = *self.start.get_or_insert(now);
        let now_secs = now.duration_since(start).as_secs_f64();
        let wired = self.wired_trigger_nodes();
        let levels = self.topo_levels();
        for level in levels {
            let set: std::collections::HashSet<Uid> = level.iter().copied().collect();

            // Phase A — run every runnable node in this level in parallel. Each
            // closure touches only its own entry (disjoint `&mut`), so there is no
            // shared state and the result is independent of thread scheduling.
            let ran: Vec<Uid> = {
                let batch: Vec<(Uid, &mut NodeEntry)> = self
                    .nodes
                    .iter_mut()
                    .filter(|(uid, e)| {
                        if !set.contains(uid) {
                            return false;
                        }
                        // A pure source free-runs; a fresh trigger fires; autotrigger
                        // free-runs only a node with no *wired* trigger (Python parity).
                        let wants_run = e.trigger_pending
                            || !e.has_trigger_inputs
                            || (e.run_policy.autotrigger && !wired.contains(uid));
                        let since_last = e.last_run.map(|t| now.saturating_duration_since(t).as_secs_f64());
                        e.run_policy.should_run(since_last, wants_run)
                    })
                    .map(|(uid, e)| {
                        e.last_run = Some(now);
                        e.ctx.now = now_secs;
                        (*uid, e)
                    })
                    .collect();
                let ran: Vec<Uid> = batch.iter().map(|(u, _)| *u).collect();
                batch.into_par_iter().for_each(|(_, entry)| run_node(entry));
                ran
            };

            // Phase B — propagate this level's fresh frames to their consumers
            // (serial; one-wire-per-input means each input has a single writer).
            for uid in ran {
                let produced: Vec<(&'static str, Data)> = self.nodes[&uid]
                    .outputs
                    .iter()
                    .filter_map(|(k, v)| v.as_ref().map(|d| (*k, d.clone())))
                    .collect();
                if produced.is_empty() {
                    continue;
                }
                let outgoing: Vec<(&'static str, Uid, &'static str)> = self
                    .links
                    .iter()
                    .filter(|l| l.node_out == uid)
                    .map(|l| (l.slot_out, l.node_in, l.slot_in))
                    .collect();
                for (slot_out, tgt, slot_in) in outgoing {
                    if let Some(d) = produced
                        .iter()
                        .find(|(s, _)| *s == slot_out)
                        .map(|(_, d)| d.clone())
                    {
                        if let Some(te) = self.nodes.get_mut(&tgt) {
                            if let Some(slot) = te.inputs.get_mut(slot_in) {
                                *slot = Some(d);
                            }
                            // A fresh frame on a triggering input wakes the consumer.
                            if te
                                .manifest
                                .inputs
                                .iter()
                                .any(|i| i.name == slot_in && i.trigger_process)
                            {
                                te.trigger_pending = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Run a single node's `process` in place: clear its outputs, tick its context,
/// stamp each emitted frame's continuity index, and capture any error or panic on
/// its error channel. Panic isolation keeps one faulty node from unwinding through
/// the scheduler (and, in the bridge, poisoning the graph mutex). Called from the
/// parallel phase, so it touches only `entry` (index stamping included — the
/// counter and both I/O buffers all live in `entry`, so it stays disjoint).
fn run_node(entry: &mut NodeEntry) {
    entry.trigger_pending = false;
    entry.ctx.tick += 1;
    for v in entry.outputs.values_mut() {
        *v = None;
    }
    let inp = Inputs::new(&entry.inputs);
    let node = &mut entry.node;
    let ctx = &mut entry.ctx;
    let mut out = Outputs::new(&mut entry.outputs);
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| node.process(&inp, &mut out, ctx)));
    entry.last_error = match result {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e.0),
        Err(p) => Some(panic_message(p)),
    };
    stamp_meta(entry);
}

/// The number of frames a `Data` spans — its total element count (numpy `.size`
/// for an array, `len` for a string/table). This, not a static per-slot flag, is
/// the timeline discriminator: a length-preserving transform's output matches its
/// input's frame count; a generator or length-changing transform does not.
fn frame_count(d: &Data) -> usize {
    match d.value() {
        goofi_core::Value::Array(s) => s.shape().iter().product(),
        goofi_core::Value::Str(s) => s.chars().count(),
        goofi_core::Value::Table(m) => m.len(),
    }
}

/// Stamp the engine-owned meta — `index` and `ufreq` — on every frame this node
/// just emitted (the node never touches either).
///
/// **index**: for each output, propagate the index of the SINGLE index-bearing
/// TRIGGERING input whose frame count equals the output's — that input is the same
/// data timeline, so an upstream drop stays visible downstream. A non-triggering
/// (control/reference) input — an oscillator's scalar frequency, say — is never a
/// timeline candidate even if its length happens to match. With zero, or more than
/// one, matching inputs (a generator, a length-changing transform, or an ambiguous
/// fan-in) the slot starts a fresh per-output counter that advances one per emit.
/// Ported from the Python node's `_next_index`/`_propagated_index`.
///
/// **ufreq**: the slot's measured emit rate (Hz) — an EMA of the inter-emit
/// interval keyed on `ctx.now`, `None` until a second emit gives one interval.
/// Advanced only when the slot actually emits, so it is that slot's true cadence
/// (correctly lower than an input's for a rate-capped or dropping transform), and
/// authoritative — overwritten every emit, never inherited from upstream meta.
fn stamp_meta(entry: &mut NodeEntry) {
    // Only triggering inputs carry the data timeline; control inputs are excluded.
    let triggering: std::collections::HashSet<&str> = entry
        .manifest
        .inputs
        .iter()
        .filter(|s| s.trigger_process)
        .map(|s| s.name)
        .collect();
    // Snapshot the index-bearing triggering inputs (index, frame_count) — no borrow held.
    let input_frames: Vec<(u64, usize)> = entry
        .inputs
        .iter()
        .filter(|(name, _)| triggering.contains(*name))
        .filter_map(|(_, o)| o.as_ref())
        .filter_map(|d| d.meta().index.map(|i| (i, frame_count(d))))
        .collect();
    // This tick's wall clock — the timestamp source for the ufreq interval.
    let now = entry.ctx.now;
    // Disjoint field borrows: rewrite outputs while advancing the index counters and
    // ufreq meters.
    let outputs = &mut entry.outputs;
    let counters = &mut entry.index_counters;
    let meters = &mut entry.ufreq_meters;
    for (slot, slot_opt) in outputs.iter_mut() {
        let Some(d) = slot_opt else { continue };
        let of = frame_count(d);
        let mut matched: Option<u64> = None;
        let mut count = 0usize;
        for (idx, f) in &input_frames {
            if *f == of {
                count += 1;
                matched = Some(*idx);
            }
        }
        let index = if count == 1 {
            matched.unwrap()
        } else {
            let c = counters.entry(*slot).or_insert(0);
            let v = *c;
            *c += 1;
            v
        };
        // Measured update rate: EMA of the inter-emit interval, inverted. (Two-step
        // contains/get — the `get_mut`/else get-or-insert form is NLL problem case #3.)
        let ufreq = if meters.contains_key(*slot) {
            let m = meters.get_mut(*slot).unwrap();
            let dt = now - m.last_emit;
            m.last_emit = now;
            if dt > 0.0 {
                let ema = m.ema.map_or(dt, |prev| UFREQ_EMA_ALPHA * dt + (1.0 - UFREQ_EMA_ALPHA) * prev);
                m.ema = Some(ema);
                Some(1.0 / ema)
            } else {
                // Non-advancing clock (same `now`): keep the prior estimate, no divide-by-zero.
                m.ema.map(|e| 1.0 / e)
            }
        } else {
            // First emit on this slot: record the timestamp, no interval to measure yet.
            meters.insert(*slot, UfreqMeter { last_emit: now, ema: None });
            None
        };
        *d = d.with_stamps(index, ufreq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{DType, Meta, SlotType, Value};
    use goofi_node::{
        Isolation, Node, NodeManifest, NodeResult, OutputDecl, SlotDecl,
    };

    // A test-only passthrough node (ARRAY "in" -> ARRAY "out") to exercise links.
    struct Echo;
    impl Node for Echo {
        fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            if let Some(d) = inp.get("in") {
                out.set("out", d.clone());
            }
            Ok(())
        }
    }
    fn echo_params() -> ParamGroups {
        ParamGroups::new()
    }
    fn echo_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(Echo)
    }
    static E_IN: &[SlotDecl] = &[SlotDecl {
        name: "in",
        kind: SlotType::Array,
        trigger_process: true,
    }];
    static E_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestEcho",
            category: "test",
            doc: "test passthrough",
            inputs: E_IN,
            outputs: E_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: echo_make,
        }
    }

    // A source that only emits on every other run (to exercise trigger arbitration).
    struct GatedSource {
        n: i64,
    }
    impl Node for GatedSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            let emit = self.n % 2 == 0;
            self.n += 1;
            if emit {
                let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                    .map_err(|e| e.to_string())?;
                out.set("out", d);
            }
            Ok(())
        }
    }
    fn gated_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(GatedSource { n: 0 })
    }
    static G_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestGated",
            category: "test",
            doc: "gated source",
            inputs: &[],
            outputs: G_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: gated_make,
        }
    }

    // A triggered node that counts the number of times it actually ran.
    struct Counter {
        runs: i64,
    }
    impl Node for Counter {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            self.runs += 1;
            let d = Data::from_array_bytes(DType::F32, vec![1], (self.runs as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    fn counter_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(Counter { runs: 0 })
    }
    static C_IN: &[SlotDecl] = &[SlotDecl {
        name: "in",
        kind: SlotType::Array,
        trigger_process: true,
    }];
    static C_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestCounter",
            category: "test",
            doc: "run counter",
            inputs: C_IN,
            outputs: C_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: counter_make,
        }
    }

    // A two-input node summing a[0]+b[0] — exercises fan-in convergence, where a
    // consumer at a later level must receive fresh frames from two producers that
    // ran (in parallel) at the same earlier level.
    struct Adder;
    impl Node for Adder {
        fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            let (Some(a), Some(b)) = (inp.get("a"), inp.get("b")) else {
                return Ok(());
            };
            let sum = first_f32(a) + first_f32(b);
            let d = Data::from_array_bytes(DType::F32, vec![1], sum.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    fn adder_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(Adder)
    }
    static ADD_IN: &[SlotDecl] = &[
        SlotDecl { name: "a", kind: SlotType::Array, trigger_process: true },
        SlotDecl { name: "b", kind: SlotType::Array, trigger_process: true },
    ];
    static ADD_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestAdder",
            category: "test",
            doc: "a[0] + b[0]",
            inputs: ADD_IN,
            outputs: ADD_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: adder_make,
        }
    }

    // A source that sleeps in process() — used to prove independent nodes at the
    // same topological level actually run concurrently (wall-clock < sum).
    struct Slow {
        ms: u64,
    }
    impl Node for Slow {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            std::thread::sleep(std::time::Duration::from_millis(self.ms));
            let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    fn slow_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(Slow { ms: 20 })
    }
    static SLOW_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestSlow",
            category: "test",
            doc: "sleeps 20ms then emits",
            inputs: &[],
            outputs: SLOW_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: slow_make,
        }
    }

    // A node that panics in process() — to verify the engine survives it.
    struct Panicky;
    impl Node for Panicky {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            panic!("boom");
        }
    }
    fn panicky_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(Panicky)
    }
    static P_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestPanic",
            category: "test",
            doc: "panics",
            inputs: &[],
            outputs: P_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: panicky_make,
        }
    }

    // A free-running counter capped at 10 Hz via a `common` group — exercises the
    // wall-clock rate gate. Emits its run count so a test can read how often it ran.
    struct CappedSource {
        runs: i64,
    }
    impl Node for CappedSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            self.runs += 1;
            let d = Data::from_array_bytes(DType::F32, vec![1], (self.runs as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    fn capped_params() -> ParamGroups {
        let mut common = IndexMap::new();
        common.insert("autotrigger".to_string(), Param::boolean(true));
        common.insert("max_frequency".to_string(), Param::float(10.0, 0.0, 60.0)); // 10 Hz -> 0.1s
        common.insert("frequency_mode".to_string(), Param::str_free("updates-per-second"));
        let mut g = ParamGroups::new();
        g.insert("common".to_string(), common);
        g
    }
    fn capped_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(CappedSource { runs: 0 })
    }
    inventory::submit! {
        NodeManifest {
            type_name: "_TestCapped",
            category: "test",
            doc: "10 Hz free-running counter",
            inputs: &[],
            outputs: G_OUT,
            default_params: capped_params,
            isolation: Isolation::InProcess,
            make: capped_make,
        }
    }

    // A node with a TRIGGERING "data" input and a NON-triggering "ref" (control)
    // input, emitting a length-1 frame. Used to prove index propagation ignores a
    // control input even when its length coincidentally matches the output's.
    struct RefLenChange;
    impl Node for RefLenChange {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    fn ref_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(RefLenChange)
    }
    static REF_IN: &[SlotDecl] = &[
        SlotDecl { name: "data", kind: SlotType::Array, trigger_process: true },
        SlotDecl { name: "ref", kind: SlotType::Array, trigger_process: false },
    ];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestRefLenChange",
            category: "test",
            doc: "triggering data + non-triggering ref; emits len-1",
            inputs: REF_IN,
            outputs: C_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: ref_make,
        }
    }

    // A source that emits the engine-supplied wall clock (ctx.now) as its value,
    // to prove NodeCtx::now advances deterministically under an injected clock.
    struct NowSource;
    impl Node for NowSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx) -> NodeResult {
            let d = Data::from_array_bytes(DType::F32, vec![1], (c.now as f32).to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    fn now_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(NowSource)
    }
    inventory::submit! {
        NodeManifest {
            type_name: "_TestNow",
            category: "test",
            doc: "emits ctx.now",
            inputs: &[],
            outputs: G_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: now_make,
        }
    }

    // A pure source with two output slots at different cadences: "fast" emits every
    // run, "slow" every other run — to prove ufreq is measured per output slot.
    struct TwoRate {
        n: i64,
    }
    impl Node for TwoRate {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            self.n += 1;
            let mk = || {
                Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
                    .map_err(|e| e.to_string())
            };
            out.set("fast", mk()?);
            if self.n % 2 == 0 {
                out.set("slow", mk()?);
            }
            Ok(())
        }
    }
    fn two_rate_make(_: &ParamGroups) -> Box<dyn Node> {
        Box::new(TwoRate { n: 0 })
    }
    static TWO_OUT: &[OutputDecl] = &[
        OutputDecl { name: "fast", kind: SlotType::Array },
        OutputDecl { name: "slow", kind: SlotType::Array },
    ];
    inventory::submit! {
        NodeManifest {
            type_name: "_TestTwoRate",
            category: "test",
            doc: "fast slot every run, slow slot every other run",
            inputs: &[],
            outputs: TWO_OUT,
            default_params: echo_params,
            isolation: Isolation::InProcess,
            make: two_rate_make,
        }
    }

    fn first_f32(d: &Data) -> f32 {
        if let Value::Array(s) = d.value() {
            f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap())
        } else {
            panic!("not an array")
        }
    }

    #[test]
    fn source_streams_latest_frame() {
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(7.0, -1e9, 1e9))
            .unwrap();
        g.tick();
        let f = g.latest_frame(src, "out").expect("frame");
        assert_eq!(first_f32(&f), 7.0);
    }

    #[test]
    fn link_propagates_in_one_tick() {
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(5.0, -1e9, 1e9))
            .unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10))
            .unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        g.add_link(src, "out", echo, "in").unwrap();

        g.tick();

        let f = g.latest_frame(echo, "out").expect("echo produced a frame");
        if let Value::Array(s) = f.value() {
            assert_eq!(s.shape(), &[2]);
        } else {
            panic!("expected array");
        }
        assert_eq!(first_f32(&f), 5.0);
    }

    #[test]
    fn one_wire_per_input_evicts_prior_source() {
        let mut g = Graph::new();
        let a = g.add_node("ConstantArray", None).unwrap();
        let b = g.add_node("ConstantArray", None).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        g.update_param(a, "constant", "value", Param::float(1.0, -1e9, 1e9))
            .unwrap();
        g.update_param(b, "constant", "value", Param::float(2.0, -1e9, 1e9))
            .unwrap();
        g.add_link(a, "out", echo, "in").unwrap();
        g.add_link(b, "out", echo, "in").unwrap(); // evicts a
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(echo, "out").unwrap()), 2.0);
    }

    #[test]
    fn remove_node_drops_links() {
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        g.add_link(src, "out", echo, "in").unwrap();
        g.remove_node(src).unwrap();
        assert!(!g.contains(src));
        g.tick(); // must not panic; echo has no input now
        assert!(g.latest_frame(echo, "out").is_none());
    }

    #[test]
    fn trigger_arbitration_gates_downstream() {
        let mut g = Graph::new();
        let src = g.add_node("_TestGated", None).unwrap(); // emits every other tick
        let cnt = g.add_node("_TestCounter", None).unwrap(); // triggered
        g.add_link(src, "out", cnt, "in").unwrap();
        for _ in 0..6 {
            g.tick();
        }
        // The gated source emits on 3 of 6 ticks, so the counter ran exactly 3 times.
        assert_eq!(first_f32(&g.latest_frame(cnt, "out").expect("counter ran")), 3.0);
    }

    #[test]
    fn unwired_triggered_node_never_runs() {
        let mut g = Graph::new();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        for _ in 0..5 {
            g.tick();
        }
        assert!(
            g.latest_frame(cnt, "out").is_none(),
            "a triggered node with no wired input must never run"
        );
    }

    #[test]
    fn gfi_v3_serialize_load_roundtrip() {
        let mut g = Graph::new();
        let c = g.add_node("ConstantArray", None).unwrap();
        g.update_param(c, "constant", "value", Param::float(7.5, -1e9, 1e9))
            .unwrap();
        g.rename_node(c, "myconst").unwrap();
        g.set_node_pos(c, [11.0, 22.0]).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        g.add_link(c, "out", echo, "in").unwrap();

        let yaml = g.serialize();
        assert!(yaml.contains("version: 3"));

        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_count(), 2);

        let restored = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.name(*u) == Some("myconst"))
            .expect("named node restored");
        assert_eq!(g2.type_name(restored), Some("ConstantArray"));
        assert_eq!(g2.pos(restored), Some([11.0, 22.0]));
        assert_eq!(
            goofi_node::param(g2.params(restored).unwrap(), "constant", "value")
                .unwrap()
                .as_f64(),
            Some(7.5)
        );

        // The link round-trips: ticking drives the echo from the restored source.
        g2.tick();
        let echo2 = g2
            .node_uids()
            .into_iter()
            .find(|u| g2.type_name(*u) == Some("_TestEcho"))
            .unwrap();
        assert!(g2.latest_frame(echo2, "out").is_some(), "restored link must carry data");
        assert_eq!(first_f32(&g2.latest_frame(echo2, "out").unwrap()), 7.5);
    }

    #[test]
    fn load_doc_rejects_unknown_type_before_teardown() {
        let mut g = Graph::new();
        g.add_node("ConstantArray", None).unwrap();
        let before = g.node_count();
        let bad = "version: 3\nnodes:\n  \"00000000000a\":\n    type: NotAReal Node\n    pos: [0, 0]\nlinks: []\n";
        assert!(g.load_doc(bad).is_err());
        // validate-before-teardown: the existing graph is untouched on failure.
        assert_eq!(g.node_count(), before);
    }

    #[test]
    fn independent_nodes_run_in_parallel() {
        // Eight sources with no edges between them all sit in topo level 0, so a
        // parallel scheduler runs them concurrently. Each sleeps 20ms: a
        // sequential tick would take >= 160ms; a parallel one must finish well
        // under that. Generous bound to stay robust on a loaded machine.
        let mut g = Graph::new();
        for _ in 0..8 {
            g.add_node("_TestSlow", None).unwrap();
        }
        g.tick(); // warm the rayon pool (first use pays thread-spawn cost)
        let t = std::time::Instant::now();
        g.tick();
        let elapsed = t.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "8 independent 20ms nodes took {elapsed:?}; expected concurrent execution (< 100ms)"
        );
    }

    #[test]
    fn independent_branches_both_produce_correctly() {
        // Two disjoint ConstantArray -> Echo branches must both propagate in one
        // tick regardless of the parallel scheduling of their level-0 sources.
        let mut g = Graph::new();
        let a = g.add_node("ConstantArray", None).unwrap();
        let ea = g.add_node("_TestEcho", None).unwrap();
        g.update_param(a, "constant", "value", Param::float(3.0, -1e9, 1e9)).unwrap();
        g.add_link(a, "out", ea, "in").unwrap();

        let b = g.add_node("ConstantArray", None).unwrap();
        let eb = g.add_node("_TestEcho", None).unwrap();
        g.update_param(b, "constant", "value", Param::float(4.0, -1e9, 1e9)).unwrap();
        g.add_link(b, "out", eb, "in").unwrap();

        g.tick();
        assert_eq!(first_f32(&g.latest_frame(ea, "out").unwrap()), 3.0);
        assert_eq!(first_f32(&g.latest_frame(eb, "out").unwrap()), 4.0);
    }

    // A runtime source built by a captured closure (not a bare fn pointer) —
    // stands in for a pyo3 node whose factory captures a Python class handle.
    struct RtSource {
        base: f32,
    }
    impl Node for RtSource {
        fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
            let d = Data::from_array_bytes(DType::F32, vec![1], self.base.to_le_bytes().to_vec(), Meta::empty())
                .map_err(|e| e.to_string())?;
            out.set("out", d);
            Ok(())
        }
    }
    fn rt_params() -> ParamGroups {
        ParamGroups::new()
    }
    fn rt_stub_make(_: &ParamGroups) -> Box<dyn Node> {
        unreachable!("a runtime dyn type is constructed by its registered factory, not manifest.make")
    }
    static RT_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: SlotType::Array,
    }];
    static RT_MANIFEST: NodeManifest = NodeManifest {
        type_name: "_RuntimeDyn",
        category: "runtime",
        doc: "runtime-registered node type",
        inputs: &[],
        outputs: RT_OUT,
        default_params: rt_params,
        isolation: Isolation::InProcess,
        make: rt_stub_make,
    };

    // A runtime manifest whose name collides with a built-in catalog type.
    static COLLIDE_MANIFEST: NodeManifest = NodeManifest {
        type_name: "Oscillator",
        category: "runtime",
        doc: "collides with the built-in Oscillator",
        inputs: &[],
        outputs: RT_OUT,
        default_params: rt_params,
        isolation: Isolation::InProcess,
        make: rt_stub_make,
    };

    #[test]
    fn register_dyn_type_refuses_collisions() {
        let mut g = Graph::new();
        // Collides with the built-in "Oscillator": refused, and add_node still
        // resolves the native node (the dyn factory would panic via rt_stub_make).
        assert!(!g.register_dyn_type(&COLLIDE_MANIFEST, Box::new(|_| unreachable!())));
        assert!(g.dyn_type_manifests().is_empty());
        let osc = g.add_node("Oscillator", None).unwrap();
        assert_eq!(g.manifest(osc).unwrap().category, "inputs"); // the native one

        // A fresh name registers once; a second registration of the same name is
        // refused rather than overwriting (which would orphan the first's manifest).
        assert!(g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 }))));
        assert!(!g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 2.0 }))));
        assert_eq!(g.dyn_type_manifests().len(), 1);
    }

    #[test]
    fn hosts_a_runtime_registered_dyn_type() {
        let mut g = Graph::new();
        // Register a node TYPE that is not in the compile-time inventory. The
        // factory captures state (base = 42.0), which a fn pointer could not.
        let base = 42.0f32;
        g.register_dyn_type(
            &RT_MANIFEST,
            Box::new(move |_params| Box::new(RtSource { base })),
        );
        // add_node resolves it transparently, like any catalog node.
        let uid = g.add_node("_RuntimeDyn", None).unwrap();
        assert_eq!(g.type_name(uid), Some("_RuntimeDyn"));
        assert_eq!(g.manifest(uid).unwrap().category, "runtime");
        g.tick();
        assert_eq!(first_f32(&g.latest_frame(uid, "out").unwrap()), 42.0);
    }

    #[test]
    fn dyn_type_manifests_enumerates_registered_runtime_types() {
        let mut g = Graph::new();
        assert!(g.dyn_type_manifests().is_empty());
        g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        let ms = g.dyn_type_manifests();
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].type_name, "_RuntimeDyn");
        assert_eq!(ms[0].category, "runtime");
    }

    #[test]
    fn dyn_type_survives_gfi_roundtrip() {
        // A .gfi referencing a runtime type must load into a graph that has the
        // type registered (validation consults both inventory and dyn types).
        let mut g = Graph::new();
        g.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        g.add_node("_RuntimeDyn", None).unwrap();
        let yaml = g.serialize();

        let mut g2 = Graph::new();
        g2.register_dyn_type(&RT_MANIFEST, Box::new(|_| Box::new(RtSource { base: 1.0 })));
        g2.load_doc(&yaml).unwrap();
        assert_eq!(g2.node_count(), 1);

        // Loading a .gfi with an *unregistered* runtime type is rejected up front.
        let mut g3 = Graph::new();
        assert!(g3.load_doc(&yaml).is_err());
        assert_eq!(g3.node_count(), 0);
    }

    #[test]
    fn diamond_converges_through_levels_in_one_tick() {
        // src -> echoA, src -> echoB, {echoA,echoB} -> adder. Levels: src(0),
        // {echoA,echoB}(1, parallel), adder(2). The adder must see BOTH branch
        // outputs — proving level-2 propagation waits for the whole level-1 batch.
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        g.update_param(src, "constant", "value", Param::float(5.0, -1e9, 1e9)).unwrap();
        let ea = g.add_node("_TestEcho", None).unwrap();
        let eb = g.add_node("_TestEcho", None).unwrap();
        let add = g.add_node("_TestAdder", None).unwrap();
        g.add_link(src, "out", ea, "in").unwrap();
        g.add_link(src, "out", eb, "in").unwrap();
        g.add_link(ea, "out", add, "a").unwrap();
        g.add_link(eb, "out", add, "b").unwrap();

        g.tick();
        assert_eq!(first_f32(&g.latest_frame(add, "out").expect("adder produced")), 10.0);
    }

    #[test]
    fn cycle_is_tolerated_without_hanging() {
        // A pure 2-cycle of triggered nodes (echoA -> echoB -> echoA) has no
        // level-0 seed: both land in the cycle-remainder final level. tick() must
        // terminate (not spin) and, unseeded, produce nothing.
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let b = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", b, "in").unwrap();
        g.add_link(b, "out", a, "in").unwrap();
        g.tick(); // must return
        assert!(g.latest_frame(a, "out").is_none());
        assert!(g.latest_frame(b, "out").is_none());
    }

    #[test]
    fn sustained_load_reference_stress_shape_stays_stable() {
        use std::time::Duration;
        // The reference stress-patch shape: one Oscillator fanning out to a PSD and
        // 8 Buffers — all at topo level 1, so they run concurrently on the pool each
        // tick. Drive it hard and assert every consumer keeps producing with a clean
        // error channel (sustained parallel stability, no drift into a faulted state).
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let psd = g.add_node("PSD", None).unwrap();
        g.add_link(osc, "out", psd, "data").unwrap();
        let mut buffers = Vec::new();
        for _ in 0..8 {
            let b = g.add_node("Buffer", None).unwrap();
            g.add_link(osc, "out", b, "data").unwrap();
            buffers.push(b);
        }

        // Advance a synthetic clock 10 ms/tick so the wall-clock-paced Oscillator
        // emits a real block each tick (default 1 kHz -> ~10 samples) and keeps its
        // consumers fed — a tight `tick()` loop would pass no time and starve them.
        let t0 = Instant::now();
        for i in 0..5000u64 {
            g.tick_at(t0 + Duration::from_millis(10 * i));
        }

        assert!(g.last_error(osc).is_none(), "oscillator faulted: {:?}", g.last_error(osc));
        assert!(g.last_error(psd).is_none(), "psd faulted: {:?}", g.last_error(psd));
        assert!(g.latest_frame(psd, "psd").is_some(), "psd must keep producing under load");
        for b in &buffers {
            assert!(g.last_error(*b).is_none(), "buffer faulted: {:?}", g.last_error(*b));
            assert!(g.latest_frame(*b, "out").is_some(), "each buffer must keep producing");
        }
    }

    #[test]
    fn generator_stamps_fresh_incrementing_index() {
        // A source (no index-bearing input) gets a fresh per-output counter that
        // advances once per emit: after 3 ticks the latest frame carries index 2.
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        for _ in 0..3 {
            g.tick();
        }
        let f = g.latest_frame(src, "out").expect("frame");
        assert_eq!(f.meta().index, Some(2), "3 emits -> indices 0,1,2 (latest 2)");
    }

    #[test]
    fn length_preserving_node_propagates_source_index() {
        // ConstantArray(len 2) -> Echo (echoes -> len 2). The echo's output frame
        // count matches its single index-bearing input, so it PROPAGATES the
        // source's origin index rather than starting a fresh counter — an upstream
        // drop stays visible at the sink. Pre-tick the source unwired so its index
        // is a non-zero 3, distinguishable from a fresh-from-0 counter.
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10)).unwrap();
        let echo = g.add_node("_TestEcho", None).unwrap();
        for _ in 0..3 {
            g.tick(); // src advances to index 2; echo (unwired, triggered) never runs
        }
        g.add_link(src, "out", echo, "in").unwrap();
        g.tick(); // src -> index 3; echo runs, matches len -> propagates 3
        let f = g.latest_frame(echo, "out").expect("echo ran");
        assert_eq!(f.meta().index, Some(3), "propagates the source's index, not fresh 0");
    }

    #[test]
    fn length_changing_node_uses_fresh_index() {
        // ConstantArray(len 2) -> Counter (emits len 1). The output frame count (1)
        // never matches the input (2), so no input is the same timeline: the counter
        // starts its OWN fresh index at 0, independent of the source's index (3).
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        g.update_param(src, "constant", "length", Param::int(2, 1, 10)).unwrap();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        for _ in 0..3 {
            g.tick(); // src advances to index 2; counter (unwired) never runs
        }
        g.add_link(src, "out", cnt, "in").unwrap();
        g.tick(); // src -> index 3; counter runs, len mismatch -> fresh index 0
        let f = g.latest_frame(cnt, "out").expect("counter ran");
        assert_eq!(f.meta().index, Some(0), "fresh counter, not the source's 3");
    }

    #[test]
    fn every_node_gets_a_common_group() {
        // The engine merges a universal `common` scheduling group into every node
        // (like Python's DEFAULT_PARAMS), so rate controls exist uniformly.
        let mut g = Graph::new();
        let c = g.add_node("ConstantArray", None).unwrap();
        let p = g.params(c).unwrap();
        let common = p.get("common").expect("common group injected");
        assert!(common.contains_key("autotrigger"));
        assert!(common.contains_key("max_frequency"));
        assert!(common.contains_key("frequency_mode"));
        // Default is unbounded + not autotriggering (behavior-preserving).
        assert_eq!(common["max_frequency"].as_f64(), Some(0.0));
        assert_eq!(common["autotrigger"].as_bool(), Some(false));
    }

    #[test]
    fn common_max_frequency_caps_a_production_node() {
        use std::time::Duration;
        // Cap a real source (ConstantArray, a free-running generator) at 10 Hz via
        // its `common` group; its emit index advances only on admitted ticks.
        let mut g = Graph::new();
        let c = g.add_node("ConstantArray", None).unwrap();
        g.update_param(c, "common", "max_frequency", Param::float(10.0, 0.0, 60.0)).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // run -> index 0
        g.tick_at(t0 + Duration::from_millis(50)); // skip
        g.tick_at(t0 + Duration::from_millis(100)); // run -> index 1
        g.tick_at(t0 + Duration::from_millis(210)); // run -> index 2
        assert_eq!(g.latest_frame(c, "out").unwrap().meta().index, Some(2), "capped to 3 emits");
    }

    #[test]
    fn run_policy_survives_gfi_roundtrip() {
        use std::time::Duration;
        // A saved max_frequency must re-derive into the loaded node's run gate.
        let mut g = Graph::new();
        let c = g.add_node("ConstantArray", None).unwrap();
        g.update_param(c, "common", "max_frequency", Param::float(10.0, 0.0, 60.0)).unwrap();
        let yaml = g.serialize();

        let mut g2 = Graph::new();
        g2.load_doc(&yaml).unwrap();
        let c2 = g2.node_uids()[0];
        assert_eq!(
            goofi_node::param(g2.params(c2).unwrap(), "common", "max_frequency").unwrap().as_f64(),
            Some(10.0),
            "max_frequency round-trips"
        );
        let t0 = Instant::now();
        g2.tick_at(t0);
        g2.tick_at(t0 + Duration::from_millis(50)); // skip -> gate active after load
        g2.tick_at(t0 + Duration::from_millis(100));
        assert_eq!(g2.latest_frame(c2, "out").unwrap().meta().index, Some(1), "gate active post-load");
    }

    #[test]
    fn autotrigger_does_not_free_run_a_wired_trigger_node() {
        // A wired triggered node with common.autotrigger=true must still run ONLY
        // when a fresh frame arrives on its wired trigger — matching Python's
        // `autotrigger AND _has_no_triggering_inputs()`. Gated source emits every
        // other tick; over 6 ticks the counter must run exactly 3 times, not 6.
        let mut g = Graph::new();
        let src = g.add_node("_TestGated", None).unwrap();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        g.add_link(src, "out", cnt, "in").unwrap();
        g.update_param(cnt, "common", "autotrigger", Param::boolean(true)).unwrap();
        for _ in 0..6 {
            g.tick();
        }
        assert_eq!(
            first_f32(&g.latest_frame(cnt, "out").expect("counter ran")),
            3.0,
            "autotrigger must not fire a wired-trigger node on its idle ticks"
        );
    }

    #[test]
    fn autotrigger_free_runs_an_unwired_trigger_node() {
        // The faithful counterpart: a node that DECLARES a trigger input but has it
        // UNWIRED, with autotrigger=true, free-runs every tick (Python:
        // `_has_no_triggering_inputs()` is true when the slot has no source). This
        // guards the fix from over-correcting the wired case into this one.
        let mut g = Graph::new();
        let cnt = g.add_node("_TestCounter", None).unwrap();
        g.update_param(cnt, "common", "autotrigger", Param::boolean(true)).unwrap();
        for _ in 0..3 {
            g.tick();
        }
        assert_eq!(
            first_f32(&g.latest_frame(cnt, "out").expect("free-ran")),
            3.0,
            "an unwired trigger node with autotrigger must free-run"
        );
    }

    #[test]
    fn ctx_now_is_seconds_since_first_tick() {
        use std::time::Duration;
        let mut g = Graph::new();
        let n = g.add_node("_TestNow", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // first tick anchors the reference -> now == 0
        assert_eq!(first_f32(&g.latest_frame(n, "out").unwrap()), 0.0);
        g.tick_at(t0 + Duration::from_millis(250)); // 0.25 s later
        assert!((first_f32(&g.latest_frame(n, "out").unwrap()) - 0.25).abs() < 1e-4);
    }

    #[test]
    fn rate_cap_gates_runs_by_wall_clock() {
        use std::time::Duration;
        // A 10 Hz (0.1s period) free-running source. Drive tick_at with a synthetic
        // clock and assert it runs only once the period has elapsed since last run.
        let mut g = Graph::new();
        let src = g.add_node("_TestCapped", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // never run -> runs (count 1)
        g.tick_at(t0 + Duration::from_millis(50)); // 0.05 < 0.1 -> skip
        g.tick_at(t0 + Duration::from_millis(100)); // 0.10 elapsed -> run (count 2)
        g.tick_at(t0 + Duration::from_millis(120)); // 0.02 since last -> skip
        g.tick_at(t0 + Duration::from_millis(210)); // 0.11 since last -> run (count 3)
        assert_eq!(
            first_f32(&g.latest_frame(src, "out").unwrap()),
            3.0,
            "10 Hz cap admitted exactly 3 of 5 ticks"
        );
    }

    fn ufreq(g: &Graph, uid: Uid, slot: &str) -> Option<f64> {
        g.latest_frame(uid, slot).unwrap().meta().ufreq
    }

    #[test]
    fn ufreq_measures_steady_source_rate() {
        use std::time::Duration;
        // A pure source ticked every 10 ms emits at a steady 100 Hz. The first frame
        // has no interval to measure; from the second on, a steady period reads exact.
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0);
        assert_eq!(ufreq(&g, src, "out"), None, "first emit: no interval yet");
        g.tick_at(t0 + Duration::from_millis(10));
        let uf = ufreq(&g, src, "out").expect("measured after 2nd emit");
        assert!((uf - 100.0).abs() < 1e-6, "10 ms period -> 100 Hz, got {uf}");
        g.tick_at(t0 + Duration::from_millis(20));
        let uf3 = ufreq(&g, src, "out").expect("still measured");
        assert!((uf3 - 100.0).abs() < 1e-6, "steady source stays exact, got {uf3}");
    }

    #[test]
    fn ufreq_reflects_the_rate_cap_not_the_tick_rate() {
        use std::time::Duration;
        // A 10 Hz-capped source ticked at 100 Hz emits every ~0.1 s. Its ufreq must
        // read the emit rate (~10 Hz), NOT the tick rate.
        let mut g = Graph::new();
        let src = g.add_node("_TestCapped", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // run (emit 1) -> no interval yet
        assert_eq!(ufreq(&g, src, "out"), None);
        g.tick_at(t0 + Duration::from_millis(50)); // skipped by the cap
        g.tick_at(t0 + Duration::from_millis(100)); // run (emit 2): dt = 0.1 s
        let uf = ufreq(&g, src, "out").expect("measured after 2nd emit");
        assert!((uf - 10.0).abs() < 1e-6, "capped emit rate -> 10 Hz, got {uf}");
    }

    #[test]
    fn ufreq_is_measured_per_output_slot() {
        use std::time::Duration;
        // "fast" emits every 10 ms run (100 Hz); "slow" every other run (50 Hz).
        // Each slot's meter advances only on its own emits, so the two disagree.
        let mut g = Graph::new();
        let src = g.add_node("_TestTwoRate", None).unwrap();
        let t0 = Instant::now();
        for i in 0..6 {
            g.tick_at(t0 + Duration::from_millis(10 * i));
        }
        let fast = ufreq(&g, src, "fast").expect("fast measured");
        let slow = ufreq(&g, src, "slow").expect("slow measured");
        assert!((fast - 100.0).abs() < 1e-6, "fast slot -> 100 Hz, got {fast}");
        assert!((slow - 50.0).abs() < 1e-6, "slow slot -> 50 Hz, got {slow}");
    }

    #[test]
    fn ufreq_guards_nonadvancing_clock() {
        use std::time::Duration;
        // Two emits at the SAME instant (dt == 0) must never yield inf/NaN: before a
        // measurement exists it stays None; afterwards it keeps the prior estimate.
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0); // emit 1
        g.tick_at(t0); // emit 2, dt == 0, no prior estimate
        assert_eq!(ufreq(&g, src, "out"), None, "dt==0 with no estimate stays None");
        g.tick_at(t0 + Duration::from_millis(10)); // emit 3: dt = 0.01 -> 100 Hz
        assert!((ufreq(&g, src, "out").unwrap() - 100.0).abs() < 1e-6);
        g.tick_at(t0 + Duration::from_millis(10)); // emit 4, dt == 0: keep prior estimate
        let uf = ufreq(&g, src, "out").unwrap();
        assert!(uf.is_finite(), "dt==0 must not produce inf/NaN, got {uf}");
        assert!((uf - 100.0).abs() < 1e-6, "keeps the prior 100 Hz estimate, got {uf}");
    }

    #[test]
    fn ufreq_survives_the_data_plane_wire() {
        use std::time::Duration;
        // End-to-end through the bridge's exact seam: an engine-stamped frame,
        // encoded as `goofi_codec::encode(latest_frame(..))` (see bridge/lib.rs),
        // carries ufreq across the wire so the browser inspector shows it.
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        let t0 = Instant::now();
        g.tick_at(t0);
        g.tick_at(t0 + Duration::from_millis(10)); // steady 100 Hz
        let frame = g.latest_frame(src, "out").unwrap();
        assert!((frame.meta().ufreq.unwrap() - 100.0).abs() < 1e-6);

        let wire = goofi_codec::encode(&frame);
        let back = goofi_codec::decode(&wire).expect("data-plane frame decodes");
        assert_eq!(back.meta().ufreq, frame.meta().ufreq, "ufreq round-trips the data plane");
        assert!((back.meta().ufreq.unwrap() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn default_policy_runs_every_tick_regardless_of_clock() {
        use std::time::Duration;
        // A default-policy source (unbounded) must run on every tick even when the
        // clock barely advances — proving the rate gate is inert without a cap
        // (backward compatibility with the pre-RunPolicy scheduler).
        let mut g = Graph::new();
        let src = g.add_node("ConstantArray", None).unwrap();
        let t0 = Instant::now();
        for i in 0..5 {
            g.tick_at(t0 + Duration::from_nanos(i)); // clock essentially frozen
        }
        // 5 emits -> the generator's index advanced to 4 (ran every tick).
        assert_eq!(g.latest_frame(src, "out").unwrap().meta().index, Some(4));
    }

    #[test]
    fn control_input_is_not_an_index_timeline() {
        // A non-triggering "ref" (control) input must NOT drive meta["index"], even
        // when its length coincidentally equals the output's. `ref`'s index is
        // advanced to 3 while the consumer is dormant (its "data" trigger unwired),
        // then a length-4 data frame triggers the consumer, which emits length 1 —
        // matching only the length-1 ref. The output index must be a FRESH 0, not
        // ref's 3 (which a naive length-only match would wrongly propagate).
        let mut g = Graph::new();
        let rs = g.add_node("ConstantArray", None).unwrap(); // ref source, len 1
        let ds = g.add_node("ConstantArray", None).unwrap();
        g.update_param(ds, "constant", "length", Param::int(4, 1, 10)).unwrap(); // data source, len 4
        let c = g.add_node("_TestRefLenChange", None).unwrap();
        g.add_link(rs, "out", c, "ref").unwrap();
        for _ in 0..3 {
            g.tick(); // rs -> index 2; c dormant (data unwired, triggered node)
        }
        g.add_link(ds, "out", c, "data").unwrap();
        g.tick(); // rs -> index 3 (len 1); ds -> index 0 (len 4); c emits len 1
        let f = g.latest_frame(c, "out").expect("consumer ran");
        assert_eq!(f.meta().index, Some(0), "control input must not be the timeline");
    }

    #[test]
    fn panicking_node_does_not_crash_the_engine() {
        // Silence the default panic backtrace during this test.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let mut g = Graph::new();
        let boom = g.add_node("_TestPanic", None).unwrap();
        let ok = g.add_node("ConstantArray", None).unwrap();
        g.update_param(ok, "constant", "value", Param::float(9.0, -1e9, 1e9))
            .unwrap();

        g.tick(); // must NOT unwind past here (would poison the graph lock)

        std::panic::set_hook(prev);

        // The panic is captured as the node's error; the healthy node still ran.
        assert!(
            g.last_error(boom).unwrap_or("").contains("panic"),
            "panic must be captured as an error"
        );
        assert_eq!(first_f32(&g.latest_frame(ok, "out").unwrap()), 9.0);
    }
}
