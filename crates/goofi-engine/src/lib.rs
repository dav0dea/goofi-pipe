//! goofi-engine — the graph + a minimal single-threaded tick scheduler (M1).
//!
//! Grows into the work-stealing compute pool + reserved RT sub-pool + timer-wheel
//! autotrigger in M2. For now: instantiate catalog nodes, wire one-wire-per-input
//! links, and `tick()` all nodes once in topological order, moving each node's
//! outputs into its consumers' inputs (latest-wins) so a single pass propagates
//! through an acyclic graph. Each node's latest output frame is exposed for the
//! data plane.

use std::collections::HashMap;

use goofi_core::{Data, Param};
use goofi_node::{Inputs, NodeCtx, NodeManifest, Outputs, ParamGroups, ParamKey};
use indexmap::IndexMap;

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

struct NodeEntry {
    type_name: &'static str,
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

/// The authoritative graph + scheduler.
pub struct Graph {
    nodes: IndexMap<Uid, NodeEntry>,
    links: Vec<Link>,
    next_uid: u64,
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
        }
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
        self.nodes.get(&uid).map(|e| e.type_name)
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

    /// Instantiate a catalog node. `params` defaults to the manifest defaults.
    pub fn add_node(
        &mut self,
        type_name: &str,
        params: Option<ParamGroups>,
    ) -> Result<Uid, String> {
        let manifest = goofi_node::find(type_name)
            .ok_or_else(|| format!("unknown node type `{type_name}`"))?;
        let params = params.unwrap_or_else(|| (manifest.default_params)());
        let mut node = (manifest.make)(&params);
        let mut ctx = NodeCtx::new();
        let last_error = node.setup(&mut ctx).err().map(|e| e.0);

        let inputs: IndexMap<&'static str, Option<Data>> =
            manifest.inputs.iter().map(|s| (s.name, None)).collect();
        let outputs = manifest.output_buffer();

        let name = self.fresh_name(&manifest.type_name.to_lowercase());
        let has_trigger_inputs = manifest.inputs.iter().any(|i| i.trigger_process);
        let uid = self.mint();
        self.nodes.insert(
            uid,
            NodeEntry {
                type_name: manifest.type_name,
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
            },
        );
        Ok(uid)
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
                json!({ "type": e.type_name, "name": e.name, "pos": e.pos, "params": Value::Object(params) }),
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
            if goofi_node::find(ty).is_none() {
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

    /// Kahn topological order (producers before consumers); a cycle's remaining
    /// nodes are appended in insertion order (latest-wins tolerates the back-edge).
    fn topo_order(&self) -> Vec<Uid> {
        let mut indeg: HashMap<Uid, usize> = self.nodes.keys().map(|k| (*k, 0)).collect();
        for l in &self.links {
            if self.nodes.contains_key(&l.node_out) && indeg.contains_key(&l.node_in) {
                *indeg.get_mut(&l.node_in).unwrap() += 1;
            }
        }
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut ready: Vec<Uid> = self
            .nodes
            .keys()
            .copied()
            .filter(|u| indeg[u] == 0)
            .collect();
        let mut visited: HashMap<Uid, bool> = HashMap::new();
        while let Some(u) = ready.pop() {
            if visited.insert(u, true).is_some() {
                continue;
            }
            order.push(u);
            for l in &self.links {
                if l.node_out == u {
                    if let Some(d) = indeg.get_mut(&l.node_in) {
                        if *d > 0 {
                            *d -= 1;
                            if *d == 0 {
                                ready.push(l.node_in);
                            }
                        }
                    }
                }
            }
        }
        // Append any cycle remainder in insertion order.
        for u in self.nodes.keys() {
            if !visited.contains_key(u) {
                order.push(*u);
            }
        }
        order
    }

    /// Run one tick of the whole graph. A node runs iff it free-runs (no
    /// triggering inputs) or a triggering input received a fresh frame this round
    /// (trigger arbitration); a skipped node keeps its previous outputs.
    pub fn tick(&mut self) {
        let order = self.topo_order();
        for uid in order {
            let should_run = {
                let entry = self.nodes.get(&uid).expect("node in order exists");
                !entry.has_trigger_inputs || entry.trigger_pending
            };
            if !should_run {
                continue;
            }

            let outgoing: Vec<(&'static str, Uid, &'static str)> = self
                .links
                .iter()
                .filter(|l| l.node_out == uid)
                .map(|l| (l.slot_out, l.node_in, l.slot_in))
                .collect();

            let produced: Vec<(&'static str, Data)> = {
                let entry = self.nodes.get_mut(&uid).expect("node in order exists");
                entry.trigger_pending = false;
                entry.ctx.tick += 1;
                for v in entry.outputs.values_mut() {
                    *v = None;
                }
                let inp = Inputs::new(&entry.inputs);
                let node = &mut entry.node;
                let ctx = &mut entry.ctx;
                let mut out = Outputs::new(&mut entry.outputs);
                // Catch a node panic so one faulty node can't unwind through the
                // graph lock (poisoning the Mutex) and take down the whole engine.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    node.process(&inp, &mut out, ctx)
                }));
                entry.last_error = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(e)) => Some(e.0),
                    Err(p) => Some(panic_message(p)),
                };
                entry
                    .outputs
                    .iter()
                    .filter_map(|(k, v)| v.as_ref().map(|d| (*k, d.clone())))
                    .collect()
            };

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
        length_preserving: true,
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
        length_preserving: false,
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
        length_preserving: false,
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
        length_preserving: false,
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
