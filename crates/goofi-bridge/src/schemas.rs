//! JSON projections of the engine graph into the exact shapes the frontend
//! mirrors (`control.ts` types). These are the wire contract; keep field names
//! and shapes aligned with the frontend or co-edit it.

use goofi_core::Param;
use goofi_engine::{ExprInfo, Graph, LinkView, Uid};
use goofi_node::{NodeManifest, ParamGroups};
use serde_json::{json, Map, Value};

pub const ROOT_ID: &str = "__root__";
pub const PROTOCOL_VERSION: i64 = 1;

/// A single param's current value as the frontend's `descriptor.value` JSON. The one
/// place param → wire-value lives, shared by [`describe_param`] and the live
/// `param_values` projection so the preview and the descriptor always agree.
pub fn param_value_json(p: &Param) -> Value {
    match p {
        Param::Float { value, .. } => json!(value),
        Param::Int { value, .. } => json!(value),
        Param::Bool { value } => json!(value),
        Param::Trigger { fired } => json!(fired),
        Param::Str { value, .. } => json!(value),
    }
}

/// A single param descriptor (discriminated on `type`). `expr` is the instance's
/// expression binding (or `None` for a plain literal / a palette type-level param); `doc` is
/// the static help text from the type's declaration, which the runtime [`Param`] cannot carry.
pub fn describe_param(p: &Param, expr: Option<&ExprInfo>, doc: Option<&str>) -> Value {
    let mut m = Map::new();
    m.insert("value".into(), param_value_json(p));
    m.insert("doc".into(), doc.map(|d| json!(d)).unwrap_or(Value::Null));
    m.insert("save_param".into(), json!(true));
    m.insert(
        "refreshable".into(),
        json!(matches!(p, Param::Str { refresh: true, .. })),
    );
    // Real expression state (or nulls/false for an unbound param). `expression_error`
    // drives the per-param field indicator. Auto-eval is always on, so there is no
    // autoeval flag on the wire.
    m.insert("expression".into(), expr.map(|e| json!(e.source)).unwrap_or(Value::Null));
    m.insert("expression_enabled".into(), json!(expr.is_some_and(|e| e.enabled)));
    m.insert("expression_triggers_process".into(), json!(expr.is_some_and(|e| e.triggers_process)));
    m.insert(
        "expression_error".into(),
        expr.and_then(|e| e.error.as_ref()).map(|s| json!(s)).unwrap_or(Value::Null),
    );
    match p {
        Param::Float { vmin, vmax, .. } => {
            m.insert("type".into(), json!("float"));
            m.insert("vmin".into(), json!(vmin));
            m.insert("vmax".into(), json!(vmax));
        }
        Param::Int { vmin, vmax, .. } => {
            m.insert("type".into(), json!("int"));
            m.insert("vmin".into(), json!(vmin));
            m.insert("vmax".into(), json!(vmax));
        }
        Param::Bool { .. } => {
            m.insert("type".into(), json!("bool"));
            m.insert("trigger".into(), json!(false));
        }
        Param::Trigger { .. } => {
            m.insert("type".into(), json!("bool"));
            m.insert("trigger".into(), json!(true));
        }
        Param::Str { options, .. } => {
            m.insert("type".into(), json!("string"));
            m.insert(
                "options".into(),
                options.as_ref().map(|o| json!(o)).unwrap_or(Value::Null),
            );
        }
    }
    Value::Object(m)
}

/// Look up a param's declared help text. A node's own declaration wins over the universal
/// `common` fallback, matching `with_common`'s keep-what-the-node-declared rule.
fn param_doc(decls: &[goofi_node::ParamDecl], group: &str, name: &str) -> Option<&'static str> {
    decls
        .iter()
        .chain(goofi_node::COMMON_DECLS)
        .find(|d| d.group == group && d.name == name)
        .and_then(|d| d.doc)
}

/// Type-level / literal params (no expression bindings) — used for the palette. This is the
/// projection the frontend renders param tooltips from: the instance descriptors override only
/// value/expression/options, so `doc` has to be right *here*.
pub fn describe_params(p: &ParamGroups, decls: &[goofi_node::ParamDecl]) -> Value {
    let mut groups = Map::new();
    for (gname, g) in p {
        let mut names = Map::new();
        for (n, param) in g {
            names.insert(n.clone(), describe_param(param, None, param_doc(decls, gname, n)));
        }
        groups.insert(gname.clone(), Value::Object(names));
    }
    Value::Object(groups)
}

/// A node instance's params, each carrying its real expression binding state (source /
/// enabled / triggers / error) for the fx toggle + field error indicator.
pub fn describe_node_params(g: &Graph, uid: Uid) -> Value {
    let Some(params) = g.params(uid) else {
        return Value::Object(Map::new());
    };
    let decls = g.manifest(uid).map(|m| m.params).unwrap_or(&[]);
    let mut groups = Map::new();
    for (gname, group) in params {
        let mut names = Map::new();
        for (n, param) in group {
            let expr = g.param_expression(uid, gname, n);
            names.insert(n.clone(), describe_param(param, expr.as_ref(), param_doc(decls, gname, n)));
        }
        groups.insert(gname.clone(), Value::Object(names));
    }
    Value::Object(groups)
}

/// The live values of a node's expression-driven params, shaped `{group: {name: value}}`
/// for the `param_values` event. Empty object when the node has no active expressions.
/// Unlike [`describe_node_params`] this carries ONLY the evaluated values (no descriptor
/// metadata), so the frontend applies it surgically — it can never clobber a concurrent
/// edit the way a full-params replace would.
pub fn expression_value_map(g: &Graph, uid: Uid) -> Value {
    let mut groups = Map::new();
    for (group, name, p) in g.expression_values(uid) {
        let entry = groups.entry(group.to_string()).or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(names) = entry {
            names.insert(name.to_string(), param_value_json(p));
        }
    }
    Value::Object(groups)
}

/// Project `(slot_name, dtype_name)` pairs into a `{name: dtype}` JSON object — shared by
/// [`input_slots`] / [`output_slots`], whose only difference was the source collection.
fn slot_map<'a>(slots: impl Iterator<Item = (&'a str, &'a str)>) -> Value {
    Value::Object(slots.map(|(name, dtype)| (name.to_string(), json!(dtype))).collect())
}
fn input_slots(m: &NodeManifest) -> Value {
    slot_map(m.inputs.iter().map(|s| (s.name, s.kind.name())))
}

/// The names of the node type's `multi` (variadic) input slots — static shape the
/// frontend reads to render those slots tall and accept many cables. Peer of the
/// dtype in [`input_slots`]; not a mutable per-instance flag.
fn input_multi(m: &NodeManifest) -> Value {
    Value::Array(m.inputs.iter().filter(|s| s.multi).map(|s| json!(s.name)).collect())
}
fn output_slots(m: &NodeManifest) -> Value {
    slot_map(m.outputs.iter().map(|s| (s.name, s.kind.name())))
}

pub fn node_type_info(m: &NodeManifest) -> Value {
    json!({
        "type": m.type_name,
        // A node's pillar (signal/audio/video) routes it to its editor panel. All current
        // node types are signal; audio/video manifests will declare their own (layering §9).
        "pillar": "signal",
        "category": m.category,
        "doc": m.doc,
        "available": true,
        "dynamic": false,
        "missing_deps": [],
        "input_slots": input_slots(m),
        "input_multi": input_multi(m),
        "output_slots": output_slots(m),
        // Project the same universal `common` group instances carry, so the palette
        // and an instantiated node agree on a type's params.
        "params": describe_params(&goofi_node::with_common(m.default_params()), m.params),
    })
}

/// The `list_nodes` palette catalog, sorted by (category, type). Includes both
/// the compile-time catalog and the graph's runtime-registered types (e.g.
/// discovered Python nodes). Hidden test nodes (`_`-prefixed) are excluded.
pub fn catalog_types(g: &Graph) -> Value {
    let mut items: Vec<(&'static str, &'static str, Value)> = goofi_node::catalog()
        .chain(g.dyn_type_manifests())
        .filter(|m| !m.type_name.starts_with('_'))
        .map(|m| (m.category, m.type_name, node_type_info(m)))
        .collect();
    items.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));
    Value::Array(items.into_iter().map(|(_, _, v)| v).collect())
}

/// A node's scope membership: which scope it lives in (ROOT if top-level). Rides the `membership`
/// field of every node payload (`node_added` + the `hello`/`graph_replaced` snapshot). The frontend
/// reconciles each member's scope from the doc forest; `local_name` is just the display name now (no
/// template locals in the flat model).
pub fn membership(g: &Graph, uid: Uid) -> Value {
    let name = g.name(uid).unwrap_or("").to_string();
    json!({
        "instance": g.scope_of(uid).map(|s| s.to_hex()).unwrap_or_else(|| ROOT_ID.to_string()),
        "local_name": name,
    })
}

pub fn node_instance_info(g: &Graph, uid: Uid) -> Value {
    let m = g.manifest(uid).expect("node exists");
    let name = g.name(uid).unwrap_or("").to_string();
    json!({
        "uid": uid.to_hex(),
        "name": name,
        "type": g.type_name(uid).unwrap_or(""),
        "pillar": "signal",
        "category": m.category,
        "doc": m.doc,
        "input_slots": input_slots(m),
        "input_multi": input_multi(m),
        "output_slots": output_slots(m),
        "params": describe_node_params(g, uid),
        "pos": g.pos(uid).unwrap_or([0.0, 0.0]),
        "viewers": g.viewers(uid).cloned().unwrap_or_else(|| json!({})),
        "inputs": {},
        "membership": membership(g, uid),
        "error": g.last_error(uid),
        "stage": "ready",
        "stats": Value::Null,
        "restarts": 0,
        "log_endpoint": Value::Null,
    })
}

pub fn link_info(l: &LinkView) -> Value {
    json!({
        "node_out": l.node_out.to_hex(),
        "slot_out": l.slot_out,
        "node_in": l.node_in.to_hex(),
        "slot_in": l.slot_in,
    })
}

/// The ROOT scope the editor renders its canvas from: ROOT-scoped leaf nodes plus top-level
/// sub-patch instances (`is_instance: true`). Nodes/instances inside a sub-patch are NOT
/// members of ROOT — they belong to their instance's scope.
fn root_instance(g: &Graph) -> Value {
    let mut members = Map::new();
    for uid in g.node_uids() {
        if g.scope_of(uid).is_none() {
            let name = g.name(uid).unwrap_or("").to_string();
            members.insert(name, json!({ "uid": uid.to_hex(), "is_instance": false }));
        }
    }
    for scope in g.scope_uids() {
        if g.scope_of(scope).is_none() {
            if let Some(s) = g.scope(scope) {
                members.insert(s.name.clone(), json!({ "uid": scope.to_hex(), "is_instance": true }));
            }
        }
    }
    json!({
        "uid": ROOT_ID,
        "name": "root",
        "kind": "unique",
        "def_id": Value::Null,
        "parent": Value::Null,
        "pos": [0.0, 0.0],
        "interface": {},
        "members": Value::Object(members),
        "slots": { "input": {}, "output": {} },
        "siblings": [],
        "error": Value::Null,
        "viewers": {},
    })
}

/// The first errored descendant of a scope (recursing into nested scopes), for the collapsed
/// sub-patch's error badge. `Null` if the whole subtree is healthy.
fn instance_error(g: &Graph, uid: Uid) -> Value {
    if g.scope(uid).is_none() {
        return Value::Null;
    }
    for muid in g.scope_members(uid) {
        if g.scope(muid).is_some() {
            let e = instance_error(g, muid);
            if !e.is_null() {
                return e;
            }
        } else if let Some(err) = g.last_error(muid) {
            return json!(err);
        }
    }
    Value::Null
}

/// The `InstanceInfo` the frontend types (`control.ts`): a scope's stubs chain-resolved to their
/// inner leaf, wired stubs projected as input/output slots, its member list, and the deep error.
/// With sharing dropped, `kind` is always unique and there are no siblings/def_id. (The frontend is
/// doc-authoritative for the live forest; this rides the snapshot for the initial render.)
pub fn describe_instance(g: &Graph, uid: Uid) -> Value {
    use goofi_engine::subpatch::Dir;
    let Some(scope) = g.scope(uid) else { return Value::Null };

    let mut interface = Map::new();
    let mut in_slots = Map::new();
    let mut out_slots = Map::new();
    for (id, st) in scope.stubs.iter() {
        // Emit the stub's DIRECT inner (member uid + direct slot/nested-StubId), not the chain-
        // resolved leaf — the editor reroutes per level. (The data plane resolves server-side.)
        interface.insert(
            id.clone(),
            json!({
                "dir": match st.dir { Dir::In => "in", Dir::Out => "out" },
                "dtype": st.dtype.name(),
                "inner_node": st.inner.as_ref().map(|(u, _)| u.to_hex()),
                "inner_slot": st.inner.as_ref().map(|(_, s)| s.clone()),
                "pos": st.pos,
                "name": st.name,
            }),
        );
        if st.inner.is_some() {
            match st.dir {
                Dir::In => in_slots.insert(id.clone(), json!(st.dtype.name())),
                Dir::Out => out_slots.insert(id.clone(), json!(st.dtype.name())),
            };
        }
    }

    let mut members = Map::new();
    for muid in g.scope_members(uid) {
        let name = g
            .name(muid)
            .map(str::to_string)
            .or_else(|| g.scope(muid).map(|s| s.name.clone()))
            .unwrap_or_default();
        members.insert(name, json!({ "uid": muid.to_hex(), "is_instance": g.scope(muid).is_some() }));
    }

    json!({
        "uid": uid.to_hex(),
        "name": scope.name,
        "kind": "unique",
        "def_id": Value::Null,
        // A top-level scope (no engine parent) parents to ROOT_ID, not null — the editor's
        // `childrenOfScope` renders a facade on the root canvas iff `instance.parent === ROOT_ID`.
        "parent": g.scope_of(uid).map(|p| json!(p.to_hex())).unwrap_or_else(|| json!(ROOT_ID)),
        "pos": scope.pos,
        "interface": Value::Object(interface),
        "members": Value::Object(members),
        "slots": { "input": Value::Object(in_slots), "output": Value::Object(out_slots) },
        "siblings": [],
        "error": instance_error(g, uid),
        "viewers": {},
    })
}

/// The full graph snapshot (`hello` / `graph_replaced` payload).
pub fn snapshot(g: &Graph, instance_id: &str, with_protocol: bool) -> Value {
    let nodes: Vec<Value> = g.node_uids().iter().map(|u| node_instance_info(g, *u)).collect();
    let links: Vec<Value> = g.links_view().iter().map(link_info).collect();
    let mut instances = Map::new();
    instances.insert(ROOT_ID.to_string(), root_instance(g));
    for scope in g.scope_uids() {
        instances.insert(scope.to_hex(), describe_instance(g, scope));
    }
    let mut snap = json!({
        "instance_id": instance_id,
        // The pillars this backend build actually hosts — the frontend shows only these
        // editors. Signal-only for now; audio/video are added as their runtimes land.
        "pillars": ["signal"],
        "nodes": nodes,
        "links": links,
        "instances": Value::Object(instances),
        "save_path": Value::Null,
        "unsaved_changes": false,
        "layout": Value::Null,
    });
    if with_protocol {
        snap["protocol_version"] = json!(PROTOCOL_VERSION);
        // hello / graph_replaced carry the node palette so the client has descriptors in hand
        // immediately — no async `list_nodes` round-trip, so the doc is authoritative for node
        // identity from the first render (no catalog-loading fallback window). Structural echoes
        // (subpatch_changed, with_protocol=false) omit it — the catalog changes only when a
        // runtime type registers, which arrives on the next hello/graph_replaced.
        snap["node_types"] = catalog_types(g);
    }
    snap
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_node::{Isolation, OutputDecl, ParamDecl, SlotDecl};

    static STUB_PARAMS: &[ParamDecl] = &[];
    fn stub_factory() -> Box<dyn goofi_node::Node> {
        unreachable!("catalog_types never instantiates")
    }
    static T_OUT: &[OutputDecl] = &[OutputDecl {
        name: "out",
        kind: goofi_core::SlotType::Array,
    }];
    static T_MANIFEST: NodeManifest = NodeManifest {
        type_name: "MyPyThing",
        category: "python",
        doc: "runtime type",
        inputs: &[],
        outputs: T_OUT,
        params: STUB_PARAMS,
        isolation: Isolation::InProcess,
        factory: stub_factory,
    };

    static MULTI_IN: &[SlotDecl] = &[
        SlotDecl { name: "many", kind: goofi_core::SlotType::Table, trigger_process: true, multi: true },
        SlotDecl { name: "one", kind: goofi_core::SlotType::Array, trigger_process: true, multi: false },
    ];
    static MULTI_MANIFEST: NodeManifest = NodeManifest {
        type_name: "MultiThing",
        category: "test",
        doc: "has a multi input slot",
        inputs: MULTI_IN,
        outputs: T_OUT,
        params: STUB_PARAMS,
        isolation: Isolation::InProcess,
        factory: stub_factory,
    };

    static DOCUMENTED_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "welch",
        name: "nperseg",
        spec: goofi_node::ParamSpec::Int { default: 256, min: 16, max: 4096 },
        default_expr: None,
        doc: Some("Samples per Welch segment: longer means finer frequency resolution."),
    }];
    static DOCUMENTED_MANIFEST: NodeManifest = NodeManifest {
        type_name: "DocumentedThing",
        category: "test",
        doc: "has a documented param",
        inputs: &[],
        outputs: T_OUT,
        params: DOCUMENTED_PARAMS,
        isolation: Isolation::InProcess,
        factory: stub_factory,
    };

    #[test]
    fn the_catalog_carries_each_params_doc() {
        // The frontend renders a param tooltip from the CATALOG descriptor (the instance path
        // only overrides value/expression/options), so this is the projection that matters.
        let mut g = Graph::new();
        g.register_dyn_type(&DOCUMENTED_MANIFEST, Box::new(|_| unreachable!()));
        let cat = catalog_types(&g);
        let ty = cat
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["type"] == "DocumentedThing")
            .expect("type registered")
            .clone();

        assert_eq!(
            ty["params"]["welch"]["nperseg"]["doc"],
            json!("Samples per Welch segment: longer means finer frequency resolution.")
        );
        // An undocumented param stays null rather than "" so the UI shows no tooltip at all.
        assert_eq!(ty["params"]["common"]["autotrigger"]["doc"].is_null(), false, "common params are documented too");
    }

    #[test]
    fn catalog_includes_runtime_registered_types() {
        let mut g = Graph::new();
        g.register_dyn_type(&T_MANIFEST, Box::new(|_| unreachable!()));
        let cat = catalog_types(&g);
        let arr = cat.as_array().unwrap();
        let ty = |v: &Value| v.get("type").and_then(|t| t.as_str()).map(str::to_string);
        assert!(
            arr.iter().any(|v| ty(v).as_deref() == Some("MyPyThing")),
            "runtime-registered type must appear in the palette"
        );
        // Native catalog types remain present alongside the runtime ones.
        assert!(arr.iter().any(|v| ty(v).as_deref() == Some("Oscillator")));
    }

    #[test]
    fn hello_snapshot_embeds_the_node_catalog() {
        let g = Graph::new();
        // hello / graph_replaced (with_protocol=true) carry the palette so the client needs no
        // async `list_nodes` round-trip before it can build nodes from the doc (retires the
        // catalog-loading fallback window).
        let hello = snapshot(&g, "iid", true);
        assert_eq!(
            hello["node_types"],
            catalog_types(&g),
            "hello embeds the same palette `list_nodes` returns"
        );
        assert!(hello["node_types"].as_array().is_some_and(|a| !a.is_empty()));
        // A structural echo (subpatch_changed, with_protocol=false) must NOT re-ship the whole
        // catalog on every group/expand/share — it changes only when a runtime type registers.
        let echo = snapshot(&g, "iid", false);
        assert!(echo.get("node_types").is_none(), "structural echoes omit the catalog");
    }

    #[test]
    fn input_multi_lists_the_variadic_input_slots() {
        // A node's multi slots appear in input_multi (static shape the frontend reads
        // to render them tall); single slots do not.
        assert_eq!(node_type_info(&MULTI_MANIFEST)["input_multi"], json!(["many"]));
        // A node with only single inputs reports an empty list.
        assert_eq!(node_type_info(&T_MANIFEST)["input_multi"], json!([]));
    }

    #[test]
    fn catalog_projects_the_common_scheduling_group() {
        // The palette catalog must show the same universal `common` group every
        // instantiated node carries, so type-level and instance-level params agree.
        let info = node_type_info(&T_MANIFEST); // STUB_PARAMS -> empty groups
        let common = &info["params"]["common"];
        assert_eq!(common["max_frequency"]["type"], json!("float"));
        assert_eq!(common["autotrigger"]["type"], json!("bool"));
        assert_eq!(common["frequency_mode"]["type"], json!("string"));
    }

    #[test]
    fn node_type_info_carries_the_signal_pillar() {
        // The pillar tag rides the control contract so the frontend can route a node to its
        // editor panel; every current type is signal.
        assert_eq!(node_type_info(&T_MANIFEST)["pillar"], json!("signal"));
    }
}
