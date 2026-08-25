//! JSON projections of the engine graph into the shapes the frontend mirrors (`control.ts`).
//! These are the wire contract: co-edit the frontend when a field or shape changes.

use goofi_core::Param;
use goofi_engine::{ExprInfo, Graph, Uid};
use goofi_node::{NodeManifest, ParamGroups};
use serde_json::{json, Map, Value};

pub const PROTOCOL_VERSION: i64 = 3;

/// A single param descriptor, discriminated on `type`. `doc` is the type declaration's help text,
/// which the runtime [`Param`] cannot carry.
pub fn describe_param(p: &Param, expr: Option<&ExprInfo>, doc: Option<&str>) -> Value {
    let mut m = Map::new();
    m.insert("value".into(), goofi_engine::param_value_json(p, true));
    m.insert("doc".into(), doc.map(|d| json!(d)).unwrap_or(Value::Null));
    m.insert(
        "refreshable".into(),
        json!(matches!(p, Param::Str { refresh: true, .. })),
    );
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

/// A param's declared help text; a node's own declaration wins over the universal `common` one.
fn param_doc(m: &NodeManifest, group: &str, name: &str) -> Option<&'static str> {
    m.params
        .iter()
        .copied()
        .chain(goofi_node::common_decls(m))
        .find(|d| d.group == group && d.name == name)
        .and_then(|d| d.doc)
}

/// Type-level params for the palette, and the projection param tooltips are rendered from.
pub fn describe_params(p: &ParamGroups, m: &NodeManifest) -> Value {
    let mut groups = Map::new();
    for (gname, g) in p {
        let mut names = Map::new();
        for (n, param) in g {
            names.insert(n.clone(), describe_param(param, None, param_doc(m, gname, n)));
        }
        groups.insert(gname.clone(), Value::Object(names));
    }
    Value::Object(groups)
}

/// A node instance's params, each carrying its real expression binding state.
pub fn describe_node_params(g: &Graph, uid: Uid) -> Value {
    let (Some(params), Some(m)) = (g.params(uid), g.manifest(uid)) else {
        return Value::Object(Map::new());
    };
    let mut groups = Map::new();
    for (gname, group) in &*params {
        let mut names = Map::new();
        for (n, param) in group {
            let expr = g.param_expression(uid, gname, n);
            names.insert(n.clone(), describe_param(param, expr.as_ref(), param_doc(m, gname, n)));
        }
        groups.insert(gname.clone(), Value::Object(names));
    }
    Value::Object(groups)
}

/// The live values of a node's expression-driven params, `{group: {name: value}}`. Values only, so
/// the frontend applies it surgically and cannot clobber a concurrent edit.
pub fn expression_value_map(g: &Graph, uid: Uid) -> Value {
    let mut groups = Map::new();
    for (group, name, p) in g.expression_values(uid) {
        let entry = groups.entry(group.to_string()).or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(names) = entry {
            names.insert(name.to_string(), goofi_engine::param_value_json(p, true));
        }
    }
    Value::Object(groups)
}

/// A node instance's param VALUES, `{group: {name: value}}`, without descriptor metadata.
pub fn param_value_map(params: &goofi_node::ParamGroups) -> Value {
    Value::Object(
        params
            .iter()
            .map(|(gname, group)| {
                let names = group.iter().map(|(n, p)| (n.clone(), goofi_engine::param_value_json(p, true)));
                (gname.clone(), Value::Object(names.collect()))
            })
            .collect(),
    )
}

/// Project `(slot_name, dtype_name)` pairs into a `{name: dtype}` JSON object.
fn slot_map<'a>(slots: impl Iterator<Item = (&'a str, &'a str)>) -> Value {
    Value::Object(slots.map(|(name, dtype)| (name.to_string(), json!(dtype))).collect())
}
fn input_slots(m: &NodeManifest) -> Value {
    slot_map(m.inputs.iter().map(|s| (s.name, s.kind.name())))
}

/// The names of the node type's `multi` (variadic) input slots — static shape, not per-instance.
fn input_multi(m: &NodeManifest) -> Value {
    Value::Array(m.inputs.iter().filter(|s| s.multi).map(|s| json!(s.name)).collect())
}
fn output_slots(m: &NodeManifest) -> Value {
    slot_map(m.outputs.iter().map(|s| (s.name, s.kind.name())))
}

/// Where a palette row's type came from, for the add-menu badge: the open patch, or `builtin`.
fn source_of(g: &Graph, type_name: &str) -> &'static str {
    if g.is_patch_type(type_name) {
        "patch"
    } else {
        "builtin"
    }
}

pub fn node_type_info(m: &NodeManifest, source: &str) -> Value {
    json!({
        "type": m.type_name,
        "source": source,
        "pillar": "signal",
        "category": m.category,
        "doc": m.doc,
        "available": true,
        "missing_deps": [],
        "input_slots": input_slots(m),
        "input_multi": input_multi(m),
        "output_slots": output_slots(m),
        // The same universal `common` group instances carry, so palette and instance agree.
        "params": describe_params(&goofi_node::with_common(m.default_params(), m), m),
    })
}

/// The `list_nodes` palette catalog, sorted by (category, type), compile-time and runtime types
/// alike. Hidden test nodes (`_`-prefixed) are excluded.
pub fn catalog_types(g: &Graph) -> Value {
    let mut items: Vec<(String, String, Value)> = goofi_node::catalog()
        .chain(g.dyn_type_manifests())
        .filter(|m| !m.type_name.starts_with('_'))
        .map(|m| {
            (m.category.to_string(), m.type_name.to_string(), node_type_info(m, source_of(g, m.type_name)))
        })
        .collect();
    // Node files that exist but cannot load are listed too, greyed and with the reason.
    items.extend(g.unavailable_types().map(|(name, reason)| {
        (
            "unavailable".to_string(),
            name.to_string(),
            json!({
                "type": name,
                "source": source_of(g, name),
                "pillar": "signal",
                "category": "unavailable",
                "doc": format!("This node could not be loaded: {reason}"),
                "available": false,
                "missing_deps": [reason],
                "input_slots": {},
                "input_multi": [],
                "output_slots": {},
                "params": {},
            }),
        )
    }));
    items.extend(crate::vocab::boundary_catalog());
    items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    Value::Array(items.into_iter().map(|(_, _, v)| v).collect())
}

/// The per-node RUNTIME overlay that never enters the doc. It rides the snapshot because its live
/// stream pushes only transitions. EVERY node is in it — a facade's health is its members' and a
/// port reaches no stage, and a client that had to work either out would be a second owner.
pub fn runtime_overlay(g: &Graph) -> Value {
    let mut m = Map::new();
    for uid in g.all_uids() {
        m.insert(
            uid.to_hex(),
            json!({
                "stage": g.node_stage(uid),
                "error": g.last_error(uid),
                // Absent for a port and a facade, which run nowhere — as `stage` is for inspect.
                "runtime": g.manifest(uid).map(|m| m.isolation.get().wire()),
            }),
        );
    }
    Value::Object(m)
}

/// The `hello` / `graph_replaced` payload: the session frame plus the truths the doc never holds.
/// It carries NO graph structure — that lives in the document alone.
pub fn snapshot(
    g: &Graph,
    instance_id: &str,
    with_protocol: bool,
    unsaved: bool,
    save_path: Option<&str>,
    harnesses: Value,
) -> Value {
    let mut snap = json!({
        "instance_id": instance_id,
        "pillars": ["signal"],
        "runtime": runtime_overlay(g),
        // Seeded for the same reason the runtime overlay is: `harness_changed` pushes transitions.
        "harnesses": harnesses,
        "save_path": save_path,
        "unsaved_changes": unsaved,
        "viewpoint": g.viewpoint().clone(),
    });
    if with_protocol {
        snap["protocol_version"] = json!(PROTOCOL_VERSION);
        // The palette rides along, so the first render needs no `list_nodes` round-trip.
        snap["node_types"] = catalog_types(g);
    }
    snap
}
