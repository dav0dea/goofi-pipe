//! JSON projections of the engine graph into the shapes the frontend mirrors (`control.ts`).
//! These are the wire contract: co-edit the frontend when a field or shape changes.

use goofi_core::Param;
use goofi_graph::{Graph, SourceInfo, Uid};
use goofi_node::{NodeManifest, ParamGroups};
use serde_json::{json, Map, Value};

pub const PROTOCOL_VERSION: i64 = 4;

/// A single param descriptor, discriminated on `type`. `doc` is the type declaration's help text,
/// which the runtime [`Param`] cannot carry. The source fields are the record's: an empty text is
/// `null`, and a param with no record is a constant.
pub fn describe_param(p: &Param, source: Option<&SourceInfo>, doc: Option<&str>) -> Value {
    let mut m = Map::new();
    m.insert("value".into(), goofi_graph::param_value_json(p));
    m.insert("doc".into(), doc.map(|d| json!(d)).unwrap_or(Value::Null));
    m.insert(
        "refreshable".into(),
        json!(matches!(p, Param::Str { refresh: true, .. })),
    );
    let text = |t: &str| if t.is_empty() { Value::Null } else { json!(t) };
    m.insert("mode".into(), json!(source.map(|s| s.state.mode).unwrap_or_default()));
    m.insert("expression".into(), source.map(|s| text(&s.state.expression)).unwrap_or(Value::Null));
    m.insert("reference".into(), source.map(|s| text(&s.state.reference)).unwrap_or(Value::Null));
    m.insert("triggers".into(), json!(source.is_some_and(|s| s.state.triggers)));
    m.insert(
        "error".into(),
        source.and_then(|s| s.error.as_ref()).map(|s| json!(s)).unwrap_or(Value::Null),
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
        }
        Param::Str { options, .. } => {
            m.insert("type".into(), json!("string"));
            m.insert(
                "options".into(),
                options.as_ref().map(|o| json!(o)).unwrap_or(Value::Null),
            );
        }
        Param::Pulse => {
            m.insert("type".into(), json!("pulse"));
        }
    }
    Value::Object(m)
}

/// A param's declared help text; a node's own declaration wins over the owning engine's
/// universal one — resolved once per node by the caller, not once per param.
fn param_doc(
    m: &NodeManifest,
    universal: &[goofi_node::ParamDecl],
    group: &str,
    name: &str,
) -> Option<&'static str> {
    m.params
        .iter()
        .copied()
        .chain(universal.iter().copied())
        .find(|d| d.group == group && d.name == name)
        .and_then(|d| d.doc)
}

/// Type-level params for the palette, and the projection param tooltips are rendered from.
pub fn describe_params(g: &Graph, engine: &str, p: &ParamGroups, m: &'static NodeManifest) -> Value {
    let universal = g.universal_decls(engine, m);
    let mut groups = Map::new();
    for (gname, grp) in p {
        let mut names = Map::new();
        for (n, param) in grp {
            names.insert(n.clone(), describe_param(param, None, param_doc(m, &universal, gname, n)));
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
    let ty = g.node_type(uid).unwrap_or_default();
    let universal = g.universal_decls(goofi_node::split_type_id(&ty).0.unwrap_or_default(), m);
    let mut groups = Map::new();
    for (gname, group) in &*params {
        let mut names = Map::new();
        for (n, param) in group {
            let source = g.param_source(uid, gname, n);
            let mut v = describe_param(param, source.as_ref(), param_doc(m, &universal, gname, n));
            if let (Param::Str { .. }, Some(live)) = (param, g.refreshed_options(uid, gname, n)) {
                v["options"] = json!(live);
            }
            names.insert(n.clone(), v);
        }
        groups.insert(gname.clone(), Value::Object(names));
    }
    Value::Object(groups)
}

/// The live values of a node's expression-driven params, `{group: {name: value}}`. Values only, so
/// the frontend applies it surgically and cannot clobber a concurrent edit.
pub fn expression_value_map(g: &Graph, uid: Uid) -> Value {
    let mut groups = Map::new();
    for (group, name, p) in g.driven_values(uid) {
        let entry = groups.entry(group.to_string()).or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(names) = entry {
            names.insert(name.to_string(), goofi_graph::param_value_json(p));
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
                let names = group.iter().map(|(n, p)| (n.clone(), goofi_graph::param_value_json(p)));
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
pub(crate) fn source_of(g: &Graph, type_name: &str) -> &'static str {
    if g.is_patch_type(type_name) {
        "patch"
    } else {
        "builtin"
    }
}

pub fn node_type_info(g: &Graph, engine: &'static str, m: &'static NodeManifest, source: &str) -> Value {
    let ty = goofi_node::qualify(engine, m.type_name);
    json!({
        "type": ty,
        "engine": engine,
        "source": source,
        "tags": m.tags.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "doc": m.doc,
        "available": true,
        "missing_deps": [],
        "input_slots": input_slots(m),
        "input_multi": input_multi(m),
        "output_slots": output_slots(m),
        // The owning engine's own normalization, so palette and instance agree.
        "params": describe_params(g, engine, &g.default_params_of(&ty, None).unwrap_or_default(), m),
    })
}

/// The `list_nodes` palette catalog, sorted by (engine, bare name). Hidden test nodes
/// (`_`-prefixed) are excluded.
pub fn catalog_types(g: &Graph) -> Value {
    let mut items: Vec<(String, String, Value)> = g
        .library_entries()
        .into_iter()
        .filter(|(_, l)| !l.manifest.type_name.starts_with('_'))
        .map(|(engine, l)| {
            let ty = goofi_node::qualify(engine, l.manifest.type_name);
            let info = node_type_info(g, engine, l.manifest, source_of(g, &ty));
            (engine.to_string(), l.manifest.type_name.to_string(), info)
        })
        .collect();
    // Node files that exist but cannot load are listed too, greyed and with the reason — carrying
    // the shape they last had, because the instances born from it are still running and wired.
    let greyed: Vec<(String, String)> = g
        .unavailable_types()
        .map(|(name, reason)| (name.to_string(), reason.to_string()))
        .collect();
    items.extend(greyed.into_iter().map(|(name, reason)| {
        let last = g.last_manifest(&name);
        let (engine, bare) = goofi_node::split_type_id(&name);
        (
            engine.unwrap_or_default().to_string(),
            bare.to_string(),
            json!({
                "type": name,
                "engine": engine,
                "source": source_of(g, &name),
                "tags": [],
                "doc": format!("This node could not be loaded: {reason}"),
                "available": false,
                "missing_deps": [reason],
                "input_slots": last.map_or_else(|| json!({}), input_slots),
                "input_multi": last.map_or_else(|| json!([]), input_multi),
                "output_slots": last.map_or_else(|| json!({}), output_slots),
                "params": last.map_or_else(|| json!({}), |m| {
                    let params = g.default_params_of(&name, None).unwrap_or_default();
                    describe_params(g, engine.unwrap_or_default(), &params, m)
                }),
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
        m.insert(uid.to_hex(), runtime_json(g, uid));
    }
    Value::Object(m)
}

/// The `{stage, error, runtime}` triple, ONE spelling for every wire site that carries it.
/// `runtime` is absent for a port and a facade, which run nowhere — as `stage` is for inspect.
pub(crate) fn runtime_json(g: &Graph, uid: Uid) -> Value {
    json!({
        "stage": g.node_stage(uid),
        "error": g.last_error(uid),
        "runtime": g.node_tier(uid).map(goofi_node::Isolation::wire),
    })
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
