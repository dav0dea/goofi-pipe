//! JSON projections of the engine graph into the exact shapes the frontend
//! mirrors (`control.ts` types). These are the wire contract; keep field names
//! and shapes aligned with the frontend or co-edit it.

use goofi_core::Param;
use goofi_engine::{Graph, LinkView, Uid};
use goofi_node::{NodeManifest, ParamGroups};
use serde_json::{json, Map, Value};

pub const ROOT_ID: &str = "__root__";
pub const PROTOCOL_VERSION: i64 = 1;

/// A single param descriptor (discriminated on `type`).
pub fn describe_param(p: &Param) -> Value {
    let mut m = Map::new();
    let value = match p {
        Param::Float { value, .. } => json!(value),
        Param::Int { value, .. } => json!(value),
        Param::Bool { value } => json!(value),
        Param::Trigger { fired } => json!(fired),
        Param::Str { value, .. } => json!(value),
    };
    m.insert("value".into(), value);
    m.insert("doc".into(), Value::Null);
    m.insert("save_param".into(), json!(true));
    m.insert(
        "refreshable".into(),
        json!(matches!(p, Param::Str { refresh: Some(_), .. })),
    );
    m.insert("expression".into(), Value::Null);
    m.insert("expression_enabled".into(), json!(false));
    m.insert("expression_triggers_process".into(), json!(false));
    m.insert("expression_autoeval".into(), json!(false));
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

pub fn describe_params(p: &ParamGroups) -> Value {
    let mut groups = Map::new();
    for (gname, g) in p {
        let mut names = Map::new();
        for (n, param) in g {
            names.insert(n.clone(), describe_param(param));
        }
        groups.insert(gname.clone(), Value::Object(names));
    }
    Value::Object(groups)
}

fn input_slots(m: &NodeManifest) -> Value {
    let mut o = Map::new();
    for s in m.inputs {
        o.insert(s.name.to_string(), json!(s.kind.name()));
    }
    Value::Object(o)
}
fn output_slots(m: &NodeManifest) -> Value {
    let mut o = Map::new();
    for s in m.outputs {
        o.insert(s.name.to_string(), json!(s.kind.name()));
    }
    Value::Object(o)
}

pub fn node_type_info(m: &NodeManifest) -> Value {
    json!({
        "type": m.type_name,
        "category": m.category,
        "doc": m.doc,
        "available": true,
        "dynamic": false,
        "missing_deps": [],
        "input_slots": input_slots(m),
        "output_slots": output_slots(m),
        "params": describe_params(&(m.default_params)()),
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

pub fn node_instance_info(g: &Graph, uid: Uid) -> Value {
    let m = g.manifest(uid).expect("node exists");
    let name = g.name(uid).unwrap_or("").to_string();
    json!({
        "uid": uid.to_hex(),
        "name": name,
        "type": g.type_name(uid).unwrap_or(""),
        "category": m.category,
        "doc": m.doc,
        "input_slots": input_slots(m),
        "output_slots": output_slots(m),
        "params": describe_params(g.params(uid).expect("params")),
        "pos": g.pos(uid).unwrap_or([0.0, 0.0]),
        "viewers": {},
        "inputs": {},
        "membership": { "instance": ROOT_ID, "local_name": name },
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

/// The ROOT scope the editor renders its canvas from (no sub-patches yet).
fn root_instance(g: &Graph) -> Value {
    let mut members = Map::new();
    for uid in g.node_uids() {
        let name = g.name(uid).unwrap_or("").to_string();
        members.insert(name, json!({ "uid": uid.to_hex(), "is_instance": false }));
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

/// The full graph snapshot (`hello` / `graph_replaced` payload).
pub fn snapshot(g: &Graph, instance_id: &str, with_protocol: bool) -> Value {
    let nodes: Vec<Value> = g.node_uids().iter().map(|u| node_instance_info(g, *u)).collect();
    let links: Vec<Value> = g.links_view().iter().map(link_info).collect();
    let mut instances = Map::new();
    instances.insert(ROOT_ID.to_string(), root_instance(g));
    let mut snap = json!({
        "instance_id": instance_id,
        "nodes": nodes,
        "links": links,
        "instances": Value::Object(instances),
        "save_path": Value::Null,
        "unsaved_changes": false,
        "layout": Value::Null,
    });
    if with_protocol {
        snap["protocol_version"] = json!(PROTOCOL_VERSION);
    }
    snap
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_node::{Isolation, OutputDecl};

    fn stub_params() -> ParamGroups {
        ParamGroups::new()
    }
    fn stub_make(_: &ParamGroups) -> Box<dyn goofi_node::Node> {
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
        default_params: stub_params,
        isolation: Isolation::InProcess,
        make: stub_make,
    };

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
}
