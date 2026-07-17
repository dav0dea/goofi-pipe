//! JSON projections of the engine graph into the exact shapes the frontend
//! mirrors (`control.ts` types). These are the wire contract; keep field names
//! and shapes aligned with the frontend or co-edit it.

use goofi_core::Param;
use goofi_engine::{ExprInfo, Graph, LinkView, Uid};
use goofi_node::{NodeManifest, ParamGroups};
use serde_json::{json, Map, Value};

pub const ROOT_ID: &str = "__root__";
pub const PROTOCOL_VERSION: i64 = 1;

/// A single param descriptor (discriminated on `type`). `expr` is the instance's
/// expression binding (or `None` for a plain literal / a palette type-level param).
pub fn describe_param(p: &Param, expr: Option<&ExprInfo>) -> Value {
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

/// Type-level / literal params (no expression bindings) — used for the palette.
pub fn describe_params(p: &ParamGroups) -> Value {
    let mut groups = Map::new();
    for (gname, g) in p {
        let mut names = Map::new();
        for (n, param) in g {
            names.insert(n.clone(), describe_param(param, None));
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
    let mut groups = Map::new();
    for (gname, group) in params {
        let mut names = Map::new();
        for (n, param) in group {
            let expr = g.param_expression(uid, gname, n);
            names.insert(n.clone(), describe_param(param, expr.as_ref()));
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

/// The names of the node type's `multi` (variadic) input slots — static shape the
/// frontend reads to render those slots tall and accept many cables. Peer of the
/// dtype in [`input_slots`]; not a mutable per-instance flag.
fn input_multi(m: &NodeManifest) -> Value {
    Value::Array(m.inputs.iter().filter(|s| s.multi).map(|s| json!(s.name)).collect())
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
        "params": describe_params(&goofi_node::with_common(m.default_params())),
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
        "membership": {
            "instance": g.scope_of(uid).map(|s| s.to_hex()).unwrap_or_else(|| ROOT_ID.to_string()),
            "local_name": g.local_of(uid).unwrap_or(&name),
        },
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
    for inst in g.instance_uids() {
        if g.scope_of(inst).is_none() {
            if let Some(i) = g.instance(inst) {
                members.insert(i.name.clone(), json!({ "uid": inst.to_hex(), "is_instance": true }));
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

/// The first errored descendant of an instance (recursing into nested instances), for the
/// collapsed sub-patch's error badge. `Null` if the whole subtree is healthy.
fn instance_error(g: &Graph, uid: Uid) -> Value {
    let Some(inst) = g.instance(uid) else { return Value::Null };
    for muid in inst.members.values() {
        if g.instance(*muid).is_some() {
            let e = instance_error(g, *muid);
            if !e.is_null() {
                return e;
            }
        } else if let Some(err) = g.last_error(*muid) {
            return json!(err);
        }
    }
    Value::Null
}

/// The `InstanceInfo` the frontend types (`control.ts`): kind derived from the def refcount,
/// interface ports chain-resolved to their inner leaf, wired boundaries projected as
/// input/output slots, sibling instances of the same def, and the deep error.
pub fn describe_instance(g: &Graph, uid: Uid) -> Value {
    use goofi_engine::subpatch::Dir;
    let Some(inst) = g.instance(uid) else { return Value::Null };
    let refcount = g.def_refcount(inst.def_id);
    let shared = refcount > 1;

    let mut interface = Map::new();
    let mut in_slots = Map::new();
    let mut out_slots = Map::new();
    if let Some(def) = g.def(inst.def_id) {
        for (bnd, b) in def.interface.iter() {
            let resolved = g.resolve_boundary(uid, bnd);
            interface.insert(
                bnd.clone(),
                json!({
                    "dir": match b.dir { Dir::In => "in", Dir::Out => "out" },
                    "dtype": b.dtype.name(),
                    "inner_node": resolved.as_ref().map(|(u, _)| u.to_hex()),
                    "inner_slot": resolved.as_ref().map(|(_, s)| s.clone()),
                    "pos": b.pos,
                    "name": b.name,
                }),
            );
            if resolved.is_some() {
                match b.dir {
                    Dir::In => in_slots.insert(bnd.clone(), json!(b.dtype.name())),
                    Dir::Out => out_slots.insert(bnd.clone(), json!(b.dtype.name())),
                };
            }
        }
    }

    let mut members = Map::new();
    for (local, muid) in inst.members.iter() {
        members.insert(local.clone(), json!({ "uid": muid.to_hex(), "is_instance": g.instance(*muid).is_some() }));
    }

    let def_id = inst.def_id;
    let siblings: Vec<Value> = g
        .instance_uids()
        .into_iter()
        .filter(|&o| o != uid && g.instance(o).map(|i| i.def_id) == Some(def_id))
        .map(|o| json!(o.to_hex()))
        .collect();

    json!({
        "uid": uid.to_hex(),
        "name": inst.name,
        "kind": if shared { "shared" } else { "unique" },
        "def_id": if shared { json!(def_id.to_hex()) } else { Value::Null },
        "parent": g.scope_of(uid).map(|p| json!(p.to_hex())).unwrap_or(Value::Null),
        "pos": inst.pos,
        "interface": Value::Object(interface),
        "members": Value::Object(members),
        "slots": { "input": Value::Object(in_slots), "output": Value::Object(out_slots) },
        "siblings": siblings,
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
    for inst in g.instance_uids() {
        instances.insert(inst.to_hex(), describe_instance(g, inst));
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
