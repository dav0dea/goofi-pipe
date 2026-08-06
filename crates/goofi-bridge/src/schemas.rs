//! JSON projections of the engine graph into the exact shapes the frontend
//! mirrors (`control.ts` types). These are the wire contract; keep field names
//! and shapes aligned with the frontend or co-edit it.

use goofi_core::Param;
use goofi_engine::{ExprInfo, Graph, Uid};
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
    let mut items: Vec<(String, String, Value)> = goofi_node::catalog()
        .chain(g.dyn_type_manifests())
        .filter(|m| !m.type_name.starts_with('_'))
        .map(|m| (m.category.to_string(), m.type_name.to_string(), node_type_info(m)))
        .collect();
    // Node files that exist but cannot load are listed too, greyed and with the reason. Dropping
    // them would leave the author of a node with an uninstalled dependency staring at a palette
    // where it simply is not, which reads as "my file was ignored".
    items.extend(g.unavailable_types().map(|(name, reason)| {
        (
            "unavailable".to_string(),
            name.to_string(),
            json!({
                "type": name,
                "pillar": "signal",
                "category": "unavailable",
                "doc": format!("This node could not be loaded: {reason}"),
                "available": false,
                "dynamic": true,
                "missing_deps": [reason],
                "input_slots": {},
                "input_multi": [],
                "output_slots": {},
                "params": {},
            }),
        )
    }));
    items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    Value::Array(items.into_iter().map(|(_, _, v)| v).collect())
}

/// The per-node RUNTIME overlay: the event-sourced state that never enters the CRDT doc, keyed by
/// node uid. Rides the `hello`/`graph_replaced` snapshot because its live stream (the stats sweep)
/// pushes only *transitions* — without this seed a client that connects to a running graph would
/// show an errored node as healthy until it happened to change.
pub fn runtime_overlay(g: &Graph) -> Value {
    let mut m = Map::new();
    for uid in g.node_uids() {
        m.insert(
            uid.to_hex(),
            json!({ "stage": g.node_stage(uid), "error": g.last_error(uid) }),
        );
    }
    Value::Object(m)
}

/// The snapshot (`hello` / `graph_replaced` payload). Deliberately carries NO graph structure:
/// nodes, links and the sub-patch forest live in the CRDT doc alone (the client assembles them
/// from doc + catalog). What it does carry is the session frame — instance id, palette, save
/// path, layout — plus [`runtime_overlay`], the one piece of per-node truth the doc never holds.
pub fn snapshot(g: &Graph, instance_id: &str, with_protocol: bool, unsaved: bool) -> Value {
    let mut snap = json!({
        "instance_id": instance_id,
        // The pillars this backend build actually hosts — the frontend shows only these
        // editors. Signal-only for now; audio/video are added as their runtimes land.
        "pillars": ["signal"],
        "runtime": runtime_overlay(g),
        "save_path": Value::Null,
        "unsaved_changes": unsaved,
        "layout": g.layout().clone(),
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

    static OVERRIDE_PARAMS: &[ParamDecl] = &[ParamDecl {
        group: "common",
        name: "autotrigger",
        spec: goofi_node::ParamSpec::Bool { default: true },
        default_expr: None,
        doc: Some("On by default: this node is a source."),
    }];
    static OVERRIDE_MANIFEST: NodeManifest = NodeManifest {
        type_name: "OverridesCommon",
        category: "test",
        doc: "declares its own common.autotrigger",
        inputs: &[],
        outputs: T_OUT,
        params: OVERRIDE_PARAMS,
        isolation: Isolation::InProcess,
        factory: stub_factory,
    };

    #[test]
    fn a_nodes_own_doc_wins_over_the_universal_common_one() {
        // A node that redeclares a `common.*` param owns its help text too — otherwise a source
        // node's "on by default" tooltip would be replaced by the generic explanation of what
        // autotrigger means in general.
        let mut g = Graph::new();
        g.register_dyn_type(&OVERRIDE_MANIFEST, Box::new(|_| unreachable!()));
        let cat = catalog_types(&g);
        let ty = cat.as_array().unwrap().iter().find(|v| v["type"] == "OverridesCommon").unwrap();

        assert_eq!(ty["params"]["common"]["autotrigger"]["doc"], json!("On by default: this node is a source."));
        // The params it did NOT redeclare still get the universal text.
        assert!(
            ty["params"]["common"]["max_frequency"]["doc"].as_str().unwrap().contains("Rate cap"),
            "the fallback still applies to the rest of the group"
        );
    }

    #[test]
    fn an_unloadable_node_is_listed_greyed_with_its_reason() {
        let mut g = Graph::new();
        g.register_unavailable("PsdScipy".into(), "scipy".into());
        let cat = catalog_types(&g);
        let ty = cat
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["type"] == "PsdScipy")
            .expect("an unloadable node is still listed");

        assert_eq!(ty["available"], json!(false));
        assert_eq!(ty["missing_deps"], json!(["scipy"]), "the machine-readable reason");
        // `doc` is what the palette tooltip SHOWS, for an unavailable node as much as an
        // available one — `reason` is a bare module name only for a ModuleNotFoundError, so
        // the sentence has to be phrased here, where both cases are known.
        assert_eq!(
            ty["doc"],
            json!("This node could not be loaded: scipy"),
            "the reason arrives already phrased, so no client has to guess at it"
        );
        // No slots or params are known — the probe never got far enough to report them.
        assert_eq!(ty["input_slots"], json!({}));
    }

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
        let hello = snapshot(&g, "iid", true, false);
        assert_eq!(
            hello["node_types"],
            catalog_types(&g),
            "hello embeds the same palette `list_nodes` returns"
        );
        assert!(hello["node_types"].as_array().is_some_and(|a| !a.is_empty()));
        // A structural echo (subpatch_changed, with_protocol=false) must NOT re-ship the whole
        // catalog on every group/expand/share — it changes only when a runtime type registers.
        let echo = snapshot(&g, "iid", false, false);
        assert!(echo.get("node_types").is_none(), "structural echoes omit the catalog");
    }

    #[test]
    fn the_snapshot_carries_no_second_graph_projection_only_the_runtime_overlay() {
        // SSOT: structure (nodes/links/scopes) lives in the CRDT doc ONLY. The snapshot carries
        // just what the doc cannot: the event-sourced per-node runtime, whose stream (the stats
        // sweep) emits transitions and so has no value to give a freshly-connected client.
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        g.add_link(a, "out", b, "data").unwrap();
        g.set_expression(a, "common", "max_frequency", "@@@ not an expression @@@", true, false).unwrap();

        let snap = snapshot(&g, "iid", true, false);
        for dead in ["nodes", "links", "instances"] {
            assert!(snap.get(dead).is_none(), "`{dead}` is the doc's job, not the snapshot's");
        }

        let rt = &snap["runtime"];
        assert_eq!(rt.as_object().map(|m| m.len()), Some(2), "one entry per node");
        assert_eq!(rt[b.to_hex()]["error"], Value::Null);
        assert_eq!(rt[b.to_hex()]["stage"], json!("ready"));
        assert!(
            rt[a.to_hex()]["error"].as_str().is_some_and(|e| !e.is_empty()),
            "a reconnecting client learns the node is errored: {:?}",
            rt[a.to_hex()]["error"]
        );
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
