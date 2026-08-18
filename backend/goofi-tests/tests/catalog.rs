//! The palette — what a user reads BEFORE adding a node, and what a client builds every node from.
//!
//! It is projected from the manifest alone, so everything a row carries has to survive that
//! projection. Registering a node type is boot-time configuration (the CLI's node scan does exactly
//! this), which is why these reach the graph directly rather than through an op: there is no op for
//! "a type that failed to load", and inventing one to test it would be the tail wagging the dog.

use goofi_core::SlotType;
use goofi_node::{Isolation, NodeManifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl};
use goofi_tests::{j, Client, Goofi};
use serde_json::Value;

static OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
fn never() -> Box<dyn goofi_node::Node> {
    unreachable!("the catalog never instantiates")
}

const fn manifest(
    type_name: &'static str,
    inputs: &'static [SlotDecl],
    params: &'static [ParamDecl],
    producer: bool,
) -> NodeManifest {
    NodeManifest {
        type_name,
        category: "test",
        doc: "a catalog fixture",
        inputs,
        outputs: OUT,
        params,
        isolation: Isolation::InProcess,
        producer,
        factory: never,
    }
}

static SOURCE: NodeManifest = manifest("MyPyThing", &[], &[], true);

static MULTI_IN: &[SlotDecl] = &[
    SlotDecl { name: "many", kind: SlotType::Table, trigger_process: true, multi: true, required: false },
    SlotDecl { name: "one", kind: SlotType::Array, trigger_process: true, multi: false, required: false },
];
static TRANSFORM: NodeManifest = manifest("MultiThing", MULTI_IN, &[], false);

static DOCUMENTED_PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "welch",
    name: "nperseg",
    spec: ParamSpec::Int { default: 256, min: 16, max: 4096 },
    expression: None,
    doc: Some("Samples per Welch segment: longer means finer frequency resolution."),
}];
static DOCUMENTED: NodeManifest = manifest("DocumentedThing", &[], DOCUMENTED_PARAMS, true);

static OVERRIDE_PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "common",
    name: "autotrigger",
    spec: ParamSpec::Bool { default: true },
    expression: None,
    doc: Some("On by default: this node is a source."),
}];
static OVERRIDES_COMMON: NodeManifest = manifest("OverridesCommon", &[], OVERRIDE_PARAMS, false);

/// One palette row by type name.
fn row(g: &Goofi, type_name: &str) -> Value {
    g.call("list_nodes", j!({}))["types"]
        .as_array()
        .expect("a palette")
        .iter()
        .find(|v| v["type"] == type_name)
        .unwrap_or_else(|| panic!("{type_name} is in the palette"))
        .clone()
}

#[test]
fn a_producer_is_paced_by_itself_and_a_consumer_by_its_input() {
    // Both fixtures declare no `common.*` param of their own; the only difference is the flag.
    let g = Goofi::new();
    g.register_dyn(&SOURCE, Box::new(|_| never()));
    g.register_dyn(&TRANSFORM, Box::new(|_| never()));

    assert_eq!(row(&g, "MyPyThing")["params"]["common"]["autotrigger"]["value"], true,
               "a source paces itself");
    assert_eq!(row(&g, "MultiThing")["params"]["common"]["autotrigger"]["value"], false,
               "a transform is driven by its input");
}

#[test]
fn a_nodes_own_doc_wins_over_the_universal_common_one() {
    // A node that redeclares a `common.*` param owns its help text too — otherwise a source node's
    // "on by default" tooltip would be replaced by the generic explanation of the flag.
    let g = Goofi::new();
    g.register_dyn(&OVERRIDES_COMMON, Box::new(|_| never()));
    let ty = row(&g, "OverridesCommon");

    assert_eq!(ty["params"]["common"]["autotrigger"]["doc"], "On by default: this node is a source.");
    assert!(ty["params"]["common"]["max_frequency"]["doc"].as_str().unwrap().contains("Rate cap"),
            "the fallback still applies to the rest of the group");
}

#[test]
fn a_params_doc_reaches_the_row_the_tooltip_is_rendered_from() {
    // The frontend renders a param tooltip from the CATALOG descriptor — an instance only overrides
    // value, expression and options — so this is the projection that matters.
    let g = Goofi::new();
    g.register_dyn(&DOCUMENTED, Box::new(|_| never()));
    let ty = row(&g, "DocumentedThing");
    assert_eq!(ty["params"]["welch"]["nperseg"]["doc"],
               "Samples per Welch segment: longer means finer frequency resolution.");
    assert!(!ty["params"]["common"]["autotrigger"]["doc"].is_null(), "common params are documented too");
}

#[test]
fn an_unloadable_node_is_listed_greyed_with_its_reason_already_phrased() {
    let g = Goofi::new();
    g.state.graph.lock().unwrap().register_unavailable("PsdScipy".into(), "scipy".into());
    let ty = row(&g, "PsdScipy");

    assert_eq!(ty["available"], false);
    assert_eq!(ty["missing_deps"], j!(["scipy"]), "the machine-readable reason");
    // `doc` is what the tooltip SHOWS, for an unavailable node as much as an available one, and
    // `reason` is a bare module name only for a ModuleNotFoundError — so the sentence is phrased
    // here, where both cases are known, rather than by every client that renders it.
    assert_eq!(ty["doc"], "This node could not be loaded: scipy");
    assert_eq!(ty["input_slots"], j!({}), "the probe never got far enough to report them");
}

#[test]
fn every_row_says_which_tree_its_type_came_from() {
    // Provenance is "this node came with the patch you opened" — the ONLY thing the scan knows that
    // the catalog cannot re-derive. It has to reach every row INCLUDING the greyed one: a patch
    // whose node needs a missing dependency is exactly where "where did this come from" is asked.
    let g = Goofi::new();
    g.register_dyn(&SOURCE, Box::new(|_| never()));
    {
        let mut graph = g.state.graph.lock().unwrap();
        graph.register_unavailable("PsdScipy".into(), "scipy".into());
        graph.set_patch_types(["MyPyThing".to_string(), "PsdScipy".to_string()].into());
    }
    assert_eq!(row(&g, "MyPyThing")["source"], "patch");
    assert_eq!(row(&g, "PsdScipy")["source"], "patch", "a greyed row is provenanced too");
    assert_eq!(row(&g, "Oscillator")["source"], "builtin", "a compiled-in node ships with goofi");
}

#[test]
fn a_row_carries_its_variadic_slots_its_common_group_and_its_pillar() {
    let g = Goofi::new();
    g.register_dyn(&SOURCE, Box::new(|_| never()));
    g.register_dyn(&TRANSFORM, Box::new(|_| never()));

    // Multi slots are the static shape the frontend reads to render them tall.
    assert_eq!(row(&g, "MultiThing")["input_multi"], j!(["many"]));
    assert_eq!(row(&g, "MyPyThing")["input_multi"], j!([]));

    // The palette must show the same universal `common` group every instantiated node carries, so
    // type-level and instance-level params agree.
    let common = row(&g, "MyPyThing")["params"]["common"].clone();
    assert_eq!(common["max_frequency"]["type"], "float");
    assert_eq!(common["autotrigger"]["type"], "bool");
    assert_eq!(common["frequency_mode"]["type"], "string");

    // The pillar tag rides the contract so a client can route a node to its editor panel.
    assert_eq!(row(&g, "MyPyThing")["pillar"], "signal");

    // `dynamic` was written by both catalog arms and read by nobody, and the two arms disagreed
    // about its meaning. Availability is the only palette-visible distinction; re-add a flag
    // together with the consumer that reads it.
    for ty in g.call("list_nodes", j!({}))["types"].as_array().unwrap() {
        assert!(ty.get("dynamic").is_none(), "{} carries a dynamic flag", ty["type"]);
    }
}

#[tokio::test]
async fn hello_carries_the_palette_and_the_runtime_overlay_and_no_second_graph_projection() {
    // Structure lives in the doc ONLY. The snapshot carries just what the doc cannot: the
    // event-sourced per-node runtime, whose live stream emits TRANSITIONS and so has nothing to
    // give a client that joins a graph already running.
    let g = Goofi::new();
    let a = g.add("Oscillator");
    let b = g.add("Buffer");
    g.link(a, "out", b, "data");
    g.call("set_expression", j!({ "node": goofi_tests::hex(a), "group": "common",
                                 "name": "max_frequency", "expression": "@@@ not an expression @@@",
                                 "enabled": true }));
    g.ready(b);

    let (_c, hello) = Client::connect(&g.serve().await).await;
    assert_eq!(hello["node_types"], g.call("list_nodes", j!({}))["types"],
               "hello embeds the same palette `list_nodes` answers");
    for dead in ["nodes", "links", "instances"] {
        assert!(hello.get(dead).is_none(), "`{dead}` is the doc's job, not the snapshot's");
    }

    let rt = &hello["runtime"];
    assert_eq!(rt.as_object().map(|m| m.len()), Some(2), "one entry per node");
    assert_eq!(rt[goofi_tests::hex(b)]["stage"], "ready");
    assert_eq!(rt[goofi_tests::hex(b)]["error"], Value::Null);
    assert!(rt[goofi_tests::hex(a)]["error"].as_str().is_some_and(|e| !e.is_empty()),
            "a reconnecting client learns the node is errored: {rt}");
}

#[test]
fn graph_replaced_omits_the_catalog_because_a_separate_event_carries_it() {
    // A load brings the patch's own node types and drops the last patch's, so the catalog changes
    // — but it rides `node_types`, not the replacement snapshot, which would otherwise re-ship the
    // whole palette on every load.
    let g = Goofi::new();
    g.add("Oscillator");
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();

    let mut ev = g.events();
    g.call("load_text", j!({ "content": yaml }));
    assert!(ev.next("graph_replaced").get("node_types").is_none(), "the echo omits the catalog");
    assert!(ev.next("node_types")["types"].as_array().is_some_and(|a| !a.is_empty()),
            "…and a separate event carries it");
}
