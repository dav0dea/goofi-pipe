//! The contracts between goofi and everything that reads it: the op registry (which GENERATES the
//! frontend's op union, its word vocabulary and the MCP tool list), the palette a client builds
//! every node from, and the GOOF frame the browser decodes.

use std::collections::HashSet;

use goofi_bridge::ops::{find, registry, typescript};
use goofi_bridge::vocab;
use goofi_core::{Data, Meta, SlotType, Value as DataValue};
use goofi_node::{NodeManifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl};
use goofi_tests::{hex, j, Client, Goofi};
use serde_json::Value;

/// A generated file, kept honest: on drift it is REWRITTEN and the test fails once.
fn regenerated(rel: &str, want: String) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    if std::fs::read_to_string(&path).ok().as_deref() != Some(want.as_str()) {
        std::fs::write(&path, &want).expect("rewriting the generated file");
        panic!("{rel} was stale; it has been regenerated — review and commit it");
    }
}

#[test]
fn every_op_row_is_well_formed_and_reachable() {
    // An op's name is its phrase, words joined with single spaces. The phrase layer resolves a
    // line by the FIRST complete phrase it finds, so the set must be PREFIX-FREE: a phrase that
    // is a word-prefix of another would swallow it whole.
    const ARG_TYPES: &[&str] = &["uid", "string", "float", "int", "bool", "float2", "json",
                                 "any", "param_addr", "endpoint", "panel_type",
                                 "uid[]", "string[]", "float[]", "json[]"];
    // ONE bare namespace: the registry and the client's reserved phrases are prefix-free
    // together, which is what lets `session list` sit beside `session status`.
    let namespace: Vec<&str> =
        registry().iter().map(|o| o.name).chain(goofi_bridge::ops::RESERVED.iter().copied()).collect();
    let mut seen = HashSet::new();
    for name in &namespace {
        assert!(seen.insert(*name), "`{name}` is declared twice");
        let words: Vec<&str> = name.split(' ').collect();
        assert!(!words.is_empty() && words.iter().all(|w| !w.is_empty()
                && w.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')),
                "`{name}` is not space-joined [a-z0-9_]+ words");
        for other in &namespace {
            let shorter: Vec<&str> = other.split(' ').collect();
            assert!(name == other || words.get(..shorter.len()) != Some(&shorter[..]),
                    "`{other}` is a word-prefix of `{name}` and would swallow it");
        }
    }
    for op in registry() {
        // The args schema is a STRING, so a typo in it would otherwise be a fact only at read time.
        assert_eq!(op.args().count(), op.args.split_whitespace().count(),
                   "`{}` has an argument with no `name:type`: {:?}", op.name, op.args);
        for (arg, ty, _) in op.args() {
            assert!(ARG_TYPES.contains(&ty), "`{}`'s `{arg}` has unknown type `{ty}`", op.name);
            // `--json` is client-consumed and `--help` server-intercepted, ANYWHERE on a line —
            // an op declaring either would silently never receive it.
            assert!(arg != "json" && arg != "help",
                    "`{}` declares the reserved flag `--{arg}`", op.name);
        }
        assert!(!op.doc.is_empty() && !op.result.is_empty(), "`{}` is undocumented", op.name);
        assert!(op.positional <= op.args().count() && op.positional <= 2,
                "`{}` claims more positionals than it declares args", op.name);
        assert!(!op.doc().contains("{panel_types}") && !op.doc().contains("{viewer_kinds}") && !op.doc().contains("{boundary_types}"),
                "`{}` has an unexpanded placeholder — a model would read it verbatim", op.name);
    }
    // The `!` has to reach the parse, or every argument is advertised as optional.
    let add: Vec<_> = find("node add").expect("node add is registered").args().collect();
    assert_eq!((add[0], add[1]), (("type", "string", true), ("pos", "float2", false)));

    // A row with no dispatch arm answers `unknown op` while palette and tool list advertise it.
    let g = Goofi::new();
    for op in registry() {
        if let Err(e) = g.try_call(op.name, j!({})) {
            assert!(!e.contains(&format!("unknown op `{}`", op.name)),
                    "`{}` is in the registry but dispatch has no arm for it: {e}", op.name);
        }
    }
}

#[test]
fn the_generated_frontend_artifacts_still_match_the_tables_they_come_from() {
    regenerated("frontend/src/lib/api/ops.ts", typescript());
    regenerated("frontend/src/lib/api/vocab.ts", vocab::typescript());

    // `PROTOCOL_VERSION` is the one number both halves declare by hand, and each comments that the
    // other must be bumped with it — which is the definition of a pair that drifts. A client one
    // version behind still connects and then half-works, so neither side's suite can catch it.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../frontend/src/lib/api/control.ts");
    let src = std::fs::read_to_string(&path).expect("the control client");
    let declared: i64 = src
        .split("export const PROTOCOL_VERSION")
        .nth(1)
        .and_then(|rest| rest.split(';').next())
        .and_then(|rest| rest.trim_start_matches([' ', '=', ':']).trim().parse().ok())
        .unwrap_or_else(|| panic!("no PROTOCOL_VERSION in {}", path.display()));
    assert_eq!(
        declared,
        goofi_bridge::schemas::PROTOCOL_VERSION,
        "the client declares protocol {declared} and this manager speaks {} — bump both together",
        goofi_bridge::schemas::PROTOCOL_VERSION
    );
}

#[test]
fn a_vocabulary_word_is_emittable_documented_and_offered_where_it_is_asked_for() {
    // Each op that takes a vocabulary word enumerates the set in its own description, by expansion.
    let doc = find("layout panel edit").expect("registered").doc();
    for word in ["parameters", "node-editor", "viewer", "line", "trajectory", "topomap"] {
        assert!(doc.contains(word), "`{word}` is not offered by the panel edit doc: {doc}");
    }
    // The description is the ONLY text an agent reads, so node edit's has to carry the viewer
    // vocabulary, and node param edit's the two words that decide what an expression does.
    let doc = find("node edit").expect("registered").doc();
    for word in ["line", "topomap", "table"] {
        assert!(doc.contains(word), "`{word}` is not offered by node edit's doc: {doc}");
    }
    let doc = find("node param edit").expect("registered").doc();
    for word in ["`triggers` defaults false", "triggers: true"] {
        assert!(doc.contains(word), "`{word}` is not offered by node param edit's doc: {doc}");
    }

    // The generator emits TS string literals with NO escaping, so a quote or newline breaks the file.
    let mut seen = HashSet::new();
    for (id, doc) in vocab::PANEL_TYPES.iter().map(|p| (p.id, p.doc))
        .chain(vocab::VIEWER_KINDS.iter().map(|k| (k.id, k.doc)))
    {
        assert!(seen.insert(id), "`{id}` is declared twice");
        assert!(!doc.is_empty(), "`{id}` is undocumented");
        for s in [id, doc] {
            assert!(!s.contains('\'') && !s.contains('\\') && !s.contains('\n'), "unquotable: {s}");
        }
    }
    // The engine mints panel entries of its own, each naming a type as a bare string.
    for ty in [goofi_graph::layout::DEFAULT_PANEL_TYPE, goofi_graph::layout::EMPTY_PANEL_TYPE] {
        assert!(vocab::panel_type(ty).is_some(), "`{ty}` is not a declared panel type");
    }
    // A kind's ViewSpec has to accept everything its component draws, or the frame is filtered out.
    for k in vocab::VIEWER_KINDS {
        if let vocab::Draws::Array { draws, accepts } = k.draws {
            assert!(draws.0 <= draws.1 && accepts.0 <= accepts.1, "`{}` has an empty range", k.id);
            assert!(accepts.0 <= draws.0 && accepts.1 >= draws.1,
                    "`{}` draws {draws:?} but its ViewSpec accepts only {accepts:?}", k.id);
        }
    }
}

static OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
fn never() -> Box<dyn goofi_signal_sdk::Node> {
    unreachable!("the catalog never instantiates")
}
const fn manifest(type_name: &'static str, inputs: &'static [SlotDecl],
                  params: &'static [ParamDecl], producer: bool) -> NodeManifest {
    NodeManifest { type_name, category: "test", doc: "a catalog fixture", inputs, outputs: OUT,
                   params, producer }
}
static SOURCE: NodeManifest = manifest("MyPyThing", &[], &[], true);
static MULTI_IN: &[SlotDecl] = &[
    SlotDecl { name: "many", kind: SlotType::Table, trigger_process: true, multi: true, required: false },
    SlotDecl { name: "one", kind: SlotType::Array, trigger_process: true, multi: false, required: false },
];
static TRANSFORM: NodeManifest = manifest("MultiThing", MULTI_IN, &[], false);
static DOCUMENTED_PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "welch", name: "nperseg", spec: ParamSpec::Int { default: 256, min: 16, max: 4096 },
    expression: None,
    doc: Some("Samples per Welch segment: longer means finer frequency resolution."),
}];
static DOCUMENTED: NodeManifest = manifest("DocumentedThing", &[], DOCUMENTED_PARAMS, true);
static OVERRIDE_PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "common", name: "autotrigger", spec: ParamSpec::Bool { default: true },
    expression: None, doc: Some("On by default: this node is a source."),
}];
static OVERRIDES_COMMON: NodeManifest = manifest("OverridesCommon", &[], OVERRIDE_PARAMS, false);

fn row(g: &Goofi, type_name: &str) -> Value {
    g.call("library list", j!({}))["types"].as_array().expect("a palette").iter()
        .find(|v| v["type"] == type_name)
        .unwrap_or_else(|| panic!("{type_name} is in the palette")).clone()
}

#[test]
fn a_palette_row_carries_everything_a_client_renders_a_node_from() {
    // Registering a type is boot-time configuration, and there is no op for "a type that failed to load".
    let g = Goofi::new();
    g.register_dyn(&SOURCE, Box::new(|_| never()), &goofi_node::NATIVE);
    g.register_dyn(&TRANSFORM, Box::new(|_| never()), &goofi_node::NATIVE);
    g.register_dyn(&DOCUMENTED, Box::new(|_| never()), &goofi_node::NATIVE);
    g.register_dyn(&OVERRIDES_COMMON, Box::new(|_| never()), &goofi_node::NATIVE);

    // The two fixtures differ only in the `producer` flag, and it decides who paces the node.
    assert_eq!(row(&g, "MyPyThing")["params"]["common"]["autotrigger"]["value"], true,
               "a source paces itself");
    assert_eq!(row(&g, "MultiThing")["params"]["common"]["autotrigger"]["value"], false,
               "a transform is driven by its input");
    let common = row(&g, "MyPyThing")["params"]["common"].clone();
    assert_eq!((&common["max_frequency"]["type"], &common["autotrigger"]["type"],
                &common["frequency_mode"]["type"]), (&j!("float"), &j!("bool"), &j!("string")));
    assert_eq!(row(&g, "MultiThing")["input_multi"], j!(["many"]));
    assert_eq!(row(&g, "MyPyThing")["input_multi"], j!([]));

    // A tooltip is rendered from the CATALOG descriptor, so a node that redeclares a `common.*`
    // param owns its help text too.
    assert_eq!(row(&g, "DocumentedThing")["params"]["welch"]["nperseg"]["doc"],
               "Samples per Welch segment: longer means finer frequency resolution.");
    let overridden = row(&g, "OverridesCommon");
    assert_eq!(overridden["params"]["common"]["autotrigger"]["doc"],
               "On by default: this node is a source.");
    assert!(overridden["params"]["common"]["max_frequency"]["doc"].as_str().unwrap()
                .contains("Rate cap"), "the fallback still applies to the rest of the group");
}

#[test]
fn a_node_that_could_not_load_explains_itself_instead_of_vanishing() {
    let g = Goofi::new();
    {
        let mut graph = g.state.graph.lock().unwrap();
        graph.register_unavailable("PsdScipy".into(), "scipy".into());
        goofi_bridge::register_dyn_type(&mut graph, &SOURCE, Box::new(|_| never()), &goofi_node::NATIVE);
        // Provenance is the only thing the scan knows that the catalog cannot re-derive, greyed rows too.
        graph.set_patch_types(["MyPyThing".to_string(), "PsdScipy".to_string()].into());
    }
    let ty = row(&g, "PsdScipy");
    assert_eq!(ty["available"], false);
    assert_eq!(ty["missing_deps"], j!(["scipy"]), "the machine-readable reason");
    // `reason` is a bare module name only for a ModuleNotFoundError, so the sentence is phrased here.
    assert_eq!(ty["doc"], "This node could not be loaded: scipy");
    assert_eq!(ty["input_slots"], j!({}), "the probe never got far enough to report them");

    assert_eq!(ty["source"], "patch", "a greyed row is provenanced too");
    assert_eq!(row(&g, "MyPyThing")["source"], "patch");
    assert_eq!(row(&g, "Oscillator")["source"], "builtin", "a shipped node ships with goofi");
}

#[tokio::test]
async fn the_palette_rides_the_snapshot_and_the_graph_never_does() {
    // The snapshot carries only what the doc cannot: the per-node runtime, whose stream emits transitions.
    let g = Goofi::new();
    let a = g.add("Oscillator");
    let b = g.add("Buffer");
    g.link(a, "out", b, "data");
    g.call("node param edit", j!({ "node": hex(a), "param": "common/max_frequency",
                                   "expression": "@@@ not an expression @@@" }));
    g.ready(b);

    let (_c, hello) = Client::connect(&g.serve().await).await;
    assert_eq!(hello["node_types"], g.call("library list", j!({}))["types"],
               "hello embeds the same palette `library list` answers");
    for dead in ["nodes", "links", "instances"] {
        assert!(hello.get(dead).is_none(), "`{dead}` is the doc's job, not the snapshot's");
    }
    let rt = &hello["runtime"];
    assert_eq!(rt.as_object().map(|m| m.len()), Some(2), "one entry per node");
    assert_eq!((&rt[hex(b)]["stage"], &rt[hex(b)]["error"]), (&j!("ready"), &Value::Null));
    assert!(rt[hex(a)]["error"].as_str().is_some_and(|e| !e.is_empty()),
            "a reconnecting client learns the node is errored: {rt}");

    // The catalog rides its OWN event, not the replacement snapshot, which would re-ship on every load.
    let yaml = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    let mut ev = g.events();
    g.call("session load", j!({ "content": yaml }));
    assert!(ev.next("graph_replaced").get("node_types").is_none(), "the echo omits the catalog");
    assert!(ev.next("node_types")["types"].as_array().is_some_and(|a| !a.is_empty()),
            "…and a separate event carries it");
}

// The GOOF frame — mirrored in `frontend/src/lib/codec/`.

#[test]
fn a_frame_survives_the_round_trip_the_browser_makes_it_do() {
    // Array data is ALWAYS f32 on the wire, and a meta entry shadowing a header key is written once.
    let labels = goofi_core::Axes::new()
        .with(0, goofi_core::Axis::coords(vec![goofi_core::Coord::Str("Fz".into()),
                                               goofi_core::Coord::Str("Cz".into())]));
    let meta = Meta::new().with_sfreq(Some(250.0)).with_channels(labels.clone());
    let body: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let d = Data::array_f32(vec![2, 4], body.clone(), meta).unwrap();
    let back = goofi_codec::decode(&goofi_codec::encode(&d)).expect("a frame goofi wrote");
    let DataValue::Array(a) = back.value() else { panic!("not an array") };
    assert_eq!((a.shape(), a.as_bytes()), (&[2usize, 4][..], &body[..]));
    assert_eq!(back.meta().sfreq(), Some(250.0));
    assert_eq!(back.meta().channels(), &labels, "positional axis labels ride the frame");

    for (name, d) in [("a string", Data::string(String::from("hello"), Meta::new())),
                      ("a table", Data::table(Default::default(), Meta::new()))] {
        let raw = goofi_codec::encode(&d);
        assert!(goofi_codec::decode(&raw).is_ok(), "{name} did not survive the round trip");
    }
}

#[test]
fn a_malformed_frame_is_refused_rather_than_trusted() {
    // The decoder reads lengths out of the frame, and runs in the browser: it must refuse, never panic.
    let good = goofi_codec::encode(&Data::array_f32(
        vec![2], vec![0u8; 8], Meta::new().with_sfreq(Some(1.0))).unwrap());
    assert!(goofi_codec::decode(&good).is_ok(), "the fixture is a frame that DOES decode");

    assert!(goofi_codec::decode(b"NOPE").is_err(), "bad magic");
    for cut in 0..good.len() {
        // Every prefix — the shape a partially-flushed socket delivers.
        assert!(goofi_codec::decode(&good[..cut]).is_err(), "a truncated frame decoded at {cut}");
    }
    let mut wrong_tag = good.clone();
    wrong_tag[5] = 0x7f;
    assert!(goofi_codec::decode(&wrong_tag).is_err(), "an unknown dtype tag");
}

#[test]
fn every_slot_name_is_letters_and_digits() {
    // A reference spells `node.slot`, and an expression reads a slot as an attribute: one rule,
    // held by every manifest a fresh goofi offers — the shipped nodes and the fixtures.
    let g = Goofi::new();
    let graph = g.state.graph.lock().unwrap();
    for m in graph.library_manifests() {
        let slots = m.inputs.iter().map(|s| s.name).chain(m.outputs.iter().map(|o| o.name));
        for slot in slots {
            assert!(goofi_core::globals::is_valid_name(slot),
                    "{}: slot `{slot}` — {}", m.type_name, goofi_core::globals::NAME_RULE);
        }
    }
    assert!(goofi_core::globals::is_valid_name("out") && goofi_core::globals::is_valid_name("cutoff2"));
    for bad in ["in", "max_frequency", "2x", "", "a.b"] {
        assert!(!goofi_core::globals::is_valid_name(bad), "`{bad}` must be refused");
    }
    // The graph mints a display name from the type, `_Test*` included: what it mints must pass
    // its own rule, or a node is born under a name nothing can reference.
    for m in graph.library_manifests() {
        let minted = format!("{}0", goofi_graph::name_base(m.type_name));
        assert!(goofi_core::globals::is_valid_name(&minted), "{}: minted `{minted}`", m.type_name);
    }
}

#[test]
fn every_declared_expression_reads_only_a_global_a_fresh_patch_has() {
    // Cheap and evaluator-free: a typo'd `globals.defualt_ufreq` compiles, binds, then errors on every
    // instance. Read AS EACH TYPE SEES IT, since a declaration may condition on the manifest.
    let g = Goofi::new();
    let graph = g.state.graph.lock().unwrap();
    let decls = graph.library_manifests().into_iter().flat_map(|m| {
        m.params.iter().copied().chain(graph.universal_decls_of(m.type_name)).map(move |d| (m.type_name, d))
    });
    for (owner, decl) in decls {
        let Some(expr) = decl.expression else { continue };
        assert!(!expr.source.trim().is_empty(),
                "{owner}: {}/{} has an empty expression", decl.group, decl.name);
        for read in goofi_node::scan_globals(expr.source) {
            assert!(goofi_core::globals::SYSTEM_GLOBALS.iter().any(|g| g.name == read.name),
                    "{owner}: the expression on {}/{} reads `globals.{}`, which no fresh patch has",
                    decl.group, decl.name, read.name);
        }
    }
}

#[test]
fn every_test_node_is_registered_and_hidden_from_the_palette() {
    // Both or neither: registering without the `_` prefix ships a product node, and the prefix
    // without registration is invisible to the tests it exists for.
    let g = Goofi::new();
    let names: Vec<&str> = g.state.graph.lock().unwrap().library_manifests().into_iter().map(|m| m.type_name).collect();
    for want in ["_TestEcho", "_TestSink", "_TestFail", "_TestPanic", "_TestSetupFail", "_TestSlow",
                 "_TestCounter", "_TestRequired", "_TestPicker", "_TestMute", "_TestConst"] {
        assert!(names.contains(&want), "{want} is not in the catalog: {names:?}");
        assert!(want.starts_with('_'), "{want} would show in the palette");
    }
    let palette = g.state.call("library list", j!({}), "t").unwrap();
    let listed: Vec<&str> = palette["types"].as_array().unwrap().iter()
        .map(|t| t["type"].as_str().unwrap()).collect();
    assert!(!listed.iter().any(|t| t.starts_with('_')), "a test node reached the palette: {listed:?}");
    assert!(listed.contains(&"Oscillator") && listed.contains(&"Buffer"), "{listed:?}");
}

#[test]
fn the_control_plane_document_carries_no_null_leaf() {
    // A delta is an RFC 7386 merge patch, which spends `null` on "delete this key", so a null leaf
    // would be ambiguous. If one is ever needed this fails and NAMES the path.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    g.call("node param edit", j!({ "node": hex(osc), "param": "oscillator/frequency",
                                   "expression": "globals.default_ufreq" }));
    g.call("global add", j!({ "name": "subject", "value": "P07", "type": "string" }));
    let inst = g.call("nodes group", j!({ "nodes": [hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    // Grouping a node fed from outside mints a WIRED port, which is how this reaches the two
    // optional leaves a scope has: a record's `scope` key, and the link that is its inner wire.
    let doc = g.doc();
    let ports: Vec<String> = doc["nodes"].as_object().unwrap().iter()
        .filter(|(_, n)| n["scope"] == inst).map(|(u, _)| u.clone()).collect();
    assert!(!ports.is_empty(), "the group left no port: {}", doc["nodes"]);
    assert!(doc["links"].as_array().unwrap().iter().any(|l| {
                ports.iter().any(|p| l["node_out"] == p.as_str() || l["node_in"] == p.as_str())
            }),
            "no port is wired, so this test would not reach the inner-wire link: {}", doc["links"]);
    g.call("layout panel add", j!({ "beside": panel_id(&g), "side": "right", "ratio": 0.5 }));

    let doc = g.doc();
    for root in ["nodes", "links", "globals", "arrangement"] {
        assert!(doc.get(root).is_some(), "the document is missing its `{root}` root: {doc}");
    }
    let mut nulls = Vec::new();
    find_nulls(&doc, &mut Vec::new(), &mut nulls);
    assert!(nulls.is_empty(), "these leaves are null, so a merge patch cannot express them: {nulls:?}");
}

fn panel_id(g: &Goofi) -> String {
    goofi_tests::panel_ids(&g.doc()["arrangement"]).first().cloned().expect("the default panel")
}

fn find_nulls(v: &Value, path: &mut Vec<String>, out: &mut Vec<String>) {
    match v {
        Value::Null => out.push(path.join(".")),
        Value::Object(m) => {
            for (k, x) in m {
                path.push(k.clone());
                find_nulls(x, path, out);
                path.pop();
            }
        }
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                path.push(i.to_string());
                find_nulls(x, path, out);
                path.pop();
            }
        }
        _ => {}
    }
}
