//! The contracts between goofi and everything that reads it: the op registry (which GENERATES the
//! frontend's op union, its word vocabulary and the MCP tool list), the palette a client builds
//! every node from, and the GOOF frame the browser decodes.
//!
//! These are the only tests here that assert about a table rather than about a session. They earn
//! that because the artifacts they judge are consumed OUTSIDE this process — a stale generated file
//! or a drifted wire format is not a bug any scenario can reach, and the failure lands in a
//! different language a commit later.

use std::collections::HashSet;

use goofi_bridge::ops::{find, typescript, Surface, MCP_PREFIX, REGISTRY};
use goofi_bridge::vocab;
use goofi_core::{Data, Meta, SlotType, Value as DataValue};
use goofi_node::{Isolation, NodeManifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl};
// Linked for its side effect: the catalog is an `inventory` registry, and a crate nothing NAMES is
// a crate rustc does not link — so a walk over the catalog would find no native node at all.
use goofi_nodes as _;
use goofi_tests::{hex, j, Client, Goofi};
use serde_json::Value;

/// A generated file, kept honest. On drift it is REWRITTEN and the test fails once, so the fix is
/// to re-run and commit rather than to hand-transcribe a table.
fn regenerated(rel: &str, want: String) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    if std::fs::read_to_string(&path).ok().as_deref() != Some(want.as_str()) {
        std::fs::write(&path, &want).expect("rewriting the generated file");
        panic!("{rel} was stale; it has been regenerated — review and commit it");
    }
}

#[test]
fn every_op_row_is_well_formed_and_reachable() {
    // The registry closed two directions at once: `dispatch` was a string-keyed match with no way
    // to say "this arm is missing", and `op` was a free string at scattered call sites. What is
    // pinned here is what a malformed row costs — a name outside `[a-z0-9_]+`, or one pushing
    // `mcp__goofi__<name>` past 64 characters, makes a provider reject the ENTIRE tool list with a
    // 400; a duplicate name does the same AND makes `find` silently prefer the first.
    const ARG_TYPES: &[&str] = &["uid", "string", "float", "int", "bool", "float2", "json",
                                 "panel_type", "uid[]", "string[]", "float[]"];
    let mut seen = HashSet::new();
    for op in REGISTRY {
        assert!(seen.insert(op.name), "`{}` is declared twice", op.name);
        assert!(!op.name.is_empty()
                && op.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "`{}` is not [a-z0-9_]+", op.name);
        assert!(MCP_PREFIX.len() + op.name.len() <= 64,
                "`{MCP_PREFIX}{}` is over the 64 characters a tool name may have", op.name);
        // The args schema is a STRING, so a typo in it would otherwise be a fact only at read time.
        assert_eq!(op.args().count(), op.args.split_whitespace().count(),
                   "`{}` has an argument with no `name:type`: {:?}", op.name, op.args);
        for (arg, ty, _) in op.args() {
            assert!(ARG_TYPES.contains(&ty), "`{}`'s `{arg}` has unknown type `{ty}`", op.name);
        }
        assert!(!op.doc.is_empty() && !op.result.is_empty(), "`{}` is undocumented", op.name);
        assert!(!op.doc().contains("{panel_types}") && !op.doc().contains("{viewer_kinds}"),
                "`{}` has an unexpanded placeholder — a model would read it verbatim", op.name);
    }
    // The `!` has to reach the parse, or every argument is advertised as optional and a model omits
    // the one the op cannot run without.
    let add: Vec<_> = find("add_node").expect("add_node is registered").args().collect();
    assert_eq!((add[0], add[1]), (("type", "string", true), ("pos", "float2", false)));

    // A row with no dispatch arm falls through to the catch-all and answers `unknown op` — an op
    // the palette, the tool list and the frontend's union all advertise and nothing can call. (The
    // converse needs no test: the gate refuses an unregistered op before the match is reached, so
    // an arm without a row is unreachable rather than silently live.)
    let g = Goofi::new();
    for op in REGISTRY {
        if let Err(e) = g.try_call(op.name, j!({})) {
            assert!(!e.contains(&format!("unknown op `{}`", op.name)),
                    "`{}` is in the registry but dispatch has no arm for it: {e}", op.name);
        }
    }
}

#[test]
fn the_ops_kept_off_the_agent_surface_are_a_decision_and_are_named_here() {
    // `surface` is the one column with a SAFETY consequence, and the whole tool list is generated
    // from it — so it is pinned as a SET, not as a property. Each name below either replaces the
    // patch an agent is working inside (with, for the three sharing the `load` arm, its undo
    // history), is the human file browser's half of that door, or is a harness op: an agent able to
    // spawn or kill a harness could spawn itself a peer, or terminate the process it speaks
    // through. Adding a row here is a decision; this is where it gets made deliberately.
    let control_only: Vec<&str> =
        REGISTRY.iter().filter(|o| o.surface == Surface::ControlOnly).map(|o| o.name).collect();
    assert_eq!(control_only, ["list_dir", "set_viewpoint", "serialize", "save", "load_text",
                              "load", "new", "list_harnesses", "spawn_harness", "stop_harness"]);
}

#[test]
fn the_generated_frontend_artifacts_still_match_the_tables_they_come_from() {
    regenerated("frontend/src/lib/api/ops.ts", typescript());
    regenerated("frontend/src/lib/api/vocab.ts", vocab::typescript());
}

#[test]
fn a_vocabulary_word_is_emittable_documented_and_offered_where_it_is_asked_for() {
    // A caller that has to GUESS a vocabulary word gets it wrong (`params` for `parameters`), and
    // the guess used to be answered `{ok: true}`. So each op that takes one enumerates the set in
    // its own description — by expansion, not by a hand-copied list, which would be the very
    // duplication `vocab.rs` exists to remove.
    let doc = find("page_set_panel").expect("registered").doc();
    for word in ["parameters", "node-editor", "viewer", "line", "trajectory", "topomap"] {
        assert!(doc.contains(word), "`{word}` is not offered by page_set_panel's doc: {doc}");
    }
    let doc = find("set_node_viewers").expect("registered").doc();
    for word in ["line", "topomap", "table"] {
        assert!(doc.contains(word), "`{word}` is not offered by set_node_viewers's doc: {doc}");
    }
    // Agents set `triggers: true` on every expression they bound. The description is the ONLY text
    // they read — the tool list projects `doc` + `result`, and the input schema carries no
    // per-argument text — and this doc named NEITHER boolean, so both read as one "turn it on".
    let doc = find("set_expression").expect("registered").doc();
    for phrase in ["`enabled` defaults false", "`triggers` defaults false", "enabled: true"] {
        assert!(doc.contains(phrase), "set_expression's doc does not say {phrase:?}: {doc}");
    }

    // The generator emits TypeScript string literals with NO escaping, so a word carrying a quote
    // or a newline emits a file that does not parse — caught here rather than by `npm run check`.
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
    // The engine mints panel entries of its own — the default page's, and the empty one a split
    // births — each naming a type as a bare string, so both tables answer to one vocabulary.
    for ty in [goofi_engine::layout::DEFAULT_PANEL_TYPE, goofi_engine::layout::EMPTY_PANEL_TYPE] {
        assert!(vocab::panel_type(ty).is_some(), "`{ty}` is not a declared panel type");
    }
    // A kind's ViewSpec has to accept everything its component draws, or a frame the viewer WOULD
    // render is filtered out of the merge and never arrives.
    for k in vocab::VIEWER_KINDS {
        if let vocab::Draws::Array { draws, accepts } = k.draws {
            assert!(draws.0 <= draws.1 && accepts.0 <= accepts.1, "`{}` has an empty range", k.id);
            assert!(accepts.0 <= draws.0 && accepts.1 >= draws.1,
                    "`{}` draws {draws:?} but its ViewSpec accepts only {accepts:?}", k.id);
        }
    }
}

// ---------------------------------------------------------------------------
// The palette — what a user reads BEFORE adding a node, and what a client
// builds every node from. It is projected from the manifest alone.
// ---------------------------------------------------------------------------

static OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
fn never() -> Box<dyn goofi_node::Node> {
    unreachable!("the catalog never instantiates")
}
const fn manifest(type_name: &'static str, inputs: &'static [SlotDecl],
                  params: &'static [ParamDecl], producer: bool) -> NodeManifest {
    NodeManifest { type_name, category: "test", doc: "a catalog fixture", inputs, outputs: OUT,
                   params, isolation: Isolation::InProcess, producer, factory: never }
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
    g.call("list_nodes", j!({}))["types"].as_array().expect("a palette").iter()
        .find(|v| v["type"] == type_name)
        .unwrap_or_else(|| panic!("{type_name} is in the palette")).clone()
}

#[test]
fn a_palette_row_carries_everything_a_client_renders_a_node_from() {
    // Registering a type is boot-time configuration — what the CLI's node scan does — so these
    // reach the graph directly: there is no op for "a type that failed to load", and inventing one
    // to test it would be the tail wagging the dog.
    let g = Goofi::new();
    g.register_dyn(&SOURCE, Box::new(|_| never()));
    g.register_dyn(&TRANSFORM, Box::new(|_| never()));
    g.register_dyn(&DOCUMENTED, Box::new(|_| never()));
    g.register_dyn(&OVERRIDES_COMMON, Box::new(|_| never()));

    // The two fixtures differ only in the `producer` flag, and it decides who paces the node.
    assert_eq!(row(&g, "MyPyThing")["params"]["common"]["autotrigger"]["value"], true,
               "a source paces itself");
    assert_eq!(row(&g, "MultiThing")["params"]["common"]["autotrigger"]["value"], false,
               "a transform is driven by its input");
    // The palette shows the same universal `common` group every instantiated node carries, so
    // type-level and instance-level params agree.
    let common = row(&g, "MyPyThing")["params"]["common"].clone();
    assert_eq!((&common["max_frequency"]["type"], &common["autotrigger"]["type"],
                &common["frequency_mode"]["type"]), (&j!("float"), &j!("bool"), &j!("string")));
    // Multi slots are the static shape the frontend reads to render them tall, and the pillar tag
    // is how a client routes a node to its editor panel.
    assert_eq!(row(&g, "MultiThing")["input_multi"], j!(["many"]));
    assert_eq!(row(&g, "MyPyThing")["input_multi"], j!([]));
    assert_eq!(row(&g, "MyPyThing")["pillar"], "signal");

    // A tooltip is rendered from the CATALOG descriptor — an instance overrides only value,
    // expression and options — so this is the projection that matters. A node that redeclares a
    // `common.*` param owns its help text too, or a source's "on by default" would be replaced by
    // the generic explanation of the flag.
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
        graph.register_dyn_type(&SOURCE, Box::new(|_| never()));
        // Provenance is "this node came with the patch you opened" — the ONLY thing the scan knows
        // that the catalog cannot re-derive, and it has to reach the greyed row too: a patch whose
        // node needs a missing dependency is exactly where "where did this come from" is asked.
        graph.set_patch_types(["MyPyThing".to_string(), "PsdScipy".to_string()].into());
    }
    let ty = row(&g, "PsdScipy");
    assert_eq!(ty["available"], false);
    assert_eq!(ty["missing_deps"], j!(["scipy"]), "the machine-readable reason");
    // `doc` is what the tooltip SHOWS, and `reason` is a bare module name only for a
    // ModuleNotFoundError — so the sentence is phrased here, where both cases are known, rather
    // than by every client that renders it.
    assert_eq!(ty["doc"], "This node could not be loaded: scipy");
    assert_eq!(ty["input_slots"], j!({}), "the probe never got far enough to report them");

    assert_eq!(ty["source"], "patch", "a greyed row is provenanced too");
    assert_eq!(row(&g, "MyPyThing")["source"], "patch");
    assert_eq!(row(&g, "Oscillator")["source"], "builtin", "a compiled-in node ships with goofi");
}

#[tokio::test]
async fn the_palette_rides_the_snapshot_and_the_graph_never_does() {
    // Structure lives in the doc ONLY. The snapshot carries just what the doc cannot: the
    // event-sourced per-node runtime, whose live stream emits TRANSITIONS and so has nothing to
    // give a client joining a graph that is already running.
    let g = Goofi::new();
    let a = g.add("Oscillator");
    let b = g.add("Buffer");
    g.link(a, "out", b, "data");
    g.call("set_expression", j!({ "node": hex(a), "group": "common", "name": "max_frequency",
                                 "expression": "@@@ not an expression @@@", "enabled": true }));
    g.ready(b);

    let (_c, hello) = Client::connect(&g.serve().await).await;
    assert_eq!(hello["node_types"], g.call("list_nodes", j!({}))["types"],
               "hello embeds the same palette `list_nodes` answers");
    for dead in ["nodes", "links", "instances"] {
        assert!(hello.get(dead).is_none(), "`{dead}` is the doc's job, not the snapshot's");
    }
    let rt = &hello["runtime"];
    assert_eq!(rt.as_object().map(|m| m.len()), Some(2), "one entry per node");
    assert_eq!((&rt[hex(b)]["stage"], &rt[hex(b)]["error"]), (&j!("ready"), &Value::Null));
    assert!(rt[hex(a)]["error"].as_str().is_some_and(|e| !e.is_empty()),
            "a reconnecting client learns the node is errored: {rt}");

    // A load brings the patch's own node types and drops the last patch's, so the catalog changes —
    // but it rides its OWN event, not the replacement snapshot, which would otherwise re-ship the
    // whole palette on every load.
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    let mut ev = g.events();
    g.call("load_text", j!({ "content": yaml }));
    assert!(ev.next("graph_replaced").get("node_types").is_none(), "the echo omits the catalog");
    assert!(ev.next("node_types")["types"].as_array().is_some_and(|a| !a.is_empty()),
            "…and a separate event carries it");
}

// ---------------------------------------------------------------------------
// The GOOF frame — mirrored in `frontend/src/lib/codec/`
// ---------------------------------------------------------------------------

#[test]
fn a_frame_survives_the_round_trip_the_browser_makes_it_do() {
    // Array data is ALWAYS f32 on the wire, and `Meta` is a plain map with a few derived keys the
    // header already carries — so a meta entry shadowing one of them must not be written twice.
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
    // The decoder reads lengths out of the frame itself, so every one of them is an attacker's (or
    // a bug's) number. It must refuse rather than index — and it must refuse rather than PANIC:
    // this same decoder runs in the browser, where a panic is a dead tab.
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
fn every_declared_expression_reads_only_a_global_a_fresh_patch_has() {
    // Seeding runs on a fresh ADD, where the only globals in the store are the system ones — so a
    // typo'd `globals.defualt_ufreq` compiles, binds, and then errors at evaluation on every
    // instance of that type: the param silently falls back to its literal and the node wears an
    // error badge. Cheap, evaluator-free, and over the whole linked catalog PLUS the universal
    // `common` group, read AS EACH TYPE SEES IT — a declaration may condition its source on the
    // manifest, so a producer form and a consumer form can differ.
    let decls = goofi_node::catalog().flat_map(|m| {
        m.params.iter().copied().chain(goofi_node::common_decls(m)).map(move |d| (m.type_name, d))
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
    // The registration is what makes the shared `_Test*` library reachable from a suite in another
    // crate, and the `_` prefix is what keeps it out of the user's palette. Both or neither: one
    // that registers without the prefix ships as a product node, and one with the prefix that
    // fails to register is invisible to the tests it exists for.
    let names: Vec<&str> = goofi_node::catalog().map(|m| m.type_name).collect();
    for want in ["_TestEcho", "_TestSink", "_TestFail", "_TestPanic", "_TestSetupFail", "_TestSlow",
                 "_TestCounter", "_TestRequired", "_TestPicker", "_TestMute", "_TestConst"] {
        assert!(names.contains(&want), "{want} is not in the catalog: {names:?}");
        assert!(want.starts_with('_'), "{want} would show in the palette");
    }
    let palette = goofi_bridge::AppState::new().call("list_nodes", j!({}), "t").unwrap();
    let listed: Vec<&str> = palette["types"].as_array().unwrap().iter()
        .map(|t| t["type"].as_str().unwrap()).collect();
    assert!(!listed.iter().any(|t| t.starts_with('_')), "a test node reached the palette: {listed:?}");
    assert!(listed.contains(&"Oscillator") && listed.contains(&"Buffer"), "{listed:?}");
}

#[test]
fn the_control_plane_document_carries_no_null_leaf() {
    // What a delta on the wire is allowed to mean. A delta is an RFC 7386 merge patch, which spends
    // `null` on "delete this key" — so a document that could hold a null leaf would be ambiguous:
    // a replica could not tell "this value is null" from "drop this key". The projection has no
    // null today, and this is what says so. If it ever needs one, this fails and NAMES the path,
    // which is the moment to give the delta an explicit tombstone instead.
    //
    // Driven over a graph that reaches every root and both optional leaves — an expression binding
    // and a wired boundary — because a null in a shape nothing built would go unseen.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    g.call("set_expression", j!({ "node": hex(osc), "group": "oscillator", "name": "frequency",
                                 "source": "globals.default_ufreq", "enabled": true }));
    g.call("add_global", j!({ "name": "subject", "value": "P07", "type": "string" }));
    let inst = g.call("group_nodes", j!({ "members": [hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    // Grouping a node whose input comes from outside mints a WIRED stub for it, which is the
    // optional `inner_node`/`inner_slot` pair — the leaves the projection omits when unwired.
    let stubs = g.doc()["instances"][&inst]["stubs"].clone();
    assert!(stubs.as_object().is_some_and(|m| m.values().any(|s| s.get("inner_node").is_some())),
            "the group left no wired stub, so this test would not reach the optional leaves: {stubs}");
    g.call("page_split_panel", j!({ "page": "Layout", "panel": panel_id(&g), "direction": "row",
                                    "ratio": 0.5 }));

    let doc = g.doc();
    for root in ["nodes", "links", "instances", "globals", "arrangement"] {
        assert!(doc.get(root).is_some(), "the document is missing its `{root}` root: {doc}");
    }
    let mut nulls = Vec::new();
    find_nulls(&doc, &mut Vec::new(), &mut nulls);
    assert!(nulls.is_empty(), "these leaves are null, so a merge patch cannot express them: {nulls:?}");
}

fn panel_id(g: &Goofi) -> String {
    g.doc()["arrangement"].as_object().unwrap().iter()
        .find(|(_, e)| e["kind"] == "panel").map(|(id, _)| id.clone()).expect("the default panel")
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
