//! What an agent READS: the patch as a diagram, a node as a tab of text, the globals an
//! expression can name, and a node type's source.
//!
//! Goldens on purpose: the text IS the interface a model reads and acts on.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use goofi_core::Param;
use goofi_node::{BindingId, Compiled, EvalCtx, ExprError, ExprEvaluator};
use goofi_tests::{hex, j, Goofi};
use serde_json::Value;

/// Ages and the workspace path are per-run, so the golden pins everything but their values.
fn stable(s: &str, mount: &str) -> String {
    s.replace(mount, "<workspace>")
        .split_inclusive('\n')
        .map(|l| match l.split_once(" — for ") {
            Some((head, tail)) => format!("{head} — for <age>{}", &tail[tail.len() - 1..]),
            None => l.to_string(),
        })
        .collect()
}

fn text(g: &Goofi, op: &str, payload: Value) -> String {
    let mount = goofi_core::path::to_slash(&g.state.mount());
    stable(g.call(op, payload)["text"].as_str().expect("a text answer"), &mount)
}

/// A patch with a sub-patch (a Buffer behind an input boundary port), a top-level source wired
/// through that port, and a node that is erroring.
fn fixture() -> (Goofi, String) {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let boom = g.add("_TestFail");
    let buf = g.add("Buffer");
    let scope = g.call("group_nodes", j!({ "members": [hex(buf)], "pos": [40.0, 10.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let bnd = g.call("add_node", j!({ "type": "InArray", "inst_id": scope,
                                     "pos": [0.0, 0.0] }))["uid"].as_str().unwrap().to_string();
    g.call("add_link", j!({ "node_out": bnd, "slot_out": "value",
                           "node_in": hex(buf), "slot_in": "data" }));
    g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                           "node_in": scope, "slot_in": bnd }));
    // The fault is a REPORT — the graph does not hold it until the node has run and said so.
    g.until("the failing node's first fault", |g| g.error(boom));
    (g, scope)
}

#[test]
fn inspect_patch_draws_the_scope_asked_for_and_get_patch_says_what_is_broken() {
    let (g, _) = fixture();
    assert_eq!(
        text(&g, "inspect_patch", j!({})),
        "\
patch: (never saved)
workspace: <workspace>
unsaved changes: yes
scope: root

```mermaid
flowchart LR
  n000000000001[\"oscillator0: Oscillator<br/>000000000001\"]
  n000000000002[\"⚠ _testfail0: _TestFail<br/>000000000002\"]
  n000000000004[[\"subpatch0<br/>000000000004\"]]
  n000000000001 -- out→value --> n000000000004
```

uids: a uid is its mermaid id without the leading `n`.
"
    );

    // What is BROKEN is the patch's business, not this scope's — one read, and the node's path
    // says where it lives, so a scope view never has to carry a neighbour's fault.
    let health = g.call("get_patch", j!({}));
    let errs = health["errors"].as_array().cloned().unwrap_or_default();
    assert_eq!(errs.len(), 1, "one standing error: {health}");
    assert_eq!(errs[0]["node"], "000000000002");
    assert_eq!(errs[0]["path"], "_testfail0");
    assert_eq!(errs[0]["error"], "the sensor is unplugged");
    assert!(errs[0]["standing"].as_f64().is_some(), "and how long it has stood: {health}");
}

#[test]
fn inspect_patch_draws_a_sub_patchs_boundary_ports_as_the_nodes_they_are() {
    let (g, scope) = fixture();
    assert_eq!(
        text(&g, "inspect_patch", j!({ "scope": scope })),
        "\
patch: (never saved)
workspace: <workspace>
unsaved changes: yes
scope: subpatch0 (000000000004)

```mermaid
flowchart LR
  n000000000003[\"buffer0: Buffer<br/>000000000003\"]
  n000000000005([\"in0: InArray<br/>000000000005\"])
  n000000000005 -- value\u{2192}data --> n000000000003
```

uids: a uid is its mermaid id without the leading `n`.
"
    );
    // The erroring node is in ROOT, and this is a sub-patch: asking about one scope used to report
    // every fault in the patch, so the same list arrived again under each scope.
    assert!(!text(&g, "inspect_patch", j!({ "scope": scope })).contains("_testfail0"));
}

#[test]
fn a_wire_inside_a_collapsed_sub_patch_is_not_drawn_as_a_self_loop_on_its_facade() {
    // Both ends of an internal wire fold onto the same facade; the wire is a fact one level down.
    let g = Goofi::new();
    let a = g.add("Oscillator");
    let b = g.add("Buffer");
    g.link(a, "out", b, "data");
    g.call("group_nodes", j!({ "members": [hex(a), hex(b)], "pos": [0.0, 0.0] }));

    let out = text(&g, "inspect_patch", j!({}));
    assert!(out.contains("[["), "the facade is drawn: {out}");
    assert!(!out.contains("-->"), "…with no edge at all at this level: {out}");
}

#[test]
fn a_node_wired_to_itself_keeps_its_edge() {
    // A node wired to its OWN input folds onto itself honestly, and the engine tolerates the cycle.
    let g = Goofi::new();
    let buf = g.add("Buffer");
    g.link(buf, "out", buf, "data");
    let out = text(&g, "inspect_patch", j!({}));
    assert!(out.contains(&format!("n{0} -- out→data --> n{0}\n", hex(buf))), "{out}");
}

#[test]
fn an_empty_scope_says_so_rather_than_drawing_an_empty_diagram() {
    let g = Goofi::new();
    let out = text(&g, "inspect_patch", j!({}));
    assert!(out.contains("(no nodes)"), "{out}");
    assert!(!out.contains("mermaid"), "no diagram for an empty scope: {out}");
    assert_eq!(g.call("get_patch", j!({}))["errors"], j!([]), "and nothing is broken");

    // A scope uid that names a LEAF is refused rather than drawn as empty.
    let n = g.add("Oscillator");
    g.refuse("inspect_patch", j!({ "scope": hex(n) }));
}

const BLEW_UP: &str = "the expression blew up";

/// An evaluator that compiles anything and hands the target value back, or refuses while `broken`.
struct Flaky {
    broken: Arc<AtomicBool>,
}

impl ExprEvaluator for Flaky {
    fn compile(&self, _source: &str) -> Result<Compiled, ExprError> {
        Ok(Compiled { id: 1 })
    }
    fn eval(&self, _id: BindingId, ctx: &EvalCtx<'_>) -> Result<Param, ExprError> {
        if self.broken.load(Ordering::Relaxed) {
            return Err(BLEW_UP.into());
        }
        Ok(ctx.target.clone())
    }
    fn release(&self, _id: BindingId) {}
}

#[test]
fn inspect_node_reports_params_whether_each_slot_is_emitting_and_the_error() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    g.call("edit_node", j!({ "node": hex(osc), "params": { "oscillator": {
                                 "amplitude": { "expression": "globals.default_ufreq / 30" } } } }));
    // A rate is MEASURED, so it needs two emits and a report across the status service.
    g.until("the oscillator's measured rate", |g| {
        g.state.graph.lock().unwrap().node_ufreq(osc)
    });

    let out = text(&g, "inspect_node", j!({ "node": hex(osc) }));
    assert!(out.starts_with(&format!("oscillator0: Oscillator (uid {}, native, stage ready)", hex(osc))),
            "{out}");
    // The goldened inline param format, round-trippable into edit_node…
    assert!(out.contains("  oscillator.frequency = 1 (float 0..100)"), "{out}");
    assert!(out.contains("  common.frequency_mode = \"updates-per-second\" (string one of [updates-per-second, "),
            "{out}");
    // …and into edit_node’s expression half. This binding cannot compile (no evaluator here), shown inline.
    assert!(out.contains("  oscillator.amplitude = expr: globals.default_ufreq / 30 → 1 (on) [error: "),
            "{out}");
    // The slot line never carries the frame: there is one door onto a node's data and it is `/data`.
    assert!(out.contains("  out: ARRAY — emitting at "), "the emitting line: {out}");
    assert!(!out.contains("f32["), "no frame contents leak into an inspection: {out}");
    // The wording is the GRAPH's: it could not bind the source, so the node never saw it.
    let err = out.lines().find(|l| l.starts_with("error: ")).unwrap_or_else(|| panic!("{out}"));
    assert!(err.contains("no expression evaluator available") && err.contains(" — for <age>"),
            "an error line carries its age, so a settling node reads differently from a broken one: {out}");

    // A node that has never emitted says so, rather than reading as healthy silence.
    let idle = g.add("Buffer");
    let idle_out = text(&g, "inspect_node", j!({ "node": hex(idle), "params": false }));
    assert!(idle_out.contains("  out: ARRAY — nothing emitted yet"), "{idle_out}");
    assert!(idle_out.ends_with("error: none\n"), "{idle_out}");

    // The flags actually gate their sections.
    let bare = text(&g, "inspect_node", j!({ "node": hex(osc), "params": false, "error": false }));
    assert!(!bare.contains("params:") && !bare.contains("error:"), "{bare}");

    // An unknown slot is refused by naming the ones that exist.
    let why = g.refuse("inspect_node", j!({ "node": hex(osc), "slot": "psd" }));
    assert!(why.contains("no output slot `psd`") && why.contains("out"), "{why}");

    // With an evaluator the error changes hands to the NODE. A node is handed one at BIRTH, so this
    // one is born after the injection.
    let broken = Arc::new(AtomicBool::new(true));
    g.state.graph.lock().unwrap().set_evaluator(Arc::new(Flaky { broken: broken.clone() }));
    let bound = g.add("Oscillator");
    g.call("edit_node", j!({ "node": hex(bound), "params": { "oscillator": {
                                 "amplitude": { "expression": "globals.default_ufreq / 30" } } } }));
    let live = g.until("the node's own evaluation error", |g| {
        Some(text(g, "inspect_node", j!({ "node": hex(bound) }))).filter(|t| t.contains(BLEW_UP))
    });
    assert!(live.contains(&format!("(on) [error: {BLEW_UP}]")),
            "the bound param's own field carries it too: {live}");

    // The finding belongs to the INSTANCE, and a restart is a new one with nothing to report.
    broken.store(false, Ordering::Relaxed);
    g.call("restart_node", j!({ "node": hex(bound) }));
    let reborn = text(&g, "inspect_node", j!({ "node": hex(bound) }));
    assert!(!reborn.contains(BLEW_UP) && reborn.ends_with("error: none\n"),
            "a reborn node draws none of the corpse's binding errors: {reborn}");
}

#[test]
fn list_globals_names_the_system_globals_an_expression_can_read() {
    let g = Goofi::new();
    let first = g.call("list_globals", j!({}))["globals"][0].clone();
    assert_eq!(first["name"], "default_ufreq");
    assert_eq!(first["type"], "float");
    assert_eq!(first["value"], 30.0);
    assert_eq!(first["system"], true);
}

#[test]
fn one_named_type_is_the_catalog_entry_plus_the_file_behind_it() {
    let g = Goofi::new();
    // The catalog and one entry of it are the same read at two widths, so the narrow one must agree
    // with the wide one rather than be assembled a second way.
    let all = g.call("list_nodes", j!({}))["types"].as_array().cloned().unwrap_or_default();
    let listed = all.iter().find(|t| t["type"] == "Oscillator").expect("Oscillator is in the palette");

    let v = g.call("list_nodes", j!({ "type": "Oscillator" }));
    assert_eq!(v["type"], listed["type"]);
    assert_eq!(v["doc"], listed["doc"], "one entry says what the catalog says");
    assert_eq!(v["params"], listed["params"]);
    assert_eq!(v["language"], "rust");
    assert_eq!(v["tier"], "native");
    assert_eq!(v["source"], Value::Null);
    assert!(v["provenance"].as_str().unwrap().contains("copy a python node"), "{v}");
    // The manifest a caller needs instead comes along.
    assert_eq!(v["output_slots"]["out"], "ARRAY");
    assert!(g.refuse("list_nodes", j!({ "type": "Nope" })).contains("no node type `Nope`"));
}

#[test]
fn a_discovered_types_file_is_found_by_re_deriving_its_name() {
    // The path is RE-DERIVED from the type name rather than recorded, so this pins the derivation.
    static OUT: &[goofi_node::OutputDecl] =
        &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }];
    static BOOM_TIER: goofi_node::IsolationCell =
        goofi_node::IsolationCell::new(goofi_node::Isolation::InProcess);
    static BOOM: goofi_node::NodeManifest = goofi_node::NodeManifest {
        type_name: "Boom",
        category: "python",
        doc: "a discovered type",
        inputs: &[],
        outputs: OUT,
        params: &[],
        isolation: &BOOM_TIER,
        producer: true,
        factory: || unreachable!("a catalog read never instantiates"),
    };

    let g = Goofi::new();
    g.register_dyn(&BOOM, Box::new(|_| unreachable!()));
    // The patch's OWN node directory, which is where the arm looks first.
    let nodes = g.state.mount().join("nodes");
    std::fs::create_dir_all(&nodes).unwrap();
    std::fs::write(nodes.join("boom.py"), "class Boom:\n    pass\n").unwrap();

    let v = g.call("list_nodes", j!({ "type": "Boom" }));
    assert_eq!(v["provenance"], "patch", "{v}");
    assert_eq!(v["path"], goofi_core::path::to_slash(&nodes.join("boom.py")), "{v}");
    assert_eq!(v["source"], "class Boom:\n    pass\n", "{v}");
    assert_eq!(v["language"], "python", "{v}");

    // A tree holding no such file leaves the discovery half empty and SAYS so, rather than null.
    std::fs::remove_file(nodes.join("boom.py")).unwrap();
    let v = g.call("list_nodes", j!({ "type": "Boom" }));
    assert_eq!(v["source"], Value::Null);
    assert!(v["provenance"].as_str().unwrap().contains("compiled in"), "{v}");
}
