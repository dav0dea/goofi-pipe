//! What an agent READS: the patch as a diagram, a node as a page of text, the globals an
//! expression can name, and a node type's source.
//!
//! These are goldens on purpose. The text IS the interface — a model reads it and acts on it — so
//! a drift in the wording is a change to the product, not to a formatting detail.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use goofi_core::Param;
use goofi_node::{BindingId, Compiled, EvalCtx, ExprError, ExprEvaluator};
use goofi_tests::{hex, j, Goofi};
use serde_json::Value;

/// Ages and the workspace path are wall-clock and per-run, so the golden pins everything BUT those
/// — while still requiring that each is there at all: a line that loses its age keeps its text and
/// fails the compare.
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
    let bnd = g.call("add_boundary", j!({ "inst_id": scope, "dir": "in", "dtype": "ARRAY",
                                         "pos": [0.0, 0.0] }))["bnd_id"].as_str().unwrap().to_string();
    g.call("wire_boundary", j!({ "inst_id": scope, "bnd_id": bnd,
                                "inner_node": hex(buf), "inner_slot": "data" }));
    g.call("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                           "node_in": scope, "slot_in": bnd }));
    // The fault is a REPORT — the graph does not hold it until the node has run and said so.
    g.until("the failing node's first fault", |g| g.error(boom));
    (g, scope)
}

#[test]
fn inspect_patch_draws_the_root_scope_and_the_whole_patchs_errors() {
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
  n000000000004[[\"subpatch4<br/>000000000004\"]]
  n000000000001 -- out→data --> n000000000004
```

uids: a node's uid is its mermaid id without the leading `n`; a boundary port's id is its mermaid id verbatim.

errors (whole patch):
  ⚠ _testfail0 (000000000002): the sensor is unplugged — for <age>
"
    );
}

#[test]
fn inspect_patch_draws_a_sub_patchs_boundary_ports_by_their_own_ids() {
    let (g, scope) = fixture();
    assert_eq!(
        text(&g, "inspect_patch", j!({ "scope": scope })),
        "\
patch: (never saved)
workspace: <workspace>
unsaved changes: yes
scope: subpatch4 (000000000004)

```mermaid
flowchart LR
  in0([\"in0 · in ARRAY\"])
  n000000000003[\"buffer0: Buffer<br/>000000000003\"]
  in0 -- data --> n000000000003
```

uids: a node's uid is its mermaid id without the leading `n`; a boundary port's id is its mermaid id verbatim.

errors (whole patch):
  ⚠ _testfail0 (000000000002): the sensor is unplugged — for <age>
"
    );
}

#[test]
fn a_wire_inside_a_collapsed_sub_patch_is_not_drawn_as_a_self_loop_on_its_facade() {
    // Both ends of an internal wire fold onto the same facade. Drawing it would put a loop on the
    // sub-patch that states nothing about the scope being read — the wire is a fact one level down.
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
    // The fold above is why `a == b` is skipped — but a node wired to its OWN input folds onto
    // itself for the honest reason. The engine tolerates the cycle, so the diagram must show it:
    // an agent debugging a feedback loop would otherwise read that the wire is not there.
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
    assert!(out.contains("errors (whole patch):\n  none"), "{out}");

    // A scope uid that names a LEAF is refused rather than drawn as empty.
    let n = g.add("Oscillator");
    g.refuse("inspect_patch", j!({ "scope": hex(n) }));
}

const BLEW_UP: &str = "the expression blew up";

/// An evaluator that compiles anything and hands the target value straight back — or, while
/// `broken` is set, refuses. The harness injects none, so an error the NODE reports about a
/// binding, as against one the graph found before it would ship it, has no other producer here.
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
    g.call("set_expression", j!({ "node": hex(osc), "group": "oscillator", "name": "amplitude",
                                 "expression": "globals.default_ufreq / 30", "enabled": true }));
    // A rate is MEASURED, so it needs two emits and a report to have crossed the status service —
    // which is the very thing the emitting line reads.
    g.until("the oscillator's measured rate", |g| {
        g.state.graph.lock().unwrap().node_ufreq(osc)
    });

    let out = text(&g, "inspect_node", j!({ "node": hex(osc) }));
    assert!(out.starts_with(&format!("oscillator0: Oscillator (uid {}, in-process, stage ready)", hex(osc))),
            "{out}");
    // The goldened inline param format, round-trippable into update_param…
    assert!(out.contains("  oscillator.frequency = 1 (float 0..100)"), "{out}");
    assert!(out.contains("  common.frequency_mode = \"updates-per-second\" (string one of [updates-per-second, "),
            "{out}");
    // …and into set_expression, for a param that is bound instead of literal. This binding cannot
    // compile (no evaluator in this build), which is itself worth showing inline.
    assert!(out.contains("  oscillator.amplitude = expr: globals.default_ufreq / 30 → 1 (on) [error: "),
            "{out}");
    // The slot line: name, kind, and whether the node is emitting — never the frame. There is one
    // door onto a node's data and it is `/data`.
    assert!(out.contains("  out: ARRAY — emitting at "), "the emitting line: {out}");
    assert!(!out.contains("f32["), "no frame contents leak into an inspection: {out}");
    // The wording is the GRAPH's: it could not bind this source at all, so the node was never
    // handed a binding to have a second opinion about — one error, from the end that found it.
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

    // With an evaluator the error line changes hands: the graph can bind the source, so what the
    // page carries is what the NODE found EVALUATING it. A node is handed the evaluator at BIRTH,
    // so this one is born after the injection.
    let broken = Arc::new(AtomicBool::new(true));
    g.state.graph.lock().unwrap().set_evaluator(Arc::new(Flaky { broken: broken.clone() }));
    let bound = g.add("Oscillator");
    g.call("set_expression", j!({ "node": hex(bound), "group": "oscillator", "name": "amplitude",
                                 "expression": "globals.default_ufreq / 30", "enabled": true }));
    let live = g.until("the node's own evaluation error", |g| {
        Some(text(g, "inspect_node", j!({ "node": hex(bound) }))).filter(|t| t.contains(BLEW_UP))
    });
    assert!(live.contains(&format!("(on) [error: {BLEW_UP}]")),
            "the bound param's own field carries it too: {live}");

    // That finding belongs to the INSTANCE, and a restart is a new one: it evaluates cleanly and
    // has nothing to report, so nothing is what the page must say.
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
fn read_node_source_says_a_native_type_has_no_file_to_edit() {
    let g = Goofi::new();
    let v = g.call("read_node_source", j!({ "type": "Oscillator" }));
    assert_eq!(v["language"], "rust");
    assert_eq!(v["tier"], "native");
    assert_eq!(v["source"], Value::Null);
    assert!(v["provenance"].as_str().unwrap().contains("copy a python node"), "{v}");
    // The manifest a caller needs instead comes along.
    assert_eq!(v["output_slots"]["out"], "ARRAY");
    assert!(g.refuse("read_node_source", j!({ "type": "Nope" })).contains("no node type `Nope`"));
}

#[test]
fn read_node_source_finds_a_discovered_types_file_by_re_deriving_its_name() {
    // The half a caller actually wants: the text to edit, the path to write it back to, and which
    // tree it came from. The path is RE-DERIVED from the type name rather than recorded, so this
    // pins the derivation — a scan stores no path for it to disagree with.
    static OUT: &[goofi_node::OutputDecl] =
        &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }];
    static BOOM: goofi_node::NodeManifest = goofi_node::NodeManifest {
        type_name: "Boom",
        category: "python",
        doc: "a discovered type",
        inputs: &[],
        outputs: OUT,
        params: &[],
        isolation: goofi_node::Isolation::InProcess,
        producer: true,
        factory: || unreachable!("read_node_source never instantiates"),
    };

    let g = Goofi::new();
    g.register_dyn(&BOOM, Box::new(|_| unreachable!()));
    // The patch's OWN node directory, which is where the arm looks first.
    let nodes = g.state.mount().join("nodes");
    std::fs::create_dir_all(&nodes).unwrap();
    std::fs::write(nodes.join("boom.py"), "class Boom:\n    pass\n").unwrap();

    let v = g.call("read_node_source", j!({ "type": "Boom" }));
    assert_eq!(v["provenance"], "patch", "{v}");
    assert_eq!(v["path"], goofi_core::path::to_slash(&nodes.join("boom.py")), "{v}");
    assert_eq!(v["source"], "class Boom:\n    pass\n", "{v}");
    assert_eq!(v["language"], "python", "{v}");

    // A tree that holds no file of that name leaves the discovery half empty and SAYS so, rather
    // than handing back a bare null the caller has to interpret.
    std::fs::remove_file(nodes.join("boom.py")).unwrap();
    let v = g.call("read_node_source", j!({ "type": "Boom" }));
    assert_eq!(v["source"], Value::Null);
    assert!(v["provenance"].as_str().unwrap().contains("compiled in"), "{v}");
}
