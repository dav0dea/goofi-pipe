//! Nodes, params, expressions and the replicated projection they land in.
//!
//! The doc is the ONLY graph projection: an op's effect is read back through `get_state`, never
//! from an event echo. What events DO carry is the per-node runtime truth the doc never holds —
//! `state_update` for a re-enriched descriptor, `param_values` for a live evaluation.

use goofi_tests::{hex, j, Goofi};

fn bind(g: &Goofi, node: &str, group: &str, name: &str, source: &str) {
    g.call("set_expression", j!({ "node": node, "group": group, "name": name,
                                  "expression": source, "enabled": true, "triggers": false }));
}

#[test]
fn a_param_edit_and_a_drag_land_in_the_state_clients_replicate() {
    // Both used to be client doc writes. They are commands now: the manager applies them to the
    // authoritative graph and re-mirrors, so every client reads them from one place.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    g.call("update_param", j!({ "node": hex(osc), "group": "common",
                               "name": "max_frequency", "value": 12.0 }));
    g.call("set_node_pos", j!({ "node": hex(osc), "pos": [123.0, 456.0] }));

    let n = &g.doc()["nodes"][hex(osc)];
    assert_eq!(n["params"]["common"]["max_frequency"]["value"], 12.0);
    assert_eq!(n["pos"]["x"], 123.0);
    assert_eq!(n["pos"]["y"], 456.0);
}

#[test]
fn an_expression_binds_and_the_descriptor_echoes_with_its_error_field() {
    // `set_expression` routes through an `EditParam` command and echoes the runtime-enriched
    // descriptor as a `state_update` — the binding round-trips AND carries `expression_error`, the
    // per-param red indicator, which is runtime-derived and never in the doc.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let mut ev = g.events();
    bind(&g, &hex(osc), "common", "max_frequency", "1 + 2");

    let d = g.until("the descriptor echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(osc)).then(|| p["params"]["common"]["max_frequency"].clone())
    });
    assert_eq!(d["expression"], "1 + 2", "source round-trips");
    assert_eq!(d["expression_enabled"], true);
    assert_eq!(d["expression_triggers_process"], false);
    // This harness injects no evaluator, so the binding round-trips WITH an error — the point is
    // that the field exists as a string to drive the indicator.
    assert!(d["expression_error"].is_string(), "got {:?}", d["expression_error"]);
    assert!(d.get("expression_autoeval").is_none(), "auto-eval is always on, so it is not on the wire");
}

#[test]
fn a_live_evaluation_is_broadcast_so_the_inspector_preview_tracks_it() {
    // The value comes from the NODE (`Status::ParamValues`), not from a graph-side evaluation — so
    // a report is what this drives. Asserting the event never arrives would pass against a
    // broadcaster that had been deleted.
    //
    // The status is injected directly because a node's own report is the ONLY producer of this
    // event, and no shipped node evaluates a Rust-side binding.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    bind(&g, &hex(osc), "common", "max_frequency", "1 + 2");
    let mut ev = g.events();
    g.state.graph.lock().unwrap().apply_status(
        osc,
        goofi_engine::runtime::Status::ParamValues {
            evaluated: vec![(goofi_node::ParamKey::new("common", "max_frequency"),
                             goofi_core::Param::float(3.0, 0.0, 100.0))],
        },
    );

    let values = g.until("a param_values broadcast", |_| {
        let p = ev.next("param_values");
        (p["node"] == hex(osc)).then(|| p["values"].clone())
    });
    // The EVALUATED value, not the literal — Oscillator's own default is 30.0, so `is_number()`
    // could not tell the two apart.
    assert_eq!(values["common"]["max_frequency"].as_f64(), Some(3.0), "got {values}");
}

#[test]
fn renaming_a_node_rewrites_the_nd_references_that_name_it() {
    let g = Goofi::new();
    let producer = g.add("Oscillator");
    let consumer = g.add("Oscillator");
    g.call("rename_node", j!({ "node": hex(producer), "name": "src" }));
    bind(&g, &hex(consumer), "common", "max_frequency", "nd('src')");

    let mut ev = g.events();
    g.call("rename_node", j!({ "node": hex(producer), "name": "signal" }));

    let expr = g.until("the referrer's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(consumer))
            .then(|| p["params"]["common"]["max_frequency"]["expression"].clone())
    });
    assert_eq!(expr, "nd('signal')", "the referrer's nd() reference followed the rename");
}

#[test]
fn rename_node_refuses_a_name_another_node_holds() {
    // `Command::EditNode` TOLERATES a collision as a no-op, so a stale undo-replay converges instead
    // of wedging the stack. The forward user error therefore has to be raised at the RPC boundary.
    let g = Goofi::new();
    let a = g.add("Oscillator");
    g.add("Buffer"); // "buffer0"

    g.refuse("rename_node", j!({ "node": hex(a), "name": "buffer0" }));
    g.call("rename_node", j!({ "node": hex(a), "name": "myosc" }));
}

#[test]
fn a_node_can_be_born_configured_at_a_chosen_uid_and_name() {
    // Paste, duplicate and undo-of-delete replay a node's params inline, so it is born CONFIGURED —
    // params applied under the graph lock, before `node_added`. A post-add `update_param` used to
    // do it, and silently dropped the values.
    let g = Goofi::new();
    let mut ev = g.events();
    let born = g.call("add_node", j!({ "type": "Oscillator",
                                      "params": { "common": { "max_frequency": 42.0 } } }));
    let uid = born["uid"].as_str().unwrap().to_string();
    assert_eq!(ev.next("node_added")["uid"], uid);
    assert_eq!(g.doc()["nodes"][&uid]["params"]["common"]["max_frequency"]["value"], 42.0);

    // …and a caller reconstructing a known graph can ask for the uid and name its links and panels
    // already point at. Undo/redo do NOT come through here — they restore via the command history.
    g.call("remove_node", j!({ "node": uid }));
    let again = g.call("add_node", j!({ "type": "Oscillator", "member_uid": uid,
                                       "name": "restored_osc" }));
    assert_eq!(again["uid"], uid, "the requested uid");
    assert_eq!(g.doc()["nodes"][&uid]["name"], "restored_osc", "and the requested name");
}

#[test]
fn a_viewer_bag_persists_and_refuses_a_word_outside_its_vocabulary() {
    // The same bug `page_set_panel` was taught out of, one door over: this bag carries the SAME
    // viewer kinds keyed by the SAME slot names, and both were stored as free strings.
    let g = Goofi::new();
    let osc = g.add("Oscillator");

    let why = g.refuse("set_node_viewers", j!({ "node": hex(osc),
                                               "viewers": { "out": { "kind": "waveform" } } }));
    assert!(why.contains("waveform") && why.contains("line") && why.contains("topomap"), "{why}");

    let why = g.refuse("set_node_viewers", j!({ "node": hex(osc),
                                               "viewers": { "psd": { "kind": "line" } } }));
    assert!(why.contains("psd") && why.contains("out"), "an unknown slot names the real ones: {why}");

    // A uid naming no node stays the ENGINE's refusal: the slot check must not shadow it with
    // "has no output slot" on a node that is not there at all.
    let why = g.refuse("set_node_viewers", j!({ "node": "0000000000ff",
                                               "viewers": { "out": { "kind": "line" } } }));
    assert!(why.contains("no such node"), "{why}");
    let why = g.refuse("set_node_viewers", j!({ "node": hex(osc), "viewers": 7 }));
    assert!(why.contains("map"), "a bag that is not a map says what one looks like: {why}");

    // A real kind on a real slot lands, and rides the `.gfi` — so the check is a gate, not a wall.
    g.call("set_node_viewers", j!({ "node": hex(osc),
                                   "viewers": { "out": { "collapsed": false, "kind": "line",
                                                         "settings": { "yScale": 2 } } } }));
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    assert!(yaml.contains("yScale"), "the view state persists: {yaml}");
}

#[test]
fn a_patch_round_trips_through_its_own_serialization() {
    let g = Goofi::new();
    g.add("Oscillator");
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    assert!(yaml.contains("version: 7"), "gfi v7 header");
    assert!(yaml.contains("Oscillator") && yaml.contains("default_ufreq"));

    let mut ev = g.events();
    g.call("load_text", j!({ "content": yaml }));
    assert!(ev.next("graph_replaced")["runtime"].is_object(),
            "graph_replaced seeds the runtime overlay the doc never holds");

    // The restored GRAPH arrives through the doc — the snapshot carries no second projection of it.
    let uid = g.nodes().pop().expect("the loaded node");
    assert_eq!(g.doc()["nodes"][&uid]["type"], "Oscillator");
}

#[test]
fn concurrent_writers_all_land_and_none_deadlock() {
    // Every write races through the same `apply -> graph -> re-mirror` chain (graph, then history,
    // then crdt). Two properties at once: the contended chain never deadlocks, and a reader
    // converges on ALL the distinct final values, so not one command was dropped by the re-mirror.
    const N: usize = 8;
    const ROUNDS: usize = 5;

    let g = Goofi::new();
    let uids: Vec<_> = (0..N).map(|_| g.add("Oscillator")).collect();
    std::thread::scope(|s| {
        for (i, u) in uids.iter().enumerate() {
            let client = g.client(&format!("s{i}"));
            s.spawn(move || {
                for r in 1..=ROUNDS {
                    client.call("update_param", j!({ "node": hex(*u), "group": "common",
                                                    "name": "max_frequency", "value": r as f64 }));
                }
            });
        }
    });

    let doc = g.doc();
    let got: Vec<_> = uids.iter()
        .map(|u| doc["nodes"][hex(*u)]["params"]["common"]["max_frequency"]["value"].as_f64())
        .collect();
    assert!(got.iter().all(|v| *v == Some(ROUNDS as f64)), "a write was lost: {got:?}");
}
