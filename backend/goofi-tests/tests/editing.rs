//! Editing a patch, with two people in it: what a write answers, what a refusal teaches, and how
//! every change comes back.

use serde_json::{Map, Value};

use goofi_tests::{hex, j, Goofi};

/// The arrangement flattened to an id-keyed map with a `parent` on each node.
fn entries(g: &Goofi) -> Map<String, Value> {
    fn down(n: &Value, parent: &str, out: &mut Map<String, Value>) {
        let mut e = n.as_object().cloned().unwrap_or_default();
        let id = e["id"].as_str().unwrap().to_string();
        e.insert("parent".into(), Value::from(parent));
        if let Some(kids) = n["children"].as_array() {
            for k in kids {
                down(k, &id, out);
            }
        }
        out.insert(id, Value::Object(e));
    }
    let mut out = Map::new();
    let arrangement = g.doc()["arrangement"].clone();
    for (i, t) in arrangement["tabs"].as_array().cloned().unwrap_or_default().iter().enumerate() {
        let id = t["id"].as_str().unwrap().to_string();
        out.insert(
            id.clone(),
            j!({ "kind": "tab", "name": t["name"].clone(), "order": i }),
        );
        down(&t["root"], &id, &mut out);
    }
    out
}

fn panels(g: &Goofi) -> Vec<String> {
    let mut v: Vec<String> =
        entries(g).iter().filter(|(_, e)| e["kind"] == "panel").map(|(id, _)| id.clone()).collect();
    v.sort();
    v
}

/// The id of the tab LABELLED `name`, resolved as the UI does.
fn tab_id(g: &Goofi, name: &str) -> String {
    entries(g).iter().find(|(_, e)| e["name"] == name).map(|(id, _)| id.clone())
        .unwrap_or_else(|| panic!("no tab labelled `{name}`"))
}

/// The tab strip, in the order it draws.
fn strip(g: &Goofi) -> Vec<String> {
    g.doc()["arrangement"]["tabs"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect()
}

/// The manager's own loader, asked to open what the manager just saved.
fn reload_warning(g: &Goofi) -> Value {
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("load_text", j!({ "content": yaml }))["layout_warning"].clone()
}

fn split(g: &Goofi, panel: &str) -> String {
    g.call("split_panel", j!({ "panel": panel, "direction": "row" }))
        .as_str().expect("a split answers the new panel's id").to_string()
}

fn first_panel(g: &Goofi) -> String {
    panels(g).first().cloned().expect("the default tab's one panel")
}

#[test]
fn a_session_of_edits_walks_all_the_way_back_and_forward_again() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 512 }));
    g.call("rename_node", j!({ "node": hex(osc), "name": "carrier" }));
    g.call("add_global", j!({ "name": "subj", "value": "P01", "type": "string" }));
    g.call("rename_global", j!({ "old": "subj", "new": "participant" }));
    g.call("set_node_pos", j!({ "node": hex(osc), "pos": [40.0, 60.0] }));
    let scope = g.call("group_nodes", j!({ "members": [hex(osc), hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let built = g.doc();

    // A rename_global is ONE step though it is an add plus a remove composed.
    let mut steps = 0;
    while g.call("undo", j!({}))["changed"] == true {
        steps += 1;
        assert!(steps < 50, "the stack never emptied");
    }
    assert!(g.nodes().is_empty() && g.instances().is_empty(), "back to an empty patch");
    assert!(g.doc()["globals"]["participant"].is_null() && g.doc()["globals"]["subj"].is_null());
    assert_eq!(steps, 9, "one step per command — a rename_global is an add plus a remove, but ONE step");

    while g.call("redo", j!({}))["changed"] == true {}
    assert_eq!(g.doc(), built, "redo rebuilt the patch it undid, uid for uid");
    let _ = scope;

    // A REORDER can silently invert to a no-op: its content IS a position.
    let g = Goofi::new();
    g.call("add_tab", j!({ "name": "Two" }));
    g.call("add_tab", j!({ "name": "Three" }));
    let settled = strip(&g);
    assert_eq!(settled, ["Tab 1", "Two", "Three"]);

    let three = tab_id(&g, "Three");
    g.call("reorder_tab", j!({ "tab": three, "to_index": 0 }));
    assert_eq!(strip(&g), ["Three", "Tab 1", "Two"], "the tab moved to the head of the strip");
    assert_eq!(g.call("undo", j!({}))["changed"], true);
    assert_eq!(strip(&g), settled, "a reorder's undo puts the tab back where it came from");
    assert_eq!(g.call("redo", j!({}))["changed"], true);
    assert_eq!(strip(&g), ["Three", "Tab 1", "Two"], "and the redo moves it again");

    while g.call("undo", j!({}))["changed"] == true {}
    assert_eq!(strip(&g), ["Tab 1"], "back to the arrangement a fresh patch opens with");
}

#[test]
fn a_fresh_command_clears_the_redo_run_and_a_session_undoes_only_its_own_work() {
    let one = Goofi::new();
    let two = one.client("s2");
    let a = one.add("Oscillator");
    let b = two.add("Buffer");

    one.call("undo", j!({}));
    assert_eq!(one.nodes(), vec![hex(b)], "s1's undo left s2's node standing");
    let r = one.call("redo", j!({}));
    assert_eq!(r["changed"], true);
    assert_eq!(one.nodes().len(), 2);

    one.call("undo", j!({}));
    one.add("Buffer"); // a fresh command discards the redo future
    let r = one.call("redo", j!({}));
    assert_eq!(r["changed"], false, "the redo run went with the new command");
    assert_eq!(r["can_redo"], false);
    let _ = a;
}

#[test]
fn a_stale_toggle_converges_instead_of_wedging_the_stack() {
    let one = Goofi::new();
    let two = one.client("s2");
    let osc = one.add("Oscillator");
    let buf = one.add("Buffer");
    let link = j!({ "node_out": hex(osc), "slot_out": "out", "node_in": hex(buf), "slot_in": "data" });
    one.call("add_link", link.clone());
    one.call("remove_link", link);
    two.call("remove_node", j!({ "node": hex(buf) })); // s1's newest toggle now names a dead uid

    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert_eq!(one.call("redo", j!({}))["changed"], true);
    for _ in 0..4 {
        assert_eq!(one.call("undo", j!({}))["changed"], true, "the stack stays walkable to empty");
    }
}

#[test]
fn a_deleted_sub_patch_comes_back_whole_with_the_panels_that_named_it() {
    let g = Goofi::new();
    let a = g.add("Oscillator");
    let b = g.add("Buffer");
    g.link(a, "out", b, "data");
    let inst = g.call("group_nodes", j!({ "members": [hex(a), hex(b)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let panel = first_panel(&g);
    g.call("set_panel", j!({ "panel": panel, "type": "viewer",
                                 "state": { "node": hex(a) } }));

    g.call("remove_node", j!({ "node": inst }));
    assert!(g.nodes().is_empty() && g.instances().is_empty(), "the subtree went with the scope");
    assert_eq!(entries(&g)[&panel]["state"], "{\"node\":null}", "and the binding with it");

    g.call("undo", j!({}));
    assert_eq!(g.instances(), vec![inst], "the scope is back at the same uid");
    assert_eq!(g.nodes().len(), 2, "with both members");
    assert_eq!(entries(&g)[&panel]["state"], format!("{{\"node\":\"{}\"}}", hex(a)),
               "and the panel names its node again");
}

/// Every layout write op driven through the one interleaving that shows a raw-state restore: a
/// peer edits between the op and its undo. The op list comes from the REGISTRY, with no catch-all.
#[test]
fn no_layout_undo_puts_back_a_slot_a_peer_has_since_built_over() {
    let ops: Vec<&str> = goofi_bridge::ops::REGISTRY.iter()
        .filter(|o| {
            o.writes
                && (o.name.ends_with("_tab") || o.name.ends_with("_panel") || o.name.ends_with("_split"))
        })
        .map(|o| o.name)
        .collect();
    assert!(ops.contains(&"remove_panel") && ops.contains(&"remove_tab"),
            "the registry filter still finds the layout write ops: {ops:?}");

    let mut stranded = Vec::new();
    for op in &ops {
        let one = Goofi::new();
        let two = one.client("s2");
        let a = first_panel(&one);
        let b = split(&one, &a);
        one.call("add_tab", j!({ "name": "Two" }));
        let c = panels(&one).into_iter().find(|p| *p != a && *p != b).expect("the tab's panel");
        let e = split(&one, &c);
        let far = entries(&one)[&e]["parent"].as_str().unwrap().to_string();
        let near = entries(&one)[&b]["parent"].as_str().unwrap().to_string();

        let two_id = tab_id(&one, "Two");
        one.call(op, match *op {
            "add_tab" => j!({ "name": "Fresh" }),
            "remove_tab" => j!({ "tab": two_id }),
            "rename_tab" => j!({ "tab": two_id, "name": "Deux" }),
            "reorder_tab" => j!({ "tab": two_id, "to_index": 0 }),
            "split_panel" => j!({ "panel": a }),
            "set_panel" => j!({ "panel": b, "type": "console" }),
            "move_panel" => j!({ "panel": b, "new_parent": far, "order_index": 0 }),
            "insert_at_panel" => j!({ "subtree": b, "target": c }),
            "resize_split" => j!({ "split": near, "fractions": [0.3, 0.7] }),
            "remove_panel" => j!({ "panel": b }),
            new => panic!("`{new}` is a layout write op with no case here — drive it through this \
                           guard, and say why if its inverse may restore a slot"),
        });
        // The peer builds exactly where a slot-restore inverse would want to write.
        if op.ends_with("_tab") {
            two.call("add_tab", j!({ "name": "Peer" }));
        } else {
            two.call("split_panel", j!({ "panel": a }));
        }
        assert_eq!(one.call("undo", j!({}))["changed"], true, "{op}: the undo flipped nothing");

        if reload_warning(&one) != Value::Null {
            stranded.push(*op);
        }
    }
    let empty: [&str; 0] = [];
    assert_eq!(stranded, empty, "an undo left an arrangement the manager cannot itself open");
}

#[test]
fn a_peers_panel_survives_every_shape_of_foreign_undo() {
    let one = Goofi::new();
    let two = one.client("s2");
    let a = first_panel(&one);

    let mine = split(&one, &a);
    let theirs = split(&two, &mine);
    assert_eq!(one.call("undo", j!({}))["changed"], true);
    let up = entries(&one)[&theirs]["parent"].as_str().unwrap().to_string();
    assert!(entries(&one).contains_key(&up), "the peer's panel still hangs off something");

    let peer2 = split(&two, &a);
    assert_eq!(one.call("redo", j!({}))["changed"], true);
    assert!(panels(&one).contains(&peer2), "the peer's panel survived a foreign redo");

    one.call("add_tab", j!({ "name": "Signals" }));
    let over = panels(&one).into_iter()
        .find(|p| ![&a, &mine, &theirs, &peer2].contains(&p)).expect("the new tab's panel");
    let far = split(&one, &over);
    let dest = entries(&one)[&far]["parent"].as_str().unwrap().to_string();
    one.call("move_panel", j!({ "panel": mine,
                                    "new_parent": dest, "order_index": 0 }));
    let peer3 = split(&two, &a);
    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert!(panels(&one).contains(&peer3), "the peer's panel survived a foreign undo");
    assert_eq!(reload_warning(&one), Value::Null);
}

#[test]
fn each_frozen_drag_gesture_is_one_op_and_therefore_one_undo() {
    // The drag feel is FROZEN UX; as primitive ops one drop would cost three to five commands.
    let g = Goofi::new();
    let first = first_panel(&g);
    let mine = split(&g, &first);
    g.call("add_tab", j!({ "name": "Signals", "index": 0 }));
    let target = panels(&g).into_iter().find(|p| *p != first && *p != mine).expect("its panel");
    let before = entries(&g);

    g.call("insert_at_panel", j!({ "subtree": mine, "target": target,
                                       "direction": "column", "place_before": true, "ratio": 0.3 }));
    assert_ne!(entries(&g), before, "the drop moved something");
    assert_eq!(g.call("undo", j!({}))["changed"], true);
    assert_eq!(entries(&g), before, "ONE ctrl-Z put the whole drag back");

    g.call("add_tab", j!({ "name": "Torn off", "index": 0, "subtree": mine }));
    assert_eq!(g.doc()["arrangement"]["tabs"][0]["root"]["id"], mine.as_str(),
               "the dragged panel is the new tab's whole root");
    g.call("undo", j!({}));
    assert_eq!(entries(&g), before, "and one ctrl-Z put that back too");
}

#[test]
fn a_restart_is_recovery_and_touches_neither_the_stack_nor_the_file() {
    // `restart_node` is the one op where "could have mutated the graph" does not imply dirty.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("load_text", j!({ "content": yaml })); // the patch now matches "disk"
    assert_eq!(g.call("get_patch", j!({}))["dirty"], false);

    let uid = g.nodes()[0].clone();
    let before = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("restart_node", j!({ "node": uid }));

    assert_eq!(g.call("serialize", j!({}))["yaml"].as_str().unwrap(), before,
               "a restart changes nothing that reaches the .gfi");
    assert_eq!(g.call("get_patch", j!({}))["dirty"], false, "so it must not dirty the patch");
    assert_eq!(g.call("undo", j!({}))["changed"], false, "and records no history entry");
}

/// A canonical 12-hex uid that names nothing.
const GHOST: &str = "ffffffffffff";

#[test]
fn a_reply_says_what_the_write_actually_did() {
    let g = Goofi::new();

    let born = g.call("add_node", j!({ "type": "Oscillator" }));
    let osc = born["uid"].as_str().unwrap().to_string();
    assert!(born["name"].as_str().is_some_and(|n| !n.is_empty()), "{born}");
    assert_eq!(born["output_slots"]["out"], "ARRAY", "{born}");
    assert_eq!(born["params"]["oscillator"]["frequency"], 1.0, "{born}");

    // A literal is COERCED to the param's declared type, so the value stored may differ.
    let buf = g.add("Buffer");
    let coerced = g.call("update_param", j!({ "node": hex(buf), "group": "buffer",
                                             "name": "size", "value": 512.6 }));
    assert_eq!(coerced["value"], 513, "an int param rounds: {coerced}");

    let wired = g.call("add_link", j!({ "node_out": osc, "slot_out": "out",
                                        "node_in": hex(buf), "slot_in": "data" }));
    assert_eq!((&wired["node_out"], &wired["dtype"]), (&j!(osc), &j!("ARRAY")), "{wired}");

    assert_eq!(g.call("remove_node", j!({ "node": GHOST }))["removed"], false);
    assert_eq!(g.call("remove_node", j!({ "node": osc }))["removed"], true);
    assert_eq!(g.call("remove_link", j!({ "node_out": osc, "slot_out": "out",
                                         "node_in": hex(buf), "slot_in": "data" }))["removed"],
               false);
}

#[test]
fn a_refusal_names_what_the_caller_could_try_instead() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");

    // A global's TYPE is what every expression reading it depends on, so `set_global` keeps it.
    let why = g.refuse("set_global", j!({ "name": "default_ufreq", "value": "fast", "type": "string" }));
    assert!(why.contains("float") && why.contains("default_ufreq"), "{why}");
    assert_eq!(g.call("set_global", j!({ "name": "default_ufreq", "value": 12.5,
                                        "type": "float" }))["value"], 12.5);

    let why = g.refuse("spawn_harness", j!({ "harness": "claude-code" }));
    assert!(why.contains("claude") && why.contains("codex"), "{why}");

    let why = g.refuse("add_link", j!({ "node_out": hex(osc), "slot_out": "out",
                                        "node_in": GHOST, "slot_in": "data" }));
    assert!(why.contains("node_in") && why.contains(GHOST), "{why}");

    for (op, payload) in [
        ("expand_instance", j!({ "inst_id": GHOST })),
        ("set_node_pos", j!({ "node": GHOST, "pos": [1.0, 2.0] })),
        ("rename_node", j!({ "node": GHOST, "name": "renamed" })),
        ("set_expression", j!({ "node": GHOST, "group": "buffer", "name": "size",
                                "expression": "1", "enabled": true })),
    ] {
        g.refuse(op, payload);
    }

    // A rename splices into expression SOURCE, so a quote yields invalid Python the referrer carries.
    for bad in ["a'b", "a\\b", "a\"b"] {
        g.refuse("rename_node", j!({ "node": hex(osc), "name": bad }));
    }
    g.call("rename_node", j!({ "node": hex(osc), "name": "a b-2" }));
    // The command tolerates a collision so replay converges; the RPC boundary raises the user error.
    g.add("Buffer");
    g.refuse("rename_node", j!({ "node": hex(osc), "name": "buffer0" }));
}

#[test]
fn an_expression_binds_carries_its_error_and_follows_the_rename_of_what_it_names() {
    let g = Goofi::new();
    let producer = g.add("Oscillator");
    let consumer = g.add("Oscillator");
    g.call("rename_node", j!({ "node": hex(producer), "name": "src" }));

    // A binding that cannot compile is STORED, so the refusal has to travel in the reply.
    let set = |expr: &str| g.call("set_expression", j!({ "node": hex(consumer), "group": "common",
                                                        "name": "max_frequency", "expression": expr,
                                                        "enabled": true, "triggers": false }));
    assert!(set("@@ not an expression @@")["error"].as_str().is_some_and(|e| !e.is_empty()),
            "the compile error must ride the reply");
    assert!(set("")["error"].is_null(), "an empty expression clears the binding");

    let mut ev = g.events();
    set("nd('src')");
    // `expression_error` is runtime-derived and rides `state_update`, never the doc.
    let d = g.until("the descriptor echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(consumer)).then(|| p["params"]["common"]["max_frequency"].clone())
    });
    assert_eq!((&d["expression"], &d["expression_enabled"], &d["expression_triggers_process"]),
               (&j!("nd('src')"), &j!(true), &j!(false)));
    assert!(d["expression_error"].is_string(), "got {:?}", d["expression_error"]);
    assert!(d.get("expression_autoeval").is_none(), "auto-eval is always on, so it is not on the wire");

    g.call("rename_node", j!({ "node": hex(producer), "name": "signal" }));
    let expr = g.until("the referrer's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(consumer))
            .then(|| p["params"]["common"]["max_frequency"]["expression"].clone())
    });
    assert_eq!(expr, "nd('signal')", "the referrer's nd() reference followed the rename");

    // Injected: a node's own `Status::ParamValues` is this event's only producer.
    let mut ev = g.events();
    g.state.graph.lock().unwrap().apply_status(consumer, goofi_engine::runtime::Status::ParamValues {
        evaluated: vec![(goofi_node::ParamKey::new("common", "max_frequency"),
                         goofi_core::Param::float(3.0, 0.0, 100.0))],
    });
    let values = g.until("a param_values broadcast", |_| {
        let p = ev.next("param_values");
        (p["node"] == hex(consumer)).then(|| p["values"].clone())
    });
    // The EVALUATED value, not the literal — the default is 30.0, which `is_number()` cannot tell apart.
    assert_eq!(values["common"]["max_frequency"].as_f64(), Some(3.0), "got {values}");
}

#[test]
fn a_node_can_be_born_configured_at_a_chosen_uid_and_name() {
    // Params are applied under the graph lock, before `node_added`, so the node is born configured.
    let g = Goofi::new();
    let mut ev = g.events();
    let born = g.call("add_node", j!({ "type": "Oscillator",
                                       "params": { "common": { "max_frequency": 42.0 } } }));
    let uid = born["uid"].as_str().unwrap().to_string();
    assert_eq!(ev.next("node_added")["uid"], uid);
    assert_eq!(g.doc()["nodes"][&uid]["params"]["common"]["max_frequency"]["value"], 42.0);

    // Undo/redo do NOT come through here — they restore via the command history.
    g.call("remove_node", j!({ "node": uid.clone() }));
    let again = g.call("add_node", j!({ "type": "Oscillator", "member_uid": uid.clone(),
                                        "name": "restored_osc" }));
    assert_eq!((&again["uid"], &g.doc()["nodes"][&uid]["name"]), (&j!(uid), &j!("restored_osc")));
}

#[test]
fn a_viewer_bag_persists_and_refuses_a_word_outside_its_vocabulary() {
    // `viewers(uid)` answers `Some({})` for every node, so an unconditional insert would stamp them all.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    assert!(g.doc()["nodes"][hex(osc)].get("viewers").is_none(), "no viewers leaf when empty");

    let why = g.refuse("set_node_viewers", j!({ "node": hex(osc),
                                                "viewers": { "out": { "kind": "waveform" } } }));
    assert!(why.contains("waveform") && why.contains("line") && why.contains("topomap"), "{why}");
    let why = g.refuse("set_node_viewers", j!({ "node": hex(osc),
                                                "viewers": { "psd": { "kind": "line" } } }));
    assert!(why.contains("psd") && why.contains("out"), "an unknown slot names the real ones: {why}");
    let why = g.refuse("set_node_viewers", j!({ "node": GHOST,
                                                "viewers": { "out": { "kind": "line" } } }));
    assert!(why.contains("no such node"), "{why}");
    let why = g.refuse("set_node_viewers", j!({ "node": hex(osc), "viewers": 7 }));
    assert!(why.contains("map"), "a bag that is not a map says what one looks like: {why}");

    g.call("set_node_viewers", j!({ "node": hex(osc),
                                    "viewers": { "out": { "collapsed": false, "kind": "line",
                                                          "settings": { "yScale": 2 } } } }));
    assert!(!g.doc()["nodes"][hex(osc)]["viewers"].is_null(), "…and the leaf appears once set");
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    assert!(yaml.contains("yScale"), "the view state persists: {yaml}");

    g.call("add_global", j!({ "name": "subject", "value": "P01", "type": "string" }));
    assert_eq!(g.doc()["globals"]["default_ufreq"]["system"], true);
    assert_eq!(g.doc()["globals"]["subject"]["system"], false);
}

#[test]
fn eight_writers_all_land_and_none_deadlock() {
    // Both the param path and the position path, because they are separate mirror writers.
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
                    client.call("set_node_pos", j!({ "node": hex(*u), "pos": [r as f64, r as f64] }));
                }
            });
        }
    });
    let doc = g.doc();
    for u in &uids {
        let n = &doc["nodes"][hex(*u)];
        assert_eq!(n["params"]["common"]["max_frequency"]["value"].as_f64(), Some(ROUNDS as f64),
                   "a param write was lost on {u}");
        assert_eq!(n["pos"]["x"].as_f64(), Some(ROUNDS as f64), "a drag was lost on {u}");
    }
}
