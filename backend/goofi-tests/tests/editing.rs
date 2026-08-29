//! Editing a patch, with two people in it: what a write answers, what a refusal teaches, and how
//! every change comes back.

use serde_json::{Map, Value};

use goofi_tests::{f32s, Goofi, ep, hex, j};

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
    let yaml = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("session load", j!({ "content": yaml }))["layout_warning"].clone()
}

fn split(g: &Goofi, panel: &str) -> String {
    g.call("layout panel add", j!({ "beside": panel, "side": "right" }))["id"]
        .as_str().expect("a placement answers what it placed").to_string()
}

fn first_panel(g: &Goofi) -> String {
    panels(g).first().cloned().expect("the default tab's one panel")
}

/// A uid that names nothing, which every refusal path is asked about.
const GHOST: &str = "ffffffffffff";

#[test]
fn a_session_of_edits_walks_all_the_way_back_and_forward_again() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    g.set_param(buf, "buffer", "size", 512);
    // ONE step, whatever it carries: a rename, a move and a viewer in a single node edit.
    g.call("node edit", j!({ "node": hex(osc), "name": "carrier", "pos": [40.0, 60.0],
                             "viewer": [{ "slot": "out", "kind": "line" }] }));
    g.set_param(osc, "oscillator", "sfreq", 128.0);
    g.call("global add", j!({ "name": "subj", "value": "P01", "type": "string" }));
    // A rename is a compound: set the new name, delete the old, ONE undo step.
    g.call("compound", j!({ "ops": [
        { "op": "global add", "payload": { "name": "participant", "value": "P01", "type": "string" } },
        { "op": "global remove", "payload": { "name": "subj" } },
    ] }));
    // …and a compound is a UNIT: a refused step takes back the one that landed, and records nothing,
    // which is what the step count below would catch.
    let why = g.refuse("compound", j!({ "ops": [
        { "op": "global add", "payload": { "name": "tmp", "value": 1.0, "type": "float" } },
        { "op": "node edit", "payload": { "node": GHOST, "name": "renamed" } },
    ] }));
    assert!(why.contains("step 1"), "the refusal names the step that failed: {why}");
    assert!(g.doc()["globals"]["tmp"].is_null(), "the step that landed was taken back: {why}");
    // A READ rides a batch — its result in the bare list the batch answers — while an EFFECT is
    // refused: its consequences are not the history's to take back, so it runs alone.
    let ridden = g.call("compound", j!({ "ops": [
        { "op": "nodes inspect" },
        { "op": "node edit", "payload": { "node": hex(osc), "pos": [1.0, 2.0] } },
    ] }));
    assert!(ridden[0]["text"].as_str().is_some_and(|t| t.contains("carrier")),
            "the read's own reply rides the list: {ridden}");
    for bad in ["undo", "compound", "session load", "session new", "session save",
                "node restart", "library refresh"] {
        let why = g.refuse("compound", j!({ "ops": [{ "op": bad }] }));
        assert!(why.contains("not a step"), "`{bad}`: {why}");
    }
    // A batch's deliveries settle ONCE: a compound editing one param twice lands on the node as a
    // single write, carrying the final value. The meter counts `on_param_changed` calls.
    let meter = g.add("_TestParamWrites");
    let mprobe = g.probe(meter, "out");
    g.link(osc, "out", meter, "in");
    let base = g.until("the meter to settle after init", |_| {
        mprobe.latest().map(|d| f32s(&d)[0])
    });
    g.call("compound", j!({ "ops": [
        { "op": "node param edit", "payload": { "node": hex(meter), "param": "control/value", "value": 1.0 } },
        { "op": "node param edit", "payload": { "node": hex(meter), "param": "control/value", "value": 2.0 } },
    ] }));
    g.until("exactly one more delivery, carrying the final value", |g| {
        let n = f32s(&mprobe.latest()?)[0];
        (n >= base + 1.0).then(|| {
            assert_eq!(n, base + 1.0, "two edits in one batch settled as one write");
            assert_eq!(g.doc()["nodes"][hex(meter)]["params"]["control"]["value"]["value"], j!(2.0));
        })
    });
    let scope = g.call("nodes group", j!({ "nodes": [hex(osc), hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let built = g.doc();

    // A compound is ONE step though it is an add plus a remove composed.
    let mut steps = 0;
    while g.call("undo", j!({}))["changed"] == true {
        steps += 1;
        assert!(steps < 50, "the stack never emptied");
    }
    assert!(g.nodes().is_empty() && g.instances().is_empty(), "back to an empty patch");
    assert!(g.doc()["globals"]["participant"].is_null() && g.doc()["globals"]["subj"].is_null());
    assert_eq!(steps, 13, "one step per command — a compound (the rename, the two-edit batch) and a three-field node edit are each ONE");

    while g.call("redo", j!({}))["changed"] == true {}
    assert_eq!(g.doc(), built, "redo rebuilt the patch it undid, uid for uid");
    let _ = scope;

    // The NAME is the arrangement's to mint: a caller that asks for none gets the first free
    // `Tab n`, so nobody has to reserve one against a strip they cannot see settle.
    let g = Goofi::new();
    g.call("layout panel add", j!({}));
    assert_eq!(strip(&g), ["Tab 1", "Tab 2"], "minted, not asked for");
    // …and a label is NOT unique. It addresses nothing — every op names an id — and uniqueness was
    // enforceable only on the way in: a rename's inverse must not refuse, so a peer taking the
    // freed name left an arrangement the loader would not open.
    g.call("layout tab edit", j!({ "tab": tab_id(&g, "Tab 2"), "name": "Tab 1" }));
    assert_eq!(strip(&g), ["Tab 1", "Tab 1"]);
    assert_eq!(reload_warning(&g), Value::Null, "and it still opens");
    while g.call("undo", j!({}))["changed"] == true {}

    // A REORDER can silently invert to a no-op: its content IS a position.
    let g = Goofi::new();
    g.call("layout panel add", j!({ "name": "Two" }));
    g.call("layout panel add", j!({ "name": "Three" }));
    let settled = strip(&g);
    assert_eq!(settled, ["Tab 1", "Two", "Three"]);

    let three = tab_id(&g, "Three");
    g.call("layout move", j!({ "entry": three, "index": 0 }));
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
    let link = j!({ "from": ep(hex(osc), "out"), "to": ep(hex(buf), "data") });
    one.call("link add", link.clone());
    one.call("link remove", link);
    two.call("node remove", j!({ "node": hex(buf) })); // s1's newest toggle now names a dead uid

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
    let inst = g.call("nodes group", j!({ "nodes": [hex(a), hex(b)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let panel = first_panel(&g);
    g.call("layout panel edit", j!({ "panel": panel, "type": "viewer",
                                     "state": { "node": hex(a) } }));

    // A panel can name a boundary PORT — it exposes a real stream — so a removed port has to take
    // its binding with it exactly as a removed node does, or the panel renders empty for good and
    // refuses even a change of viewer kind, because the dead uid has no slots to check against.
    let port = g.call("node add", j!({ "type": "InArray", "inst_id": inst, "pos": [0.0, 0.0] }))
        ["uid"].as_str().expect("a port uid").to_string();
    let second = split(&g, &panel);
    g.call("layout panel edit", j!({ "panel": second, "type": "viewer",
                                     "state": { "node": port, "slot": "value" } }));
    g.call("node remove", j!({ "node": port }));
    let unbound = |p: &str| entries(&g)[p]["state"].as_str().unwrap_or("").contains("\"node\":null");
    assert!(unbound(&second), "the port took its panel binding: {}", entries(&g)[&second]["state"]);
    g.call("undo", j!({}));
    assert!(entries(&g)[&second]["state"].as_str().is_some_and(|s| s.contains(&port)),
            "and one undo gives the port and the binding back together");

    g.call("node remove", j!({ "node": inst }));
    assert!(g.nodes().is_empty() && g.instances().is_empty(), "the subtree went with the scope");
    assert_eq!(entries(&g)[&panel]["state"], "{\"node\":null}", "and the binding with it");
    assert!(unbound(&second),
            "…including a panel that named one of its PORTS, which the subtree sweep must reach");

    g.call("undo", j!({}));
    assert_eq!(g.instances(), vec![inst], "the scope is back at the same uid");
    assert_eq!(g.nodes().len(), 2, "with both members");
    assert_eq!(entries(&g)[&panel]["state"], format!("{{\"node\":\"{}\"}}", hex(a)),
               "and the panel names its node again");
}

/// Every SHAPE a layout write comes in, driven through the one interleaving that shows a raw-state
/// restore: a peer edits between the op and its undo. The merged ops each carry several — a tab's
/// name and a split's shares are both `edit_panel` — so the rows are shapes, and the op list from
/// the REGISTRY is what proves no op slipped past without one.
#[test]
fn no_layout_undo_puts_back_a_slot_a_peer_has_since_built_over() {
    let ops: Vec<&str> = goofi_bridge::ops::registry().iter()
        .filter(|o| o.handler.is_write() && o.name.starts_with("layout "))
        .map(|o| o.name)
        .collect();
    assert!(ops.contains(&"layout remove") && ops.contains(&"layout move"),
            "the registry filter still finds the layout write ops: {ops:?}");

    // (shape, the op it goes out as). One row per way a caller can spell a layout write — the
    // SHAPES are the truth here and the op names are not, which is how they survived the merged
    // ops splitting into one op per record kind.
    const SHAPES: &[(&str, &str)] = &[
        ("a fresh tab", "layout panel add"),
        ("a tab built around a subtree", "layout move"),
        ("a split", "layout panel add"),
        ("a tab's name", "layout tab edit"),
        ("a panel's type", "layout panel edit"),
        ("a split's shares", "layout split edit"),
        ("a move into a split", "layout move"),
        ("a move beside a panel", "layout move"),
        ("a move within the strip", "layout move"),
        ("a closed panel", "layout remove"),
        ("a closed tab", "layout remove"),
    ];
    for op in &ops {
        assert!(SHAPES.iter().any(|(_, o)| o == op),
                "`{op}` is a layout write op with no shape here — drive it through this guard, and \
                 say why if its inverse may restore a slot");
    }

    let mut stranded = Vec::new();
    for (shape, op) in SHAPES {
        let one = Goofi::new();
        let two = one.client("s2");
        let a = first_panel(&one);
        let b = split(&one, &a);
        one.call("layout panel add", j!({}));
        let c = panels(&one).into_iter().find(|p| *p != a && *p != b).expect("the tab's panel");
        let e = split(&one, &c);
        let far = entries(&one)[&e]["parent"].as_str().unwrap().to_string();
        let near = entries(&one)[&b]["parent"].as_str().unwrap().to_string();
        let two_id = tab_id(&one, "Tab 2");

        one.call(op, match *shape {
            "a fresh tab" => j!({}),
            "a tab built around a subtree" => j!({ "entry": b }),
            "a split" => j!({ "beside": a }),
            "a tab's name" => j!({ "tab": two_id, "name": "Deux" }),
            "a panel's type" => j!({ "panel": b, "type": "console" }),
            "a split's shares" => j!({ "split": near, "fraction": [0.3, 0.7] }),
            "a move into a split" => j!({ "entry": b, "in": far, "index": 0 }),
            "a move beside a panel" => j!({ "entry": b, "beside": c, "side": "bottom" }),
            "a move within the strip" => j!({ "entry": two_id, "index": 0 }),
            "a closed panel" => j!({ "entry": b }),
            "a closed tab" => j!({ "entry": two_id }),
            new => panic!("`{new}` is a shape with no payload here"),
        });
        // The peer builds exactly where a slot-restore inverse would want to write — both places,
        // because a merged op's shapes do not all reach for the same one.
        two.call("layout panel add", j!({}));
        two.call("layout panel add", j!({ "beside": a }));
        assert_eq!(one.call("undo", j!({}))["changed"], true, "{shape}: the undo flipped nothing");

        if reload_warning(&one) != Value::Null {
            stranded.push(*shape);
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

    one.call("layout panel add", j!({ "name": "Signals" }));
    let over = panels(&one).into_iter()
        .find(|p| ![&a, &mine, &theirs, &peer2].contains(&p)).expect("the new tab's panel");
    let far = split(&one, &over);
    let dest = entries(&one)[&far]["parent"].as_str().unwrap().to_string();
    one.call("layout move", j!({ "entry": mine, "in": dest, "index": 0 }));
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
    g.call("layout panel add", j!({ "name": "Signals", "index": 0 }));
    let target = panels(&g).into_iter().find(|p| *p != first && *p != mine).expect("its panel");
    let before = entries(&g);

    g.call("layout move", j!({ "entry": mine, "beside": target,
                               "side": "top", "ratio": 0.3 }));
    assert_ne!(entries(&g), before, "the drop moved something");
    // The SIDE really landed: `top` means a column split with the mover FIRST. This is the pin
    // that caught `--side` being read off a dead key and every drop silently going right.
    let parent = entries(&g)[&mine]["parent"].as_str().unwrap().to_string();
    let split = &entries(&g)[&parent];
    assert_eq!(split["axis"], "column", "a `top` drop splits vertically: {split}");
    assert_eq!(split["children"][0]["id"], j!(mine.as_str()),
               "…with the mover on the side it was dropped on: {split}");
    g.refuse("layout move", j!({ "entry": mine, "beside": target, "side": "sideways" }));
    assert_eq!(g.call("undo", j!({}))["changed"], true);
    assert_eq!(entries(&g), before, "ONE ctrl-Z put the whole drag back");

    g.call("layout move", j!({ "entry": mine, "name": "Torn off", "index": 0 }));
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
    let yaml = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("session load", j!({ "content": yaml })); // the patch now matches "disk"
    assert_eq!(g.call("session status", j!({}))["dirty"], false);

    let uid = g.nodes()[0].clone();
    let before = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("node restart", j!({ "node": uid }));

    assert_eq!(g.call("session manifest", j!({}))["yaml"].as_str().unwrap(), before,
               "a restart changes nothing that reaches the .gfi");
    assert_eq!(g.call("session status", j!({}))["dirty"], false, "so it must not dirty the patch");
    assert_eq!(g.call("undo", j!({}))["changed"], false, "and records no history entry");

    // The batch decides dirtiness ONCE, from settled state: a batch that is fully taken back
    // leaves the patch exactly as clean as it found it, though a step landed on the way.
    g.refuse("compound", j!({ "ops": [
        { "op": "node edit", "payload": { "node": uid, "pos": [5.0, 5.0] } },
        { "op": "node edit", "payload": { "node": "ffffffffffff", "name": "ghost" } },
    ] }));
    assert_eq!(g.call("session status", j!({}))["dirty"], false,
               "a rolled-back batch must not leave the unsaved dot raised");
}

/// A canonical 12-hex uid that names nothing.

#[test]
fn a_reply_says_what_the_write_actually_did() {
    let g = Goofi::new();

    let born = g.call("node add", j!({ "type": "Oscillator" }));
    let osc = born["uid"].as_str().unwrap().to_string();
    assert!(born["name"].as_str().is_some_and(|n| !n.is_empty()), "{born}");
    assert_eq!(born["output_slots"]["out"], "ARRAY", "{born}");
    assert_eq!(born["params"]["oscillator"]["frequency"], 1.0, "{born}");

    // A literal is COERCED to the param's declared type, so the value stored may differ.
    let buf = g.add("Buffer");
    let coerced = g.set_param(buf, "buffer", "size", 512.6);
    assert_eq!(coerced["value"], 513, "an int param rounds: {coerced}");

    let wired = g.call("link add", j!({ "from": ep(&osc, "out"), "to": ep(hex(buf), "data") }));
    assert_eq!((&wired["from"], &wired["dtype"]), (&j!(ep(&osc, "out")), &j!("ARRAY")), "{wired}");

    assert_eq!(g.call("node remove", j!({ "node": GHOST }))["removed"], false);
    assert_eq!(g.call("node remove", j!({ "node": osc }))["removed"], true);
    assert_eq!(g.call("link remove", j!({ "from": ep(&osc, "out"), "to": ep(hex(buf), "data") }))["removed"],
               false);
}

#[test]
fn a_refusal_names_what_the_caller_could_try_instead() {
    let g = Goofi::new();
    let osc = g.add("Oscillator");

    // A global's TYPE is what every expression reading it depends on, so it is immutable: an
    // edit coerces to the type held, and a value the type cannot read is refused by naming it.
    let why = g.refuse("global edit", j!({ "name": "default_ufreq", "value": "fast" }));
    assert!(why.contains("float") && why.contains("fast"), "{why}");
    let why = g.refuse("global add", j!({ "name": "default_ufreq", "value": 9.0, "type": "float" }));
    assert!(why.contains("already exists") && why.contains("global edit"), "{why}");
    assert_eq!(g.call("global edit", j!({ "name": "default_ufreq", "value": 12.5 }))["value"], 12.5);

    // A LOCKED global is the machine's: the value refuses the edit the way the name refuses the
    // remove, and what it holds is this machine's .goofi folder, not anything a patch said.
    let home = g.call("global list", j!({}))["globals"].as_array().unwrap().iter()
        .find(|e| e["name"] == "goofi_home").cloned().expect("goofi_home is seeded");
    assert_eq!((&home["locked"], &home["type"]), (&j!(true), &j!("string")), "{home}");
    assert_eq!(home["value"], j!(goofi_core::path::to_slash(&goofi_core::home::dir())));
    let why = g.refuse("global edit", j!({ "name": "goofi_home", "value": "/tmp/elsewhere" }));
    assert!(why.contains("read-only"), "{why}");
    let why = g.refuse("global remove", j!({ "name": "goofi_home" }));
    assert!(why.contains("system"), "{why}");

    let why = g.refuse("agent start", j!({ "name": "claude-code" }));
    assert!(why.contains("claude") && why.contains("codex"), "{why}");

    let why = g.refuse("link add", j!({ "from": ep(hex(osc), "out"), "to": ep(GHOST, "data") }));
    assert!(why.contains("`to`") && why.contains(GHOST), "{why}");

    for (op, payload) in [
        ("nodes ungroup", j!({ "subpatch": GHOST })),
        ("node edit", j!({ "node": GHOST, "pos": [1.0, 2.0] })),
        ("node edit", j!({ "node": GHOST, "name": "renamed" })),
        ("node param edit", j!({ "node": GHOST, "param": "buffer/size", "expression": "1" })),
    ] {
        g.refuse(op, payload);
    }

    // An expression reads a name as an ATTRIBUTE — `globals.gain`, `nd('sub').out.slot` — so a name
    // Python cannot parse as one is refused, whatever makes it unparseable. The refusal says the
    // rule rather than the one character it caught.
    for bad in ["a'b", "a\\b", "a\"b", "a b-2", "nd()", "1st", "class", ""] {
        let why = g.refuse("node edit", j!({ "node": hex(osc), "name": bad }));
        assert!(why.contains("letters, digits or _"), "the refusal states the rule: {why}");
    }
    g.call("node edit", j!({ "node": hex(osc), "name": "a_b_2" }));
    // The command tolerates a collision so replay converges; the RPC boundary raises the user error.
    g.add("Buffer");
    g.refuse("node edit", j!({ "node": hex(osc), "name": "buffer0" }));
    // Nothing at all is a caller error: an op that means "edit" must be told what to edit.
    g.refuse("node edit", j!({ "node": hex(osc) }));
}

#[test]
fn an_expression_binds_carries_its_error_and_follows_the_rename_of_what_it_names() {
    let g = Goofi::new();
    let producer = g.add("Oscillator");
    let consumer = g.add("Oscillator");
    g.call("node edit", j!({ "node": hex(producer), "name": "src" }));

    // A binding that cannot compile is STORED, so the refusal has to travel in the reply.
    // An expression given with no `mode` binds: that is what writing one means.
    let set = |expr: &str| g.call("node param edit", j!({ "node": hex(consumer),
                                                          "param": "common/max_frequency",
                                                          "expression": expr }));
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

    g.call("node edit", j!({ "node": hex(producer), "name": "signal" }));
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
    let born = g.call("node add", j!({ "type": "Oscillator",
                                       "param": [{ "name": "common/max_frequency", "value": 42.0 }] }));
    let uid = born["uid"].as_str().unwrap().to_string();
    assert_eq!(ev.next("node_added")["uid"], uid);
    assert_eq!(g.doc()["nodes"][&uid]["params"]["common"]["max_frequency"]["value"], 42.0);

    // Undo/redo do NOT come through here — they restore via the command history.
    g.call("node remove", j!({ "node": uid.clone() }));
    let again = g.call("node add", j!({ "type": "Oscillator", "member_uid": uid.clone(),
                                        "name": "restored_osc" }));
    assert_eq!((&again["uid"], &g.doc()["nodes"][&uid]["name"]), (&j!(uid), &j!("restored_osc")));
}

#[test]
fn a_viewer_bag_persists_and_refuses_a_word_outside_its_vocabulary() {
    // `viewers(uid)` answers `Some({})` for every node, so an unconditional insert would stamp them all.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    assert!(g.doc()["nodes"][hex(osc)].get("viewers").is_none(), "no viewers leaf when empty");

    let why = g.refuse("node edit", j!({ "node": hex(osc),
                                         "viewer": [{ "slot": "out", "kind": "waveform" }] }));
    assert!(why.contains("waveform") && why.contains("line") && why.contains("topomap"), "{why}");
    let why = g.refuse("node edit", j!({ "node": hex(osc),
                                         "viewer": [{ "slot": "psd", "kind": "line" }] }));
    assert!(why.contains("psd") && why.contains("out"), "an unknown slot names the real ones: {why}");
    let why = g.refuse("node edit", j!({ "node": GHOST,
                                         "viewer": [{ "slot": "out", "kind": "line" }] }));
    assert!(why.contains("no such node"), "{why}");
    let why = g.refuse("node edit", j!({ "node": hex(osc), "viewer": 7 }));
    assert!(why.contains("list"), "an arg that is not a list says what one looks like: {why}");
    let why = g.refuse("node edit", j!({ "node": hex(osc), "viewer": [{ "kind": "line" }] }));
    assert!(why.contains("slot"), "an entry without a slot is refused by naming it: {why}");

    g.call("node edit", j!({ "node": hex(osc),
                             "viewer": [{ "slot": "out", "collapsed": false, "kind": "line",
                                          "settings": { "yScale": 2 } }] }));
    assert!(!g.doc()["nodes"][hex(osc)]["viewers"].is_null(), "…and the leaf appears once set");
    // Entries MERGE, slot by slot: naming one setting leaves the kind and the others where they were.
    g.call("node edit", j!({ "node": hex(osc),
                             "viewer": [{ "slot": "out", "settings": { "xScale": 3 } }] }));
    let view = |g: &Goofi| g.doc()["nodes"][hex(osc)]["viewers"].as_str().unwrap_or("").to_string();
    let merged = view(&g);
    for kept in ["\"kind\":\"line\"", "\"yScale\":2", "\"xScale\":3"] {
        assert!(merged.contains(kept), "the patch merged rather than replaced: {merged}");
    }
    // …and it is UNDOABLE, which is what makes it an op rather than a side write.
    g.call("undo", j!({}));
    assert!(!view(&g).contains("xScale"), "the undo took the merge back off: {}", view(&g));
    g.call("redo", j!({}));
    assert!(view(&g).contains("xScale"), "…and the redo put it back: {}", view(&g));
    let yaml = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    assert!(yaml.contains("yScale"), "the view state persists: {yaml}");

    // `clear` removes the slot's stored view — the entry form's spelling of the old null.
    let why = g.refuse("node edit", j!({ "node": hex(osc),
                                         "viewer": [{ "slot": "out", "clear": false }] }));
    assert!(why.contains("clear"), "a false `clear` is refused rather than read as a set: {why}");
    g.call("node edit", j!({ "node": hex(osc), "viewer": [{ "slot": "out", "clear": true }] }));
    assert!(!view(&g).contains("yScale"), "the clear dropped the stored view: {}", view(&g));

    g.call("global add", j!({ "name": "subject", "value": "P01", "type": "string" }));
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
                    client.set_param(*u, "common", "max_frequency", r as f64);
                    client.call("node edit", j!({ "node": hex(*u), "pos": [r as f64, r as f64] }));
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
