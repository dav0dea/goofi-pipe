//! The node runtime as the control plane sees it: a node reports its own state, and the
//! status-drain worker is the one thing that applies those reports.
//!
//! There is no tick. A node is known when `add_node` answers and addressable only when it reports
//! `Ready`, so every assertion here is a poll, never a single look.

use goofi_tests::{hex, j, Goofi};

/// A node whose FIRST instance fails to boot and whose second succeeds — so a restart is observable
/// as recovery rather than as "the op did not error".
static FLAKY: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "FlakyBoot",
    category: "test",
    doc: "fails setup once, then succeeds",
    inputs: &[],
    outputs: &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }],
    params: &[],
    isolation: goofi_node::Isolation::InProcess,
    producer: true,
    factory: || unreachable!("a dyn type is built by its registered factory"),
};

struct FlakyBoot {
    fail: bool,
}
impl goofi_node::Node for FlakyBoot {
    fn setup(&mut self, _c: &mut goofi_node::NodeCtx, _p: &goofi_node::Params<'_>) -> goofi_node::NodeResult {
        if self.fail { Err("boot failed".into()) } else { Ok(()) }
    }
    fn process(&mut self, _i: &goofi_node::Inputs<'_>, _o: &mut goofi_node::Outputs<'_>,
               _c: &mut goofi_node::NodeCtx, _p: &goofi_node::Params<'_>) -> goofi_node::NodeResult {
        Ok(())
    }
}

#[test]
fn restart_rebuilds_the_instance_and_clears_the_error() {
    // The button exists to rescue a crashed node, so the proof is that the node's error goes away —
    // not merely that the op returned Ok.
    let g = Goofi::new();
    let builds = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    g.register_dyn(&FLAKY, Box::new(move |_| {
        let n = builds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::new(FlakyBoot { fail: n == 0 })
    }));

    let uid = g.add("FlakyBoot");
    let why = g.until("the first instance to fail", |g| g.error(uid));
    assert!(why.contains("boot failed"), "{why}");

    g.call("restart_node", j!({ "node": hex(uid) }));
    g.until("the second instance to boot clean", |g| g.error(uid).is_none().then_some(()));
}

#[test]
fn a_restart_is_recovery_and_leaves_the_undo_stack_alone() {
    // The client records no history entry for a restart, so the manager must not either — else undo
    // would flip the restart instead of the user's last real edit.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");

    g.call("restart_node", j!({ "node": hex(buf) }));
    g.call("undo", j!({}));
    assert_eq!(g.doc()["links"].as_array().map(Vec::len), Some(0), "undo removed the LINK");
    assert_eq!(g.nodes().len(), 2, "both nodes survive");
}

#[test]
fn a_restart_changes_nothing_that_reaches_the_gfi_and_so_does_not_dirty_the_patch() {
    // The dirty gate derives "the patch differs from disk" from "the op could have mutated the
    // graph". `restart_node` is the one op where that inference is simply false — and it is reached
    // from the RECOVERY path, where a user is least able to tell a spurious dot from a real one.
    let g = Goofi::new();
    g.add("Oscillator");
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    // A load is how a patch becomes "the same as disk" without a filesystem.
    g.call("load_text", j!({ "content": yaml }));
    assert_eq!(g.call("get_patch", j!({}))["dirty"], false, "a freshly loaded patch matches disk");

    let uid = g.nodes().pop().expect("the loaded node");
    let before = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("restart_node", j!({ "node": uid }));
    let after = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();

    assert_eq!(before, after, "a restart changes nothing that reaches the .gfi");
    assert_eq!(g.call("get_patch", j!({}))["dirty"], false, "so it must not dirty the patch");
}

#[test]
fn refreshing_a_param_echoes_the_fresh_options_and_clears_the_spinner() {
    // Options live only in runtime state, never in the doc, so they reach the browser ONLY via this
    // echo — and `refreshed_params` is what lifts the ⟳ spinner. The RPC reply says only that the
    // request went out; the node answers on its own thread.
    let g = Goofi::new();
    let uid = g.add("_TestPicker");
    g.ready(uid);
    let mut ev = g.events();
    g.call("refresh_param", j!({ "node": hex(uid), "group": "io", "name": "device" }));

    let p = g.until("the picker's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(uid)).then_some(p)
    });
    let options = p["params"]["io"]["device"]["options"].as_array().cloned().unwrap_or_default();
    assert!(options.iter().any(|o| o.as_str().is_some_and(|s| s.starts_with("dev"))),
            "the re-enumerated list reached the client: {options:?}");
    assert_eq!(p["refreshed_params"], j!([["io", "device"]]), "…and the spinner is cleared");
}

#[test]
fn refreshing_reports_completion_even_when_the_node_offers_nothing() {
    // A node that declares a refreshable param but implements no hook must still get its echo:
    // without it the button spins for its full safety timeout on every such node.
    let g = Goofi::new();
    let uid = g.add("_TestMute");
    g.ready(uid);
    let mut ev = g.events();
    g.call("refresh_param", j!({ "node": hex(uid), "group": "io", "name": "device" }));

    let p = g.until("the mute picker's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(uid)).then_some(p)
    });
    assert_eq!(p["refreshed_params"], j!([["io", "device"]]), "the spinner is cleared");
    assert_eq!(p["params"]["io"]["device"]["options"], j!(["none"]),
               "and the declared options are left as they were");
}

#[test]
fn refreshing_a_param_that_is_not_refreshable_is_refused() {
    // Oscillator's waveform is a fixed list. The frontend lifts the spinner on a rejected call.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let why = g.refuse("refresh_param", j!({ "node": hex(osc), "group": "oscillator",
                                            "name": "waveform" }));
    assert!(why.contains("not refreshable"), "{why}");
}

#[test]
fn a_free_running_node_reports_its_measured_update_rate() {
    // The node header's live rate. The producer was once orphaned entirely, so what this drives is
    // the report reaching a client — asserting the event never arrives would pass against a
    // broadcaster that had been deleted.
    let g = Goofi::new();
    let mut ev = g.events();
    let src = g.add("Oscillator");

    let stats = g.until("a node_stats broadcast", |_| {
        let p = ev.next("node_stats");
        (p["node"] == hex(src)).then_some(p)
    });
    assert!(stats["stats"]["updates_per_second"].is_number(), "got {stats}");
}

#[test]
fn the_dirty_flag_tracks_mutations_and_a_save_clears_both_halves_of_it() {
    // The title-bar dot and the unload guard both read this. It is DERIVED, not stored: any
    // successful mutation dirties the patch, and a save (or a load) makes it clean.
    let g = Goofi::new();
    let dirty = |g: &Goofi| g.call("get_patch", j!({}))["dirty"] == true;
    assert!(!dirty(&g), "a fresh session is clean");

    let mut ev = g.events();
    g.add("Oscillator");
    assert_eq!(ev.next("unsaved_changes")["unsaved_changes"], true, "adding a node dirties it");

    // `unsaved_changes` is a COMPOSITE — the graph flag OR a workspace that drifted from its
    // archive — but the announcement used to ride the flag's transition alone. So a patch dirtied
    // only by a file written into the mount saved silently, and every tab kept its dot armed on a
    // patch that was by then entirely on disk.
    std::fs::write(g.state.mount().join("agent.md"), b"notes").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.call("save", j!({ "path": path.to_string_lossy() }));
    assert_eq!(ev.next("unsaved_changes")["unsaved_changes"], false, "every tab is told it is saved");

    // …and it STAYS clean: the dispatch tail sets the flag on any op it does not recognise as
    // read-only, which is why `save` is in that set despite writing a file.
    g.call("list_nodes", j!({}));
    g.call("serialize", j!({}));
    assert!(!dirty(&g), "a read must not re-dirty the patch");
}

#[test]
fn deleting_a_busy_node_never_waits_for_it_under_the_graph_lock() {
    // The one lock property the async runtime does NOT get for free. `remove_node` runs under the
    // same mutex every control RPC needs, and a node only observes its halt flag BETWEEN runs — so
    // waiting on one parked inside `process()` would freeze the whole app for that window (a real
    // subprocess node waits out its 10 s timeout).
    //
    // Its sibling — "a running node never strands the lock" — went with the tick it described:
    // nothing holds the lock across a `process()` any more, so that assertion now holds against
    // every implementation, correct or not. This one is real code that could wait, and
    // `Graph::shutdown`, which waits on purpose at exit, is the proof that waiting is one line away.
    let g = Goofi::new();
    let uid = g.add("_TestSlow");
    // Long enough for the node's own thread to be inside its run, and far short of its end.
    std::thread::sleep(std::time::Duration::from_millis(60));

    let t0 = std::time::Instant::now();
    g.call("remove_node", j!({ "node": hex(uid) }));
    let held = t0.elapsed();
    assert!(held < std::time::Duration::from_millis(100),
            "delete returned in {held:?} — it waited on the busy node under the graph lock");
}
