//! Node files in a workspace: written, rescanned, edited, shadowed, saved and reopened — through
//! the real scan on the real Python, so the running node IS the file's code.

use std::path::Path;

use goofi_core::Data;
use goofi_tests::{drive, j, require_python, Goofi, OutputProbe};

/// A producer that emits the number it was written with — which FILE a node runs, observable.
fn write_node(dir: &Path, file: &str, value: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let source = format!(
        "import goofi\nimport numpy as np\n\nclass Emit(goofi.Node):\n    \
         OUTPUTS = {{\"out\": goofi.DataType.ARRAY}}\n    PRODUCER = True\n\n    \
         def process(self):\n        return {{\"out\": np.array([{value}], dtype=\"float32\")}}\n"
    );
    std::fs::write(dir.join(file), source).unwrap();
}

/// The same producer in Rust — built through cargo into the cache a scan reads.
fn write_rust_node(dir: &Path, file: &str, value: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let source = format!(
        "use goofi_core::{{Data, Meta, SlotType}};\n\
         use goofi_signal_sdk::{{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, Params}};\n\n\
         #[derive(Default)]\nstruct Emit;\n\n\
         impl Node for Emit {{\n    \
         fn process(&mut self, _i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {{\n        \
         let bytes = {value}f32.to_le_bytes().to_vec();\n        \
         o.set(\"out\", Data::array_f32(vec![1], bytes, Meta::new()).map_err(|e| e.to_string())?);\n        \
         Ok(())\n    }}\n}}\n\n\
         static OUTPUTS: &[OutputDecl] = &[OutputDecl {{ name: \"out\", kind: SlotType::Array }}];\n\
         static MANIFEST: Manifest = Manifest {{\n    \
         tags: &[], doc: \"emits a number\", inputs: &[], outputs: OUTPUTS, params: &[], producer: true,\n}};\n\n\
         goofi_signal_sdk::export!(Emit, MANIFEST);\n"
    );
    std::fs::write(dir.join(file), source).unwrap();
}

fn first_f32(d: &Data) -> f32 {
    match d.value() {
        goofi_core::Value::Array(s) => f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()),
        _ => panic!("not an array"),
    }
}

/// Watch one node's `out` slot until it carries `want`. The probe is opened per call because a
/// restart is a REBIRTH onto the next generation's service, and the frame in flight may be old.
fn emits(g: &Goofi, uid: goofi_tests::Uid, want: f32) {
    let probe = OutputProbe::open(&g.state.graph.lock().unwrap(), uid, "out");
    g.until(&format!("{uid} to emit {want}"), |g| {
        let mut graph = g.state.graph.lock().unwrap();
        probe.frame(&mut graph).filter(|d| first_f32(d) == want)
    });
}

fn rescan(g: &Goofi) -> serde_json::Value {
    g.call("library refresh", j!({}))
}

#[test]
fn a_node_file_in_the_workspace_is_live_after_a_rescan_and_follows_its_edits() {
    let _py = require_python();
    let g = Goofi::new();
    let mount = g.state.mount();
    write_node(&mount.join("nodes_signal"), "my_thing.py", "1.0");

    assert_eq!(rescan(&g)["added"], j!(["signal:MyThing"]), "the file becomes a type");
    // The baseline is what the LAST scan found, so refresh with nothing edited says nothing changed.
    let again = rescan(&g);
    assert_eq!((&again["added"], &again["changed"], &again["removed"]), (&j!([]), &j!([]), &j!([])),
               "a rescan of an unchanged tree changes nothing");

    let live = g.add("MyThing");
    emits(&g, live, 1.0);

    write_node(&mount.join("nodes_signal"), "my_thing.py", "2.0");
    let diff = rescan(&g);
    assert_eq!(diff["changed"], j!(["signal:MyThing"]), "an edited file reports as changed");
    assert_eq!((&diff["added"], &diff["removed"]), (&j!([]), &j!([])));
    emits(&g, live, 2.0); // the running node is the new code

    // Removal closes the door; it does not reach into the graph.
    std::fs::remove_file(mount.join("nodes_signal").join("my_thing.py")).unwrap();
    assert_eq!(rescan(&g)["removed"], j!(["signal:MyThing"]));
    g.refuse("node add", j!({ "type": "MyThing" }));
    emits(&g, live, 2.0); // …and its instance still runs
}

#[test]
fn a_patch_local_node_wins_the_name_and_is_marked_as_the_patchs_own() {
    // The patch is scanned SECOND so its own file wins a name the shipped root also uses.
    let _py = require_python();
    let mut g = Goofi::new();
    let shipped = tempfile::tempdir().unwrap();
    write_node(shipped.path(), "my_thing.py", "1.0");
    write_node(shipped.path(), "only_shipped.py", "7.0");
    g.state.roots = vec![shipped.path().to_path_buf()];
    write_node(&g.state.mount().join("nodes_signal"), "my_thing.py", "9.0");
    rescan(&g);

    let uid = g.add("MyThing");
    emits(&g, uid, 9.0); // the patch's own file wins the name
    let source = |ty: &str| g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == ty).unwrap()["source"].clone();
    assert_eq!(source("signal:MyThing"), "patch", "…and says where it came from");
    assert_eq!(source("signal:OnlyShipped"), "builtin", "the shipped root's own node is not the patch's");
}

#[test]
fn a_later_shipped_root_wins_the_name_without_dropping_the_earlier_one() {
    // Adding a root must not COST one — the failure a REPLACING flag causes.
    let _py = require_python();
    let mut g = Goofi::new();
    let builtin = tempfile::tempdir().unwrap();
    let mine = tempfile::tempdir().unwrap();
    write_node(builtin.path(), "my_thing.py", "1.0");
    write_node(builtin.path(), "only_builtin.py", "7.0");
    write_node(mine.path(), "my_thing.py", "5.0");
    g.state.roots = vec![builtin.path().to_path_buf(), mine.path().to_path_buf()];
    rescan(&g);

    let shadowed = g.add("MyThing");
    emits(&g, shadowed, 5.0); // the later root wins the name
    let kept = g.add("OnlyBuiltin");
    emits(&g, kept, 7.0); // and the earlier root's other nodes are still registered
}

#[test]
fn a_named_type_hands_back_the_file_that_is_actually_running() {
    // `rescan` overwrites forwards, so this first-match-wins search has to walk the roots
    // backwards to agree; dropping that `.rev()` passes every other test here.
    let _py = require_python();
    let mut g = Goofi::new();
    let builtin = tempfile::tempdir().unwrap();
    let mine = tempfile::tempdir().unwrap();
    write_node(builtin.path(), "my_thing.py", "1.0");
    write_node(mine.path(), "my_thing.py", "5.0");
    g.state.roots = vec![builtin.path().to_path_buf(), mine.path().to_path_buf()];
    rescan(&g);

    let r = g.call("library get", j!({ "type": "MyThing" }));
    assert!(r["source"].as_str().is_some_and(|s| s.contains("[5.0]")),
            "the file that RUNS is the file handed back: {r}");
    assert_eq!(r["provenance"], "shipped", "{r}");
    assert_eq!(r["path"], goofi_core::path::to_slash(&mine.path().join("my_thing.py")),
               "…and it names the winning root, not the shadowed one: {r}");

    write_node(&g.state.mount().join("nodes_signal"), "my_thing.py", "9.0");
    rescan(&g);
    let r = g.call("library get", j!({ "type": "MyThing" }));
    assert_eq!(r["provenance"], "patch", "{r}");
    assert!(r["source"].as_str().is_some_and(|s| s.contains("[9.0]")), "{r}");
}

#[test]
fn loading_a_patch_registers_the_nodes_it_ships_before_resolving_them() {
    // The ORDER is load-bearing: `load_doc` rejects a type it does not know.
    let _py = require_python();
    let g = Goofi::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("patch.gfi");
    write_node(&g.state.mount().join("nodes_signal"), "my_thing.py", "5.0");
    rescan(&g);
    g.add("MyThing");
    g.call("session save", j!({ "path": target.to_string_lossy() }));

    // A SECOND manager, which is the real case: it has never seen this type.
    let opened = Goofi::new();
    opened.call("session load", j!({ "path": target.to_string_lossy() }));
    assert_eq!(opened.nodes().len(), 1);
    let uid = opened.state.graph.lock().unwrap().node_uids()[0];
    emits(&opened, uid, 5.0); // the instance runs the patch's code

    // `new` swaps in an empty workspace, so a type the previous patch brought stops being addable.
    opened.call("session new", j!({}));
    opened.refuse("node add", j!({ "type": "MyThing" }));
}

#[test]
fn a_rust_node_file_builds_loads_follows_its_edits_and_shadows_a_shipped_one() {
    let g = Goofi::new();
    // A shipped node is SOURCE in the shipped root, where `library get` finds it.
    let r = g.call("library get", j!({ "type": "Oscillator" }));
    assert_eq!((&r["provenance"], &r["language"], &r["tier"]), (&j!("shipped"), &j!("rust"), &j!("native")), "{r}");
    assert!(r["source"].as_str().is_some_and(|s| s.contains("impl Node for Oscillator")), "{r}");
    let shipped_osc = std::path::PathBuf::from(r["path"].as_str().unwrap());

    // An authored file builds through cargo into the same cache, and runs.
    let mount = g.state.mount();
    write_rust_node(&mount.join("nodes_signal"), "Twice.rs", "2.0");
    assert_eq!(rescan(&g)["added"], j!(["signal:Twice"]), "the file becomes a type");
    let live = g.add("Twice");
    emits(&g, live, 2.0);
    write_rust_node(&mount.join("nodes_signal"), "Twice.rs", "3.0");
    assert_eq!(rescan(&g)["changed"], j!(["signal:Twice"]), "an edited file reports as changed");
    emits(&g, live, 3.0); // the running node is the new code

    // A file that does not compile greys the type out with rustc's own words, and the instance
    // built from the last good file runs on.
    std::fs::write(mount.join("nodes_signal").join("Twice.rs"), "fn broken( {\n").unwrap();
    rescan(&g);
    let row = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == "signal:Twice").cloned().expect("the type stays listed, greyed");
    assert_eq!(row["available"], false, "{row}");
    assert!(row["missing_deps"].to_string().contains("error"), "rustc's words reach the palette: {row}");
    // The greyed row keeps the SHAPE it last loaded, because the canvas draws a node's slots and
    // params from it: the instance is still running and still wired, and a row with no slots
    // erased it from every open tab while its data kept flowing.
    assert_eq!(row["output_slots"], j!({ "out": "ARRAY" }), "the greyed row keeps its slots: {row}");
    assert!(row["params"].as_object().is_some_and(|p| !p.is_empty()), "and its params: {row}");
    let why = g.refuse("node add", j!({ "type": "Twice" }));
    assert!(why.contains("unavailable"), "{why}");
    emits(&g, live, 3.0);
    write_rust_node(&mount.join("nodes_signal"), "Twice.rs", "3.0");
    assert_eq!(rescan(&g)["changed"], j!(["signal:Twice"]), "the fix is a change, built from the cache");

    // A stem the name rule refuses is not a node, in either language, exactly as a `_` stem is not:
    // nothing is built, nothing is listed.
    write_rust_node(&mount.join("nodes_signal"), "my-node.rs", "1.0");
    write_node(&mount.join("nodes_signal"), "2d.py", "1.0");
    assert_eq!(rescan(&g)["added"], j!([]));
    let listed = g.call("library list", j!({}));
    let names: Vec<&str> = listed["types"].as_array().unwrap().iter().filter_map(|v| v["type"].as_str()).collect();
    assert!(!names.iter().any(|n| ["my-node", "2d", "My-node"].contains(n)), "{names:?}");

    // An author's `Drop` that panics costs its own instance, never the process: the boundary
    // catches at every entry, teardown included.
    std::fs::write(
        mount.join("nodes_signal").join("Doomed.rs"),
        "use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, Outputs, Params};\n\
         #[derive(Default)]\nstruct Doomed;\n\
         impl Drop for Doomed { fn drop(&mut self) { panic!(\"doomed\") } }\n\
         impl Node for Doomed {\n    \
         fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult { Ok(()) }\n}\n\
         static MANIFEST: Manifest = Manifest { tags: &[], doc: \"panics on drop\", inputs: &[], outputs: &[], params: &[], producer: true };\n\
         goofi_signal_sdk::export!(Doomed, MANIFEST);\n",
    )
    .unwrap();
    assert_eq!(rescan(&g)["added"], j!(["signal:Doomed"]));
    let doomed = g.add("Doomed");
    g.call("node remove", j!({ "node": goofi_tests::hex(doomed) }));
    g.until("the doomed instance to be gone", |g| (g.state.graph.lock().unwrap().node_count() == 1).then_some(()));
    assert!(g.call("library list", j!({}))["types"].as_array().is_some(), "the server answers after the drop");

    // An audio slot belongs to the audio SDK: a signal node that declares one is greyed out with
    // the SDK named, never registered.
    std::fs::write(
        mount.join("nodes_signal").join("Loud.rs"),
        "use goofi_core::SlotType;\n\
         use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, Params};\n\
         #[derive(Default)]\nstruct Loud;\n\
         impl Node for Loud {\n    \
         fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult { Ok(()) }\n}\n\
         static OUTS: &[OutputDecl] = &[OutputDecl { name: \"out\", kind: SlotType::Audio }];\n\
         static MANIFEST: Manifest = Manifest { tags: &[], doc: \"claims an audio slot\", inputs: &[], outputs: OUTS, params: &[], producer: true };\n\
         goofi_signal_sdk::export!(Loud, MANIFEST);\n",
    )
    .unwrap();
    rescan(&g);
    let loud = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == "signal:Loud").cloned().expect("listed, greyed");
    assert_eq!(loud["available"], false, "{loud}");
    assert!(loud["missing_deps"].to_string().contains("goofi_audio_sdk"), "{loud}");

    // A shipped file copied into the patch shadows it, and the palette says so — a copy that kept
    // its source's mtime included, which is what a Finder copy and `fs::copy` on macOS make.
    let copy = mount.join("nodes_signal").join("Oscillator.rs");
    std::fs::copy(&shipped_osc, &copy).unwrap();
    let kept = std::fs::metadata(&shipped_osc).unwrap().modified().unwrap();
    std::fs::File::options().write(true).open(&copy).unwrap().set_modified(kept).unwrap();
    assert!(rescan(&g)["changed"].as_array().unwrap().contains(&j!("signal:Oscillator")));
    let source = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == "signal:Oscillator").unwrap()["source"].clone();
    assert_eq!(source, "patch");

    // The archive carries the SOURCE; a second goofi builds or finds the artifact and runs it.
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("rust.gfi");
    g.call("session save", j!({ "path": target.to_string_lossy() }));
    let opened = Goofi::new();
    opened.call("session load", j!({ "path": target.to_string_lossy() }));
    let uid = opened.state.graph.lock().unwrap().node_uids()[0];
    emits(&opened, uid, 3.0);
}

/// An audio producer that holds the level it was written with — which FILE a node runs, heard.
fn write_audio_node(dir: &Path, file: &str, value: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let source = format!(
        "use goofi_audio_sdk::goofi_core::SlotType;\n\
         use goofi_audio_sdk::{{AudioNode, Block, Manifest, OutputDecl}};\n\n\
         #[derive(Default)]\nstruct Level;\n\n\
         impl AudioNode for Level {{\n    \
         fn prepare(&mut self, _rate: f64) {{}}\n    \
         fn process(&mut self, b: &mut Block<'_>) {{\n        \
         b.outs[0].chan_mut(0).fill({value}f32);\n    \
         }}\n}}\n\n\
         static OUTS: &[OutputDecl] = &[OutputDecl {{ name: \"out\", kind: SlotType::Audio }}];\n\
         static MANIFEST: Manifest = Manifest {{ tags: &[], doc: \"holds a level\", inputs: &[], outputs: OUTS, params: &[] }};\n\n\
         goofi_audio_sdk::export!(Level, MANIFEST);\n"
    );
    std::fs::write(dir.join(file), source).unwrap();
}

const PY_LEVEL: &str = "import goofi\nimport numpy as np\nclass Level(goofi.Node):\n    OUTPUTS = {\"out\": goofi.DataType.ARRAY}\n    def process(self):\n        return {\"out\": np.zeros(1, dtype=np.float32)}\n";

/// Drive the audio clock and watch one node's `out` tap until it holds `want`.
fn holds(g: &Goofi, uid: goofi_tests::Uid, want: f32) {
    let probe = OutputProbe::open(&g.state.graph.lock().unwrap(), uid, "out");
    g.until(&format!("{uid} to hold {want}"), |g| {
        drive(g, 4800);
        probe.frame(&mut g.state.graph.lock().unwrap()).filter(|d| first_f32(d) == want)
    });
}

#[test]
fn an_audio_node_file_builds_loads_follows_its_edits_and_rides_an_archive() {
    let g = Goofi::new();
    let r = g.call("library get", j!({ "type": "Osc" }));
    assert_eq!((&r["provenance"], &r["language"], &r["tier"]), (&j!("shipped"), &j!("rust"), &j!("native")), "{r}");
    assert!(r["source"].as_str().is_some_and(|s| s.contains("impl AudioNode for Osc")), "{r}");

    let mount = g.state.mount();
    write_audio_node(&mount.join("nodes_audio"), "Level.rs", "0.25");
    assert_eq!(rescan(&g)["added"], j!(["audio:Level"]), "the file becomes a type");
    let live = g.add("Level");
    holds(&g, live, 0.25);
    // A signal node with the same stem is another type, and `library get` finds each one's file.
    std::fs::create_dir_all(mount.join("nodes_signal")).unwrap();
    std::fs::write(mount.join("nodes_signal").join("Level.py"), PY_LEVEL).unwrap();
    assert_eq!(rescan(&g)["added"], j!(["signal:Level"]), "two engines offer one name");
    let (audio, signal) = (g.call("library get", j!({ "type": "audio:Level" })), g.call("library get", j!({ "type": "signal:Level" })));
    assert!(audio["source"].as_str().is_some_and(|s| s.contains("impl AudioNode for Level")), "{audio}");
    assert!(signal["source"].as_str().is_some_and(|s| s.contains("class Level")), "{signal}");
    write_audio_node(&mount.join("nodes_audio"), "Level.rs", "0.5");
    assert_eq!(rescan(&g)["changed"], j!(["audio:Level"]), "an edited file reports as changed");
    holds(&g, live, 0.5);

    // A file that does not compile greys the type out with rustc's own words, and the instance
    // built from the last good file runs on.
    std::fs::write(mount.join("nodes_audio").join("Level.rs"), "fn broken( {\n").unwrap();
    rescan(&g);
    let row = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == "audio:Level").cloned().expect("the type stays listed, greyed");
    assert_eq!(row["available"], false, "{row}");
    assert!(row["missing_deps"].to_string().contains("error"), "rustc's words reach the palette: {row}");
    assert!(g.refuse("node add", j!({ "type": "audio:Level" })).contains("unavailable"));
    holds(&g, live, 0.5);
    write_audio_node(&mount.join("nodes_audio"), "Level.rs", "0.5");
    assert_eq!(rescan(&g)["changed"], j!(["audio:Level"]), "the fix is a change, built from the cache");

    // A file whose stem is a built-in node's name is not a node file: nothing is added, nothing
    // changes when it is edited, the built-in's row stands alone and is not the patch's.
    std::fs::write(mount.join("nodes_audio").join("AudioOut.rs"), "fn never_built() {}\n").unwrap();
    let scanned = rescan(&g);
    assert_eq!((&scanned["added"], &scanned["changed"]), (&j!([]), &j!([])), "{scanned}");
    std::fs::write(mount.join("nodes_audio").join("AudioOut.rs"), "fn never_built_either() {}\n").unwrap();
    assert_eq!(rescan(&g)["changed"], j!([]));
    let rows: Vec<serde_json::Value> = g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .filter(|v| v["type"] == "audio:AudioOut").cloned().collect();
    assert!(rows.len() == 1 && rows[0]["available"] == true && rows[0]["source"] != "patch", "{rows:?}");
    assert_ne!(g.call("library get", j!({ "type": "AudioOut" }))["provenance"], "patch");

    // The archive carries the SOURCE; a second goofi builds or finds the artifact and runs it.
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("audio.gfi");
    g.call("session save", j!({ "path": target.to_string_lossy() }));
    let opened = Goofi::new();
    opened.call("session load", j!({ "path": target.to_string_lossy() }));
    let uid = opened.state.graph.lock().unwrap().node_uids()[0];
    holds(&opened, uid, 0.5);
}
