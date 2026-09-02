//! Node files in a workspace: written, rescanned, edited, shadowed, saved and reopened — through
//! the real scan on the real Python, so the running node IS the file's code.

use std::path::Path;

use goofi_core::Data;
use goofi_tests::{j, require_python, Goofi, OutputProbe};

/// A producer that emits the number it was written with — which FILE a node runs, observable.
fn write_node(root: &Path, file: &str, value: &str) {
    let dir = root.join("nodes_signal");
    std::fs::create_dir_all(&dir).unwrap();
    let source = format!(
        "import goofi\nimport numpy as np\n\nclass Emit(goofi.Node):\n    \
         OUTPUTS = {{\"out\": goofi.DataType.ARRAY}}\n    PRODUCER = True\n\n    \
         def process(self):\n        return {{\"out\": np.array([{value}], dtype=\"float32\")}}\n"
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
    write_node(&mount, "my_thing.py", "1.0");

    assert_eq!(rescan(&g)["added"], j!(["MyThing"]), "the file becomes a type");
    // The baseline is what the LAST scan found, so refresh with nothing edited says nothing changed.
    let again = rescan(&g);
    assert_eq!((&again["added"], &again["changed"], &again["removed"]), (&j!([]), &j!([]), &j!([])),
               "a rescan of an unchanged tree changes nothing");

    let live = g.add("MyThing");
    emits(&g, live, 1.0);

    write_node(&mount, "my_thing.py", "2.0");
    let diff = rescan(&g);
    assert_eq!(diff["changed"], j!(["MyThing"]), "an edited file reports as changed");
    assert_eq!((&diff["added"], &diff["removed"]), (&j!([]), &j!([])));
    emits(&g, live, 2.0); // the running node is the new code

    // Removal closes the door; it does not reach into the graph.
    std::fs::remove_file(mount.join("nodes_signal").join("my_thing.py")).unwrap();
    assert_eq!(rescan(&g)["removed"], j!(["MyThing"]));
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
    write_node(&g.state.mount(), "my_thing.py", "9.0");
    rescan(&g);

    let uid = g.add("MyThing");
    emits(&g, uid, 9.0); // the patch's own file wins the name
    let source = |ty: &str| g.call("library list", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == ty).unwrap()["source"].clone();
    assert_eq!(source("MyThing"), "patch", "…and says where it came from");
    assert_eq!(source("OnlyShipped"), "builtin", "the shipped root's own node is not the patch's");
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
    assert_eq!(r["path"], goofi_core::path::to_slash(&mine.path().join("nodes_signal").join("my_thing.py")),
               "…and it names the winning root, not the shadowed one: {r}");

    write_node(&g.state.mount(), "my_thing.py", "9.0");
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
    write_node(&g.state.mount(), "my_thing.py", "5.0");
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
