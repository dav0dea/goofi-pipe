//! Node files in a workspace: written, rescanned, edited, shadowed, saved and reopened.
//!
//! The scan seam is injected, and the stub captures each file's CONTENT at scan time — which is
//! what makes "the running node is the NEW code" observable at all.

use std::path::Path;
use std::sync::Arc;

use goofi_bridge::{ScannedType, Tier};
use goofi_core::{Data, Meta};
use goofi_engine::Graph;
use goofi_node::{Inputs, Isolation, Node, NodeCtx, NodeError, NodeManifest, NodeResult, OutputDecl,
                 Outputs, Params};
use goofi_tests::{j, Goofi, OutputProbe};

static OUT: &[OutputDecl] = &[OutputDecl { name: "out", kind: goofi_core::SlotType::Array }];

/// A node that emits the number its file held WHEN THE SCAN RAN.
struct Emit(f32);
impl Node for Emit {
    fn process(&mut self, _i: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        let d = Data::array_f32(vec![1], self.0.to_le_bytes().to_vec(), Meta::empty())
            .map_err(|e| NodeError(e.to_string()))?;
        out.set("out", d);
        Ok(())
    }
}

fn stub_scan(g: &mut Graph, dir: &Path) -> Vec<ScannedType> {
    let mut paths: Vec<_> =
        std::fs::read_dir(dir).into_iter().flatten().filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    let mut out = Vec::new();
    for path in paths {
        if path.extension().and_then(|e| e.to_str()) != Some("py") {
            continue;
        }
        let name = goofi_node::discover::camel(&path.file_stem().unwrap().to_string_lossy());
        let value: f32 =
            std::fs::read_to_string(&path).unwrap_or_default().trim().parse().unwrap_or(0.0);
        let manifest: &'static NodeManifest = Box::leak(Box::new(NodeManifest {
            type_name: Box::leak(name.clone().into_boxed_str()),
            category: "python",
            doc: "a scanned node",
            inputs: &[],
            outputs: OUT,
            params: &[],
            isolation: Isolation::InProcess,
            producer: true,
            factory: || unreachable!("a scanned type is built by its registered factory"),
        }));
        out.push(ScannedType {
            type_name: name,
            tier: Tier::InProcess,
            stamp: std::fs::metadata(&path).ok().map(|m| (m.len(), m.modified().unwrap())),
            registration: g.register_dyn_type(manifest, Box::new(move |_| Box::new(Emit(value)))),
        });
    }
    out
}

/// A manager whose node scan is the stub above.
fn scanning() -> Goofi {
    let mut g = Goofi::new();
    g.state.scan_nodes = Arc::new(stub_scan);
    g
}

fn write_node(dir: &Path, file: &str, body: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(file), body).unwrap();
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
    g.call("rescan_nodes", j!({}))
}

#[test]
fn a_node_file_in_the_workspace_is_live_after_a_rescan_and_follows_its_edits() {
    let g = scanning();
    let nodes = g.state.mount().join("nodes");
    write_node(&nodes, "my_thing.py", "1.0");

    assert_eq!(rescan(&g)["added"], j!(["MyThing"]), "the file becomes a type");
    // The baseline is what the LAST scan found, so refresh with nothing edited says nothing changed.
    let again = rescan(&g);
    assert_eq!((&again["added"], &again["changed"], &again["removed"]), (&j!([]), &j!([]), &j!([])),
               "a rescan of an unchanged tree changes nothing");

    let live = g.add("MyThing");
    emits(&g, live, 1.0);

    write_node(&nodes, "my_thing.py", "2.0");
    let diff = rescan(&g);
    assert_eq!(diff["changed"], j!(["MyThing"]), "an edited file reports as changed");
    assert_eq!((&diff["added"], &diff["removed"]), (&j!([]), &j!([])));
    emits(&g, live, 2.0); // the running node is the new code

    // Removal closes the door; it does not reach into the graph.
    std::fs::remove_file(nodes.join("my_thing.py")).unwrap();
    assert_eq!(rescan(&g)["removed"], j!(["MyThing"]));
    g.refuse("add_node", j!({ "type": "MyThing" }));
    emits(&g, live, 2.0); // …and its instance still runs
}

#[test]
fn a_patch_local_node_wins_the_name_and_is_marked_as_the_patchs_own() {
    // The patch is scanned SECOND so its own file wins a name the shipped tree also uses.
    let mut g = scanning();
    let shipped = tempfile::tempdir().unwrap();
    write_node(shipped.path(), "my_thing.py", "1.0");
    write_node(shipped.path(), "only_shipped.py", "7.0");
    g.state.system_nodes = vec![shipped.path().to_path_buf()];
    write_node(&g.state.mount().join("nodes"), "my_thing.py", "9.0");
    rescan(&g);

    let uid = g.add("MyThing");
    emits(&g, uid, 9.0); // the patch's own file wins the name
    let source = |ty: &str| g.call("list_nodes", j!({}))["types"].as_array().unwrap().iter()
        .find(|v| v["type"] == ty).unwrap()["source"].clone();
    assert_eq!(source("MyThing"), "patch", "…and says where it came from");
    assert_eq!(source("OnlyShipped"), "builtin", "the shipped tree's own node is not the patch's");
}

#[test]
fn a_later_shipped_directory_wins_the_name_without_dropping_the_earlier_tree() {
    // Adding a directory must not COST one — the failure a REPLACING flag causes.
    let mut g = scanning();
    let builtin = tempfile::tempdir().unwrap();
    let mine = tempfile::tempdir().unwrap();
    write_node(builtin.path(), "my_thing.py", "1.0");
    write_node(builtin.path(), "only_builtin.py", "7.0");
    write_node(mine.path(), "my_thing.py", "5.0");
    g.state.system_nodes = vec![builtin.path().to_path_buf(), mine.path().to_path_buf()];
    rescan(&g);

    let shadowed = g.add("MyThing");
    emits(&g, shadowed, 5.0); // the later directory wins the name
    let kept = g.add("OnlyBuiltin");
    emits(&g, kept, 7.0); // and the earlier directory's other nodes are still registered
}

#[test]
fn a_named_type_hands_back_the_file_that_is_actually_running() {
    // `rescan` overwrites forwards, so this first-match-wins search has to walk the list backwards
    // to agree; dropping that `.rev()` passes every other test here.
    let mut g = scanning();
    let builtin = tempfile::tempdir().unwrap();
    let mine = tempfile::tempdir().unwrap();
    write_node(builtin.path(), "my_thing.py", "1.0");
    write_node(mine.path(), "my_thing.py", "5.0");
    g.state.system_nodes = vec![builtin.path().to_path_buf(), mine.path().to_path_buf()];
    rescan(&g);

    let r = g.call("list_nodes", j!({ "type": "MyThing" }));
    assert_eq!(r["source"], "5.0", "the file that RUNS is the file handed back: {r}");
    assert_eq!(r["provenance"], "shipped", "{r}");
    assert_eq!(r["path"], goofi_core::path::to_slash(&mine.path().join("my_thing.py")),
               "…and it names the winning directory, not the shadowed one: {r}");

    write_node(&g.state.mount().join("nodes"), "my_thing.py", "9.0");
    rescan(&g);
    let r = g.call("list_nodes", j!({ "type": "MyThing" }));
    assert_eq!(r["provenance"], "patch", "{r}");
    assert_eq!(r["source"], "9.0", "{r}");
}

#[test]
fn loading_a_patch_registers_the_nodes_it_ships_before_resolving_them() {
    // The ORDER is load-bearing: `load_doc` rejects a type it does not know.
    let g = scanning();
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("patch.gfi");
    write_node(&g.state.mount().join("nodes"), "my_thing.py", "5.0");
    rescan(&g);
    g.add("MyThing");
    g.call("save", j!({ "path": target.to_string_lossy() }));

    // A SECOND manager, which is the real case: it has never seen this type.
    let opened = scanning();
    opened.call("load", j!({ "path": target.to_string_lossy() }));
    assert_eq!(opened.nodes().len(), 1);
    let uid = opened.state.graph.lock().unwrap().node_uids()[0];
    emits(&opened, uid, 5.0); // the instance runs the patch's code

    // `new` swaps in an empty workspace, so a type the previous patch brought stops being addable.
    opened.call("load", j!({}));
    opened.refuse("add_node", j!({ "type": "MyThing" }));
}
