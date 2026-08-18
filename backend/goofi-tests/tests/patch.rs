//! A patch is an ARCHIVE: a `.gfi` zip holding `patch.yaml` beside the workspace tree it was saved
//! with. The manager owns where it lives, and both halves of "where" fail separately — the live
//! broadcast converges the tabs already open, `get_patch` converges the ones that ask later.
//!
//! A load is all-or-nothing: it extracts into a FRESH mount, parses, and only then swaps. Graph and
//! workspace together, or neither.

use goofi_tests::{j, Goofi};

/// A path as goofi spells it back: `/` on every platform. A test comparing against the platform's
/// own spelling would pass on unix and pin the Windows bug — it once asserted `C:\Users\…` against
/// the `C:/Users/…` the wire actually carries.
fn spelled(p: &std::path::Path) -> String {
    goofi_core::path::to_slash(p)
}

fn save_path(g: &Goofi) -> Option<String> {
    g.call("get_patch", j!({}))["save_path"].as_str().map(str::to_string)
}

fn dirty(g: &Goofi) -> bool {
    g.call("get_patch", j!({}))["dirty"] == true
}

fn yaml(g: &Goofi) -> String {
    g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string()
}

#[test]
fn list_dir_browses_the_backend_filesystem() {
    let g = Goofi::new();
    // No path ⇒ home, which is where the Save/Load modal opens on a fresh patch.
    let home = g.call("list_dir", j!({}));
    assert!(std::path::Path::new(home["path"].as_str().unwrap()).is_absolute(), "{home}");
    assert!(home["roots"].as_array().unwrap().iter().any(|r| r["label"] == "Home"),
            "the sidebar needs at least a Home root: {}", home["roots"]);

    let repo = std::env::current_dir().unwrap();
    let listing = g.call("list_dir", j!({ "path": repo.to_string_lossy() }));
    let entries = listing["entries"].as_array().unwrap();
    let src = entries.iter().find(|e| e["name"] == "src").expect("this crate has a src/ dir");
    assert_eq!(src["kind"], "dir");
    assert_eq!(src["is_gfi"], false);
    assert_eq!(src["hidden"], false);
    assert_eq!(listing["parent"].as_str(), repo.parent().map(spelled).as_deref());
}

#[test]
fn a_save_writes_an_archive_and_needs_a_path_to_write_it_to() {
    // "Save in browser" is gone (user decision, 2026-08-08): a save's ONLY job is writing the patch
    // to a backend path. The old no-path form returned YAML for a download and left the dirty flag
    // standing — a second save semantics. A save with no path is a malformed request, not a mode.
    let g = Goofi::new();
    g.add("Oscillator");
    let why = g.refuse("save", j!({}));
    assert!(why.contains("save") && why.contains("path"), "{why}");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.call("save", j!({ "path": path.to_string_lossy() }));

    // Readable only through `read_gfi` — a bare-YAML write would leave it "not a zip archive".
    let dest = dir.path().join("unpacked");
    let manifest = goofi_engine::archive::read_gfi(&path, &dest).unwrap();
    assert!(manifest.contains("Oscillator"), "the manifest is the serialized patch: {manifest}");
    assert!(dest.is_dir(), "the workspace tree rides along, empty or not");
}

#[test]
fn only_a_patch_with_a_file_behind_it_keeps_a_name() {
    // The stored path always names a file this patch was really written to or read from, because a
    // plain Save overwrites it silently, from any tab, with no second prompt.
    let g = Goofi::new();
    assert_eq!(save_path(&g), None, "an unsaved patch has no home yet");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.add("Oscillator");
    g.call("save", j!({ "path": path.to_string_lossy() }));
    assert_eq!(save_path(&g).as_deref(), Some(spelled(&path).as_str()));

    // A save that FAILS leaves the previous home standing: the patch has never been written to the
    // file it was refused, so naming it would aim the next silent overwrite at it.
    let nowhere = dir.path().join("no-such-dir").join("patch.gfi");
    g.refuse("save", j!({ "path": nowhere.to_string_lossy() }));
    assert_eq!(save_path(&g).as_deref(), Some(spelled(&path).as_str()), "the old home stands");

    // An upload (`load_text`) carries no file, so the patch it installs is UNNAMED. Inheriting the
    // previous path here is the silent-overwrite hazard in its purest form: a different patch
    // entirely, saved over a file it never came from.
    let content = yaml(&g);
    g.call("load_text", j!({ "content": content }));
    assert_eq!(save_path(&g), None, "an uploaded patch has no home");
}

#[test]
fn a_save_tells_every_open_tab_where_the_patch_now_lives() {
    // C38: the manager owns the path, so every client agrees about it. It used to own none — only
    // the `load` arm announced one — so a save named the patch in the tab that performed it and
    // nowhere else.
    let g = Goofi::new();
    let mut ev = g.events();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.call("save", j!({ "path": path.to_string_lossy() }));
    assert_eq!(ev.next("save_path_changed")["save_path"].as_str(), Some(spelled(&path).as_str()));
}

#[test]
fn a_panel_binding_survives_a_load_into_an_instance_that_held_other_nodes() {
    // A viewer panel binds by node UID and a load does not remap it, so the load has to bring the
    // uid back. It must survive into an instance that has already held OTHER nodes — the only
    // arrangement that can fail. A load into a fresh instance renumbers to the very values it
    // saved, and looks perfect.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let panel = g.doc()["arrangement"].as_object().unwrap().iter()
        .find(|(_, e)| e["kind"] == "panel").map(|(id, _)| id.clone()).expect("the default panel");
    g.call("page_set_panel", j!({ "page": "Layout", "panel": panel, "type": "viewer",
                                 "state": { "node": goofi_tests::hex(osc), "slot": "out" } }));

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.call("save", j!({ "path": path.to_string_lossy() }));

    // Make the instance a USED one: three more nodes, whose uids the old load handed the saved
    // patch on the way back in.
    for _ in 0..3 {
        g.add("Buffer");
    }
    g.call("load", j!({ "path": path.to_string_lossy() }));

    assert_eq!(g.nodes(), vec![goofi_tests::hex(osc)], "the patch came back with the uid it was saved with");
    let state = g.doc()["arrangement"][&panel]["state"].as_str().unwrap_or("").to_string();
    assert!(state.contains(&goofi_tests::hex(osc)), "…so the panel still names a node that exists: {state}");
}

#[test]
fn a_load_restores_the_graph_and_the_workspace_together() {
    let g = Goofi::new();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.add("Oscillator");
    std::fs::write(g.state.mount().join("agent.md"), b"notes").unwrap();
    // The orientation is workspace content like any other, so what the patch carries is whatever its
    // own workspace holds: an `AGENTS.md` the agent rewrote, and NO `CLAUDE.md` at all — deleted, as
    // a patch saved before goofi ever seeded one would have none.
    let learned = b"goofi-pipe: this patch's EEG source is on channel 3.\n";
    std::fs::write(g.state.mount().join("AGENTS.md"), learned).unwrap();
    std::fs::remove_file(g.state.mount().join("CLAUDE.md")).unwrap();
    // The ignore list is the patch's on the same terms, and it is the ONE file the pack consults as
    // it packs: narrowed here to prove it rides its own archive rather than filtering itself out.
    let ignores = "*.wav\n";
    std::fs::write(g.state.mount().join(goofi_engine::archive::IGNORE_FILE), ignores).unwrap();
    g.call("save", j!({ "path": path.to_string_lossy() }));

    // Diverge on BOTH planes — a node the patch does not have, and a workspace that no longer
    // matches the one it packed — then load it back off disk.
    let stale = g.state.mount();
    g.add("Buffer");
    std::fs::remove_file(stale.join("agent.md")).unwrap();
    std::fs::write(stale.join("scratch.txt"), b"written since the save").unwrap();
    g.call("load", j!({ "path": path.to_string_lossy() }));

    assert_eq!(g.nodes().len(), 1, "the on-disk patch replaced the diverged graph");
    assert_eq!(save_path(&g).as_deref(), Some(spelled(&path).as_str()), "and says where it came from");

    let mount = g.state.mount();
    assert_ne!(mount, stale, "a load mounts fresh");
    assert_eq!(std::fs::read(mount.join("agent.md")).unwrap(), b"notes");
    assert!(!mount.join("scratch.txt").exists(), "the diverged workspace did not follow");
    // A load seeds NOTHING: goofi initialises a workspace it CREATED, never one it unpacked
    // someone's patch into. These two are what stops the seed call being put back on this path.
    assert_eq!(std::fs::read(mount.join("AGENTS.md")).unwrap(), learned,
               "the load seeded over the orientation the patch was saved with");
    assert!(!mount.join("CLAUDE.md").exists(), "the load invented a file the archive never held");
    assert_eq!(std::fs::read_to_string(mount.join(goofi_engine::archive::IGNORE_FILE)).unwrap(), ignores);
    assert!(!stale.exists(), "the mount the load replaced is released, not leaked");
}

#[test]
fn a_refused_load_leaves_the_graph_and_the_workspace_untouched() {
    let g = Goofi::new();
    g.add("Oscillator");
    let mount = g.state.mount();
    std::fs::write(mount.join("agent.md"), b"notes").unwrap();

    // The three ways a load is refused, in the order the arm reaches them: no such file, a file that
    // is not an archive, and — the one that pins commit-AFTER-parse — a perfectly good archive,
    // workspace and all, whose manifest the engine will not accept.
    let dir = tempfile::tempdir().unwrap();
    let junk = dir.path().join("junk.gfi");
    std::fs::write(&junk, "this: is: not: a patch").unwrap();
    let packed = dir.path().join("ws");
    std::fs::create_dir(&packed).unwrap();
    std::fs::write(packed.join("intruder.txt"), b"from the refused archive").unwrap();
    let bad = dir.path().join("bad.gfi");
    goofi_engine::archive::write_gfi(&bad, "this: is: not: a patch", &packed).unwrap();

    for target in [dir.path().join("absent.gfi"), junk, bad] {
        g.refuse("load", j!({ "path": target.to_string_lossy() }));
    }

    assert!(yaml(&g).contains("Oscillator"), "the pre-load graph survives every failure");
    assert_eq!(g.state.mount(), mount, "the live mount is still the open patch's");
    assert_eq!(std::fs::read(mount.join("agent.md")).unwrap(), b"notes");
    assert!(!mount.join("intruder.txt").exists(), "nothing from a refused archive landed in it");
}

#[test]
fn a_new_patch_is_empty_clean_unnamed_and_has_nothing_to_undo() {
    // New is reached from a patch that had grown all three things a patch can have — a graph, an
    // arrangement and a file on disk — and inherits none of them. Each half fails separately: the
    // arrangement is not graph content, and the dispatch tail dirties any op it does not recognise,
    // so a New patch would be born asking to be saved over the last real one.
    let g = Goofi::new();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.add("Oscillator");
    g.call("session_add_page", j!({ "name": "Second" }));
    g.call("save", j!({ "path": path.to_string_lossy() }));

    g.call("new", j!({}));
    assert!(g.nodes().is_empty(), "an open tab's canvas is emptied too");
    assert_eq!(dirty(&g), false, "a New patch has nothing to save");
    assert_eq!(save_path(&g), None, "…and no file behind it");
    let pages = g.call("inspect_layout", j!({}))["text"].as_str().unwrap().matches("page `").count();
    assert_eq!(pages, 1, "…and none of the previous patch's panels");
    assert!(!yaml(&g).contains("Oscillator"), "…and none of its nodes");

    // The command history goes with the patch too: an entry belonging to the graph that just went
    // away has nothing left to flip against, and its redo would put the node back.
    let undo = g.call("undo", j!({}));
    assert_eq!(undo["changed"], false, "nothing to undo across a New");
    assert_eq!(undo["can_undo"], false, "…and none offered");
}

#[test]
fn a_new_patch_mounts_an_empty_workspace_of_its_own() {
    // The workspace is half of what a patch is. `open_workspace` is how a client learns where that
    // is at all: the mount is a per-run temp directory under a random name.
    let g = Goofi::new();
    let before = std::path::PathBuf::from(
        g.call("open_workspace", j!({}))["path"].as_str().unwrap());
    assert_eq!(before, g.state.mount(), "open_workspace names the LIVE mount");
    std::fs::write(before.join("agent.md"), b"notes").unwrap();

    g.call("new", j!({}));

    let after = std::path::PathBuf::from(
        g.call("open_workspace", j!({}))["path"].as_str().unwrap());
    assert_eq!(after, g.state.mount(), "…and follows it when New swaps it");
    assert_ne!(after, before, "New mounts fresh");
    assert!(!after.join("agent.md").exists(), "so nothing the previous patch wrote survives");
    assert!(!before.exists(), "and the mount it replaced is released, not leaked");
    // Empty of the previous patch, but NOT of the orientation: `new` mints the workspace, so `new`
    // is exactly the case that initialises it. It shares its dispatch arm with `load`, which must
    // not be seeded, and the two are one line apart — this is the half that says which is which.
    let agents = std::fs::read_to_string(after.join("AGENTS.md"))
        .expect("a New patch's workspace is seeded with the orientation");
    assert!(agents.contains("goofi-pipe is a live"), "…the real one: {agents}");
    assert_eq!(std::fs::read_to_string(after.join("CLAUDE.md")).unwrap(), "@AGENTS.md\n");
    assert_eq!(dirty(&g), false, "and asking where it is did not dirty anything");
}

#[test]
fn a_save_packs_the_live_mount_refuses_to_pack_into_it_and_never_truncates_a_good_archive() {
    // `save_archive` is the save primitive both the op and the `/patch.gfi` route go through, and
    // two of its three properties can only be staged by calling it: a pack that fails AFTER the
    // zip's first entry, and a target inside the very tree being packed.
    let tmp = tempfile::tempdir().unwrap();
    let mount = tmp.path().join("goofi-0123").join("workspace");
    std::fs::create_dir_all(&mount).unwrap();
    std::fs::write(mount.join("agent.md"), b"notes").unwrap();

    let target = tmp.path().join("patch.gfi");
    goofi_bridge::save_archive(&target, "version: 7\n", &mount).unwrap();
    let dest = tmp.path().join("unpacked");
    assert_eq!(goofi_engine::archive::read_gfi(&target, &dest).unwrap(), "version: 7\n");
    assert_eq!(std::fs::read(dest.join("agent.md")).unwrap(), b"notes", "the LIVE mount is packed");

    // A mount that is not on disk fails the workspace walk, which happens after the first zip entry
    // is written — exactly the window in which packing straight onto the target would truncate a
    // good `.gfi` into a half-written one. It sits OUTSIDE the target's directory, or the mount
    // refusal below answers first and the pack never runs.
    let good = tmp.path().join("previous.gfi");
    std::fs::write(&good, b"the previous save").unwrap();
    let gone = tmp.path().join("mnt").join("gone").join("workspace");
    let err = goofi_bridge::save_archive(&good, "version: 7\n", &gone).unwrap_err();
    assert!(err.contains("save failed"), "the refusal names the operation: {err}");
    assert_eq!(std::fs::read(&good).unwrap(), b"the previous save");
    assert!(!tmp.path().join("previous.gfi.tmp").exists(), "the half-written sibling is cleaned up");

    // A target inside the mount would pack the archive into itself.
    for inside in [mount.join("patch.gfi"), mount.parent().unwrap().join("patch.gfi")] {
        let err = goofi_bridge::save_archive(&inside, "version: 7\n", &mount).unwrap_err();
        assert!(err.contains("temporary workspace"), "the refusal says why: {err}");
        assert!(!inside.exists(), "a refused save writes nothing");
    }
}

#[test]
fn an_external_workspace_edit_makes_the_patch_differ_from_its_archive() {
    // A workspace file edited OUTSIDE goofi — by the agent the harness runs in it, or by the user's
    // own editor — makes the patch differ from its `.gfi` exactly as a moved node does. There is no
    // watcher: the manager compares the mount against the fingerprint it took when it last packed.
    let g = Goofi::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("patch.gfi");
    std::fs::write(g.state.mount().join("agent.md"), b"notes").unwrap();
    g.call("save", j!({ "path": target.to_string_lossy() }));
    assert!(!dirty(&g), "the patch was just written to disk, workspace and all");

    // A file the archive does not have — what an agent writing into the workspace does.
    std::fs::write(g.state.mount().join("scratch.txt"), b"written since the save").unwrap();
    assert!(dirty(&g), "a workspace file the archive lacks is an unsaved change");

    // A file that WAS packed but whose content has since changed: the fingerprint has to carry more
    // than the set of names, or the commonest edit of all goes unnoticed.
    g.call("save", j!({ "path": target.to_string_lossy() }));
    assert!(!dirty(&g), "saving again re-baselines the workspace");
    std::fs::write(g.state.mount().join("agent.md"), b"notes, and then some more notes").unwrap();
    assert!(dirty(&g), "an edit to a packed file is an unsaved change too");

    // …including one that leaves the LENGTH alone — an editor rewriting a line in place. That is
    // the half a `(name, len)` fingerprint would silently drop.
    g.call("save", j!({ "path": target.to_string_lossy() }));
    std::fs::write(g.state.mount().join("agent.md"), b"NOTES, AND THEN SOME MORE NOTES").unwrap();
    assert!(dirty(&g), "a same-length in-place edit is an unsaved change");

    // And a save that FAILED re-baselines nothing: it packed no file, so those edits still live
    // only in the mount — a per-run temp tree a graceful exit deletes.
    g.refuse("save", j!({ "path": tmp.path().join("no-such-dir").join("patch.gfi").to_string_lossy() }));
    assert!(dirty(&g), "a save that wrote nothing cannot call the workspace packed");
}

#[test]
fn a_freshly_loaded_patch_is_clean_though_every_file_in_it_was_just_written() {
    // What the load arm's re-baseline buys: a freshly opened patch has the dot off and the unload
    // guard down, on a graph and a workspace that are byte-for-byte the file's.
    let g = Goofi::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("patch.gfi");
    std::fs::write(g.state.mount().join("agent.md"), b"notes").unwrap();
    g.call("save", j!({ "path": target.to_string_lossy() }));

    // Loaded into a SECOND manager, which is the real case — the goofi that opens a patch is rarely
    // the one that wrote it, and it has no baseline of its own to fall back on.
    let opened = Goofi::new();
    opened.call("load", j!({ "path": target.to_string_lossy() }));
    assert_eq!(std::fs::read(opened.state.mount().join("agent.md")).unwrap(), b"notes");
    assert!(!dirty(&opened), "a patch is not unsaved the moment it finishes loading");
}
