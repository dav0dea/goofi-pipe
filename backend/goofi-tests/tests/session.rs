//! A whole authoring session: build a patch, save it, find it again, and open it somewhere else.
//!
//! One scenario across every layer a user's work passes through — catalog, graph, params,
//! expressions, sub-patch scopes, boundaries, the panel arrangement, the archive, and the uid
//! identity that has to survive all of it. A per-op test can pass on every one of those and still
//! lose the patch on the way back in, which is what this exists to catch.
//!
//! A `.gfi` is a ZIP: `patch.yaml` beside the workspace tree it was saved with. The manager owns
//! where it lives, and both halves of "where" fail separately — the live broadcast converges the
//! tabs already open, `get_patch` converges the ones that ask later. A load is all-or-nothing: it
//! extracts into a FRESH mount, parses, and only then swaps. Graph and workspace, or neither.

use goofi_tests::{hex, j, Goofi};
use serde_json::Value;

fn panel(g: &Goofi) -> String {
    goofi_tests::panel_ids(&g.doc()["arrangement"]).first().cloned().expect("the default panel")
}

#[test]
fn a_patch_is_built_saved_and_opened_somewhere_else_unchanged() {
    let g = Goofi::new();

    // The palette a user picks from, before anything exists.
    let types = g.call("list_nodes", j!({}))["types"].as_array().cloned().unwrap();
    for want in ["Oscillator", "Buffer"] {
        assert!(types.iter().any(|t| t["type"] == want), "{want} is in the palette");
    }
    assert!(!types.iter().any(|t| t["type"] == "_TestEcho"), "test nodes are not");

    // Build: a source, a window, and a sink downstream of it.
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let sink = g.add("Buffer");
    g.call("rename_node", j!({ "node": hex(osc), "name": "carrier" }));
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 128 }));
    g.link(osc, "out", buf, "data");
    g.link(buf, "out", sink, "data");

    // A param driven by another node, and one driven by a global.
    g.call("add_global", j!({ "name": "gain", "value": 2.0, "type": "float" }));
    g.call("set_expression", j!({ "node": hex(sink), "group": "buffer", "name": "size",
                                 "expression": "globals.gain * 64", "enabled": true }));

    // Fold the middle of the chain into a sub-patch. The cut wires become boundary ports.
    let scope = g.call("group_nodes", j!({ "members": [hex(buf)], "pos": [40.0, 10.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let stubs = g.doc()["instances"][&scope]["stubs"].as_object().cloned().unwrap_or_default();
    assert_eq!(stubs.len(), 2, "both cuts are exposed: {stubs:?}");

    // Bind a panel to a node, so the arrangement carries a reference into the graph.
    g.call("set_panel", j!({ "panel": panel(&g), "type": "viewer",
                                 "state": { "node": hex(osc), "slot": "out" } }));

    let before = g.doc();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    std::fs::write(g.state.mount().join("notes.md"), b"the EEG source is on channel 3").unwrap();
    g.call("save", j!({ "path": path.to_string_lossy() }));
    assert_eq!(g.call("get_patch", j!({}))["dirty"], false, "a saved patch is clean");

    // Open it in a DIFFERENT instance that has already held other nodes — the only arrangement
    // that can fail, since a load into a fresh one renumbers to the very uids it saved.
    let other = Goofi::new();
    for _ in 0..5 {
        other.add("Oscillator");
    }
    other.call("load", j!({ "path": path.to_string_lossy() }));

    let after = other.doc();
    assert_eq!(after["nodes"], before["nodes"], "every node came back as it was, uid for uid");
    assert_eq!(after["links"], before["links"], "and so did every wire");
    assert_eq!(after["instances"], before["instances"], "and the scope with its boundary ports");
    assert_eq!(after["globals"], before["globals"]);
    assert_eq!(after["arrangement"], before["arrangement"],
               "…so the panel still names a node that exists");
    assert_eq!(std::fs::read(other.state.mount().join("notes.md")).unwrap(),
               b"the EEG source is on channel 3", "the workspace travelled with the patch");
    assert_eq!(other.call("get_patch", j!({}))["dirty"], false,
               "a patch is not unsaved the moment it finishes loading");
}

#[test]
fn a_refused_load_leaves_the_open_patch_exactly_as_it_was() {
    // The other half of the round trip: a load is all-or-nothing, graph and workspace together.
    let g = Goofi::new();
    g.add("Oscillator");
    std::fs::write(g.state.mount().join("notes.md"), b"work in progress").unwrap();
    let before = g.doc();
    let mount = g.state.mount();

    let dir = tempfile::tempdir().unwrap();
    // The three refusals, in the order the arm reaches them — and the third is the one that pins
    // commit-AFTER-parse: a perfectly good archive whose manifest the engine will not accept.
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

    assert_eq!(g.doc(), before, "the open patch is untouched");
    assert_eq!(g.state.mount(), mount, "on the mount it was already using");
    assert_eq!(std::fs::read(mount.join("notes.md")).unwrap(), b"work in progress");
    assert!(!mount.join("intruder.txt").exists(), "and nothing from the refused archive landed");
}

#[test]
fn a_new_patch_inherits_nothing_from_the_one_before_it() {
    // New is reached from a patch that had grown all three things a patch can have — a graph, an
    // arrangement and a file on disk. Each half fails separately, and a New patch born unsaved
    // would offer to be written over the last real one.
    let g = Goofi::new();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.add("Oscillator");
    g.call("add_tab", j!({ "name": "Second" }));
    g.call("save", j!({ "path": path.to_string_lossy() }));
    let old_mount = g.state.mount();
    std::fs::write(old_mount.join("notes.md"), b"the previous patch's").unwrap();

    g.call("new", j!({}));

    assert!(g.nodes().is_empty(), "no nodes");
    assert_eq!(g.call("inspect_layout", j!({}))["text"].as_str().unwrap().matches("tab `").count(), 1,
               "no tabs of the previous patch");
    assert_eq!(g.call("get_patch", j!({}))["save_path"], Value::Null, "no file behind it");
    assert_eq!(g.call("get_patch", j!({}))["dirty"], false, "and nothing to save");
    assert_eq!(g.call("undo", j!({}))["changed"], false, "the history went with the patch");

    let mount = g.state.mount();
    assert_ne!(mount, old_mount, "a fresh workspace");
    assert!(!old_mount.exists(), "and the one it replaced is released, not leaked");
    // Empty of the previous patch, but NOT of the orientation: `new` MINTS the workspace, so `new`
    // is exactly the case that seeds it — while `load`, one line away, must not.
    assert!(!mount.join("notes.md").exists());
    assert!(std::fs::read_to_string(mount.join("AGENTS.md")).unwrap().contains("goofi-pipe is a live"));
    assert_eq!(std::fs::read_to_string(mount.join("CLAUDE.md")).unwrap(), "@AGENTS.md\n");
}

#[test]
fn a_patch_whose_arrangement_cannot_be_rendered_still_opens() {
    // The graph is the value and the arrangement is chrome: a layout the flat model admits but
    // cannot render must never make a patch unopenable — and the fallback must be stated.
    let g = Goofi::new();
    g.add("Oscillator");
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    // A DUPLICATE id — the one corruption the tree admits and the flat map could not, since a map
    // key cannot repeat. The shapes the flat model used to admit (an orphan, a cycle, a tab with two
    // roots) have no spelling here at all.
    let broken = yaml.replace("id: panel-2", "id: tab-1");
    assert_ne!(broken, yaml, "the fixture actually corrupted something");

    let r = g.call("load_text", j!({ "content": broken }));
    assert_eq!(r["ok"], true, "the patch still opens: {r}");
    assert!(r["layout_warning"].as_str().is_some_and(|w| w.contains("appears twice")),
            "…and says why the arrangement was dropped: {r}");
    assert_eq!(g.nodes().len(), 1, "with the graph intact");
}

fn save_path(g: &Goofi) -> Option<String> {
    g.call("get_patch", j!({}))["save_path"].as_str().map(str::to_string)
}

fn dirty(g: &Goofi) -> bool {
    g.call("get_patch", j!({}))["dirty"] == true
}

/// A path as goofi spells it back: `/` on every platform. Comparing against the platform's own
/// spelling would pass on unix and pin the Windows bug — this once asserted `C:\Users\…` against
/// the `C:/Users/…` the wire actually carries.
fn spelled(p: &std::path::Path) -> String {
    goofi_core::path::to_slash(p)
}

#[test]
fn only_a_patch_with_a_file_behind_it_keeps_a_name_and_every_tab_is_told_which() {
    // The stored path always names a file this patch was really written to or read from, because a
    // plain Save overwrites it silently, from any tab, with no second prompt. The manager owns it
    // — it used to own none, and only the `load` arm announced one, so a save named the patch in
    // the tab that performed it and nowhere else.
    let g = Goofi::new();
    assert_eq!(save_path(&g), None, "an unsaved patch has no home yet");
    g.add("Oscillator");

    // "Save in browser" is gone: a save's ONLY job is writing to a backend path, so a save with no
    // path is a malformed request rather than a second mode.
    let why = g.refuse("save", j!({}));
    assert!(why.contains("save") && why.contains("path"), "{why}");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    let mut ev = g.events();
    g.call("save", j!({ "path": path.to_string_lossy() }));
    assert_eq!(ev.next("save_path_changed")["save_path"].as_str(), Some(spelled(&path).as_str()));
    assert_eq!(save_path(&g).as_deref(), Some(spelled(&path).as_str()));
    // Readable only through `read_gfi` — a bare-YAML write would leave it "not a zip archive".
    let dest = dir.path().join("unpacked");
    let manifest = goofi_engine::archive::read_gfi(&path, &dest).unwrap();
    assert!(manifest.contains("Oscillator"), "the manifest is the serialized patch: {manifest}");
    assert!(dest.is_dir(), "the workspace tree rides along, empty or not");

    // A save that FAILS leaves the previous home standing: the patch has never been written to the
    // file it was refused, so naming it would aim the next silent overwrite at it.
    let nowhere = dir.path().join("no-such-dir").join("patch.gfi");
    g.refuse("save", j!({ "path": nowhere.to_string_lossy() }));
    assert_eq!(save_path(&g).as_deref(), Some(spelled(&path).as_str()), "the old home stands");

    // An upload carries no file, so the patch it installs is UNNAMED. Inheriting the previous path
    // here is the silent-overwrite hazard in its purest form: a different patch entirely, saved
    // over a file it never came from.
    let content = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("load_text", j!({ "content": content }));
    assert_eq!(save_path(&g), None, "an uploaded patch has no home");
}

#[test]
fn a_save_packs_the_live_mount_refuses_to_pack_into_it_and_never_truncates_a_good_archive() {
    // `save_archive` is the primitive both the op and the `/patch.gfi` route go through, and two of
    // its three properties can only be staged by calling it: a pack that fails AFTER the zip's
    // first entry, and a target inside the very tree being packed.
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
fn the_workspace_counts_as_unsaved_work_and_a_fresh_load_is_clean() {
    // A workspace file edited OUTSIDE goofi — by the agent the harness runs in it, or by the user's
    // own editor — makes the patch differ from its `.gfi` exactly as a moved node does. There is no
    // watcher: the manager compares the mount against the fingerprint it took when it last packed.
    let g = Goofi::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("patch.gfi");
    std::fs::write(g.state.mount().join("agent.md"), b"notes").unwrap();
    g.call("save", j!({ "path": target.to_string_lossy() }));
    assert!(!dirty(&g), "the patch was just written to disk, workspace and all");

    std::fs::write(g.state.mount().join("scratch.txt"), b"written since the save").unwrap();
    assert!(dirty(&g), "a workspace file the archive lacks is an unsaved change");

    // A file that WAS packed but whose content has since changed: the fingerprint has to carry more
    // than the set of names, or the commonest edit of all goes unnoticed — INCLUDING one that
    // leaves the length alone, which is what an editor rewriting a line in place does.
    g.call("save", j!({ "path": target.to_string_lossy() }));
    assert!(!dirty(&g), "saving again re-baselines the workspace");
    std::fs::write(g.state.mount().join("agent.md"), b"NOTES").unwrap();
    assert!(dirty(&g), "an edit to a packed file is an unsaved change too");
    g.call("save", j!({ "path": target.to_string_lossy() }));
    std::fs::write(g.state.mount().join("agent.md"), b"note!").unwrap();
    assert!(dirty(&g), "a same-length in-place edit is an unsaved change");

    // A save that FAILED re-baselines nothing: it packed no file, so those edits still live only in
    // the mount — a per-run temp tree a graceful exit deletes.
    g.refuse("save", j!({ "path": tmp.path().join("no-such-dir").join("patch.gfi").to_string_lossy() }));
    assert!(dirty(&g), "a save that wrote nothing cannot call the workspace packed");

    // Loaded into a SECOND manager, which is the real case — the goofi that opens a patch is rarely
    // the one that wrote it, and it has no baseline of its own to fall back on.
    let opened = Goofi::new();
    opened.call("load", j!({ "path": target.to_string_lossy() }));
    assert_eq!(std::fs::read(opened.state.mount().join("agent.md")).unwrap(), b"NOTES");
    assert!(!dirty(&opened), "a patch is not unsaved the moment it finishes loading");
}

#[test]
fn the_file_browser_answers_a_path_the_way_save_and_load_take_it() {
    // The browser behind Save and Load. Expectations are built from `goofi_core::path`'s own
    // primitives rather than from the browser's answer: asserting with the function under test on
    // both sides would accept, say, a normalizer that reversed the segments it re-attaches.
    use goofi_core::path::{canonical, to_slash};
    let g = Goofi::new();
    let list = |p: Option<&str>| match p {
        Some(p) => g.call("list_dir", j!({ "path": p })),
        None => g.call("list_dir", j!({})),
    };
    let names = |l: &Value| -> Vec<String> {
        l["entries"].as_array().unwrap().iter()
            .map(|e| e["name"].as_str().unwrap().to_string()).collect()
    };

    let tmp = tempfile::tempdir().unwrap();
    for f in ["Beta.txt", "alpha.txt", ".hidden", "patch.gfi"] {
        std::fs::write(tmp.path().join(f), b"x").unwrap();
    }
    for d in ["Zeta", "apples"] {
        std::fs::create_dir_all(tmp.path().join(d)).unwrap();
    }
    // A name this platform accepts but UTF-8 cannot express. Lossy-encoding one would emit an entry
    // that cannot be opened, and whose `path` key can collide with another undecodable one.
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let bad = std::ffi::OsString::from_vec(b"bad\xff".to_vec()); // starts no valid sequence
        std::fs::write(tmp.path().join(bad), b"x").unwrap();
    }

    let here = to_slash(&canonical(tmp.path()).unwrap());
    let listing = list(Some(&tmp.path().to_string_lossy()));
    assert_eq!(names(&listing), ["apples", "Zeta", ".hidden", "alpha.txt", "Beta.txt", "patch.gfi"],
               "directories first, then case-insensitively by name, and nothing undecodable");

    let by_name = |n: &str| listing["entries"].as_array().unwrap().iter()
        .find(|e| e["name"] == n).unwrap_or_else(|| panic!("{n} is listed")).clone();
    let entry = by_name("alpha.txt");
    for key in ["name", "path", "kind", "is_gfi", "hidden"] {
        assert!(entry.get(key).is_some(), "an entry is missing `{key}`");
    }
    assert_eq!(entry["kind"], "file");
    assert_eq!(entry["path"], format!("{here}/alpha.txt"));
    // The browser renders neither a size nor a date column, so the row carries neither. Re-add them
    // together with the column that shows them.
    for key in ["size", "mtime"] {
        assert!(entry.get(key).is_none(), "an entry carries an unrendered `{key}`");
    }
    // Hidden entries are EMITTED, not filtered — the browser owns that toggle.
    assert_eq!((&by_name(".hidden")["hidden"], &by_name("patch.gfi")["hidden"]), (&j!(true), &j!(false)));
    assert_eq!((&by_name("patch.gfi")["is_gfi"], &by_name("alpha.txt")["is_gfi"]),
               (&j!(true), &j!(false)));

    // A FILE path lists its parent, so a path typed into Save-As navigates.
    assert_eq!(list(Some(&tmp.path().join("patch.gfi").to_string_lossy()))["path"], here);

    // The frontend omits `path` on the first open; a cleared input sends "".
    let home = std::env::var("HOME").expect("HOME is set in the test environment");
    let home = to_slash(&canonical(std::path::Path::new(&home)).unwrap_or_else(|_| home.into()));
    assert_eq!(list(None)["path"].as_str().unwrap(), home);
    assert_eq!(list(Some("")), list(None));
    assert_eq!(list(Some("~"))["path"].as_str().unwrap(), home, "a leading tilde expands");
    // The sidebar marks a root active by raw string equality against `path`, so a root whose
    // spelling differs from what `list_dir` echoes back never highlights.
    let roots = list(None)["roots"].as_array().unwrap().clone();
    assert!(roots.iter().any(|r| r["label"] == "Home"), "expected a Home root");
    for root in &roots {
        let path = root["path"].as_str().unwrap();
        assert_eq!(list(Some(path))["path"].as_str().unwrap(), path, "root {path} is not stable");
    }
    // The topmost ancestor of a real path — `/` is not a root on Windows, where it absolutizes
    // against the current drive.
    let top = to_slash(canonical(&std::env::temp_dir()).unwrap().ancestors().last().unwrap());
    assert!(list(Some(&top))["parent"].is_null(), "root has no parent to climb to");

    // Navigation must NOT error, or the browser keeps showing the previous directory. The
    // Save-As case: the longest existing ancestor is canonicalized and the tail re-attached — and
    // `..` has no `file_name()`, so an ancestor walk would silently drop it and land elsewhere.
    let missing = format!("{top}definitely/not/a/directory");
    let listing = list(Some(&missing));
    assert_eq!((names(&listing), listing["path"].as_str().unwrap()), (Vec::new(), missing.as_str()));
    assert_eq!(list(Some(&tmp.path().join("new/deeper").to_string_lossy()))["path"],
               format!("{here}/new/deeper"));
    assert_eq!(list(Some(&tmp.path().join("missing/../also-missing").to_string_lossy()))["path"],
               format!("{here}/also-missing"));

    // Save and load share this resolution, so a path a user can navigate to is one they can write
    // to. Proven through a REFUSAL that names the expanded path — actually saving into `$HOME` is
    // not something a test may do.
    let why = g.refuse("load", j!({ "path": "~/definitely-not-a-patch-goofi-wrote.gfi" }));
    assert!(why.contains(&home), "the refusal names the expanded path, not the tilde: {why}");
}
