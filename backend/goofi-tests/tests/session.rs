//! A whole authoring session: build a patch, save it, find it again, and open it somewhere else.

use goofi_tests::{hex, j, Goofi};
use serde_json::Value;

fn panel(g: &Goofi) -> String {
    goofi_tests::panel_ids(&g.doc()["arrangement"]).first().cloned().expect("the default panel")
}

#[test]
fn a_patch_is_built_saved_and_opened_somewhere_else_unchanged() {
    let g = Goofi::new();

    let types = g.call("library list", j!({}))["types"].as_array().cloned().unwrap();
    for want in ["signal:Oscillator", "signal:Buffer"] {
        assert!(types.iter().any(|t| t["type"] == want), "{want} is in the palette");
    }
    assert!(!types.iter().any(|t| t["type"] == "signal:_TestEcho"), "test nodes are not");

    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let sink = g.add("Buffer");
    g.call("node edit", j!({ "node": hex(osc), "name": "carrier" }));
    g.set_param(buf, "buffer", "size", 128);
    g.link(osc, "out", buf, "data");
    g.link(buf, "out", sink, "data");

    g.call("global add", j!({ "name": "gain", "value": 2.0, "type": "float" }));
    g.call("node param edit", j!({ "node": hex(sink), "param": "buffer/size",
                                   "expression": "globals.gain * 64" }));
    // …and a reference over it: the archive carries the whole record, the expression retained.
    let level = g.add("_TestScalar");
    g.call("node edit", j!({ "node": hex(level), "name": "level" }));
    g.call("node param edit", j!({ "node": hex(sink), "param": "buffer/size", "reference": "level.out" }));

    let scope = g.call("nodes group", j!({ "nodes": [hex(buf)], "pos": [40.0, 10.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    assert_eq!(g.ports(&scope).len(), 2, "both cuts are exposed: {:?}", g.ports(&scope));

    // Nest it, and leave one port with nothing behind it. Between them these are every shape the
    // archive's one entity kind has to carry: a facade inside a facade, a port wired to another
    // scope's port, and a port whose inner wire is simply absent.
    let outer = g.call("nodes group", j!({ "nodes": [&scope], "pos": [80.0, 10.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let spare = g.call("node add", j!({ "type": "OutTable", "inst_id": outer, "pos": [5.0, 6.0] }))
        ["uid"].as_str().unwrap().to_string();
    g.call("node edit", j!({ "node": spare, "name": "spare" }));

    // The file says it in ONE vocabulary: a facade and a port are node records like any other, and
    // a port's inner wire is a link like any other.
    let yaml = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    let saved: serde_json::Value = serde_yaml_ng::from_str(&yaml).unwrap();
    assert_eq!(saved["goofi"], env!("CARGO_PKG_VERSION"), "the manifest names its writer");
    assert!(saved["root"].get("scopes").is_none(), "no block of its own for the structure");
    let recs = saved["root"]["nodes"].as_object().unwrap();
    assert_eq!(recs[&outer]["type"], "SubPatch", "the facade is a node record: {:?}", recs[&outer]);
    assert_eq!(recs[&spare]["type"], "OutTable", "…and so is the port");
    assert_eq!(recs[&scope]["scope"], outer, "membership rides the record it belongs to");
    let source = &recs[&hex(sink)]["sources"][0];
    assert_eq!((&source["mode"], &source["expression"], &source["reference"]),
               (&j!("reference"), &j!("globals.gain * 64"), &j!("level.out")), "{source}");

    g.call("layout panel edit", j!({ "panel": panel(&g), "type": "viewer",
                                        "state": { "node": hex(osc), "slot": "out" } }));

    let before = g.doc();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    std::fs::write(g.state.mount().join("notes.md"), b"the EEG source is on channel 3").unwrap();
    g.call("session save", j!({ "path": path.to_string_lossy() }));
    assert_eq!(g.call("session status", j!({}))["dirty"], false, "a saved patch is clean");

    // Opened in an instance that has already held other nodes — a fresh one renumbers to the saved uids.
    let other = Goofi::new();
    for _ in 0..5 {
        other.add("Oscillator");
    }
    other.call("session load", j!({ "path": path.to_string_lossy() }));

    let after = other.doc();
    assert_eq!(after["nodes"], before["nodes"],
               "every node came back as it was, uid for uid — facades and ports among them");
    assert_eq!(after["links"], before["links"], "and so did every wire, inner ones included");
    assert_eq!(after["globals"], before["globals"]);
    assert_eq!(after["arrangement"], before["arrangement"],
               "…so the panel still names a node that exists");
    assert_eq!(std::fs::read(other.state.mount().join("notes.md")).unwrap(),
               b"the EEG source is on channel 3", "the workspace travelled with the patch");
    assert_eq!(other.call("session status", j!({}))["dirty"], false,
               "a patch is not unsaved the moment it finishes loading");

    // …and reopened over ITSELF, in the session that has been running it all along.
    let mut ev = g.events();
    let late = g.add("Oscillator");
    g.ready(late);
    // A uid the status worker has never reported on, so its `ready` is the tick that also memoized
    // every node already running.
    loop {
        let told = ev.next("node_stage");
        if told["node"] == hex(late) && told["stage"] == "ready" {
            break;
        }
    }
    g.call("session save", j!({ "path": path.to_string_lossy() }));
    g.call("session load", j!({ "path": path.to_string_lossy() }));

    // A locked global is the machine's: the manifest never carries it, and the load re-derives it.
    let manifest = g.call("session manifest", j!({}));
    assert!(!manifest["yaml"].as_str().unwrap().contains("goofi_home"),
            "a machine path in a patch file travels to the wrong machine");
    let held = g.call("global list", j!({}))["globals"].as_array().unwrap().iter()
        .find(|e| e["name"] == "goofi_home").cloned().unwrap();
    assert_eq!(held["value"], j!(goofi_core::path::to_slash(&goofi_core::home::dir())));

    let snap = ev.next("graph_replaced");
    assert_eq!(snap["runtime"][hex(late)]["stage"], "creating",
               "the load rebuilt it at the uid it was saved with, and the snapshot caught it starting");
    g.ready(late);
    // The stage stream is a delta over that snapshot, so the node reaching `ready` again HAS to be
    // said — the uid it came back at reported the same thing in its previous life.
    loop {
        let told = ev.next("node_stage");
        if told["node"] == hex(late) && told["stage"] == "ready" {
            break;
        }
    }
}

#[test]
fn a_refused_load_leaves_the_open_patch_exactly_as_it_was() {
    let g = Goofi::new();
    g.add("Oscillator");
    std::fs::write(g.state.mount().join("notes.md"), b"work in progress").unwrap();
    let before = g.doc();
    let mount = g.state.mount();

    let dir = tempfile::tempdir().unwrap();
    // In the order the arm reaches them; the third pins commit-AFTER-parse.
    let junk = dir.path().join("junk.gfi");
    std::fs::write(&junk, "this: is: not: a patch").unwrap();
    let packed = dir.path().join("ws");
    std::fs::create_dir(&packed).unwrap();
    std::fs::write(packed.join("intruder.txt"), b"from the refused archive").unwrap();
    let bad = dir.path().join("bad.gfi");
    goofi_graph::archive::write_gfi(&bad, "this: is: not: a patch", &packed).unwrap();
    for target in [dir.path().join("absent.gfi"), junk, bad] {
        g.refuse("session load", j!({ "path": target.to_string_lossy() }));
    }
    // Valid YAML from a FUTURE goofi: the version gate refuses, and the refusal names the writer.
    let future = dir.path().join("future.gfi");
    goofi_graph::archive::write_gfi(&future, "version: 99\ngoofi: \"9.9.9\"\nroot: {}", &packed).unwrap();
    let refusal = g.refuse("session load", j!({ "path": future.to_string_lossy() }));
    assert!(refusal.contains("written by goofi 9.9.9"), "the writer is named: {refusal}");

    assert_eq!(g.doc(), before, "the open patch is untouched");
    assert_eq!(g.state.mount(), mount, "on the mount it was already using");
    assert_eq!(std::fs::read(mount.join("notes.md")).unwrap(), b"work in progress");
    assert!(!mount.join("intruder.txt").exists(), "and nothing from the refused archive landed");
}

#[test]
fn a_new_patch_inherits_nothing_from_the_one_before_it() {
    // New is reached from a patch with a graph, an arrangement and a file; each half fails separately.
    let g = Goofi::new();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    g.add("Oscillator");
    g.call("layout panel add", j!({ "name": "Second" }));
    g.call("session save", j!({ "path": path.to_string_lossy() }));
    let old_mount = g.state.mount();
    std::fs::write(old_mount.join("notes.md"), b"the previous patch's").unwrap();

    g.call("session new", j!({}));

    assert!(g.nodes().is_empty(), "no nodes");
    assert_eq!(g.call("layout inspect", j!({}))["text"].as_str().unwrap().matches("tab `").count(), 1,
               "no tabs of the previous patch");
    assert_eq!(g.call("session status", j!({}))["save_path"], Value::Null, "no file behind it");
    assert_eq!(g.call("session status", j!({}))["dirty"], false, "and nothing to save");
    assert_eq!(g.call("undo", j!({}))["changed"], false, "the history went with the patch");

    let mount = g.state.mount();
    assert_ne!(mount, old_mount, "a fresh workspace");
    assert!(!old_mount.exists(), "and the one it replaced is released, not leaked");
    // `new` MINTS the workspace, so it seeds the orientation while `load`, one line away, must not.
    assert!(!mount.join("notes.md").exists());
    assert!(std::fs::read_to_string(mount.join("AGENTS.md")).unwrap().contains("goofi-pipe is a live"));
    assert_eq!(std::fs::read_to_string(mount.join("CLAUDE.md")).unwrap(), "@AGENTS.md\n");
}

#[test]
fn a_patch_whose_arrangement_cannot_be_rendered_still_opens() {
    // A layout the flat model admits but cannot render must never make a patch unopenable.
    let g = Goofi::new();
    g.add("Oscillator");
    let yaml = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    // A DUPLICATE id is the one corruption the tree admits and a flat map could not.
    let broken = yaml.replace("id: panel-2", "id: tab-1");
    assert_ne!(broken, yaml, "the fixture actually corrupted something");

    // The two doors are one op, and never both at once: a manifest inline, or an archive at a path.
    let why = g.refuse("session load", j!({ "content": yaml.clone(), "path": "/tmp/nope.gfi" }));
    assert!(why.contains("never both"), "{why}");

    let r = g.call("session load", j!({ "content": broken }));
    assert_eq!(r["ok"], true, "the patch still opens: {r}");
    assert!(r["layout_warning"].as_str().is_some_and(|w| w.contains("appears twice")),
            "…and says why the arrangement was dropped: {r}");
    assert_eq!(g.nodes().len(), 1, "with the graph intact");
}

fn save_path(g: &Goofi) -> Option<String> {
    g.call("session status", j!({}))["save_path"].as_str().map(str::to_string)
}

fn dirty(g: &Goofi) -> bool {
    g.call("session status", j!({}))["dirty"] == true
}

/// A path as goofi spells it back: canonical and `/`-separated, the same resolution a save
/// applies — so an OS shorthand for the same file (a Windows 8.3 name) still compares equal.
fn spelled(p: &std::path::Path) -> String {
    goofi_core::path::to_slash(&goofi_core::path::canonical(p).unwrap())
}

#[test]
fn only_a_patch_with_a_file_behind_it_keeps_a_name_and_every_tab_is_told_which() {
    // The manager owns the stored path, because a plain Save overwrites it silently from any tab.
    let g = Goofi::new();
    assert_eq!(save_path(&g), None, "an unsaved patch has no home yet");
    g.add("Oscillator");

    // A save's ONLY job is writing to a backend path, so a save with no path is malformed.
    let why = g.refuse("session save", j!({}));
    assert!(why.contains("session save") && why.contains("path"), "{why}");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("patch.gfi");
    let mut ev = g.events();
    g.call("session save", j!({ "path": path.to_string_lossy() }));
    assert_eq!(ev.next("save_path_changed")["save_path"].as_str(), Some(spelled(&path).as_str()));
    assert_eq!(save_path(&g).as_deref(), Some(spelled(&path).as_str()));
    // Readable only through `read_gfi` — a bare-YAML write would leave it "not a zip archive".
    let dest = dir.path().join("unpacked");
    let manifest = goofi_graph::archive::read_gfi(&path, &dest).unwrap();
    assert!(manifest.contains("Oscillator"), "the manifest is the serialized patch: {manifest}");
    assert!(dest.is_dir(), "the workspace tree rides along, empty or not");

    // A save that FAILS leaves the previous home standing; naming it would aim the next overwrite at it.
    let nowhere = dir.path().join("no-such-dir").join("patch.gfi");
    g.refuse("session save", j!({ "path": nowhere.to_string_lossy() }));
    assert_eq!(save_path(&g).as_deref(), Some(spelled(&path).as_str()), "the old home stands");

    // An upload carries no file, so inheriting the previous path would save a different patch over it.
    let content = g.call("session manifest", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("session load", j!({ "content": content }));
    assert_eq!(save_path(&g), None, "an uploaded patch has no home");
}

#[test]
fn a_save_packs_the_live_mount_refuses_to_pack_into_it_and_never_truncates_a_good_archive() {
    // Two of `save_archive`'s three properties can only be staged by calling it directly.
    let tmp = tempfile::tempdir().unwrap();
    let mount = tmp.path().join("goofi-0123").join("workspace");
    std::fs::create_dir_all(&mount).unwrap();
    std::fs::write(mount.join("agent.md"), b"notes").unwrap();

    let target = tmp.path().join("patch.gfi");
    goofi_bridge::save_archive(&target, "version: 7\n", &mount).unwrap();
    let dest = tmp.path().join("unpacked");
    assert_eq!(goofi_graph::archive::read_gfi(&target, &dest).unwrap(), "version: 7\n");
    assert_eq!(std::fs::read(dest.join("agent.md")).unwrap(), b"notes", "the LIVE mount is packed");

    // The workspace walk fails after the first zip entry is written. It sits OUTSIDE the target's
    // directory, or the mount refusal below answers first and the pack never runs.
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
    // There is no watcher: the manager compares the mount against the fingerprint of the last pack.
    let g = Goofi::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("patch.gfi");
    std::fs::write(g.state.mount().join("agent.md"), b"notes").unwrap();
    g.call("session save", j!({ "path": target.to_string_lossy() }));
    assert!(!dirty(&g), "the patch was just written to disk, workspace and all");

    std::fs::write(g.state.mount().join("scratch.txt"), b"written since the save").unwrap();
    assert!(dirty(&g), "a workspace file the archive lacks is an unsaved change");

    // The fingerprint carries more than the set of names, including an edit that keeps the length.
    g.call("session save", j!({ "path": target.to_string_lossy() }));
    assert!(!dirty(&g), "saving again re-baselines the workspace");
    std::fs::write(g.state.mount().join("agent.md"), b"NOTES").unwrap();
    assert!(dirty(&g), "an edit to a packed file is an unsaved change too");
    g.call("session save", j!({ "path": target.to_string_lossy() }));
    std::fs::write(g.state.mount().join("agent.md"), b"note!").unwrap();
    assert!(dirty(&g), "a same-length in-place edit is an unsaved change");

    // A save that FAILED packed no file, so those edits still live only in the mount.
    g.refuse("session save", j!({ "path": tmp.path().join("no-such-dir").join("patch.gfi").to_string_lossy() }));
    assert!(dirty(&g), "a save that wrote nothing cannot call the workspace packed");

    // A SECOND manager, which is the real case: it has no baseline of its own to fall back on.
    let opened = Goofi::new();
    opened.call("session load", j!({ "path": target.to_string_lossy() }));
    assert_eq!(std::fs::read(opened.state.mount().join("agent.md")).unwrap(), b"NOTES");
    assert!(!dirty(&opened), "a patch is not unsaved the moment it finishes loading");
}

#[test]
fn the_file_browser_answers_a_path_the_way_save_and_load_take_it() {
    // Expectations built from `goofi_core::path`'s own primitives: the function under test on both
    // sides would accept a normalizer that reversed the segments it re-attaches.
    use goofi_core::path::{canonical, to_slash};
    let g = Goofi::new();
    let list = |p: Option<&str>| match p {
        Some(p) => g.call("dir list", j!({ "path": p })),
        None => g.call("dir list", j!({})),
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
    // A name this platform accepts but UTF-8 cannot express; a lossy entry can collide with
    // another. Best-effort: a filesystem that refuses the bytes (macOS, EILSEQ) has no
    // undecodable names to filter, and Linux CI proves the filter either way.
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let bad = std::ffi::OsString::from_vec(b"bad\xff".to_vec()); // starts no valid sequence
        let _ = std::fs::write(tmp.path().join(bad), b"x");
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
    // The browser renders neither a size nor a date column, so the row carries neither.
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
    let home = std::env::home_dir().expect("a home directory in the test environment");
    let home = to_slash(&canonical(&home).unwrap_or_else(|_| home.clone()));
    assert_eq!(list(None)["path"].as_str().unwrap(), home);
    assert_eq!(list(Some("")), list(None));
    assert_eq!(list(Some("~"))["path"].as_str().unwrap(), home, "a leading tilde expands");
    // The sidebar marks a root active by raw string equality against `path`.
    let roots = list(None)["roots"].as_array().unwrap().clone();
    assert!(roots.iter().any(|r| r["label"] == "Home"), "expected a Home root");
    for root in &roots {
        let path = root["path"].as_str().unwrap();
        assert_eq!(list(Some(path))["path"].as_str().unwrap(), path, "root {path} is not stable");
    }
    // The topmost ancestor of a real path — `/` is not a root on Windows.
    let top = to_slash(canonical(&std::env::temp_dir()).unwrap().ancestors().last().unwrap());
    assert!(list(Some(&top))["parent"].is_null(), "root has no parent to climb to");

    // Navigation must NOT error, or the browser keeps showing the previous directory. `..` has no
    // `file_name()`, so an ancestor walk would silently drop it and land elsewhere.
    let missing = format!("{top}definitely/not/a/directory");
    let listing = list(Some(&missing));
    assert_eq!((names(&listing), listing["path"].as_str().unwrap()), (Vec::new(), missing.as_str()));
    assert_eq!(list(Some(&tmp.path().join("new/deeper").to_string_lossy()))["path"],
               format!("{here}/new/deeper"));
    assert_eq!(list(Some(&tmp.path().join("missing/../also-missing").to_string_lossy()))["path"],
               format!("{here}/also-missing"));

    // Proven through a REFUSAL that names the expanded path: saving into `$HOME` is not something
    // a test may do.
    let why = g.refuse("session load", j!({ "path": "~/definitely-not-a-patch-goofi-wrote.gfi" }));
    assert!(why.contains(&home), "the refusal names the expanded path, not the tilde: {why}");
}
