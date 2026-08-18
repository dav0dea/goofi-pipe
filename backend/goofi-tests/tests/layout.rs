//! The arrangement — the fifth doc root, and manager-owned like the graph.
//!
//! **The rule: no layout inverse restores raw state; every inverse re-plans through the forward
//! planners.** Putting a slot back resurrects a container a peer's children no longer belong under,
//! which strands them and corrupts the arrangement on the next save. Every instance of that class
//! was found by driving TWO sessions — a single session's undo is provably clean, so nothing
//! single-session can see it.
//!
//! The judge throughout is the manager's own loader: it is asked to open what the manager just
//! saved, and a `layout_warning` means it could not.

use serde_json::{Map, Value};

use goofi_tests::{hex, j, Goofi};

/// The arrangement's ENTRIES. The root also carries the manager's monotone id counter under a
/// reserved key, which no minted id can take — a reader walks entries, not keys.
fn entries(g: &Goofi) -> Map<String, Value> {
    let mut m = g.doc()["arrangement"].as_object().cloned().unwrap_or_default();
    m.retain(|_, e| e.get("kind").is_some());
    m
}

fn panels(g: &Goofi) -> Vec<String> {
    let mut v: Vec<String> =
        entries(g).iter().filter(|(_, e)| e["kind"] == "panel").map(|(id, _)| id.clone()).collect();
    v.sort();
    v
}

/// The id of the entry a page hangs off. A page holds exactly one, so a second root IS the
/// corruption a resurrected container makes.
fn page_roots(g: &Goofi, name: &str) -> Vec<String> {
    let m = entries(g);
    let Some(page) = m.iter().find(|(_, e)| e["name"] == name).map(|(id, _)| id.clone()) else {
        return Vec::new();
    };
    m.iter().filter(|(_, e)| e["parent"] == page.as_str()).map(|(id, _)| id.clone()).collect()
}

fn size_of(g: &Goofi, id: &str) -> f64 {
    g.doc()["arrangement"][id]["size"].as_f64().unwrap_or(f64::NAN)
}

/// The manager's own loader, asked to open what the manager just saved. `Null` when it can.
fn reload_warning(g: &Goofi) -> Value {
    let yaml = g.call("serialize", j!({}))["yaml"].as_str().unwrap().to_string();
    g.call("load_text", j!({ "content": yaml }))["layout_warning"].clone()
}

fn split(g: &Goofi, page: &str, panel: &str, dir: &str) -> String {
    g.call("page_split_panel", j!({ "page": page, "panel": panel, "direction": dir }))
        .as_str().expect("a split answers the new panel's id").to_string()
}

/// The one panel a fresh patch's default page holds.
fn first_panel(g: &Goofi) -> String {
    panels(g).first().cloned().expect("the default page's one panel")
}

/// The split an entry sits under.
fn parent(g: &Goofi, id: &str) -> String {
    g.doc()["arrangement"][id]["parent"].as_str().expect("an entry has a parent").to_string()
}

#[test]
fn deleting_a_node_empties_the_panels_bound_to_it_and_an_undo_binds_them_back() {
    // A panel bound to a node that is gone renders empty and explains nothing, so the binding goes
    // with the node — INSIDE `RemoveNode`. The client used to do it, back when it owned the layout;
    // doing it there now would cost a second command per delete and leave a peer's copy of the
    // panel pointing at a uid that no longer exists.
    let g = Goofi::new();
    let n = g.add("Oscillator");
    let panel = first_panel(&g);
    g.call("page_set_panel", j!({ "page": "Layout", "panel": panel, "type": "viewer",
                                 "state": { "node": hex(n) } }));

    g.call("remove_node", j!({ "node": hex(n) }));
    assert_eq!(g.doc()["arrangement"][&panel]["state"], "{\"node\":null}");

    // …and comes back with it: the manager owns the inverse, so ONE undo restores both.
    g.call("undo", j!({}));
    assert_eq!(g.doc()["arrangement"][&panel]["state"], format!("{{\"node\":\"{}\"}}", hex(n)));
}

#[test]
fn inspect_layout_reads_the_whole_tree_or_one_page() {
    let g = Goofi::new();
    let first = first_panel(&g);
    g.call("session_add_page", j!({ "name": "Signals" }));
    let fresh = g.call("page_split_panel", j!({ "page": "Layout", "panel": first,
                                               "direction": "row", "ratio": 0.25 }))
        .as_str().unwrap().to_string();

    let text = |p: Value| g.call("inspect_layout", p)["text"].as_str().unwrap().to_string();
    let whole = text(j!({}));
    assert!(whole.contains("Layout") && whole.contains("Signals"), "no arg is every page: {whole}");

    let one = text(j!({ "page": "Layout" }));
    assert!(one.contains(&first) && one.contains(&fresh), "the page's own panels: {one}");
    assert!(!one.contains("Signals"), "…and NOT a page the caller did not ask about: {one}");
    assert!(one.contains("0.25"), "each entry's share is annotated: {one}");
    assert_eq!(g.doc()["arrangement"][&fresh]["panel_type"], "empty", "a split births an empty");

    // A page is addressed by NAME, so an unknown one has to say which exist.
    let why = g.refuse("inspect_layout", j!({ "page": "Nope" }));
    assert!(why.contains("Layout") && why.contains("Signals"), "{why}");
}

#[test]
fn a_split_undoes_and_redoes_at_the_same_panel_id() {
    let g = Goofi::new();
    let panel = first_panel(&g);
    let fresh = split(&g, "Layout", &panel, "column");
    assert_eq!(entries(&g).len(), 4, "page + split + 2 panels");

    assert_eq!(g.call("undo", j!({}))["changed"], true);
    assert_eq!(panels(&g), vec![panel.clone()], "the arrangement is exactly what it was");
    assert_eq!(entries(&g).len(), 2, "the wrapper split went too");

    g.call("redo", j!({}));
    assert!(!g.doc()["arrangement"][&fresh].is_null(), "redo re-splits at the SAME id");
}

#[test]
fn a_split_undone_after_a_peers_split_leaves_the_peers_panel_standing() {
    let one = Goofi::new();
    let two = one.client("s2");
    let panel = first_panel(&one);
    let mine = split(&one, "Layout", &panel, "row");
    // The peer splits the panel s1 just made, so its own panel hangs off s1's wrapper split.
    let theirs = split(&two, "Layout", &mine, "row");

    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert!(one.doc()["arrangement"][&mine].is_null(), "s1's panel went");
    let up = parent(&one, &theirs);
    assert!(entries(&one).contains_key(&up), "the peer's panel still hangs off something");
    assert!(panels(&one).contains(&panel), "and so did the panel that was split");
    assert_eq!(reload_warning(&one), Value::Null);
}

#[test]
fn undoing_a_page_closes_it_whole_rather_than_stranding_the_peers_panel_on_it() {
    // The other birth is a PAGE, and the semantics differ: undoing "add page" closes the page WHOLE,
    // so a peer's panel on it goes too — a lost update, but a CONVERGENT one. Restoring the slots
    // would instead leave the peer's panel hanging off a page that no longer exists.
    let one = Goofi::new();
    let two = one.client("s2");
    let standing = panels(&one);
    let before = entries(&one).len();

    one.call("session_add_page", j!({ "name": "Second" }));
    let second = panels(&one).into_iter().find(|p| !standing.contains(p)).expect("its panel");
    split(&two, "Second", &second, "row");

    assert_eq!(one.call("undo", j!({}))["changed"], true);
    let pages = entries(&one).values().filter(|e| e["kind"] == "page").count();
    assert_eq!(pages, 1, "the page went whole");
    assert_eq!(entries(&one).len(), before, "and took the peer's split with it, leaving no orphan");
    assert_eq!(reload_warning(&one), Value::Null);
}

#[test]
fn a_move_undone_after_a_peers_split_moves_back_rather_than_resurrecting_a_dead_split() {
    // A move that empties a split promotes the survivor and drops the split. Restoring the slots
    // the move displaced puts that DEAD split back at the page root — while the wrapper a peer has
    // since hung its own panel off stays where it is, so the page ends up with two roots.
    let one = Goofi::new();
    let two = one.client("s2");
    let first = first_panel(&one);
    let mine = split(&one, "Layout", &first, "row");
    one.call("session_add_page", j!({ "name": "Signals" }));
    let theirs = panels(&one).into_iter().find(|p| *p != first && *p != mine).expect("its panel");
    let far = split(&one, "Signals", &theirs, "row");
    let dest = parent(&one, &far);

    one.call("page_move_panel", j!({ "page": "Layout", "panel": mine,
                                    "new_parent": dest, "order_index": 0 }));
    assert_eq!(page_roots(&one, "Layout"), vec![first.clone()], "the survivor took the page root");

    // The peer splits that survivor, so its panel hangs off a wrapper sitting in the very slot the
    // dead split wants back.
    let peers = split(&two, "Layout", &first, "column");

    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert_eq!(page_roots(&one, "Layout").len(), 1, "a dead split did not come back");
    assert!(panels(&one).contains(&peers), "the peer's panel survived a foreign undo");
    let page = one.call("inspect_layout", j!({ "page": "Layout" }))["text"].as_str().unwrap().to_string();
    assert!(page.contains(&mine), "and the undo did move the panel back: {page}");
    assert_eq!(reload_warning(&one), Value::Null);
}

#[test]
fn the_two_subtree_drags_undo_without_resurrecting_a_dead_split_either() {
    // Dropping a subtree on a panel and tearing it off onto the tab bar are the same class: each
    // LIFTS a subtree, which can promote its split away, and a slot-restore undo brings that dead
    // split back on top of whatever the peer built where it stood.
    let one = Goofi::new();
    let two = one.client("s2");
    let first = first_panel(&one);
    let mine = split(&one, "Layout", &first, "row");
    one.call("session_add_page", j!({ "name": "Signals" }));
    let target = panels(&one).into_iter().find(|p| *p != first && *p != mine).expect("its panel");

    one.call("page_insert_at_panel", j!({ "page": "Signals", "subtree": mine,
                                         "target": target, "direction": "column" }));
    assert_eq!(page_roots(&one, "Layout"), vec![first.clone()], "the survivor took the page root");
    let peers = split(&two, "Layout", &first, "column");
    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert_eq!(page_roots(&one, "Layout").len(), 1, "a dead split did not come back");
    assert!(panels(&one).contains(&peers), "the peer's panel survived a foreign undo");

    // The tab-bar tear-off carries a subtree onto a page of its own, and lifting it can promote a
    // split away just the same.
    one.call("session_add_page", j!({ "name": "Torn off", "subtree": mine }));
    assert_eq!(page_roots(&one, "Layout").len(), 1, "the page it left kept exactly one root");
    let theirs = split(&two, "Layout", &first, "row");
    assert_eq!(one.call("undo", j!({}))["changed"], true);
    assert_eq!(page_roots(&one, "Layout").len(), 1, "a dead split did not come back");
    assert!(panels(&one).contains(&theirs), "the peer's panel survived the tear-off's undo");
    assert_eq!(reload_warning(&one), Value::Null);
}

#[test]
fn an_undisturbed_move_undoes_to_exactly_where_it_was() {
    // Re-planning the inverse must not cost the single-session expectation that ctrl-Z puts a panel
    // back exactly — same split, same position among its siblings, same shares for all three.
    let g = Goofi::new();
    let first = first_panel(&g);
    let last = split(&g, "Layout", &first, "row");
    let mid = split(&g, "Layout", &first, "row"); // a THREE-child split, so it survives the move
    g.call("session_add_page", j!({ "name": "Signals" }));
    let theirs = panels(&g).into_iter()
        .find(|p| *p != first && *p != mid && *p != last).expect("the new page's panel");
    let far = split(&g, "Signals", &theirs, "row");
    let dest = parent(&g, &far);
    let before = entries(&g);

    g.call("page_move_panel", j!({ "page": "Layout", "panel": mid,
                                  "new_parent": dest, "order_index": 1 }));
    assert_eq!(g.call("undo", j!({}))["changed"], true);
    assert_eq!(entries(&g), before, "one ctrl-Z put the panel back exactly where it was");
}

#[test]
fn a_type_change_undone_after_a_peers_split_leaves_the_peer_its_slot() {
    // `page_set_panel` edits what a panel HOLDS, not where it sits — but its inverse restored the
    // WHOLE entry, `order` among it, pinning the panel back into the slot a peer's adjacent split
    // had taken. Two children of one split at one order is an arrangement the loader refuses.
    let one = Goofi::new();
    let two = one.client("s2");
    let a = first_panel(&one);
    let b = split(&one, "Layout", &a, "row");

    one.call("page_set_panel", j!({ "page": "Layout", "panel": b, "type": "console" }));
    let peer = split(&two, "Layout", &a, "row");
    assert_eq!(one.call("undo", j!({}))["changed"], true);

    assert_eq!(one.doc()["arrangement"][&b]["panel_type"], "empty", "the type it had came back");
    assert_ne!(one.doc()["arrangement"][&b]["order"], one.doc()["arrangement"][&peer]["order"],
               "and the peer kept the order it took");
    assert_eq!(reload_warning(&one), Value::Null);
}

#[test]
fn a_resize_undone_after_a_peers_split_re_asserts_shares_without_re_pinning_slots() {
    // A set of shares is CONTENTS too, and restoring each whole entry to undo them puts the orders
    // back with them. The undo re-asserts the shares it found and renormalizes around whatever the
    // peer added, so the split still divides exactly one slot.
    let one = Goofi::new();
    let two = one.client("s2");
    let a = first_panel(&one);
    let b = split(&one, "Layout", &a, "row");
    let near = parent(&one, &b);

    one.call("page_resize_split", j!({ "page": "Layout", "split": near, "fractions": [0.3, 0.7] }));
    let peer = split(&two, "Layout", &a, "row");
    assert_eq!(one.call("undo", j!({}))["changed"], true);

    assert!((size_of(&one, &a) - size_of(&one, &b)).abs() < 1e-9, "the equal shares it found");
    let total = size_of(&one, &a) + size_of(&one, &b) + size_of(&one, &peer);
    assert!((total - 1.0).abs() < 1e-9, "and the split still divides one slot: {total}");
    assert_eq!(reload_warning(&one), Value::Null);
}

#[test]
fn a_contents_undo_follows_the_panel_a_peer_has_since_carried_off() {
    // A slot is not only an order. A peer may have carried the panel to another split entirely — and
    // the two-child split it left promoted away behind it. Restoring the entry's own `parent` hangs
    // the panel off a container that is no longer there, which reaches no page at all.
    let one = Goofi::new();
    let two = one.client("s2");
    let a = first_panel(&one);
    let b = split(&one, "Layout", &a, "row");
    one.call("session_add_page", j!({ "name": "Two" }));
    let c = panels(&one).into_iter().find(|p| *p != a && *p != b).expect("the second page's panel");
    let e = split(&one, "Two", &c, "row");
    let far = parent(&one, &e);

    one.call("page_set_panel", j!({ "page": "Layout", "panel": b, "type": "console" }));
    two.call("page_move_panel", j!({ "page": "Layout", "panel": b,
                                    "new_parent": far, "order_index": 0 }));
    assert_eq!(one.call("undo", j!({}))["changed"], true);

    assert_eq!(one.doc()["arrangement"][&b]["panel_type"], "empty");
    assert_eq!(one.doc()["arrangement"][&b]["parent"], far,
               "the type came back where the peer had carried the panel to");
    assert_eq!(reload_warning(&one), Value::Null);
}

#[test]
fn a_rename_undone_after_a_peers_reorder_keeps_the_tab_index_it_finds() {
    // A page's NAME is contents and its tab index is the slot. A peer's reorder renumbers the strip,
    // and restoring the whole page entry puts back an index another tab now holds.
    let one = Goofi::new();
    let two = one.client("s2");
    one.call("session_add_page", j!({ "name": "Two" }));
    one.call("session_add_page", j!({ "name": "Three" }));

    one.call("session_rename_page", j!({ "from": "Two", "to": "Deux" }));
    two.call("session_reorder_page", j!({ "name": "Three", "to_index": 0 }));
    assert_eq!(one.call("undo", j!({}))["changed"], true);

    let mut tabs: Vec<i64> = entries(&one).values().filter(|e| e["kind"] == "page")
        .filter_map(|e| e["order"].as_i64()).collect();
    tabs.sort_unstable();
    tabs.dedup();
    assert_eq!(tabs.len(), 3, "the undo renamed the page without taking a tab index twice");
    assert_eq!(reload_warning(&one), Value::Null);
}

/// **The rule, enforced rather than remembered.** Three rounds found three instances of the class
/// BY HAND, which is two too many — so every layout write op is driven through the one interleaving
/// that makes it visible: a peer edits between the op and its undo. The op list comes from the
/// REGISTRY and the match has no catch-all, so a new layout op is red until it is driven too.
#[test]
fn no_layout_undo_puts_back_a_slot_a_peer_has_since_built_over() {
    let ops: Vec<&str> = goofi_bridge::ops::REGISTRY.iter()
        .filter(|o| o.writes && (o.name.starts_with("page_") || o.name.starts_with("session_")))
        .map(|o| o.name)
        .collect();
    assert!(ops.contains(&"page_remove_panel") && ops.contains(&"session_remove_page"),
            "the registry filter still finds the layout write ops: {ops:?}");

    let mut stranded = Vec::new();
    for op in &ops {
        // A fresh manager per op, so each meets the same arrangement: `Layout` holds a two-child
        // split and `Two` holds another. Between them every op has an argument that exists, and each
        // leaves `a` standing for the peer to build on.
        let one = Goofi::new();
        let two = one.client("s2");
        let a = first_panel(&one);
        let b = split(&one, "Layout", &a, "row");
        one.call("session_add_page", j!({ "name": "Two" }));
        let c = panels(&one).into_iter().find(|p| *p != a && *p != b).expect("the page's panel");
        let e = split(&one, "Two", &c, "row");
        let far = parent(&one, &e);
        let near = parent(&one, &b);

        let payload = match *op {
            "session_add_page" => j!({ "name": "Fresh" }),
            "session_remove_page" => j!({ "name": "Two" }),
            "session_rename_page" => j!({ "from": "Two", "to": "Deux" }),
            "session_reorder_page" => j!({ "name": "Two", "to_index": 0 }),
            "page_split_panel" => j!({ "page": "Layout", "panel": a }),
            "page_set_panel" => j!({ "page": "Layout", "panel": b, "type": "console" }),
            "page_move_panel" => j!({ "page": "Layout", "panel": b, "new_parent": far, "order_index": 0 }),
            "page_insert_at_panel" => j!({ "page": "Two", "subtree": b, "target": c }),
            "page_resize_split" => j!({ "page": "Layout", "split": near, "fractions": [0.3, 0.7] }),
            "page_remove_panel" => j!({ "page": "Layout", "panel": b }),
            new => panic!("`{new}` is a layout write op with no case here — drive it through this \
                           guard, and say why if its inverse may restore a slot"),
        };
        one.call(op, payload);
        // The peer builds exactly where that op's slot-restore inverse would want to write: over the
        // survivor `a` for a structural op, over the tab index the page ops renumber.
        if op.starts_with("session_") {
            two.call("session_add_page", j!({ "name": "Peer" }));
        } else {
            two.call("page_split_panel", j!({ "page": "Layout", "panel": a }));
        }
        assert_eq!(one.call("undo", j!({}))["changed"], true, "{op}: the undo flipped nothing");

        if reload_warning(&one) != Value::Null {
            stranded.push(*op);
        }
    }
    // EMPTY, and it stays empty. The two this guard found (`page_set_panel`, `page_resize_split`)
    // now invert through `Command::LayoutContents`, which re-reads each slot at flip time. The one
    // op whose inverse still restores an `order` is `session_reorder_page` — where the order IS the
    // content, so carrying the live one over would make its undo a no-op. It is driven above
    // regardless, so the day it does strand, this list is what says so.
    let empty: [&str; 0] = [];
    assert_eq!(stranded, empty, "an undo left an arrangement the manager cannot itself open");
}
