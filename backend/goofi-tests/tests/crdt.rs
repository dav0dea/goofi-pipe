//! The CRDT document itself: the shape-agnostic mirror every client replicates, and the sync
//! handshake that carries it.
//!
//! Shape-agnostic on purpose — the doc knows maps and leaves, and `crdt_mirror` is the only place
//! the graph's roots are named. What is pinned here is the ALGEBRA: reconcile is idempotent,
//! prunes what went, normalizes numbers, and a hostile state vector degrades instead of aborting.

use goofi_bridge::crdt::*;

// ---- test shims: read leaves through the generic reader (the typed getters were removed) ----
// Numbers come back from `to_json` in shortest form (a whole f64 `30.0` reads as integer `30`),
// so numeric assertions compare via `as_f64`, not exact `json!` equality.
fn nstr(doc: &GraphDoc, uid: &str, key: &str) -> Option<String> {
    doc.read_at(&["nodes", uid, key]).and_then(|v| v.as_str().map(String::from))
}
fn npos(doc: &GraphDoc, root: &str, uid: &str) -> Option<[f64; 2]> {
    let x = doc.read_at(&[root, uid, "pos", "x"])?.as_f64()?;
    let y = doc.read_at(&[root, uid, "pos", "y"])?.as_f64()?;
    Some([x, y])
}
fn pval(doc: &GraphDoc, uid: &str, g: &str, n: &str) -> Option<serde_json::Value> {
    doc.read_at(&["nodes", uid, "params", g, n, "value"])
}
fn pnum(doc: &GraphDoc, uid: &str, g: &str, n: &str) -> Option<f64> {
    pval(doc, uid, g, n).and_then(|v| v.as_f64())
}
fn pexpr_src(doc: &GraphDoc, uid: &str, g: &str, n: &str) -> Option<String> {
    doc.read_at(&["nodes", uid, "params", g, n, "expr", "source"])
            .and_then(|v| v.as_str().map(String::from))
    }
    fn viewers(doc: &GraphDoc, uid: &str) -> Option<serde_json::Value> {
        doc.read_at(&["nodes", uid, "viewers"])
            .and_then(|v| v.as_str().and_then(|s| serde_json::from_str(s).ok()))
    }
    fn links(doc: &GraphDoc) -> Vec<serde_json::Value> {
        doc.read_at(&["links"]).and_then(|v| v.as_array().cloned()).unwrap_or_default()
    }

    #[test]
    fn a_fresh_doc_has_no_nodes() {
        let doc = GraphDoc::new();
        assert!(doc.node_ids().is_empty());
    }

    #[test]
    fn viewers_blob_and_links_round_trip() {
        use serde_json::json;
        // A node's viewers blob is a STRING leaf; build it via the generic reconciler and read it
        // back through the generic reader (parsing the string), then exercise the wholesale replace.
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&json!({
            "nodes": { "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0},
                "params": {}, "viewers": "{\"out\":{\"kind\":\"line\"}}" } },
            "links": [], "instances": {}
        }));
        assert_eq!(viewers(&doc, "1"), Some(json!({"out": {"kind": "line"}})));

        doc.replace_links(&[json!({
            "node_out": "1", "slot_out": "out", "node_in": "2", "slot_in": "data",
        })]);
        assert_eq!(links(&doc).len(), 1);
        assert_eq!(links(&doc)[0]["slot_in"], json!("data"));
        doc.replace_links(&[]);
        assert!(links(&doc).is_empty());
    }

    #[test]
    fn replace_links_is_idempotent() {
        // The re-mirror re-asserts the whole link set after every op. Re-asserting the SAME links
        // must produce NO doc ops — else the link array churns (new items + tombstones) on every
        // unrelated edit, defeating the empty-diff broadcast-skip for any patch that has links.
        let mut doc = GraphDoc::new();
        let l = |a: &str, b: &str| {
            serde_json::json!({ "node_out": a, "slot_out": "out", "node_in": b, "slot_in": "in" })
        };
        doc.replace_links(&[l("1", "2"), l("2", "3")]);

        let before = doc.to_json();
        doc.replace_links(&[l("1", "2"), l("2", "3")]);
        assert_eq!(doc.to_json(), before, "re-asserting the same link set must be a no-op");
        // A real change (an added link) still applies.
        doc.replace_links(&[l("1", "2"), l("2", "3"), l("3", "4")]);
        assert_ne!(doc.to_json(), before, "a real link change is a logical change");
        assert_eq!(links(&doc).len(), 3);
        // Order matters — a reordering is a real change.
        let before2 = doc.to_json();
        doc.replace_links(&[l("3", "4"), l("1", "2"), l("2", "3")]);
        assert_ne!(doc.to_json(), before2, "a reordering is a change");
    }

    #[test]
    fn remove_node_and_state_round_trip() {
        use serde_json::json;
        let node2 = || json!({ "type": "Buffer", "name": "buf", "pos": {"x": 1.0, "y": 2.0}, "params": {} });
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {} },
            "2": node2()
        }, "links": [], "instances": {} }));
        // Removal is wholesale: re-mirror the projection with node 1 omitted → it is pruned.
        doc.reconcile_root(&json!({ "nodes": { "2": node2() }, "links": [], "instances": {} }));
        assert_eq!(doc.node_ids(), vec!["2"]);

        // Read back through the PROTOCOL rather than the primitive under it: a full-state frame
        // is exactly what a joining client is sent.
        let mut copy = GraphDoc::new();
        copy.on_sync(SyncMsg::decode(&doc.full_state_frame()).expect("a full-state frame"));
        assert_eq!(copy.node_ids(), vec!["2"]);
        assert_eq!(nstr(&copy, "2", "name").as_deref(), Some("buf"));
    }

    #[test]
    fn sync_diff_converges_two_replicas() {
        use serde_json::json;
        // The relay handshake: a peer advertises its state vector, the other returns a diff,
        // the peer applies it and converges — the primitive the /control sync relay uses.
        let node = |name: &str| json!({ "nodes": {
            "1": { "type": "Oscillator", "name": name, "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} });
        let mut server = GraphDoc::new();
        server.reconcile_root(&node("osc"));

        let client = GraphDoc::new(); // empty replica just joined
        let diff = server.diff(&client.state_vector());
        let mut client = client;
        client.on_sync(SyncMsg::Update(diff.clone()));
        assert_eq!(nstr(&client, "1", "name").as_deref(), Some("osc"), "client converged via diff");

        // A later incremental edit on the server produces a small diff the client applies.
        server.reconcile_root(&node("osc2"));
        let diff2 = server.diff(&client.state_vector());
        client.on_sync(SyncMsg::Update(diff2.clone()));
        assert_eq!(nstr(&client, "1", "name").as_deref(), Some("osc2"));
    }

    #[test]
    fn a_hostile_state_vector_degrades_instead_of_aborting() {
        use serde_json::json;
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));

        // Six bytes off the wire declaring ~4e9 entries. yrs pre-allocates the map from the
        // DECLARED count before reading a single entry, so an unvalidated decode aborts the
        // WHOLE PROCESS via `handle_alloc_error` — not a catchable panic, not an `Err`.
        let full = server.diff(&[]);
        assert_eq!(server.diff(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F]), full, "count exceeds the bytes backing it");
        assert_eq!(server.diff(&[0xFF; 12]), full, "var-uint that never terminates");
        assert_eq!(server.diff(&[0x04, 0x01, 0x00]), full, "4 entries in 2 bytes");

        // An honest state vector still computes a real (here: empty) diff.
        assert!(server.diff(&server.state_vector()).len() < full.len(), "an up-to-date peer is owed nothing");
    }

    #[test]
    fn sync_msg_encode_decode_round_trip() {
        for m in [
            SyncMsg::StateVector(vec![1, 2, 3]),
            SyncMsg::Update(vec![9, 8]),
        ] {
            let bytes = m.clone().encode();
            assert_eq!(SyncMsg::decode(&bytes), Some(m));
        }
        assert_eq!(SyncMsg::decode(&[]), None, "empty is not a message");
        assert_eq!(SyncMsg::decode(&[7, 0]), None, "unknown tag rejected");
    }

    #[test]
    fn on_sync_pairwise_handshake_converges() {
        use serde_json::json;
        // The symmetric handshake: each side sends its SV on connect; receiving a peer's SV
        // yields an Update carrying what the peer lacks; receiving an Update applies it.
        let node1 = || json!({ "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {} });
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": { "1": node1() }, "links": [], "instances": {} }));
        let mut client = GraphDoc::new();

        // Connect: both emit their SV.
        let server_hello = server.sync_hello();
        let client_hello = client.sync_hello();

        // Server receives client's SV → replies with the diff the client is missing.
        let to_client = server.on_sync(SyncMsg::decode(&client_hello).unwrap());
        // Client receives server's SV → replies with the diff the server is missing (none here).
        let _to_server = client.on_sync(SyncMsg::decode(&server_hello).unwrap());

        // Client applies the server's diff → converges.
        for m in to_client {
            client.on_sync(m);
        }
        assert_eq!(nstr(&client, "1", "name").as_deref(), Some("osc"), "client converged via on_sync");

        // A live server edit, relayed as one Update, lands on the client. reconcile_root is
        // wholesale, so add node 2 while KEEPING node 1 (omitting it would prune it).
        server.reconcile_root(&json!({ "nodes": {
            "1": node1(),
            "2": { "type": "Buffer", "name": "buf", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));
        let live = server.diff(&client.state_vector());
        client.on_sync(SyncMsg::Update(live));
        assert_eq!(nstr(&client, "2", "name").as_deref(), Some("buf"));
    }

    #[test]
    fn full_state_frame_recovers_a_gapped_replica() {
        // The recovery contract: when a client has missed deltas (lag/reconnect), the server
        // ships its FULL STATE as an Update; applying it converges the client regardless of
        // what it missed — including a change that DEPENDS on a missed one (which yrs would
        // otherwise buffer as an unresolvable pending update). This is why recovery must send
        // full state, not the server's state vector (which a reader answers with an empty diff).
        use serde_json::json;
        let mut server = GraphDoc::new();
        server.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));

        let mut client = GraphDoc::new();
        client.on_sync(SyncMsg::Update(server.diff(&client.state_vector())));
        assert_eq!(client.node_ids(), vec!["1"], "client synced node 1");

        // The client now MISSES everything below (dropped deltas): a new node, a param edit, and a
        // rename of node 1 (a struct that chains off the earlier one). One wholesale re-mirror.
        server.reconcile_root(&json!({ "nodes": {
            "1": { "type": "Oscillator", "name": "renamed", "pos": {"x": 0.0, "y": 0.0},
                "params": { "common": { "max_frequency": { "value": 50.0 } } } },
            "2": { "type": "Buffer", "name": "buf", "pos": {"x": 0.0, "y": 0.0}, "params": {} } },
            "links": [], "instances": {} }));

        // Recovery: apply the framed full state. Convergence, not divergence.
        let SyncMsg::Update(full) = SyncMsg::decode(&server.full_state_frame()).unwrap() else {
            panic!("full_state_frame is an Update");
        };
        client.on_sync(SyncMsg::Update(full.clone()));
        assert_eq!(client.node_ids().len(), 2, "gapped node arrived");
        assert_eq!(nstr(&client, "1", "name").as_deref(), Some("renamed"), "dependent change resolved");
        assert_eq!(nstr(&client, "2", "name").as_deref(), Some("buf"));
        assert_eq!(pnum(&client, "1", "common", "max_frequency"), Some(50.0));
    }

    // ---- generic reconcile_root: the single writer that subsumes the typed writer zoo ----

    /// A doc-projection covering every shape the reconciler must handle: a node with params (one
    /// plain, one expression-bound), a viewers blob, a link, and a sub-patch instance with a member
    /// and a wired output boundary. Exactly the doc's field set — no runtime fields.
    fn full_projection() -> serde_json::Value {
        serde_json::json!({
            "nodes": {
                "1": {
                    "type": "Oscillator", "name": "osc", "pos": {"x": 10.0, "y": 20.0},
                    "params": {
                        "common": { "max_frequency": { "value": 30.0 } },
                        "oscillator": { "waveform": { "value": "sine",
                            "expr": { "source": "nd('lfo')", "enabled": true, "triggers": false } } }
                    },
                    "viewers": "{\"out\":{\"kind\":\"line\"}}"
                },
                "2": { "type": "Buffer", "name": "buf", "pos": {"x": 0.0, "y": 0.0}, "params": {} }
            },
            "links": [ { "node_out": "1", "slot_out": "out", "node_in": "2", "slot_in": "data" } ],
            "instances": {
                "i1": {
                    "name": "subpatch0", "parent": ROOT_MARK, "pos": {"x": 5.0, "y": 6.0},
                    "members": { "buffer0": "2" },
                    "interface": { "out0": { "dir": "out", "dtype": "ARRAY", "name": "wave",
                        "pos": {"x": 1.0, "y": 2.0}, "inner_node": "2", "inner_slot": "out" } }
                }
            }
        })
    }
    const ROOT_MARK: &str = "__root__";

    #[test]
    fn reconcile_mirrors_globals_and_is_idempotent() {
        use serde_json::json;
        let proj = || json!({
            "nodes": {}, "links": [], "instances": {},
            "globals": {
                "default_ufreq": { "value": 30.0, "type": "float", "system": true },
                "subject": { "value": "P07", "type": "string", "system": false },
            }
        });
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&proj());
        assert_eq!(doc.read_at(&["globals", "default_ufreq", "value"]).and_then(|v| v.as_f64()), Some(30.0));
        assert_eq!(doc.read_at(&["globals", "default_ufreq", "system"]), Some(json!(true)));
        assert_eq!(doc.read_at(&["globals", "subject", "value"]).and_then(|v| v.as_str().map(str::to_string)), Some("P07".into()));
        // Idempotent: re-mirroring the same globals produces no logical change — the params lesson.
        let before = doc.to_json();
        doc.reconcile_root(&proj());
        assert_eq!(doc.to_json(), before, "re-mirroring identical globals is a no-op");
    }

    #[test]
    fn reconcile_mirrors_the_flat_arrangement_root() {
        use serde_json::json;
        // The fifth root, and the whole reason the arrangement is flat: an id-keyed map of scalars
        // is exactly the `nodes` shape this reconciler already handles. A NESTED panel tree would
        // have lost its `children` arrays here, silently.
        let proj = |ty: &str| json!({ "nodes": {}, "links": [], "instances": {},
            "arrangement": {
                "page-1": { "kind": "page", "name": "Layout", "order": 0 },
                "split-3": { "kind": "split", "parent": "page-1", "order": 0, "size": 1.0, "axis": "row" },
                "panel-2": { "kind": "panel", "parent": "split-3", "order": 0, "size": 0.5,
                             "panel_type": ty, "state": "{\"node\":\"osc0\"}" },
            }});
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&proj("viewer"));
        assert_eq!(doc.read_at(&["arrangement", "panel-2", "parent"]), Some(json!("split-3")));
        assert_eq!(doc.read_at(&["arrangement", "split-3", "axis"]), Some(json!("row")));
        assert_eq!(doc.read_at(&["arrangement", "panel-2", "size"]).and_then(|v| v.as_f64()), Some(0.5));
        assert_eq!(doc.read_at(&["arrangement", "panel-2", "state"]), Some(json!("{\"node\":\"osc0\"}")));

        let before = doc.to_json();
        doc.reconcile_root(&proj("viewer"));
        assert_eq!(doc.to_json(), before, "re-mirroring an unchanged arrangement writes nothing");
        doc.reconcile_root(&proj("console"));
        assert_ne!(doc.to_json(), before, "a real panel-type change is a logical change");

        // Closing a panel prunes its entry, exactly as removing a global prunes its key.
        doc.reconcile_root(&json!({ "nodes": {}, "links": [], "instances": {}, "arrangement": {} }));
        assert!(doc.read_at(&["arrangement", "panel-2"]).is_none());
    }

    #[test]
    fn reconcile_prunes_a_removed_global() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&json!({ "nodes": {}, "links": [], "instances": {},
            "globals": { "g": { "value": 1, "type": "int", "system": false } } }));
        assert!(doc.read_at(&["globals", "g", "value"]).is_some());
        // A re-mirror without `g` prunes it (mirror of a user delete applied to the engine).
        doc.reconcile_root(&json!({ "nodes": {}, "links": [], "instances": {}, "globals": {} }));
        assert!(doc.read_at(&["globals", "g"]).is_none());
    }

    #[test]
    fn reconcile_root_builds_the_whole_graph() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());

        // Nodes + identity + params (value AND binding) + viewers, via the generic reader.
        assert_eq!(doc.node_ids().len(), 2);
        assert_eq!(nstr(&doc, "1", "type").as_deref(), Some("Oscillator"));
        assert_eq!(nstr(&doc, "1", "name").as_deref(), Some("osc"));
        assert_eq!(npos(&doc, "nodes", "1"), Some([10.0, 20.0]));
        assert_eq!(pnum(&doc, "1", "common", "max_frequency"), Some(30.0));
        assert_eq!(pval(&doc, "1", "oscillator", "waveform"), Some(json!("sine")));
        assert_eq!(pexpr_src(&doc, "1", "oscillator", "waveform").as_deref(), Some("nd('lfo')"));
        assert_eq!(viewers(&doc, "1"), Some(json!({"out": {"kind": "line"}})));
        // Links.
        assert_eq!(links(&doc).len(), 1);
        assert_eq!(links(&doc)[0]["node_in"], json!("2"));
        // The sub-patch forest — read the instance object from the generic reader.
        let j = doc.to_json();
        let rec = &j["instances"]["i1"];
        assert_eq!(rec["parent"], json!("__root__"));
        assert_eq!(npos(&doc, "instances", "i1"), Some([5.0, 6.0]));
        assert!(rec.get("def_id").is_none(), "a unique instance omits def_id");
        assert_eq!(rec["members"], json!({ "buffer0": "2" }));
        let out = rec["interface"]
            .as_object()
            .unwrap()
            .values()
            .find(|b| b["dir"] == json!("out"))
            .expect("output boundary");
        assert_eq!(out["inner_node"], json!("2"));
        assert_eq!(out["inner_slot"], json!("out"));
    }

    #[test]
    fn reconcile_root_is_idempotent() {
        // The load-bearing invariant: re-asserting an UNCHANGED projection produces ZERO doc ops
        // (else the re-mirror churns tombstones and manufactures competing writes that race a
        // client's leaf-edit — the "params lesson" the typed writers hand-rolled per field).
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());
        let before = doc.to_json();
        doc.reconcile_root(&full_projection());
        assert_eq!(doc.to_json(), before, "re-reconciling an unchanged graph is a no-op");
    }

    #[test]
    fn a_null_leaf_is_idempotent_like_every_other_scalar() {
        use serde_json::json;
        // `insert_scalar` stores a JSON null as `Any::Null`, so `read_scalar` must read it back as
        // a null — otherwise the leaf is forever "changed", `reconcile_map` rewrites it on every
        // re-mirror, and the idempotence invariant holds for three of the four scalar kinds only.
        // `to_json` cannot see this (a rewritten null looks identical), so assert on the doc's own
        // clock: a write bumps the state vector, a no-op does not.
        let mut doc = GraphDoc::new();
        let proj = json!({ "nodes": { "1": { "type": "Buffer", "name": "buf",
            "pos": {"x": 0.0, "y": 0.0},
            "params": { "buffer": { "size": { "value": null } } } } },
            "links": [], "instances": {} });
        doc.reconcile_root(&proj);
        let sv = doc.state_vector();
        doc.reconcile_root(&proj);
        assert_eq!(doc.state_vector(), sv, "re-asserting an unchanged null leaf writes nothing");
    }

    #[test]
    fn reconcile_normalizes_int_vs_float_numbers() {
        use serde_json::json;
        // Numbers are stored as f64. A projection carrying an INT param value (e.g. Buffer.size)
        // must not churn against its stored f64 form on the next re-mirror.
        let mut doc = GraphDoc::new();
        let mut proj = json!({ "nodes": { "1": { "type": "Buffer", "name": "buf",
            "pos": {"x": 0.0, "y": 0.0}, "params": { "buffer": { "size": { "value": 1000 } } } } },
            "links": [], "instances": {} });
        doc.reconcile_root(&proj);
        let before = doc.to_json();
        // Re-assert with the value as a float — the same number, different JSON repr.
        proj["nodes"]["1"]["params"]["buffer"]["size"]["value"] = json!(1000.0);
        doc.reconcile_root(&proj);
        assert_eq!(doc.to_json(), before, "int 1000 vs f64 1000.0 is not a change");
    }

    #[test]
    fn reconcile_prunes_removed_keys() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());
        assert!(pexpr_src(&doc, "1", "oscillator", "waveform").is_some());

        // A shrunk projection: node 2 gone, node 1's expr binding cleared, the instance's member
        // dropped, and the instance itself removed. Every stale key must be pruned.
        let shrunk = json!({
            "nodes": { "1": { "type": "Oscillator", "name": "osc", "pos": {"x": 10.0, "y": 20.0},
                "params": { "oscillator": { "waveform": { "value": "sine" } } } } },
            "links": [],
            "instances": {}
        });
        doc.reconcile_root(&shrunk);
        assert_eq!(doc.node_ids(), vec!["1"], "node 2 pruned");
        assert_eq!(pexpr_src(&doc, "1", "oscillator", "waveform"), None, "cleared binding pruned");
        assert!(pval(&doc, "1", "common", "max_frequency").is_none(), "removed param group pruned");
        assert!(links(&doc).is_empty(), "links cleared");
        assert!(doc.instance_ids().is_empty(), "instance pruned");
    }

    // ---- generic read (to_json / read_at) ----

    #[test]
    fn to_json_and_read_at_expose_the_whole_doc_generically() {
        use serde_json::json;
        let mut doc = GraphDoc::new();
        doc.reconcile_root(&full_projection());

        // to_json yields the doc's three roots as plain JSON — the generic reader. Numbers come
        // back in shortest form (a whole f64 30.0 serializes as 30), so compare them numerically.
        let j = doc.to_json();
        assert_eq!(j["nodes"]["1"]["name"], json!("osc"));
        assert_eq!(j["nodes"]["1"]["params"]["common"]["max_frequency"]["value"].as_f64(), Some(30.0));
        assert_eq!(j["instances"]["i1"]["parent"], json!("__root__"));
        assert_eq!(j["links"][0]["node_in"], json!("2"));

        // read_at navigates by path (serde-pointer semantics), None when absent.
        assert_eq!(read_at_val(&doc, &["nodes", "1", "pos", "x"]).and_then(|v| v.as_f64()), Some(10.0));
        assert_eq!(
            read_at_val(&doc, &["nodes", "1", "params", "oscillator", "waveform", "expr", "source"]),
            Some(json!("nd('lfo')"))
        );
        assert_eq!(read_at_val(&doc, &["nodes", "nope"]), None);
    }
    // Small shim so the test reads a path without repeating the join.
    fn read_at_val(doc: &GraphDoc, path: &[&str]) -> Option<serde_json::Value> {
        doc.read_at(path)
    }
