//! Demo mode: the ONE mode a PUBLIC goofi serves in — the graph, and nothing of the host around
//! it. It is not a sandbox and never claims to be: a param expression is still Python, and what
//! this mode withholds is the convenient doors. `roadmap/demo-mode.md` states the rest.

use goofi_bridge::phrase;
use goofi_tests::{host, http, j, Goofi};

/// Every op a demo drops, and the one it keeps because a visitor needs a reset.
const DROPPED: [&str; 4] = ["dir list", "agent list", "session save", "session load"];

#[tokio::test]
async fn a_public_goofi_serves_the_graph_and_none_of_the_host_around_it() {
    let full = Goofi::new();
    let g = Goofi::demo();

    // The rows are ABSENT, not filtered on read — the index, dispatch and the resolver agree,
    // exactly as they do for headless.
    let names: Vec<String> = g.call("op list", j!({}))["ops"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|o| o["op"].as_str().map(str::to_string))
        .collect();
    for dropped in DROPPED {
        assert!(!names.iter().any(|n| n == dropped), "`{dropped}` is served: {names:?}");
    }
    assert!(names.iter().any(|n| n == "session new"), "the visitor's reset stays: {names:?}");
    assert!(names.iter().any(|n| n == "node add"), "the graph is the whole point: {names:?}");

    // A refusal teaches what is missing rather than naming the mode, because several modes
    // withhold and the caller wants the group.
    let why = g.refuse("agent list", j!({}));
    assert!(why.contains("unknown op"), "{why}");
    let words = vec!["agent".to_string(), "list".to_string()];
    let Err(why) = phrase::resolve(g.state.ops(), &words) else { panic!("resolved on a demo") };
    assert!(why.contains("does not serve the `agent` ops"), "{why}");

    // Completion follows the served set, so nothing offers a door that is not there.
    let offered = |g: &Goofi, word: &str| {
        phrase::complete(g.state.ops(), None, "").iter().any(|(w, _)| w == word)
    };
    for word in ["agent", "dir"] {
        assert!(offered(&full, word), "a full server offers `{word}`");
        assert!(!offered(&g, word), "a demo does not offer `{word}`");
    }

    // No audio ENGINE, which is the one line that also empties the audio half of the catalog.
    let types = |g: &Goofi| -> Vec<String> {
        goofi_bridge::catalog_type_names(&g.state.graph.lock().unwrap())
    };
    assert!(types(&full).iter().any(|t| t.starts_with("audio:")), "a full server has audio nodes");
    assert!(
        !types(&g).iter().any(|t| t.starts_with("audio:")),
        "a demo has none: {:?}",
        types(&g)
    );
    assert!(types(&g).iter().any(|t| t.starts_with("signal:")), "…and every signal node stands");

    // The graph itself is untouched by the mode: a node is added, and it runs.
    let osc = g.add("Oscillator");
    assert_eq!(g.call("node state", j!({ "node": goofi_tests::hex(osc) }))["error"], j!(null));

    let base = g.serve().await;
    let addr = host(&base);
    let other = host(&full.serve().await).to_string();

    // The host-facing routes are NOT MOUNTED. A 404 is the whole statement: there is no handler
    // to refuse from — and the SAME path on a full server answers, so the 404 is the mode rather
    // than a misspelled route.
    for (method, path) in [("POST", "/exec"), ("POST", "/mcp"), ("GET", "/patch.gfi")] {
        let (status, ..) = http(addr, method, path, "", b"").await;
        assert_eq!(status, 404, "{method} {path} is absent on a demo");
        let (status, ..) = http(&other, method, path, "", b"").await;
        assert_ne!(status, 404, "{method} {path} is mounted on a full server");
    }

    // The origin guard is the other half of the mode: a demo lives behind a DNS name, so an
    // Origin that IS this request's Host is admitted. Nothing else changes — a name the request
    // was not sent to is still refused.
    let named = |h: &str, o: &str| {
        format!("Host: {h}\r\nOrigin: http://{o}\r\nContent-Type: application/json\r\n")
    };
    let (status, ..) =
        http(addr, "POST", "/control", &named("goofi.example.com", "goofi.example.com"), b"").await;
    assert_ne!(status, 403, "a demo answers for the name it was reached at");
    let (status, ..) =
        http(addr, "POST", "/control", &named("goofi.example.com", "elsewhere.example.com"), b"").await;
    assert_eq!(status, 403, "…and never for a name it was not");

    // The same request on a FULL server is refused: the mode is what opened that door.
    let (status, ..) = http(
        &other,
        "POST",
        "/control",
        &named("goofi.example.com", "goofi.example.com"),
        b"",
    )
    .await;
    assert_eq!(status, 403, "a local goofi admits no DNS name");
}
