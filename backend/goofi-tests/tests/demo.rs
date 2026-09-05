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
    let osc = g.add("LFO");
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

/// The demo's other half: a PUBLIC goofi is billed for as long as it talks, so an untouched one
/// announces a countdown and then goes quiet. It does not exit — an exited container does not wake
/// on the next visitor's request, and the demo would be gone until someone redeployed it.
#[tokio::test]
async fn an_untouched_demo_counts_down_and_hands_itself_back() {
    let mut g = Goofi::demo();
    // The production window is ten minutes and a minute of grace; a test says the same thing in
    // milliseconds, which is what the policy being injectable is for.
    g.state.idle = goofi_bridge::IdlePolicy {
        warn_after: std::time::Duration::from_millis(250),
        grace: std::time::Duration::from_millis(400),
    };
    let base = g.serve().await;

    let (mut a, _) = goofi_tests::Client::connect(&base).await;
    let (mut b, _) = goofi_tests::Client::connect_as(&base, "second-tab").await;

    // Untouched for the window: the countdown is announced, and it carries how long stands so
    // every tab counts toward one instant rather than starting a clock of its own.
    let said = a.event("demo_idle").await;
    assert_eq!(said["closing_in_ms"], j!(400), "the announcement names the grace");
    assert_eq!(b.event("demo_idle").await["closing_in_ms"], j!(400), "…to every tab, not the idle one");

    // One visitor speaking withdraws it FOR EVERYONE — the patch is shared, so the instance is
    // idle only when nobody at all has touched it.
    b.call("session status", j!({})).await;
    assert_eq!(a.event("demo_idle").await["closing_in_ms"], j!(null), "a peer's word withdraws it");

    // Left alone again, it announces once more and this time runs out. The sockets close.
    assert_eq!(a.event("demo_idle").await["closing_in_ms"], j!(400), "silence starts it again");
    a.until_closed().await;
    b.until_closed().await;

    // …and the SAME process is still there to serve the next visitor, who finds a working demo.
    // The instance id is what says it never died: a restart would mint a new one.
    let (mut c, hello) = goofi_tests::Client::connect(&base).await;
    assert_eq!(hello["demo"], j!(true), "the mode survives the handback");
    let status = c.call("session status", j!({})).await;
    assert_eq!(status["instance_id"], hello["instance_id"], "it went quiet, it did not exit");

    // Arriving is speaking, so this visitor is not swept up by the countdown that closed the last.
    assert_eq!(c.event("demo_idle").await["closing_in_ms"], j!(400), "their own window starts fresh");
}
