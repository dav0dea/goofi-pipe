//! The CLI session: the client library against the real `/exec` door — target resolution over
//! `$GOOFI_HOME`, every phrase reachable, the batch as one undo step per actor, and the raw
//! read round-tripped through NPY. The true argv-to-process path is e2e's, which spawns the
//! real binary; nothing here depends on the bin crate.

use goofi_client as client;
use goofi_core::home;
use goofi_tests::Goofi;

fn lines(cmds: &[&str]) -> Vec<String> {
    cmds.iter().map(|c| c.to_string()).collect()
}

/// One entry's rendered text, for a command that must succeed.
fn ok(url: &str, actor: &str, cmd: &str) -> String {
    let entries = client::exec(url, &lines(&[cmd]), Some(actor))
        .unwrap_or_else(|e| panic!("`{cmd}` was refused: {e}"));
    entries.into_iter().next().map(|e| e["text"].as_str().unwrap_or_default().to_string()).unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shell_finds_its_server_and_drives_the_whole_vocabulary_through_exec() {
    // The ONE test in this binary, so the process-global env is nobody else's.
    let tmp = std::env::temp_dir().join(format!("goofi-cli-home-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::env::set_var("GOOFI_HOME", &tmp);
    std::env::remove_var("GOOFI_SESSION");

    // 0 sessions: refused by telling how to start one.
    let why = client::resolve_target().unwrap_err();
    assert!(why.contains("no running goofi"), "{why}");

    // The server is in-process (`serve_app`), which records nothing — the RECORDS under test
    // are written here, as the binary's serve path writes its own.
    let g = Goofi::new();
    let base = g.serve().await;
    let url = format!("http://{}", base.trim_start_matches("ws://"));
    let id = g.state.instance_id.to_string();
    home::write_session(&id, &url);

    // A record whose id the probe contradicts is swept; the true record resolves alone.
    home::write_session("an_impostor", &url);
    // …and a refused connection sweeps too.
    home::write_session("long_gone", "http://127.0.0.1:1");
    let target = tokio::task::spawn_blocking(client::resolve_target).await.unwrap().unwrap();
    assert_eq!((target.id.as_str(), target.url.as_str()), (id.as_str(), url.as_str()));
    assert_eq!(
        home::sessions().len(),
        1,
        "the impostor and the dead record were swept; the live one stays"
    );

    // A connect the caller's OWN side blocks (a sandboxed agent shell) proves nothing about the
    // server: the record survives the probe, still resolves, and the exec failure names the
    // sandbox instead of declaring no goofi runs. The broadcast address is the portable stand-in —
    // every platform refuses it on the caller's side.
    home::write_session("sandbox_jailed", "http://255.255.255.255:1");
    let rows = tokio::task::spawn_blocking(client::list).await.unwrap();
    assert!(
        rows.iter().any(|(s, p)| s.id == "sandbox_jailed" && *p == client::Probed::Unresponsive),
        "kept tentatively, never swept: {rows:?}"
    );
    std::env::set_var("GOOFI_SESSION", "sandbox_jailed");
    let target = tokio::task::spawn_blocking(client::resolve_target).await.unwrap().unwrap();
    let why = client::exec(&target.url, &lines(&["session status"]), None).unwrap_err();
    assert!(why.contains("sandbox"), "the failure names the real cause: {why}");
    std::env::remove_var("GOOFI_SESSION");
    home::remove_session("sandbox_jailed");

    // A listener that never answers is INCONCLUSIVE: kept, listed unresponsive — and now the
    // bare resolution is ambiguous and says so by naming both.
    let mute = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mute_url = format!("http://{}", mute.local_addr().unwrap());
    home::write_session("busy_peer", &mute_url);
    let rows = tokio::task::spawn_blocking(client::list).await.unwrap();
    assert_eq!(rows.len(), 2, "kept tentatively: {rows:?}");
    assert!(rows.iter().any(|(s, p)| s.id == "busy_peer" && *p == client::Probed::Unresponsive));
    let why = tokio::task::spawn_blocking(client::resolve_target).await.unwrap().unwrap_err();
    assert!(why.contains("several") && why.contains("busy_peer") && why.contains(&id), "{why}");
    // GOOFI_SESSION breaks the tie — and one naming NOTHING is refused by pointing at
    // the listing.
    std::env::set_var("GOOFI_SESSION", "no_such_goofi");
    let why = tokio::task::spawn_blocking(client::resolve_target).await.unwrap().unwrap_err();
    assert!(why.contains("no_such_goofi") && why.contains("session list"), "{why}");
    std::env::set_var("GOOFI_SESSION", &id);
    let rows = tokio::task::spawn_blocking(client::list).await.unwrap();
    assert!(rows.iter().any(|(s, _)| s.id == id), "{rows:?}");
    let target = tokio::task::spawn_blocking(client::resolve_target).await.unwrap().unwrap();
    assert_eq!(target.id, id);
    home::remove_session("busy_peer");
    drop(mute);

    // Every phrase is reachable through the real door: `--help` on each resolves and answers.
    let ops: serde_json::Value = serde_json::from_str(&ok(&url, "default", "op list")).unwrap();
    let all: Vec<String> = ops["ops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["op"].as_str().unwrap().to_string())
        .collect();
    assert!(all.len() > 40, "the whole registry rides the index: {}", all.len());
    for phrase in &all {
        let text = ok(&url, "default", &format!("{phrase} --help"));
        assert!(text.contains(phrase) && text.contains("answers:"),
                "`{phrase} --help` explains itself, result shape included: {text}");
    }
    // The reserved client set is pinned AS the list, and help teaches the door words.
    assert_eq!(
        goofi_bridge::ops::RESERVED,
        ["serve", "help", "session list", "agent term", "plugin", "completions"]
    );
    let top = ok(&url, "default", "help");
    assert!(top.contains("session list") && top.contains("node"), "{top}");
    let group = ok(&url, "default", "help node");
    assert!(group.contains("node param edit"), "a group listing: {group}");
    let err = client::exec(&url, &lines(&["frobnicate"]), None).unwrap_err();
    assert!(err.contains("unknown op"), "{err}");

    // Completion is a vocabulary read: every candidate is `word<TAB>doc`, and the walk follows
    // the phrase tree — groups with their own doc line, then an op's flags.
    let complete = |line: &str| ok(&url, "default", &format!("op complete --line '{line}'"));
    let top = complete("");
    assert!(top.lines().any(|l| l.starts_with("node\t") && l.contains("one node instance")),
            "a group word completes with its own doc: {top}");
    let sub = complete("node ");
    assert!(sub.lines().any(|l| l.starts_with("param\t")) && !sub.contains("session"),
            "inside a group only its children answer: {sub}");
    let flags = complete("node add --");
    assert!(flags.contains("--type\t<string>, required") && flags.contains("--pos\t<float2>"),
            "an op completes its flags, typed and marked: {flags}");
    assert!(complete("node ad").lines().all(|l| l.starts_with("ad")),
            "a partial word filters");

    // A param edit lands on the graph, and a multi-line batch is ONE step in ITS actor's stack.
    let born: serde_json::Value =
        serde_json::from_str(&ok(&url, "shell_a", "node add --type Oscillator --name osc")).unwrap();
    let uid = born["uid"].as_str().unwrap().to_string();
    ok(&url, "shell_a", &format!("node param edit {uid} oscillator/sfreq --value 99"));
    assert_eq!(
        g.doc()["nodes"][&uid]["params"]["oscillator"]["sfreq"]["value"], 99.0,
        "the edit reached the graph"
    );
    // …and completion turns LIVE: a `uid` position offers the patch's own nodes, by the NAME a
    // human types, with the uid riding in the doc column.
    assert!(ok(&url, "default", "op complete --line 'node edit '")
                .contains(&format!("osc\t(signal:Oscillator) {uid}")),
            "a uid position completes from the graph");
    // A node reference is the uid OR the unique name — resolved server-side, in the one
    // namespace nd() reads, so every transport takes both and the endpoint's node half follows.
    ok(&url, "names", "node param edit osc oscillator/sfreq --value 55");
    assert_eq!(
        g.doc()["nodes"][&uid]["params"]["oscillator"]["sfreq"]["value"], 55.0,
        "the name reached the same node"
    );
    let why = client::exec(&url, &lines(&["node state nosuch"]), None).unwrap_err();
    assert!(why.contains("names no node"), "{why}");
    let batch = lines(&[
        "node add --type Buffer --name win",
        "node add --type Buffer --name sink",
    ]);
    assert_eq!(client::exec(&url, &batch, Some("shell_a")).unwrap().len(), 2);
    // Both halves of a wire spelled by name, made and taken back — the round-trip proves both
    // endpoint spellings resolve to the same wire.
    ok(&url, "names", "link add osc/out win/data");
    let gone: serde_json::Value =
        serde_json::from_str(&ok(&url, "names", "link remove osc/out win/data")).unwrap();
    assert_eq!(gone["removed"], true, "the named wire was there to remove");
    // Another actor's undo takes back ITS work, never shell_a's; bare shells share `default`.
    let d: serde_json::Value = serde_json::from_str(&ok(&url, "default", "node add --type Buffer")).unwrap();
    ok(&url, "default", "undo");
    assert!(g.doc()["nodes"][d["uid"].as_str().unwrap()].is_null(), "default undid its own");
    assert_eq!(g.doc()["nodes"].as_object().unwrap().len(), 3, "shell_a's three still stand");
    ok(&url, "shell_a", "undo");
    assert_eq!(
        g.doc()["nodes"].as_object().unwrap().len(), 1,
        "one undo took back the whole batch, in shell_a's stack"
    );

    // The raw read round-trips: the entry's rendered form IS the NPY bytes, ready for a pipe.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let npy = loop {
        let entries =
            client::exec(&url, &lines(&[&format!("node snapshot {uid}/out")]), None).unwrap();
        let bytes = client::rendered(&entries[0]);
        if bytes.starts_with(b"\x93NUMPY") {
            break bytes;
        }
        assert!(std::time::Instant::now() < deadline, "no frame ever reached the raw read");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let hlen = u16::from_le_bytes([npy[8], npy[9]]) as usize;
    assert!((npy.len() - 10 - hlen).is_multiple_of(4) && npy.len() > 10 + hlen, "a whole f32 payload");

    let _ = std::fs::remove_dir_all(&tmp);
}
