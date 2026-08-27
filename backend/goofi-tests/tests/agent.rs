//! An agent inside the patch: `/mcp` is the vocabulary it drives goofi with, and `/term/<id>` is the
//! PTY of a harness goofi launched itself.

use std::ffi::OsString;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::ops::REGISTRY;
use goofi_bridge::{term, AppState};
use goofi_tests::{host, http, Client, Goofi};
use serde_json::{json, Value};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

async fn start_server() -> (Goofi, String, AppState) {
    let g = Goofi::new();
    // The `_`-test agents are CONFIG entries now, written once into the test-scoped home the
    // fixture minted — never advertised, exactly as a user's own test entry would be.
    static CONFIG: std::sync::Once = std::sync::Once::new();
    CONFIG.call_once(|| {
        let at = goofi_core::home::config_file();
        let _ = std::fs::create_dir_all(at.parent().unwrap());
        std::fs::write(at, concat!(
            "[[agents]]\nname = \"_sh\"\ncommand = \"sh\"\n\n",
            // One that reports the SIGTERM and refuses to leave. The loop matters: a bare
            // `sleep` is a child of the same group and would die of the group signal.
            "[[agents]]\nname = \"_deaf\"\n",
            "command = \"trap 'echo GOT-TERM' TERM; while :; do echo armed; sleep 0.2; done\"\n\n",
            "[[agents]]\nname = \"visible_probe\"\ncommand = \"echo hi\"\n\n",
            "[[agents]]\nname = \"_gone\"\ncommand = \"definitely-not-a-cmd-4712\"\n",
        )).expect("the test config");
    });
    let addr = host(&g.serve().await).to_string();
    let state = g.state.clone();
    (g, addr, state)
}

/// One JSON-RPC request to an MCP address.
async fn rpc(addr: &str, path: &str, id: i64, method: &str, params: Value) -> Value {
    let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string();
    let headers = "Content-Type: application/json\r\n\
                   Accept: application/json, text/event-stream\r\n\
                   MCP-Protocol-Version: 2025-06-18\r\n";
    let (status, _, raw) = http(addr, "POST", path, headers, body.as_bytes()).await;
    assert_eq!(status, 200, "{method} answered {status}");
    let reply: Value = serde_json::from_slice(&raw).expect("a JSON-RPC reply");
    assert_eq!(reply["jsonrpc"], "2.0");
    assert_eq!(reply["id"].as_i64(), Some(id), "reply carried another caller's id: {reply}");
    reply
}

/// A `goofi_exec` call with one or more command LINES, answering `(rendered text, is_error)`.
async fn exec(addr: &str, path: &str, id: i64, commands: &[&str]) -> (String, bool) {
    let args = json!({ "name": "goofi_exec", "arguments": { "commands": commands } });
    let r = rpc(addr, path, id, "tools/call", args).await;
    let result = &r["result"];
    (result["content"][0]["text"].as_str().unwrap_or_default().to_string(),
     result["isError"] == json!(true))
}

/// One command that must succeed, answering ITS result — the tool answers one shape whatever
/// the count, the list of results, so this unwraps the one element.
async fn ok_exec(addr: &str, id: i64, command: &str) -> String {
    let (text, err) = exec(addr, "/mcp", id, &[command]).await;
    assert!(!err, "`{command}` failed: {text}");
    let list: Value = serde_json::from_str(&text).expect("the reply is the results list");
    let one = list.as_array().filter(|l| l.len() == 1).unwrap_or_else(|| panic!("{text}"));
    serde_json::to_string_pretty(&one[0]).expect("two strings")
}


async fn tools(addr: &str) -> Vec<Value> {
    rpc(addr, "/mcp", 1, "tools/list", json!({})).await["result"]["tools"]
        .as_array().expect("a tools array").clone()
}

#[tokio::test]
async fn the_one_tool_speaks_the_whole_op_vocabulary_in_command_lines() {
    let (g, addr, _s) = start_server().await;
    let served = tools(&addr).await;
    assert_eq!(served.len(), 1, "ONE tool, whatever the registry holds: {served:?}");
    let t = &served[0];
    assert_eq!(t["name"], json!("goofi_exec"));
    assert!(t["description"].as_str().is_some_and(|d| d.contains("op list")),
            "the description points a model at the index: {t}");
    // Even a one-argument tool advertises an object schema, or a client rejects it.
    assert_eq!(t["inputSchema"]["type"], json!("object"));
    assert_eq!(t["inputSchema"]["required"], json!(["commands"]));

    // The index is an op like any other, and it carries what a caller derives a client from.
    let ops: Value = serde_json::from_str(&ok_exec(&addr, 2, "op list").await).unwrap();
    let ops = ops["ops"].as_array().expect("an ops list");
    assert_eq!(ops.len(), REGISTRY.len(), "every registry row is in the index");
    let add = ops.iter().find(|o| o["op"] == json!("node add")).unwrap();
    assert_eq!(add["kind"], json!("write"));
    assert!(add["args"].as_str().is_some_and(|a| a.contains("type:string!")),
            "the args schema rides the index: {add}");

    // One command executes directly: flags typed by the schema — a NEGATIVE float2 value, a
    // chosen name, and a `json` flag quoted as bash would quote it.
    let born = ok_exec(&addr, 3, "node add --type Oscillator --pos -100,-50 --name osc").await;
    let born: Value = serde_json::from_str(&born).expect("the rendered reply is the op's JSON");
    let uid = born["uid"].as_str().expect("a uid").to_string();
    assert_eq!(born["name"], json!("osc"));
    // …and both 2b positionals at once: the uid, then the joined `group/param` address.
    let line = format!("node param edit {uid} oscillator/frequency --value 7.5");
    let edited = ok_exec(&addr, 4, &line).await;
    assert!(edited.contains("7.5"), "the param came back as stored: {edited}");
    let patch = ok_exec(&addr, 5, "nodes inspect").await;
    assert!(patch.contains(&uid), "the diagram rides the result's `text`: {patch}");

    // Idempotence still SAYS which of the two happened.
    let missing = ok_exec(&addr, 6, "node remove --node aaaaaaaaaaaa").await;
    assert!(missing.contains("\"removed\": false"), "a no-op success says so: {missing}");
    let real = ok_exec(&addr, 7, &format!("node remove --node {uid}")).await;
    assert!(real.contains("\"removed\": true"), "a real delete is distinguishable: {real}");

    // An Effect runs when it is the ONLY command — undo brings the node back.
    let undone: Value = serde_json::from_str(&ok_exec(&addr, 8, "undo").await).unwrap();
    assert_eq!(undone["changed"], json!(true));
    assert!(g.call("session state", json!({}))["nodes"].as_object().is_some_and(|n| n.len() == 1),
            "the undo really landed");

    // A refusal teaches: the op index, the op's own flags, the required set.
    let (text, err) = exec(&addr, "/mcp", 9, &["frobnicate --hard"]).await;
    assert!(err && text.contains("unknown op") && text.contains("op list"), "{text}");
    let (text, err) = exec(&addr, "/mcp", 10, &["node add --type Oscillator --sideways 3"]).await;
    assert!(err && text.contains("--sideways") && text.contains("--pos"), "{text}");
    let (text, err) = exec(&addr, "/mcp", 11, &["node add"]).await;
    assert!(err && text.contains("--type") && text.contains("required"), "{text}");
    let (text, err) = exec(&addr, "/mcp", 12, &["undo --hard"]).await;
    assert!(err && text.contains("takes no arguments"), "{text}");
    let (text, err) = exec(&addr, "/mcp", 13, &["node add --type Oscillator --type Buffer"]).await;
    assert!(err && text.contains("twice"), "{text}");
    let (text, err) = exec(&addr, "/mcp", 14, &["node add --type"]).await;
    assert!(err && text.contains("needs a value"), "{text}");

    // …and each spelling the schema promises parses: `--flag=value`, the bool's `--no-` form on
    // a declared bool only, `any` as JSON-or-bare-string, and a variadic positional list.
    let born = ok_exec(&addr, 15, "node add --type=Oscillator --name inline_osc").await;
    let born: Value = serde_json::from_str(&born).unwrap();
    assert_eq!(born["name"], json!("inline_osc"));
    let bare = ok_exec(&addr, 16, &format!("node state {} --no-params", born["uid"].as_str().unwrap())).await;
    assert!(!bare.contains("params:"), "the bool's negative spelling gates the section: {bare}");
    let (text, err) = exec(&addr, "/mcp", 17, &["node add --type Oscillator --no-name x"]).await;
    assert!(err && text.contains("--no-name"), "`--no-` binds only to a declared bool: {text}");
    ok_exec(&addr, 18, "global add gain --type float --value 2.5").await;
    let tag: Value =
        serde_json::from_str(&ok_exec(&addr, 19, "global add tag --type string --value hello").await)
            .unwrap();
    assert_eq!(tag["value"], json!("hello"), "`any` falls back to the bare string");
    let second = ok_exec(&addr, 20, "node add --type Buffer").await;
    let second: Value = serde_json::from_str(&second).unwrap();
    let grouped = ok_exec(&addr, 21, &format!("nodes group {} {} --pos 0,0",
        born["uid"].as_str().unwrap(), second["uid"].as_str().unwrap())).await;
    let grouped: Value = serde_json::from_str(&grouped).unwrap();
    let doc = g.call("session state", json!({}));
    for uid in [born["uid"].as_str().unwrap(), second["uid"].as_str().unwrap()] {
        assert_eq!(doc["nodes"][uid]["scope"], grouped["inst_id"],
                   "the variadic positional took BOTH words into the new scope: {doc}");
    }
}

#[tokio::test]
async fn several_commands_are_one_batch_and_a_refused_step_takes_the_whole_batch_back() {
    let (g, addr, _s) = start_server().await;
    let nodes = |g: &Goofi| g.call("session state", json!({}))["nodes"]
        .as_object().map(|n| n.len()).unwrap_or(0);

    // A tab watching from the start: the batch must reach it as ONE doc_patch carrying both
    // births — a per-step broadcast would land a patch holding only the first.
    let (mut tab, _hello) = Client::connect(&g.serve().await).await;
    let (text, err) =
        exec(&addr, "/mcp", 1, &["node add --type Oscillator", "node add --type Buffer"]).await;
    assert!(!err, "{text}");
    let results: Value = serde_json::from_str(&text).expect("the batch answers a JSON list");
    assert_eq!(results.as_array().map(|a| a.len()), Some(2), "each step's result, in order: {text}");
    assert_eq!(nodes(&g), 2);
    let patch = tab.event("doc_patch").await;
    for step in results.as_array().unwrap() {
        let uid = step["uid"].as_str().expect("a birth reply");
        assert!(patch["patch"]["nodes"].get(uid).is_some(),
                "the ONE settle patch carries {uid}: {patch}");
    }

    // ONE undo step covers the whole batch.
    ok_exec(&addr, 2, "undo").await;
    assert_eq!(nodes(&g), 0, "one undo took back both steps");

    // A step that is not an undoable write refuses the batch BEFORE anything lands.
    let (text, err) = exec(&addr, "/mcp", 3, &["node add --type Oscillator", "session new"]).await;
    assert!(err && text.contains("not a step"), "{text}");
    assert_eq!(nodes(&g), 0, "a refused batch left nothing behind");

    // A READ rides a batch, its reply in order beside the write's.
    let (text, err) = exec(&addr, "/mcp", 5, &["node add --type Oscillator", "nodes inspect"]).await;
    assert!(!err, "{text}");
    let results: Value = serde_json::from_str(&text).unwrap();
    assert!(results[1]["text"].as_str().is_some_and(|t| t.contains("Oscillator")), "{text}");
    ok_exec(&addr, 6, "undo").await;

    // The same Effect alone is legal — the total surface includes the lifecycle.
    let loaded = ok_exec(&addr, 4, "session new").await;
    assert!(loaded.contains("\"ok\": true"), "{loaded}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_agents_drive_one_server_at_once_and_read_their_work_back_out_of_it() {
    let (_g, addr, _s) = start_server().await;
    let init = rpc(&addr, "/mcp", 1, "initialize", json!({
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": { "name": "test", "version": "0" } })).await;
    assert!(init["result"]["capabilities"]["tools"].is_object(), "tools are advertised: {init}");
    assert!(init["result"]["serverInfo"]["name"].as_str().is_some_and(|n| n.contains("goofi")));
    assert!(init["result"]["instructions"].is_null(), "{init}");

    let uid_of = |t: String| serde_json::from_str::<Value>(&t).unwrap()["uid"].as_str().unwrap().to_string();
    let spawn = |ty: &'static str, base: i64| {
        let addr = addr.clone();
        tokio::spawn(async move {
            let mut uids = Vec::new();
            for i in 0..4 {
                let line = format!("node add --type {ty}");
                uids.push(uid_of(ok_exec(&addr, base + i, &line).await));
            }
            uids
        })
    };
    let (a, b) = (spawn("Oscillator", 100), spawn("Buffer", 200));
    let mut uids = a.await.unwrap();
    uids.extend(b.await.unwrap());
    assert_eq!(uids.iter().collect::<std::collections::HashSet<_>>().len(), 8,
               "two clients minted a colliding uid: {uids:?}");
    let patch = ok_exec(&addr, 300, "nodes inspect").await;
    for uid in &uids {
        assert!(patch.contains(uid), "{uid} is missing from the shared patch:\n{patch}");
    }

    let (text, err) = exec(&addr, "/mcp", 400, &["node add --type NoSuchNodeType"]).await;
    assert!(err, "an unknown type was not refused: {text}");
    assert!(text.contains("NoSuchNodeType"), "the refusal names what it refused: {text}");

    for asked in ["2024-11-05", "2025-06-18", "2025-11-25"] {
        let r = rpc(&addr, "/mcp", 500, "initialize", json!({ "protocolVersion": asked })).await;
        assert_eq!(r["result"]["protocolVersion"], json!(asked), "a supported ask is granted: {r}");
    }
    for asked in ["2026-07-28", "not-a-version"] {
        let r = rpc(&addr, "/mcp", 501, "initialize", json!({ "protocolVersion": asked })).await;
        assert_eq!(r["result"]["protocolVersion"], json!("2025-11-25"), "`{asked}` came back: {r}");
    }
    let r = rpc(&addr, "/mcp", 502, "initialize", json!({})).await;
    assert_eq!(r["result"]["protocolVersion"], json!("2025-06-18"), "the documented default: {r}");
}

async fn recv_text_by(ws: &mut Ws, deadline: tokio::time::Instant) -> Value {
    loop {
        let msg = tokio::time::timeout_at(deadline, ws.next())
            .await.expect("recv timed out").expect("stream ended").expect("ws error");
        if let Message::Text(t) = msg {
            return serde_json::from_str(t.as_str()).expect("json");
        }
    }
}

async fn recv_text(ws: &mut Ws) -> Value {
    recv_text_by(ws, tokio::time::Instant::now() + Duration::from_secs(5)).await
}

/// Send an RPC on `/control` and return its reply, skipping interleaved broadcast events.
async fn call(ws: &mut Ws, id: i64, op: &str, payload: Value) -> Value {
    ws.send(Message::Text(json!({ "id": id, "op": op, "payload": payload }).to_string().into()))
        .await.unwrap();
    loop {
        let v = recv_text(ws).await;
        if v["id"] == json!(id) {
            assert!(v["error"].is_null(), "{op} failed: {v}");
            return v["result"].clone();
        }
    }
}

/// A control socket past its `hello`, and one spawned `_sh` harness with its PTY attached.
async fn harness(addr: &str, kind: &str) -> (Ws, String, Ws) {
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "agent start", json!({ "name": kind })).await["instance_id"]
        .as_str().expect("a spawn answers an instance id").to_string();
    let (term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    (ctl, id, term)
}

/// Read `/term` until `want` has been seen in the PTY bytes.
async fn read_until(ws: &mut Ws, want: &str) -> String {
    let mut seen = String::new();
    let mut answered = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !seen.contains(want) {
        // A real terminal answers ConPTY's cursor-position query, and the child stays BLOCKED until one does.
        if !answered && seen.contains('\u{1b}') && seen.contains("[6n") {
            answered = true;
            ws.send(Message::Binary(b"\x1b[1;1R".to_vec().into())).await.unwrap();
        }
        let msg = match tokio::time::timeout_at(deadline, ws.next()).await {
            Err(_) => panic!("{want:?} never arrived; the PTY said {seen:?}"),
            Ok(Some(Ok(m))) => m,
            Ok(end) => panic!("the socket ended before {want:?} ({end:?}); the PTY said {seen:?}"),
        };
        match msg {
            Message::Binary(b) => seen.push_str(&String::from_utf8_lossy(&b)),
            Message::Text(t) => seen.push_str(t.as_str()),
            _ => {}
        }
    }
    seen
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_harness_spawns_carries_bytes_both_ways_and_is_reaped_with_the_code_it_chose() {
    let (g, addr, state) = start_server().await;
    let (mut ctl, id, mut term) = harness(&addr, "_sh").await;

    // `6*7` is echoed input; only `42` can come from the child having run.
    term.send(Message::Binary(b"echo $((6*7))\n".to_vec().into())).await.unwrap();
    read_until(&mut term, "42").await;

    // An edit under the shell's own actor, so the exit below can prove the stack dies with it.
    g.client(&term::actor_of(&id)).call("node add", json!({ "type": "Oscillator" }));

    // `TAIL''MARK` is what the terminal echoes, so `TAILMARK` can only come from the child.
    term.send(Message::Binary(
        b"i=0; while [ $i -lt 400 ]; do i=$((i+1)); echo L$i; done; echo TAIL''MARK; exit 7\n".to_vec().into()))
        .await.unwrap();
    // A second view attaches MID-BURST: replay meets live exactly once — a snapshot taken after
    // the subscribe doubles a line, a gap between them loses one.
    let (mut mid, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    let seen = read_until(&mut term, "exit_code").await;
    assert!(seen.contains("L400"), "the burst was truncated");
    assert!(seen.contains("TAILMARK"), "the child's last line was dropped");
    assert!(seen.contains("\"exit_code\":7"), "the exit frame carries the code the child chose");
    let seen = read_until(&mut mid, "exit_code").await;
    assert_eq!(seen.matches("L200\r").count(), 1, "replay and live overlap or gap: {}", seen.len());

    let (mut late, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    assert!(read_until(&mut late, "exit_code").await.contains("\"exit_code\":7"),
            "a late attach was told nothing");

    // A shell that ends on its OWN loses its undo stack too — the reaper owns the drop, and its
    // broadcast comes after it, so the exited event is the settled signal.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let ev = recv_text_by(&mut ctl, deadline).await;
        if ev["event"] == json!("harness_changed")
            && ev["payload"]["instances"][0]["state"] == json!("exited")
        {
            break;
        }
    }
    assert_eq!(g.client(&term::actor_of(&id)).call("undo", json!({}))["changed"], json!(false),
               "the self-exited shell's stack has something to undo");

    let roster = call(&mut ctl, 2, "agent list", json!({})).await;
    let inst = &roster["instances"][0];
    assert_eq!((&inst["id"], &inst["harness"], &inst["state"], &inst["exit_code"]),
               (&json!(id), &json!("_sh"), &json!("exited"), &json!(7)), "{roster}");
    state.release_mount();
}

#[tokio::test]
async fn a_harness_runs_unwatched_and_its_roster_survives_a_reconnect() {
    // On Windows ConPTY blocks the child on a cursor query, so with no viewer goofi must answer it.
    let (_g, addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    let hello = recv_text(&mut ctl).await;
    assert_eq!(hello["payload"]["harnesses"]["instances"], json!([]), "a fresh backend: {hello}");

    let id = call(&mut ctl, 1, "agent start", json!({ "name": "_deaf" })).await["instance_id"]
        .as_str().unwrap().to_string();
    tokio::time::sleep(Duration::from_millis(600)).await; // deliberately no `/term` socket yet
    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    read_until(&mut term, "armed").await;

    drop(ctl);
    let (mut later, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    let hello = recv_text(&mut later).await;
    let instances = &hello["payload"]["harnesses"]["instances"];
    assert_eq!((&instances[0]["id"], &instances[0]["state"]), (&json!(id), &json!("running")),
               "the roster was not seeded: {hello}");
    let agents = &hello["payload"]["harnesses"]["agents"];
    assert!(agents.as_array().is_some_and(|a| a.iter()
                .any(|e| e["name"] == "visible_probe" && e["command"] == "echo hi")),
            "…and the CONFIG list rides the same shape, so a joining tab offers the buttons: {hello}");
    assert!(!agents.to_string().contains("_sh"), "a `_` test entry is never advertised: {agents}");
    assert_eq!(hello["payload"]["unsaved_changes"], json!(false), "a spawn dirtied the patch");

    call(&mut later, 2, "agent stop", json!({ "instance": id })).await;
    state.release_mount();
}

/// Turn the line discipline's echo OFF, then have the child report where it is and what it holds.
/// The readiness marker is spelled `REA''DY`, which only the child prints joined.
async fn report(term: &mut Ws) -> String {
    term.send(Message::Binary(b"stty -echo; echo REA''DY\n".to_vec().into())).await.unwrap();
    read_until(term, "READY").await;
    term.send(Message::Binary(
        b"printf 'CWD[%s]TERM[%s]COLOR[%s]LC[%s]HOME[%s]KEPT[%s]E''ND\\n' \"$(pwd -P)\" \
          \"$TERM\" \"$COLORTERM\" \"$LC_ALL\" \"$HOME\" \"$STATED_BY_THE_TEST\"\n".to_vec().into()))
        .await.unwrap();
    read_until(term, "END").await
}

/// One `KEY[value]` field out of that report — the LAST, since ConPTY echoes the `printf` line first.
fn field<'a>(seen: &'a str, key: &str) -> &'a str {
    let at = seen.rfind(key).unwrap_or_else(|| panic!("no {key} in {seen:?}"));
    seen[at + key.len()..].split_once(']').expect("a closed field").0
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_harness_runs_in_the_patchs_workspace_with_the_terminal_contract_overlaid() {
    let (_g, addr, state) = start_server().await;
    let (mut ctl, id, mut term) = harness(&addr, "_sh").await;
    let seen = report(&mut term).await;

    // An MSYS `sh` maps `%TEMP%` onto `/tmp`, so only the tail goofi minted is comparable.
    let mount = state.mount();
    let mut tail = mount.components().rev().map(|c| c.as_os_str().to_string_lossy().into_owned());
    let (workspace, nonce) = (tail.next().unwrap(), tail.next().unwrap());
    let want = format!("{nonce}/{workspace}");
    assert!(field(&seen, "CWD[").ends_with(&want), "the harness is not in the workspace: {seen:?}");

    // Stated, not read from the suite's environment: an ambient value passes with no overlay at all.
    let env: Vec<(OsString, OsString)> = vec![
        ("TERM".into(), "dumb".into()), ("COLORTERM".into(), "no".into()),
        ("LANG".into(), "C".into()), ("STATED_BY_THE_TEST".into(), "kept".into())];
    let stated = state.harnesses
        .spawn("_sh", &state.mount(), "http://127.0.0.1:1", &env, state.events.clone(),
               state.history.clone())
        .expect("a spawn with a stated parent environment");
    let (mut term, _) = connect_async(format!("ws://{addr}/term/{stated}")).await.unwrap();
    let seen = report(&mut term).await;
    assert_eq!(field(&seen, "TERM["), "xterm-256color", "a dumb TERM was not overlaid: {seen:?}");
    assert_eq!(field(&seen, "COLOR["), "truecolor", "{seen:?}");
    assert_eq!(field(&seen, "LC["), "C.UTF-8", "a parent with no UTF-8 locale gets one: {seen:?}");
    assert_eq!(field(&seen, "KEPT["), "kept", "the stated parent never reached the child: {seen:?}");
    // Asserted by what it must not be: an MSYS shell reports `/c/Users/x` for `C:\Users\x`.
    let home = field(&seen, "HOME[");
    assert!(!home.is_empty() && !home.contains(&nonce), "HOME was redirected: {seen:?}");

    // Reap both: a leaked PTY child outlives the suite and corrupts every later measurement.
    call(&mut ctl, 2, "agent stop", json!({ "instance": id })).await;
    state.harnesses.stop(&stated).unwrap();
    state.release_mount();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_asks_before_it_insists_and_reaps_a_harness_that_will_not_go() {
    // Waiting for `armed` first matters: a signal delivered before the trap proves nothing.
    let (_g, addr, state) = start_server().await;
    let (mut ctl, id, mut term) = harness(&addr, "_deaf").await;
    read_until(&mut term, "armed").await;
    call(&mut ctl, 2, "agent stop", json!({ "instance": id })).await;

    // Read BEFORE waiting on the trap: the ask is synchronous, and a slow echo can outlast
    // the grace — after which `stopping` has already moved on.
    let roster = call(&mut ctl, 3, "agent list", json!({})).await;
    assert_eq!(roster["instances"][0]["state"], json!("stopping"), "{roster}");

    // The graceful ask is only OBSERVABLE where signals are; Windows refuses `taskkill` without `/F`.
    #[cfg(unix)]
    read_until(&mut term, "GOT-TERM").await;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let ev = recv_text_by(&mut ctl, deadline).await;
        let inst = ev["payload"]["instances"][0].clone();
        if ev["event"] == json!("harness_changed") && inst["state"] == json!("exited") {
            assert_eq!(inst["id"], json!(id), "another instance exited: {ev}");
            assert!(inst["exit_code"].is_number(), "no exit code captured: {ev}");
            break;
        }
    }

    // A command that cannot launch fails ON its PTY — and the tail replays it to a socket
    // that arrives after the words, the race the panel always loses to a fast shell.
    let gone = call(&mut ctl, 4, "agent start", json!({ "name": "_gone" })).await["instance_id"]
        .as_str().expect("the spawn succeeds; the SHELL is what fails").to_string();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    for probe in 100.. {
        let roster = call(&mut ctl, probe, "agent list", json!({})).await;
        let done = roster["instances"].as_array().unwrap().iter()
            .any(|i| i["id"] == json!(gone) && i["state"] == json!("exited"));
        if done {
            break;
        }
        assert!(tokio::time::Instant::now() < deadline, "the shell never exited: {roster}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let (mut late, _) = connect_async(format!("ws://{addr}/term/{gone}")).await.unwrap();
    let said = read_until(&mut late, "not found").await;
    assert!(said.contains("definitely-not-a-cmd-4712"), "the failure names the command: {said}");
    call(&mut ctl, 5, "agent stop", json!({ "instance": gone })).await;
    state.release_mount();
}

/// Read `/term` text frames until an authoritative-size one arrives.
async fn recv_size(ws: &mut Ws) -> (u64, u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let v = recv_text_by(ws, deadline).await;
        if v["op"] == json!("size") {
            return (v["cols"].as_u64().expect("cols"), v["rows"].as_u64().expect("rows"));
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn several_views_of_one_terminal_agree_on_one_size() {
    // A PTY has one window: the last view to speak wins, and every view is told the answer.
    let (_g, addr, state) = start_server().await;
    let (mut ctl, id, mut a) = harness(&addr, "_sh").await;
    let url = format!("ws://{addr}/term/{id}");
    let (mut b, _) = connect_async(&url).await.unwrap();

    let resize = |cols: u16, rows: u16| {
        Message::Text(json!({ "op": "resize", "cols": cols, "rows": rows }).to_string().into())
    };
    a.send(resize(100, 30)).await.unwrap();
    assert_eq!(recv_size(&mut a).await, (100, 30), "the view that asked is told the answer");
    assert_eq!(recv_size(&mut b).await, (100, 30), "a view that did not ask is told too");
    b.send(resize(80, 24)).await.unwrap();
    assert_eq!(recv_size(&mut b).await, (80, 24), "the last writer wins");
    assert_eq!(recv_size(&mut a).await, (80, 24), "…for every view");

    // A view whose panel unmounted keeps its socket but stops speaking for the terminal: zero size.
    b.send(resize(0, 0)).await.unwrap();
    assert_eq!(recv_size(&mut a).await, (100, 30), "a retracted view hands the size back");
    assert_eq!(recv_size(&mut b).await, (100, 30), "…and hears the answer it caused");

    let (mut c, _) = connect_async(&url).await.unwrap();
    assert_eq!(recv_size(&mut c).await, (100, 30), "a late view starts at the current size");
    c.send(resize(120, 40)).await.unwrap();
    assert_eq!(recv_size(&mut a).await, (120, 40));
    drop(c);
    assert_eq!(recv_size(&mut a).await, (100, 30), "a view that left hands the size back");

    // The arbitration reaches the KERNEL: `30 100` is nowhere in the input.
    a.send(Message::Binary(b"stty size\n".to_vec().into())).await.unwrap();
    read_until(&mut a, "30 100").await;

    call(&mut ctl, 2, "agent stop", json!({ "instance": id })).await;
    state.release_mount();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_agent_carries_its_identity_in_its_environment_and_dies_with_the_patch() {
    let (g, addr, state) = start_server().await;
    let (mut ctl, id, mut term) = harness(&addr, "_sh").await;

    let mount = state.mount();
    let agents = std::fs::read_to_string(mount.join("AGENTS.md"))
        .expect("the workspace was seeded with the orientation");
    assert!(agents.contains("goofi-pipe is a live"), "the orientation is the real one: {agents}");
    assert_eq!(std::fs::read_to_string(mount.join("CLAUDE.md")).unwrap(), "@AGENTS.md\n");

    // Identity travels in the ENVIRONMENT, and the shell itself answers what it was handed: the
    // server's id, its own undo actor, and a `goofi` that IS this server's binary — the shim,
    // first on PATH, laid in the instance's own config dir.
    term.send(Message::Binary(
        b"stty -echo; printf 'S[%s]A[%s]G[%s]EN''D\n' \
          \"$GOOFI_SESSION\" \"$GOOFI_ACTOR\" \"$(command -v goofi)\"\n".to_vec().into()))
        .await.unwrap();
    let said = read_until(&mut term, "END").await;
    assert!(said.contains(&format!("S[{}]", &*state.instance_id)), "{said}");
    assert!(said.contains(&format!("A[{}]", term::actor_of(&id))), "{said}");
    // A login profile may rebuild PATH over the shim, so the pin is the word's TARGET, not
    // its seat.
    let word = std::path::PathBuf::from(field(&said, "G["));
    let me = std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
    assert_eq!(std::fs::canonicalize(&word).unwrap_or_default(), me, "{said}");
    let shim = term::config_dir(&mount, &id).join("goofi");
    assert_eq!(std::fs::read_link(&shim).unwrap(), std::env::current_exe().unwrap(),
               "the shim IS this very binary");

    // A stack's lifetime follows its actor: the stopped shell keeps its edits, loses its undo.
    // The reaper drops the stack and THEN broadcasts, so the exited event is the settled signal.
    let actor = term::actor_of(&id);
    g.client(&actor).call("node add", json!({ "type": "Oscillator" }));
    call(&mut ctl, 3, "agent stop", json!({ "instance": id })).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let ev = recv_text_by(&mut ctl, deadline).await;
        if ev["event"] == json!("harness_changed")
            && ev["payload"]["instances"][0]["state"] == json!("exited")
        {
            break;
        }
    }
    assert_eq!(g.client(&actor).call("undo", json!({}))["changed"], json!(false),
               "the dropped stack has nothing to undo");
    assert_eq!(g.call("session state", json!({}))["nodes"].as_object().map(|n| n.len()), Some(1),
               "…and the graph keeps what the agent built");

    // Replacing the patch reaps the running agents, and the central /mcp stays open.
    let (mut ctl2, _, mut second) = harness(&addr, "_sh").await;
    call(&mut ctl2, 4, "session new", json!({})).await;
    let roster = call(&mut ctl2, 5, "agent list", json!({})).await;
    assert_eq!(roster["instances"], json!([]), "the replaced patch's harnesses stayed: {roster}");
    // `read_until` itself is the pin: no exit frame within its deadline panics.
    read_until(&mut second, "exit_code").await;
    let (ok, err) = exec(&addr, "/mcp", 6, &["node add --type Buffer"]).await;
    assert!(!err, "replacing the patch closed the central endpoint too: {ok}");

    let (_ctl, _, mut left) = harness(&addr, "_sh").await;
    state.release_mount();
    read_until(&mut left, "exit_code").await; // no frame within its deadline panics
}
