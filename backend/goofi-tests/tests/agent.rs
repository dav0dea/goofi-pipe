//! An agent inside the patch: `/mcp` is the vocabulary it drives goofi with, and `/term/<id>` is the
//! PTY of a harness goofi launched itself.

use std::ffi::OsString;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::ops::{Surface, MCP_PREFIX, REGISTRY};
use goofi_bridge::{term, AppState};
use goofi_tests::{host, http, Goofi};
use serde_json::{json, Value};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

async fn start_server() -> (Goofi, String, AppState) {
    let g = Goofi::new();
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

/// A `tools/call`, answering `(rendered text, is_error)`.
async fn tool(addr: &str, path: &str, id: i64, name: &str, args: Value) -> (String, bool) {
    let r = rpc(addr, path, id, "tools/call", json!({ "name": name, "arguments": args })).await;
    let result = &r["result"];
    (result["content"][0]["text"].as_str().unwrap_or_default().to_string(),
     result["isError"] == json!(true))
}

/// A tool call that must succeed.
async fn ok_tool(addr: &str, id: i64, name: &str, args: Value) -> String {
    let (text, err) = tool(addr, "/mcp", id, name, args).await;
    assert!(!err, "{name} failed: {text}");
    text
}

async fn tools(addr: &str) -> Vec<Value> {
    rpc(addr, "/mcp", 1, "tools/list", json!({})).await["result"]["tools"]
        .as_array().expect("a tools array").clone()
}

#[tokio::test]
async fn the_served_tools_are_exactly_the_registrys_agent_surface_and_each_states_what_it_answers() {
    // The list is GENERATED from the op registry: adding an op without a surface decision fails here.
    let (_g, addr, _s) = start_server().await;
    let served = tools(&addr).await;
    let mut names: Vec<String> =
        served.iter().map(|t| t["name"].as_str().unwrap().to_string()).collect();
    let mut want: Vec<String> = REGISTRY.iter().filter(|o| o.surface == Surface::Mcp)
        .map(|o| o.name.to_string()).collect();
    names.sort();
    want.sort();
    assert_eq!(names, want);

    // Each exclusion replaces the patch the agent is working inside, undo history included.
    for off in ["load", "save", "serialize", "list_dir", "set_viewpoint"] {
        assert!(REGISTRY.iter().any(|o| o.name == off && o.surface == Surface::ControlOnly),
                "`{off}` is no longer a control-only row");
        assert!(!names.iter().any(|t| t == off), "`{off}` reached the agent surface");
    }

    for t in &served {
        // `mcp__goofi__<name>` past 64 characters makes a provider reject the WHOLE tool list.
        let name = t["name"].as_str().unwrap();
        assert!(name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "`{name}` is not [a-z0-9_]+");
        assert!(MCP_PREFIX.len() + name.len() <= 64,
                "`{MCP_PREFIX}{name}` is over the 64 characters a tool name may have");
        assert!(t["description"].as_str().is_some_and(|d| !d.is_empty()), "`{name}` is undescribed");
        // Even an op with no arguments advertises an object schema, or a client rejects the tool.
        assert_eq!(t["inputSchema"]["type"], json!("object"), "`{name}`");
    }

    // A required argument advertised as optional is one a model omits.
    let add = served.iter().find(|t| t["name"] == json!("add_node")).unwrap();
    assert_eq!(add["inputSchema"]["properties"]["type"]["type"], json!("string"));
    assert_eq!(add["inputSchema"]["properties"]["pos"]["type"], json!("array"));
    assert_eq!(add["inputSchema"]["required"], json!(["type"]));
    let list = served.iter().find(|t| t["name"] == json!("list_nodes")).unwrap();
    assert_eq!(list["inputSchema"]["required"], json!([]));

    let described = |name: &str| {
        served.iter().find(|t| t["name"] == json!(name)).unwrap()["description"]
            .as_str().unwrap().to_string()
    };
    ok_tool(&addr, 2, "add_node", json!({ "type": "Oscillator" })).await;
    let undone: Value = serde_json::from_str(&ok_tool(&addr, 3, "undo", json!({})).await).unwrap();
    assert!(undone["changed"].is_boolean(), "undo's `changed` is a bool: {undone}");
    assert!(described("undo").contains("changed: bool"), "{}", described("undo"));

    let missing = ok_tool(&addr, 4, "remove_node", json!({ "node": "aaaaaaaaaaaa" })).await;
    assert!(missing.contains("\"removed\": false"), "a no-op success has to SAY so: {missing}");
    let uid = ok_tool(&addr, 5, "add_node", json!({ "type": "Buffer" })).await;
    let uid: Value = serde_json::from_str(&uid).unwrap();
    let real = ok_tool(&addr, 6, "remove_node", json!({ "node": uid["uid"] })).await;
    assert!(real.contains("\"removed\": true"), "…and a real delete is distinguishable: {real}");
    let d = described("remove_node");
    assert!(d.contains("Idempotent") && d.contains("removed"), "{d}");
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
                uids.push(uid_of(ok_tool(&addr, base + i, "add_node", json!({ "type": ty })).await));
            }
            uids
        })
    };
    let (a, b) = (spawn("Oscillator", 100), spawn("Buffer", 200));
    let mut uids = a.await.unwrap();
    uids.extend(b.await.unwrap());
    assert_eq!(uids.iter().collect::<std::collections::HashSet<_>>().len(), 8,
               "two clients minted a colliding uid: {uids:?}");
    let patch = ok_tool(&addr, 300, "inspect_patch", json!({})).await;
    for uid in &uids {
        assert!(patch.contains(uid), "{uid} is missing from the shared patch:\n{patch}");
    }

    for (name, args, names) in [("add_node", json!({ "type": "NoSuchNodeType" }), "NoSuchNodeType"),
                                ("load", json!({}), "load")] {
        let (text, err) = tool(&addr, "/mcp", 400, name, args).await;
        assert!(err, "`{name}` was not refused: {text}");
        assert!(text.contains(names), "the refusal names what it refused: {text}");
    }

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
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": kind })).await["instance_id"]
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
    let (_g, addr, state) = start_server().await;
    let (mut ctl, id, mut term) = harness(&addr, "_sh").await;

    // `6*7` is echoed input; only `42` can come from the child having run.
    term.send(Message::Binary(b"echo $((6*7))\n".to_vec().into())).await.unwrap();
    read_until(&mut term, "42").await;

    // `TAIL''MARK` is what the terminal echoes, so `TAILMARK` can only come from the child.
    term.send(Message::Binary(
        b"i=0; while [ $i -lt 400 ]; do i=$((i+1)); echo L$i; done; echo TAIL''MARK; exit 7\n".to_vec().into()))
        .await.unwrap();
    let seen = read_until(&mut term, "exit_code").await;
    assert!(seen.contains("L400"), "the burst was truncated");
    assert!(seen.contains("TAILMARK"), "the child's last line was dropped");
    assert!(seen.contains("\"exit_code\":7"), "the exit frame carries the code the child chose");

    let (mut late, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    assert!(read_until(&mut late, "exit_code").await.contains("\"exit_code\":7"),
            "a late attach was told nothing");

    let roster = call(&mut ctl, 2, "list_harnesses", json!({})).await;
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

    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_deaf" })).await["instance_id"]
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
    assert!(hello["payload"]["harnesses"]["detected"].is_array(),
            "…and `detected` rides the same shape, so a joining tab can offer the launch buttons");
    assert_eq!(hello["payload"]["unsaved_changes"], json!(false), "a spawn dirtied the patch");

    call(&mut later, 2, "stop_harness", json!({ "instance": id })).await;
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
        .spawn("_sh", &state.mount(), "http://127.0.0.1:1", &env, state.events.clone())
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
    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;
    state.harnesses.stop(&stated).unwrap();
    state.release_mount();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_asks_before_it_insists_and_reaps_a_harness_that_will_not_go() {
    // Waiting for `armed` first matters: a signal delivered before the trap proves nothing.
    let (_g, addr, state) = start_server().await;
    let (mut ctl, id, mut term) = harness(&addr, "_deaf").await;
    read_until(&mut term, "armed").await;
    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;

    // The graceful ask is only OBSERVABLE where signals are; Windows refuses `taskkill` without `/F`.
    #[cfg(unix)]
    read_until(&mut term, "GOT-TERM").await;

    let roster = call(&mut ctl, 3, "list_harnesses", json!({})).await;
    assert_eq!(roster["instances"][0]["state"], json!("stopping"), "{roster}");

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

    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;
    state.release_mount();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_minted_address_serves_its_own_agent_and_dies_with_the_patch_that_spawned_it() {
    let (_g, addr, state) = start_server().await;
    let (mut ctl, id, mut term) = harness(&addr, "_sh").await;

    let mount = state.mount();
    let agents = std::fs::read_to_string(mount.join("AGENTS.md"))
        .expect("the workspace was seeded with the orientation");
    assert!(agents.contains("goofi-pipe is a live"), "the orientation is the real one: {agents}");
    assert_eq!(std::fs::read_to_string(mount.join("CLAUDE.md")).unwrap(), "@AGENTS.md\n");

    let cfg = std::fs::read_to_string(term::config_dir(&mount, &id).join("mcp.json"))
        .expect("the spawn wrote the harness's MCP config");
    assert!(cfg.contains(&format!("/mcp/{id}")), "the config names the minted address: {cfg}");
    assert!(cfg.contains(&format!("127.0.0.1:{}", addr.split(':').nth(1).unwrap())),
            "the config names a URL that reaches this server: {cfg}");

    let path = format!("/mcp/{id}");
    let (born, err) = tool(&addr, &path, 1, "add_node", json!({ "type": "Oscillator" })).await;
    assert!(!err, "the instance's own address serves its tools: {born}");
    let (undone, _) = tool(&addr, "/mcp", 2, "undo", json!({})).await;
    assert!(undone.contains("\"changed\": false"), "the central session undid another's: {undone}");

    call(&mut ctl, 3, "load", json!({})).await;
    let roster = call(&mut ctl, 4, "list_harnesses", json!({})).await;
    assert_eq!(roster["instances"], json!([]), "the replaced patch's harnesses stayed: {roster}");
    let (refused, err) = tool(&addr, &path, 5, "add_node", json!({ "type": "Oscillator" })).await;
    assert!(err, "a harness from the replaced patch still edited the new one: {refused}");
    assert!(refused.contains(&id), "the refusal names the instance it refused: {refused}");
    assert!(read_until(&mut term, "exit_code").await.contains("exit_code"),
            "the child outlived the patch that spawned it");
    let (ok, err) = tool(&addr, "/mcp", 6, "add_node", json!({ "type": "Buffer" })).await;
    assert!(!err, "replacing the patch closed the central endpoint too: {ok}");

    let (_ctl, _, mut left) = harness(&addr, "_sh").await;
    state.release_mount();
    assert!(read_until(&mut left, "exit_code").await.contains("exit_code"),
            "the harness outlived the goofi that spawned it");
}
