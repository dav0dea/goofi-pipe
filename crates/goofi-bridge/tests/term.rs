//! The harness plane, driven the way the agent panel will: `/control` mints an instance,
//! `/term/<id>` carries its PTY, and `/mcp/<id>` is the address goofi handed *that* harness.
//!
//! Nothing here needs Claude Code, Codex or opencode to be installed. The harnesses spawned are
//! the hidden `_sh` and `_deaf` adapters — the same `_`-prefixed idiom the node catalog uses for
//! its test types — so the PTY, the roster, the reaper, the stop escalation and the per-instance
//! MCP route are all driven by `/bin/sh`, which every machine that can run this suite has.

use std::ffi::OsString;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use goofi_bridge::{serve_app, term, AppState};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

/// A live server, answering `/control`, `/term` and both MCP addresses from one `AppState`. The
/// MCP port is pointed at the ephemeral listener, so the URL a spawn mints is one this very test
/// can call back into — which is the whole claim `/mcp/<id>` makes.
async fn start_server() -> (String, AppState) {
    let state = AppState::new();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    state.set_mcp_port(addr.port());
    let served = state.clone();
    tokio::spawn(async move {
        serve_app(listener, served, None).await.unwrap();
    });
    (addr.to_string(), state)
}

async fn recv_text(ws: &mut Ws) -> Value {
    recv_text_by(ws, tokio::time::Instant::now() + Duration::from_secs(5)).await
}

/// As [`recv_text`], but bounded by an absolute deadline — what a caller waiting on something the
/// server only does after a grace period needs, since a per-message timeout shorter than that grace
/// would fire on the silence in between.
async fn recv_text_by(ws: &mut Ws, deadline: tokio::time::Instant) -> Value {
    loop {
        let msg = tokio::time::timeout_at(deadline, ws.next())
            .await
            .expect("recv timed out")
            .expect("stream ended")
            .expect("ws error");
        if let Message::Text(t) = msg {
            return serde_json::from_str(t.as_str()).expect("json");
        }
    }
}

/// Send an RPC and return its reply, skipping interleaved broadcast events.
async fn call(ws: &mut Ws, id: i64, op: &str, payload: Value) -> Value {
    ws.send(Message::Text(json!({ "id": id, "op": op, "payload": payload }).to_string()))
        .await
        .unwrap();
    loop {
        let v = recv_text(ws).await;
        if v["id"] == json!(id) {
            assert!(v["error"].is_null(), "{op} failed: {v}");
            return v["result"].clone();
        }
    }
}

/// One HTTP POST to an MCP address, returning the parsed JSON-RPC reply. Hand-rolled over a
/// `TcpStream` exactly as `tests/mcp.rs` does: `Connection: close` makes the reply read-to-EOF.
async fn mcp(addr: &str, path: &str, id: i64, method: &str, params: Value) -> Value {
    let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string();
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), s.read_to_end(&mut raw))
        .await
        .expect("the endpoint answered within 5s")
        .unwrap();
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (_, body) = text.split_once("\r\n\r\n").expect("a well-formed HTTP reply");
    serde_json::from_str(body).unwrap_or_else(|e| panic!("a JSON-RPC reply, got {text:?}: {e}"))
}

/// A `tools/call` against an MCP address, answering `(text, is_error)`.
async fn tool(addr: &str, path: &str, id: i64, name: &str, args: Value) -> (String, bool) {
    let reply = mcp(addr, path, id, "tools/call", json!({ "name": name, "arguments": args })).await;
    let result = &reply["result"];
    (
        result["content"][0]["text"].as_str().unwrap_or_default().to_string(),
        result["isError"] == json!(true),
    )
}

/// Read `/term` until `want` has been seen in the PTY bytes, answering the exit frame if one
/// arrives first. Panics on timeout, naming what did arrive — a garbled stream is otherwise a
/// hang with nothing to read.
async fn read_until(ws: &mut Ws, want: &str) -> String {
    let mut seen = String::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !seen.contains(want) {
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

/// The whole lifecycle in one pass: a spawn mints an instance, the PTY carries bytes BOTH ways
/// (the child's own output, not merely the terminal's echo of the input), and a stop reaps the
/// child with the exit code it really chose.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_harness_spawns_carries_bytes_and_is_reaped_with_its_exit_code() {
    let (addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await; // hello

    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .expect("a spawn answers an instance id")
        .to_string();

    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    // `6*7` is in the input the terminal echoes back; `42` can only come from the child having
    // run. Without that distinction the assertion would pass on the line discipline alone.
    term.send(Message::Binary(b"echo $((6*7))\n".to_vec())).await.unwrap();
    read_until(&mut term, "42").await;

    // The child chooses 7, so the reaped code cannot be confused with a default or a signal.
    term.send(Message::Binary(b"exit 7\n".to_vec())).await.unwrap();
    let tail = read_until(&mut term, "exit_code").await;
    assert!(tail.contains("\"exit_code\":7"), "the exit frame carries the child's code: {tail:?}");

    let roster = call(&mut ctl, 2, "list_harnesses", json!({})).await;
    let inst = &roster["instances"][0];
    assert_eq!(inst["id"], json!(id));
    assert_eq!(inst["harness"], json!("_sh"));
    assert_eq!(inst["state"], json!("exited"), "a reaped instance is exited: {roster}");
    assert_eq!(inst["exit_code"], json!(7), "{roster}");
    state.release_mount();
}

/// Turn the line discipline's echo OFF, then have the child report where it is and what it was
/// handed. Without `stty -echo` every assertion on the answer would pass on the terminal repeating
/// the question — so even the readiness marker is spelled `REA''DY`, which the echo shows verbatim
/// and only the child prints joined.
async fn report(term: &mut Ws) -> String {
    term.send(Message::Binary(b"stty -echo; echo REA''DY\n".to_vec())).await.unwrap();
    read_until(term, "READY").await;
    term.send(Message::Binary(
        b"printf 'CWD[%s]TERM[%s]COLOR[%s]LC[%s]HOME[%s]KEPT[%s]END\\n' \"$(pwd -P)\" \
          \"$TERM\" \"$COLORTERM\" \"$LC_ALL\" \"$HOME\" \"$STATED_BY_THE_TEST\"\n"
            .to_vec(),
    ))
    .await
    .unwrap();
    read_until(term, "END").await
}

/// One `KEY[value]` field out of that report.
fn field<'a>(seen: &'a str, key: &str) -> &'a str {
    let rest = seen.split_once(key).unwrap_or_else(|| panic!("no {key} in {seen:?}")).1;
    rest.split_once(']').expect("a closed field").0
}

/// The task's two headline behaviours, driven through a real child.
///
/// **The cwd is the patch's ephemeral workspace** — the whole reason the agent and the user edit
/// one patch together — so that half goes through the `spawn_harness` RPC, mount and all.
///
/// **The environment is the parent's, with only the terminal contract overlaid.** That half states
/// its own parent rather than reading the suite's: on a machine whose terminal already says
/// `TERM=xterm-256color` an assertion against the ambient value would pass whether goofi overlaid
/// anything or not, and cargo runs the suite as threads in one process, where no test may set a
/// variable without setting it for every other.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_harness_runs_in_the_workspace_with_the_terminal_contract_overlaid() {
    let (addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();
    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    let seen = report(&mut term).await;
    assert_eq!(
        std::path::Path::new(field(&seen, "CWD[")),
        std::fs::canonicalize(state.mount()).unwrap(),
        "the harness does not run in the patch's workspace: {seen:?}"
    );

    // A stated parent: a `TERM` no TUI will draw on, a colour answer of `no`, a locale that is not
    // UTF-8, and one variable goofi knows nothing about — which is also what proves the stated
    // parent reached the child at all, and with it that each overlay really had something to beat.
    let env: Vec<(OsString, OsString)> = vec![
        ("TERM".into(), "dumb".into()),
        ("COLORTERM".into(), "no".into()),
        ("LANG".into(), "C".into()),
        ("STATED_BY_THE_TEST".into(), "kept".into()),
    ];
    let stated = state
        .harnesses
        .spawn("_sh", &state.mount(), "http://127.0.0.1:1", &env, state.events.clone())
        .expect("a spawn with a stated parent environment");
    let (mut term, _) = connect_async(format!("ws://{addr}/term/{stated}")).await.unwrap();
    let seen = report(&mut term).await;
    assert_eq!(field(&seen, "TERM["), "xterm-256color", "a dumb TERM was not overlaid: {seen:?}");
    assert_eq!(field(&seen, "COLOR["), "truecolor", "{seen:?}");
    assert_eq!(field(&seen, "LC["), "C.UTF-8", "a parent with no UTF-8 locale gets one: {seen:?}");
    assert_eq!(field(&seen, "KEPT["), "kept", "the stated parent never reached the child: {seen:?}");
    // …and a variable goofi was never told about — `HOME`, the one the credentials follow — comes
    // through untouched, which is the "inherited whole, no HOME redirection" half of the contract.
    assert_eq!(field(&seen, "HOME["), std::env::var("HOME").unwrap_or_default(), "{seen:?}");

    // Reap both: a leaked PTY child outlives the suite and corrupts every later measurement.
    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;
    state.harnesses.stop(&stated).unwrap();
    state.release_mount();
}

/// The end of a long transcript, for a failure message: 3 kB of `L1…L400` is unreadable.
fn tail(seen: &str) -> String {
    let end: Vec<char> = seen.chars().rev().take(120).collect();
    format!("{} bytes ending {:?}", seen.len(), end.into_iter().rev().collect::<String>())
}

/// A dying harness's LAST words are the whole reason to watch one — the stack trace, the auth
/// failure, the rate-limit message — and they are written in the instant before the exit. So the
/// exit frame is the last thing this socket sends rather than the first thing it reaches for: the
/// child's exit and the PTY's end-of-stream are two different events, and everything between them
/// used to be thrown away. The burst is long enough that the socket cannot have drained it before
/// `child.wait()` returns, which is exactly the race that dropped it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dying_harness_delivers_its_last_output_before_its_exit_code() {
    let (addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();

    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    // `TAIL''MARK` is what the terminal echoes back, so `TAILMARK` can only come from the child
    // having run — the assertion cannot pass on the line discipline repeating the command.
    term.send(Message::Binary(
        b"i=0; while [ $i -lt 400 ]; do i=$((i+1)); echo L$i; done; echo TAIL''MARK; exit 5\n"
            .to_vec(),
    ))
    .await
    .unwrap();

    let seen = read_until(&mut term, "exit_code").await;
    assert!(seen.contains("L400"), "the burst was truncated: {}", tail(&seen));
    assert!(seen.contains("TAILMARK"), "the child's last line was dropped: {}", tail(&seen));
    assert!(seen.contains("\"exit_code\":5"), "{}", tail(&seen));
    state.release_mount();
}

/// Wait on the control socket for the reaper's announcement that `id` has exited.
async fn await_exit(ctl: &mut Ws, id: &str) -> Value {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let ev = recv_text_by(ctl, deadline).await;
        let inst = ev["payload"]["instances"][0].clone();
        if ev["event"] == json!("harness_changed") && inst["state"] == json!("exited") {
            assert_eq!(inst["id"], json!(id), "another instance exited: {ev}");
            assert!(inst["exit_code"].is_number(), "no exit code captured: {ev}");
            return inst;
        }
    }
}

/// A stop asks before it insists, and returns without waiting for either: `stop_harness` closes the
/// address and answers at once, and the reaper announces the exit behind it — which is why the exit
/// is WAITED FOR on the broadcast rather than read back from the reply.
///
/// The `_deaf` harness reports the SIGTERM it was sent and then refuses to leave, so all three
/// halves are watchable in one pass: the graceful signal arrives, it is not enough, and the SIGKILL
/// after the grace reaps the child anyway. The test costs that grace — nothing shorter can prove
/// the second signal fired. It waits for `armed` first: a signal delivered before the trap was
/// installed would prove nothing at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_signals_first_and_kills_after_the_grace() {
    let (addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_deaf" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();
    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    read_until(&mut term, "armed").await;

    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;
    read_until(&mut term, "GOT-TERM").await;
    await_exit(&mut ctl, &id).await;
    state.release_mount();
}

/// A harness belongs to the patch that spawned it: its cwd IS that patch's workspace and its MCP
/// address edits that patch's graph. So opening another patch tears every one of them down through
/// the same stop path. Without it a `new` swaps the mount underneath a live agent, which goes on
/// editing a patch it was never launched for, from a directory that no longer exists, while the
/// roster still calls it running.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_new_patch_tears_down_the_harnesses_the_old_one_spawned() {
    let (addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();
    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    // Wait for the child to answer before replacing the patch, so the teardown cannot be racing
    // the spawn it is meant to undo.
    term.send(Message::Binary(b"echo $((6*7))\n".to_vec())).await.unwrap();
    read_until(&mut term, "42").await;

    call(&mut ctl, 2, "new", json!({})).await;

    let roster = call(&mut ctl, 3, "list_harnesses", json!({})).await;
    assert_eq!(roster["instances"], json!([]), "the replaced patch's harnesses stayed: {roster}");
    let (refused, err) =
        tool(&addr, &format!("/mcp/{id}"), 1, "add_node", json!({ "type": "Oscillator" })).await;
    assert!(err, "a harness from the replaced patch still edited the new one: {refused}");
    // …and the child is gone, not merely forgotten: the socket that was watching it sees the exit.
    let tail = read_until(&mut term, "exit_code").await;
    assert!(tail.contains("exit_code"), "the child outlived the patch that spawned it: {tail:?}");
    state.release_mount();
}

/// The design's centre: the address is MINTED by goofi and written into the harness's own config,
/// so identity never travels through the agent. Stopping drops the route — teachably — while the
/// central `/mcp` every external agent uses stays open, and the two undo stacks stay apart.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stopped_instances_address_refuses_while_the_central_one_stays_open() {
    let (addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();

    // goofi wrote the config, in this instance's own directory, naming this instance's URL.
    let cfg = std::fs::read_to_string(term::config_dir(&state.mount(), &id).join("mcp.json"))
        .expect("the spawn wrote the harness's MCP config");
    assert!(cfg.contains(&format!("/mcp/{id}")), "the config names the minted address: {cfg}");
    assert!(
        cfg.contains(&format!("127.0.0.1:{}", addr.split(':').nth(1).unwrap())),
        "the config names a URL that reaches this server: {cfg}"
    );

    let path = format!("/mcp/{id}");
    let (uid, err) = tool(&addr, &path, 1, "add_node", json!({ "type": "Oscillator" })).await;
    assert!(!err, "the instance's own address serves its tools: {uid}");

    // The session follows the address: the central agent's undo must not reach into this
    // instance's edit, which is the whole reason the address keys the history.
    let (undone, _) = tool(&addr, "/mcp", 2, "undo", json!({})).await;
    assert!(undone.contains("\"changed\": false"), "the central session undid another's: {undone}");

    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;
    let (refused, err) = tool(&addr, &path, 3, "add_node", json!({ "type": "Oscillator" })).await;
    assert!(err, "a stopped address still served a tool call: {refused}");
    assert!(refused.contains(&id), "the refusal names the instance it refused: {refused}");

    // …and the central endpoint is untouched by any of it.
    let (ok, err) = tool(&addr, "/mcp", 4, "add_node", json!({ "type": "Buffer" })).await;
    assert!(!err, "stopping an instance closed the central endpoint too: {ok}");
    state.release_mount();
}

/// The roster rides the snapshot, for the same reason the `runtime` overlay does: the live stream
/// carries only transitions, so a tab that joins after a spawn would otherwise draw an empty
/// switcher over a running harness.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_roster_survives_a_reconnect() {
    let (addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    let hello = recv_text(&mut ctl).await;
    assert_eq!(
        hello["payload"]["harnesses"]["instances"],
        json!([]),
        "a fresh backend has no instances: {hello}"
    );
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();
    drop(ctl);

    let (mut later, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    let hello = recv_text(&mut later).await;
    let instances = &hello["payload"]["harnesses"]["instances"];
    assert_eq!(instances[0]["id"], json!(id), "the roster was not seeded: {hello}");
    assert_eq!(instances[0]["state"], json!("running"), "{hello}");
    // Launching an agent is not authoring, so it must not put the unsaved dot on an untouched
    // patch — which is why the config goofi writes lands BESIDE the workspace and not in it.
    assert_eq!(hello["payload"]["unsaved_changes"], json!(false), "a spawn dirtied the patch");
    // `detected` rides the same shape, so a joining tab can offer the launch buttons too.
    assert!(hello["payload"]["harnesses"]["detected"].is_array(), "{hello}");
    state.release_mount();
}
