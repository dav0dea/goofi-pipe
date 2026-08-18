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
use goofi_bridge::{term, AppState};
use goofi_tests::{host, Goofi};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

type Ws = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

/// A live server answering `/control`, `/term` and both MCP addresses from one instance. The
/// `Goofi` comes back too: dropping it releases the workspace mount the harnesses run in.
async fn start_server() -> (Goofi, String, AppState) {
    let g = Goofi::new();
    let addr = host(&g.serve().await).to_string();
    let state = g.state.clone();
    (g, addr, state)
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
    let mut answered = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !seen.contains(want) {
        // A real terminal answers ConPTY's cursor-position query, and the child stays BLOCKED
        // until one does — goofi only answers on its behalf while nobody is attached, precisely so
        // a live viewer can give the true position. This client is standing in for that viewer, so
        // it has to behave like one; without this it is not a weak fixture but a wrong one, and
        // every test here times out having seen exactly these four bytes.
        if !answered && seen.contains('\u{1b}') && seen.contains("[6n") {
            answered = true;
            ws.send(Message::Binary(b"\x1b[1;1R".to_vec())).await.unwrap();
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

/// The whole lifecycle in one pass: a spawn mints an instance, the PTY carries bytes BOTH ways
/// (the child's own output, not merely the terminal's echo of the input), and a stop reaps the
/// child with the exit code it really chose.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_harness_spawns_carries_bytes_and_is_reaped_with_its_exit_code() {
    let (_g, addr, state) = start_server().await;
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
    // A tab that opens the panel AFTER the harness died sees why, rather than an empty terminal:
    // the exit is served to a socket that was never there for it, until the instance is dismissed.
    let (mut late, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    let seen = read_until(&mut late, "exit_code").await;
    assert!(seen.contains("\"exit_code\":7"), "a late attach was told nothing: {seen:?}");

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
        // `E''ND` for the same reason `REA''DY` is spelled that way: the shell concatenates the
        // quotes away so the OUTPUT ends in `END`, while the echoed command line shows `E''ND` and
        // cannot satisfy the wait. Without it `read_until(…, "END")` returns on the echo — before
        // the printf has run — and every field below reads the format string instead of a value.
        b"printf 'CWD[%s]TERM[%s]COLOR[%s]LC[%s]HOME[%s]KEPT[%s]E''ND\\n' \"$(pwd -P)\" \
          \"$TERM\" \"$COLORTERM\" \"$LC_ALL\" \"$HOME\" \"$STATED_BY_THE_TEST\"\n"
            .to_vec(),
    ))
    .await
    .unwrap();
    read_until(term, "END").await
}

/// One `KEY[value]` field out of that report — the LAST such field, not the first.
///
/// The terminal echoes the `printf` line before the shell runs it, so `TERM[%s]` appears in the
/// stream before `TERM[xterm-256color]` does. `stty -echo` cannot prevent that on every platform:
/// ConPTY echoes on its own account, outside the line discipline `stty` speaks to. Reading the last
/// occurrence is what makes this parser independent of whether the echo was suppressed, rather than
/// dependent on a command that silently does nothing on Windows.
fn field<'a>(seen: &'a str, key: &str) -> &'a str {
    let at = seen.rfind(key).unwrap_or_else(|| panic!("no {key} in {seen:?}"));
    let rest = &seen[at + key.len()..];
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
    let (_g, addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();
    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    let seen = report(&mut term).await;
    // The child reports its cwd in ITS OWN spelling: an MSYS `sh` maps `%TEMP%` onto `/tmp`, so the
    // Windows path and this string name one directory under two schemes and can never compare
    // equal. What IS comparable is the tail goofi minted — the nonce directory and the workspace
    // inside it — which is unique to this run and so still proves the harness landed in this
    // patch's workspace rather than merely in some directory.
    let mount = state.mount();
    let mut tail = mount.components().rev().map(|c| c.as_os_str().to_string_lossy().into_owned());
    let (workspace, nonce) = (tail.next().unwrap(), tail.next().unwrap());
    let want = format!("{nonce}/{workspace}");
    assert!(
        field(&seen, "CWD[").ends_with(&want),
        "the harness does not run in the patch's workspace (wanted a cwd ending {want:?}): {seen:?}"
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
    // …and `HOME` — the one the credentials follow — is NOT redirected into the workspace, which is
    // the "inherited whole, no HOME redirection" half of the contract. Asserted by what it must not
    // be, rather than against the parent's own string: an MSYS shell reports `/c/Users/philipp` for
    // the directory Windows spells `C:\Users\philipp`, so byte equality would be testing that path
    // translator instead of goofi. That the parent's environment reaches the child at all is
    // already proven above by `KEPT`, on a variable no shell rewrites.
    let home = field(&seen, "HOME[");
    assert!(!home.is_empty(), "the child was handed no HOME at all: {seen:?}");
    assert!(!home.contains(&nonce), "HOME was redirected into the workspace: {seen:?}");

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
    let (_g, addr, state) = start_server().await;
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
    let (_g, addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_deaf" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();
    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    read_until(&mut term, "armed").await;

    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;
    // The graceful ask is only OBSERVABLE where signals are. This harness traps SIGTERM and says
    // so; on Windows there is no SIGTERM to trap — `taskkill` without `/F` is refused outright by a
    // console process — so the harness leaves at the END of the grace rather than the start, and
    // there is nothing to hear. Gated rather than deleted, because where the ask does happen it is
    // a real guarantee worth keeping. What must hold on every platform is asserted below: the stop
    // is announced, and the harness actually stops.
    #[cfg(unix)]
    read_until(&mut term, "GOT-TERM").await;
    // A stop that has been asked for but not yet obeyed is its own state. Without it the roster
    // this arm broadcasts is byte-identical to the one before it — a vacuous event — and every
    // client shows a live-looking harness whose address is already refusing, for as long as the
    // grace runs against a harness that ignores SIGTERM. Which is the case the grace exists for,
    // and the case this harness IS.
    let roster = call(&mut ctl, 3, "list_harnesses", json!({})).await;
    assert_eq!(roster["instances"][0]["state"], json!("stopping"), "{roster}");

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
    let (_g, addr, state) = start_server().await;
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
    let (_g, addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();

    // The harness's cwd IS the patch workspace, and the orientation is waiting in it — the only
    // placement a harness actually reads as a project doc, and the one that rides the `.gfi`.
    let mount = state.mount();
    let agents = std::fs::read_to_string(mount.join("AGENTS.md"))
        .expect("the workspace was seeded with the orientation");
    assert!(agents.contains("goofi-pipe is a live"), "the orientation is the real one: {agents}");
    assert_eq!(std::fs::read_to_string(mount.join("CLAUDE.md")).unwrap(), "@AGENTS.md\n");

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

/// Read `/term` text frames until an authoritative-size one arrives, answering `(cols, rows)`.
async fn recv_size(ws: &mut Ws) -> (u64, u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let v = recv_text_by(ws, deadline).await;
        if v["op"] == json!("size") {
            return (v["cols"].as_u64().expect("cols"), v["rows"].as_u64().expect("rows"));
        }
    }
}

/// Several views of ONE terminal, and one size between them. A PTY has a single window: two panels
/// showing the same harness cannot each have their own, so the size is arbitrated — **the last view
/// to speak wins**, and the answer is broadcast to every view (the others letterbox against it).
///
/// The two halves that are easy to get wrong are both driven here. A view that **retracts** (its
/// panel unmounted, so it has nothing on screen to speak for) and a view that **leaves** must both
/// hand the terminal back to the survivors — otherwise closing the size-owning view strands every
/// other at a size nobody can see. And a view that attaches LATE is told the current size at once,
/// because there is no change event coming to tell it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn several_views_of_one_terminal_agree_on_one_size() {
    let (_g, addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await;
    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_sh" })).await["instance_id"]
        .as_str()
        .unwrap()
        .to_string();
    let url = format!("ws://{addr}/term/{id}");
    let (mut a, _) = connect_async(&url).await.unwrap();
    let (mut b, _) = connect_async(&url).await.unwrap();

    let resize = |cols: u16, rows: u16| {
        Message::Text(json!({ "op": "resize", "cols": cols, "rows": rows }).to_string())
    };
    a.send(resize(100, 30)).await.unwrap();
    assert_eq!(recv_size(&mut a).await, (100, 30), "the view that asked is told the answer");
    assert_eq!(recv_size(&mut b).await, (100, 30), "a view that did not ask is told too");

    b.send(resize(80, 24)).await.unwrap();
    assert_eq!(recv_size(&mut b).await, (80, 24), "the last writer wins");
    assert_eq!(recv_size(&mut a).await, (80, 24), "…for every view");

    // A view whose panel unmounted keeps its socket (that is what keeps its scrollback flowing) but
    // stops speaking for the terminal, which it says with a zero size.
    b.send(resize(0, 0)).await.unwrap();
    assert_eq!(recv_size(&mut a).await, (100, 30), "a retracted view hands the size back");
    assert_eq!(recv_size(&mut b).await, (100, 30), "…and hears the answer it caused");

    // A late attach is told the current size without waiting for anyone to resize.
    let (mut c, _) = connect_async(&url).await.unwrap();
    assert_eq!(recv_size(&mut c).await, (100, 30), "a late view starts at the current size");

    c.send(resize(120, 40)).await.unwrap();
    assert_eq!(recv_size(&mut a).await, (120, 40));
    drop(c);
    assert_eq!(recv_size(&mut a).await, (100, 30), "a view that left hands the size back");

    // …and the arbitration reaches the KERNEL, not just the other views: the child's own idea of
    // its window is what a TUI lays itself out against. `30 100` is nowhere in the input.
    a.send(Message::Binary(b"stty size\n".to_vec())).await.unwrap();
    read_until(&mut a, "30 100").await;

    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;
    state.release_mount();
}

/// The roster rides the snapshot, for the same reason the `runtime` overlay does: the live stream
/// carries only transitions, so a tab that joins after a spawn would otherwise draw an empty
/// switcher over a running harness.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_roster_survives_a_reconnect() {
    let (_g, addr, state) = start_server().await;
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


/// A harness runs with NOBODY watching. That is not an edge case: an agent spawned over MCP has no
/// panel open, and goofi is expected to work headless. On Windows ConPTY asks the terminal where
/// the cursor is the moment the child starts and blocks it until something answers — so with no
/// viewer attached there is nobody to answer, and the harness hangs before it runs a single
/// command. Attaching afterwards cannot rescue it: the query went out to a broadcast with no
/// subscribers and is gone.
///
/// `_deaf` prints on its own schedule and needs no input, so seeing its output through a socket
/// that attached LATE proves the child was running the whole time it was unobserved.
#[tokio::test]
async fn a_harness_runs_with_nobody_watching_and_is_still_going_when_a_viewer_arrives() {
    let (_g, addr, state) = start_server().await;
    let (mut ctl, _) = connect_async(format!("ws://{addr}/control")).await.unwrap();
    recv_text(&mut ctl).await; // hello

    let id = call(&mut ctl, 1, "spawn_harness", json!({ "harness": "_deaf" })).await["instance_id"]
        .as_str()
        .expect("a spawn answers an instance id")
        .to_string();

    // Deliberately no `/term` socket yet — this is the whole point of the test.
    tokio::time::sleep(Duration::from_millis(600)).await;

    let (mut term, _) = connect_async(format!("ws://{addr}/term/{id}")).await.unwrap();
    read_until(&mut term, "armed").await;

    call(&mut ctl, 2, "stop_harness", json!({ "instance": id })).await;
    state.release_mount();
}

