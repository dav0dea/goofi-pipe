//! The graph in use: nodes run, stream, fail, recover, and pace themselves — with no tick anywhere.
//!
//! Every node owns one thread that parks on its own doorbell. Frames travel node to node over
//! shared memory, so no node runs under the graph mutex and no user action waits on a `process()`.
//! What makes any of it observable is the status-drain worker: a node is known when `add_node`
//! answers and addressable only when it reports `Ready`. So each assertion here polls.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use goofi_tests::{hex, holds_within, j, Goofi, Viewer};

fn f32s(d: &goofi_core::Data) -> Vec<f32> {
    let goofi_core::Value::Array(a) = d.value() else { panic!("not an array: {d:?}") };
    a.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn a_chain_runs_streams_and_follows_the_params_edited_under_it() {
    // The reference shape: a free-running producer feeding a consumer, both edited while running.
    // Nothing is stepped and nothing is waited on — the producer paces itself off the patch rate.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 64 }));
    g.call("update_param", j!({ "node": hex(osc), "group": "oscillator", "name": "sfreq", "value": 64.0 }));
    let probe = g.probe(buf, "out"); // opened BEFORE the wire: the data services keep no history
    g.link(osc, "out", buf, "data");

    let full = g.until("the window to fill", |_| {
        probe.latest().filter(|d| f32s(d).len() == 64).map(|d| f32s(&d))
    });
    assert!(full.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "a unit sine: {:?}", &full[..4]);
    assert_eq!(probe.latest().unwrap().meta().sfreq(), Some(64.0), "sfreq rides the frame");

    // A param edit reaches a node that is already running, without a restart and without a tick.
    g.call("update_param", j!({ "node": hex(buf), "group": "buffer", "name": "size", "value": 16 }));
    g.until("the window to shrink under the running node", |_| {
        probe.latest().filter(|d| f32s(d).len() == 16).map(|_| ())
    });

    // And the measured rate reaches a client, which is what the node header draws.
    let mut ev = g.events();
    let stats = g.until("a node_stats broadcast", |_| {
        let p = ev.next("node_stats");
        (p["node"] == hex(osc)).then_some(p)
    });
    assert!(stats["stats"]["updates_per_second"].as_f64().is_some_and(|r| r > 0.0), "{stats}");
}

#[test]
fn a_producer_paces_itself_to_its_rate_cap_and_follows_a_live_change() {
    // `common.max_frequency` is what stops a source running as fast as its thread can. Counting
    // emitted frames is the only way to see it: the param's stated value reads correct against a
    // node that ignores it entirely. (What BINDS this cap to the patch's `default_ufreq` global is
    // an expression, so that half is proven where an evaluator exists — see `python.rs`.)
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let probe = g.probe(osc, "out");
    g.call("update_param", j!({ "node": hex(osc), "group": "common", "name": "max_frequency", "value": 5.0 }));
    g.ready(osc);

    let runs = |window: Duration| {
        let (mut seen, mut last, end) = (0, None, Instant::now() + window);
        while Instant::now() < end {
            let now = probe.latest().map(|d| d.meta().index());
            if now.is_some() && now != last {
                seen += 1;
                last = now;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        seen
    };
    let slow = runs(Duration::from_millis(800));
    assert!(slow <= 8, "5 Hz produced {slow} frames in 0.8 s — the cap is not honoured");

    g.call("update_param", j!({ "node": hex(osc), "group": "common", "name": "max_frequency", "value": 60.0 }));
    g.until("the re-paced producer", |_| (runs(Duration::from_millis(400)) > 8).then_some(()));
}

#[test]
fn each_way_a_node_can_fail_is_reported_and_none_of_them_stops_the_patch() {
    // Three failure modes on three channels, in ONE patch beside a healthy chain — because the
    // property that matters is containment: a node's failure is its own, and the control plane
    // stays answerable. A lifecycle panic is the sharp case: `setup` runs under the graph mutex the
    // bridge locks with `.unwrap()`, so one panic used to poison it and take the app down for good.
    let g = Goofi::new();
    let bad = g.add("_TestFail");
    let panics = g.add("_TestPanic");
    let unborn = g.add("_TestSetupFail");
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    let probe = g.probe(buf, "out");
    g.link(osc, "out", buf, "data");

    let why = g.until("the failing node's error", |g| g.error(bad));
    assert!(why.contains("the sensor is unplugged"), "the node's own words reach the client: {why}");
    let why = g.until("the panicking node's error", |g| g.error(panics));
    assert!(why.contains("panic"), "a panic is reported as an error, not as a crash: {why}");
    let why = g.until("the un-initialized node's error", |g| g.error(unborn));
    assert!(why.contains("the device did not open"), "{why}");
    assert_eq!(g.stage(unborn), "error", "a node whose setup failed does not read as healthy");
    // `_TestSetupFail` emits from `process`, so a frame is proof that it ran after setup refused.
    assert!(g.probe(unborn, "out").latest().is_none(), "process must not run against a failed setup");
    g.until("the healthy chain to keep streaming", |_| probe.latest().map(|_| ()));
    assert_eq!(g.nodes().len(), 5, "and the control plane still answers");
}

#[test]
fn a_required_input_refuses_to_run_on_a_hole_and_says_so() {
    // The input contract. A `required` slot is the promise that `process` may read it
    // unconditionally, which is only worth anything if the runtime enforces the refusal.
    let g = Goofi::new();
    let need = g.add("_TestRequired");
    let probe = g.probe(need, "out");
    // Set to free-run without its input wired — the shape a user reaches by hand. The gate is on
    // PRESENCE, never on wiring, so a slot fed by a producer that has emitted nothing reads the
    // same as this one.
    g.call("update_param", j!({ "node": hex(need), "group": "common", "name": "autotrigger", "value": true }));
    g.call("update_param", j!({ "node": hex(need), "group": "common", "name": "max_frequency", "value": 20.0 }));
    let why = g.until("the refusal", |g| g.error(need));
    assert!(why.contains("in"), "the refusal names the slot that is empty: {why}");
    assert!(g.stays(|_| probe.latest().is_none()), "and nothing ran");

    let src = g.add("_TestCounter");
    g.link(src, "out", need, "in");
    g.until("the fed node to run", |_| probe.latest().map(|_| ()));
    assert!(g.until("the error to clear", |g| g.error(need).is_none().then_some(true)));
}

/// A node whose FIRST instance fails to boot and whose second succeeds — so a restart is observable
/// as RECOVERY, rather than as "the op did not error".
static FLAKY: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "_TestFlaky",
    category: "test",
    doc: "fails setup once, then succeeds",
    inputs: &[],
    outputs: &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }],
    params: &[],
    isolation: goofi_node::Isolation::InProcess,
    producer: true,
    factory: || unreachable!("a dyn type is built by its registered factory"),
};

/// Its frames carry WHICH instance emitted them. A per-instance run count cannot say: the first
/// frames of a reborn node are emitted while the reducer is still on the dead generation's name, so
/// by the time a viewer is following again the count has already passed where the corpse left off.
struct Flaky {
    generation: f32,
    n: f32,
}
impl goofi_node::Node for Flaky {
    fn setup(&mut self, _c: &mut goofi_node::NodeCtx, _p: &goofi_node::Params<'_>) -> goofi_node::NodeResult {
        if self.generation == 0.0 { Err("the device did not open".into()) } else { Ok(()) }
    }
    fn process(&mut self, _i: &goofi_node::Inputs<'_>, o: &mut goofi_node::Outputs<'_>,
               _c: &mut goofi_node::NodeCtx, _p: &goofi_node::Params<'_>) -> goofi_node::NodeResult {
        self.n += 1.0;
        let mut body = self.generation.to_le_bytes().to_vec();
        body.extend_from_slice(&self.n.to_le_bytes());
        o.set("out", goofi_core::Data::array_f32(vec![2], body, goofi_core::Meta::new()).unwrap());
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restart_recovers_a_node_and_the_viewer_follows_it_to_its_new_home() {
    // A rebirth publishes under a NEW service name, so nothing downstream can notice by failing:
    // the subscriber on the corpse's name receives cleanly and for ever, and simply never receives
    // again — a viewer frozen on its last frame with no error anywhere. The reducer re-derives the
    // name on its own clock, which is what makes the restart button a recovery rather than a mute.
    let g = Goofi::new();
    let builds = Arc::new(AtomicUsize::new(0));
    g.register_dyn(&FLAKY, Box::new(move |_| {
        Box::new(Flaky { generation: builds.fetch_add(1, Ordering::SeqCst) as f32, n: 0.0 })
    }));
    let base = g.serve().await;
    let uid = g.add("_TestFlaky");
    // Paced, so the run count steps in ones. An uncapped producer laps its own viewer — the socket
    // lags, the broadcast drops what it could not keep up with, and the one frame this test is
    // about is exactly the frame most likely to be dropped.
    g.call("update_param", j!({ "node": hex(uid), "group": "common", "name": "max_frequency", "value": 10.0 }));

    let why = tokio::task::block_in_place(|| g.until("the first instance to fail", |g| g.error(uid)));
    assert!(why.contains("the device did not open"), "{why}");

    g.call("restart_node", j!({ "node": hex(uid) }));
    tokio::task::block_in_place(|| {
        g.until("the second instance to boot clean", |g| g.error(uid).is_none().then_some(()))
    });

    let mut v = Viewer::open(&base, &hex(uid), "out").await;
    assert_eq!(f32s(&v.decoded().await)[0], 1.0, "the stream is live on the recovered generation");
    let before = g.state.graph.lock().unwrap().output_service_of(uid, "out");
    g.call("restart_node", j!({ "node": hex(uid) }));
    assert_ne!(g.state.graph.lock().unwrap().output_service_of(uid, "out"), before,
               "a rebirth is a new name");
    // The frame itself says which instance made it. Reading until a later generation appears is
    // what the re-home costs: for up to one rehome interval the reducer is still listening on the
    // dead name, and everything the new instance emits in that window is simply gone.
    v.until(|d| f32s(d)[0] == 2.0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_viewers_of_one_slot_share_one_reducer_and_each_gets_what_it_can_draw() {
    // The data plane's headline claim: N viewers of a slot cost ONE reduce per frame, on the
    // bridge's own subscription — so no number of tabs can slow a `process()` down. The fold is
    // richest-per-dim, so a wide viewer beside a narrow one leaves neither short.
    let g = Goofi::new();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    // Far more samples per frame than any viewer asks for, so there is a reduction to fold at all.
    g.call("update_param", j!({ "node": hex(osc), "group": "oscillator", "name": "sfreq", "value": 20000.0 }));
    let key = (osc, "out".to_string());

    let spec = |max: usize| j!([{ "dtype": "array", "ndim": [["le", 2]], "dims": [],
                                  "reduce": [{ "dim": -1, "max": max, "method": "envelope" }] }]);
    let mut wide = Viewer::open(&base, &hex(osc), "out").await;
    wide.view(spec(256)).await;
    let mut narrow = Viewer::open(&base, &hex(osc), "out").await;
    narrow.view(spec(32)).await;
    // Four more tabs on the same slot. The count is what separates "one reduce per frame" from
    // "one per subscriber": with two viewers both readings look alike under jitter.
    let mut rest = Vec::new();
    for _ in 0..4 {
        let mut v = Viewer::open(&base, &hex(osc), "out").await;
        v.view(spec(32)).await;
        rest.push(v);
    }

    // Both are served the SAME merged frame: the fold takes the largest need per dim, and an
    // envelope of width W carries 2·W samples. A viewer is never sent less than it asked for.
    for v in [&mut wide, &mut narrow] {
        let d = v.until(|d| f32s(d).len() == 512).await;
        assert!(d.meta().reduced().is_some(), "the frame says it is a reduction");
    }
    assert_eq!(g.state.reducers.active_slots(), 1, "six viewers, one reducer");
    assert_eq!(g.state.reducers.subscribers(&key), 6);

    // Bounded by the producer's rate, not multiplied by subscribers — the property a
    // per-connection loop would break while every other assertion here still passed.
    let passes = g.state.reducers.reductions(&key);
    tokio::time::sleep(Duration::from_millis(400)).await;
    let grew = g.state.reducers.reductions(&key) - passes;
    assert!(grew < 30, "{grew} reduce passes for ~12 frames — the reducer is running per subscriber");

    drop(rest);
    drop(narrow);
    assert!(holds_within(Duration::from_secs(5), || g.state.reducers.subscribers(&key) == 1).await);
    drop(wide);
    assert!(holds_within(Duration::from_secs(5), || g.state.reducers.active_slots() == 0).await,
            "the last viewer left and the reducer went with it");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_viewer_that_stops_answering_is_reclaimed_and_a_merely_slow_one_is_not() {
    // A tab closed by a killed browser never sends a close frame. Without a liveness probe its
    // reducer runs for ever against nobody. The hard half is the second one: a viewer on a slow
    // link falls behind and MUST NOT be mistaken for a dead one.
    let g = Goofi::impatient();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    let key = (osc, "out".to_string());

    let dead = Viewer::open(&base, &hex(osc), "out").await;
    let mut slow = Viewer::open(&base, &hex(osc), "out").await;
    slow.decoded().await;
    assert_eq!(g.state.reducers.subscribers(&key), 2);

    // `dead` never reads and never pongs — a socket whose peer is gone. `slow` reads far behind the
    // producer, which is a live tab on a bad connection.
    let ticker = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(300)).await;
            slow.decoded().await;
        }
    });
    assert!(holds_within(Duration::from_secs(10), || g.state.reducers.subscribers(&key) == 1).await,
            "the silent peer was never reclaimed");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(g.state.reducers.subscribers(&key), 1, "…and the slow one kept its subscription");
    ticker.abort();
    drop(dead);
}

#[test]
fn a_busy_node_never_holds_up_the_control_plane_and_never_wedges_the_exit() {
    // `remove_node` runs under the same mutex every control RPC needs, and a node observes its halt
    // flag only BETWEEN runs — so waiting on one parked inside `process()` would freeze the app for
    // that window (a real subprocess node waits out its ten-second timeout). Exit waits on purpose,
    // to a CEILING: a wedged node must release its shared memory without wedging the shutdown.
    let g = Goofi::new();
    let slow = g.add("_TestSlow");
    let other = g.add("_TestSlow");
    std::thread::sleep(Duration::from_millis(60)); // both threads are now inside a ten-second run

    let t0 = Instant::now();
    g.call("remove_node", j!({ "node": hex(slow) }));
    assert!(t0.elapsed() < Duration::from_millis(100),
            "the delete took {:?} — it waited on the busy node under the graph lock", t0.elapsed());

    let t0 = Instant::now();
    g.state.graph.lock().unwrap().shutdown();
    // Five seconds sits between the two answers rather than beside one of them: the ceiling is two
    // (`SHUTDOWN_WAIT`) and a node that was JOINED costs the full ten this node sleeps.
    assert!(t0.elapsed() < Duration::from_secs(5),
            "the exit took {:?} — it JOINED the busy node instead of waiting to a ceiling", t0.elapsed());
    let _ = other;
}

#[test]
fn a_refreshable_param_is_re_enumerated_on_the_nodes_own_thread() {
    // The ⟳ round trip has no other observable surface: options live in runtime state, never in the
    // doc, so they reach a client ONLY through the status worker's echo — and that echo is what
    // lifts the spinner. The RPC reply says merely that the request went out.
    let g = Goofi::new();
    let picker = g.add("_TestPicker");
    let mute = g.add("_TestMute");
    g.ready(picker);
    g.ready(mute);
    let mut ev = g.events();

    g.call("refresh_param", j!({ "node": hex(picker), "group": "io", "name": "device" }));
    let p = g.until("the picker's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(picker)).then_some(p)
    });
    let options = p["params"]["io"]["device"]["options"].as_array().cloned().unwrap_or_default();
    assert!(options.iter().any(|o| o.as_str().is_some_and(|s| s.starts_with("dev"))),
            "the re-enumerated list reached the client: {options:?}");
    assert_eq!(p["refreshed_params"], j!([["io", "device"]]), "…and the spinner is cleared");

    // A node that declares the param but implements no hook must still get its echo, or the button
    // spins for its full safety timeout on every such node.
    g.call("refresh_param", j!({ "node": hex(mute), "group": "io", "name": "device" }));
    let p = g.until("the mute picker's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(mute)).then_some(p)
    });
    assert_eq!(p["refreshed_params"], j!([["io", "device"]]), "the spinner is cleared regardless");
    assert_eq!(p["params"]["io"]["device"]["options"], j!(["none"]), "options left as declared");

    // A fixed list is refused, which is what lifts the spinner on the frontend's side.
    let osc = g.add("Oscillator");
    let why = g.refuse("refresh_param",
                       j!({ "node": hex(osc), "group": "oscillator", "name": "waveform" }));
    assert!(why.contains("not refreshable"), "{why}");
}
