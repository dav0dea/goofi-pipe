//! The graph in use: nodes run, stream, fail, recover, and pace themselves — with no tick anywhere.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use goofi_tests::{f32s, Goofi, Viewer, ep, hex, holds_within, j};

// A wall-clock oracle needs a QUIET machine: a measuring test takes it alone (write), the rest
// share it (read) — CI's two cores made parallel tests corrupt each other's time.
static MACHINE: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

/// A snapshot reply's NPY, decoded; empty when the reply holds none.
fn npy(r: &serde_json::Value) -> Vec<u8> {
    use base64::Engine;
    r["npy_b64"].as_str()
        .map(|s| base64::engine::general_purpose::STANDARD.decode(s).expect("valid base64"))
        .unwrap_or_default()
}

/// The f32 payload behind an NPY header; empty when there is no valid header.
fn npy_f32s(bytes: &[u8]) -> Vec<f32> {
    if bytes.len() < 10 {
        return Vec::new();
    }
    let data = 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    bytes[data..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn a_chain_runs_streams_and_follows_the_params_edited_under_it() {
    let _machine = MACHINE.blocking_read();
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.set_param(buf, "buffer", "size", 64);
    g.set_param(osc, "oscillator", "sfreq", 64.0);
    // Nothing has flowed yet, so the raw read answers null WITH the reason — and the ask itself
    // is what opens the slot's feed.
    let idle = g.call("node snapshot", j!({ "output": ep(hex(buf), "out") }));
    assert!(idle["frame"].is_null() && idle["reason"].as_str().is_some_and(|r| r.contains("emit")),
            "{idle}");
    let probe = g.probe(buf, "out"); // opened BEFORE the wire: the data services keep no history
    g.link(osc, "out", buf, "data");

    let full = g.until("the window to fill", |_| {
        probe.latest().filter(|d| f32s(d).len() == 64).map(|d| f32s(&d))
    });
    assert!(full.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "a unit sine: {:?}", &full[..4]);
    assert_eq!(probe.latest().unwrap().meta().sfreq(), Some(64.0), "sfreq rides the frame");

    g.set_param(buf, "buffer", "size", 16);
    g.until("the window to shrink under the running node", |_| {
        probe.latest().filter(|d| f32s(d).len() == 16).map(|_| ())
    });

    // The raw one-shot read: exactly the frame the node emitted, as NPY — no subscription, no
    // reduction, and refused by naming the real slots when the address is wrong.
    let why = g.refuse("node snapshot", j!({ "output": ep(hex(buf), "psd") }));
    // `it has:` anchors the real-slot list — a bare `out` also matches "output" in the same line.
    assert!(why.contains("no output slot `psd`") && why.contains("it has: out"), "{why}");
    let snap = g.until("a 16-wide raw frame in the snapshot cache", |g| {
        Some(g.call("node snapshot", j!({ "output": ep(hex(buf), "out") })))
            .filter(|r| npy_f32s(&npy(r)).len() == 16)
    });
    let bytes = npy(&snap);
    assert!(bytes.starts_with(b"\x93NUMPY\x01\x00"), "an NPY magic: {:?}", &bytes[..10]);
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    assert_eq!((10 + hlen) % 64, 0, "the data offset is 64-aligned");
    let header = String::from_utf8_lossy(&bytes[10..10 + hlen]).to_string();
    assert!(header.contains("'descr': '<f4'") && header.contains("(16,)"), "{header}");
    let vals = npy_f32s(&bytes);
    assert!(vals.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "the unit sine, raw: {vals:?}");
    assert_eq!(snap["meta"]["sfreq"], 64.0, "meta rides the snapshot: {snap}");

    let mut ev = g.events();
    let stats = g.until("a node_stats broadcast", |_| {
        let p = ev.next("node_stats");
        (p["node"] == hex(osc)).then_some(p)
    });
    assert!(stats["stats"]["updates_per_second"].as_f64().is_some_and(|r| r > 0.0), "{stats}");
}

#[test]
fn a_producer_paces_itself_to_its_rate_cap_and_follows_a_live_change() {
    let _machine = MACHINE.blocking_write();
    // Counting emitted frames is the only way to see a cap: a stated value reads correct anyway.
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let probe = g.probe(osc, "out");
    g.set_param(osc, "common", "max_frequency", 5.0);
    g.ready(osc);

    // Read from the index STAMP: a data wire is one deep, so a poll loop counts its own rate.
    let runs = |window: Duration| {
        let (mut first, mut last, end) = (None, 0, Instant::now() + window);
        while Instant::now() < end {
            if let Some(i) = probe.latest().and_then(|d| d.meta().index()) {
                first.get_or_insert(i);
                last = i;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        last - first.unwrap_or(last)
    };
    let slow = runs(Duration::from_millis(800));
    assert!(slow <= 8, "5 Hz produced {slow} frames in 0.8 s — the cap is not honoured");

    g.set_param(osc, "common", "max_frequency", 60.0);
    g.until("the re-paced producer", |_| (runs(Duration::from_millis(400)) > 8).then_some(()));

    // The delivered rate sits just UNDER the cap, never over — and a low cap hides that, so the
    // window that judges it is a fast one.
    g.set_param(osc, "oscillator", "sfreq", 1000.0);
    g.set_param(osc, "common", "max_frequency", 200.0);
    runs(Duration::from_millis(300)); // let the new cap take hold before the window that judges it
    // The floor is the MACHINE's, not a constant: the pacer spends one sleep per period, so what
    // a 5 ms sleep costs here bounds any cap — macOS CI rounds it to ~16 ms and tops out near 60.
    let t0 = Instant::now();
    for _ in 0..40 {
        std::thread::sleep(Duration::from_millis(5));
    }
    let paceable = (40_000u128 / t0.elapsed().as_millis().max(1)).min(200) as u64;
    let fast = runs(Duration::from_millis(1000));
    assert!(fast <= 201, "a 200 Hz cap delivered {fast} frames in a second — OVER the cap");
    assert!(fast >= paceable * 85 / 100,
            "a 200 Hz cap delivered {fast} frames in a second, where this machine paces {paceable}");
}

#[test]
fn each_way_a_node_can_fail_is_reported_and_none_of_them_stops_the_patch() {
    let _machine = MACHINE.blocking_read();
    // Containment: `setup` runs under the graph mutex the bridge unwraps, so a panic must not poison it.
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
    let _machine = MACHINE.blocking_read();
    // A `required` slot lets `process` read it unconditionally, which is worth something only if
    // the runtime enforces the refusal.
    let g = Goofi::new();
    let need = g.add("_TestRequired");
    let probe = g.probe(need, "out");
    // The gate is on PRESENCE, never on wiring.
    g.set_param(need, "common", "autotrigger", true);
    g.set_param(need, "common", "max_frequency", 20.0);
    let why = g.until("the refusal", |g| g.error(need));
    assert!(why.contains("in"), "the refusal names the slot that is empty: {why}");
    assert!(g.stays(|_| probe.latest().is_none()), "and nothing ran");

    let src = g.add("_TestCounter");
    g.link(src, "out", need, "in");
    g.until("the fed node to run", |_| probe.latest().map(|_| ()));
    assert!(g.until("the error to clear", |g| g.error(need).is_none().then_some(true)));
}

/// A node whose FIRST instance fails to boot and whose second succeeds, so a restart shows as recovery.
static FLAKY: goofi_node::NodeManifest = goofi_node::NodeManifest {
    type_name: "_TestFlaky",
    category: "test",
    doc: "fails setup once, then succeeds",
    inputs: &[],
    outputs: &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }],
    params: &[],
    isolation: &goofi_node::NATIVE,
    producer: true,
    factory: || unreachable!("a dyn type is built by its registered factory"),
};

/// Its frames carry WHICH instance emitted them; a per-instance run count cannot say.
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
    let _machine = MACHINE.read().await;
    // A rebirth publishes under a NEW service name, so a stale subscriber never errors, it just stops.
    let g = Goofi::new();
    let builds = Arc::new(AtomicUsize::new(0));
    g.register_dyn(&FLAKY, Box::new(move |_| {
        Box::new(Flaky { generation: builds.fetch_add(1, Ordering::SeqCst) as f32, n: 0.0 })
    }));
    let base = g.serve().await;
    let uid = g.add("_TestFlaky");
    // Paced, so the run count steps in ones: an uncapped producer laps its own viewer.
    g.set_param(uid, "common", "max_frequency", 10.0);

    let why = tokio::task::block_in_place(|| g.until("the first instance to fail", |g| g.error(uid)));
    assert!(why.contains("the device did not open"), "{why}");

    g.call("node restart", j!({ "node": hex(uid) }));
    tokio::task::block_in_place(|| {
        g.until("the second instance to boot clean", |g| g.error(uid).is_none().then_some(()))
    });

    let mut v = Viewer::open(&base, &hex(uid), "out").await;
    assert_eq!(f32s(&v.decoded().await)[0], 1.0, "the stream is live on the recovered generation");
    let before = g.state.graph.lock().unwrap().output_service_of(uid, "out");
    g.call("node restart", j!({ "node": hex(uid) }));
    assert_ne!(g.state.graph.lock().unwrap().output_service_of(uid, "out"), before,
               "a rebirth is a new name");
    // For up to one rehome interval the reducer is still listening on the dead name.
    v.until(|d| f32s(d)[0] == 2.0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_viewers_of_one_slot_share_one_reducer_and_each_gets_what_it_can_draw() {
    let _machine = MACHINE.read().await;
    let g = Goofi::new();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    // Far more samples per frame than any viewer asks for, so there is a reduction to fold at all.
    g.set_param(osc, "oscillator", "sfreq", 20000.0);
    let key = (osc, "out".to_string());

    let spec = |max: usize| j!([{ "dtype": "array", "ndim": [["le", 2]], "dims": [],
                                  "reduce": [{ "dim": -1, "max": max, "method": "envelope" }] }]);
    let mut wide = Viewer::open(&base, &hex(osc), "out").await;
    wide.view(spec(256)).await;
    let mut narrow = Viewer::open(&base, &hex(osc), "out").await;
    narrow.view(spec(32)).await;
    // The count separates "one reduce per frame" from "one per subscriber"; two viewers look alike.
    let mut rest = Vec::new();
    for _ in 0..4 {
        let mut v = Viewer::open(&base, &hex(osc), "out").await;
        v.view(spec(32)).await;
        rest.push(v);
    }

    // The fold takes the largest need per dim, and an envelope of width W carries 2·W samples.
    for v in [&mut wide, &mut narrow] {
        let d = v.until(|d| f32s(d).len() == 512).await;
        assert!(d.meta().reduced().is_some(), "the frame says it is a reduction");
    }
    assert_eq!(g.state.reducers.active_slots(), 1, "six viewers, one reducer");
    assert_eq!(g.state.reducers.subscribers(&key), 6);

    // A sub-patch boundary port is a NAMING indirection over this same stream — it never runs and
    // never holds a frame — so a viewer on one has to land on the reducer already here rather than
    // opening a second on a slot that produces nothing.
    let buf = g.add("Buffer");
    g.link(osc, "out", buf, "data");
    let inst = g.call("nodes group", j!({ "nodes": [hex(buf)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    // The group minted the port itself: the cable it cut is what a boundary port IS.
    let port = g.ports(&inst).first().cloned().expect("the cut is exposed as a port");
    let mut through = Viewer::open(&base, &port, "value").await;
    through.view(spec(32)).await;
    assert!(!f32s(&through.until(|d| !f32s(d).is_empty()).await).is_empty(),
            "the port draws the stream behind it");
    assert_eq!(g.state.reducers.active_slots(), 1, "…on the reducer that was already open");
    assert_eq!(g.state.reducers.subscribers(&key), 7);

    // An UNWIRED port is a node with no data, never a node that is absent: its socket opens and
    // idles, exactly as one on a leaf nobody has connected does. Nothing is behind it, so it joins
    // no reducer and opens none.
    let mid = g.call("node add", j!({ "type": "Buffer", "inst_id": inst, "pos": [0.0, 0.0] }))
        ["uid"].as_str().unwrap().to_string();
    let bare = g.call("node add", j!({ "type": "InArray", "inst_id": inst, "pos": [0.0, 0.0] }))
        ["uid"].as_str().unwrap().to_string();
    let mut pending = Viewer::open(&base, &bare, "value").await;
    pending.view(spec(32)).await;
    assert_eq!(g.state.reducers.active_slots(), 1, "nothing is behind it yet, so no second reducer");
    assert_eq!(g.state.reducers.subscribers(&key), 7, "and it joined nothing");

    // Wiring BOTH sides is what puts a stream behind it, and neither half alone does — so the
    // socket has to still be there for the second one. It then joins the reducer already serving
    // that stream, rather than staying frozen on the answer it got when it opened.
    g.call("link add", j!({ "from": ep(&bare, "value"), "to": ep(&mid, "data") }));
    assert_eq!(g.state.reducers.subscribers(&key), 7, "the inside alone feeds it nothing");
    g.call("link add", j!({ "from": ep(hex(osc), "out"), "to": ep(&inst, &bare) }));
    assert!(holds_within(Duration::from_secs(5), || g.state.reducers.subscribers(&key) == 8).await,
            "the open socket joined the stream its port now stands in front of");
    assert_eq!(g.state.reducers.active_slots(), 1, "still one reducer for the one physical slot");
    assert!(!f32s(&pending.until(|d| !f32s(d).is_empty()).await).is_empty(),
            "and it draws, on a socket that never closed");
    drop(pending);
    assert!(holds_within(Duration::from_secs(5), || g.state.reducers.subscribers(&key) == 7).await);
    drop(through);
    assert!(holds_within(Duration::from_secs(5), || g.state.reducers.subscribers(&key) == 6).await);
    g.call("node remove", j!({ "node": inst }));

    let passes = g.state.reducers.reductions(&key);
    tokio::time::sleep(Duration::from_millis(400)).await;
    let grew = g.state.reducers.reductions(&key) - passes;
    assert!(grew < 30, "{grew} reduce passes for ~12 frames — the reducer is running per subscriber");

    drop(rest);
    drop(narrow);
    assert!(holds_within(Duration::from_secs(5), || g.state.reducers.subscribers(&key) == 1).await);

    // One emit per hundred seconds, so every frame below can only be the one the reducer holds.
    g.set_param(osc, "common", "max_frequency", 0.01);
    tokio::time::sleep(Duration::from_millis(250)).await; // the last emit in flight lands

    drop(wide);
    assert!(holds_within(Duration::from_secs(5), || g.state.reducers.subscribers(&key) == 0).await,
            "the last viewer left");
    assert_eq!(g.state.reducers.active_slots(), 1,
               "the reducer went with its last viewer, and the slot's only copy of the last frame \
                went with it — the producer keeps no history, so the next viewer has nothing to draw");

    // A closing socket is no evidence that a slot stopped being watched.
    let mut back = Viewer::open(&base, &hex(osc), "out").await;
    back.view(spec(32)).await;
    let cached = back.until(|d| !f32s(d).is_empty()).await.meta().index();
    assert!(cached.is_some(), "a served frame carries the emit index of the producer's own frame");
    drop(back);

    // 300 rounds is past the data service's own `max_nodes` (256), so a rebuild that does not give
    // its node and subscriber back spends a ceiling the viewer count never reaches.
    for round in 0..300 {
        let mut v = Viewer::open(&base, &hex(osc), "out").await;
        v.view(spec(32)).await;
        let served = v.until(|d| !f32s(d).is_empty()).await;
        assert_eq!(served.meta().index(), cached, "round {round}: not the frame the slot held");
        assert_eq!(g.state.reducers.active_slots(), 1, "round {round}: one viewer, one reducer");
        drop(v);
        // Tight rather than `holds_within`: the round must SEE zero, and 300 polls of 25 ms is 8 s.
        let gone = Instant::now() + Duration::from_secs(5);
        while g.state.reducers.subscribers(&key) != 0 && Instant::now() < gone {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(g.state.reducers.subscribers(&key), 0, "round {round}: the viewer's socket closed");
    }

    g.call("node remove", j!({ "node": hex(osc) }));
    assert!(holds_within(Duration::from_secs(5), || g.state.reducers.active_slots() == 0).await,
            "the node left and its reducer went with it");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_viewer_that_stops_answering_is_reclaimed_and_a_merely_slow_one_is_not() {
    let _machine = MACHINE.write().await;
    // The hard half is the second one: a viewer on a slow link must not be mistaken for a dead one.
    let g = Goofi::impatient();
    let base = g.serve().await;
    let osc = g.add("Oscillator");
    let key = (osc, "out".to_string());

    let dead = Viewer::open(&base, &hex(osc), "out").await;
    let mut slow = Viewer::open(&base, &hex(osc), "out").await;
    slow.decoded().await;
    assert_eq!(g.state.reducers.subscribers(&key), 2);

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
    let _machine = MACHINE.blocking_write();
    // A node observes its halt flag only BETWEEN runs, so exit waits to a CEILING, never a join.
    let g = Goofi::new();
    let slow = g.add("_TestSlow");
    let other = g.add("_TestSlow");
    std::thread::sleep(Duration::from_millis(60)); // both threads are now inside a ten-second run

    let t0 = Instant::now();
    g.call("node remove", j!({ "node": hex(slow) }));
    assert!(t0.elapsed() < Duration::from_millis(100),
            "the delete took {:?} — it waited on the busy node under the graph lock", t0.elapsed());

    // A Python node's build EXECUTES its module, so the honest fixture is the dyn seam a discovered
    // node arrives through, not a trivial native factory.
    struct Echo;
    impl goofi_node::Node for Echo {
        fn process(&mut self, i: &goofi_node::Inputs<'_>, o: &mut goofi_node::Outputs<'_>,
                   _: &mut goofi_node::NodeCtx, _: &goofi_node::Params<'_>)
                   -> goofi_node::NodeResult {
            if let Some(d) = i.get("in") {
                o.set("out", d.clone());
            }
            Ok(())
        }
    }
    static SLOW_IN: &[goofi_node::SlotDecl] = &[goofi_node::SlotDecl {
        name: "in", kind: goofi_core::SlotType::Array,
        trigger_process: true, multi: false, required: false }];
    static SLOW_OUT: &[goofi_node::OutputDecl] =
        &[goofi_node::OutputDecl { name: "out", kind: goofi_core::SlotType::Array }];
    static SLOW_BUILD: goofi_node::NodeManifest = goofi_node::NodeManifest {
        type_name: "_TestSlowBuild", category: "test", doc: "takes 700 ms to construct",
        inputs: SLOW_IN, outputs: SLOW_OUT, params: &[],
        isolation: &goofi_node::NATIVE, producer: false,
        factory: || unreachable!("a dyn type is built by its registered factory"),
    };
    g.register_dyn(&SLOW_BUILD, Box::new(|_| {
        std::thread::sleep(Duration::from_millis(700));
        Box::new(Echo)
    }));

    let src = g.add("_TestCounter");
    let t0 = Instant::now();
    let heavy = g.add("_TestSlowBuild");
    let added = t0.elapsed();
    assert!(added < Duration::from_millis(250),
            "the add took {added:?} — the instance was built under the graph lock");
    assert_eq!(g.stage(heavy), "creating", "a node still being built says it is being built");
    let t0 = Instant::now();
    let quick = g.add("_TestSlow");
    assert!(t0.elapsed() < Duration::from_millis(250),
            "an op behind the build took {:?}", t0.elapsed());

    // A node is ADDRESSABLE only once ready, so this wire is planned against one that cannot hear it.
    let probe = g.probe(heavy, "out");
    g.link(src, "out", heavy, "in");
    assert_eq!(g.stage(heavy), "creating", "…and it is still building while that wire is planned");
    g.ready(heavy);
    g.until("the wire planned during the build to carry a frame", |_| probe.latest());
    g.call("node remove", j!({ "node": hex(quick) }));

    let t0 = Instant::now();
    g.state.graph.lock().unwrap().shutdown();
    // Five seconds sits between the two answers: the ceiling is two and a JOIN would cost ten.
    assert!(t0.elapsed() < Duration::from_secs(5),
            "the exit took {:?} — it JOINED the busy node instead of waiting to a ceiling", t0.elapsed());
    let _ = other;

    // An idle node parks on its doorbell with no timeout, so only `signal_stop` ringing it ends the wait.
    let idle = Goofi::new();
    for _ in 0..8 {
        idle.ready(idle.add("Buffer")); // unwired, so it parks rather than running
    }
    let t0 = Instant::now();
    idle.state.graph.lock().unwrap().shutdown();
    assert!(t0.elapsed() < Duration::from_millis(500),
            "an exit with nothing but parked nodes took {:?} — a doorbell was lost and it waited \
             out the ceiling", t0.elapsed());
}

#[test]
fn a_refreshable_param_is_re_enumerated_on_the_nodes_own_thread() {
    let _machine = MACHINE.blocking_read();
    // Options live in runtime state, never the doc, so they reach a client only through the status echo.
    let g = Goofi::new();
    let picker = g.add("_TestPicker");
    let mute = g.add("_TestMute");
    g.ready(picker);
    g.ready(mute);
    let mut ev = g.events();

    g.call("node param refresh", j!({ "node": hex(picker), "param": "io/device" }));
    let p = g.until("the picker's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(picker)).then_some(p)
    });
    let options = p["params"]["io"]["device"]["options"].as_array().cloned().unwrap_or_default();
    assert!(options.iter().any(|o| o.as_str().is_some_and(|s| s.starts_with("dev"))),
            "the re-enumerated list reached the client: {options:?}");
    assert_eq!(p["refreshed_params"], j!([["io", "device"]]), "…and the spinner is cleared");

    // A node with no hook must still get its echo, or the button spins for its full safety timeout.
    g.call("node param refresh", j!({ "node": hex(mute), "param": "io/device" }));
    let p = g.until("the mute picker's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(mute)).then_some(p)
    });
    assert_eq!(p["refreshed_params"], j!([["io", "device"]]), "the spinner is cleared regardless");
    assert_eq!(p["params"]["io"]["device"]["options"], j!(["none"]), "options left as declared");

    // A fixed list is refused, which is what lifts the spinner on the frontend's side.
    let osc = g.add("Oscillator");
    let why = g.refuse("node param refresh",
                       j!({ "node": hex(osc), "param": "oscillator/waveform" }));
    assert!(why.contains("not refreshable"), "{why}");
}
