//! A scheduled engine beside the signal one (roadmap/multi-engine-graph.md): two skeletons — one
//! audio-shaped, one graphics-shaped — each publishing static data at its own fixed tick over
//! the shared transport, registered through the one seam and driven through the one op surface.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use goofi_audio_sdk::{AudioNode, Block, ParamDecl, ParamSpec, Port, PortMut, BLOCK};
use goofi_core::SlotType;
use goofi_node::{
    DrainWaker, Engine, EventId, GraphView, LibraryEntry, NodeManifest, NodeStage,
    OutputDecl, ParamGroups, Request, SlotDecl, Status, Touched, Uid,
};
use goofi_tests::{ep, f32s, hex, j, shape, Goofi};

static AUDIO_OUTS: &[OutputDecl] = &[
    OutputDecl { name: "out", kind: SlotType::Audio },
    OutputDecl { name: "echo", kind: SlotType::Array },
];
static AUDIO_INS: &[SlotDecl] = &[
    SlotDecl { name: "input", kind: SlotType::Array, trigger_process: false, multi: false, required: false },
    SlotDecl { name: "audio", kind: SlotType::Audio, trigger_process: false, multi: false, required: false },
];
static AUDIO: NodeManifest = NodeManifest {
    type_name: "SkelAudioOsc",
    category: "audio",
    doc: "a fixed block per tick, and the freshest boundary input echoed back",
    inputs: AUDIO_INS,
    outputs: AUDIO_OUTS,
    params: &[],
    producer: true,
};

static GFX_OUTS: &[OutputDecl] = &[OutputDecl { name: "frame", kind: SlotType::Array }];
static GFX: NodeManifest = NodeManifest {
    type_name: "SkelGfxFrame",
    category: "graphics",
    doc: "a static frame per tick",
    inputs: &[],
    outputs: GFX_OUTS,
    params: &[],
    producer: true,
};

/// The doorbells one output slot rings after each publish.
type Rings = Vec<(goofi_transport::Doorbell, EventId)>;

/// One skeleton node's tick-side state, rebuilt by `settle` from the settled view.
#[derive(Default)]
struct Feed {
    generation: u64,
    /// One entry per output slot: its publisher, and who to wake.
    outs: Vec<(&'static str, goofi_transport::BytePublisher, Rings)>,
    /// The boundary wires feeding `in`, drained latest-wins before each tick.
    ins: Vec<goofi_transport::ByteSubscriber>,
    /// The freshest boundary frame, kept ENCODED — the echo republishes it verbatim.
    latest_in: Option<Vec<u8>>,
}

struct Shared {
    feeds: HashMap<Uid, Feed>,
    /// The static payload every non-echo slot publishes, pre-encoded once.
    block: Vec<u8>,
    /// Last: every port above is built from it, and fields drop in declaration order.
    iox: goofi_transport::IoxNode,
}

/// A scheduled engine: one thread, one fixed tick, static data. No doorbells in — it drains its
/// boundary before each tick — and none of the signal engine's machinery.
struct Skeleton {
    id: &'static str,
    class: &'static NodeManifest,
    waker: Arc<DrainWaker>,
    shared: Arc<Mutex<Shared>>,
    pending: Vec<(Uid, Status)>,
    /// An insert or remove happened and only a settle can rebuild the feeds for it.
    dirty: bool,
    stop: Arc<AtomicBool>,
    tick: Option<std::thread::JoinHandle<()>>,
}

impl Skeleton {
    fn new(
        id: &'static str,
        class: &'static NodeManifest,
        block: &goofi_core::Data,
        period: Duration,
        waker: Arc<DrainWaker>,
    ) -> Skeleton {
        let shared = Arc::new(Mutex::new(Shared {
            feeds: HashMap::new(),
            block: goofi_codec::encode(block),
            iox: goofi_transport::iox_node().expect("an iceoryx2 node for the skeleton"),
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let tick = {
            let (shared, stop) = (shared.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    tick_once(&shared);
                    std::thread::sleep(period);
                }
            })
        };
        Skeleton {
            id,
            class,
            waker,
            shared,
            pending: Vec::new(),
            dirty: false,
            stop,
            tick: Some(tick),
        }
    }
}

/// One tick: drain every boundary input to its freshest frame, then publish and ring.
fn tick_once(shared: &Mutex<Shared>) {
    let Ok(mut s) = shared.lock() else { return };
    let mut fresh: HashMap<Uid, Vec<u8>> = HashMap::new();
    for (uid, feed) in &s.feeds {
        let mut newest = None;
        for sub in &feed.ins {
            while let Ok(Some(sample)) = sub.receive() {
                newest = Some(sample.payload().to_vec());
            }
        }
        if let Some(bytes) = newest {
            fresh.insert(*uid, bytes);
        }
    }
    for (uid, bytes) in fresh {
        s.feeds.get_mut(&uid).expect("just seen").latest_in = Some(bytes);
    }
    for feed in s.feeds.values() {
        for (slot, publisher, rings) in &feed.outs {
            let payload = match *slot {
                "echo" => match &feed.latest_in {
                    Some(bytes) => bytes.as_slice(),
                    None => continue,
                },
                _ => s.block.as_slice(),
            };
            goofi_transport::publish(publisher, payload, rings.iter().map(|(b, id)| (b, *id)));
        }
    }
}

impl Engine for Skeleton {
    fn id(&self) -> &'static str {
        self.id
    }

    fn doorbell_driven(&self) -> bool {
        false
    }

    fn dirty(&self) -> bool {
        self.dirty
    }

    fn library(&self) -> Vec<LibraryEntry> {
        vec![LibraryEntry { manifest: self.class, isolation: &goofi_node::NATIVE }]
    }

    fn normalize_params(
        &self,
        type_name: &str,
        supplied: Option<ParamGroups>,
    ) -> Result<ParamGroups, String> {
        if type_name != self.class.type_name {
            return Err(format!("no node type `{type_name}` in the {} library", self.id));
        }
        Ok(supplied.unwrap_or_else(|| self.class.default_params()))
    }

    fn insert(
        &mut self,
        uid: Uid,
        _type_name: &str,
        generation: u64,
        _params: &ParamGroups,
    ) -> Option<String> {
        self.shared
            .lock()
            .unwrap()
            .feeds
            .insert(uid, Feed { generation, ..Feed::default() });
        // A synchronous engine is ready the moment its insert answers.
        self.pending.push((uid, Status::Stage { stage: NodeStage::Ready }));
        self.dirty = true;
        self.waker.notify();
        None
    }

    fn remove(&mut self, uid: Uid) {
        self.shared.lock().unwrap().feeds.remove(&uid);
        self.pending.retain(|(u, _)| *u != uid);
        self.dirty = true;
    }

    fn settle(&mut self, view: &GraphView<'_>, _touched: &[Touched]) {
        self.dirty = false;
        let mut s = self.shared.lock().unwrap();
        let Shared { feeds, block, iox } = &mut *s;
        for (uid, node) in &view.nodes {
            if node.engine != self.id {
                continue;
            }
            let Some(feed) = feeds.get_mut(uid) else { continue };
            let base = goofi_transport::service_base(view.instance, *uid, node.generation);
            // A rebirth renamed every service, so its publishers are rebuilt; a same-generation
            // settle reuses each slot's publisher, taken by NAME before anything is created —
            // two publishers on one one-publisher service, however briefly, is a refusal.
            let mut held: HashMap<&'static str, goofi_transport::BytePublisher> =
                if feed.generation == node.generation {
                    std::mem::take(&mut feed.outs).into_iter().map(|(n, p, _)| (n, p)).collect()
                } else {
                    feed.outs.clear();
                    HashMap::new()
                };
            feed.generation = node.generation;
            let mut outs = Vec::new();
            for out in node.manifest.outputs {
                let publisher = held.remove(out.name).unwrap_or_else(|| {
                    let service = goofi_transport::output_service(&base, out.name);
                    let svc = goofi_transport::data_service(iox, &service)
                        .expect("the skeleton's data service");
                    goofi_transport::publisher(&svc, out.name, 64 * 1024)
                        .expect("the skeleton's publisher")
                });
                let rings = rings_for(view, *uid, out.name);
                outs.push((out.name, publisher, rings));
            }
            feed.outs = outs;
            feed.ins = view
                .wires_into(*uid, "input")
                .filter_map(|(producer, slot)| {
                    let p = view.nodes.get(&producer)?;
                    let service = goofi_transport::output_service(
                        &goofi_transport::service_base(view.instance, producer, p.generation),
                        slot,
                    );
                    goofi_transport::open_output_subscriber(iox, &service).ok()
                })
                .collect();
            // The static block needs no tick to be current: publish on settle, so a consumer
            // wired later still sees data within one of ITS OWN wakes, and the tick re-publishes.
            for (slot, publisher, rings) in &feed.outs {
                if *slot != "echo" {
                    goofi_transport::publish(publisher, block, rings.iter().map(|(b, id)| (b, *id)));
                }
            }
        }
    }

    fn drain(&mut self, apply: &mut dyn FnMut(Uid, Status)) -> usize {
        let pending = std::mem::take(&mut self.pending);
        let n = pending.len();
        for (uid, status) in pending {
            apply(uid, status);
        }
        n
    }

    fn request(&mut self, _uid: Uid, _request: Request) {}

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(tick) = self.tick.take() {
            let _ = tick.join();
        }
        self.shared.lock().unwrap().feeds.clear();
    }
}

impl Drop for Skeleton {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Every doorbell one of this engine's output slots must ring, read off the view — only for
/// consumers whose engine wakes on doorbells.
fn rings_for(view: &GraphView<'_>, producer: Uid, slot: &'static str) -> Rings {
    let node_iox = goofi_transport::iox_node().expect("an iceoryx2 node for the rings");
    view.ringers(producer, slot)
        .into_iter()
        .filter_map(|r| {
            let door = goofi_transport::door_of(view, r.consumer)?;
            Some((goofi_transport::Doorbell::open(&node_iox, &door).ok()?, r.event_id))
        })
        .collect()
}

/// The one DSP node behind the audio skeleton, written as an author writes one: a level param
/// read by index, one output filled a channel at a time.
struct Tone;

goofi_audio_sdk::params! {
    LEVEL = ParamDecl {
        group: "tone",
        name: "level",
        spec: ParamSpec::Float { default: 0.5, min: 0.0, max: 1.0 },
        doc: Some("the block's level"),
        expression: None,
    },
}

impl AudioNode for Tone {
    fn prepare(&mut self, _rate: f64) {}
    fn process(&mut self, b: &mut Block<'_>) {
        let level = b.params[P::LEVEL].chan(0)[0];
        let out = b.outs[0].chan_mut(0);
        out.fill(level);
        out[0] = level / 2.0;
    }
}

/// The block the audio skeleton publishes: `Tone` run once over a local arena — 64 samples, the
/// first one distinctive so a binding that reads it lands a value no default carries.
fn audio_block() -> goofi_core::Data {
    let level = [0.5f32; BLOCK];
    let params = [Port::new(&level, 1, true)];
    let mut arena = [0f32; BLOCK];
    let mut outs = [PortMut::new(&mut arena, 1)];
    Tone.process(&mut Block { ins: &[], outs: &mut outs, params: &params });
    goofi_tests::frame(&arena)
}

/// An 8×8 static gradient — the graphics skeleton's whole output.
fn gfx_frame() -> goofi_core::Data {
    let bytes: Vec<u8> = (0..64).flat_map(|i| (i as f32 / 64.0).to_le_bytes()).collect();
    goofi_core::Data::array_f32(vec![8, 8], bytes, goofi_core::Meta::empty()).expect("a frame")
}

fn register_skeletons(t: &Goofi) {
    let mut g = t.state.graph.lock().unwrap();
    let waker = g.drain_waker();
    g.set_evaluator(Arc::new(goofi_tests::FirstVar));
    g.register_engine(Box::new(Skeleton::new(
        "skel",
        &AUDIO,
        &audio_block(),
        Duration::from_millis(5),
        waker.clone(),
    )));
    g.register_engine(Box::new(Skeleton::new(
        "graphics",
        &GFX,
        &gfx_frame(),
        Duration::from_millis(16),
        waker,
    )));
}

#[test]
fn a_scheduled_engine_beside_the_signal_one() {
    let t = Goofi::new();
    register_skeletons(&t);

    // Step: both libraries join the ONE palette, beside the signal catalog, and an audio slot
    // is a kind the palette spells.
    let types = t.call("library list", j!({}));
    let names: Vec<&str> =
        types["types"].as_array().unwrap().iter().filter_map(|r| r["type"].as_str()).collect();
    for want in ["SkelAudioOsc", "SkelGfxFrame", "Oscillator", "InAudio", "OutAudio"] {
        assert!(names.contains(&want), "`{want}` is missing from the merged palette: {names:?}");
    }
    let skel = types["types"].as_array().unwrap().iter().find(|r| r["type"] == "SkelAudioOsc").unwrap();
    assert_eq!(skel["output_slots"]["out"], "AUDIO", "{skel}");
    assert_eq!(skel["input_slots"]["audio"], "AUDIO", "{skel}");

    // Step: a skeleton node is born through the one op surface and reports ready through the
    // one health plane.
    let audio = t.add("SkelAudioOsc");
    t.ready(audio);
    let state = t.call("node state", j!({ "node": hex(audio) }))["text"].as_str().unwrap().to_string();
    assert!(state.contains("stage ready"), "the generic projection carries a foreign engine: {state}");

    // Step: a viewer on a skeleton slot is a plain subscriber on the derived name.
    let audio_probe = t.probe(audio, "out");
    let block = audio_probe.expect_frame(&mut t.state.graph.lock().unwrap(), "the audio block");
    assert_eq!(f32s(&block)[0], 0.25, "the static block, decoded off the shared transport");
    assert_eq!(f32s(&block).len(), 64);

    // Step: an audio output feeds an audio input, or an ARRAY input through the tap; nothing but
    // audio feeds an audio input, and the refusal names both kinds.
    let other = t.add("SkelAudioOsc");
    t.link(audio, "out", other, "audio");
    let refused = t.refuse("link add", j!({ "from": ep(hex(audio), "echo"), "to": ep(hex(other), "audio") }));
    assert!(refused.contains("ARRAY") && refused.contains("AUDIO"), "{refused}");

    // Step: an audio port is a node like any other — born in a sub-patch, wired on both faces,
    // relaying audio in and audio out — and the archive brings the whole arrangement back.
    let inst = t.call("nodes group", j!({ "nodes": [hex(other)], "pos": [0.0, 0.0] }))["inst_id"]
        .as_str().unwrap().to_string();
    let port_in = t.call("node add", j!({ "type": "InAudio", "inst_id": inst, "pos": [0.0, 0.0] }))["uid"]
        .as_str().unwrap().to_string();
    let port_out = t.call("node add", j!({ "type": "OutAudio", "inst_id": inst, "pos": [0.0, 0.0] }))["uid"]
        .as_str().unwrap().to_string();
    t.call("link add", j!({ "from": ep(&port_in, "value"), "to": ep(hex(other), "audio") }));
    t.call("link add", j!({ "from": ep(hex(other), "out"), "to": ep(&port_out, "value") }));
    t.call("link add", j!({ "from": ep(hex(audio), "out"), "to": ep(&port_in, "value") }));
    let wires = t.doc()["links"].clone();
    let archive = tempfile::tempdir().unwrap();
    let path = archive.path().join("audio-ports.gfi");
    t.call("session save", j!({ "path": path.to_string_lossy() }));
    let again = Goofi::new();
    register_skeletons(&again);
    again.call("session load", j!({ "path": path.to_string_lossy() }));
    let doc = again.doc();
    for (uid, ty) in [(&port_in, "InAudio"), (&port_out, "OutAudio")] {
        assert_eq!(doc["nodes"][uid]["type"], ty, "the port is back at its uid: {doc}");
    }
    assert_eq!(doc["nodes"][&port_in]["scope"], inst, "…in the sub-patch it is a port of");
    assert_eq!(doc["links"], wires, "every audio cable is back, the ports' inner wires included");
    drop(again);
    t.call("node remove", j!({ "node": inst }));

    // Step: skeleton → signal. The signal consumer subscribes to the derived name and the
    // skeleton rings its slot doorbell, so data crosses without a protocol.
    let echo = t.add("_TestEcho");
    t.link(audio, "out", echo, "input");
    let echoed = t.probe(echo, "out");
    let crossed = echoed.expect_frame(&mut t.state.graph.lock().unwrap(), "the crossed block");
    assert_eq!(f32s(&crossed), f32s(&block), "the signal node re-emits what the skeleton made");

    // Step: signal → skeleton. The boundary is drained at the SKELETON's own tick — no doorbell
    // exists on a scheduled engine — and the echo slot republishes the freshest frame verbatim.
    let osc = t.add("Oscillator");
    t.link(osc, "out", audio, "input");
    let back = t.probe(audio, "echo");
    let boundary = back.expect_frame(&mut t.state.graph.lock().unwrap(), "the boundary echo");
    assert!(!f32s(&boundary).is_empty(), "the oscillator's block came back through the boundary");

    // Step: cross-engine modulation. A signal param binds to nd('SkelAudioOsc'); the skeleton
    // rings the binding's own event id, the mailbox holds the frame latest-wins, and the
    // evaluated value lands exactly once — static data cannot spam writes.
    let meter = t.add("_TestParamWrites");
    t.link(audio, "out", meter, "input");
    t.call(
        "node param edit",
        j!({ "node": hex(meter), "param": "control/value",
             "expression": "nd('SkelAudioOsc')", "mode": "expression" }),
    );
    t.until("the binding's one write landed (init replay + the bound arrival)", |t| {
        let count = t.probe(meter, "out").frame(&mut t.state.graph.lock().unwrap())?;
        (f32s(&count)[0] >= 2.0).then_some(f32s(&count)[0])
    });
    assert!(
        t.stays(|t| {
            let count = t.probe(meter, "out").frame(&mut t.state.graph.lock().unwrap());
            count.is_none_or(|c| f32s(&c)[0] <= 2.0)
        }),
        "latest-wins modulation of a STATIC value writes once, however many ticks pass"
    );

    // Step: a reference obeys the same rule a cable does — a Float param may read an audio
    // output, a Str param may not — and the refusal names the kinds.
    let audio_name = t.doc()["nodes"][hex(audio)]["name"].as_str().unwrap().to_string();
    let bound = t.call(
        "node param edit",
        j!({ "node": hex(meter), "param": "control/value", "reference": format!("{audio_name}.out"), "mode": "reference" }),
    );
    assert!(bound["error"].is_null(), "a Float param references an audio output: {bound}");
    let picker = t.add("_TestPicker");
    let refused = t.call(
        "node param edit",
        j!({ "node": hex(picker), "param": "io/device", "reference": format!("{audio_name}.out"), "mode": "reference" }),
    );
    assert!(refused["error"].as_str().is_some_and(|e| e.contains("AUDIO") && e.contains("STRING")), "{refused}");
    t.call("node remove", j!({ "node": hex(picker) }));

    // Step: a restart is a rebirth through the same trait doors — new generation, new services.
    let generation = t.state.graph.lock().unwrap().node_generation(audio);
    let stale_probe = t.probe(audio, "out");
    t.call("node restart", j!({ "node": hex(audio) }));
    t.ready(audio);
    assert_eq!(
        t.state.graph.lock().unwrap().node_generation(audio),
        generation + 1,
        "the rebirth minted a fresh generation"
    );
    let reborn = t.probe(audio, "out");
    reborn.expect_frame(&mut t.state.graph.lock().unwrap(), "the reborn skeleton's block");
    let seen = stale_probe.count();
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(stale_probe.count(), seen, "the corpse's service name went silent");

    // Step: the second scheduled engine ticks beside the first, with its own shape and pace.
    let gfx = t.add("SkelGfxFrame");
    t.ready(gfx);
    let frame_probe = t.probe(gfx, "frame");
    let frame = frame_probe.expect_frame(&mut t.state.graph.lock().unwrap(), "the gfx frame");
    assert_eq!(shape(&frame), vec![8, 8], "the graphics skeleton's static frame");

    // Step: a remove through the one op surface tears the foreign node down and the rest stand.
    t.call("node remove", j!({ "node": hex(gfx) }));
    assert!(!t.nodes().contains(&hex(gfx)), "the graphics node is gone");
    audio_probe.expect_frame(&mut t.state.graph.lock().unwrap(), "the audio skeleton still runs");
}
