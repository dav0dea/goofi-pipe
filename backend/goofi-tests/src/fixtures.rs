//! Deterministic `_`-prefixed test nodes that every suite can reach, kept out of the palette by
//! the prefix and registered at harness boot through the one door a discovered type takes. Keep
//! this set SMALL: a node earns a place only for a runtime behaviour.

use std::sync::atomic::{AtomicU64, Ordering};

use goofi_core::{Data, Meta, SlotType};
use goofi_graph::Graph;
use goofi_node::{
    Engine, GraphView, LibraryEntry, NodeManifest, NodeStage, OutputDecl, ParamDecl, ParamGroups, ParamKey,
    ParamSpec, Params, SlotDecl, Status, Touched, Uid,
};
use goofi_signal_sdk::{Inputs, Node, NodeCtx, NodeResult, Outputs};

const fn manifest(
    type_name: &'static str,
    doc: &'static str,
    inputs: &'static [SlotDecl],
    outputs: &'static [OutputDecl],
    params: &'static [ParamDecl],
    producer: bool,
) -> NodeManifest {
    NodeManifest { type_name, category: "test", doc, inputs, outputs, params, producer }
}

fn add(g: &mut Graph, manifest: NodeManifest, make: fn() -> Box<dyn Node>) {
    let manifest: &'static NodeManifest = Box::leak(Box::new(manifest));
    goofi_bridge::register_dyn_type(g, manifest, Box::new(move |_| make()), &goofi_node::NATIVE);
}

/// Every fixture into `g`.
pub fn register(g: &mut Graph) {
    add(g, manifest("_TestEcho", "passes its input straight through", IN_ARRAY, OUT_ARRAY, NO_PARAMS, false), || Box::new(Echo));
    add(g, manifest("_TestSink", "consumes a wire and carries one param", IN_ARRAY, &[], SINK_PARAMS, false), || Box::new(Sink));
    add(g, manifest("_TestScalar", "emits its `control/value` as a one-element frame", &[], OUT_ARRAY, SINK_PARAMS, true), || Box::new(Scalar));
    add(g, manifest("_TestFail", "process always errors", &[], OUT_ARRAY, NO_PARAMS, true), || Box::new(Failing));
    add(g, manifest("_TestPanic", "process panics", &[], OUT_ARRAY, NO_PARAMS, true), || Box::new(Panicking));
    add(g, manifest("_TestSetupFail", "setup always errors, so process never runs", &[], OUT_ARRAY, NO_PARAMS, true), || Box::new(SetupFail));
    add(g, manifest("_TestSlow", "one run takes ten seconds", &[], OUT_ARRAY, NO_PARAMS, true), || Box::new(Slow));
    add(g, manifest("_TestCounter", "emits its own run count", IN_ARRAY, OUT_ARRAY, NO_PARAMS, true), || Box::new(Counter::default()));
    add(g, manifest("_TestParamWrites", "emits how many param writes were delivered", IN_ARRAY, OUT_ARRAY, SINK_PARAMS, false), || Box::new(ParamWrites::default()));
    add(g, manifest("_TestRequired", "refuses to run without its input", IN_REQUIRED, OUT_ARRAY, NO_PARAMS, false), || Box::new(RequiredCounter::default()));
    add(g, manifest("_TestGrid", "a [3, 4] frame of three offset rising signals", &[], OUT_ARRAY, NO_PARAMS, true), || Box::new(Grid::default()));
    add(g, manifest("_TestPicker", "a refreshable device list", &[], &[], PICKER_PARAMS, false), || Box::new(Picker));
    add(g, manifest("_TestMute", "a refreshable list with no hook behind it", &[], &[], PICKER_PARAMS, false), || Box::new(MutePicker));
    add(g, manifest("_TestConst", "constant float32 array source (value+length) — hidden test/bench scaffolding.", &[], OUT_ARRAY, CONST_PARAMS, true), || Box::new(TestConst));
    add(g, manifest("_TestRamp", "a [C, T] ramp frame at `sfreq`: channel c rises from c to c + 1", &[], OUT_ARRAY, RAMP_PARAMS, true), || Box::new(Ramp));
}

static IN_ARRAY: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: false,
}];
static IN_REQUIRED: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUT_ARRAY: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
static NO_PARAMS: &[ParamDecl] = &[];

/// An engine that is nothing but a library: it advertises the names it was built with, and runs
/// them nowhere. What a test registers to put a second engine behind a name a shipped one offers.
pub struct LibraryEngine {
    id: &'static str,
    types: Vec<LibraryEntry>,
    pending: Vec<(Uid, Status)>,
}

impl LibraryEngine {
    pub fn named(id: &'static str, types: &[&str]) -> LibraryEngine {
        let types = types
            .iter()
            .map(|name| {
                let name: &'static str = Box::leak(name.to_string().into_boxed_str());
                let m = manifest(name, "a name a second engine also offers", &[], OUT_ARRAY, NO_PARAMS, true);
                LibraryEntry { manifest: Box::leak(Box::new(m)), isolation: &goofi_node::NATIVE }
            })
            .collect();
        LibraryEngine { id, types, pending: Vec::new() }
    }
}

impl Engine for LibraryEngine {
    fn id(&self) -> &'static str {
        self.id
    }

    fn doorbell_driven(&self) -> bool {
        false
    }

    fn dirty(&self) -> bool {
        false
    }

    fn library(&self) -> Vec<LibraryEntry> {
        self.types.clone()
    }

    fn insert(&mut self, uid: Uid, _type_name: &str, _generation: u64, _params: &ParamGroups) -> Option<String> {
        self.pending.push((uid, Status::Stage { stage: NodeStage::Ready }));
        None
    }

    fn remove(&mut self, uid: Uid) {
        self.pending.retain(|(u, _)| *u != uid);
    }

    fn settle(&mut self, _view: &GraphView<'_>, _touched: &[Touched]) {}

    fn drain(&mut self, apply: &mut dyn FnMut(Uid, Status)) -> usize {
        let pending = std::mem::take(&mut self.pending);
        let n = pending.len();
        for (uid, status) in pending {
            apply(uid, status);
        }
        n
    }

    fn refresh_param(&mut self, _uid: Uid, _key: ParamKey) {}

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn shutdown(&mut self) {}
}

struct Echo;
impl Node for Echo {
    fn process(&mut self, i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        if let Some(d) = i.get("input") {
            o.set("out", d.clone());
        }
        Ok(())
    }
}

static SINK_PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "control",
    name: "value",
    spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
    expression: None,
    doc: None,
}];

struct Sink;
impl Node for Sink {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
}

/// Emits its one param as a one-element frame — the scalar a reference copies.
struct Scalar;
impl Node for Scalar {
    fn process(&mut self, _i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        let value = p.f64("control", "value").unwrap_or(0.0) as f32;
        o.set("out", Data::array_f32(vec![1], value.to_le_bytes().to_vec(), Meta::new()).unwrap());
        Ok(())
    }
}

struct Failing;
impl Node for Failing {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Err("the sensor is unplugged".into())
    }
}

struct Panicking;
impl Node for Panicking {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        panic!("_TestPanic panics on purpose");
    }
}

struct SetupFail;
impl Node for SetupFail {
    fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Err("the device did not open".into())
    }
    fn process(&mut self, _i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        o.set("out", Data::array_f32(vec![1], 0f32.to_le_bytes().to_vec(), Meta::new()).unwrap());
        Ok(())
    }
}

/// Sleeps far past the shutdown ceiling, so a teardown that JOINED would hang instead of returning.
struct Slow;
impl Node for Slow {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        std::thread::sleep(std::time::Duration::from_secs(10));
        Ok(())
    }
}

/// Emits how many times it has run, as a length-1 array.
#[derive(Default)]
struct Counter {
    runs: u64,
}
impl Node for Counter {
    fn process(&mut self, _i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        self.runs += 1;
        let bytes = (self.runs as f32).to_le_bytes().to_vec();
        o.set("out", Data::array_f32(vec![1], bytes, Meta::new()).map_err(|e| e.to_string())?);
        Ok(())
    }
}

/// Emits how many param writes the engine DELIVERED — the settle dedup's own meter. Not a
/// producer, so no seeded binding can move the count between a test's reads.
#[derive(Default)]
struct ParamWrites {
    writes: u64,
}
impl Node for ParamWrites {
    fn process(&mut self, _i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        let bytes = (self.writes as f32).to_le_bytes().to_vec();
        o.set("out", Data::array_f32(vec![1], bytes, Meta::new()).map_err(|e| e.to_string())?);
        Ok(())
    }
    fn on_param_changed(&mut self, _k: &ParamKey, _v: &goofi_core::Param) -> NodeResult {
        self.writes += 1;
        Ok(())
    }
}

/// The same, but its input is REQUIRED.
#[derive(Default)]
struct RequiredCounter {
    runs: u64,
}
impl Node for RequiredCounter {
    fn process(&mut self, i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        let _ = i.get("input").expect("a required slot is never empty when process runs");
        self.runs += 1;
        let bytes = (self.runs as f32).to_le_bytes().to_vec();
        o.set("out", Data::array_f32(vec![1], bytes, Meta::new()).map_err(|e| e.to_string())?);
        Ok(())
    }
}

/// A `[3, 4]` frame — the only source here that is not a vector. Each row rises, offset by a round
/// 100, so a leak between channels, a reversed window or a degenerate measure all show up.
#[derive(Default)]
struct Grid {
    /// Samples emitted so far; each frame continues the sequence rather than restarting it.
    n: u64,
}
impl Node for Grid {
    fn process(&mut self, _i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        let (rows, cols) = (3u64, 4u64);
        let first = self.n;
        self.n += cols;
        let buf = (0..rows)
            .flat_map(|r| {
                (0..cols).flat_map(move |t| {
                    let n = (first + t) as f64;
                    let v = r as f64 * 100.0 + n + 0.4 * (std::f64::consts::TAU * n / 8.0).sin();
                    (v as f32).to_le_bytes()
                })
            })
            .collect();
        let meta = Meta::new().with_sfreq(Some(256.0));
        let shape = vec![rows as usize, cols as usize];
        o.set("out", Data::array_f32(shape, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

/// Process-wide, because the answer must CHANGE between scans for a test to tell them apart.
static PICKER_SCANS: AtomicU64 = AtomicU64::new(0);

static PICKER_PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "io",
    name: "device",
    spec: ParamSpec::Str { default: "none", options: &["none"], refresh: true },
    expression: None,
    doc: None,
}];

struct Picker;
impl Node for Picker {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
    fn on_param_refreshed(&mut self, key: &ParamKey, _p: &Params<'_>) -> Option<Vec<String>> {
        if key.group != "io" || key.name != "device" {
            return None;
        }
        let n = PICKER_SCANS.fetch_add(1, Ordering::Relaxed);
        Some(vec![format!("dev{n}"), "none".to_string()])
    }
}

/// The same declaration with NO hook behind it, so the ⟳ spinner runs to its safety timeout.
struct MutePicker;
impl Node for MutePicker {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
}

/// `_TestConst` — a constant float32 array source.
struct TestConst;

impl Node for TestConst {
    fn process(&mut self, _inp: &Inputs<'_>, out: &mut Outputs<'_>, _ctx: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        let nan = p.bool("constant", "nan").unwrap_or(false);
        let value = if nan { f32::NAN } else { p.f64("constant", "value").unwrap_or(0.0) as f32 };
        let length = p.i64("constant", "length").unwrap_or(1).max(1) as usize;
        let buf: Vec<u8> = (0..length).flat_map(|_| value.to_le_bytes()).collect();
        let data = Data::array_f32(vec![length], buf, Meta::empty())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }
}

static CONST_PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "constant",
        name: "value",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The value every element of the emitted array carries."),
    },
    ParamDecl {
        group: "constant",
        name: "length",
        spec: ParamSpec::Int { default: 1, min: 1, max: 1_000_000 },
        expression: None,
        doc: Some("How many elements the emitted array has."),
    },
    ParamDecl {
        group: "constant",
        name: "nan",
        spec: ParamSpec::Bool { default: false },
        expression: None,
        doc: Some("Emit NaN in place of the value."),
    },
];

/// `_TestRamp` — one `[C, T]` frame per run, channel `c` rising from `c` to `c + 1` over
/// `length` samples at `sfreq`: ordered samples with a rate, which is what a crossing resamples.
struct Ramp;

impl Node for Ramp {
    fn process(&mut self, _inp: &Inputs<'_>, out: &mut Outputs<'_>, _ctx: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        let sfreq = p.f64("ramp", "sfreq").unwrap_or(256.0);
        let length = p.i64("ramp", "length").unwrap_or(512).max(1) as usize;
        let channels = p.i64("ramp", "channels").unwrap_or(1).max(1) as usize;
        let buf: Vec<u8> = (0..channels)
            .flat_map(|c| (0..length).flat_map(move |i| (c as f32 + i as f32 / length as f32).to_le_bytes()))
            .collect();
        let data = Data::array_f32(vec![channels, length], buf, Meta::new().with_sfreq(Some(sfreq)))
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }
}

static RAMP_PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "ramp",
        name: "sfreq",
        spec: ParamSpec::Float { default: 256.0, min: 1.0, max: 100_000.0 },
        expression: None,
        doc: None,
    },
    ParamDecl {
        group: "ramp",
        name: "length",
        spec: ParamSpec::Int { default: 512, min: 1, max: 1_000_000 },
        expression: None,
        doc: None,
    },
    ParamDecl {
        group: "ramp",
        name: "channels",
        spec: ParamSpec::Int { default: 1, min: 1, max: 16 },
        expression: None,
        doc: None,
    },
];
