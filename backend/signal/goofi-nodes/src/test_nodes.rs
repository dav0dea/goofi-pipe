//! Deterministic `_`-prefixed test nodes that every suite can reach, kept out of the palette by
//! the prefix. Keep this set SMALL: a node earns a place only for a runtime behaviour.

use std::sync::atomic::{AtomicU64, Ordering};

use goofi_core::{Data, Meta, SlotType};
use goofi_node::{NodeManifest, OutputDecl, ParamDecl, ParamKey, ParamSpec, Params, SlotDecl};
use goofi_signal::{default_factory, NodeClass};
use goofi_signal_sdk::{Inputs, Node, NodeCtx, NodeResult, Outputs};

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

/// Builds a test-category registration.
pub const fn class(
    type_name: &'static str,
    doc: &'static str,
    inputs: &'static [SlotDecl],
    outputs: &'static [OutputDecl],
    params: &'static [ParamDecl],
    producer: bool,
    factory: fn() -> Box<dyn Node>,
) -> NodeClass {
    NodeClass {
        manifest: NodeManifest {
            type_name,
            category: "test",
            doc,
            inputs,
            outputs,
            params,
            producer,
        },
        isolation: &goofi_node::NATIVE,
        factory,
    }
}

#[derive(Default)]
struct Echo;
impl Node for Echo {
    fn process(&mut self, i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        if let Some(d) = i.get("input") {
            o.set("out", d.clone());
        }
        Ok(())
    }
}
inventory::submit! {
    class("_TestEcho", "passes its input straight through", IN_ARRAY, OUT_ARRAY, NO_PARAMS, false, default_factory::<Echo>)
}

static SINK_PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "control",
    name: "value",
    spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
    expression: None,
    doc: None,
}];

#[derive(Default)]
struct Sink;
impl Node for Sink {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
}
inventory::submit! {
    class("_TestSink", "consumes a wire and carries one param", IN_ARRAY, &[], SINK_PARAMS, false, default_factory::<Sink>)
}

/// Emits its one param as a one-element frame — the scalar a reference copies.
#[derive(Default)]
struct Scalar;
impl Node for Scalar {
    fn process(&mut self, _i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        let value = p.f64("control", "value").unwrap_or(0.0) as f32;
        o.set("out", Data::array_f32(vec![1], value.to_le_bytes().to_vec(), Meta::new()).unwrap());
        Ok(())
    }
}
inventory::submit! {
    class("_TestScalar", "emits its `control/value` as a one-element frame", &[], OUT_ARRAY, SINK_PARAMS, true, default_factory::<Scalar>)
}

#[derive(Default)]
struct Failing;
impl Node for Failing {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Err("the sensor is unplugged".into())
    }
}
inventory::submit! {
    class("_TestFail", "process always errors", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<Failing>)
}

#[derive(Default)]
struct Panicking;
impl Node for Panicking {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        panic!("_TestPanic panics on purpose");
    }
}
inventory::submit! {
    class("_TestPanic", "process panics", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<Panicking>)
}

#[derive(Default)]
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
inventory::submit! {
    class("_TestSetupFail", "setup always errors, so process never runs", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<SetupFail>)
}

/// Sleeps far past the shutdown ceiling, so a teardown that JOINED would hang instead of returning.
#[derive(Default)]
struct Slow;
impl Node for Slow {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        std::thread::sleep(std::time::Duration::from_secs(10));
        Ok(())
    }
}
inventory::submit! {
    class("_TestSlow", "one run takes ten seconds", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<Slow>)
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
inventory::submit! {
    class("_TestCounter", "emits its own run count", IN_ARRAY, OUT_ARRAY, NO_PARAMS, true, default_factory::<Counter>)
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
inventory::submit! {
    class("_TestParamWrites", "emits how many param writes were delivered", IN_ARRAY, OUT_ARRAY, SINK_PARAMS, false, default_factory::<ParamWrites>)
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
inventory::submit! {
    class("_TestRequired", "refuses to run without its input", IN_REQUIRED, OUT_ARRAY, NO_PARAMS, false, default_factory::<RequiredCounter>)
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
inventory::submit! {
    class("_TestGrid", "a [3, 4] frame of three offset rising signals", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<Grid>)
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

#[derive(Default)]
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
inventory::submit! {
    class("_TestPicker", "a refreshable device list", &[], &[], PICKER_PARAMS, false, default_factory::<Picker>)
}

/// The same declaration with NO hook behind it, so the ⟳ spinner runs to its safety timeout.
#[derive(Default)]
struct MutePicker;
impl Node for MutePicker {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
}
inventory::submit! {
    class("_TestMute", "a refreshable list with no hook behind it", &[], &[], PICKER_PARAMS, false, default_factory::<MutePicker>)
}
