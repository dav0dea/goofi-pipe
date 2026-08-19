//! The shared test-node library — deterministic `_`-prefixed nodes that EVERY suite can reach.
//!
//! `test_source`'s doc has always stated the intent: scaffolding lives here "so the engine, bridge,
//! and goofi-python test suites share one simple deterministic source instead of each defining its
//! own". Only `_TestConst` ever followed it. The engine grew twenty-eight more inside its own
//! `#[cfg(test)]` module, where `inventory` registers them for the engine's test binary alone — so
//! an integration test in `goofi-bridge` or `goofi-python` could not name one, and every such test
//! hand-rolled its own `NodeManifest` instead. That is the reason a headless integration test could
//! not replace a unit test, and it is what this module removes.
//!
//! **Not gated behind `cfg(test)`, and that is the point.** An integration test is a separate crate
//! linking the ordinary build; a node only registered under `cfg(test)` is invisible to it. These
//! ship in the binary and stay out of the palette on the `_` prefix, exactly as `_TestConst` does
//! (`goofi_bridge::schemas` and `catalog_type_names` both filter it).
//!
//! Keep this set SMALL. It exists to let one integration test stand in for many unit tests, not to
//! mirror every fixture a deleted unit test once had. A node earns a place here only when a
//! behaviour of the RUNTIME cannot be observed without it.

use std::sync::atomic::{AtomicU64, Ordering};

use goofi_core::{Data, Meta, SlotType};
use goofi_node::{
    default_factory, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl,
    Outputs, ParamDecl, ParamKey, ParamSpec, Params, SlotDecl,
};

/// One array input that wakes `process`.
static IN_ARRAY: &[SlotDecl] = &[SlotDecl {
    name: "in",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: false,
}];
/// The same slot, but the node refuses to run without it — the input contract's gate.
static IN_REQUIRED: &[SlotDecl] = &[SlotDecl {
    name: "in",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUT_ARRAY: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];
static NO_PARAMS: &[ParamDecl] = &[];

/// Every manifest here differs in four fields at most, so they are stated once rather than
/// twenty-eight times. This is the `const fn` the engine's test module already used internally; it
/// is public now so a test in any crate can build one too.
pub const fn manifest(
    type_name: &'static str,
    doc: &'static str,
    inputs: &'static [SlotDecl],
    outputs: &'static [OutputDecl],
    params: &'static [ParamDecl],
    producer: bool,
    factory: fn() -> Box<dyn Node>,
) -> NodeManifest {
    NodeManifest {
        type_name,
        category: "test",
        doc,
        inputs,
        outputs,
        params,
        isolation: Isolation::InProcess,
        producer,
        factory,
    }
}

// ---------------------------------------------------------------------------
// Passthrough and sink — the two shapes a wire needs at each end
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Echo;
impl Node for Echo {
    fn process(&mut self, i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        if let Some(d) = i.get("in") {
            o.set("out", d.clone());
        }
        Ok(())
    }
}
inventory::submit! {
    manifest("_TestEcho", "passes its input straight through", IN_ARRAY, OUT_ARRAY, NO_PARAMS, false, default_factory::<Echo>)
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
    manifest("_TestSink", "consumes a wire and carries one param", IN_ARRAY, &[], SINK_PARAMS, false, default_factory::<Sink>)
}

// ---------------------------------------------------------------------------
// Faults — the three ways a node can fail, each surfacing on a different channel
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Failing;
impl Node for Failing {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Err("the sensor is unplugged".into())
    }
}
inventory::submit! {
    manifest("_TestFail", "process always errors", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<Failing>)
}

#[derive(Default)]
struct Panicking;
impl Node for Panicking {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        panic!("_TestPanic panics on purpose");
    }
}
inventory::submit! {
    manifest("_TestPanic", "process panics", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<Panicking>)
}

#[derive(Default)]
struct SetupFail;
impl Node for SetupFail {
    fn setup(&mut self, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Err("the device did not open".into())
    }
    fn process(&mut self, _i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        // Never reached while setup fails — a node that failed to initialize does not run at all.
        o.set("out", Data::array_f32(vec![1], 0f32.to_le_bytes().to_vec(), Meta::new()).unwrap());
        Ok(())
    }
}
inventory::submit! {
    manifest("_TestSetupFail", "setup always errors, so process never runs", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<SetupFail>)
}

// ---------------------------------------------------------------------------
// Timing — a node that takes real time, and one that counts its own runs
// ---------------------------------------------------------------------------

/// Sleeps far past the shutdown ceiling, so a teardown that JOINED rather than waiting to a bound
/// would hang instead of returning.
#[derive(Default)]
struct Slow;
impl Node for Slow {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        std::thread::sleep(std::time::Duration::from_secs(10));
        Ok(())
    }
}
inventory::submit! {
    manifest("_TestSlow", "one run takes ten seconds", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<Slow>)
}

/// Emits how many times it has run, as a length-1 array. The only node here whose OUTPUT carries a
/// run count, so a test can assert a rate or a trigger without reading engine internals.
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
    manifest("_TestCounter", "emits its own run count", IN_ARRAY, OUT_ARRAY, NO_PARAMS, true, default_factory::<Counter>)
}

/// The same, but its input is REQUIRED: it must refuse to run until a frame is present, and report
/// why rather than running with a hole.
#[derive(Default)]
struct RequiredCounter {
    runs: u64,
}
impl Node for RequiredCounter {
    fn process(&mut self, i: &Inputs<'_>, o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        // Read unconditionally: a required slot is exactly the promise that this cannot be None.
        let _ = i.get("in").expect("a required slot is never empty when process runs");
        self.runs += 1;
        let bytes = (self.runs as f32).to_le_bytes().to_vec();
        o.set("out", Data::array_f32(vec![1], bytes, Meta::new()).map_err(|e| e.to_string())?);
        Ok(())
    }
}
inventory::submit! {
    manifest("_TestRequired", "refuses to run without its input", IN_REQUIRED, OUT_ARRAY, NO_PARAMS, false, default_factory::<RequiredCounter>)
}

/// A `[3, 4]` frame — the only source here that is not a vector, and the reason it exists. A
/// fixture that only ever carries one dimension cannot tell a node that PRESERVES rank from one
/// that flattens, which is the failure the whole signal set is built against.
///
/// Each row is the same rising signal offset by a round 100, so a channel that leaked into its
/// neighbour shows up as arithmetic that stops working. Rising strictly, so a window that came out
/// backwards does too; and carrying a small ripple, so a measure taken over it is not degenerate.
#[derive(Default)]
struct Grid {
    /// Samples emitted so far — the frame's values continue the sequence rather than restarting it.
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
    manifest("_TestGrid", "a [3, 4] frame of three offset rising signals", &[], OUT_ARRAY, NO_PARAMS, true, default_factory::<Grid>)
}

// ---------------------------------------------------------------------------
// A refreshable param — the ⟳ round trip, which has no other observable surface
// ---------------------------------------------------------------------------

/// How many times any `_TestPicker` has been asked to re-enumerate. A process-wide counter because
/// the answer has to CHANGE between scans for a test to tell a fresh scan from a cached one.
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
    manifest("_TestPicker", "a refreshable device list", &[], &[], PICKER_PARAMS, false, default_factory::<Picker>)
}

/// The same declaration with NO hook behind it — a node that offers nothing to re-enumerate. The
/// ⟳ spinner is cleared by the echo, so without one this button spins for its full safety timeout.
#[derive(Default)]
struct MutePicker;
impl Node for MutePicker {
    fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
}
inventory::submit! {
    manifest("_TestMute", "a refreshable list with no hook behind it", &[], &[], PICKER_PARAMS, false, default_factory::<MutePicker>)
}
