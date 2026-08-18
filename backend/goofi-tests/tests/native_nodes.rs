//! The native node library — Oscillator and Buffer — and the catalog invariants every node
//! type has to satisfy, whichever crate declared it.

/// Catalog validation: every `expression` a node declares must READ ONLY GLOBALS THAT EXIST
/// in a fresh patch — i.e. `goofi_core::globals::SYSTEM_GLOBALS`. Seeding runs on a fresh add
/// (`seed_default_expressions`), where the only globals in the store are the system ones, so a
/// typo'd `globals.defualt_ufreq` compiles, binds, and then errors at eval on every instance of
/// that node type — the param falls back to its literal and the node wears an error badge.
///
/// The "targets a declared param" check this test used to make cannot fail and is gone: an
/// `expression` lives ON the decl it targets, and `with_common` keeps whatever `common.*`
/// keys a node declared, so the target always exists by construction.
///
/// Cheap, evaluator-free, and runs over the whole linked catalog PLUS the universal `common`
/// group, which every node carries and which now declares one itself.
#[test]
fn every_declared_expression_reads_only_system_globals() {
    // The universal declarations are read AS EACH TYPE SEES THEM, so both the producer and the
    // consumer forms are covered (the catalog holds both kinds) — a declaration is free to
    // condition its source on the manifest, and one naming a global no fresh patch has would
    // otherwise fail on one kind of node only.
    let decls = goofi_node::catalog().flat_map(|m| {
        m.params
            .iter()
            .copied()
            .chain(goofi_node::common_decls(m))
            .map(move |d| (m.type_name, d))
    });
    for (owner, decl) in decls {
        let Some(expr) = decl.expression else { continue };
        assert!(!expr.source.trim().is_empty(), "{}: {}/{} has an empty expression", owner, decl.group, decl.name);
        for read in goofi_node::scan_globals(expr.source) {
            let name = read.name;
            assert!(
                goofi_core::globals::SYSTEM_GLOBALS.iter().any(|g| g.name == name),
                "{}: the expression on {}/{} reads `globals.{}`, which no fresh patch has",
                owner,
                decl.group,
                decl.name,
                name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Oscillator
// ---------------------------------------------------------------------------

// Linked for its side effect: the catalog is an `inventory` registry, and a crate nothing in this
// file NAMES is a crate rustc does not link — so `find("Oscillator")` would answer None.
use goofi_nodes as _;

use goofi_core::{Data, Meta, Param, Value};
use indexmap::IndexMap;
use goofi_node::{Inputs, NodeCtx, Outputs, ParamGroups, ParamKey, Params};
/// Build an Oscillator + its params (frequency/sfreq/amplitude/waveform), seeding the
/// mirrored `sfreq` field via the replay path the engine uses. Returns the params so
/// the caller passes them (cold reads) into `process`.
fn build(
    freq: f64,
    sfreq: f64,
    amp: f64,
    waveform: &str,
) -> (Box<dyn goofi_node::Node>, &'static goofi_node::NodeManifest, ParamGroups) {
    let m = goofi_node::find("Oscillator").expect("Oscillator registered");
    let mut params = m.default_params();
    params["oscillator"].insert("frequency".into(), Param::float(freq, 0.0, 1e6));
    params["oscillator"].insert("sfreq".into(), Param::float(sfreq, 1.0, 1e6));
    params["oscillator"].insert("amplitude".into(), Param::float(amp, 0.0, 1e6));
    params["oscillator"].insert(
        "waveform".into(),
        Param::Str { value: waveform.to_string(), options: None, refresh: false },
    );
    let mut node = (m.factory)();
    // Seed the mirrored sfreq (+ its re-anchor state).
    node.on_param_changed(&ParamKey::new("oscillator", "sfreq"), &params["oscillator"]["sfreq"])
        .unwrap();
    (node, m, params)
}

fn run_at(
    node: &mut Box<dyn goofi_node::Node>,
    m: &goofi_node::NodeManifest,
    params: &ParamGroups,
    now: f64,
) -> Option<Vec<f32>> {
    let inputs_map = IndexMap::new();
    let inp = Inputs::new(&inputs_map);
    let mut outbuf = m.output_buffer();
    let mut ctx = NodeCtx { now, ..Default::default() };
    {
        let mut out = Outputs::new(&mut outbuf);
        node.process(&inp, &mut out, &mut ctx, &Params::new(params)).unwrap();
    }
    outbuf.get("out").unwrap().as_ref().map(|d| match d.value() {
        Value::Array(s) => s
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        _ => panic!("expected array"),
    })
}

#[test]
fn first_tick_emits_nothing_then_paces_by_wall_clock() {
    let (mut node, m, params) = build(1.0, 1000.0, 1.0, "sine");
    // First tick anchors at now=0 -> zero elapsed -> no frame.
    assert!(run_at(&mut node, m, &params, 0.0).is_none(), "no time elapsed yet");
    // 10 ms later at 1 kHz -> exactly 10 samples.
    assert_eq!(run_at(&mut node, m, &params, 0.010).expect("a paced block").len(), 10);
    assert_eq!(run_at(&mut node, m, &params, 0.020).unwrap().len(), 10);
}

#[test]
fn drift_free_across_uneven_ticks() {
    // Cumulative samples equal round(sfreq·elapsed), never lost to per-tick rounding.
    let (mut node, m, params) = build(1.0, 1000.0, 1.0, "sine");
    run_at(&mut node, m, &params, 0.0); // anchor
    let a = run_at(&mut node, m, &params, 0.0015).map_or(0, |v| v.len()); // round(1.5)=2
    let b = run_at(&mut node, m, &params, 0.0035).map_or(0, |v| v.len()); // round(3.5)-2 = 2
    assert_eq!(a + b, 4, "cumulative tracks round(sfreq·elapsed): round(3.5)=4");
}

#[test]
fn emits_sine_values_and_sfreq_meta() {
    // freq = sfreq/4 -> phase step π/2 -> samples 0, 1, 0, -1 (× amplitude).
    let (mut node, m, params) = build(250.0, 1000.0, 2.0, "sine");
    let inputs_map = IndexMap::new();
    let inp = Inputs::new(&inputs_map);
    let mut o0 = m.output_buffer();
    node.process(&inp, &mut Outputs::new(&mut o0), &mut NodeCtx { now: 0.0, ..Default::default() }, &Params::new(&params)).unwrap();
    let mut o1 = m.output_buffer();
    node.process(&inp, &mut Outputs::new(&mut o1), &mut NodeCtx { now: 0.004, ..Default::default() }, &Params::new(&params)).unwrap();
    let d = o1.get("out").unwrap().as_ref().unwrap();
    assert_eq!(d.meta().sfreq(), Some(1000.0));
    if let Value::Array(s) = d.value() {
        assert_eq!(s.shape(), &[4]);
        let v = |i: usize| f32::from_le_bytes(s.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(v(0).abs() < 1e-5, "sin(0)*2 ~ 0");
        assert!((v(1) - 2.0).abs() < 1e-5, "sin(pi/2)*2 ~ 2");
        assert!((v(3) + 2.0).abs() < 1e-5, "sin(3pi/2)*2 ~ -2");
    } else {
        panic!("expected array");
    }
}

#[test]
fn square_waveform_is_plus_minus_amplitude() {
    // freq = sfreq/4 -> phases 0, π/2, π, 3π/2. Phase-based square: <π -> +1 else -1,
    // so [+1, +1, -1, -1].
    let (mut node, m, params) = build(250.0, 1000.0, 1.0, "square");
    let inputs_map = IndexMap::new();
    let inp = Inputs::new(&inputs_map);
    let mut o0 = m.output_buffer();
    node.process(&inp, &mut Outputs::new(&mut o0), &mut NodeCtx { now: 0.0, ..Default::default() }, &Params::new(&params)).unwrap();
    let mut o1 = m.output_buffer();
    node.process(&inp, &mut Outputs::new(&mut o1), &mut NodeCtx { now: 0.004, ..Default::default() }, &Params::new(&params)).unwrap();
    if let Value::Array(s) = o1.get("out").unwrap().as_ref().unwrap().value() {
        let v: Vec<f32> = s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert!(v.iter().all(|x| x.abs() == 1.0), "square is ±1, got {v:?}");
        assert_eq!(v, vec![1.0, 1.0, -1.0, -1.0]);
    } else {
        panic!("expected array");
    }
}

#[test]
fn same_value_sfreq_write_does_not_reanchor_pacing() {
    let (mut node, m, params) = build(1.0, 1000.0, 1.0, "sine");
    run_at(&mut node, m, &params, 0.0); // anchor at t=0
    assert_eq!(run_at(&mut node, m, &params, 0.010).unwrap().len(), 10); // emitted = 10
    // Redundant same-value sfreq write must NOT reset pacing.
    node.on_param_changed(&ParamKey::new("oscillator", "sfreq"), &Param::float(1000.0, 1.0, 1e6)).unwrap();
    // Drift-free pacing continues (10 more, not a re-anchor to zero elapsed).
    assert_eq!(run_at(&mut node, m, &params, 0.020).unwrap().len(), 10);
}

#[test]
fn changed_sfreq_reanchors_pacing() {
    let (mut node, m, params) = build(1.0, 1000.0, 1.0, "sine");
    run_at(&mut node, m, &params, 0.0);
    run_at(&mut node, m, &params, 0.010); // emitted = 10
    // A real change re-anchors: the next tick anchors at its own `now` -> no frame yet.
    node.on_param_changed(&ParamKey::new("oscillator", "sfreq"), &Param::float(500.0, 1.0, 1e6)).unwrap();
    assert!(run_at(&mut node, m, &params, 0.020).is_none(), "changed sfreq re-anchors -> no frame this tick");
}

#[test]
fn oscillation_frequency_is_independent_of_sfreq() {
    // The emitted signal's real-world frequency (cycles per wall-clock second) is set SOLELY by
    // `frequency`; `sfreq` only changes how many samples represent that second. Measure it by
    // counting upward zero-crossings over one wall-second of ticks, at two very different sfreqs.
    // (The perceived "sfreq changes the frequency" is a DISPLAY artifact — a fixed-sample Buffer +
    // a sample-index viewer x-axis shrinks the visible time window as sfreq rises — not the DSP.)
    fn cycles_in_one_second(sfreq: f64) -> usize {
        let (mut node, m, params) = build(3.0, sfreq, 1.0, "sine"); // 3 Hz
        let mut all: Vec<f32> = Vec::new();
        for k in 0..=30 {
            if let Some(v) = run_at(&mut node, m, &params, k as f64 / 30.0) {
                all.extend(v);
            }
        }
        all.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count()
    }
    let low = cycles_in_one_second(80.0);
    let high = cycles_in_one_second(2000.0);
    assert_eq!(low, high, "cycles/sec must not depend on sfreq: {low} (80 Hz) vs {high} (2 kHz)");
    assert!((3..=4).contains(&low), "frequency=3 -> ~3 cycles per wall-second, got {low}");
}

#[test]
fn a_non_finite_drive_does_not_poison_the_phase() {
    // A param expression can legitimately yield inf/NaN (`float('inf')`, or `nd()` on a node
    // emitting NaN) — the Float coercion deliberately does not reject it. Folding that into
    // `self.phase` would leave it NaN for the node's LIFETIME: `phase.rem_euclid(TAU)` maps
    // both ±inf and NaN to NaN, and `on_param_changed` re-anchors pacing but never resets
    // phase. So a non-finite drive is a boundary error, not new state.
    let (mut node, m, mut params) = build(1.0, 1000.0, 1.0, "sine");
    let inputs_map = IndexMap::new();
    let inp = Inputs::new(&inputs_map);
    let mut anchor = m.output_buffer();
    node.process(&inp, &mut Outputs::new(&mut anchor), &mut NodeCtx { now: 0.0, ..Default::default() }, &Params::new(&params)).unwrap();

    params["oscillator"].insert("frequency".into(), Param::float(f64::INFINITY, 0.0, 1e6));
    let mut poisoned = m.output_buffer();
    let r = node.process(&inp, &mut Outputs::new(&mut poisoned), &mut NodeCtx { now: 0.004, ..Default::default() }, &Params::new(&params));
    assert!(r.is_err(), "a non-finite frequency is a per-tick node error, not a frame");

    // The moment the expression yields a finite value again, the node emits real samples.
    params["oscillator"].insert("frequency".into(), Param::float(250.0, 0.0, 1e6));
    let v = run_at(&mut node, m, &params, 0.008).expect("a frame once the drive is finite again");
    assert!(v.iter().all(|x| x.is_finite()), "phase survived the non-finite tick: {v:?}");
}

#[test]
fn has_no_control_input_slots() {
    // The `frequency` control-input slot is gone — its role is now served by a param
    // expression (frequency = nd('lfo')...), so the oscillator is a pure producer.
    let m = goofi_node::find("Oscillator").expect("Oscillator registered");
    assert!(m.inputs.is_empty(), "oscillator is a producer with no input slots; got {:?}",
        m.inputs.iter().map(|s| s.name).collect::<Vec<_>>());
}

#[test]
fn defaults_to_the_producer_update_rate() {
    use goofi_node::{with_common, RunPolicy};
    let m = goofi_node::find("Oscillator").expect("Oscillator registered");
    // The live default is the global — editing globals.default_ufreq re-rates every Oscillator
    // (the P5 seeding mechanism drives the binding); the 30.0 literal is the graceful fallback.
    let decl = m
        .params
        .iter()
        .find(|d| d.group == "common" && d.name == "max_frequency")
        .expect("oscillator declares common.max_frequency");
    let expr = decl.expression.expect("oscillator binds its rate to the global");
    assert_eq!(expr.source, "globals.default_ufreq");
    // Without an evaluator, the fallback literal must still pace it at the producer default
    // (30 Hz) and free-run it — never unbounded (which would saturate the tick loop).
    let policy = RunPolicy::from_params(&with_common(m.default_params(), m));
    assert_eq!(policy.max_frequency, 30.0, "fallback rate is the producer default");
    assert!(policy.autotrigger, "the oscillator is a free-running producer");
}

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------


fn f32_frame(vals: &[f32]) -> Data {
    let buf: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    Data::array_f32(vec![vals.len()], buf, Meta::empty()).unwrap()
}

/// Params with `buffer.size` = `size` (cold-read live by `process`).
fn params_with_size(size: i64) -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("size".to_string(), Param::int(size, 1, 100));
    let mut groups = ParamGroups::new();
    groups.insert("buffer".to_string(), g);
    groups
}

fn run(node: &mut Box<dyn goofi_node::Node>, m: &goofi_node::NodeManifest, params: &ParamGroups, frame: Data) -> Data {
    let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    inmap.insert("data", Some(frame));
    let inp = Inputs::new(&inmap);
    let mut outbuf = m.output_buffer();
    let mut ctx = NodeCtx::new();
    {
        let mut out = Outputs::new(&mut outbuf);
        node.process(&inp, &mut out, &mut ctx, &Params::new(params)).unwrap();
    }
    outbuf.get("out").unwrap().as_ref().unwrap().clone()
}

fn to_vec(d: &Data) -> Vec<f32> {
    if let Value::Array(s) = d.value() {
        s.as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else {
        panic!("not array")
    }
}

#[test]
fn buffer_rolls_to_window_size() {
    let m = goofi_node::find("Buffer").unwrap();
    let mut node = (m.factory)();
    // `size` is read live from params — a size of 3, no on_param_changed needed.
    let params = params_with_size(3);

    let o1 = run(&mut node, m, &params, f32_frame(&[1.0, 2.0]));
    assert_eq!(to_vec(&o1), vec![1.0, 2.0]);

    let o2 = run(&mut node, m, &params, f32_frame(&[3.0, 4.0]));
    // last 3 of [1,2,3,4]
    assert_eq!(to_vec(&o2), vec![2.0, 3.0, 4.0]);
}

#[test]
fn default_params_declare_size() {
    let m = goofi_node::find("Buffer").unwrap();
    let p = m.default_params();
    assert_eq!(p["buffer"]["size"].as_i64(), Some(1000));
}

// ---------------------------------------------------------------------------
// The shared test-node library
// ---------------------------------------------------------------------------


fn emit(node: &mut Box<dyn goofi_node::Node>, m: &goofi_node::NodeManifest, params: &ParamGroups) -> Value {
    let inputs_map = IndexMap::new();
    let inp = Inputs::new(&inputs_map);
    let mut outbuf = m.output_buffer();
    node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new(), &Params::new(params)).unwrap();
    outbuf.get("out").unwrap().as_ref().expect("emitted a frame").value().clone()
}

#[test]
fn emits_a_constant_array_and_reacts_to_params() {
    let m = goofi_node::find("_TestConst").expect("_TestConst registered");
    assert_eq!(m.category, "test");
    let mut params = m.default_params();
    params["constant"].insert("value".into(), Param::float(2.5, -1.0e9, 1.0e9));
    params["constant"].insert("length".into(), Param::int(4, 1, 1_000_000));
    let mut node = (m.factory)();

    match emit(&mut node, m, &params) {
        Value::Array(s) => {
            assert_eq!(s.shape(), &[4]);
            assert_eq!(f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()), 2.5);
        }
        _ => panic!("expected array"),
    }

    // `length` is read live — changing the param map changes the next emit.
    params["constant"].insert("length".into(), Param::int(3, 1, 10));
    match emit(&mut node, m, &params) {
        Value::Array(s) => assert_eq!(s.shape(), &[3]),
        _ => panic!("expected array"),
    }
}

#[test]
fn every_test_node_is_registered_and_hidden_from_the_palette() {
    // The registration is what makes this library reachable from another crate's integration
    // test, and the `_` prefix is what keeps it out of the user's palette. Both or neither:
    // a node that registers without the prefix ships as a product node, and one with the
    // prefix that fails to register is invisible to the tests it exists for.
    let names: Vec<&str> = goofi_node::catalog().map(|m| m.type_name).collect();
    for want in [
        "_TestEcho", "_TestSink", "_TestFail", "_TestPanic", "_TestSetupFail", "_TestSlow",
        "_TestCounter", "_TestRequired", "_TestPicker", "_TestMute",
    ] {
        assert!(names.contains(&want), "{want} is not in the catalog: {names:?}");
        assert!(want.starts_with('_'), "{want} would show in the palette");
    }
}
