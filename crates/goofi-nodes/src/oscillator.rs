//! Oscillator — a general low-frequency / biosignal-regime oscillator. Each tick it
//! emits the block of samples real time has advanced by (drift-free against
//! `meta["sfreq"]`), carrying phase forward so blocks join click-free. A `frequency`
//! slider (LFO range) and a `waveform` selector (sine/square/sawtooth/triangle) shape
//! the signal; a wired `frequency` control input overrides the slider each tick.
//! Self-contained (no audio-synth infrastructure) — one of the two seed nodes for the
//! redesigned library.
//!
//! `frequency`, `amplitude`, and `waveform` are cold params (read live from `p` each
//! tick). `sfreq` is the one stateful param: changing it must re-anchor the drift-free
//! pacing, so it is mirrored to a field via `on_param_changed`.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Meta, Value};
use goofi_node::{
    default_factory, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl,
    Outputs, ParamDecl, ParamKey, ParamSpec, Params, SlotDecl,
};
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy)]
enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

impl Waveform {
    fn parse(s: &str) -> Waveform {
        match s {
            "square" => Waveform::Square,
            "sawtooth" => Waveform::Sawtooth,
            "triangle" => Waveform::Triangle,
            _ => Waveform::Sine,
        }
    }
    /// The waveform's value at `phase` radians, in [-1, 1]. All are phase-aligned with
    /// sine (zero-rising at phase 0) except sawtooth, which ramps across the period.
    fn sample(self, phase: f64) -> f64 {
        match self {
            Waveform::Sine => phase.sin(),
            // Phase-based (not sign-of-sine, which flips on the π rounding boundary):
            // +1 over the first half of the period, −1 over the second.
            Waveform::Square => {
                if phase.rem_euclid(TAU) < PI {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => 2.0 * (phase.rem_euclid(TAU) / TAU) - 1.0,
            // Piecewise-linear (phase-aligned with sine: 0 at 0, +1 at π/2, -1 at 3π/2)
            // — no per-sample transcendentals.
            Waveform::Triangle => {
                let t = phase.rem_euclid(TAU) / TAU;
                if t < 0.25 {
                    4.0 * t
                } else if t < 0.75 {
                    2.0 - 4.0 * t
                } else {
                    4.0 * t - 4.0
                }
            }
        }
    }
}

#[derive(Default)]
struct Oscillator {
    /// Sample rate — mirrored from the `sfreq` param; changing it re-anchors pacing.
    sfreq: f64,
    /// Phase in radians, carried across ticks for click-free blocks.
    phase: f64,
    /// Wall-clock anchor (`ctx.now` at the first emit) — pacing is measured from here.
    start: Option<f64>,
    /// Total samples emitted since `start`; the running count keeps the per-tick block
    /// size drift-free (`n = round(sfreq·elapsed) − emitted`) rather than rounding each
    /// tick independently.
    emitted: u64,
}

impl Node for Oscillator {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        // `sfreq` is seeded (>= 1) via on_param_changed; an unseeded node emits nothing.
        let sfreq = self.sfreq;
        // Drift-free sample count: how many samples real time has advanced by since the
        // anchor, minus what we've already emitted. Zero on the first tick (now == anchor).
        let start = *self.start.get_or_insert(c.now);
        let total = (sfreq * (c.now - start)).round().max(0.0) as u64;
        let n = total.saturating_sub(self.emitted) as usize;
        if n == 0 {
            return Ok(());
        }
        self.emitted = total;

        // A wired `frequency` control input drives frequency in real time (its first
        // value); unwired, the `frequency` slider (a cold param) holds.
        let freq = first_f32(inp.get("frequency"))
            .map(|v| v as f64)
            .or_else(|| p.f64("oscillator", "frequency"))
            .unwrap_or(1.0);
        let amp = p.f64("oscillator", "amplitude").unwrap_or(1.0);
        let wave = Waveform::parse(p.str("oscillator", "waveform").unwrap_or("sine"));
        let step = TAU * freq / sfreq; // phase increment per sample
        let mut buf = Vec::with_capacity(n * 4);
        for _ in 0..n {
            buf.extend_from_slice(&((wave.sample(self.phase) * amp) as f32).to_le_bytes());
            self.phase += step;
        }
        self.phase = self.phase.rem_euclid(TAU); // keep bounded over long runs

        let meta = Meta { sfreq: Some(sfreq), ..Default::default() };
        let data = Data::from_array_bytes(DType::F32, vec![n], buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &goofi_core::Param) -> NodeResult {
        // Only `sfreq` is stateful: mirror it and re-anchor pacing so the new rate takes
        // effect from the next tick. frequency/amplitude/waveform are read live in process.
        if key.group == "oscillator" && key.name == "sfreq" {
            if let Some(x) = v.as_f64() {
                self.sfreq = x.max(1.0);
                self.start = None;
                self.emitted = 0;
            }
        }
        Ok(())
    }
}

/// First element of an f32 array `Data`, if the slot holds one (a scalar control).
fn first_f32(d: Option<&Data>) -> Option<f32> {
    match d?.value() {
        Value::Array(s) if s.dtype() == DType::F32 => s
            .as_bytes()
            .get(0..4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        _ => None,
    }
}

static PARAMS: &[ParamDecl] = &[
    // LFO / biosignal regime — a slow frequency slider, biosignal-typical sample rate.
    ParamDecl { group: "oscillator", name: "frequency", spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 100.0 } },
    ParamDecl { group: "oscillator", name: "amplitude", spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 1.0e6 } },
    ParamDecl { group: "oscillator", name: "sfreq", spec: ParamSpec::Float { default: 250.0, min: 1.0, max: 10_000.0 } },
    ParamDecl {
        group: "oscillator",
        name: "waveform",
        spec: ParamSpec::Str { default: "sine", options: &["sine", "square", "sawtooth", "triangle"], refresh: false },
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];
// A single non-triggering control input: the oscillator free-runs (real-time paced)
// and reads the latest `frequency` each tick rather than being woken by it.
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "frequency",
    kind: SlotType::Array,
    trigger_process: false,
    multi: false,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Oscillator",
        category: "inputs",
        doc: "LFO/biosignal oscillator (sine/square/sawtooth/triangle), frequency slider, meta sfreq.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        params: PARAMS,
        isolation: Isolation::InProcess,
        factory: default_factory::<Oscillator>,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{DType, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamGroups, ParamKey, Params};
    use indexmap::IndexMap;

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
        let mut ctx = NodeCtx { tick: 0, now };
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
        node.process(&inp, &mut Outputs::new(&mut o0), &mut NodeCtx { tick: 0, now: 0.0 }, &Params::new(&params)).unwrap();
        let mut o1 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o1), &mut NodeCtx { tick: 1, now: 0.004 }, &Params::new(&params)).unwrap();
        let d = o1.get("out").unwrap().as_ref().unwrap();
        assert_eq!(d.meta().sfreq, Some(1000.0));
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
        node.process(&inp, &mut Outputs::new(&mut o0), &mut NodeCtx { tick: 0, now: 0.0 }, &Params::new(&params)).unwrap();
        let mut o1 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o1), &mut NodeCtx { tick: 1, now: 0.004 }, &Params::new(&params)).unwrap();
        if let Value::Array(s) = o1.get("out").unwrap().as_ref().unwrap().value() {
            let v: Vec<f32> = s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            assert!(v.iter().all(|x| x.abs() == 1.0), "square is ±1, got {v:?}");
            assert_eq!(v, vec![1.0, 1.0, -1.0, -1.0]);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn frequency_control_input_overrides_the_slider() {
        use goofi_core::{Data, Meta};
        // Slider frequency 10 Hz; a wired control input of 250 Hz (= sfreq/4) must win.
        let (mut node, m, params) = build(10.0, 1000.0, 1.0, "sine");
        let freq = Data::from_array_bytes(DType::F32, vec![1], 250.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("frequency", Some(freq));
        let inp = Inputs::new(&inmap);

        let mut o0 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o0), &mut NodeCtx { tick: 0, now: 0.0 }, &Params::new(&params)).unwrap();
        let mut o1 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o1), &mut NodeCtx { tick: 1, now: 0.004 }, &Params::new(&params)).unwrap();
        if let Value::Array(s) = o1.get("out").unwrap().as_ref().unwrap().value() {
            assert_eq!(s.shape(), &[4]);
            let v = |i: usize| f32::from_le_bytes(s.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap());
            assert!(v(0).abs() < 1e-5 && (v(1) - 1.0).abs() < 1e-5 && (v(3) + 1.0).abs() < 1e-5,
                "control 250 Hz -> quarter-period [0,1,0,-1], got [{},{},{},{}]", v(0), v(1), v(2), v(3));
        } else {
            panic!("expected array");
        }
    }
}
