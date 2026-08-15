//! Oscillator — a general low-frequency / biosignal-regime oscillator. Each tick it
//! emits the block of samples real time has advanced by (drift-free against
//! `meta["sfreq"]`), carrying phase forward so blocks join click-free. A `frequency`
//! slider (LFO range) and a `waveform` selector (sine/square/sawtooth/triangle) shape
//! the signal. It is a pure producer with no input slots: drive `frequency` live by
//! binding a param expression (e.g. `frequency = nd('lfo').out[0]`) rather than wiring
//! a control cable. Self-contained (no audio-synth infrastructure) — one of the two
//! seed nodes for the redesigned library.
//!
//! `frequency`, `amplitude`, and `waveform` are cold params (read live from `p` each
//! tick). `sfreq` is the one stateful param: changing it must re-anchor the drift-free
//! pacing, so it is mirrored to a field via `on_param_changed`.

use goofi_core::SlotType;
use goofi_core::{Data, Meta};
use goofi_node::{
    default_factory, ExprDecl, ExprMode, Inputs, Isolation, Node, NodeCtx, NodeManifest,
    NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, ParamSpec, Params,
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
    fn process(&mut self, _inp: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
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

        // Frequency comes from the `frequency` param (a cold read). Drive it live by
        // binding a param expression (e.g. frequency = nd('lfo').out[0]).
        let freq = p.f64("oscillator", "frequency").unwrap_or(1.0);
        let amp = p.f64("oscillator", "amplitude").unwrap_or(1.0);
        let wave = Waveform::parse(p.str("oscillator", "waveform").unwrap_or("sine"));
        // A bound expression can yield inf/NaN (the Float coercion does not reject it). Refuse
        // it BEFORE it reaches `self.phase`: `rem_euclid` maps ±inf and NaN alike to NaN, and
        // nothing ever resets phase, so one bad tick would emit NaN for the node's lifetime.
        if !freq.is_finite() || !amp.is_finite() {
            return Err(format!("non-finite drive: frequency={freq}, amplitude={amp}").into());
        }
        let step = TAU * freq / sfreq; // phase increment per sample
        let mut buf = Vec::with_capacity(n * 4);
        for _ in 0..n {
            buf.extend_from_slice(&((wave.sample(self.phase) * amp) as f32).to_le_bytes());
            self.phase += step;
        }
        self.phase = self.phase.rem_euclid(TAU); // keep bounded over long runs

        let meta = Meta::new().with_sfreq(Some(sfreq));
        let data = Data::array_f32(vec![n], buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &goofi_core::Param) -> NodeResult {
        // Only `sfreq` is stateful: mirror it and re-anchor pacing so the new rate takes
        // effect from the next tick. frequency/amplitude/waveform are read live in process.
        if key.group == "oscillator" && key.name == "sfreq" {
            if let Some(x) = v.as_f64() {
                let new = x.max(1.0);
                // Re-anchor ONLY on an actual change: a redundant same-value write must
                // not reset pacing (which would drop the elapsed interval / stall a tick).
                if new != self.sfreq {
                    self.sfreq = new;
                    self.start = None;
                    self.emitted = 0;
                }
            }
        }
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    // LFO / biosignal regime — a slow frequency slider, biosignal-typical sample rate.
    ParamDecl {
        group: "oscillator",
        name: "frequency",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 100.0 },
        expression: None,
        doc: Some("Oscillation frequency in Hz, within the signal band rather than the audio band."),
    },
    ParamDecl {
        group: "oscillator",
        name: "amplitude",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 1.0e6 },
        expression: None,
        doc: Some("Peak value of the waveform; it swings between -amplitude and +amplitude."),
    },
    ParamDecl {
        group: "oscillator",
        name: "sfreq",
        spec: ParamSpec::Float { default: 250.0, min: 1.0, max: 10_000.0 },
        expression: None,
        doc: Some(
            "Sample rate WITHIN each emitted frame, in Hz. Together with this node's update rate \
             (common.max_frequency) it decides how many samples each frame carries.",
        ),
    },
    ParamDecl {
        group: "oscillator",
        name: "waveform",
        spec: ParamSpec::Str { default: "sine", options: &["sine", "square", "sawtooth", "triangle"], refresh: false },
        expression: None,
        doc: Some("Shape of one cycle."),
    },
    // The producer contract: free-running, paced by the patch's `default_ufreq` global (30 Hz by
    // default). The 30.0 literal is the no-evaluator fallback; the expression makes a live
    // `globals.default_ufreq` edit re-rate every Oscillator. `common.autotrigger` is NOT declared
    // here — `producer: true` on the manifest is the one place a source says so.
    ParamDecl {
        group: "common",
        name: "max_frequency",
        spec: ParamSpec::Float { default: 30.0, min: 0.0, max: 1000.0 },
        expression: Some(ExprDecl {
            source: "globals.default_ufreq",
            mode: ExprMode::On,
            trigger: false,
        }),
        doc: Some(
            "How many frames per second to emit. Bound to the patch's `default_ufreq` global by \
             default, so editing that global re-rates every Oscillator at once.",
        ),
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Oscillator",
        category: "inputs",
        doc: "LFO/biosignal oscillator (sine/square/sawtooth/triangle), frequency slider, meta sfreq.",
        inputs: &[],
        outputs: OUTPUTS,
        params: PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: default_factory::<Oscillator>,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Param, Value};
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
        let policy = RunPolicy::from_params(&with_common(m.default_params(), m.producer));
        assert_eq!(policy.max_frequency, 30.0, "fallback rate is the producer default");
        assert!(policy.autotrigger, "the oscillator is a free-running producer");
    }
}
