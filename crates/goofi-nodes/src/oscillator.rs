//! Oscillator — a general low-frequency / biosignal-regime oscillator. Each tick it
//! emits the block of samples real time has advanced by (drift-free against
//! `meta["sfreq"]`), carrying phase forward so blocks join click-free. A `frequency`
//! slider (LFO range) and a `waveform` selector (sine/square/sawtooth/triangle) shape
//! the signal; a wired `frequency` control input overrides the slider each tick.
//! Self-contained (no audio-synth infrastructure) — one of the two seed nodes for the
//! redesigned library.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Meta, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;
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

struct Oscillator {
    frequency: f64,
    amplitude: f64,
    sfreq: f64,
    waveform: Waveform,
    /// Phase in radians, carried across ticks for click-free blocks.
    phase: f64,
    /// Wall-clock anchor (`ctx.now` at the first emit) — pacing is measured from here.
    start: Option<f64>,
    /// Total samples emitted since `start`; the running count keeps the per-tick block
    /// size drift-free (`n = round(sfreq·elapsed) − emitted`) rather than rounding each
    /// tick independently.
    emitted: u64,
}

impl Oscillator {
    fn new(frequency: f64, amplitude: f64, sfreq: f64, waveform: Waveform) -> Oscillator {
        Oscillator { frequency, amplitude, sfreq, waveform, phase: 0.0, start: None, emitted: 0 }
    }
}

impl Node for Oscillator {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx) -> NodeResult {
        // Drift-free sample count: how many samples real time has advanced by since the
        // anchor, minus what we've already emitted. Zero on the first tick (now == anchor).
        let start = *self.start.get_or_insert(c.now);
        let total = (self.sfreq * (c.now - start)).round().max(0.0) as u64;
        let n = total.saturating_sub(self.emitted) as usize;
        if n == 0 {
            return Ok(());
        }
        self.emitted = total;

        // A wired `frequency` control input drives frequency in real time (its first
        // value); unwired, the `frequency` slider holds.
        let freq = first_f32(inp.get("frequency")).map(|v| v as f64).unwrap_or(self.frequency);
        let step = TAU * freq / self.sfreq; // phase increment per sample
        let amp = self.amplitude;
        let wave = self.waveform;
        let mut buf = Vec::with_capacity(n * 4);
        for _ in 0..n {
            buf.extend_from_slice(&((wave.sample(self.phase) * amp) as f32).to_le_bytes());
            self.phase += step;
        }
        self.phase = self.phase.rem_euclid(TAU); // keep bounded over long runs

        let meta = Meta { sfreq: Some(self.sfreq), ..Default::default() };
        let data = Data::from_array_bytes(DType::F32, vec![n], buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("oscillator", "frequency") => {
                if let Some(x) = v.as_f64() {
                    self.frequency = x;
                }
            }
            ("oscillator", "amplitude") => {
                if let Some(x) = v.as_f64() {
                    self.amplitude = x;
                }
            }
            ("oscillator", "sfreq") => {
                if let Some(x) = v.as_f64() {
                    // Re-anchor pacing so the new rate takes effect from the next tick.
                    self.sfreq = x.max(1.0);
                    self.start = None;
                    self.emitted = 0;
                }
            }
            ("oscillator", "waveform") => {
                if let Some(s) = v.as_str() {
                    self.waveform = Waveform::parse(s); // phase-preserving
                }
            }
            _ => {}
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

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    // LFO / biosignal regime — a slow frequency slider, biosignal-typical sample rate.
    g.insert("frequency".to_string(), Param::float(1.0, 0.0, 100.0));
    g.insert("amplitude".to_string(), Param::float(1.0, 0.0, 1.0e6));
    g.insert("sfreq".to_string(), Param::float(250.0, 1.0, 10_000.0));
    g.insert(
        "waveform".to_string(),
        Param::Str {
            value: "sine".to_string(),
            options: Some(
                ["sine", "square", "sawtooth", "triangle"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            refresh: None,
        },
    );
    let mut groups = ParamGroups::new();
    groups.insert("oscillator".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let f = |name, dflt| param(p, "oscillator", name).and_then(Param::as_f64).unwrap_or(dflt);
    let waveform = param(p, "oscillator", "waveform")
        .and_then(Param::as_str)
        .map(Waveform::parse)
        .unwrap_or(Waveform::Sine);
    Box::new(Oscillator::new(
        f("frequency", 1.0),
        f("amplitude", 1.0),
        f("sfreq", 250.0).max(1.0),
        waveform,
    ))
}

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
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{DType, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs};
    use indexmap::IndexMap;

    fn run_at(
        node: &mut Box<dyn goofi_node::Node>,
        m: &goofi_node::NodeManifest,
        now: f64,
    ) -> Option<Vec<f32>> {
        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx { tick: 0, now };
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
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

    /// Build an Oscillator with explicit frequency/sfreq (and optional waveform).
    fn osc(freq: f64, sfreq: f64, waveform: Option<&str>) -> (Box<dyn goofi_node::Node>, &'static goofi_node::NodeManifest) {
        let m = goofi_node::find("Oscillator").expect("Oscillator registered");
        let mut params = (m.default_params)();
        params["oscillator"].insert("frequency".into(), goofi_core::Param::float(freq, 0.0, 1e6));
        params["oscillator"].insert("sfreq".into(), goofi_core::Param::float(sfreq, 1.0, 1e6));
        if let Some(w) = waveform {
            params["oscillator"].insert(
                "waveform".into(),
                goofi_core::Param::Str { value: w.to_string(), options: None, refresh: None },
            );
        }
        ((m.make)(&params), m)
    }

    #[test]
    fn first_tick_emits_nothing_then_paces_by_wall_clock() {
        let (mut node, m) = osc(1.0, 1000.0, None);
        // First tick anchors at now=0 -> zero elapsed -> no frame.
        assert!(run_at(&mut node, m, 0.0).is_none(), "no time elapsed yet");
        // 10 ms later at 1 kHz -> exactly 10 samples.
        assert_eq!(run_at(&mut node, m, 0.010).expect("a paced block").len(), 10);
        assert_eq!(run_at(&mut node, m, 0.020).unwrap().len(), 10);
    }

    #[test]
    fn drift_free_across_uneven_ticks() {
        // Cumulative samples equal round(sfreq·elapsed), never lost to per-tick rounding.
        let (mut node, m) = osc(1.0, 1000.0, None);
        run_at(&mut node, m, 0.0); // anchor
        let a = run_at(&mut node, m, 0.0015).map_or(0, |v| v.len()); // round(1.5)=2
        let b = run_at(&mut node, m, 0.0035).map_or(0, |v| v.len()); // round(3.5)-2 = 2
        assert_eq!(a + b, 4, "cumulative tracks round(sfreq·elapsed): round(3.5)=4");
    }

    #[test]
    fn emits_sine_values_and_sfreq_meta() {
        // freq = sfreq/4 -> phase step π/2 -> samples 0, 1, 0, -1 (× amplitude).
        let m = goofi_node::find("Oscillator").unwrap();
        let mut params = (m.default_params)();
        params["oscillator"].insert("frequency".into(), goofi_core::Param::float(250.0, 0.0, 1e6));
        params["oscillator"].insert("sfreq".into(), goofi_core::Param::float(1000.0, 1.0, 1e6));
        params["oscillator"].insert("amplitude".into(), goofi_core::Param::float(2.0, 0.0, 1e6));
        let mut node = (m.make)(&params);

        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut o0 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o0), &mut NodeCtx { tick: 0, now: 0.0 }).unwrap();
        let mut o1 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o1), &mut NodeCtx { tick: 1, now: 0.004 }).unwrap();
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
        let (mut node, m) = osc(250.0, 1000.0, Some("square"));
        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut o0 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o0), &mut NodeCtx { tick: 0, now: 0.0 }).unwrap();
        let mut o1 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o1), &mut NodeCtx { tick: 1, now: 0.004 }).unwrap();
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
        let (mut node, m) = osc(10.0, 1000.0, None);
        let freq = Data::from_array_bytes(DType::F32, vec![1], 250.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("frequency", Some(freq));
        let inp = Inputs::new(&inmap);

        let mut o0 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o0), &mut NodeCtx { tick: 0, now: 0.0 }).unwrap();
        let mut o1 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut o1), &mut NodeCtx { tick: 1, now: 0.004 }).unwrap();
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
