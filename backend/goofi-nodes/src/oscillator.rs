//! Oscillator — an LFO/biosignal producer. Each tick emits the block of samples real time has
//! advanced by, carrying phase forward so blocks join click-free.

use goofi_core::SlotType;
use goofi_core::{Data, Meta};
use goofi_node::{
    default_factory, ExprDecl, ExprMode, Inputs,  Node, NodeClass, NodeCtx, NodeManifest,
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
    /// The waveform's value at `phase` radians, in [-1, 1].
    fn sample(self, phase: f64) -> f64 {
        match self {
            Waveform::Sine => phase.sin(),
            // Phase-based, not sign-of-sine, which flips on the π rounding boundary.
            Waveform::Square => {
                if phase.rem_euclid(TAU) < PI {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => 2.0 * (phase.rem_euclid(TAU) / TAU) - 1.0,
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
    sfreq: f64,
    phase: f64,
    /// Wall-clock anchor (`ctx.now` at the first emit) — pacing is measured from here.
    start: Option<f64>,
    /// Total samples emitted since `start`; counting keeps the block size drift-free.
    emitted: u64,
}

impl Node for Oscillator {
    fn process(&mut self, _inp: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        // `sfreq` is seeded (>= 1) via on_param_changed; an unseeded node emits nothing.
        let sfreq = self.sfreq;
        let start = *self.start.get_or_insert(c.now);
        let total = (sfreq * (c.now - start)).round().max(0.0) as u64;
        let n = total.saturating_sub(self.emitted) as usize;
        if n == 0 {
            return Ok(());
        }
        self.emitted = total;

        let freq = p.f64("oscillator", "frequency").unwrap_or(1.0);
        let amp = p.f64("oscillator", "amplitude").unwrap_or(1.0);
        let wave = Waveform::parse(p.str("oscillator", "waveform").unwrap_or("sine"));
        // Refuse non-finite BEFORE it reaches `self.phase`: nothing resets phase, so NaN is forever.
        if !freq.is_finite() || !amp.is_finite() {
            return Err(format!("non-finite drive: frequency={freq}, amplitude={amp}").into());
        }
        let step = TAU * freq / sfreq;
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
        if key.group == "oscillator" && key.name == "sfreq" {
            if let Some(x) = v.as_f64() {
                let new = x.max(1.0);
                // Re-anchor ONLY on an actual change: a same-value write would stall a tick.
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
    // A manifest's own `common.*` is never overwritten by the universal declaration, so this one
    // is stated in full; 30.0 stands in when no expression evaluator is wired.
    ParamDecl {
        group: "common",
        name: "max_frequency",
        spec: ParamSpec::Float { default: 30.0, min: 0.0, max: 1000.0 },
        expression: Some(ExprDecl {
            source: "globals.default_ufreq",
            mode: ExprMode::On,
            trigger: true,
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
    NodeClass {
        manifest: NodeManifest {
            type_name: "Oscillator",
            category: "inputs",
            doc: "LFO/biosignal oscillator (sine/square/sawtooth/triangle), frequency slider, meta sfreq.",
            inputs: &[],
            outputs: OUTPUTS,
            params: PARAMS,
            producer: true,
        },
        isolation: &goofi_node::NATIVE,
        factory: default_factory::<Oscillator>,
    }
}
