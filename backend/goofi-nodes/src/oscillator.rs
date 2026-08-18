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
        // Declaring `common.max_frequency` here means the universal declaration does not apply to
        // this node at all — a manifest's own common param is never overwritten, and is under no
        // obligation to match it. Stated in full for that reason: the Oscillator IS paced by the
        // patch rate, so the expression is `On`; 30.0 is the literal that stands in when no
        // evaluator is wired, which is also why this one is capped far above the universal 100.
        // `trigger` matches the universal declaration and is equally inert — spec §1.1 ignores it
        // on `common.*`. A `default_ufreq` edit re-rates this node by re-evaluating the binding,
        // not by triggering a run.
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
