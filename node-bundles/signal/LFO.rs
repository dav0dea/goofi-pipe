//! LFO — one oscillator for both planes: a sample per update to modulate a param, or the block of
//! samples real time advanced by, carrying phase across frames so blocks join without a click.

use goofi_core::{Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, Tag};

/// The waveform's value at `t` cycles in `[0, 1)`, in `[-1, 1]`.
fn wave(kind: &str, t: f64, duty: f64) -> f64 {
    match kind {
        "square" => {
            if t < duty {
                1.0
            } else {
                -1.0
            }
        }
        "sawtooth" => 2.0 * t - 1.0,
        "triangle" => {
            if t < 0.25 {
                4.0 * t
            } else if t < 0.75 {
                2.0 - 4.0 * t
            } else {
                4.0 * t - 4.0
            }
        }
        _ => (std::f64::consts::TAU * t).sin(),
    }
}

#[derive(Default)]
struct Lfo {
    /// Cycles accumulated since the last reset; bounded to `[0, 1)`.
    phase: f64,
    /// `ctx.now` at the first emit — block pacing is measured from here.
    start: Option<f64>,
    /// `ctx.now` at the last emit, which paces the value mode.
    last: Option<f64>,
    /// Samples emitted since `start`; counting keeps the block size drift-free.
    emitted: u64,
}

impl Node for Lfo {
    fn process(
        &mut self,
        _inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let freq = p.f64("lfo", "frequency").unwrap_or(1.0);
        let amp = p.f64("lfo", "amplitude").unwrap_or(1.0);
        let offset = p.f64("lfo", "offset").unwrap_or(0.0);
        // Refuse non-finite BEFORE it reaches `self.phase`: nothing but a reset clears a NaN.
        if !freq.is_finite() || !amp.is_finite() || !offset.is_finite() {
            return Err(format!("non-finite drive: frequency={freq}, amplitude={amp}, offset={offset}").into());
        }
        let kind = p.str("lfo", "waveform").unwrap_or("sine");
        let duty = p.f64("lfo", "duty").unwrap_or(0.5).clamp(0.0, 1.0);
        let skew = p.f64("lfo", "phase").unwrap_or(0.0);
        let sfreq = p.f64("output", "sfreq").unwrap_or(250.0).max(1.0);
        let block = p.str("output", "mode").unwrap_or("value") == "block";

        let mut sample = |phase: f64| ((wave(kind, (phase + skew).rem_euclid(1.0), duty) * amp + offset) as f32).to_le_bytes();

        let (shape, buf, meta) = if block {
            let start = *self.start.get_or_insert(c.now);
            let total = (sfreq * (c.now - start)).round().max(0.0) as u64;
            let n = total.saturating_sub(self.emitted) as usize;
            if n == 0 {
                return Ok(());
            }
            self.emitted = total;
            let step = freq / sfreq;
            let mut buf = Vec::with_capacity(n * 4);
            for _ in 0..n {
                buf.extend_from_slice(&sample(self.phase));
                self.phase = (self.phase + step).rem_euclid(1.0);
            }
            (vec![n], buf, Meta::new().with_sfreq(Some(sfreq)))
        } else {
            let elapsed = c.now - self.last.unwrap_or(c.now);
            self.phase = (self.phase + freq * elapsed).rem_euclid(1.0);
            (vec![1], sample(self.phase).to_vec(), Meta::new())
        };
        self.last = Some(c.now);
        out.set("out", Data::array_f32(shape, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, _v: &goofi_core::Param) -> NodeResult {
        // A new rate or mode re-anchors the block pacing; the old count measures a gone clock.
        if key.group == "output" {
            self.start = None;
            self.emitted = 0;
        }
        Ok(())
    }

    fn on_pulse(&mut self, _key: &ParamKey, _p: &Params<'_>) -> NodeResult {
        self.phase = 0.0;
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "lfo",
        name: "waveform",
        spec: ParamSpec::Str { default: "sine", options: &["sine", "triangle", "sawtooth", "square"], refresh: false },
        expression: None,
        doc: Some("Shape of one cycle."),
    },
    ParamDecl {
        group: "lfo",
        name: "frequency",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 1000.0 },
        expression: None,
        doc: Some("Cycles per second."),
    },
    ParamDecl {
        group: "lfo",
        name: "amplitude",
        spec: ParamSpec::Float { default: 1.0, min: -1.0e6, max: 1.0e6 },
        expression: None,
        doc: Some("Peak value: the wave swings between minus this and plus this, before `offset`."),
    },
    ParamDecl {
        group: "lfo",
        name: "offset",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e6, max: 1.0e6 },
        expression: None,
        doc: Some("Added to every sample, so the wave can swing around a value other than zero."),
    },
    ParamDecl {
        group: "lfo",
        name: "phase",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("Where in the cycle the wave reads, in cycles; 0.25 is a quarter turn ahead."),
    },
    ParamDecl {
        group: "lfo",
        name: "duty",
        spec: ParamSpec::Float { default: 0.5, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("Fraction of the cycle a square wave spends high; the other waveforms ignore it."),
    },
    ParamDecl {
        group: "lfo",
        name: "reset",
        spec: ParamSpec::Pulse,
        expression: None,
        doc: Some("Put the phase back to the start of the cycle."),
    },
    ParamDecl {
        group: "output",
        name: "mode",
        spec: ParamSpec::Str { default: "value", options: &["value", "block"], refresh: false },
        expression: None,
        doc: Some(
            "`value` emits one sample per update, which a param reference reads as a number; \
             `block` emits the samples elapsed at `sfreq`, which is a signal.",
        ),
    },
    ParamDecl {
        group: "output",
        name: "sfreq",
        spec: ParamSpec::Float { default: 250.0, min: 1.0, max: 10_000.0 },
        expression: None,
        doc: Some("Sample rate within an emitted block, in Hz. `value` mode ignores it."),
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Generator],
    doc: "A low-frequency oscillator: one sample per update to modulate a param, or a block of samples to feed a signal.",
    inputs: &[],
    outputs: OUTPUTS,
    params: PARAMS,
    producer: true,
};

goofi_signal_sdk::export!(Lfo, MANIFEST);
