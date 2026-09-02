use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{hz_of, AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, BLOCK, MAX_CHANNELS};

goofi_audio_sdk::params! {
    MODE = ParamDecl {
        group: "filter",
        name: "mode",
        spec: ParamSpec::Str { default: "low", options: &["low", "band", "high"], refresh: false },
        expression: None,
        doc: None,
    },
    CUTOFF = ParamDecl {
        group: "filter",
        name: "cutoff",
        spec: ParamSpec::Float { default: 2.0, min: -5.0, max: 7.0 },
        expression: None,
        doc: Some("volts per octave, 0 at C4 — the same units as `Osc.pitch`, so a reference tracks"),
    },
    Q = ParamDecl {
        group: "filter",
        name: "q",
        spec: ParamSpec::Float { default: 0.7, min: 0.5, max: 20.0 },
        expression: None,
        doc: Some("resonance; it self-oscillates as it climbs"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "A resonant state-variable filter, low, band or high, at audio-rate cutoff.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

/// One channel's two integrator states.
#[derive(Clone, Copy, Default)]
struct State {
    ic1: f32,
    ic2: f32,
}

#[derive(Default)]
struct Svf {
    rate: f32,
    state: [State; MAX_CHANNELS as usize],
}

impl AudioNode for Svf {
    fn prepare(&mut self, rate: f64) {
        self.rate = rate as f32;
        self.state = Default::default();
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let (input, cutoff, q) = (&b.ins[0], &b.params[P::CUTOFF], &b.params[P::Q]);
        let mode = b.params[P::MODE].chan(0)[0];
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            let (x, hz, res) = (input.chan(c), cutoff.chan(c), q.chan(c));
            let state = &mut self.state[c];
            let y = out.chan_mut(c);
            for i in 0..BLOCK {
                // Nyquist is the ceiling `tan` needs: at it the warp is infinite.
                let f = hz_of(hz[i]).clamp(1.0, 0.45 * self.rate);
                let g = (std::f32::consts::PI * f / self.rate).tan();
                let k = 1.0 / res[i].max(0.5);
                let a1 = 1.0 / (1.0 + g * (g + k));
                let (a2, a3) = (g * a1, g * g * a1);
                let v3 = x[i] - state.ic2;
                let v1 = a1 * state.ic1 + a2 * v3;
                let v2 = state.ic2 + a2 * state.ic1 + a3 * v3;
                state.ic1 = 2.0 * v1 - state.ic1;
                state.ic2 = 2.0 * v2 - state.ic2;
                y[i] = match mode as usize {
                    1 => v1,
                    2 => x[i] - k * v1 - v2,
                    _ => v2,
                };
            }
        }
    }
}

goofi_audio_sdk::export!(Svf, MANIFEST);
