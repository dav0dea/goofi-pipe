use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{hz_of, AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, Tag, BLOCK, MAX_CHANNELS};

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
        spec: ParamSpec::Float { default: 2.0, min: -5.0, max: 6.0 },
        expression: None,
        doc: Some("volts per octave, 0 at C4 — the same units as `Osc.pitch`, so a reference tracks"),
    },
    Q = ParamDecl {
        group: "filter",
        name: "q",
        spec: ParamSpec::Float { default: 0.7, min: 0.5, max: 20.0 },
        expression: None,
        doc: Some("resonance: it rings longer as it climbs, and peaks at `q` times full scale"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "A resonant state-variable filter, low, band or high, at audio-rate cutoff.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

#[derive(Default)]
struct Svf {
    rate: f32,
    ic1: [f32; MAX_CHANNELS as usize],
    ic2: [f32; MAX_CHANNELS as usize],
}

impl AudioNode for Svf {
    fn prepare(&mut self, rate: f64) {
        self.rate = rate as f32;
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let (input, cutoff, q) = (&b.ins[0], &b.params[P::CUTOFF], &b.params[P::Q]);
        let mode = b.params[P::MODE].chan(0)[0] as u8;
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            let (x, volts, res) = (input.chan(c), cutoff.chan(c), q.chan(c));
            let (ic1, ic2) = (&mut self.ic1[c], &mut self.ic2[c]);
            let y = out.chan_mut(c);
            for i in 0..BLOCK {
                // Nyquist is the ceiling `tan` needs: at it the warp is infinite.
                let f = hz_of(volts[i]).clamp(1.0, 0.45 * self.rate);
                let g = (std::f32::consts::PI * f / self.rate).tan();
                let k = 1.0 / res[i].max(0.5);
                let a1 = 1.0 / (1.0 + g * (g + k));
                let (a2, a3) = (g * a1, g * g * a1);
                let v3 = x[i] - *ic2;
                let v1 = a1 * *ic1 + a2 * v3;
                let v2 = *ic2 + a2 * *ic1 + a3 * v3;
                *ic1 = 2.0 * v1 - *ic1;
                *ic2 = 2.0 * v2 - *ic2;
                y[i] = match mode {
                    1 => v1,
                    2 => x[i] - k * v1 - v2,
                    _ => v2,
                };
            }
        }
    }
}

goofi_audio_sdk::export!(Svf, MANIFEST);
