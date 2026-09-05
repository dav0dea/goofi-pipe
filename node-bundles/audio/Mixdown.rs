use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, Tag, BLOCK, MAX_CHANNELS};

goofi_audio_sdk::params! {
    CHANNELS = ParamDecl {
        group: "mixdown",
        name: "channels",
        spec: ParamSpec::Int { default: 2, min: 1, max: 16 },
        expression: None,
        doc: Some("how many channels leave; 2 is the pair a speaker takes"),
    },
    SPREAD = ParamDecl {
        group: "mixdown",
        name: "spread",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("how far apart the voices sit; at 0 every one of them is in the middle"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "Many channels become few, each one placed across the field it lands in.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

#[derive(Default)]
struct Mixdown;

/// Where voice `v` of `n` sits, in 0 to 1 across the field; one voice sits in the middle.
fn position(v: usize, n: usize, spread: f32) -> f32 {
    let at = if n < 2 { 0.5 } else { v as f32 / (n - 1) as f32 };
    0.5 + (at - 0.5) * spread.clamp(0.0, 1.0)
}

impl AudioNode for Mixdown {
    fn channels(&self, _ins: &[u16], params: &[f64], outs: usize) -> Vec<u16> {
        vec![(params[P::CHANNELS] as u16).clamp(1, MAX_CHANNELS); outs]
    }

    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let input = &b.ins[0];
        let spread = b.params[P::SPREAD].chan(0)[0];
        let voices = (input.channels() as usize).min(MAX_CHANNELS as usize);
        let out = &mut b.outs[0];
        let wide = out.channels() as usize;

        for c in 0..wide {
            out.chan_mut(c).fill(0.0);
        }
        for v in 0..voices {
            let at = position(v, voices, spread) * (wide - 1) as f32;
            let (left, frac) = (at.floor() as usize, at.fract());
            // Constant power, so a voice crossing between two channels does not dip on the way.
            let (a, t) = ((1.0 - frac) * std::f32::consts::FRAC_PI_2, frac * std::f32::consts::FRAC_PI_2);
            for (c, gain) in [(left, a.sin()), ((left + 1).min(wide - 1), t.sin())] {
                let x = *input.chan(v);
                let y = out.chan_mut(c);
                for i in 0..BLOCK {
                    y[i] += x[i] * gain;
                }
            }
        }
    }
}

goofi_audio_sdk::export!(Mixdown, MANIFEST);
