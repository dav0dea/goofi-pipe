use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, BLOCK, MAX_CHANNELS};

goofi_audio_sdk::params! {
    FREQ = ParamDecl {
        group: "osc",
        name: "freq",
        spec: ParamSpec::Float { default: 440.0, min: 0.0, max: 20000.0 },
        expression: None,
        doc: Some("Hz; an audio reference here is one voice per channel"),
    },
    SHAPE = ParamDecl {
        group: "osc",
        name: "shape",
        spec: ParamSpec::Str { default: "sine", options: &["sine", "saw", "square", "tri"], refresh: false },
        expression: None,
        doc: None,
    },
}

static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

pub static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "An oscillator in [-1, 1], one channel per channel of `freq`.",
    inputs: &[],
    outputs: OUTS,
    params: PARAMS,
};

#[derive(Default)]
pub struct Osc {
    phase: [f32; MAX_CHANNELS as usize],
    step: f32,
}

impl AudioNode for Osc {
    fn prepare(&mut self, rate: f64) {
        self.step = 1.0 / rate as f32;
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let freq = &b.params[P::FREQ];
        let shape = b.params[P::SHAPE].chan(0)[0] as u8;
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            let f = freq.chan(c.min(freq.channels() as usize - 1));
            let phase = &mut self.phase[c];
            let samples = out.chan_mut(c);
            for i in 0..BLOCK {
                samples[i] = match shape {
                    1 => 2.0 * *phase - 1.0,
                    2 => if *phase < 0.5 { 1.0 } else { -1.0 },
                    3 => 1.0 - 4.0 * (*phase - 0.5).abs(),
                    _ => (std::f32::consts::TAU * *phase).sin(),
                };
                *phase = (*phase + f[i] * self.step).fract();
            }
        }
    }
}
