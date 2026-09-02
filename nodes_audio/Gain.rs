use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, BLOCK};

goofi_audio_sdk::params! {
    GAIN = ParamDecl {
        group: "gain",
        name: "gain",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 10.0 },
        expression: None,
        doc: None,
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

pub static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "Its input times `gain`; the wires into `input` sum.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

#[derive(Default)]
pub struct Gain;

impl AudioNode for Gain {
    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let input = &b.ins[0];
        let gain = &b.params[P::GAIN];
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            let x = input.chan(c);
            let g = gain.chan(c);
            let y = out.chan_mut(c);
            for i in 0..BLOCK {
                y[i] = x[i] * g[i];
            }
        }
    }
}

goofi_audio_sdk::export!(Gain, MANIFEST);
