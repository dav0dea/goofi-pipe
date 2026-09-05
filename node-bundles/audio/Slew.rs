use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, Tag, BLOCK, MAX_CHANNELS};

goofi_audio_sdk::params! {
    RISE = ParamDecl {
        group: "slew",
        name: "rise",
        spec: ParamSpec::Float { default: 0.01, min: 0.0, max: 10.0 },
        expression: None,
        doc: Some("seconds to climb one unit"),
    },
    FALL = ParamDecl {
        group: "slew",
        name: "fall",
        spec: ParamSpec::Float { default: 0.01, min: 0.0, max: 10.0 },
        expression: None,
        doc: Some("seconds to fall one unit"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "Its input with a rate limit: a step becomes a ramp, a gate an envelope, a knob a glide.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

#[derive(Default)]
struct Slew {
    rate: f32,
    held: [f32; MAX_CHANNELS as usize],
}

impl AudioNode for Slew {
    fn prepare(&mut self, rate: f64) {
        self.rate = rate as f32;
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let (input, rise, fall) = (&b.ins[0], &b.params[P::RISE], &b.params[P::FALL]);
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            let (x, up, down) = (input.chan(c), rise.chan(c), fall.chan(c));
            let held = &mut self.held[c];
            let y = out.chan_mut(c);
            for i in 0..BLOCK {
                let seconds = if x[i] > *held { up[i] } else { down[i] };
                // A zero time is a wire: the limit of an infinite rate, not a division by zero.
                let step = match seconds > 0.0 {
                    true => 1.0 / (seconds * self.rate),
                    false => f32::INFINITY,
                };
                *held += (x[i] - *held).clamp(-step, step);
                y[i] = *held;
            }
        }
    }
}

goofi_audio_sdk::export!(Slew, MANIFEST);
