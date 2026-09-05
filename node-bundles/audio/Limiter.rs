use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, Tag, BLOCK, MAX_CHANNELS};

goofi_audio_sdk::params! {
    CEILING = ParamDecl {
        group: "limiter",
        name: "ceiling",
        spec: ParamSpec::Float { default: 0.9, min: 0.05, max: 1.0 },
        expression: None,
        doc: Some("the level nothing leaving this node may pass"),
    },
    ATTACK = ParamDecl {
        group: "limiter",
        name: "attack",
        spec: ParamSpec::Float { default: 0.004, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("seconds to take hold of a peak; shorter is tighter and harder"),
    },
    RELEASE = ParamDecl {
        group: "limiter",
        name: "release",
        spec: ParamSpec::Float { default: 0.4, min: 0.0, max: 10.0 },
        expression: None,
        doc: Some("seconds to let go once the peak has passed"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "One gain across every channel, so nothing leaves above the ceiling and the picture holds.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

#[derive(Default)]
struct Limiter {
    rate: f32,
    /// The follower's level, and the gain it asks for — both linked across the channels.
    level: f32,
    gain: f32,
}

impl AudioNode for Limiter {
    fn prepare(&mut self, rate: f64) {
        self.rate = rate as f32;
        self.level = 0.0;
        self.gain = 1.0;
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let (input, ceiling, attack, release) =
            (&b.ins[0], &b.params[P::CEILING], &b.params[P::ATTACK], &b.params[P::RELEASE]);
        let channels = (b.outs[0].channels() as usize).min(MAX_CHANNELS as usize);
        let (ceil, att, rel) = (ceiling.chan(0), attack.chan(0), release.chan(0));

        // One pass to find the gain from the loudest channel, so the channels stay in proportion.
        let mut wanted = [1.0f32; BLOCK];
        for (i, g) in wanted.iter_mut().enumerate() {
            let peak = (0..channels).fold(0.0f32, |m, c| m.max(input.chan(c)[i].abs()));
            let coeff = |seconds: f32| match seconds > 0.0 {
                true => (-1.0 / (seconds * self.rate)).exp(),
                false => 0.0,
            };
            let a = if peak > self.level { coeff(att[i]) } else { coeff(rel[i]) };
            self.level = peak + a * (self.level - peak);
            let limit = ceil[i].max(0.05);
            self.gain = (limit / self.level.max(1e-6)).min(1.0);
            *g = self.gain;
        }

        for c in 0..channels {
            let x = input.chan(c);
            let y = b.outs[0].chan_mut(c);
            for i in 0..BLOCK {
                // The follower alone can be beaten by a peak that arrives inside its attack; the
                // tanh behind it is what makes the ceiling a fact rather than an aim.
                let limit = ceil[i].max(0.05);
                y[i] = limit * (x[i] * wanted[i] / limit).tanh();
            }
        }
    }
}

goofi_audio_sdk::export!(Limiter, MANIFEST);
