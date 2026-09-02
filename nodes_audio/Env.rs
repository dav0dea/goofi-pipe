use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Edge, Manifest, OutputDecl, ParamDecl, ParamSpec, BLOCK, GATE_HIGH, MAX_CHANNELS};

goofi_audio_sdk::params! {
    GATE = ParamDecl {
        group: "env",
        name: "gate",
        spec: ParamSpec::Bool { default: false },
        expression: None,
        doc: Some("HIGH at 0.5 and above; an audio reference is one voice per channel"),
    },
    ATTACK = ParamDecl {
        group: "env",
        name: "attack",
        spec: ParamSpec::Float { default: 0.01, min: 0.0, max: 10.0 },
        expression: None,
        doc: Some("seconds to full"),
    },
    DECAY = ParamDecl {
        group: "env",
        name: "decay",
        spec: ParamSpec::Float { default: 0.1, min: 0.0, max: 10.0 },
        expression: None,
        doc: Some("seconds from full to `sustain`"),
    },
    SUSTAIN = ParamDecl {
        group: "env",
        name: "sustain",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 1.0 },
        expression: None,
        doc: None,
    },
    RELEASE = ParamDecl {
        group: "env",
        name: "release",
        spec: ParamSpec::Float { default: 0.1, min: 0.0, max: 10.0 },
        expression: None,
        doc: Some("seconds from full to silence once the gate drops"),
    },
}

static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

pub static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "An ADSR in [0, 1] over `gate`, one voice per channel of it.",
    inputs: &[],
    outputs: OUTS,
    params: PARAMS,
};

#[derive(Clone, Copy, Default, PartialEq)]
enum Stage {
    #[default]
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Default)]
pub struct Env {
    stage: [Stage; MAX_CHANNELS as usize],
    level: [f32; MAX_CHANNELS as usize],
    edge: [Edge; MAX_CHANNELS as usize],
    step: f32,
}

impl AudioNode for Env {
    fn prepare(&mut self, rate: f64) {
        self.step = 1.0 / rate as f32;
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            let gate = b.params[P::GATE].chan(c);
            let attack = b.params[P::ATTACK].chan(c);
            let decay = b.params[P::DECAY].chan(c);
            let sustain = b.params[P::SUSTAIN].chan(c);
            let release = b.params[P::RELEASE].chan(c);
            let (stage, level, edge) = (&mut self.stage[c], &mut self.level[c], &mut self.edge[c]);
            let samples = out.chan_mut(c);
            for i in 0..BLOCK {
                if edge.rising(gate[i]) {
                    *stage = Stage::Attack;
                } else if gate[i] < GATE_HIGH && !matches!(*stage, Stage::Idle | Stage::Release) {
                    *stage = Stage::Release;
                }
                let per = |seconds: f32| self.step / seconds.max(1e-4);
                match *stage {
                    Stage::Attack => {
                        *level += per(attack[i]);
                        if *level >= 1.0 {
                            *level = 1.0;
                            *stage = Stage::Decay;
                        }
                    }
                    Stage::Decay => {
                        *level -= per(decay[i]) * (1.0 - sustain[i]);
                        if *level <= sustain[i] {
                            *level = sustain[i];
                            *stage = Stage::Sustain;
                        }
                    }
                    Stage::Sustain => *level = sustain[i],
                    Stage::Release => {
                        *level -= per(release[i]);
                        if *level <= 0.0 {
                            *level = 0.0;
                            *stage = Stage::Idle;
                        }
                    }
                    Stage::Idle => *level = 0.0,
                }
                samples[i] = *level;
            }
        }
    }
}

goofi_audio_sdk::export!(Env, MANIFEST);
