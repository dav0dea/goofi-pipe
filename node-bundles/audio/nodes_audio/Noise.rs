use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, BLOCK};

goofi_audio_sdk::params! {
    RATE = ParamDecl {
        group: "noise",
        name: "rate",
        spec: ParamSpec::Float { default: 2.0, min: 0.01, max: 50.0 },
        expression: None,
        doc: Some("new random target per second, per output"),
    },
    SMOOTH = ParamDecl {
        group: "noise",
        name: "smooth",
        spec: ParamSpec::Float { default: 0.7, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("0 steps hard between targets, 1 glides the whole way — the shape of the wander"),
    },
}

/// Ten decorrelated streams: enough to drive a plugin's core parameters each its own way.
const N: usize = 10;

static OUTS: &[OutputDecl] = &[
    OutputDecl { name: "a", kind: SlotType::Audio },
    OutputDecl { name: "b", kind: SlotType::Audio },
    OutputDecl { name: "c", kind: SlotType::Audio },
    OutputDecl { name: "d", kind: SlotType::Audio },
    OutputDecl { name: "e", kind: SlotType::Audio },
    OutputDecl { name: "f", kind: SlotType::Audio },
    OutputDecl { name: "g", kind: SlotType::Audio },
    OutputDecl { name: "h", kind: SlotType::Audio },
    OutputDecl { name: "i", kind: SlotType::Audio },
    OutputDecl { name: "j", kind: SlotType::Audio },
];

static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "Ten independent streams of smooth random modulation in [0, 1] — each wanders to a new \
          target `rate` times a second, gliding by `smooth`. Wire each output to a different \
          parameter for movement that never repeats.",
    inputs: &[],
    outputs: OUTS,
    params: PARAMS,
};

/// One wandering stream: where it is, where it is heading, how far along, and its own random state.
#[derive(Clone, Copy)]
struct Stream {
    val: f32,
    from: f32,
    to: f32,
    phase: f32,
    rng: u32,
}

impl Default for Stream {
    fn default() -> Stream {
        Stream { val: 0.5, from: 0.5, to: 0.5, phase: 1.0, rng: 1 }
    }
}

impl Stream {
    /// xorshift32 in [0, 1): no crate, no clock, deterministic per seed.
    fn rand(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }
}

struct Noise {
    sr: f32,
    st: [Stream; N],
}

impl Default for Noise {
    fn default() -> Noise {
        Noise { sr: 48000.0, st: [Stream::default(); N] }
    }
}

impl AudioNode for Noise {
    fn channels(&self, _ins: &[u16], _params: &[f64], outs: usize) -> Vec<u16> {
        vec![1; outs]
    }

    fn prepare(&mut self, rate: f64) {
        self.sr = rate as f32;
        // Each stream seeded by its index, so the ten wander independently rather than in lockstep.
        for (k, s) in self.st.iter_mut().enumerate() {
            s.rng = (k as u32).wrapping_mul(2_654_435_761).wrapping_add(1) | 1;
            s.to = s.rand();
            s.from = s.to;
            s.val = s.to;
            s.phase = 1.0;
        }
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let rate = b.params[P::RATE].chan(0);
        let smooth = b.params[P::SMOOTH].chan(0);
        for k in 0..N {
            let s = &mut self.st[k];
            let out = &mut b.outs[k];
            let y = out.chan_mut(0);
            for i in 0..BLOCK {
                let step = (rate[i] / self.sr).max(0.0);
                s.phase += step;
                if s.phase >= 1.0 {
                    s.phase -= s.phase.floor();
                    s.from = s.to;
                    s.to = s.rand();
                }
                // A raised-cosine ease between targets: `smooth` blends the hard step into the glide.
                let eased = 0.5 - 0.5 * (std::f32::consts::PI * s.phase).cos();
                let shaped = s.phase + (eased - s.phase) * smooth[i].clamp(0.0, 1.0);
                s.val = s.from + (s.to - s.from) * shaped;
                y[i] = s.val;
            }
        }
    }
}

goofi_audio_sdk::export!(Noise, MANIFEST);
