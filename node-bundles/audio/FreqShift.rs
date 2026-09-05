use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, Tag, BLOCK, MAX_CHANNELS};

goofi_audio_sdk::params! {
    FREQUENCY = ParamDecl {
        group: "freq_shift",
        name: "frequency",
        spec: ParamSpec::Float { default: 0.0, min: -5000.0, max: 5000.0 },
        expression: None,
        doc: Some("hertz to move every partial by, up or down; it is an addition, so harmony does not survive it"),
    },
    MODE = ParamDecl {
        group: "freq_shift",
        name: "mode",
        spec: ParamSpec::Str { default: "single", options: &["single", "ring"], refresh: false },
        expression: None,
        doc: Some("`single` moves the spectrum one way; `ring` keeps both sides and is the harsher of the two"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "Every partial moved by the same number of hertz, which no interval survives.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

/// A second-order allpass in the form the Hilbert pair is built from: one coefficient, two delays.
#[derive(Clone, Copy, Default)]
struct Allpass {
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Allpass {
    fn run(&mut self, a2: f32, x: f32) -> f32 {
        let y = a2 * (x + self.y2) - self.x2;
        (self.x2, self.x1) = (self.x1, x);
        (self.y2, self.y1) = (self.y1, y);
        y
    }
}

/// The two cascades whose outputs stay a quarter cycle apart across the band.
static CHAIN_I: [f32; 4] = [0.692_387_8, 0.936_065_4, 0.988_229_5, 0.998_748_8];
static CHAIN_Q: [f32; 4] = [0.402_192_1, 0.856_171_1, 0.972_290_9, 0.995_288_5];

#[derive(Clone, Copy, Default)]
struct Voice {
    i: [Allpass; 4],
    q: [Allpass; 4],
    /// The in-phase chain runs one sample ahead; this is what puts the two back together.
    late: f32,
    phase: f32,
}

#[derive(Default)]
struct FreqShift {
    step: f32,
    voices: [Voice; MAX_CHANNELS as usize],
}

impl AudioNode for FreqShift {
    fn prepare(&mut self, rate: f64) {
        self.step = 1.0 / rate as f32;
        self.voices = [Voice::default(); MAX_CHANNELS as usize];
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let (input, frequency) = (&b.ins[0], &b.params[P::FREQUENCY]);
        let ring = b.params[P::MODE].chan(0)[0] as u8 == 1;
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            let (x, f) = (input.chan(c), frequency.chan(c));
            let v = &mut self.voices[c];
            let y = out.chan_mut(c);
            for i in 0..BLOCK {
                let (sin, cos) = (std::f32::consts::TAU * v.phase).sin_cos();
                v.phase = (v.phase + f[i] * self.step).fract();
                if ring {
                    y[i] = x[i] * cos;
                    continue;
                }
                let mut a = x[i];
                let mut q = x[i];
                for (s, k) in v.i.iter_mut().zip(CHAIN_I) {
                    a = s.run(k, a);
                }
                for (s, k) in v.q.iter_mut().zip(CHAIN_Q) {
                    q = s.run(k, q);
                }
                y[i] = v.late * cos + q * sin;
                v.late = a;
            }
        }
    }
}

goofi_audio_sdk::export!(FreqShift, MANIFEST);
