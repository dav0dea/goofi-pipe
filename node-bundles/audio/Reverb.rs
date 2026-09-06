use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, Tag, BLOCK};

goofi_audio_sdk::params! {
    MIX = ParamDecl {
        group: "reverb",
        name: "mix",
        spec: ParamSpec::Float { default: 0.45, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("how much of what leaves is the room rather than the sound that entered it"),
    },
    DECAY = ParamDecl {
        group: "reverb",
        name: "decay",
        spec: ParamSpec::Float { default: 6.0, min: 0.2, max: 60.0 },
        expression: None,
        doc: Some("seconds for the tail to fall away"),
    },
    SIZE = ParamDecl {
        group: "reverb",
        name: "size",
        spec: ParamSpec::Float { default: 0.8, min: 0.1, max: 1.8 },
        expression: None,
        doc: Some("how far apart the walls are; small rooms ring, large ones wash"),
    },
    DAMPING = ParamDecl {
        group: "reverb",
        name: "damping",
        spec: ParamSpec::Float { default: 0.5, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("how fast the top of the tail is lost, as it is in a room with soft walls"),
    },
    PREDELAY = ParamDecl {
        group: "reverb",
        name: "predelay",
        spec: ParamSpec::Float { default: 0.02, min: 0.0, max: 0.25 },
        expression: None,
        doc: Some("seconds of silence before the room answers, which is what sets the listener back from the source"),
    },
    MODULATION = ParamDecl {
        group: "reverb",
        name: "modulation",
        spec: ParamSpec::Float { default: 0.35, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("how much the walls move; a little of it stops the tail from ringing on one note"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "A room around the sound.\n\
          Eight delay lines fed back through each other, out in stereo.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

const LINES: usize = 8;
const MAX_PREDELAY: f32 = 0.25;
const MAX_SIZE: f32 = 1.8;
/// Coprime lengths at 48 kHz, so no two lines ever line up and ring together.
static LENGTHS: [f32; LINES] = [1153.0, 1319.0, 1531.0, 1693.0, 1889.0, 2053.0, 2251.0, 2399.0];
/// The four lengths the input is diffused through before it reaches the tank.
static DIFFUSERS: [f32; 4] = [142.0, 379.0, 107.0, 277.0];
const SWING: f32 = 12.0;

/// A ring buffer read at a distance that may not be a whole number of samples.
#[derive(Default)]
struct Line {
    buf: Vec<f32>,
    write: usize,
}

impl Line {
    fn sized(len: usize) -> Line {
        Line { buf: vec![0.0; len.max(4)], write: 0 }
    }

    fn read(&self, back: f32) -> f32 {
        let len = self.buf.len();
        let back = back.clamp(1.0, len as f32 - 2.0);
        let whole = back.floor();
        let frac = back - whole;
        let at = (self.write + len - whole as usize) % len;
        let before = (at + len - 1) % len;
        self.buf[at] * (1.0 - frac) + self.buf[before] * frac
    }

    fn push(&mut self, v: f32) {
        self.buf[self.write] = v;
        self.write = (self.write + 1) % self.buf.len();
    }
}

#[derive(Default)]
struct Reverb {
    rate: f32,
    predelay: Line,
    diffusers: Vec<Line>,
    tank: Vec<Line>,
    damped: [f32; LINES],
    lfo: f32,
}

impl AudioNode for Reverb {
    fn channels(&self, _ins: &[u16], _params: &[f64], outs: usize) -> Vec<u16> {
        vec![2; outs]
    }

    fn prepare(&mut self, rate: f64) {
        self.rate = rate as f32;
        let scale = self.rate / 48_000.0;
        self.predelay = Line::sized((MAX_PREDELAY * self.rate) as usize + 4);
        self.diffusers = DIFFUSERS.iter().map(|l| Line::sized((l * scale) as usize + 4)).collect();
        self.tank = LENGTHS
            .iter()
            .map(|l| Line::sized((l * MAX_SIZE * scale + SWING) as usize + 4))
            .collect();
        self.damped = [0.0; LINES];
        self.lfo = 0.0;
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let input = &b.ins[0];
        let scale = self.rate / 48_000.0;
        let (mix, decay) = (b.params[P::MIX].chan(0), b.params[P::DECAY].chan(0));
        let (size, damping) = (b.params[P::SIZE].chan(0), b.params[P::DAMPING].chan(0));
        let (predelay, modulation) = (b.params[P::PREDELAY].chan(0), b.params[P::MODULATION].chan(0));
        let voices = b.ins[0].channels() as usize;

        let mut wet = [[0.0f32; BLOCK]; 2];
        let mut dry = [0.0f32; BLOCK];
        for i in 0..BLOCK {
            let x = (0..voices).map(|c| input.chan(c)[i]).sum::<f32>() / voices.max(1) as f32;
            dry[i] = x;

            self.predelay.push(x);
            let mut through = self.predelay.read((predelay[i] * self.rate).max(1.0));
            for (d, len) in self.diffusers.iter_mut().zip(DIFFUSERS) {
                let back = d.read(len * scale);
                let into = through + back * 0.6;
                d.push(into);
                through = back - into * 0.6;
            }

            // Householder: every line hears the average of all of them, less itself.
            let mut taken = [0.0f32; LINES];
            for (l, t) in taken.iter_mut().enumerate() {
                let wobble = SWING * modulation[i] * (std::f32::consts::TAU * (self.lfo + l as f32 * 0.125)).sin();
                *t = self.tank[l].read(LENGTHS[l] * size[i] * scale + wobble + SWING);
            }
            let sum = taken.iter().sum::<f32>() * 2.0 / LINES as f32;
            let cut = damping[i].clamp(0.0, 0.98);
            for l in 0..LINES {
                let length = LENGTHS[l] * size[i] * scale;
                // Each line loses the same 60 dB over `decay`, whatever its own length is.
                let gain = 10f32.powf(-3.0 * length / (decay[i].max(0.2) * self.rate));
                let fed = (sum - taken[l]) * gain + through * 0.35;
                self.damped[l] += (fed - self.damped[l]) * (1.0 - cut);
                self.tank[l].push(self.damped[l]);
            }
            self.lfo = (self.lfo + 0.13 * self.rate.recip()).fract();
            for (side, out) in wet.iter_mut().enumerate() {
                out[i] = taken.iter().skip(side).step_by(2).sum::<f32>() * 0.5;
            }
        }

        for c in 0..b.outs[0].channels() as usize {
            let y = b.outs[0].chan_mut(c);
            for i in 0..BLOCK {
                y[i] = dry[i] * (1.0 - mix[i]) + wet[c.min(1)][i] * mix[i];
            }
        }
    }
}

goofi_audio_sdk::export!(Reverb, MANIFEST);
