use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, BLOCK};

goofi_audio_sdk::params! {
    BPM = ParamDecl {
        group: "chords",
        name: "bpm",
        spec: ParamSpec::Float { default: 88.0, min: 30.0, max: 240.0 },
        expression: None,
        doc: Some("tempo; each chord lasts one bar of four beats"),
    },
    HOLD = ParamDecl {
        group: "chords",
        name: "hold",
        spec: ParamSpec::Float { default: 0.85, min: 0.05, max: 1.0 },
        expression: None,
        doc: Some("fraction of the bar the chord sounds before it releases into the next"),
    },
}

/// The four voices of each chord, on four channels — wire `pitch` and `gate` to an instrument's
/// voice pitch and gate and it plays the progression polyphonically.
const VOICES: usize = 4;

/// A ii–V–I–VI turnaround in C, the backbone of a jazz standard: Dm7, G7, Cmaj7, A7♭.
/// Each row is four semitone offsets from C4, so the whole thing sits in one comfortable octave.
const PROG: [[i16; VOICES]; 4] = [
    [2, 5, 9, 12],   // Dm7  — D F A C
    [7, 11, 14, 17], // G7   — G B D F
    [0, 4, 7, 11],   // Cmaj7 — C E G B
    [9, 13, 16, 19], // A7   — A C# E G
];

static OUTS: &[OutputDecl] = &[
    OutputDecl { name: "pitch", kind: SlotType::Audio },
    OutputDecl { name: "gate", kind: SlotType::Audio },
];

static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "A looping jazz progression as voice signals: `pitch` carries four notes on four channels \
          (volts per octave, C4 is 0), `gate` opens for `hold` of each bar. Wire both to an \
          instrument's voice pitch and gate for hands-off comping.",
    inputs: &[],
    outputs: OUTS,
    params: PARAMS,
};

#[derive(Default)]
struct Chords {
    sr: f32,
    /// Seconds into the current bar.
    t: f32,
    idx: usize,
}

impl AudioNode for Chords {
    fn channels(&self, _ins: &[u16], _params: &[f64], outs: usize) -> Vec<u16> {
        vec![VOICES as u16; outs]
    }

    fn prepare(&mut self, rate: f64) {
        self.sr = rate as f32;
        self.t = 0.0;
        self.idx = 0;
    }

    fn process(&mut self, b: &mut Block<'_>) {
        let bpm = b.params[P::BPM].chan(0);
        let hold = b.params[P::HOLD].chan(0);
        for i in 0..BLOCK {
            let bar = 4.0 * 60.0 / bpm[i].max(1.0);
            self.t += 1.0 / self.sr;
            if self.t >= bar {
                self.t -= bar;
                self.idx = (self.idx + 1) % PROG.len();
            }
            let chord = PROG[self.idx];
            let open = self.t < bar * hold[i].clamp(0.0, 1.0);
            for v in 0..VOICES {
                b.outs[0].chan_mut(v)[i] = chord[v] as f32 / 12.0;
                b.outs[1].chan_mut(v)[i] = if open { 1.0 } else { 0.0 };
            }
        }
    }
}

goofi_audio_sdk::export!(Chords, MANIFEST);
