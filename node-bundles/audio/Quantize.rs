use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, SlotDecl, Tag, BLOCK};

goofi_audio_sdk::params! {
    SCALE = ParamDecl {
        group: "quantize",
        name: "scale",
        spec: ParamSpec::Str {
            default: "major",
            options: &[
                "chromatic", "major", "minor", "pentatonic_major", "pentatonic_minor", "dorian",
                "phrygian", "lydian", "mixolydian", "blues",
            ],
            refresh: false,
        },
        expression: None,
        doc: Some("which notes a pitch is allowed to land on"),
    },
    ROOT = ParamDecl {
        group: "quantize",
        name: "root",
        spec: ParamSpec::Int { default: 0, min: 0, max: 11 },
        expression: None,
        doc: Some("the note the scale is built from, in semitones above C"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "A pitch in volts pulled to the nearest note of a scale.",
    inputs: INS,
    outputs: OUTS,
    params: PARAMS,
};

/// The semitones each scale admits, in the order the `scale` param offers them.
static SCALES: &[&[i32]] = &[
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    &[0, 2, 4, 5, 7, 9, 11],
    &[0, 2, 3, 5, 7, 8, 10],
    &[0, 2, 4, 7, 9],
    &[0, 3, 5, 7, 10],
    &[0, 2, 3, 5, 7, 9, 10],
    &[0, 1, 3, 5, 7, 8, 10],
    &[0, 2, 4, 6, 7, 9, 11],
    &[0, 2, 4, 5, 7, 9, 10],
    &[0, 3, 5, 6, 7, 10],
];

#[derive(Default)]
struct Quantize;

/// The nearest admitted semitone to `semis`, searched across the octave below and above so a
/// pitch just under the root does not climb a whole octave to reach it.
fn snap(semis: f32, degrees: &[i32], root: i32) -> f32 {
    let octave = (semis / 12.0).floor() as i32;
    let mut best = semis;
    let mut gap = f32::INFINITY;
    for o in octave - 1..=octave + 1 {
        for d in degrees {
            let note = (o * 12 + root + d) as f32;
            if (note - semis).abs() < gap {
                gap = (note - semis).abs();
                best = note;
            }
        }
    }
    best
}

impl AudioNode for Quantize {
    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let input = &b.ins[0];
        let degrees = SCALES[(b.params[P::SCALE].chan(0)[0] as usize).min(SCALES.len() - 1)];
        let root = b.params[P::ROOT].chan(0)[0] as i32;
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            let x = input.chan(c);
            let y = out.chan_mut(c);
            for i in 0..BLOCK {
                y[i] = snap(x[i] * 12.0, degrees, root) / 12.0;
            }
        }
    }
}

goofi_audio_sdk::export!(Quantize, MANIFEST);
