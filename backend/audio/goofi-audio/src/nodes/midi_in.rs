use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, MAX_CHANNELS};

use crate::nodes::Birth;

pub const TYPE: &str = "MidiIn";
/// What the `port` param says when no port is opened.
pub const NO_PORT: &str = "none";

goofi_audio_sdk::params! {
    PORT = ParamDecl {
        group: "midi",
        name: "port",
        spec: ParamSpec::Str { default: NO_PORT, options: &[NO_PORT], refresh: true },
        expression: None,
        doc: None,
    },
    VOICES = ParamDecl {
        group: "midi",
        name: "voices",
        spec: ParamSpec::Int { default: 4, min: 1, max: MAX_CHANNELS as i64 },
        expression: None,
        doc: Some("one channel per voice on every output; notes take voices round-robin"),
    },
}

static OUTS: &[OutputDecl] = &[
    OutputDecl { name: "gate", kind: SlotType::Audio },
    OutputDecl { name: "pitch", kind: SlotType::Audio },
    OutputDecl { name: "velocity", kind: SlotType::Audio },
];

pub static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "A MIDI port as signals: per voice a gate, a pitch in volts per octave (C4 is 0) and a \
          velocity in [0, 1]. A note lands at the start of the next block.",
    inputs: &[],
    outputs: OUTS,
    params: PARAMS,
};

/// One note message as the port's callback hands it over.
#[derive(Clone, Copy, Debug)]
pub struct Note {
    pub on: bool,
    pub note: u8,
    pub velocity: u8,
}

impl Note {
    /// The note a raw MIDI message carries, if it is one: a note-on with zero velocity is off.
    pub fn parse(bytes: &[u8]) -> Option<Note> {
        let [status, note, velocity, ..] = *bytes else { return None };
        match status & 0xF0 {
            0x90 if velocity > 0 => Some(Note { on: true, note, velocity }),
            0x90 | 0x80 => Some(Note { on: false, note, velocity }),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Voice {
    gate: bool,
    note: u8,
    velocity: f32,
}

pub struct MidiIn {
    notes: Option<rtrb::Consumer<Note>>,
    voices: [Voice; MAX_CHANNELS as usize],
    next: usize,
}

impl MidiIn {
    pub fn new(birth: Birth) -> MidiIn {
        MidiIn { notes: birth.notes, voices: [Voice::default(); MAX_CHANNELS as usize], next: 0 }
    }

    fn land(&mut self, n: Note, voices: usize) {
        if n.on {
            let free = (0..voices).map(|k| (self.next + k) % voices).find(|v| !self.voices[*v].gate);
            let v = free.unwrap_or(self.next % voices);
            self.voices[v] = Voice { gate: true, note: n.note, velocity: f32::from(n.velocity) / 127.0 };
            self.next = (v + 1) % voices;
        } else if let Some(voice) = self.voices[..voices].iter_mut().find(|v| v.gate && v.note == n.note) {
            voice.gate = false;
        }
    }
}

impl AudioNode for MidiIn {
    fn channels(&self, _ins: &[u16], params: &[f64], outs: usize) -> Vec<u16> {
        let voices = params.get(P::VOICES).copied().unwrap_or(1.0) as u16;
        vec![voices.clamp(1, MAX_CHANNELS); outs]
    }

    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let voices = (b.outs[0].channels() as usize).clamp(1, MAX_CHANNELS as usize);
        while let Some(n) = self.notes.as_mut().and_then(|r| r.pop().ok()) {
            self.land(n, voices);
        }
        for c in 0..voices {
            let voice = self.voices[c];
            b.outs[0].chan_mut(c).fill(if voice.gate { 1.0 } else { 0.0 });
            b.outs[1].chan_mut(c).fill((f32::from(voice.note) - 60.0) / 12.0);
            b.outs[2].chan_mut(c).fill(voice.velocity);
        }
    }
}
