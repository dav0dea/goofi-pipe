use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, Tag, MAX_CHANNELS};

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
    BEND = ParamDecl {
        group: "midi",
        name: "bend_range",
        spec: ParamSpec::Float { default: 2.0, min: 0.0, max: 24.0 },
        expression: None,
        doc: Some("how many semitones a full pitch wheel reaches, up and down"),
    },
    VOICES = ParamDecl {
        group: "midi",
        name: "voices",
        spec: ParamSpec::Int { default: 4, min: 1, max: MAX_CHANNELS as i64 },
        expression: None,
        doc: Some(
            "one channel per voice on every output; notes take voices round-robin. The bundled              `voices` output needs two channels per voice, so it carries the first 8 — past that,              wire gate, pitch and velocity separately",
        ),
    },
}

static OUTS: &[OutputDecl] = &[
    OutputDecl { name: "gate", kind: SlotType::Audio },
    OutputDecl { name: "pitch", kind: SlotType::Audio },
    OutputDecl { name: "velocity", kind: SlotType::Audio },
    OutputDecl { name: "voices", kind: SlotType::Audio },
];

pub static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Input, Tag::Midi],
    doc: "A MIDI port as signals.\n\
          Per voice a gate, a pitch in volts per octave (C4 is 0) and a velocity in [0, 1]. A \
          note lands at the start of the next block.",
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
    /// The sustain pedal moved rather than a key: `on` is the pedal's new state.
    pub pedal: bool,
    /// The pitch wheel moved rather than a key: `bend` is where it now sits, -1 to 1.
    pub wheel: Option<f32>,
}

impl Note {
    /// The note a raw MIDI message carries, if it is one: a note-on with zero velocity is off.
    /// The sustain pedal rides here too — it decides when a note-off is obeyed, so it has to be
    /// in the same order as the notes it holds.
    pub fn parse(bytes: &[u8]) -> Option<Note> {
        let [status, note, velocity, ..] = *bytes else { return None };
        match status & 0xF0 {
            0x90 if velocity > 0 => Some(Note { on: true, note, velocity, pedal: false, wheel: None }),
            0x90 | 0x80 => Some(Note { on: false, note, velocity, pedal: false, wheel: None }),
            // CC 64 is sustain, and by the spec anything from 64 up is DOWN.
            0xB0 if note == 64 => Some(Note { on: velocity >= 64, note, velocity, pedal: true, wheel: None }),
            // The wheel is 14 bits across the two data bytes, centred at 8192 and asymmetric:
            // 8191 up against 8192 down, so each side is scaled by its own end.
            0xE0 => {
                let raw = i32::from(u16::from(velocity) << 7 | u16::from(note)) - 8192;
                let bend = raw as f32 / if raw >= 0 { 8191.0 } else { 8192.0 };
                Some(Note { on: false, note, velocity, pedal: false, wheel: Some(bend) })
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Voice {
    gate: bool,
    note: u8,
    velocity: f32,
    /// The key is up and only the pedal is still holding this voice down.
    held_by_pedal: bool,
}

pub struct MidiIn {
    notes: Option<rtrb::Consumer<Note>>,
    voices: [Voice; MAX_CHANNELS as usize],
    next: usize,
    pedal: bool,
    bend: f32,
}

impl MidiIn {
    pub fn new(birth: Birth) -> MidiIn {
        MidiIn { notes: birth.notes, voices: [Voice::default(); MAX_CHANNELS as usize], next: 0, pedal: false, bend: 0.0 }
    }

    /// A note-on takes the next free voice round-robin — or the voice already holding that note;
    /// a note-off frees its voice wherever it is, past a shrunk count included.
    fn land(&mut self, n: Note, voices: usize) {
        if let Some(bend) = n.wheel {
            self.bend = bend;
            return;
        }
        // Lifting the pedal is what finally releases every key already let go under it.
        if n.pedal {
            self.pedal = n.on;
            if !n.on {
                for v in self.voices.iter_mut().filter(|v| v.held_by_pedal) {
                    v.gate = false;
                    v.held_by_pedal = false;
                }
            }
            return;
        }
        let held = self.voices.iter_mut().find(|v| v.gate && v.note == n.note);
        match (n.on, held) {
            // Retaking a key the pedal still holds is a fresh press, so it stops being the
            // pedal's — otherwise lifting the pedal would cut a note being played.
            (true, Some(voice)) => {
                voice.velocity = f32::from(n.velocity) / 127.0;
                voice.held_by_pedal = false;
            }
            (true, None) => {
                let free = (0..voices).map(|k| (self.next + k) % voices).find(|v| !self.voices[*v].gate);
                let v = free.unwrap_or(self.next % voices);
                self.voices[v] =
                    Voice { gate: true, note: n.note, velocity: f32::from(n.velocity) / 127.0, held_by_pedal: false };
                self.next = (v + 1) % voices;
            }
            (false, Some(voice)) => {
                if self.pedal {
                    voice.held_by_pedal = true;
                } else {
                    voice.gate = false;
                }
            }
            (false, None) => {}
        }
    }
}

impl AudioNode for MidiIn {
    /// The three plain outputs are one channel per voice; `voices` is pitches then velocities, so
    /// a reader halves the count. The gate is not carried because MIDI does not carry one either:
    /// a velocity of zero IS the note off, which is what buys the eighth voice.
    fn channels(&self, _ins: &[u16], params: &[f64], outs: usize) -> Vec<u16> {
        let voices = (params.get(P::VOICES).copied().unwrap_or(1.0) as u16).clamp(1, MAX_CHANNELS);
        (0..outs).map(|i| if i == 3 { (voices * 2).min(MAX_CHANNELS) } else { voices }).collect()
    }

    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let voices = (b.outs[0].channels() as usize).clamp(1, MAX_CHANNELS as usize);
        // The wheel bends the PITCH rather than riding a channel of its own, so every instrument
        // hears it through the one signal it already reads — the cable included.
        let semitones = b.params.get(P::BEND).map_or(2.0, |p| p.chan(0)[0]);
        while let Some(n) = self.notes.as_mut().and_then(|r| r.pop().ok()) {
            self.land(n, voices);
        }
        // The bundle is capped like any other port, so a voice count whose pair would not fit
        // carries as many whole voices as it can rather than a torn last one.
        let bundled = (b.outs[3].channels() as usize / 2).min(voices);
        for c in 0..voices {
            let voice = self.voices[c];
            let (gate, vel) = (if voice.gate { 1.0 } else { 0.0 }, voice.velocity);
            let pitch = (f32::from(voice.note) - 60.0 + self.bend * semitones) / 12.0;
            b.outs[0].chan_mut(c).fill(gate);
            b.outs[1].chan_mut(c).fill(pitch);
            b.outs[2].chan_mut(c).fill(vel);
            if c < bundled {
                b.outs[3].chan_mut(c).fill(pitch);
                // A silent voice carries zero, which is the note-off the other side reads.
                b.outs[3].chan_mut(bundled + c).fill(if voice.gate { vel.max(f32::MIN_POSITIVE) } else { 0.0 });
            }
        }
    }
}
