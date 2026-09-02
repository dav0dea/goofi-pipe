//! The shipped audio nodes, each written as an author writes one. Listed here until the audio
//! ABI lands (Step 7 of the audio program), which moves the DSP files into `nodes_audio/`; the
//! device and MIDI nodes stay, because their control halves own OS handles.

pub mod audio_in;
pub mod audio_out;
pub mod env;
pub mod feedback;
pub mod gain;
pub mod midi_in;
pub mod osc;
pub mod signal_in;

use std::sync::atomic::AtomicU16;
use std::sync::Arc;

use goofi_audio_sdk::{AudioNode, Manifest};

/// What the engine hands a node at birth: the rings a device or a port fills, and the width the
/// device answered — none of it for a node that owns no OS handle.
#[derive(Default)]
pub struct Birth {
    pub inbox: Option<rtrb::Consumer<f32>>,
    pub notes: Option<rtrb::Consumer<midi_in::Note>>,
    pub chans: Arc<AtomicU16>,
}

pub type Make = fn(Birth) -> Box<dyn AudioNode>;

pub static SHIPPED: &[(&str, &Manifest, Make)] = &[
    ("Osc", &osc::MANIFEST, |_| Box::new(osc::Osc::default())),
    ("Gain", &gain::MANIFEST, |_| Box::new(gain::Gain)),
    (audio_out::TYPE, &audio_out::MANIFEST, |_| Box::new(audio_out::AudioOut)),
    ("SignalIn", &signal_in::MANIFEST, |_| Box::new(signal_in::SignalIn)),
    ("Env", &env::MANIFEST, |_| Box::new(env::Env::default())),
    ("Feedback", &feedback::MANIFEST, |_| Box::new(feedback::Feedback)),
    (audio_in::TYPE, &audio_in::MANIFEST, |b| Box::new(audio_in::AudioIn::new(b))),
    (midi_in::TYPE, &midi_in::MANIFEST, |b| Box::new(midi_in::MidiIn::new(b))),
];
