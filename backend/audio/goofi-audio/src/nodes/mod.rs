//! The nodes built into the engine: the device and MIDI nodes, whose control halves own OS
//! handles. Every other audio node — shipped or authored — is one file in `nodes_audio/`, loaded
//! behind the audio ABI.

pub mod audio_in;
pub mod audio_out;
pub mod midi_in;

use std::sync::atomic::AtomicU16;
use std::sync::Arc;

use goofi_audio_sdk::{AudioNode, Manifest};
use goofi_node::NodeManifest;

/// What the engine hands a node at birth: the rings a device or a port fills, and the width the
/// device answered — none of it for a node that owns no OS handle.
#[derive(Default)]
pub struct Birth {
    pub inbox: Option<rtrb::Consumer<f32>>,
    pub notes: Option<rtrb::Consumer<midi_in::Note>>,
    pub chans: Arc<AtomicU16>,
}

pub type Born = fn(Birth) -> Box<dyn AudioNode>;

/// One node class the engine can build, built in or loaded.
#[derive(Clone)]
pub struct Class {
    pub manifest: &'static NodeManifest,
    pub make: Arc<dyn Fn(Birth) -> Box<dyn AudioNode> + Send + Sync>,
}

/// Whether the engine treats `type_name` by name — a file may not take it.
pub fn built_in(type_name: &str) -> bool {
    BUILT_IN.iter().any(|(name, ..)| *name == type_name)
}

pub static BUILT_IN: &[(&str, &Manifest, Born)] = &[
    (audio_out::TYPE, &audio_out::MANIFEST, |_| Box::new(audio_out::AudioOut)),
    (audio_in::TYPE, &audio_in::MANIFEST, |b| Box::new(audio_in::AudioIn::new(b))),
    (midi_in::TYPE, &midi_in::MANIFEST, |b| Box::new(midi_in::MidiIn::new(b))),
];
