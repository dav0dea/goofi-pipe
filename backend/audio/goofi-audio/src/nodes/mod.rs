//! The nodes built into the engine: the device and MIDI nodes, whose control halves own OS
//! handles. Every other audio node — shipped or authored — is one `.rs` file, loaded
//! behind the audio ABI.

pub mod audio_in;
pub mod audio_out;
pub mod audio_playback;
pub mod midi_in;

use std::sync::atomic::AtomicU16;
use std::sync::Arc;

use goofi_audio_sdk::{AudioNode, Block, Manifest};
use goofi_node::NodeManifest;

use crate::runtime::Inbox;

/// What the engine hands a node at birth: the rings a device or a port fills, and the width the
/// device answered — none of it for a node that owns no OS handle.
#[derive(Default)]
pub struct Birth {
    pub inbox: Option<rtrb::Consumer<f32>>,
    /// The take's ring, which only `AudioOut` fills: the DSP half's end of it.
    pub rec: Option<rtrb::Producer<f32>>,
    pub notes: Option<rtrb::Consumer<midi_in::Note>>,
    pub chans: Arc<AtomicU16>,
    /// The window thread, where a plugin is made and unmade; none where the machine has no display.
    pub ui: Option<crate::ui::Ui>,
    /// Which node this is, and the engine's inbox — what a plugin's editor writes through. None
    /// for the twin, which is no instance.
    pub uid: Option<goofi_node::Uid>,
    pub shared: Option<Arc<crate::control::AudioShared>>,
}

/// The DSP half of every node its own control half feeds — the device's input, and a file's. It
/// reads the ring and is as wide as whoever fills it last said.
pub struct Fed {
    inbox: Option<Inbox>,
    chans: Arc<AtomicU16>,
}

impl Fed {
    pub fn new(birth: Birth, catch_up: bool) -> Fed {
        Fed { inbox: birth.inbox.map(|ring| Inbox::new(ring, catch_up)), chans: birth.chans }
    }
}

impl AudioNode for Fed {
    fn channels(&self, _ins: &[u16], _params: &[f64], outs: usize) -> Vec<u16> {
        vec![self.chans.load(std::sync::atomic::Ordering::Relaxed).max(1); outs]
    }

    fn audio_params(&self, _declared: usize) -> usize {
        0
    }

    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let out = &mut b.outs[0];
        match &mut self.inbox {
            Some(inbox) => inbox.fill(out),
            None => {
                for c in 0..out.channels() as usize {
                    out.chan_mut(c).fill(0.0);
                }
            }
        }
    }
}

pub type Born = fn(Birth) -> Box<dyn AudioNode>;

/// One node class the engine can build, built in or loaded.
#[derive(Clone)]
pub struct Class {
    pub manifest: &'static NodeManifest,
    pub make: Arc<dyn Fn(Birth) -> Box<dyn AudioNode> + Send + Sync>,
    /// A VST3 class, by what its scan derived; goofi's own nodes hold none.
    pub plugin: Option<Arc<crate::vst3::Derived>>,
}

/// Whether the engine treats `type_name` by name — a file may not take it.
pub fn built_in(type_name: &str) -> bool {
    BUILT_IN.iter().any(|(name, ..)| *name == type_name)
}

pub static BUILT_IN: &[(&str, &Manifest, Born)] = &[
    (audio_out::TYPE, &audio_out::MANIFEST, |b| Box::new(audio_out::AudioOut::new(b))),
    // The device drops what piled up; the file keeps it, or a skip is what a late tick sounds like.
    (audio_in::TYPE, &audio_in::MANIFEST, |b| Box::new(Fed::new(b, true))),
    (audio_playback::TYPE, &audio_playback::MANIFEST, |b| Box::new(Fed::new(b, false))),
    (midi_in::TYPE, &midi_in::MANIFEST, |b| Box::new(midi_in::MidiIn::new(b))),
];
