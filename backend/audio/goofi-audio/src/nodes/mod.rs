//! The shipped audio nodes, each written as an author writes one. Listed here until the audio
//! ABI lands (Step 7 of the audio program), which moves the files into `nodes_audio/` and deletes
//! this slice.

pub mod audio_out;
pub mod gain;
pub mod osc;

use goofi_audio_sdk::{AudioNode, Manifest};

pub type Make = fn() -> Box<dyn AudioNode>;

pub static SHIPPED: &[(&str, &Manifest, Make)] = &[
    ("Osc", &osc::MANIFEST, || Box::new(osc::Osc::default())),
    ("Gain", &gain::MANIFEST, || Box::new(gain::Gain)),
    ("AudioOut", &audio_out::MANIFEST, || Box::new(audio_out::AudioOut)),
];
