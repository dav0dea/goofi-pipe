use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, SlotDecl};

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];

pub static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "The device: what reaches `input` is what is heard. One per patch is read.",
    inputs: INS,
    outputs: &[],
    params: &[],
};

pub struct AudioOut;

impl AudioNode for AudioOut {
    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, _b: &mut Block<'_>) {}
}
