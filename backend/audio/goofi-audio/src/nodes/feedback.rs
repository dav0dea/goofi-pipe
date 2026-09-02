use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, SlotDecl};

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: false, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

pub static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "Its input one block late: the one way a loop closes. It runs first each block, reading \
          what its producer left in the last one.",
    inputs: INS,
    outputs: OUTS,
    params: &[],
};

pub struct Feedback;

impl AudioNode for Feedback {
    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let input = &b.ins[0];
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            out.chan_mut(c).copy_from_slice(input.chan(c));
        }
    }

    fn feedback(&self) -> bool {
        true
    }
}
