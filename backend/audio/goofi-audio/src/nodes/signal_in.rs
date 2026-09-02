use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, SlotDecl};

static INS: &[SlotDecl] =
    &[SlotDecl { name: "data", kind: SlotType::Array, trigger_process: false, multi: false, required: false }];
static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

pub static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "The in-order crossing: a `[C, T]` signal frame with `sfreq` enters as `C` audio channels, \
          resampled to the rate. A frame with no `sfreq` enters one sample per sample, so a control \
          value is held until the next.",
    inputs: INS,
    outputs: OUTS,
    params: &[],
};

pub struct SignalIn;

impl AudioNode for SignalIn {
    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let input = &b.ins[0];
        let out = &mut b.outs[0];
        for c in 0..out.channels() as usize {
            out.chan_mut(c).copy_from_slice(input.chan(c));
        }
    }
}
