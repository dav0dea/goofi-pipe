use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;

use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, OutputDecl, ParamDecl, ParamSpec, Tag};

use crate::nodes::Birth;
use crate::runtime::Inbox;

pub const TYPE: &str = "AudioIn";

goofi_audio_sdk::params! {
    DEVICE = ParamDecl {
        group: "audio",
        name: "device",
        spec: ParamSpec::Str { default: crate::DEFAULT_DEVICE, options: &[crate::DEFAULT_DEVICE], refresh: true },
        expression: None,
        doc: Some("the input device; one other than the clock's drifts, and the ring holds or drops at its edges"),
    },
}

static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

pub static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Input],
    doc: "The device's input, as many channels as it has.",
    inputs: &[],
    outputs: OUTS,
    params: PARAMS,
};

/// The DSP half reads the inbox the device's callback fills; the twin only answers the width.
pub struct AudioIn {
    inbox: Option<Inbox>,
    chans: Arc<AtomicU16>,
}

impl AudioIn {
    pub fn new(birth: Birth) -> AudioIn {
        AudioIn { inbox: birth.inbox.map(Inbox::new), chans: birth.chans }
    }
}

impl AudioNode for AudioIn {
    fn channels(&self, _ins: &[u16], _params: &[f64], outs: usize) -> Vec<u16> {
        vec![self.chans.load(Ordering::Relaxed).max(1); outs]
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
