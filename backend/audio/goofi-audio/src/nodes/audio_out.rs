use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, ParamDecl, ParamSpec, SlotDecl, Tag, BLOCK};

use crate::nodes::Birth;

pub const TYPE: &str = "AudioOut";

goofi_audio_sdk::params! {
    DEVICE = ParamDecl {
        group: "audio",
        name: "device",
        spec: ParamSpec::Str { default: crate::DEFAULT_DEVICE, options: &[crate::DEFAULT_DEVICE], refresh: true },
        expression: None,
        doc: Some("the output device the engine's clock follows; every AudioOut names the same one"),
    },
    GAIN = ParamDecl {
        group: "audio",
        name: "gain",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 10.0 },
        expression: None,
        doc: None,
    },
    ON = ParamDecl {
        group: "record",
        name: "on",
        spec: ParamSpec::Bool { default: false },
        expression: None,
        doc: Some("write the input to the file while this is high, before gain: gain is what you hear"),
    },
    FILE = ParamDecl {
        group: "record",
        name: "file",
        spec: ParamSpec::Str { default: "take", options: &[], refresh: false },
        expression: None,
        doc: Some("a bare name lands in the recordings folder; an absolute path is taken as it is"),
    },
    UNIQUE = ParamDecl {
        group: "record",
        name: "unique",
        spec: ParamSpec::Bool { default: true },
        expression: None,
        doc: Some("join the time to the name, so a take never replaces the one before it"),
    },
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];

pub static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Output],
    doc: "The sound device: what reaches `input` is heard, times `gain`.\n\
          Every AudioOut on the device sums. `record.on` writes the same input to a WAV file, \
          before gain.",
    inputs: INS,
    outputs: &[],
    params: PARAMS,
};

/// The DSP half only fills the take's ring; the control half owns the file.
pub struct AudioOut {
    rec: Option<rtrb::Producer<f32>>,
}

impl AudioOut {
    pub fn new(birth: Birth) -> AudioOut {
        AudioOut { rec: birth.rec }
    }
}

impl AudioNode for AudioOut {
    fn audio_params(&self, _declared: usize) -> usize {
        0
    }

    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, b: &mut Block<'_>) {
        let Some(rec) = &mut self.rec else { return };
        if !goofi_audio_sdk::high(b.scalars[P::ON]) {
            return;
        }
        let input = &b.ins[0];
        let c = input.channels();
        if let Ok(chunk) = rec.write_chunk_uninit(1 + c as usize * BLOCK) {
            let samples = (0..c as usize).flat_map(|ch| input.chan(ch).iter().copied());
            chunk.fill_from_iter(std::iter::once(c as f32).chain(samples));
        }
    }
}
