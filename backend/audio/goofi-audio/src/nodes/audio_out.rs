use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{AudioNode, Block, Manifest, ParamDecl, ParamSpec, SlotDecl};

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
}

static INS: &[SlotDecl] =
    &[SlotDecl { name: "input", kind: SlotType::Audio, trigger_process: false, multi: true, required: false }];

pub static MANIFEST: Manifest = Manifest {
    category: "audio",
    doc: "The device: what reaches `input` is heard, times `gain`; every AudioOut on the device sums.",
    inputs: INS,
    outputs: &[],
    params: PARAMS,
};

pub struct AudioOut;

impl AudioNode for AudioOut {
    fn prepare(&mut self, _rate: f64) {}

    fn process(&mut self, _b: &mut Block<'_>) {}
}
