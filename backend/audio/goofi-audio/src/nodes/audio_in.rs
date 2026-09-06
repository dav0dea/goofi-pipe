use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{Manifest, OutputDecl, ParamDecl, ParamSpec, Tag};

pub const TYPE: &str = "AudioIn";

goofi_audio_sdk::params! {
    DEVICE = ParamDecl {
        group: "audio",
        name: "device",
        spec: ParamSpec::Str { default: crate::DEFAULT_DEVICE, options: &[crate::DEFAULT_DEVICE], refresh: true },
        expression: None,
        doc: Some("the input device; one other than the clock's drifts, and the ring holds or drops at its edges — an `ASIO: ` name must be the same driver the rest of the patch uses, since only one loads at a time"),
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
