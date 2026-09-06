use goofi_audio_sdk::goofi_core::SlotType;
use goofi_audio_sdk::{Manifest, OutputDecl, ParamDecl, ParamSpec, Tag};

pub const TYPE: &str = "AudioPlayback";

goofi_audio_sdk::params! {
    FILE = ParamDecl {
        group: "play",
        name: "file",
        spec: ParamSpec::Str { default: "", options: &[], refresh: false },
        expression: None,
        doc: Some("the WAV file to play; a bare name is looked for in the recordings folder"),
    },
    POSITION = ParamDecl {
        group: "play",
        name: "position",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("where in the file to play from; playback runs on its own, and a MOVE of this skips"),
    },
    RESET = ParamDecl {
        group: "play",
        name: "reset",
        spec: ParamSpec::Pulse,
        expression: None,
        doc: Some("play from the start again"),
    },
    LOOPING = ParamDecl {
        group: "play",
        name: "loop",
        spec: ParamSpec::Bool { default: false },
        expression: None,
        doc: Some("start again at the end, instead of falling silent"),
    },
}

static OUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Audio }];

pub static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Input],
    doc: "A WAV file as audio.\n\
          As many channels as the file holds, resampled to the engine's rate.",
    inputs: &[],
    outputs: OUTS,
    params: PARAMS,
};
