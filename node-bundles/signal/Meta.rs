//! Meta — write the rate or the labels a frame carries, for a source that does not know its own.

use goofi_core::{resolve_axis, Axis, Coord, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Meta;

impl Node for Meta {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let mut meta = d.meta().clone();

        let sfreq = p.f64("meta", "sfreq").unwrap_or(0.0);
        if sfreq > 0.0 {
            meta.set_sfreq(Some(sfreq));
        }
        let text = p.str("meta", "labels").unwrap_or("");
        let names: Vec<&str> = text.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        if !names.is_empty() {
            let axis = resolve_axis(p.i64("meta", "axis").unwrap_or(0), a.shape().len())?;
            let len = a.shape()[axis];
            if names.len() != len {
                return Err(format!("`labels` names {} entries for an axis of {len}", names.len()).into());
            }
            let coords: Vec<Coord> = names.iter().map(|s| Coord::Str((*s).into())).collect();
            let axes = meta.channels().clone().with(axis, Axis::coords(coords));
            meta.set_channels(axes);
        }
        out.set("out", Data::array_f32(a.shape().to_vec(), a.as_bytes().to_vec(), meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "meta",
        name: "sfreq",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 100_000.0 },
        expression: None,
        doc: Some("The sample rate to write onto the frame, in Hz. 0 keeps whatever it arrived with."),
    },
    ParamDecl {
        group: "meta",
        name: "labels",
        spec: ParamSpec::Str { default: "", options: &[], refresh: false },
        expression: None,
        doc: Some(
            "Names for the entries along the chosen axis, separated by commas, one per entry. \
             Empty keeps the labels the frame arrived with.",
        ),
    },
    ParamDecl {
        group: "meta",
        name: "axis",
        spec: ParamSpec::Int { default: 0, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis the labels name, negative from the end."),
    },
];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "Write the sample rate or the channel names onto a frame, for a source that does not carry its own.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Meta, MANIFEST);
