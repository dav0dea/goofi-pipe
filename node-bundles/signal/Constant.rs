//! Constant — a number, given a shape. With `value` in reference mode it is how a scalar from
//! anywhere in the patch becomes an array of the shape a node downstream wants.

use goofi_core::{Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, Tag};

#[derive(Default)]
struct Constant;

impl Node for Constant {
    fn process(
        &mut self,
        _inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let value = p.f64("constant", "value").unwrap_or(0.0);
        if !value.is_finite() {
            return Err(format!("`value` is not a number: {value}").into());
        }
        let text = p.str("constant", "shape").unwrap_or("1");
        let mut shape = Vec::new();
        for part in text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let n: usize = part.parse().map_err(|_| format!("`shape` takes whole numbers, not `{part}`"))?;
            if n == 0 {
                return Err("`shape` takes lengths of one or more".to_string().into());
            }
            shape.push(n);
        }
        if shape.is_empty() {
            shape.push(1);
        }
        let count: usize = shape.iter().product();
        if count > 1 << 24 {
            return Err(format!("`shape` asks for {count} entries, which is more than 16 million").into());
        }
        let buf: Vec<u8> = (value as f32).to_le_bytes().repeat(count);
        out.set("out", Data::array_f32(shape, buf, Meta::new()).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "constant",
        name: "value",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The number every entry carries; in reference mode it follows another node's output."),
    },
    ParamDecl {
        group: "constant",
        name: "shape",
        spec: ParamSpec::Str { default: "1", options: &[], refresh: false },
        expression: None,
        doc: Some("The shape to fill, as lengths separated by commas: `1` is one number, `4,64` a grid."),
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Generator],
    doc: "A number, given a shape: the way a scalar becomes an array of the size a node downstream wants.",
    inputs: &[],
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Constant, MANIFEST);
