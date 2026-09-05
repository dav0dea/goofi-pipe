//! Function — one elementwise function, chosen by name. Shape and meta are untouched.

use goofi_core::{Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Function;

/// Every name in the manifest's option list, and nothing else reaches here.
fn apply(name: &str, x: f32) -> f32 {
    match name {
        "negate" => -x,
        "square" => x * x,
        "sqrt" => x.sqrt(),
        "log" => x.ln(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "exp" => x.exp(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "tanh" => x.tanh(),
        "sign" => {
            if x == 0.0 {
                0.0
            } else {
                x.signum()
            }
        }
        "round" => x.round(),
        "floor" => x.floor(),
        "ceil" => x.ceil(),
        _ => x.abs(),
    }
}

impl Node for Function {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.as_array()?;
        let name = p.str("function", "function").unwrap_or("abs");
        let mut buf = Vec::with_capacity(a.as_bytes().len());
        for x in a.as_bytes().chunks_exact(4) {
            let v = f32::from_le_bytes(x.try_into().expect("four bytes"));
            buf.extend_from_slice(&apply(name, v).to_le_bytes());
        }
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "function",
    name: "function",
    spec: ParamSpec::Str {
        default: "abs",
        options: &[
            "abs", "negate", "square", "sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan",
            "asin", "acos", "atan", "tanh", "sign", "round", "floor", "ceil",
        ],
        refresh: false,
    },
    expression: None,
    doc: Some(
        "Which function to apply to every value. A value outside a function's domain, such as a \
         negative under `sqrt`, comes out as not-a-number rather than as an error.",
    ),
}];
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
    doc: "One elementwise function over the frame, chosen by name. Shape and metadata are untouched.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Function, MANIFEST);
