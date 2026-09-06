//! Reduce — one axis becomes one number per position on the others, so the axis and its labels go.

use goofi_core::{resolve_axis, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Reduce;

/// `v` is one axis' worth of values, gathered in order.
fn reduce(mode: &str, v: &mut [f32]) -> f32 {
    let n = v.len() as f32;
    let sum: f32 = v.iter().sum();
    match mode {
        "sum" => sum,
        "min" => v.iter().copied().fold(f32::INFINITY, f32::min),
        "max" => v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        "norm" => v.iter().map(|x| x * x).sum::<f32>().sqrt(),
        "std" => {
            let mean = sum / n;
            (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt()
        }
        "median" => {
            v.sort_by(f32::total_cmp);
            let mid = v.len() / 2;
            if v.len() % 2 == 0 {
                (v[mid - 1] + v[mid]) / 2.0
            } else {
                v[mid]
            }
        }
        _ => sum / n,
    }
}

impl Node for Reduce {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let shape = a.shape();
        let axis = resolve_axis(p.i64("reduce", "axis").unwrap_or(-1), shape.len())?;
        let mode = p.str("reduce", "mode").unwrap_or("mean");
        let n = shape[axis];
        if n == 0 {
            return Err("cannot reduce an axis of no length".to_string().into());
        }
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();

        let src = a.as_bytes();
        let at = |o: usize, k: usize, i: usize| ((o * n + k) * inner + i) * 4;
        let mut lane = vec![0f32; n];
        let mut buf = Vec::with_capacity(outer * inner * 4);
        for o in 0..outer {
            for i in 0..inner {
                for (k, slot) in lane.iter_mut().enumerate() {
                    *slot = f32::from_le_bytes(src[at(o, k, i)..at(o, k, i) + 4].try_into().expect("four bytes"));
                }
                buf.extend_from_slice(&reduce(mode, &mut lane).to_le_bytes());
            }
        }

        let mut shape_out = shape.to_vec();
        shape_out.remove(axis);
        let meta = d.meta().drop_axis(axis, shape.len());
        out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "reduce",
        name: "mode",
        spec: ParamSpec::Str {
            default: "mean",
            options: &["mean", "median", "min", "max", "std", "sum", "norm"],
            refresh: false,
        },
        expression: None,
        doc: Some("How the values along the axis become one: an average, a spread, an extreme or a total."),
    },
    ParamDecl {
        group: "reduce",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis to collapse, negative from the end. -1 is time, -2 is channels."),
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
    doc: "Collapse one axis to a single value.\n\
          One value per position on the other axes: a mean, a spread, an extreme or a total.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Reduce, MANIFEST);
