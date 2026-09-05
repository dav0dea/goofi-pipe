//! Reshape — reorder the axes, then re-cut the frame into a new shape. A reorder carries the
//! labels with it; a re-cut cannot, because the entries it makes are not the entries it was given.

use goofi_core::{Axes, Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Reshape;

/// `"1,0"` as a permutation of `ndim` axes; empty keeps the order.
fn permutation(text: &str, ndim: usize) -> Result<Vec<usize>, String> {
    let parts: Vec<&str> = text.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if parts.is_empty() {
        return Ok((0..ndim).collect());
    }
    if parts.len() != ndim {
        return Err(format!("`axes` names {} axes for a frame of {ndim}", parts.len()));
    }
    let mut perm = Vec::with_capacity(ndim);
    for part in parts {
        let at: usize = part.parse().map_err(|_| format!("`axes` takes axis numbers, not `{part}`"))?;
        if at >= ndim || perm.contains(&at) {
            return Err(format!("`axes` names axis {at} out of range or twice"));
        }
        perm.push(at);
    }
    Ok(perm)
}

/// `"4,-1"` as lengths for `count` entries; empty keeps the shape.
fn lengths(text: &str, count: usize, current: &[usize]) -> Result<Vec<usize>, String> {
    let parts: Vec<&str> = text.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if parts.is_empty() {
        return Ok(current.to_vec());
    }
    let mut shape = Vec::with_capacity(parts.len());
    let mut free = None;
    for (at, part) in parts.iter().enumerate() {
        let n: i64 = part.parse().map_err(|_| format!("`shape` takes whole numbers, not `{part}`"))?;
        match n {
            -1 if free.is_none() => {
                free = Some(at);
                shape.push(1);
            }
            -1 => return Err("`shape` takes one -1 at most".to_string()),
            n if n > 0 => shape.push(n as usize),
            _ => return Err(format!("`shape` takes lengths of one or more, not `{n}`")),
        }
    }
    let known: usize = shape.iter().product();
    if let Some(at) = free {
        if known == 0 || count % known != 0 {
            return Err(format!("{count} entries do not divide into the shape given"));
        }
        shape[at] = count / known;
    }
    if shape.iter().product::<usize>() != count {
        return Err(format!("`shape` asks for {} entries, and the frame has {count}", shape.iter().product::<usize>()));
    }
    Ok(shape)
}

impl Node for Reshape {
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
        let perm = permutation(p.str("reshape", "axes").unwrap_or(""), shape.len())?;

        let moved: Vec<usize> = perm.iter().map(|&at| shape[at]).collect();
        let src = a.as_bytes();
        let mut buf = vec![0u8; src.len()];
        if perm.iter().enumerate().any(|(at, &from)| at != from) {
            // Strides of the SOURCE, read in the permuted order, so one walk fills the output.
            let mut stride = vec![1usize; shape.len()];
            for at in (0..shape.len().saturating_sub(1)).rev() {
                stride[at] = stride[at + 1] * shape[at + 1];
            }
            let count: usize = shape.iter().product();
            let mut index = vec![0usize; moved.len()];
            for at in 0..count {
                let from: usize = index.iter().enumerate().map(|(k, &i)| i * stride[perm[k]]).sum();
                buf[at * 4..at * 4 + 4].copy_from_slice(&src[from * 4..from * 4 + 4]);
                for k in (0..index.len()).rev() {
                    index[k] += 1;
                    if index[k] < moved[k] {
                        break;
                    }
                    index[k] = 0;
                }
            }
        } else {
            buf.copy_from_slice(src);
        }

        let mut meta = d.meta().clone();
        let mut axes = Axes::new();
        for (at, &from) in perm.iter().enumerate() {
            if let Some(axis) = meta.channels().get(from) {
                axes = axes.with(at, axis.clone());
            }
        }
        meta.set_channels(axes);

        let count: usize = moved.iter().product();
        let cut = lengths(p.str("reshape", "shape").unwrap_or(""), count, &moved)?;
        if cut != moved {
            // A re-cut makes entries the input never had, so no label and no rate survives it.
            meta = Meta::new();
        }
        out.set("out", Data::array_f32(cut, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "reshape",
        name: "axes",
        spec: ParamSpec::Str { default: "", options: &[], refresh: false },
        expression: None,
        doc: Some(
            "The new order of the axes, as their old numbers separated by commas: `1,0` swaps the \
             two axes of a grid. Empty keeps the order.",
        ),
    },
    ParamDecl {
        group: "reshape",
        name: "shape",
        spec: ParamSpec::Str { default: "", options: &[], refresh: false },
        expression: None,
        doc: Some(
            "The shape to re-cut the entries into, separated by commas, with one -1 for the length \
             to work out. Empty keeps the shape.",
        ),
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
    doc: "Reorder the axes of a frame, then re-cut the entries into a new shape.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Reshape, MANIFEST);
