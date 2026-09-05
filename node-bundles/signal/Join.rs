//! Join — several frames become one, laid end to end along an axis or stacked onto a new one.
//! A stacked axis takes the sender names, which is the only place a builtin may name an entry.

use goofi_core::{resolve_axis, Axis, Coord, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Join;

/// Copy `wires` block by block along `axis`, so the entries of one frame stay together.
fn interleave(shape: &[usize], axis: usize, lens: &[usize], wires: &[&[u8]]) -> Vec<u8> {
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let mut buf = Vec::with_capacity(wires.iter().map(|w| w.len()).sum());
    for o in 0..outer {
        for (w, &len) in wires.iter().zip(lens) {
            let block = len * inner * 4;
            buf.extend_from_slice(&w[o * block..(o + 1) * block]);
        }
    }
    buf
}

impl Node for Join {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let wires = inp.get_multi("input");
        if wires.is_empty() {
            return Err("`input` needs at least one wire".to_string().into());
        }
        let stack = p.str("join", "mode").unwrap_or("concatenate") == "stack";
        let first = &wires[0].1;
        let base = first.assert_ndims().at_least(1)?;
        let ndim = base.shape().len();
        let axis = resolve_axis(p.i64("join", "axis").unwrap_or(0), if stack { ndim + 1 } else { ndim })?;

        let mut arrays = Vec::with_capacity(wires.len());
        for (source, d) in wires {
            let a = d.as_array()?;
            let ok = if stack {
                a.shape() == base.shape()
            } else {
                a.shape().len() == ndim
                    && a.shape().iter().enumerate().all(|(at, n)| at == axis || *n == base.shape()[at])
            };
            if !ok {
                return Err(format!(
                    "`{source}` is {:?} where the first wire is {:?}",
                    a.shape(),
                    base.shape()
                )
                .into());
            }
            arrays.push(a);
        }

        let metas: Vec<&goofi_core::Meta> = wires[1..].iter().map(|(_, d)| d.meta()).collect();
        if stack {
            // A stacked axis is born here, so its entries have names only if the senders do.
            let coords: Vec<Coord> = wires.iter().map(|(source, _)| Coord::Str(source.as_str().into())).collect();
            let mut shape_out = base.shape().to_vec();
            shape_out.insert(axis, wires.len());
            let lens = vec![1usize; wires.len()];
            let mut stacked = base.shape().to_vec();
            stacked.insert(axis, 1);
            let bytes: Vec<&[u8]> = arrays.iter().map(|a| a.as_bytes()).collect();
            let buf = interleave(&stacked, axis, &lens, &bytes);
            let meta = first.meta().insert_axis(axis, Axis::coords(coords), ndim);
            out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
            return Ok(());
        }

        let lens: Vec<usize> = arrays.iter().map(|a| a.shape()[axis]).collect();
        let mut shape_out = base.shape().to_vec();
        shape_out[axis] = lens.iter().sum();
        let bytes: Vec<&[u8]> = arrays.iter().map(|a| a.as_bytes()).collect();
        let buf = interleave(base.shape(), axis, &lens, &bytes);
        let meta = first.meta().concat(&metas, axis);
        out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "join",
        name: "mode",
        spec: ParamSpec::Str { default: "concatenate", options: &["concatenate", "stack"], refresh: false },
        expression: None,
        doc: Some(
            "`concatenate` lays the frames end to end along an axis they already share; `stack` \
             puts them side by side on a new axis, named after the nodes they came from.",
        ),
    },
    ParamDecl {
        group: "join",
        name: "axis",
        spec: ParamSpec::Int { default: 0, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis to join along, or where the new axis goes when stacking."),
    },
];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Array,
    trigger_process: true,
    multi: true,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "Lay several frames end to end along one axis, or stack them onto a new axis named after their senders.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Join, MANIFEST);
