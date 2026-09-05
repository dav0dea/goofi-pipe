//! Emd — empirical mode decomposition: a frame is pulled apart into the oscillations it is made
//! of, fastest first, with whatever is left over dropped. Per frame, so it follows a Buffer.

use goofi_core::{resolve_axis, stream, Axis, Coord, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Emd;

/// The indices where `v` turns, with the ends carried along so a spline spans the whole run.
fn turns(v: &[f32], up: bool) -> Vec<usize> {
    let mut at = vec![0usize];
    for i in 1..v.len().saturating_sub(1) {
        let peak = v[i] > v[i - 1] && v[i] >= v[i + 1];
        let dip = v[i] < v[i - 1] && v[i] <= v[i + 1];
        if if up { peak } else { dip } {
            at.push(i);
        }
    }
    at.push(v.len() - 1);
    at.dedup();
    at
}

/// A natural cubic spline through `(at, v[at])`, read back at every index.
fn spline(v: &[f32], at: &[usize]) -> Vec<f32> {
    let n = at.len();
    if n < 3 {
        // Two knots or fewer cannot bend: the straight line through them is the envelope.
        let (first, last) = (v[at[0]], v[at[n - 1]]);
        let span = (at[n - 1] - at[0]).max(1) as f32;
        return (0..v.len()).map(|i| first + (last - first) * (i as f32 - at[0] as f32) / span).collect();
    }
    let x: Vec<f64> = at.iter().map(|i| *i as f64).collect();
    let y: Vec<f64> = at.iter().map(|i| v[*i] as f64).collect();
    let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    // The tridiagonal solve for the second derivatives, with both ends left free.
    let (mut lower, mut diag, mut rhs) = (vec![0.0; n], vec![1.0; n], vec![0.0; n]);
    for i in 1..n - 1 {
        lower[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }
    let mut upper = vec![0.0; n];
    for i in 1..n - 1 {
        upper[i] = h[i];
    }
    for i in 1..n {
        let m = if diag[i - 1] == 0.0 { 0.0 } else { lower[i] / diag[i - 1] };
        diag[i] -= m * upper[i - 1];
        rhs[i] -= m * rhs[i - 1];
    }
    let mut second = vec![0.0; n];
    for i in (0..n).rev() {
        let ahead = if i + 1 < n { upper[i] * second[i + 1] } else { 0.0 };
        second[i] = if diag[i] == 0.0 { 0.0 } else { (rhs[i] - ahead) / diag[i] };
    }
    let mut out = Vec::with_capacity(v.len());
    let mut seg = 0;
    for i in 0..v.len() {
        while seg + 2 < n && (i as f64) > x[seg + 1] {
            seg += 1;
        }
        let (a, b) = (x[seg], x[seg + 1]);
        let step = (b - a).max(1e-9);
        let (l, r) = ((b - i as f64) / step, (i as f64 - a) / step);
        let value = l * y[seg]
            + r * y[seg + 1]
            + ((l * l * l - l) * second[seg] + (r * r * r - r) * second[seg + 1]) * step * step / 6.0;
        out.push(value as f32);
    }
    out
}

/// One sifting pass: the lane minus the mean of its two envelopes.
fn sift(lane: &[f32]) -> Vec<f32> {
    let (high, low) = (spline(lane, &turns(lane, true)), spline(lane, &turns(lane, false)));
    lane.iter().zip(high.iter().zip(&low)).map(|(x, (h, l))| x - (h + l) / 2.0).collect()
}

impl Node for Emd {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let dim = resolve_axis(p.i64("emd", "axis").unwrap_or(-1), a.shape().len())?;
        let n = a.shape()[dim];
        if n < 8 {
            return Err(format!("needs at least 8 samples along the axis, got {n}").into());
        }
        let count = p.i64("emd", "count").unwrap_or(5).clamp(1, 10) as usize;
        let rounds = p.i64("emd", "siftings").unwrap_or(10).clamp(1, 100) as usize;

        // One lane in becomes `count` lanes out, laid consecutively so the new axis sits before
        // time in the output's own order.
        let mut modes: Vec<Vec<f32>> = Vec::new();
        for lane in stream::lanes(a.shape(), dim, a.as_bytes()) {
            let mut rest = lane;
            for _ in 0..count {
                let mut mode = rest.clone();
                for _ in 0..rounds {
                    mode = sift(&mode);
                }
                rest = rest.iter().zip(&mode).map(|(x, m)| x - m).collect();
                modes.push(mode);
            }
        }

        let mut shape_out = a.shape().to_vec();
        shape_out.insert(dim, count);
        let buf = stream::unlanes(&shape_out, dim + 1, &modes);
        let names: Vec<Coord> = (1..=count).map(|i| Coord::Str(format!("IMF{i}").into())).collect();
        let meta = d.meta().insert_axis(dim, Axis::coords(names), a.shape().len());
        out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "emd",
        name: "count",
        spec: ParamSpec::Int { default: 5, min: 1, max: 10 },
        expression: None,
        doc: Some("How many oscillations to pull out, fastest first. What is left over is dropped."),
    },
    ParamDecl {
        group: "emd",
        name: "siftings",
        spec: ParamSpec::Int { default: 10, min: 1, max: 100 },
        expression: None,
        doc: Some("How many passes each oscillation is refined by. More is cleaner and slower."),
    },
    ParamDecl {
        group: "emd",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis holds the samples. -1 is time."),
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
    doc: "Pull a frame apart into the oscillations it is made of, fastest first.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Emd, MANIFEST);
