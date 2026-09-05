//! Operation — the arithmetic, the matrix product and the correlations that take more than one
//! frame. Arithmetic folds left in wire order and broadcasts the way numpy does.

use goofi_core::{resolve_axis, Axis, Coord, Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Operation;

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().expect("four bytes"))).collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The shape two frames combine into: aligned from the right, a length of one stretches.
fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for at in 0..rank {
        let (x, y) = (dim(a, rank, at), dim(b, rank, at));
        out[at] = match (x, y) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(format!("shapes {a:?} and {b:?} do not fit together")),
        };
    }
    Ok(out)
}

/// Dimension `at` of `shape` seen at rank `rank`, right-aligned; a missing leading axis is 1.
fn dim(shape: &[usize], rank: usize, at: usize) -> usize {
    let offset = rank - shape.len();
    if at < offset {
        1
    } else {
        shape[at - offset]
    }
}

/// Strides into `shape` for a walk over `out`, where a stretched axis does not advance.
fn strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let mut s = vec![0usize; out.len()];
    let offset = out.len() - shape.len();
    let mut acc = 1;
    for at in (0..shape.len()).rev() {
        s[offset + at] = if shape[at] == 1 { 0 } else { acc };
        acc *= shape[at];
    }
    s
}

fn combine(mode: &str, a: f32, b: f32) -> f32 {
    match mode {
        "subtract" => a - b,
        "multiply" => a * b,
        "divide" => a / b,
        "min" => a.min(b),
        "max" => a.max(b),
        _ => a + b,
    }
}

/// One elementwise fold of two frames, broadcast to the shape they share.
fn elementwise(mode: &str, x: (&[usize], &[f32]), y: (&[usize], &[f32])) -> Result<(Vec<usize>, Vec<f32>), String> {
    let out = broadcast(x.0, y.0)?;
    let (sx, sy) = (strides(x.0, &out), strides(y.0, &out));
    let count: usize = out.iter().product();
    let mut v = Vec::with_capacity(count);
    let mut index = vec![0usize; out.len()];
    for _ in 0..count {
        let ax: usize = index.iter().zip(&sx).map(|(i, s)| i * s).sum();
        let ay: usize = index.iter().zip(&sy).map(|(i, s)| i * s).sum();
        v.push(combine(mode, x.1[ax], y.1[ay]));
        for at in (0..index.len()).rev() {
            index[at] += 1;
            if index[at] < out[at] {
                break;
            }
            index[at] = 0;
        }
    }
    Ok((out, v))
}

/// `[.., n, k]` times `[.., k, m]`. The leading axes must match: a batch that broadcasts is not
/// something a signal patch asks for, and guessing it wrong is worse than saying so.
fn matmul(x: (&[usize], &[f32]), y: (&[usize], &[f32])) -> Result<(Vec<usize>, Vec<f32>), String> {
    let (xs, ys) = (x.0, y.0);
    if xs.len() < 2 || ys.len() < 2 {
        return Err(format!("matmul needs two axes on each side, got {xs:?} and {ys:?}"));
    }
    let (n, k) = (xs[xs.len() - 2], xs[xs.len() - 1]);
    let (k2, m) = (ys[ys.len() - 2], ys[ys.len() - 1]);
    if k != k2 || xs[..xs.len() - 2] != ys[..ys.len() - 2] {
        return Err(format!("matmul cannot fit {xs:?} against {ys:?}"));
    }
    let batch: usize = xs[..xs.len() - 2].iter().product();
    let mut v = vec![0f32; batch * n * m];
    for b in 0..batch {
        for i in 0..n {
            for j in 0..m {
                let mut acc = 0f32;
                for t in 0..k {
                    acc += x.1[(b * n + i) * k + t] * y.1[(b * k + t) * m + j];
                }
                v[(b * n + i) * m + j] = acc;
            }
        }
    }
    let mut out = xs[..xs.len() - 2].to_vec();
    out.extend_from_slice(&[n, m]);
    Ok((out, v))
}

/// Gather the lane at `(outer, inner)` along the collapsed axis.
fn lane(v: &[f32], o: usize, i: usize, n: usize, inner: usize) -> Vec<f32> {
    (0..n).map(|k| v[(o * n + k) * inner + i]).collect()
}

fn centred(v: &[f32]) -> (Vec<f32>, f32) {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let c: Vec<f32> = v.iter().map(|x| x - mean).collect();
    let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
    (c, norm)
}

/// Pearson r between two lanes; a lane that does not vary correlates with nothing.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let ((ca, na), (cb, nb)) = (centred(a), centred(b));
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    ca.iter().zip(&cb).map(|(x, y)| x * y).sum::<f32>() / (na * nb)
}

/// The correlation of `a` with `b` at every offset in `offsets`, each normalised as Pearson is.
fn lagged(a: &[f32], b: &[f32], offsets: &[i64]) -> Vec<f32> {
    let ((ca, na), (cb, nb)) = (centred(a), centred(b));
    let scale = if na == 0.0 || nb == 0.0 { 0.0 } else { 1.0 / (na * nb) };
    offsets
        .iter()
        .map(|&d| {
            let mut acc = 0f32;
            for (i, x) in ca.iter().enumerate() {
                let j = i as i64 + d;
                if j >= 0 && (j as usize) < cb.len() {
                    acc += x * cb[j as usize];
                }
            }
            acc * scale
        })
        .collect()
}

impl Node for Operation {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let wires = inp.get_multi("input");
        let (_, first) = wires.first().ok_or("`input` needs at least one wire")?;
        let base = first.as_array()?;
        let mode = p.str("operation", "mode").unwrap_or("add");

        let frames: Result<Vec<(Vec<usize>, Vec<f32>)>, String> =
            wires.iter().map(|(_, d)| d.as_array().map(|a| (a.shape().to_vec(), floats(a.as_bytes())))).collect();
        let frames = frames?;

        let (shape, values) = match mode {
            "correlation" | "crosscorrelation" | "autocorrelation" => {
                let axis = resolve_axis(p.i64("operation", "axis").unwrap_or(-1), base.shape().len())?;
                let n = base.shape()[axis];
                let outer: usize = base.shape()[..axis].iter().product();
                let inner: usize = base.shape()[axis + 1..].iter().product();
                let others: Vec<&(Vec<usize>, Vec<f32>)> = if mode == "autocorrelation" {
                    vec![&frames[0]]
                } else {
                    frames[1..].iter().collect()
                };
                if others.is_empty() {
                    return Err(format!("`{mode}` needs a second wire").into());
                }
                for (s, _) in &others {
                    if *s != base.shape() {
                        return Err(format!("`{mode}` needs matching shapes, got {s:?} and {:?}", base.shape()).into());
                    }
                }
                let offsets: Vec<i64> = if mode == "correlation" {
                    vec![0]
                } else {
                    let asked = p.i64("operation", "lags").unwrap_or(0).unsigned_abs() as usize;
                    let reach = if asked == 0 { n.saturating_sub(1) } else { asked.min(n - 1) };
                    (-(reach as i64)..=reach as i64).collect()
                };
                let mut v = Vec::with_capacity(others.len() * outer * inner * offsets.len());
                for (_, other) in &others {
                    for o in 0..outer {
                        for i in 0..inner {
                            let a = lane(&frames[0].1, o, i, n, inner);
                            let b = lane(other, o, i, n, inner);
                            if mode == "correlation" {
                                v.push(pearson(&a, &b));
                            } else {
                                v.extend(lagged(&a, &b, &offsets));
                            }
                        }
                    }
                }
                let mut shape = base.shape().to_vec();
                if mode == "correlation" {
                    shape.remove(axis);
                } else {
                    shape[axis] = offsets.len();
                }
                if others.len() > 1 {
                    shape.insert(0, others.len());
                }
                let meta = correlation_meta(first.meta(), mode, axis, base.shape().len(), &offsets, &others, wires);
                out.set("out", Data::array_f32(shape, bytes(&v), meta).map_err(|e| e.to_string())?);
                return Ok(());
            }
            "mean" => {
                let mut acc = frames[0].clone();
                for f in &frames[1..] {
                    acc = elementwise("add", (&acc.0, &acc.1), (&f.0, &f.1))?;
                }
                let n = frames.len() as f32;
                (acc.0, acc.1.iter().map(|x| x / n).collect())
            }
            "matmul" => {
                let mut acc = frames[0].clone();
                for f in &frames[1..] {
                    acc = matmul((&acc.0, &acc.1), (&f.0, &f.1))?;
                }
                acc
            }
            _ => {
                let mut acc = frames[0].clone();
                for f in &frames[1..] {
                    acc = elementwise(mode, (&acc.0, &acc.1), (&f.0, &f.1))?;
                }
                acc
            }
        };

        // The first wire's meta, keeping a label only where its axis still has the same length.
        let mut meta = first.meta().clone();
        let mut axes = goofi_core::Axes::new();
        for (at, n) in shape.iter().enumerate() {
            if let Some(axis) = meta.channels().get(at) {
                if axis.coords.as_ref().is_some_and(|c| c.len() == *n) {
                    axes = axes.with(at, axis.clone());
                }
            }
        }
        meta.set_channels(axes);
        out.set("out", Data::array_f32(shape, bytes(&values), meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

/// The meta a correlation leaves: the collapsed axis gone or relabelled with its lag offsets, and
/// the pair axis named after the wires when there is more than one pair.
fn correlation_meta(
    base: &Meta,
    mode: &str,
    axis: usize,
    ndim: usize,
    offsets: &[i64],
    others: &[&(Vec<usize>, Vec<f32>)],
    wires: &[(String, Data)],
) -> Meta {
    let mut meta = if mode == "correlation" {
        base.drop_axis(axis, ndim)
    } else {
        let coords: Vec<Coord> = offsets.iter().map(|&d| Coord::Num(d as f64)).collect();
        base.clone().with_channels(base.channels().clone().with(axis, Axis::coords(coords))).with_sfreq(None)
    };
    if others.len() > 1 {
        let names: Vec<Coord> = wires[1..].iter().map(|(source, _)| Coord::Str(source.as_str().into())).collect();
        let rank = if mode == "correlation" { ndim - 1 } else { ndim };
        meta = meta.insert_axis(0, Axis::coords(names), rank);
    }
    meta
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "operation",
        name: "mode",
        spec: ParamSpec::Str {
            default: "add",
            options: &[
                "add", "subtract", "multiply", "divide", "min", "max", "mean", "matmul",
                "correlation", "crosscorrelation", "autocorrelation",
            ],
            refresh: false,
        },
        expression: None,
        doc: Some(
            "What to do with the wires. The arithmetic folds them left in wire order and stretches \
             a length of one to fit; the correlations measure the first wire against each other one.",
        ),
    },
    ParamDecl {
        group: "operation",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis the correlations run along; the other modes ignore it. -1 is time."),
    },
    ParamDecl {
        group: "operation",
        name: "lags",
        spec: ParamSpec::Int { default: 0, min: 0, max: 100_000 },
        expression: None,
        doc: Some("How far either way a lagged correlation reaches, in samples; 0 reaches as far as it can."),
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
    doc: "Combine several frames: arithmetic folded left with numpy broadcasting, a matrix product, or a correlation.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Operation, MANIFEST);
