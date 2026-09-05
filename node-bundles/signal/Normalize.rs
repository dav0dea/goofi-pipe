//! Normalize — put a signal on a common scale, from statistics taken over a window of its own
//! past. A window of zero keeps running statistics instead, which costs no memory as it grows.

use goofi_core::{resolve_axis, stream, Data, SlotType, Stream};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, SlotDecl, Tag};

/// What a lane is scaled by: a centre and a spread, whatever the mode calls them.
#[derive(Clone, Copy, Default)]
struct Scale {
    centre: f64,
    spread: f64,
}

/// Running count, mean and sum of squared deviations, and the extremes — one per lane.
#[derive(Clone, Copy, Default)]
struct Running {
    n: f64,
    mean: f64,
    m2: f64,
    low: f64,
    high: f64,
}

impl Running {
    fn push(&mut self, x: f64) {
        if self.n == 0.0 {
            (self.low, self.high) = (x, x);
        }
        self.n += 1.0;
        let delta = x - self.mean;
        self.mean += delta / self.n;
        self.m2 += delta * (x - self.mean);
        self.low = self.low.min(x);
        self.high = self.high.max(x);
    }

    fn scale(&self, mode: &str) -> Scale {
        match mode {
            "minmax" => Scale { centre: self.low, spread: self.high - self.low },
            _ => Scale { centre: self.mean, spread: (self.m2 / self.n.max(1.0)).sqrt() },
        }
    }
}

fn median(sorted: &[f64], at: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = at * (sorted.len() - 1) as f64;
    let (low, frac) = (pos.floor() as usize, pos.fract());
    sorted[low] + (sorted[(low + 1).min(sorted.len() - 1)] - sorted[low]) * frac
}

/// The statistics of one window, by mode.
fn scale_of(mode: &str, window: &[f32]) -> Scale {
    match mode {
        "minmax" => {
            let low = window.iter().copied().fold(f32::INFINITY, f32::min) as f64;
            let high = window.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            Scale { centre: low, spread: high - low }
        }
        "robust" => {
            let mut v: Vec<f64> = window.iter().map(|x| *x as f64).collect();
            v.sort_by(f64::total_cmp);
            Scale { centre: median(&v, 0.5), spread: median(&v, 0.75) - median(&v, 0.25) }
        }
        _ => {
            let n = window.len().max(1) as f64;
            let mean = window.iter().map(|x| *x as f64).sum::<f64>() / n;
            let var = window.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
            Scale { centre: mean, spread: var.sqrt() }
        }
    }
}

/// A spread of zero means the window does not vary, and every value sits at the centre.
fn apply(scale: Scale, x: f32) -> f32 {
    if scale.spread == 0.0 {
        0.0
    } else {
        ((x as f64 - scale.centre) / scale.spread) as f32
    }
}

#[derive(Default)]
struct Normalize {
    past: Stream,
    running: Vec<Running>,
    /// The statistics `hold` froze, kept so the freeze survives the next frame.
    held: Option<Vec<Scale>>,
}

impl Node for Normalize {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let mode = p.str("normalize", "mode").unwrap_or("zscore");
        let dim = resolve_axis(p.i64("window", "axis").unwrap_or(-1), a.shape().len())?;
        let size = p.f64("window", "size").unwrap_or(1000.0).max(0.0);
        let unit = p.str("window", "unit").unwrap_or("samples");
        let hold = p.bool("window", "hold").unwrap_or(false);
        let n = a.shape()[dim];

        let width = match unit {
            "seconds" => {
                let rate = d.meta().sfreq().ok_or("`seconds` needs a frame that carries its sample rate")?;
                (size * rate).round() as usize
            }
            _ => size.round() as usize,
        };

        let (shape, stitched, at) = if width == 0 {
            // Running statistics: the node keeps no window at all, only the four numbers.
            (a.shape().to_vec(), a.as_bytes().to_vec(), 0usize)
        } else {
            self.past.push(a.shape(), dim, a.as_bytes(), width + n)
        };
        let lanes = stream::lanes(&shape, dim, &stitched);

        let scales: Vec<Scale> = match &self.held {
            Some(held) if held.len() == lanes.len() => held.clone(),
            _ => {
                if width == 0 {
                    if self.running.len() != lanes.len() {
                        self.running = vec![Running::default(); lanes.len()];
                    }
                    for (lane, run) in lanes.iter().zip(&mut self.running) {
                        for x in lane {
                            run.push(*x as f64);
                        }
                    }
                    self.running.iter().map(|r| r.scale(mode)).collect()
                } else {
                    lanes
                        .iter()
                        .map(|lane| scale_of(mode, &lane[lane.len().saturating_sub(width)..]))
                        .collect()
                }
            }
        };
        self.held = hold.then(|| scales.clone());

        let scaled: Vec<Vec<f32>> = lanes
            .iter()
            .zip(&scales)
            .map(|(lane, scale)| lane[at..at + n].iter().map(|x| apply(*scale, *x)).collect())
            .collect();
        let buf = stream::unlanes(a.shape(), dim, &scaled);
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn on_pulse(&mut self, _key: &ParamKey, _p: &Params<'_>) -> NodeResult {
        self.past.reset();
        self.running.clear();
        self.held = None;
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "normalize",
        name: "mode",
        spec: ParamSpec::Str { default: "zscore", options: &["zscore", "minmax", "robust"], refresh: false },
        expression: None,
        doc: Some(
            "`zscore` measures in standard deviations from the mean, `minmax` maps the window onto \
             0 to 1, and `robust` uses the median and the middle half, which an outlier cannot move.",
        ),
    },
    ParamDecl {
        group: "window",
        name: "size",
        spec: ParamSpec::Float { default: 1000.0, min: 0.0, max: 1.0e7 },
        expression: None,
        doc: Some(
            "How much of the past the statistics are taken over. 0 keeps running statistics over \
             everything the node has seen, which costs no memory as it grows.",
        ),
    },
    ParamDecl {
        group: "window",
        name: "unit",
        spec: ParamSpec::Str { default: "samples", options: &["samples", "seconds"], refresh: false },
        expression: None,
        doc: Some("What `size` counts."),
    },
    ParamDecl {
        group: "window",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis the statistics are taken along. -1 is time."),
    },
    ParamDecl {
        group: "window",
        name: "hold",
        spec: ParamSpec::Bool { default: false },
        expression: None,
        doc: Some("Freeze the statistics where they are, so later data is measured against them."),
    },
    ParamDecl {
        group: "window",
        name: "reset",
        spec: ParamSpec::Pulse,
        expression: None,
        doc: Some("Forget the past and any frozen statistics."),
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
    doc: "Put a signal on a common scale, from statistics over a window of its own past.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Normalize, MANIFEST);
