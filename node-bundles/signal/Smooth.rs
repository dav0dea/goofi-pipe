//! Smooth — a running mean or an exponential decay over the stitched past, so the answer is the
//! same whatever size the frames arrive in. It holds no filter state, only the input's own past.

use goofi_core::{resolve_axis, stream, Data, SlotType, Stream};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Smooth {
    past: Stream,
    /// Whole frames, for the unit that counts updates rather than samples.
    frames: std::collections::VecDeque<Vec<f32>>,
}

/// The mean of the `width` values ending at each position, the head clamped to what exists.
fn running_mean(lane: &[f32], width: usize) -> Vec<f32> {
    let mut sum = 0.0f64;
    let mut out = Vec::with_capacity(lane.len());
    for (i, v) in lane.iter().enumerate() {
        sum += *v as f64;
        if i >= width {
            sum -= lane[i - width] as f64;
        }
        out.push((sum / (i + 1).min(width) as f64) as f32);
    }
    out
}

/// A one-pole decay run forward over the whole lane; `tau` is in samples.
fn exponential(lane: &[f32], tau: f64) -> Vec<f32> {
    let alpha = if tau <= 0.0 { 1.0 } else { 1.0 - (-1.0 / tau).exp() };
    let mut y = lane.first().copied().unwrap_or(0.0) as f64;
    lane.iter()
        .map(|v| {
            y += alpha * (*v as f64 - y);
            y as f32
        })
        .collect()
}

/// `size` in the given unit as a count of samples along the axis.
fn width(size: f64, unit: &str, sfreq: Option<f64>) -> Result<usize, String> {
    match unit {
        "seconds" => {
            let rate = sfreq.ok_or("`seconds` needs a frame that carries its sample rate")?;
            Ok((size * rate).round().max(1.0) as usize)
        }
        _ => Ok(size.round().max(1.0) as usize),
    }
}

impl Node for Smooth {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let size = p.f64("smooth", "size").unwrap_or(10.0).max(0.0);
        let unit = p.str("smooth", "unit").unwrap_or("samples");
        let exp = p.str("smooth", "mode").unwrap_or("average") == "exponential";

        if unit == "updates" {
            // The stream here is the SEQUENCE of frames, so one value per position is smoothed
            // across updates rather than along an axis.
            let values: Vec<f32> = a.as_bytes().chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().expect("four bytes"))).collect();
            let keep = size.round().max(1.0) as usize;
            if self.frames.front().is_some_and(|f| f.len() != values.len()) {
                self.frames.clear();
            }
            self.frames.push_back(values);
            while self.frames.len() > keep {
                self.frames.pop_front();
            }
            let n = self.frames.len() as f32;
            let mut buf = Vec::with_capacity(a.as_bytes().len());
            for i in 0..self.frames[0].len() {
                let sum: f32 = self.frames.iter().map(|f| f[i]).sum();
                buf.extend_from_slice(&(sum / n).to_le_bytes());
            }
            out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
            return Ok(());
        }

        let dim = resolve_axis(p.i64("smooth", "axis").unwrap_or(-1), a.shape().len())?;
        let w = width(size, unit, d.meta().sfreq())?;
        // An exponential needs several time constants of past before its answer settles.
        let reach = if exp { w * 5 } else { w };
        let n = a.shape()[dim];
        let (shape, stitched, at) = self.past.push(a.shape(), dim, a.as_bytes(), reach + n);
        let smoothed: Vec<Vec<f32>> = stream::lanes(&shape, dim, &stitched)
            .iter()
            .map(|lane| {
                let full = if exp { exponential(lane, w as f64) } else { running_mean(lane, w) };
                full[at..at + n].to_vec()
            })
            .collect();
        let buf = stream::unlanes(a.shape(), dim, &smoothed);
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn on_pulse(&mut self, _key: &ParamKey, _p: &Params<'_>) -> NodeResult {
        self.past.reset();
        self.frames.clear();
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "smooth",
        name: "mode",
        spec: ParamSpec::Str { default: "average", options: &["average", "exponential"], refresh: false },
        expression: None,
        doc: Some(
            "`average` takes the mean of the last `size` values; `exponential` lets an old value \
             fade away, with `size` as the time it takes to fade.",
        ),
    },
    ParamDecl {
        group: "smooth",
        name: "size",
        spec: ParamSpec::Float { default: 10.0, min: 0.0, max: 1.0e7 },
        expression: None,
        doc: Some("How much of the past to smooth over, in the unit below."),
    },
    ParamDecl {
        group: "smooth",
        name: "unit",
        spec: ParamSpec::Str { default: "samples", options: &["samples", "seconds", "updates"], refresh: false },
        expression: None,
        doc: Some(
            "What `size` counts. `updates` smooths each position across frames instead of along \
             an axis, which is what a per-update value needs.",
        ),
    },
    ParamDecl {
        group: "smooth",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis to smooth along. -1 is time."),
    },
    ParamDecl {
        group: "smooth",
        name: "reset",
        spec: ParamSpec::Pulse,
        expression: None,
        doc: Some("Forget the past, so the node starts again from the next frame."),
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
    doc: "Take the edge off a signal, as a running mean or as an exponential decay.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Smooth, MANIFEST);
