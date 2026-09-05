//! Psd — how much power sits at each frequency. Welch averages the spectra of overlapping
//! segments, which trades resolution for a steadier answer; `fft` takes one segment, the whole run.

use std::sync::Arc;

use goofi_core::{resolve_axis, stream, Axis, Coord, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};
use rustfft::{num_complex::Complex32, FftPlanner};

struct Psd {
    planner: FftPlanner<f32>,
}

impl Default for Psd {
    fn default() -> Psd {
        Psd { planner: FftPlanner::new() }
    }
}

/// A periodic cosine window: `a0 - a1·cos(x) + a2·cos(2x)`, which covers the three on offer.
fn window(kind: &str, n: usize) -> Vec<f32> {
    let (a0, a1, a2) = match kind {
        "hamming" => (0.54, 0.46, 0.0),
        "blackman" => (0.42, 0.5, 0.08),
        "hann" => (0.5, 0.5, 0.0),
        _ => return vec![1.0; n],
    };
    (0..n)
        .map(|i| {
            let x = std::f64::consts::TAU * i as f64 / n as f64;
            (a0 - a1 * x.cos() + a2 * (2.0 * x).cos()) as f32
        })
        .collect()
}

impl Node for Psd {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let dim = resolve_axis(p.i64("psd", "axis").unwrap_or(-1), a.shape().len())?;
        let n = a.shape()[dim];
        if n < 2 {
            return Err(format!("needs at least 2 samples along the axis, got {n}").into());
        }
        let sfreq = d.meta().sfreq().ok_or("this node needs a frame that carries its sample rate")?;

        let taper = p.str("psd", "window").unwrap_or("hann");
        let seg = if p.str("psd", "mode").unwrap_or("welch") == "fft" {
            n
        } else {
            let size = p.f64("welch", "segment").unwrap_or(0.5);
            let samples = match p.str("welch", "unit").unwrap_or("seconds") {
                "samples" => size,
                "fraction" => size * n as f64,
                _ => size * sfreq,
            };
            (samples.round().max(2.0) as usize).min(n)
        };
        let overlap = p.f64("welch", "overlap").unwrap_or(0.5).clamp(0.0, 0.95);
        let hop = ((seg as f64 * (1.0 - overlap)).round() as usize).clamp(1, seg);

        let bins = seg / 2 + 1;
        let taper = window(taper, seg);
        // Dividing by `sfreq · Σw²` makes the result a DENSITY: the segment length cannot move a peak.
        let norm = 1.0 / (sfreq as f32 * taper.iter().map(|w| w * w).sum::<f32>());
        let fft = self.planner.plan_fft_forward(seg);

        let freqs: Vec<f64> = (0..bins).map(|k| k as f64 * sfreq / seg as f64).collect();
        let low = p.f64("range", "low").unwrap_or(0.0);
        let high = p.f64("range", "high").unwrap_or(0.0);
        let kept: Vec<usize> = (0..bins)
            .filter(|k| freqs[*k] >= low && (high <= 0.0 || freqs[*k] <= high))
            .collect();
        if kept.is_empty() {
            return Err(format!("no bin falls between {low} Hz and {high} Hz").into());
        }
        let log = p.str("range", "scale").unwrap_or("linear") == "log";

        let mut scratch = vec![Complex32::default(); seg];
        let spectra: Vec<Vec<f32>> = stream::lanes(a.shape(), dim, a.as_bytes())
            .iter()
            .map(|lane| {
                let mut sum = vec![0.0f32; bins];
                let mut count = 0.0f32;
                for start in (0..=n - seg).step_by(hop) {
                    for (c, (x, w)) in scratch.iter_mut().zip(lane[start..].iter().zip(&taper)) {
                        *c = Complex32::new(x * w, 0.0);
                    }
                    fft.process(&mut scratch);
                    for (k, (s, c)) in sum.iter_mut().zip(&scratch[..bins]).enumerate() {
                        // DC and, for an even length, Nyquist have no twin to fold in.
                        let fold = if k == 0 || 2 * k == seg { 1.0 } else { 2.0 };
                        *s += c.norm_sqr() * norm * fold;
                    }
                    count += 1.0;
                }
                kept.iter()
                    .map(|k| {
                        let v = sum[*k] / count;
                        if log { v.max(1e-12).ln() } else { v }
                    })
                    .collect()
            })
            .collect();

        let mut shape_out = a.shape().to_vec();
        shape_out[dim] = kept.len();
        let coords: Vec<Coord> = kept.iter().map(|k| Coord::Num(freqs[*k])).collect();
        // No longer a time series: `sfreq` would read as the spacing of a domain that is gone.
        let mut meta = d.meta().clone();
        let axes = meta.channels().clone().with(dim, Axis::coords(Arc::from(coords)));
        meta.set_channels(axes);
        meta.set_sfreq(None);
        let buf = stream::unlanes(&shape_out, dim, &spectra);
        out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "psd",
        name: "mode",
        spec: ParamSpec::Str { default: "welch", options: &["fft", "welch"], refresh: false },
        expression: None,
        doc: Some(
            "`welch` averages the spectra of overlapping segments, which is steadier; `fft` takes \
             the whole run as one segment, which is sharper and noisier.",
        ),
    },
    ParamDecl {
        group: "psd",
        name: "window",
        spec: ParamSpec::Str {
            default: "hann",
            options: &["hann", "hamming", "blackman", "rectangular"],
            refresh: false,
        },
        expression: None,
        doc: Some(
            "Taper applied to each segment. It stops a peak from smearing across the spectrum; \
             `rectangular` keeps the samples as they are.",
        ),
    },
    ParamDecl {
        group: "psd",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis holds the samples. -1 is time."),
    },
    ParamDecl {
        group: "welch",
        name: "segment",
        spec: ParamSpec::Float { default: 0.5, min: 0.0, max: 1e7 },
        expression: None,
        doc: Some("How long one segment is. A longer segment tells frequencies apart better."),
    },
    ParamDecl {
        group: "welch",
        name: "unit",
        spec: ParamSpec::Str {
            default: "seconds",
            options: &["seconds", "samples", "fraction"],
            refresh: false,
        },
        expression: None,
        doc: Some("What `segment` counts in. `fraction` is a share of the frame."),
    },
    ParamDecl {
        group: "welch",
        name: "overlap",
        spec: ParamSpec::Float { default: 0.5, min: 0.0, max: 0.95 },
        expression: None,
        doc: Some("How much of a segment the next one repeats. More overlap is steadier and slower."),
    },
    ParamDecl {
        group: "range",
        name: "low",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 10_000.0 },
        expression: None,
        doc: Some("The lowest frequency to keep, in Hz."),
    },
    ParamDecl {
        group: "range",
        name: "high",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 10_000.0 },
        expression: None,
        doc: Some("The highest frequency to keep, in Hz. 0 keeps every bin above `low`."),
    },
    ParamDecl {
        group: "range",
        name: "scale",
        spec: ParamSpec::Str { default: "linear", options: &["linear", "log"], refresh: false },
        expression: None,
        doc: Some("Whether the power comes out as it is, or as its logarithm."),
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
    tags: &[Tag::Analysis],
    doc: "How much power sits at each frequency, in Hz, over the last axis.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Psd, MANIFEST);
