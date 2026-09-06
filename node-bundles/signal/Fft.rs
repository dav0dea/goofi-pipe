//! Fft — the spectrum both ways. Forward turns a run of samples into one-sided bins, each carrying
//! a pair of numbers; inverse takes that pair back to samples, and restores the rate from the bins.

use std::sync::Arc;

use goofi_core::{resolve_axis, Axis, Coord, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};
use rustfft::{num_complex::Complex32, FftPlanner};

struct Fft {
    planner: FftPlanner<f32>,
}

impl Default for Fft {
    fn default() -> Fft {
        Fft { planner: FftPlanner::new() }
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

impl Fft {
    fn forward(&mut self, d: &Data, dim: usize, polar: bool, taper: &str, out: &mut Outputs<'_>) -> NodeResult {
        let a = d.as_array()?;
        let shape = a.shape();
        let t = shape[dim];
        if t < 2 {
            return Err(format!("needs at least 2 samples along the axis, got {t}").into());
        }
        let bins = t / 2 + 1;
        let (outer, inner) = (shape[..dim].iter().product::<usize>(), shape[dim + 1..].iter().product::<usize>());
        let taper = window(taper, t);
        let fft = self.planner.plan_fft_forward(t);
        let src = a.as_bytes();
        let read = |o: usize, k: usize, i: usize| {
            let at = (((o * t) + k) * inner + i) * 4;
            f32::from_le_bytes(src[at..at + 4].try_into().expect("four bytes"))
        };

        let mut buf = vec![0u8; outer * bins * 2 * inner * 4];
        let mut scratch = vec![Complex32::default(); t];
        for o in 0..outer {
            for i in 0..inner {
                for (k, c) in scratch.iter_mut().enumerate() {
                    *c = Complex32::new(read(o, k, i) * taper[k], 0.0);
                }
                fft.process(&mut scratch);
                for (f, c) in scratch[..bins].iter().enumerate() {
                    let pair = if polar { [c.norm(), c.arg()] } else { [c.re, c.im] };
                    for (part, v) in pair.iter().enumerate() {
                        let at = ((((o * bins) + f) * 2 + part) * inner + i) * 4;
                        buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }

        let sfreq = d.meta().sfreq().ok_or("a forward transform needs a frame that carries its sample rate")?;
        let freqs: Vec<Coord> = (0..bins).map(|k| Coord::Num(k as f64 * sfreq / t as f64)).collect();
        let parts: Vec<Coord> = if polar {
            vec![Coord::Str("magnitude".into()), Coord::Str("phase".into())]
        } else {
            vec![Coord::Str("real".into()), Coord::Str("imaginary".into())]
        };
        let mut shape_out = shape.to_vec();
        shape_out[dim] = bins;
        shape_out.insert(dim + 1, 2);
        // A spectrum is no longer a time series, so the rate would read as the spacing of a domain
        // that is gone.
        let meta = d.meta().insert_axis(dim + 1, Axis::coords(Arc::from(parts)), shape.len());
        let axes = meta.channels().clone().with(dim, Axis::coords(Arc::from(freqs)));
        let meta = meta.with_channels(axes).with_sfreq(None);
        out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn inverse(&mut self, d: &Data, dim: usize, polar: bool, out: &mut Outputs<'_>) -> NodeResult {
        let a = d.assert_ndims().at_least(2)?;
        let shape = a.shape();
        if shape.get(dim + 1) != Some(&2) {
            return Err(format!("an inverse needs a pair axis of 2 after the bins, got {shape:?}").into());
        }
        let bins = shape[dim];
        let t = (bins - 1) * 2;
        let (outer, inner) = (shape[..dim].iter().product::<usize>(), shape[dim + 2..].iter().product::<usize>());
        let fft = self.planner.plan_fft_inverse(t);
        let src = a.as_bytes();
        let read = |o: usize, f: usize, part: usize, i: usize| {
            let at = ((((o * bins) + f) * 2 + part) * inner + i) * 4;
            f32::from_le_bytes(src[at..at + 4].try_into().expect("four bytes"))
        };

        let mut buf = vec![0u8; outer * t * inner * 4];
        let mut scratch = vec![Complex32::default(); t];
        for o in 0..outer {
            for i in 0..inner {
                for f in 0..bins {
                    let (x, y) = (read(o, f, 0, i), read(o, f, 1, i));
                    let c = if polar { Complex32::from_polar(x, y) } else { Complex32::new(x, y) };
                    scratch[f] = c;
                    // The negative half is the conjugate mirror the forward pass folded away.
                    if f > 0 && f < bins - 1 {
                        scratch[t - f] = c.conj();
                    }
                }
                fft.process(&mut scratch);
                for (k, c) in scratch.iter().enumerate() {
                    let at = (((o * t) + k) * inner + i) * 4;
                    buf[at..at + 4].copy_from_slice(&(c.re / t as f32).to_le_bytes());
                }
            }
        }

        // The bins say how far apart they are, and that spacing times the length is the rate.
        let sfreq = d.meta().channels().get(dim).and_then(|x| x.coords.as_ref()).and_then(|c| {
            match (c.first(), c.get(1)) {
                (Some(Coord::Num(a)), Some(Coord::Num(b))) => Some((b - a) * t as f64),
                _ => None,
            }
        });
        let mut shape_out = shape.to_vec();
        shape_out.remove(dim + 1);
        shape_out[dim] = t;
        let meta = d.meta().drop_axis(dim + 1, shape.len());
        let axes = meta.channels().clone().with(dim, Axis::default());
        let meta = meta.with_channels(axes).with_sfreq(sfreq);
        out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

impl Node for Fft {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let polar = p.str("fft", "form").unwrap_or("polar") == "polar";
        if p.str("fft", "mode").unwrap_or("forward") == "inverse" {
            let dim = resolve_axis(p.i64("fft", "axis").unwrap_or(-1), a.shape().len().saturating_sub(1).max(1))?;
            self.inverse(d, dim, polar, out)
        } else {
            let dim = resolve_axis(p.i64("fft", "axis").unwrap_or(-1), a.shape().len())?;
            self.forward(d, dim, polar, p.str("fft", "window").unwrap_or("none"), out)
        }
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "fft",
        name: "mode",
        spec: ParamSpec::Str { default: "forward", options: &["forward", "inverse"], refresh: false },
        expression: None,
        doc: Some("`forward` turns samples into bins; `inverse` turns the same bins back into samples."),
    },
    ParamDecl {
        group: "fft",
        name: "form",
        spec: ParamSpec::Str { default: "polar", options: &["polar", "complex"], refresh: false },
        expression: None,
        doc: Some(
            "How each bin's pair of numbers reads: `polar` as a size and an angle, `complex` as a \
             real and an imaginary part. An inverse must be told the same form the forward wrote.",
        ),
    },
    ParamDecl {
        group: "fft",
        name: "window",
        spec: ParamSpec::Str { default: "none", options: &["none", "hann", "hamming", "blackman"], refresh: false },
        expression: None,
        doc: Some(
            "Taper applied before a forward transform, which stops a peak from smearing across the \
             spectrum. An inverse ignores it.",
        ),
    },
    ParamDecl {
        group: "fft",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis holds the samples going forward, or the bins coming back. -1 is time."),
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
    doc: "The spectrum, both ways.\n\
          Samples into one-sided bins, and those bins back into samples.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Fft, MANIFEST);
