//! Filter — a Butterworth band, run forwards and then backwards over the stitched past, so what
//! comes out is not shifted in time. It holds no filter state: the input's own past is the state.

use goofi_core::{resolve_axis, stream, Data, SlotType, Stream};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, SlotDecl, Tag};

/// One second-order section, normalized so `a0 == 1`.
#[derive(Clone, Copy, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Low,
    High,
    Notch,
}

impl Biquad {
    /// An RBJ section at `f0` Hz with quality `q`, for a stream sampled at `sfreq`.
    fn new(kind: Kind, f0: f64, q: f64, sfreq: f64) -> Biquad {
        // A cutoff at or past Nyquist makes `alpha` zero or negative, and the section unstable.
        let f0 = f0.clamp(sfreq * 1e-6, sfreq * 0.49);
        let w0 = std::f64::consts::TAU * f0 / sfreq;
        let (cos, alpha) = (w0.cos(), w0.sin() / (2.0 * q));
        let (b0, b1, b2) = match kind {
            Kind::Low => ((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0),
            Kind::High => ((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0),
            Kind::Notch => (1.0, -2.0 * cos, 1.0),
        };
        let a0 = 1.0 + alpha;
        Biquad {
            b0: (b0 / a0) as f32,
            b1: (b1 / a0) as f32,
            b2: (b2 / a0) as f32,
            a1: (-2.0 * cos / a0) as f32,
            a2: ((1.0 - alpha) / a0) as f32,
        }
    }

    /// What a steady input becomes at the far side of this section.
    fn dc_gain(&self) -> f32 {
        let den = 1.0 + self.a1 + self.a2;
        if den.abs() < 1e-9 {
            0.0
        } else {
            (self.b0 + self.b1 + self.b2) / den
        }
    }

    /// The state that leaves a steady input passing through unchanged, per unit of input — what
    /// stops the first sample of a pass from reading as a step out of silence.
    fn rest(&self) -> [f32; 2] {
        let g = self.dc_gain();
        let z1 = self.b2 - self.a2 * g;
        [self.b1 - self.a1 * g + z1, z1]
    }

    /// How long this section's slowest pole takes to fade, in samples.
    fn settling(&self) -> usize {
        let radius = self.a2.abs().sqrt().clamp(0.0, 0.999_9);
        if radius <= 0.0 {
            1
        } else {
            (-5.0 / radius.ln()).ceil() as usize
        }
    }
}

/// The Butterworth cascade of `order` (rounded up to even) at `f0`: each section takes a pole's Q.
fn butterworth(kind: Kind, f0: f64, order: usize, sfreq: f64, into: &mut Vec<Biquad>) {
    let n = order.max(2).div_ceil(2) * 2;
    for k in 0..n / 2 {
        let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * n) as f64;
        into.push(Biquad::new(kind, f0, 1.0 / (2.0 * theta.cos()), sfreq));
    }
}

/// One pass of the cascade over `lane`, each section starting at rest for the first sample.
fn pass(sections: &[Biquad], lane: &[f32]) -> Vec<f32> {
    let first = lane.first().copied().unwrap_or(0.0);
    let mut state: Vec<[f32; 2]> = Vec::with_capacity(sections.len());
    let mut level = first;
    for s in sections {
        let rest = s.rest();
        state.push([rest[0] * level, rest[1] * level]);
        level *= s.dc_gain();
    }
    lane.iter()
        .map(|sample| {
            let mut x = *sample;
            // Direct form II transposed, which stays accurate in f32 at low cutoffs.
            for (s, z) in sections.iter().zip(state.iter_mut()) {
                let y = s.b0 * x + z[0];
                z[0] = s.b1 * x - s.a1 * y + z[1];
                z[1] = s.b2 * x - s.a2 * y;
                x = y;
            }
            x
        })
        .collect()
}

/// `lane` with `pad` samples of its own reflection at each end, turned through the end sample —
/// so the extension continues the trend rather than facing a step.
fn odd_extend(lane: &[f32], pad: usize) -> Vec<f32> {
    let (first, last) = (lane[0], lane[lane.len() - 1]);
    let pad = pad.min(lane.len().saturating_sub(1));
    let head = (1..=pad).rev().map(|i| 2.0 * first - lane[i]);
    let tail = (1..=pad).map(|i| 2.0 * last - lane[lane.len() - 1 - i]);
    head.chain(lane.iter().copied()).chain(tail).collect()
}

#[derive(Default)]
struct Filter {
    past: Stream,
    sections: Vec<Biquad>,
}

impl Node for Filter {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let dim = resolve_axis(p.i64("filter", "axis").unwrap_or(-1), a.shape().len())?;
        let sfreq = d.meta().sfreq().ok_or(
            "no sfreq on the incoming frame: a cutoff in Hz needs to know the sample rate",
        )?;

        let mode = p.str("filter", "mode").unwrap_or("bandpass");
        let (low, high) = (p.f64("filter", "low").unwrap_or(1.0), p.f64("filter", "high").unwrap_or(40.0));
        let order = p.i64("filter", "order").unwrap_or(4).clamp(2, 16) as usize;
        if matches!(mode, "bandpass" | "notch") && low >= high {
            return Err(format!("{mode} needs low < high, got {low} and {high}").into());
        }
        self.sections.clear();
        match mode {
            "lowpass" => butterworth(Kind::Low, high, order, sfreq, &mut self.sections),
            "highpass" => butterworth(Kind::High, low, order, sfreq, &mut self.sections),
            "notch" => {
                // The RBJ notch takes a centre and a quality; this is the pair that means low..high.
                let centre = (low * high).sqrt();
                for _ in 0..order.div_ceil(2) {
                    self.sections.push(Biquad::new(Kind::Notch, centre, centre / (high - low), sfreq));
                }
            }
            _ => {
                butterworth(Kind::High, low, order, sfreq, &mut self.sections);
                butterworth(Kind::Low, high, order, sfreq, &mut self.sections);
            }
        }

        let settle = self.sections.iter().map(Biquad::settling).max().unwrap_or(1).min(1 << 16);
        // The PAST is what settles a pass, and it is real data. The reflection at each end is kept
        // short, as scipy keeps it: a long one is a block of constant the filter answers instead.
        let edge = 3 * (2 * self.sections.len() + 1);
        let n = a.shape()[dim];
        let (shape, stitched, at) = self.past.push(a.shape(), dim, a.as_bytes(), settle + n);
        let filtered: Vec<Vec<f32>> = stream::lanes(&shape, dim, &stitched)
            .iter()
            .map(|lane| {
                let pad = edge.min(lane.len().saturating_sub(1));
                let wide = odd_extend(lane, pad);
                let ahead = pass(&self.sections, &wide);
                let mut back: Vec<f32> = ahead.into_iter().rev().collect();
                back = pass(&self.sections, &back);
                back.reverse();
                back[pad + at..pad + at + n].to_vec()
            })
            .collect();
        let buf = stream::unlanes(a.shape(), dim, &filtered);
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn on_pulse(&mut self, _key: &ParamKey, _p: &Params<'_>) -> NodeResult {
        self.past.reset();
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "filter",
        name: "mode",
        spec: ParamSpec::Str {
            default: "bandpass",
            options: &["lowpass", "highpass", "bandpass", "notch"],
            refresh: false,
        },
        expression: None,
        doc: Some(
            "Which part of the spectrum survives: below `high`, above `low`, between the two, or \
             everything except between the two.",
        ),
    },
    ParamDecl {
        group: "filter",
        name: "low",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 10_000.0 },
        expression: None,
        doc: Some("The bottom edge of the band, in Hz. `lowpass` ignores it."),
    },
    ParamDecl {
        group: "filter",
        name: "high",
        spec: ParamSpec::Float { default: 40.0, min: 0.0, max: 10_000.0 },
        expression: None,
        doc: Some("The top edge of the band, in Hz. `highpass` ignores it."),
    },
    ParamDecl {
        group: "filter",
        name: "order",
        spec: ParamSpec::Int { default: 4, min: 2, max: 16 },
        expression: None,
        doc: Some("How sharply the edge cuts. A higher order is steeper and rings for longer."),
    },
    ParamDecl {
        group: "filter",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis to filter along. -1 is time."),
    },
    ParamDecl {
        group: "filter",
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
    doc: "Keep one band of the spectrum, run both ways so nothing comes out shifted in time.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Filter, MANIFEST);
