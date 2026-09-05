//! Wavelet — how a signal's frequencies come and go over the frame, which a single spectrum
//! cannot show. Per frame, so it follows a Buffer.

use goofi_core::{resolve_axis, stream, Axis, Coord, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};
use rustfft::{num_complex::Complex32, FftPlanner};

struct Wavelet {
    planner: FftPlanner<f32>,
}

impl Default for Wavelet {
    fn default() -> Wavelet {
        Wavelet { planner: FftPlanner::new() }
    }
}

/// One wavelet's answer at every frequency bin of a transform of length `n`.
fn kernel(kind: &str, freq: f64, cycles: f64, sfreq: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| {
            // Only the positive half answers: the analytic wavelet has no negative side.
            let f = if k <= n / 2 { k as f64 * sfreq / n as f64 } else { return 0.0 };
            if kind == "mexican_hat" {
                let x = f / freq;
                (x * x * (-x * x / 2.0).exp() * 2.0) as f32
            } else {
                // Morlet: a bell around `freq` whose width is set by how many cycles it spans.
                let width = freq / cycles;
                let z = (f - freq) / width;
                (2.0 * (-0.5 * z * z).exp()) as f32
            }
        })
        .collect()
}

impl Node for Wavelet {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let dim = resolve_axis(p.i64("wavelet", "axis").unwrap_or(-1), a.shape().len())?;
        let n = a.shape()[dim];
        if n < 8 {
            return Err(format!("needs at least 8 samples along the axis, got {n}").into());
        }
        let sfreq = d.meta().sfreq().ok_or("this node needs a frame that carries its sample rate")?;
        let kind = p.str("wavelet", "wavelet").unwrap_or("morlet");
        let cycles = p.f64("wavelet", "cycles").unwrap_or(7.0).clamp(1.0, 50.0);
        let low = p.f64("range", "low").unwrap_or(1.0).max(1e-6);
        let high = p.f64("range", "high").unwrap_or(40.0).max(low * 1.000001);
        let count = p.i64("range", "count").unwrap_or(40).clamp(2, 512) as usize;
        let log = p.str("range", "scale").unwrap_or("linear") == "log";

        let freqs: Vec<f64> = (0..count)
            .map(|i| {
                let t = i as f64 / (count - 1) as f64;
                (low.ln() + t * (high.ln() - low.ln())).exp()
            })
            .collect();
        let forward = self.planner.plan_fft_forward(n);
        let inverse = self.planner.plan_fft_inverse(n);
        let kernels: Vec<Vec<f32>> =
            freqs.iter().map(|f| kernel(kind, *f, cycles, sfreq, n)).collect();

        // One input lane becomes `count` lanes, one per frequency, laid consecutively so the new
        // axis lands just before time.
        let (mut mag, mut ang) = (Vec::new(), Vec::new());
        let mut spectrum = vec![Complex32::default(); n];
        let mut scratch = vec![Complex32::default(); n];
        for lane in stream::lanes(a.shape(), dim, a.as_bytes()) {
            for (c, x) in spectrum.iter_mut().zip(&lane) {
                *c = Complex32::new(*x, 0.0);
            }
            forward.process(&mut spectrum);
            for k in &kernels {
                for (s, (c, w)) in scratch.iter_mut().zip(spectrum.iter().zip(k)) {
                    *s = c * *w;
                }
                inverse.process(&mut scratch);
                let z: Vec<Complex32> = scratch.iter().map(|c| c / n as f32).collect();
                mag.push(z.iter().map(|c| c.norm()).collect::<Vec<f32>>());
                ang.push(z.iter().map(|c| c.arg()).collect::<Vec<f32>>());
            }
        }

        let mut shape_out = a.shape().to_vec();
        shape_out.insert(dim, count);
        let coords: Vec<Coord> = freqs.iter().map(|f| Coord::Num(*f)).collect();
        // The last axis is still time, so the rate rides through with it.
        let meta = d.meta().insert_axis(dim, Axis::coords(coords), a.shape().len());
        for (name, lanes) in [("out", &mag), ("phase", &ang)] {
            let scaled: Vec<Vec<f32>> = if log && name == "out" {
                lanes.iter().map(|l| l.iter().map(|v| (v.max(1e-12)).ln()).collect()).collect()
            } else {
                lanes.to_vec()
            };
            let buf = stream::unlanes(&shape_out, dim + 1, &scaled);
            out.set(name, Data::array_f32(shape_out.clone(), buf, meta.clone()).map_err(|e| e.to_string())?);
        }
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "wavelet",
        name: "wavelet",
        spec: ParamSpec::Str { default: "morlet", options: &["morlet", "mexican_hat"], refresh: false },
        expression: None,
        doc: Some(
            "The shape looked for at each frequency. `morlet` reads a steady oscillation well; \
             `mexican_hat` reads a sudden one.",
        ),
    },
    ParamDecl {
        group: "wavelet",
        name: "cycles",
        spec: ParamSpec::Float { default: 7.0, min: 1.0, max: 50.0 },
        expression: None,
        doc: Some(
            "How many cycles the Morlet shape spans. More cycles tell frequencies apart better and \
             tell moments apart worse.",
        ),
    },
    ParamDecl {
        group: "wavelet",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis holds the samples. -1 is time."),
    },
    ParamDecl {
        group: "range",
        name: "low",
        spec: ParamSpec::Float { default: 1.0, min: 0.01, max: 10_000.0 },
        expression: None,
        doc: Some("The lowest frequency to look for, in Hz."),
    },
    ParamDecl {
        group: "range",
        name: "high",
        spec: ParamSpec::Float { default: 40.0, min: 0.01, max: 10_000.0 },
        expression: None,
        doc: Some("The highest frequency to look for, in Hz."),
    },
    ParamDecl {
        group: "range",
        name: "count",
        spec: ParamSpec::Int { default: 40, min: 2, max: 512 },
        expression: None,
        doc: Some("How many frequencies to look at, spaced evenly by ratio between the two above."),
    },
    ParamDecl {
        group: "range",
        name: "scale",
        spec: ParamSpec::Str { default: "linear", options: &["linear", "log"], refresh: false },
        expression: None,
        doc: Some("Whether the strength comes out as it is, or as its logarithm."),
    },
];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[
    OutputDecl { name: "out", kind: SlotType::Array },
    OutputDecl { name: "phase", kind: SlotType::Array },
];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Analysis],
    doc: "How a signal's frequencies come and go across the frame, which one spectrum cannot show.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Wavelet, MANIFEST);
