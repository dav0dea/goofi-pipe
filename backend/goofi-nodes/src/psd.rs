//! Psd — a one-sided periodogram over the last axis: `[.., T]` in, `[.., T/2 + 1]` out.
//! Interior bins are doubled, for the energy their negative-frequency twins carry.

use std::sync::Arc;

use goofi_core::{Axis, Coord, Data, SlotType};
use goofi_node::{NodeManifest, OutputDecl, ParamDecl, ParamSpec, Params, SlotDecl};
use goofi_signal::{default_factory, Inputs, Node, NodeClass, NodeCtx, NodeResult, Outputs};
use rustfft::{num_complex::Complex32, FftPlanner};

struct Psd {
    planner: FftPlanner<f32>,
    /// The window, and the length and kind it was built for.
    window: Vec<f32>,
    built: (usize, String),
}

impl Default for Psd {
    fn default() -> Psd {
        Psd { planner: FftPlanner::new(), window: Vec::new(), built: (0, String::new()) }
    }
}

/// A periodic cosine window: `a0 - a1·cos(x) + a2·cos(2x)`, which covers the three on offer.
fn cosine_window(kind: &str, n: usize) -> Vec<f32> {
    let (a0, a1, a2) = match kind {
        "hamming" => (0.54, 0.46, 0.0),
        "blackman" => (0.42, 0.5, 0.08),
        "none" => return vec![1.0; n],
        _ => (0.5, 0.5, 0.0), // hann
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
        let Some(d) = inp.get("data") else { return Ok(()) };
        let a = d.assert_ndims().at_least(1)?;
        let shape = a.shape();
        let n = *shape.last().expect("at_least(1) rejects a rank-0 array");
        if n < 2 {
            return Err(format!("needs at least 2 samples along the last axis, got {n}").into());
        }
        // Absent, the bins are cycles per sample — a usable spectrum, so a fallback and not an error.
        let sfreq = d.meta().sfreq().unwrap_or(1.0);

        let kind = p.str("psd", "window").unwrap_or("hann");
        if self.built != (n, kind.to_string()) {
            self.window = cosine_window(kind, n);
            self.built = (n, kind.to_string());
        }
        let fft = self.planner.plan_fft_forward(n);

        let bins = n / 2 + 1;
        let rows: usize = shape[..shape.len() - 1].iter().product();
        // Dividing by `sfreq · Σw²` makes the result a DENSITY: window length cannot move a peak.
        let norm = 1.0 / (sfreq as f32 * self.window.iter().map(|w| w * w).sum::<f32>());
        let src = a.as_bytes();
        let mut scratch = vec![Complex32::default(); n];
        let mut buf = Vec::with_capacity(rows * bins * 4);
        for r in 0..rows {
            let row = &src[r * n * 4..(r + 1) * n * 4];
            for (c, (x, w)) in scratch.iter_mut().zip(row.chunks_exact(4).zip(&self.window)) {
                *c = Complex32::new(f32::from_le_bytes(x.try_into().expect("four bytes")) * w, 0.0);
            }
            fft.process(&mut scratch);
            for (k, c) in scratch[..bins].iter().enumerate() {
                // DC and, for an even length, Nyquist have no twin to fold in.
                let fold = if k == 0 || 2 * k == n { 1.0 } else { 2.0 };
                buf.extend_from_slice(&(c.norm_sqr() * norm * fold).to_le_bytes());
            }
        }

        let mut shape_out = shape.to_vec();
        let last = shape_out.len() - 1;
        shape_out[last] = bins;
        let freqs: Vec<Coord> =
            (0..bins).map(|k| Coord::Num(k as f64 * sfreq / n as f64)).collect();
        // No longer a time series: `sfreq` would read as the spacing of a domain that is gone.
        let mut meta = d.meta().clone();
        let axes = meta.channels().clone().with(last, Axis::coords(Arc::from(freqs)));
        meta.set_channels(axes);
        meta.set_sfreq(None);
        out.set("psd", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "psd",
    name: "window",
    spec: ParamSpec::Str {
        default: "hann",
        options: &["hann", "hamming", "blackman", "none"],
        refresh: false,
    },
    expression: None,
    doc: Some(
        "Taper applied before the transform. It stops a peak from smearing across the spectrum; \
         `none` keeps the samples as they are, which is correct only for a whole number of cycles.",
    ),
}];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "psd",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeClass {
        manifest: NodeManifest {
            type_name: "Psd",
            category: "signal",
            doc: "Power spectral density over the last axis: [.., T] becomes [.., T/2 + 1], in Hz.",
            inputs: INPUTS,
            outputs: OUTPUTS,
            params: PARAMS,
            producer: false,
        },
        isolation: &goofi_node::NATIVE,
        factory: default_factory::<Psd>,
    }
}
