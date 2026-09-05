//! Hilbert — the analytic signal of a frame: how big the swing is, where in the cycle it is, and
//! how fast that is turning. Per frame, so it follows a Buffer.

use goofi_core::{resolve_axis, stream, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};
use rustfft::{num_complex::Complex32, FftPlanner};

struct Hilbert {
    planner: FftPlanner<f32>,
}

impl Default for Hilbert {
    fn default() -> Hilbert {
        Hilbert { planner: FftPlanner::new() }
    }
}

impl Node for Hilbert {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let dim = resolve_axis(p.i64("hilbert", "axis").unwrap_or(-1), a.shape().len())?;
        let n = a.shape()[dim];
        if n < 4 {
            return Err(format!("needs at least 4 samples along the axis, got {n}").into());
        }
        let sfreq = d.meta().sfreq().ok_or("this node needs a frame that carries its sample rate")?;
        let forward = self.planner.plan_fft_forward(n);
        let inverse = self.planner.plan_fft_inverse(n);

        let lanes = stream::lanes(a.shape(), dim, a.as_bytes());
        let (mut env, mut ang, mut hz) = (Vec::new(), Vec::new(), Vec::new());
        let mut scratch = vec![Complex32::default(); n];
        for lane in &lanes {
            for (c, x) in scratch.iter_mut().zip(lane) {
                *c = Complex32::new(*x, 0.0);
            }
            forward.process(&mut scratch);
            // The one-sided spectrum, doubled: the negative half of a real signal says nothing new.
            let half = n / 2;
            for (k, c) in scratch.iter_mut().enumerate() {
                if k == 0 || (n % 2 == 0 && k == half) {
                    continue;
                } else if k < half {
                    *c *= 2.0;
                } else {
                    *c = Complex32::default();
                }
            }
            inverse.process(&mut scratch);
            let z: Vec<Complex32> = scratch.iter().map(|c| c / n as f32).collect();
            env.push(z.iter().map(|c| c.norm()).collect::<Vec<f32>>());
            let phase: Vec<f32> = z.iter().map(|c| c.arg()).collect();
            // The turn between two samples, brought into one revolution, is the frequency.
            let mut rate = Vec::with_capacity(n);
            for k in 0..n {
                let (a, b) = (phase[k.max(1) - 1], phase[(k + 1).min(n - 1)]);
                let step = if k == 0 || k == n - 1 { 1.0 } else { 2.0 };
                let mut turn = (b - a) as f64;
                while turn > std::f64::consts::PI {
                    turn -= std::f64::consts::TAU;
                }
                while turn < -std::f64::consts::PI {
                    turn += std::f64::consts::TAU;
                }
                rate.push((turn / step * sfreq / std::f64::consts::TAU) as f32);
            }
            ang.push(phase);
            hz.push(rate);
        }

        let shape = a.shape().to_vec();
        for (name, lanes) in [("envelope", &env), ("phase", &ang), ("frequency", &hz)] {
            let buf = stream::unlanes(&shape, dim, lanes);
            out.set(name, Data::array_f32(shape.clone(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        }
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "hilbert",
    name: "axis",
    spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
    expression: None,
    doc: Some("Which axis holds the samples. -1 is time."),
}];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[
    OutputDecl { name: "envelope", kind: SlotType::Array },
    OutputDecl { name: "phase", kind: SlotType::Array },
    OutputDecl { name: "frequency", kind: SlotType::Array },
];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Analysis],
    doc: "How big a signal's swing is, where in its cycle it stands, and how fast that is turning.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Hilbert, MANIFEST);
