//! FreqShift — move a signal up or down the spectrum without changing its speed. The oscillator's
//! phase is the stream's own sample count, so the shift stays continuous across frames with no
//! state but the input's past.

use goofi_core::{stream, Data, SlotType, Stream};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, SlotDecl, Tag};
use rustfft::{num_complex::Complex32, FftPlanner};

struct FreqShift {
    past: Stream,
    planner: FftPlanner<f32>,
    /// Samples of this stream already sent, which is what the oscillator's phase counts.
    sent: u64,
}

impl Default for FreqShift {
    fn default() -> FreqShift {
        FreqShift { past: Stream::default(), planner: FftPlanner::new(), sent: 0 }
    }
}

impl Node for FreqShift {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let dim = a.shape().len() - 1;
        let n = a.shape()[dim];
        let sfreq = d.meta().sfreq().ok_or("this node needs a frame that carries its sample rate")?;
        let shift = p.f64("freq_shift", "frequency").unwrap_or(0.0);
        let ring = p.str("freq_shift", "mode").unwrap_or("single") == "ring";

        // The analytic signal needs settling room either side, so the reach is a power of two past
        // the frame rather than a filter's ringing length.
        let reach = n.next_power_of_two();
        let (shape, stitched, at) = self.past.push(a.shape(), dim, a.as_bytes(), reach);
        let total = shape[dim];
        let forward = self.planner.plan_fft_forward(total);
        let inverse = self.planner.plan_fft_inverse(total);

        // Where the frame's first sample sits in the stream, counted from the very first sample.
        let base = self.sent as f64 - (total - at) as f64 + n as f64;
        let step = std::f64::consts::TAU * shift / sfreq;
        let mut scratch = vec![Complex32::default(); total];
        let shifted: Vec<Vec<f32>> = stream::lanes(&shape, dim, &stitched)
            .iter()
            .map(|lane| {
                for (c, x) in scratch.iter_mut().zip(lane) {
                    *c = Complex32::new(*x, 0.0);
                }
                forward.process(&mut scratch);
                let half = total / 2;
                for (k, c) in scratch.iter_mut().enumerate() {
                    if k == 0 || (total % 2 == 0 && k == half) {
                        continue;
                    } else if k < half {
                        *c *= 2.0;
                    } else {
                        *c = Complex32::default();
                    }
                }
                inverse.process(&mut scratch);
                (0..n)
                    .map(|j| {
                        let z = scratch[at + j] / total as f32;
                        let phase = step * (base + j as f64);
                        let turn = Complex32::new(phase.cos() as f32, phase.sin() as f32);
                        // Ring modulation keeps both sidebands, which is the real part alone.
                        if ring {
                            z.re * turn.re
                        } else {
                            (z * turn).re
                        }
                    })
                    .collect()
            })
            .collect();
        self.sent += n as u64;
        let buf = stream::unlanes(a.shape(), dim, &shifted);
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn on_pulse(&mut self, _key: &ParamKey, _p: &Params<'_>) -> NodeResult {
        self.past.reset();
        self.sent = 0;
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "freq_shift",
        name: "frequency",
        spec: ParamSpec::Float { default: 0.0, min: -5000.0, max: 5000.0 },
        expression: None,
        doc: Some(
            "How far to move the signal, in Hz, up for a positive number and down for a negative \
             one. Unlike a pitch change, the speed of the signal does not change with it.",
        ),
    },
    ParamDecl {
        group: "freq_shift",
        name: "mode",
        spec: ParamSpec::Str { default: "single", options: &["single", "ring"], refresh: false },
        expression: None,
        doc: Some(
            "`single` moves the signal one way only; `ring` keeps both the sum and the difference, \
             which is the harsher, older sound.",
        ),
    },
    ParamDecl {
        group: "freq_shift",
        name: "reset",
        spec: ParamSpec::Pulse,
        expression: None,
        doc: Some("Forget the past and start the shift's own phase again."),
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
    doc: "Move a signal up or down the spectrum.\n\
          By a fixed number of hertz, without changing its speed.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(FreqShift, MANIFEST);
