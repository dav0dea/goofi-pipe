//! Hold — the value the last `take` caught, kept until the next one. A pulse latches, and what
//! comes out glides to the newly caught value over `ramp` rather than stepping to it.
//!
//! The first frame is taken without being asked, so a fresh Hold answers with the signal rather
//! than with zero. A `take` that arrives between frames latches the next one to arrive.
//!
//! The node runs on its input, so the glide advances a step per frame that arrives and is only as
//! smooth as the input is fast — a ramp longer than a few frames of the source is what this is for.

use goofi_core::{Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Hold {
    /// Where the current glide started, per element.
    from: Vec<f32>,
    /// Where it is going, per element: the values the last take caught.
    to: Vec<f32>,
    /// `ctx.now` when the current glide began.
    started: f64,
    /// A `take` fired; the next frame is the one to catch.
    armed: bool,
    /// Whether anything has been caught yet. Until it has, the next frame is taken unasked.
    taken: bool,
}

impl Hold {
    /// Where the glide has reached at `now`, per element.
    fn at(&self, now: f64, ramp: f64) -> Vec<f32> {
        let t = if ramp <= 0.0 { 1.0 } else { ((now - self.started) / ramp).clamp(0.0, 1.0) as f32 };
        self.from.iter().zip(&self.to).map(|(a, b)| a + (b - a) * t).collect()
    }
}

impl Node for Hold {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.as_array()?;
        let ramp = p.f64("hold", "ramp").unwrap_or(0.0).max(0.0);

        let values: Vec<f32> = a
            .as_bytes()
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().expect("four bytes")))
            .collect();
        // A frame of a different width is a different signal: there is nothing to glide from.
        let fresh = !self.taken || values.len() != self.to.len();
        if fresh || self.armed {
            // Glide from wherever the last one had reached, so a take mid-glide does not jump.
            self.from = if fresh { values.clone() } else { self.at(c.now, ramp) };
            self.to = values;
            self.started = c.now;
            self.armed = false;
            self.taken = true;
        }

        let mut buf = Vec::with_capacity(self.to.len() * 4);
        for v in self.at(c.now, ramp) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn on_pulse(&mut self, _key: &ParamKey, _p: &Params<'_>) -> NodeResult {
        self.armed = true;
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "hold",
        name: "take",
        spec: ParamSpec::Pulse,
        expression: None,
        doc: Some(
            "Catch the input as it is now and hold it. A reference here fires on a rising edge, \
             so a Clock's pulse or a Threshold's decision drives the latch.",
        ),
    },
    ParamDecl {
        group: "hold",
        name: "ramp",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 600.0 },
        expression: None,
        doc: Some("How long in seconds to glide to a newly caught value. Zero steps to it."),
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
    tags: &[Tag::Control],
    doc: "Sample and hold: the value the last `take` caught.\n\
          It glides to a newly caught value over `ramp`. Shape and metadata are the input's.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Hold, MANIFEST);
