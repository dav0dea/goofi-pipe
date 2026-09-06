//! Threshold — a signal becomes a decision: one where it is past the level, zero where it is not.
//! Its output is what a `reset` or a `trigger` pulse elsewhere in the patch references.

use goofi_core::{Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

/// One element's decision, when it last changed, and how long the comparison has disagreed.
#[derive(Clone, Copy, Default)]
struct State {
    high: bool,
    since: f64,
    /// What the raw comparison last said, and when it started saying it.
    want: bool,
    want_since: f64,
}

#[derive(Default)]
struct Threshold {
    states: Vec<State>,
}

impl Node for Threshold {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.as_array()?;
        let level = p.f64("threshold", "level").unwrap_or(0.0) as f32;
        let band = p.f64("threshold", "hysteresis").unwrap_or(0.0).abs() as f32;
        let below = p.str("threshold", "mode").unwrap_or("above") == "below";
        let edge_only = p.bool("threshold", "edge").unwrap_or(false);
        let hold = p.f64("threshold", "hold").unwrap_or(0.0).max(0.0);
        let dwell = p.f64("threshold", "dwell").unwrap_or(0.0).max(0.0);

        let count = a.as_bytes().len() / 4;
        if self.states.len() != count {
            self.states = vec![State::default(); count];
        }
        let mut buf = Vec::with_capacity(count * 4);
        for (x, state) in a.as_bytes().chunks_exact(4).zip(&mut self.states) {
            let v = f32::from_le_bytes(x.try_into().expect("four bytes"));
            // The band is crossed from whichever side the state is on, so noise at the level
            // cannot rattle the decision.
            let past = if state.high { v > level - band } else { v > level + band };
            let want = if below { !past } else { past };
            // The comparison has to keep saying the same thing for `dwell` before the decision
            // follows it, so a brief excursion past the level is not a state.
            if want != state.want {
                state.want = want;
                state.want_since = c.now;
            }
            let mut rose = false;
            if want != state.high && c.now - state.want_since >= dwell && c.now - state.since >= hold {
                rose = want;
                state.high = want;
                state.since = c.now;
            }
            let emit = if edge_only { rose } else { state.high };
            buf.extend_from_slice(&if emit { 1.0f32 } else { 0.0f32 }.to_le_bytes());
        }
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "threshold",
        name: "mode",
        spec: ParamSpec::Str { default: "above", options: &["above", "below"], refresh: false },
        expression: None,
        doc: Some("Whether the decision is true above the level or below it."),
    },
    ParamDecl {
        group: "threshold",
        name: "level",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The value the signal is compared against."),
    },
    ParamDecl {
        group: "threshold",
        name: "hysteresis",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 1.0e9 },
        expression: None,
        doc: Some(
            "How far past the level the signal must go to switch back, so noise sitting on the \
             level cannot rattle the decision.",
        ),
    },
    ParamDecl {
        group: "threshold",
        name: "edge",
        spec: ParamSpec::Bool { default: false },
        expression: None,
        doc: Some("Emit one only on the update where the decision turns true, rather than while it holds."),
    },
    ParamDecl {
        group: "threshold",
        name: "dwell",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 3600.0 },
        expression: None,
        doc: Some(
            "How long in seconds the comparison must keep saying the same thing before the \
             decision follows it, so a brief excursion past the level is not a state. It delays \
             the release as well as the onset.",
        ),
    },
    ParamDecl {
        group: "threshold",
        name: "hold",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 3600.0 },
        expression: None,
        doc: Some("How long in seconds a decision stays before it is allowed to switch again."),
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
    doc: "Turn a signal into a decision.\n\
          One where it is past the level, zero where it is not.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Threshold, MANIFEST);
