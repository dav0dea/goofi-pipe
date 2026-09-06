//! Math — the arithmetic every patch needs on the way between two nodes: a scale and a shift,
//! then a range mapped onto another range, and how a value outside that range is brought back in.
//! Shape and meta are untouched.

use goofi_core::{Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

/// Bring `v` inside `[lo, hi]` the way `bound` says, or leave it where it is.
fn bind(v: f32, lo: f32, hi: f32, bound: &str) -> f32 {
    match bound {
        // A NaN bound has no inside; `clamp` would panic on it rather than pass the value on.
        "clamp" if lo <= hi => v.clamp(lo, hi),
        // Modulo, so the top of the range and the bottom are the same place.
        "wrap" if hi > lo => lo + (v - lo).rem_euclid(hi - lo),
        // The octave fold: the value keeps its pitch class and changes register. Only a positive
        // range has octaves, and a range narrower than one cannot hold every value — what comes
        // out is then the nearest octave of the input, which may still sit outside.
        "fold" if v > 0.0 && v.is_finite() && lo > 0.0 && hi > lo => {
            let mut v = v;
            while v > hi {
                v *= 0.5;
            }
            while v < lo {
                v *= 2.0;
            }
            v
        }
        _ => v,
    }
}

#[derive(Default)]
struct Math;

impl Node for Math {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.as_array()?;
        let pre = p.f64("math", "pre_add").unwrap_or(0.0) as f32;
        let mul = p.f64("math", "multiply").unwrap_or(1.0) as f32;
        let post = p.f64("math", "post_add").unwrap_or(0.0) as f32;
        let from_low = p.f64("range", "from_low").unwrap_or(0.0) as f32;
        let from_high = p.f64("range", "from_high").unwrap_or(1.0) as f32;
        let to_low = p.f64("range", "to_low").unwrap_or(0.0) as f32;
        let to_high = p.f64("range", "to_high").unwrap_or(1.0) as f32;
        let bound = p.str("range", "bound").unwrap_or("none");
        // A zero-width source range has no ratio to carry, so every value lands at `to_low`.
        let span = from_high - from_low;
        let scale = if span == 0.0 { 0.0 } else { (to_high - to_low) / span };
        let (lo, hi) = if to_low <= to_high { (to_low, to_high) } else { (to_high, to_low) };

        let mut buf = Vec::with_capacity(a.as_bytes().len());
        for x in a.as_bytes().chunks_exact(4) {
            let v = f32::from_le_bytes(x.try_into().expect("four bytes"));
            let v = (v + pre) * mul + post;
            let v = to_low + (v - from_low) * scale;
            buf.extend_from_slice(&bind(v, lo, hi, bound).to_le_bytes());
        }
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "math",
        name: "pre_add",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("Added to every value before the multiply, which is how you centre a signal."),
    },
    ParamDecl {
        group: "math",
        name: "multiply",
        spec: ParamSpec::Float { default: 1.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("Scales every value; a negative number turns the signal upside down."),
    },
    ParamDecl {
        group: "math",
        name: "post_add",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("Added to every value after the multiply, which is how you set a baseline."),
    },
    ParamDecl {
        group: "range",
        name: "from_low",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The bottom of the range the values are expected to arrive in."),
    },
    ParamDecl {
        group: "range",
        name: "from_high",
        spec: ParamSpec::Float { default: 1.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The top of the range the values are expected to arrive in."),
    },
    ParamDecl {
        group: "range",
        name: "to_low",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The bottom of the range they are mapped onto."),
    },
    ParamDecl {
        group: "range",
        name: "to_high",
        spec: ParamSpec::Float { default: 1.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The top of the range they are mapped onto."),
    },
    ParamDecl {
        group: "range",
        name: "bound",
        spec: ParamSpec::Str { default: "none", options: &["none", "clamp", "wrap", "fold"], refresh: false },
        expression: None,
        doc: Some(
            "What happens to a value outside the target range. `clamp` holds it at the edge; \
             `wrap` carries it round to the other end; `fold` halves or doubles it until it lands \
             inside, which is how a spectral peak becomes an audible pitch or an LFO rate. `fold` \
             needs a positive range, and one narrower than an octave cannot hold every value.",
        ),
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
    doc: "Scale and shift every value, then remap the range.\n\
          `bound` says what happens to a value that lands outside it. Shape and metadata are untouched.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Math, MANIFEST);
