//! Quantize — every value pulled onto the nearest of an allowed set. The set is either counted
//! out — `count` values spread evenly from `low` to `high` — or wired in through `levels`, which
//! is how a tuning's own ratios become the only numbers a signal may take.
//!
//! `period` is what makes a wired set of a few numbers cover the whole line: the value is folded
//! into one period of the set, matched there, and put back in the register it came from — a scale
//! that runs one octave quantizing a pitch four octaves up. `ratio` folds by multiplying, which
//! is what an octave is; `linear` folds by adding.
//!
//! NaN in `levels` is padding, not a level: a `Tuning` output feeds straight in.

use goofi_core::{Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

/// Where `v` sits in an ascending set: the index of the nearest entry.
fn nearest(set: &[f32], v: f32) -> usize {
    let i = set.partition_point(|s| *s < v);
    if i == 0 {
        0
    } else if i >= set.len() {
        set.len() - 1
    } else if v - set[i - 1] <= set[i] - v {
        i - 1
    } else {
        i
    }
}

#[derive(Default)]
struct Quantize {
    /// The allowed values, ascending, rebuilt each frame from the params or the wire.
    set: Vec<f32>,
}

impl Node for Quantize {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.as_array()?;
        let counted = p.str("quantize", "mode").unwrap_or("count") == "count";
        let strength = p.f64("quantize", "strength").unwrap_or(1.0).clamp(0.0, 1.0) as f32;

        self.set.clear();
        if counted {
            let count = p.i64("quantize", "count").unwrap_or(12).max(2) as usize;
            let low = p.f64("quantize", "low").unwrap_or(0.0) as f32;
            let high = p.f64("quantize", "high").unwrap_or(1.0) as f32;
            let step = (high - low) / (count - 1) as f32;
            self.set.extend((0..count).map(|i| low + step * i as f32));
        } else {
            let wire = inp.get("levels").ok_or("`levels` is required when `mode` is `levels`")?;
            self.set.extend(
                wire.as_array()?
                    .as_bytes()
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().expect("four bytes")))
                    .filter(|v| v.is_finite()),
            );
        }
        self.set.sort_by(|a, b| a.total_cmp(b));
        self.set.dedup();
        // A tuning whose window held no peaks is a row of NaN, which is an answer rather than a
        // fault: with nothing to land on, the signal passes and every index is a hole.
        let Some(&base) = self.set.first() else {
            let holes = (0..a.as_bytes().len() / 4).flat_map(|_| f32::NAN.to_le_bytes()).collect();
            out.set("out", d.clone());
            out.set("index", Data::array_f32(a.shape().to_vec(), holes, d.meta().clone()).map_err(|e| e.to_string())?);
            return Ok(());
        };
        let ratio = p.str("quantize", "scale").unwrap_or("ratio") == "ratio";
        let period = p.f64("quantize", "period").unwrap_or(0.0) as f32;
        // Folding needs a period that actually moves a value, and a ratio one needs a positive set.
        let folds = !counted && period.is_finite() && if ratio { period > 1.0 && base > 0.0 } else { period > 0.0 };

        let mut values = Vec::with_capacity(a.as_bytes().len());
        let mut index = Vec::with_capacity(a.as_bytes().len());
        for x in a.as_bytes().chunks_exact(4) {
            let v = f32::from_le_bytes(x.try_into().expect("four bytes"));
            if !v.is_finite() {
                values.extend_from_slice(&v.to_le_bytes());
                index.extend_from_slice(&f32::NAN.to_le_bytes());
                continue;
            }
            // Carry the register aside, match inside one period, then put the register back.
            let (folded, register) = if !folds || (ratio && v <= 0.0) {
                (v, None)
            } else if ratio {
                let (mut inside, mut n) = (v, 0i32);
                while inside >= base * period {
                    inside /= period;
                    n += 1;
                }
                while inside < base {
                    inside *= period;
                    n -= 1;
                }
                (inside, Some(n))
            } else {
                let n = ((v - base) / period).floor();
                (v - n * period, Some(n as i32))
            };
            let i = nearest(&self.set, folded);
            let pulled = match register {
                Some(n) if ratio => self.set[i] * period.powi(n),
                Some(n) => self.set[i] + period * n as f32,
                None => self.set[i],
            };
            values.extend_from_slice(&(v + (pulled - v) * strength).to_le_bytes());
            index.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let shape = a.shape().to_vec();
        out.set("out", Data::array_f32(shape.clone(), values, d.meta().clone()).map_err(|e| e.to_string())?);
        out.set("index", Data::array_f32(shape, index, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "quantize",
        name: "mode",
        spec: ParamSpec::Str { default: "count", options: &["count", "levels"], refresh: false },
        expression: None,
        doc: Some("Where the allowed values come from: counted out below, or wired into `levels`."),
    },
    ParamDecl {
        group: "quantize",
        name: "count",
        spec: ParamSpec::Int { default: 12, min: 2, max: 4096 },
        expression: None,
        doc: Some("How many values are allowed, spread evenly from `low` to `high` inclusive."),
    },
    ParamDecl {
        group: "quantize",
        name: "low",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The first of the counted values."),
    },
    ParamDecl {
        group: "quantize",
        name: "high",
        spec: ParamSpec::Float { default: 1.0, min: -1.0e9, max: 1.0e9 },
        expression: None,
        doc: Some("The last of them."),
    },
    ParamDecl {
        group: "quantize",
        name: "period",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 1.0e6 },
        expression: None,
        doc: Some(
            "How far the wired set repeats, so a scale of a few numbers covers the whole line. \
             2 is the octave. Zero leaves a value outside the set at whichever end is nearest.",
        ),
    },
    ParamDecl {
        group: "quantize",
        name: "scale",
        spec: ParamSpec::Str { default: "ratio", options: &["ratio", "linear"], refresh: false },
        expression: None,
        doc: Some("Whether `period` multiplies — which is what an octave does to a ratio — or adds."),
    },
    ParamDecl {
        group: "quantize",
        name: "strength",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 1.0 },
        expression: None,
        doc: Some("How far towards the allowed value each one is pulled. Zero passes the signal through."),
    },
];
static INPUTS: &[SlotDecl] = &[
    SlotDecl { name: "input", kind: SlotType::Array, trigger_process: true, multi: false, required: true },
    SlotDecl { name: "levels", kind: SlotType::Array, trigger_process: false, multi: false, required: false },
];
static OUTPUTS: &[OutputDecl] = &[
    OutputDecl { name: "out", kind: SlotType::Array },
    OutputDecl { name: "index", kind: SlotType::Array },
];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "Pull every value onto the nearest of an allowed set.\n\
          The set is counted out, or wired in as a tuning's own ratios. `index` says which one \
          each value landed on. Shape and metadata are the input's.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Quantize, MANIFEST);
