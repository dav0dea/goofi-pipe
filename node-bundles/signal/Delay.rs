//! Delay — the stream, later. In samples it reads back through the stitched past, so any chunking
//! gives one answer; in updates it holds whole frames and hands back the one from further back.

use goofi_core::{resolve_axis, stream, Data, SlotType, Stream};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Delay {
    past: Stream,
    /// Whole frames, for the units that count updates rather than samples.
    frames: std::collections::VecDeque<Data>,
}

impl Node for Delay {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let size = p.f64("delay", "size").unwrap_or(10.0).max(0.0);
        let unit = p.str("delay", "unit").unwrap_or("samples");
        let sfreq = d.meta().sfreq();

        // Seconds are samples when the frame says how fast it runs, and updates when it does not.
        let by_samples = match unit {
            "samples" => true,
            "seconds" => sfreq.is_some(),
            _ => false,
        };
        if !by_samples {
            let back = match (unit, sfreq) {
                ("seconds", _) => {
                    let ufreq = d.meta().get("ufreq").and_then(|v| match v {
                        goofi_core::MetaValue::Float(f) => Some(*f),
                        goofi_core::MetaValue::Int(i) => Some(*i as f64),
                        _ => None,
                    });
                    (size * ufreq.unwrap_or(1.0)).round().max(0.0) as usize
                }
                _ => size.round() as usize,
            };
            self.frames.push_back(d.clone());
            while self.frames.len() > back + 1 {
                self.frames.pop_front();
            }
            let held = self.frames.front().expect("the frame just pushed").clone();
            out.set("out", held);
            return Ok(());
        }

        let dim = resolve_axis(p.i64("delay", "axis").unwrap_or(-1), a.shape().len())?;
        let back = match unit {
            "seconds" => (size * sfreq.expect("seconds with a rate")).round().max(0.0) as usize,
            _ => size.round() as usize,
        };
        let n = a.shape()[dim];
        let (shape, stitched, at) = self.past.push(a.shape(), dim, a.as_bytes(), back + n);
        // Before the stream begins there is nothing to read back, so the earliest sample stands in.
        let shifted: Vec<Vec<f32>> = stream::lanes(&shape, dim, &stitched)
            .iter()
            .map(|lane| {
                let last = lane.len() as i64 - 1;
                (0..n).map(|j| lane[(((at + j) as i64) - back as i64).clamp(0, last) as usize]).collect()
            })
            .collect();
        let buf = stream::unlanes(a.shape(), dim, &shifted);
        out.set("out", Data::array_f32(a.shape().to_vec(), buf, d.meta().clone()).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn on_pulse(&mut self, _key: &ParamKey, _p: &Params<'_>) -> NodeResult {
        self.past.reset();
        self.frames.clear();
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "delay",
        name: "size",
        spec: ParamSpec::Float { default: 10.0, min: 0.0, max: 1.0e7 },
        expression: None,
        doc: Some("How far back to read, in the unit below. Never a sleep: the node answers at once."),
    },
    ParamDecl {
        group: "delay",
        name: "unit",
        spec: ParamSpec::Str { default: "samples", options: &["samples", "seconds", "updates"], refresh: false },
        expression: None,
        doc: Some(
            "What `size` counts. `seconds` reads as samples on a frame that carries a rate, and as \
             updates on one that does not.",
        ),
    },
    ParamDecl {
        group: "delay",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis the delay runs along when it counts samples. -1 is time."),
    },
    ParamDecl {
        group: "delay",
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
    doc: "The stream as it was a moment ago, counted in samples, seconds or updates.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Delay, MANIFEST);
