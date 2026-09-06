//! Clock — a steady tick the whole patch can share. `rate` is read as beats a minute, as hertz or
//! as seconds a tick, and `division` cuts each one into as many as you want, so the same node is a
//! sixteenth-note grid, a 40 Hz strobe or a reading taken every five minutes.
//!
//! `pulse` is one on an update a tick landed in and zero otherwise, which is what a pulse param
//! references. It cannot say that TWO landed: a clock faster than the patch's update rate is read
//! through `count`, which never skips one.

use goofi_core::{Data, Meta, SlotType};
use goofi_signal_sdk::{ExprDecl, ExprMode, Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, Tag};

/// Seconds between ticks, or `None` when the settings name no rate at all.
fn interval(rate: f64, unit: &str, division: f64) -> Option<f64> {
    let per_tick = match unit {
        "hz" => 1.0 / (rate * division),
        "seconds" => rate / division,
        _ => 60.0 / (rate * division),
    };
    (per_tick.is_finite() && per_tick > 0.0).then_some(per_tick)
}

#[derive(Default)]
struct Clock {
    /// `ctx.now` at the last update, which is what elapsed time is measured from.
    last: Option<f64>,
    /// How far into the current tick, in ticks, in `[0, 1)`.
    phase: f64,
    /// Ticks since the last reset.
    count: u64,
}

impl Node for Clock {
    fn process(
        &mut self,
        _inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let rate = p.f64("clock", "rate").unwrap_or(120.0);
        let unit = p.str("clock", "unit").unwrap_or("bpm");
        let division = p.f64("clock", "division").unwrap_or(1.0);
        let running = p.bool("clock", "running").unwrap_or(true);
        let every =
            interval(rate, unit, division).ok_or_else(|| format!("no tick to run: rate={rate} {unit}, division={division}"))?;

        let elapsed = c.now - self.last.unwrap_or(c.now);
        self.last = Some(c.now);
        // A stopped clock holds its place rather than rewinding it, so starting again resumes.
        let ticks = if running {
            // A rate far above the update rate would otherwise count more ticks than a u64 holds.
            let advanced = (self.phase + elapsed / every).min(u32::MAX as f64);
            self.phase = advanced.fract();
            advanced.trunc() as u64
        } else {
            0
        };
        self.count += ticks;

        let one = |v: f32| Data::array_f32(vec![1], v.to_le_bytes().to_vec(), Meta::new()).map_err(|e| e.to_string());
        out.set("pulse", one(if ticks > 0 { 1.0 } else { 0.0 })?);
        out.set("phase", one(self.phase as f32)?);
        out.set("count", one(self.count as f32)?);
        Ok(())
    }

    fn on_pulse(&mut self, _key: &ParamKey, _p: &Params<'_>) -> NodeResult {
        self.phase = 0.0;
        self.count = 0;
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "clock",
        name: "rate",
        spec: ParamSpec::Float { default: 120.0, min: 0.0, max: 1.0e6 },
        expression: None,
        doc: Some("How fast, read in whatever `unit` says."),
    },
    ParamDecl {
        group: "clock",
        name: "unit",
        spec: ParamSpec::Str { default: "bpm", options: &["bpm", "hz", "seconds"], refresh: false },
        expression: None,
        doc: Some("What `rate` means: beats a minute, ticks a second, or seconds between ticks."),
    },
    ParamDecl {
        group: "clock",
        name: "division",
        spec: ParamSpec::Float { default: 1.0, min: 0.001, max: 1000.0 },
        expression: None,
        doc: Some("How many ticks to cut each one into. At 120 bpm, 4 is sixteenth notes."),
    },
    ParamDecl {
        group: "clock",
        name: "running",
        spec: ParamSpec::Bool { default: true },
        expression: None,
        doc: Some("Off holds the place it had reached, so starting again resumes rather than restarts."),
    },
    ParamDecl {
        group: "clock",
        name: "reset",
        spec: ParamSpec::Pulse,
        expression: None,
        doc: Some("Put the phase and the count back to zero."),
    },
    // A manifest's own `common.*` is never overwritten by the universal declaration, and a clock
    // can tick no finer than the rate it is asked at.
    ParamDecl {
        group: "common",
        name: "max_frequency",
        spec: ParamSpec::Float { default: 30.0, min: 0.0, max: 1000.0 },
        expression: Some(ExprDecl { source: "globals.system.default_ufreq", mode: ExprMode::On, trigger: true }),
        doc: Some(
            "How many frames a second to emit. Bound to the patch's `default_ufreq` global, so \
             editing that global re-rates every generator at once.",
        ),
    },
];
static OUTPUTS: &[OutputDecl] = &[
    OutputDecl { name: "pulse", kind: SlotType::Array },
    OutputDecl { name: "phase", kind: SlotType::Array },
    OutputDecl { name: "count", kind: SlotType::Array },
];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Generator, Tag::Control],
    doc: "A steady tick, in beats a minute, hertz or seconds.\n\
          `pulse` is one on an update a tick landed in, `phase` where it is inside the current \
          tick, and `count` the ticks since the last reset.",
    inputs: &[],
    outputs: OUTPUTS,
    params: PARAMS,
    producer: true,
};

goofi_signal_sdk::export!(Clock, MANIFEST);
