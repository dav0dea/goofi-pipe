//! Filter — a streaming Butterworth filter along one axis, one delay line per signal.
//! Stateful rather than per-frame: a filter restarted on each block would ring at every boundary.

use goofi_core::{resolve_axis, Data, SlotType};
use goofi_node::{NodeManifest, OutputDecl, ParamDecl, ParamSpec, Params, SlotDecl};
use goofi_signal::{default_factory, NodeClass};
use goofi_signal_sdk::{Inputs, Node, NodeCtx, NodeResult, Outputs};

/// One second-order section, normalized so `a0 == 1`.
#[derive(Clone, Copy, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Low,
    High,
    Notch,
}

impl Biquad {
    /// An RBJ section at `f0` Hz with quality `q`, for a stream sampled at `sfreq`.
    fn new(kind: Kind, f0: f64, q: f64, sfreq: f64) -> Biquad {
        // A cutoff at or past Nyquist makes `alpha` zero or negative, and the section unstable.
        let f0 = f0.clamp(sfreq * 1e-6, sfreq * 0.49);
        let w0 = std::f64::consts::TAU * f0 / sfreq;
        let (cos, alpha) = (w0.cos(), w0.sin() / (2.0 * q));
        let (b0, b1, b2) = match kind {
            Kind::Low => ((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0),
            Kind::High => ((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0),
            Kind::Notch => (1.0, -2.0 * cos, 1.0),
        };
        let a0 = 1.0 + alpha;
        Biquad {
            b0: (b0 / a0) as f32,
            b1: (b1 / a0) as f32,
            b2: (b2 / a0) as f32,
            a1: (-2.0 * cos / a0) as f32,
            a2: ((1.0 - alpha) / a0) as f32,
        }
    }
}

/// The Butterworth cascade of `order` (rounded up to even) at `f0`: each section takes a pole's Q.
fn butterworth(kind: Kind, f0: f64, order: usize, sfreq: f64, into: &mut Vec<Biquad>) {
    let n = order.max(2).div_ceil(2) * 2;
    for k in 0..n / 2 {
        let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * n) as f64;
        into.push(Biquad::new(kind, f0, 1.0 / (2.0 * theta.cos()), sfreq));
    }
}

#[derive(Default)]
struct Filter {
    sections: Vec<Biquad>,
    /// Two delay elements per (signal, section), laid out signal-major.
    state: Vec<[f32; 2]>,
    layout: (Vec<usize>, usize, usize),
}

impl Node for Filter {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("data").ok_or("`data` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let axis = resolve_axis(p.i64("filter", "axis").unwrap_or(-1), a.ndim())?;
        let sfreq = d.meta().sfreq().ok_or(
            "no sfreq on the incoming frame: a cutoff in Hz needs to know the sample rate",
        )?;

        let mode = p.str("filter", "mode").unwrap_or("bandpass");
        let (low, high) = (p.f64("filter", "low").unwrap_or(1.0), p.f64("filter", "high").unwrap_or(40.0));
        let order = p.i64("filter", "order").unwrap_or(4).max(2) as usize;
        if matches!(mode, "bandpass" | "notch") && low >= high {
            return Err(format!("{mode} needs low < high, got {low} and {high}").into());
        }
        self.sections.clear();
        match mode {
            "lowpass" => butterworth(Kind::Low, high, order, sfreq, &mut self.sections),
            "highpass" => butterworth(Kind::High, low, order, sfreq, &mut self.sections),
            "notch" => {
                // The RBJ notch takes a centre and a quality; this is the pair that means low..high.
                let centre = (low * high).sqrt();
                for _ in 0..order.div_ceil(2) {
                    self.sections.push(Biquad::new(Kind::Notch, centre, centre / (high - low), sfreq));
                }
            }
            _ => {
                butterworth(Kind::High, low, order, sfreq, &mut self.sections);
                butterworth(Kind::Low, high, order, sfreq, &mut self.sections);
            }
        }

        let shape = a.shape();
        let inner: usize = shape[axis + 1..].iter().product();
        let outer: usize = shape[..axis].iter().product();
        let along = shape[axis];
        let signals = outer * inner;
        let layout = (shape.to_vec(), axis, self.sections.len());
        if layout != self.layout {
            self.state = vec![[0.0; 2]; signals * self.sections.len()];
            self.layout = layout;
        }

        let mut buf: Vec<f32> = a
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
            .collect();
        for o in 0..outer {
            for i in 0..inner {
                let base = o * along * inner + i;
                let st = &mut self.state[(o * inner + i) * self.sections.len()..][..self.sections.len()];
                for k in 0..along {
                    let mut x = buf[base + k * inner];
                    // Direct form II transposed, which stays accurate in f32 at low cutoffs.
                    for (s, z) in self.sections.iter().zip(st.iter_mut()) {
                        let y = s.b0 * x + z[0];
                        z[0] = s.b1 * x - s.a1 * y + z[1];
                        z[1] = s.b2 * x - s.a2 * y;
                        x = y;
                    }
                    buf[base + k * inner] = x;
                }
            }
        }

        let bytes = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.set(
            "out",
            Data::array_f32(shape.to_vec(), bytes, d.meta().clone()).map_err(|e| e.to_string())?,
        );
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "filter",
        name: "mode",
        spec: ParamSpec::Str {
            default: "bandpass",
            options: &["bandpass", "lowpass", "highpass", "notch"],
            refresh: false,
        },
        expression: None,
        doc: Some(
            "Which band to keep. `bandpass` keeps low..high, `notch` removes it; `lowpass` uses \
             `high` as its cutoff and `highpass` uses `low`.",
        ),
    },
    ParamDecl {
        group: "filter",
        name: "low",
        spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 10_000.0 },
        expression: None,
        doc: Some("Lower edge in Hz. Unused by `lowpass`."),
    },
    ParamDecl {
        group: "filter",
        name: "high",
        spec: ParamSpec::Float { default: 40.0, min: 0.0, max: 10_000.0 },
        expression: None,
        doc: Some("Upper edge in Hz. Unused by `highpass`."),
    },
    ParamDecl {
        group: "filter",
        name: "order",
        spec: ParamSpec::Int { default: 4, min: 2, max: 16 },
        expression: None,
        doc: Some(
            "How steeply the band ends, in poles. Rounded up to an even number, because the \
             cascade is built from second-order sections.",
        ),
    },
    ParamDecl {
        group: "filter",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis to filter along, negative from the end. -1 is time."),
    },
];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeClass {
        manifest: NodeManifest {
            type_name: "Filter",
            category: "signal",
            doc: "Streaming Butterworth bandpass/lowpass/highpass/notch along one axis, shape unchanged.",
            inputs: INPUTS,
            outputs: OUTPUTS,
            params: PARAMS,
            producer: false,
        },
        isolation: &goofi_node::NATIVE,
        factory: default_factory::<Filter>,
    }
}
