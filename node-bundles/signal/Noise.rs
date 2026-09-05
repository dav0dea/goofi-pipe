//! Noise — uniform, normal or pink, one independent stream per channel, paced like the LFO: a
//! sample per update, or the block real time advanced by.

use goofi_core::{Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, ParamKey, Params, ParamSpec, Tag};

/// One channel's generator state: a 64-bit stream, plus the poles pink noise needs.
#[derive(Clone, Copy)]
struct Source {
    state: u64,
    pink: [f64; 3],
    /// The second of a Box-Muller pair, held for the next normal sample.
    spare: Option<f64>,
}

impl Source {
    fn new(seed: u64) -> Source {
        Source { state: seed | 1, pink: [0.0; 3], spare: None }
    }

    /// xorshift64*, which needs no dependency and is far beyond what a noise source asks of it.
    fn next(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        (self.state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }

    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let (u, v) = (self.next().max(1e-12), self.next());
        let (r, theta) = ((-2.0 * u.ln()).sqrt(), std::f64::consts::TAU * v);
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Three one-pole filters over white noise: about 1/f across the audible decades.
    fn pink(&mut self) -> f64 {
        let w = self.uniform();
        self.pink[0] = 0.99765 * self.pink[0] + w * 0.0990460;
        self.pink[1] = 0.96300 * self.pink[1] + w * 0.2965164;
        self.pink[2] = 0.57000 * self.pink[2] + w * 1.0526913;
        (self.pink[0] + self.pink[1] + self.pink[2] + w * 0.1848) * 0.25
    }

    fn sample(&mut self, mode: &str) -> f64 {
        match mode {
            "normal" => self.normal(),
            "pink" => self.pink(),
            _ => self.uniform(),
        }
    }
}

#[derive(Default)]
struct Noise {
    sources: Vec<Source>,
    /// The seed the sources were built from, so a change rebuilds them.
    seeded: i64,
    start: Option<f64>,
    emitted: u64,
}

impl Noise {
    fn reseed(&mut self, channels: usize, seed: i64) {
        let base = if seed < 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9e37_79b9_7f4a_7c15)
        } else {
            seed as u64
        };
        // Each channel walks the golden-ratio stride off the base, so no two streams line up.
        self.sources =
            (0..channels).map(|c| Source::new(base.wrapping_add((c as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)))).collect();
        self.seeded = seed;
    }
}

impl Node for Noise {
    fn process(
        &mut self,
        _inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let channels = p.i64("output", "channels").unwrap_or(1).clamp(1, 64) as usize;
        let seed = p.i64("noise", "seed").unwrap_or(-1);
        if self.sources.len() != channels || self.seeded != seed {
            self.reseed(channels, seed);
        }
        let amp = p.f64("noise", "amplitude").unwrap_or(1.0);
        let offset = p.f64("noise", "offset").unwrap_or(0.0);
        if !amp.is_finite() || !offset.is_finite() {
            return Err(format!("non-finite drive: amplitude={amp}, offset={offset}").into());
        }
        let mode = p.str("noise", "mode").unwrap_or("uniform");
        let sfreq = p.f64("output", "sfreq").unwrap_or(250.0).max(1.0);

        if p.str("output", "mode").unwrap_or("value") != "block" {
            let mut buf = Vec::with_capacity(channels * 4);
            for s in &mut self.sources {
                buf.extend_from_slice(&((s.sample(mode) * amp + offset) as f32).to_le_bytes());
            }
            out.set("out", Data::array_f32(vec![channels], buf, Meta::new()).map_err(|e| e.to_string())?);
            return Ok(());
        }

        let start = *self.start.get_or_insert(c.now);
        let total = (sfreq * (c.now - start)).round().max(0.0) as u64;
        let n = total.saturating_sub(self.emitted) as usize;
        if n == 0 {
            return Ok(());
        }
        self.emitted = total;
        let mut buf = Vec::with_capacity(channels * n * 4);
        for s in &mut self.sources {
            for _ in 0..n {
                buf.extend_from_slice(&((s.sample(mode) * amp + offset) as f32).to_le_bytes());
            }
        }
        let meta = Meta::new().with_sfreq(Some(sfreq));
        out.set("out", Data::array_f32(vec![channels, n], buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, _v: &goofi_core::Param) -> NodeResult {
        if key.group == "output" {
            self.start = None;
            self.emitted = 0;
        }
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "noise",
        name: "mode",
        spec: ParamSpec::Str { default: "uniform", options: &["uniform", "normal", "pink"], refresh: false },
        expression: None,
        doc: Some(
            "How the samples are distributed: `uniform` fills the range evenly, `normal` clusters \
             around the offset, `pink` weights the low frequencies as living signals do.",
        ),
    },
    ParamDecl {
        group: "noise",
        name: "amplitude",
        spec: ParamSpec::Float { default: 1.0, min: -1.0e6, max: 1.0e6 },
        expression: None,
        doc: Some("Scales every sample, so the noise spans minus this to plus this."),
    },
    ParamDecl {
        group: "noise",
        name: "offset",
        spec: ParamSpec::Float { default: 0.0, min: -1.0e6, max: 1.0e6 },
        expression: None,
        doc: Some("Added to every sample, so the noise sits around a value other than zero."),
    },
    ParamDecl {
        group: "noise",
        name: "seed",
        spec: ParamSpec::Int { default: -1, min: -1, max: i32::MAX as i64 },
        expression: None,
        doc: Some("Fixes the stream so a patch replays the same noise; -1 takes a fresh one."),
    },
    ParamDecl {
        group: "output",
        name: "mode",
        spec: ParamSpec::Str { default: "value", options: &["value", "block"], refresh: false },
        expression: None,
        doc: Some(
            "`value` emits one sample per channel per update, which a param reference reads as a \
             number; `block` emits the samples elapsed at `sfreq`, which is a signal.",
        ),
    },
    ParamDecl {
        group: "output",
        name: "sfreq",
        spec: ParamSpec::Float { default: 250.0, min: 1.0, max: 10_000.0 },
        expression: None,
        doc: Some("Sample rate within an emitted block, in Hz. `value` mode ignores it."),
    },
    ParamDecl {
        group: "output",
        name: "channels",
        spec: ParamSpec::Int { default: 1, min: 1, max: 64 },
        expression: None,
        doc: Some("How many independent noise streams to emit, one per channel."),
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Generator],
    doc: "Uniform, normal or pink noise, one independent stream per channel.",
    inputs: &[],
    outputs: OUTPUTS,
    params: PARAMS,
    producer: true,
};

goofi_signal_sdk::export!(Noise, MANIFEST);
