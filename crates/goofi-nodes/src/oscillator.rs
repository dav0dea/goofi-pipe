//! Oscillator — a wall-clock-paced, phase-continuous waveform generator. Each tick
//! it asks a [`SampleClock`](goofi_audio::SampleClock) how many samples have elapsed
//! since the last tick (drift-free against `meta["sfreq"]`) and renders exactly that
//! many with a [`WaveGen`](goofi_audio::WaveGen), carrying phase forward so blocks
//! join click-free. The number of samples per tick therefore tracks *real time*, not
//! the tick count — the faithful audio-generator behavior. The engine stamps the
//! continuity `meta["index"]` (a generator gets a fresh per-emit counter).

use goofi_audio::{SampleClock, WaveGen, Waveform};
use goofi_core::SlotType;
use goofi_core::{Data, DType, Meta, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Oscillator {
    frequency: f64,
    amplitude: f64,
    sfreq: f64,
    clock: SampleClock,
    gen: WaveGen,
    /// Anchored on the first tick so pacing starts from the node's own arrival.
    started: bool,
}

impl Oscillator {
    fn new(frequency: f64, amplitude: f64, sfreq: f64, waveform: Waveform) -> Oscillator {
        Oscillator {
            frequency,
            amplitude,
            sfreq,
            clock: SampleClock::new(sfreq),
            gen: WaveGen::new(waveform),
            started: false,
        }
    }
}

impl Node for Oscillator {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx) -> NodeResult {
        if !self.started {
            self.clock.start(c.now);
            self.started = true;
        }
        // Samples elapsed since the last tick — 0 on the very first tick (now == ref).
        let n = self.clock.advance(c.now) as usize;
        if n == 0 {
            return Ok(()); // nothing elapsed yet; emit no (empty) frame
        }
        // A wired `frequency` control input drives the frequency in real time
        // (its first value); unwired, the `frequency` param holds.
        let freq = first_f32(inp.get("frequency")).map(|v| v as f64).unwrap_or(self.frequency);
        let amp = self.amplitude as f32;
        let buf: Vec<u8> = self
            .gen
            .generate(freq, self.sfreq, n)
            .iter()
            .flat_map(|s| (s * amp).to_le_bytes())
            .collect();
        let meta = Meta {
            sfreq: Some(self.sfreq),
            ..Default::default()
        };
        let data =
            Data::from_array_bytes(DType::F32, vec![n], buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("oscillator", "frequency") => {
                if let Some(x) = v.as_f64() {
                    self.frequency = x;
                }
            }
            ("oscillator", "amplitude") => {
                if let Some(x) = v.as_f64() {
                    self.amplitude = x;
                }
            }
            ("oscillator", "sfreq") => {
                if let Some(x) = v.as_f64() {
                    // The clock's rate is fixed at construction; rebuild and re-anchor
                    // so pacing tracks the new rate from the next tick.
                    self.sfreq = x.max(1.0);
                    self.clock = SampleClock::new(self.sfreq);
                    self.started = false;
                }
            }
            ("oscillator", "waveform") => {
                if let Some(s) = v.as_str() {
                    self.gen.set_waveform(Waveform::parse(s)); // phase-preserving
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// First element of an f32 array `Data`, if the slot holds one (a scalar control).
fn first_f32(d: Option<&Data>) -> Option<f32> {
    match d?.value() {
        Value::Array(s) if s.dtype() == DType::F32 => s
            .as_bytes()
            .get(0..4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        _ => None,
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("frequency".to_string(), Param::float(10.0, 0.0, 20_000.0));
    g.insert("amplitude".to_string(), Param::float(1.0, 0.0, 1.0e6));
    g.insert("sfreq".to_string(), Param::float(1000.0, 1.0, 1.0e6));
    g.insert(
        "waveform".to_string(),
        Param::Str {
            value: "sine".to_string(),
            options: Some(
                ["sine", "square", "sawtooth", "pulse"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            refresh: None,
        },
    );
    let mut groups = ParamGroups::new();
    groups.insert("oscillator".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let f = |name, dflt| param(p, "oscillator", name).and_then(Param::as_f64).unwrap_or(dflt);
    let waveform = param(p, "oscillator", "waveform")
        .and_then(Param::as_str)
        .map(Waveform::parse)
        .unwrap_or(Waveform::Sine);
    Box::new(Oscillator::new(
        f("frequency", 10.0),
        f("amplitude", 1.0),
        f("sfreq", 1000.0).max(1.0),
        waveform,
    ))
}

static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];
// A single non-triggering control input: the oscillator free-runs (wall-clock
// paced) and reads the latest `frequency` each tick rather than being woken by it.
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "frequency",
    kind: SlotType::Array,
    trigger_process: false,
    multi: false,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Oscillator",
        category: "inputs",
        doc: "Wall-clock-paced waveform generator (sine/square/sawtooth/pulse), meta sfreq.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{DType, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs};
    use indexmap::IndexMap;

    fn run_at(
        node: &mut Box<dyn goofi_node::Node>,
        m: &goofi_node::NodeManifest,
        now: f64,
    ) -> Option<Vec<f32>> {
        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx { tick: 0, now };
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        outbuf.get("out").unwrap().as_ref().map(|d| match d.value() {
            Value::Array(s) => s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => panic!("expected array"),
        })
    }

    #[test]
    fn first_tick_emits_nothing_then_paces_by_wall_clock() {
        let m = goofi_node::find("Oscillator").expect("Oscillator registered");
        let mut params = (m.default_params)();
        params["oscillator"].insert("sfreq".into(), goofi_core::Param::float(1000.0, 1.0, 1e6));
        let mut node = (m.make)(&params);
        // First tick anchors the clock at now=0 -> zero elapsed -> no frame.
        assert!(run_at(&mut node, m, 0.0).is_none(), "no time elapsed yet");
        // 10 ms later at 1 kHz -> exactly 10 samples.
        let s = run_at(&mut node, m, 0.010).expect("a paced block");
        assert_eq!(s.len(), 10, "10 ms * 1 kHz = 10 samples");
        // Another 10 ms -> 10 more (drift-free, tracks wall clock).
        assert_eq!(run_at(&mut node, m, 0.020).unwrap().len(), 10);
    }

    #[test]
    fn frequency_control_input_overrides_the_param() {
        use goofi_core::{Data, Meta};
        let m = goofi_node::find("Oscillator").unwrap();
        let mut params = (m.default_params)();
        // Param frequency is a slow 10 Hz; a wired control input of 250 Hz (= sfreq/4)
        // must win, giving the clean quarter-period samples [0, 1, 0, -1].
        params["oscillator"].insert("frequency".into(), goofi_core::Param::float(10.0, 0.0, 1e6));
        params["oscillator"].insert("sfreq".into(), goofi_core::Param::float(1000.0, 1.0, 1e6));
        let mut node = (m.make)(&params);

        let freq = Data::from_array_bytes(DType::F32, vec![1], 250.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("frequency", Some(freq));
        let inp = Inputs::new(&inmap);

        // Anchor at now=0 (empty), then 4 ms -> 4 samples at the control frequency.
        let mut ob0 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut ob0), &mut NodeCtx { tick: 0, now: 0.0 }).unwrap();
        let mut ob1 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut ob1), &mut NodeCtx { tick: 1, now: 0.004 }).unwrap();
        let d = ob1.get("out").unwrap().as_ref().unwrap();
        if let Value::Array(s) = d.value() {
            assert_eq!(s.shape(), &[4]);
            let v = |i: usize| f32::from_le_bytes(s.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap());
            assert!(v(0).abs() < 1e-5 && (v(1) - 1.0).abs() < 1e-5 && (v(3) + 1.0).abs() < 1e-5,
                "control freq 250 Hz -> quarter-period [0,1,0,-1], got [{},{},{},{}]", v(0), v(1), v(2), v(3));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn emits_sine_values_and_sfreq_meta() {
        let m = goofi_node::find("Oscillator").unwrap();
        let mut params = (m.default_params)();
        // freq = sfreq/4 -> phase step π/2 -> samples 0, 1, 0, -1 (× amplitude).
        params["oscillator"].insert("frequency".into(), goofi_core::Param::float(250.0, 0.0, 1e6));
        params["oscillator"].insert("sfreq".into(), goofi_core::Param::float(1000.0, 1.0, 1e6));
        params["oscillator"].insert("amplitude".into(), goofi_core::Param::float(2.0, 0.0, 1e6));
        let mut node = (m.make)(&params);

        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx { tick: 0, now: 0.0 })
            .unwrap();
        // 4 ms at 1 kHz -> 4 samples.
        let mut outbuf2 = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf2), &mut NodeCtx { tick: 1, now: 0.004 })
            .unwrap();
        let d = outbuf2.get("out").unwrap().as_ref().unwrap();
        assert_eq!(d.meta().sfreq, Some(1000.0));
        if let Value::Array(s) = d.value() {
            assert_eq!(s.dtype(), DType::F32);
            assert_eq!(s.shape(), &[4]);
            let v = |i: usize| f32::from_le_bytes(s.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap());
            assert!(v(0).abs() < 1e-5, "sin(0)*2 ~ 0");
            assert!((v(1) - 2.0).abs() < 1e-5, "sin(pi/2)*2 ~ 2");
            assert!((v(3) + 2.0).abs() < 1e-5, "sin(3pi/2)*2 ~ -2");
        } else {
            panic!("expected array");
        }
    }
}
