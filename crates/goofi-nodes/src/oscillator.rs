//! Oscillator — a phase-accumulator sine generator. Emits `n` float32 samples
//! per tick with `meta["sfreq"]`. A simple free-running generator for bring-up;
//! the drift-free SampleClock-paced version arrives with the audio milestone (M8).

use std::f64::consts::TAU;

use goofi_core::{Data, DType, Meta, Param};
use goofi_core::SlotType;
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Oscillator {
    frequency: f64,
    amplitude: f64,
    sfreq: f64,
    n: usize,
    phase: f64,
}

impl Node for Oscillator {
    fn process(&mut self, _inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let dphi = TAU * self.frequency / self.sfreq;
        let mut buf = Vec::with_capacity(self.n * 4);
        for _ in 0..self.n {
            let s = (self.amplitude * self.phase.sin()) as f32;
            buf.extend_from_slice(&s.to_le_bytes());
            self.phase = (self.phase + dphi).rem_euclid(TAU);
        }
        let meta = Meta {
            sfreq: Some(self.sfreq),
            ..Default::default()
        };
        let data = Data::from_array_bytes(DType::F32, vec![self.n], buf, meta)
            .map_err(|e| e.to_string())?;
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
                    self.sfreq = x.max(1.0);
                }
            }
            ("oscillator", "n_samples") => {
                if let Some(x) = v.as_i64() {
                    self.n = x.max(1) as usize;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("frequency".to_string(), Param::float(10.0, 0.0, 20_000.0));
    g.insert("amplitude".to_string(), Param::float(1.0, 0.0, 1.0e6));
    g.insert("sfreq".to_string(), Param::float(1000.0, 1.0, 1.0e6));
    g.insert("n_samples".to_string(), Param::int(64, 1, 1_000_000));
    let mut groups = ParamGroups::new();
    groups.insert("oscillator".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let f = |name, dflt| param(p, "oscillator", name).and_then(Param::as_f64).unwrap_or(dflt);
    Box::new(Oscillator {
        frequency: f("frequency", 10.0),
        amplitude: f("amplitude", 1.0),
        sfreq: f("sfreq", 1000.0).max(1.0),
        n: param(p, "oscillator", "n_samples")
            .and_then(Param::as_i64)
            .unwrap_or(64)
            .max(1) as usize,
        phase: 0.0,
    })
}

static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];
static INPUTS: &[SlotDecl] = &[];

inventory::submit! {
    NodeManifest {
        type_name: "Oscillator",
        category: "inputs",
        doc: "Generate a sine wave (n float32 samples per tick, meta sfreq).",
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

    #[test]
    fn oscillator_emits_sine_with_sfreq() {
        let m = goofi_node::find("Oscillator").expect("Oscillator registered");
        // freq=1, sfreq=4, n=4 -> phase step TAU/4; samples ~ [0, 1, 0, -1].
        let mut params = (m.default_params)();
        params["oscillator"].insert("frequency".into(), goofi_core::Param::float(1.0, 0.0, 1e6));
        params["oscillator"].insert("sfreq".into(), goofi_core::Param::float(4.0, 1.0, 1e6));
        params["oscillator"].insert("n_samples".into(), goofi_core::Param::int(4, 1, 1_000_000));
        let mut node = (m.make)(&params);

        let inputs_map = IndexMap::new();
        let inp = Inputs::new(&inputs_map);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let data = outbuf.get("out").unwrap().as_ref().unwrap();
        assert_eq!(data.meta().sfreq, Some(4.0));
        if let Value::Array(s) = data.value() {
            assert_eq!(s.dtype(), DType::F32);
            assert_eq!(s.shape(), &[4]);
            let b = s.as_bytes();
            let sample = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
            assert!(sample(0).abs() < 1e-6, "sin(0) ~ 0");
            assert!((sample(1) - 1.0).abs() < 1e-6, "sin(pi/2) ~ 1");
            assert!((sample(3) + 1.0).abs() < 1e-6, "sin(3pi/2) ~ -1");
        } else {
            panic!("expected array");
        }
    }
}
