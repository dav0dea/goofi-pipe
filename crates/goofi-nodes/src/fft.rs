//! FFT — one-sided magnitude spectrum |FFT| of a float32 signal, with the
//! frequency axis in `meta["channels"]["dim0"]`.

use std::collections::BTreeMap;
use std::sync::Arc;

use goofi_core::SlotType;
use goofi_core::{Channels, Coord, Data, DType, Meta, Value};
use goofi_node::{
    Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs, ParamGroups,
    SlotDecl,
};

struct Fft {
    // Kept across ticks so a repeated signal length never re-plans the FFT.
    planner: goofi_dsp::FftPlanner<f32>,
}

impl Node for Fft {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }
        let signal: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if signal.is_empty() {
            return Ok(());
        }
        let n = signal.len();
        let fs = d.meta().sfreq.unwrap_or(1000.0) as f32;
        let mag = goofi_dsp::magnitude_spectrum(&mut self.planner, &signal);

        let buf: Vec<u8> = mag.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut ch = BTreeMap::new();
        ch.insert(
            0usize,
            Arc::new(
                (0..mag.len())
                    .map(|k| Coord::Num((k as f32 * fs / n as f32) as f64))
                    .collect::<Vec<_>>(),
            ),
        );
        let meta = Meta {
            channels: Channels(ch),
            ..Default::default()
        };
        let data = Data::from_array_bytes(DType::F32, vec![mag.len()], buf, meta)
            .map_err(|e| e.to_string())?;
        out.set("spectrum", data);
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    ParamGroups::new()
}
fn make(_: &ParamGroups) -> Box<dyn Node> {
    Box::new(Fft {
        planner: goofi_dsp::FftPlanner::new(),
    })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "spectrum",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "FFT",
        category: "signal",
        doc: "One-sided magnitude spectrum |FFT|; frequency axis in meta channels.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use goofi_core::{Data, DType, Meta, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs};
    use indexmap::IndexMap;

    #[test]
    fn fft_peaks_at_sine_frequency() {
        let m = goofi_node::find("FFT").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let (fs, n, f0) = (64.0f32, 64usize, 8.0f32);
        let sig: Vec<f32> = (0..n).map(|i| (2.0 * PI * f0 * i as f32 / fs).sin()).collect();
        let buf: Vec<u8> = sig.iter().flat_map(|v| v.to_le_bytes()).collect();
        let meta = Meta {
            sfreq: Some(fs as f64),
            ..Default::default()
        };
        let frame = Data::from_array_bytes(DType::F32, vec![n], buf, meta).unwrap();

        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let d = outbuf.get("spectrum").unwrap().as_ref().unwrap();
        if let Value::Array(s) = d.value() {
            assert_eq!(s.shape(), &[n / 2 + 1]);
            let mag: Vec<f32> = s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let peak = mag
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(peak as f32 * fs / n as f32, f0);
        } else {
            panic!("expected array");
        }
    }
}
