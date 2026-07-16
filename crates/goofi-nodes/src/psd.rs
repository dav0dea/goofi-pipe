//! PSD — one-sided power spectral density (Hann periodogram) of a float32 signal.
//! Reads `sfreq` from the input meta; emits power with the frequency axis in
//! `meta["channels"]["dim0"]` so viewers label the x-axis in Hz.

use std::collections::BTreeMap;
use std::sync::Arc;

use goofi_core::SlotType;
use goofi_core::{Channels, Coord, Data, DType, Meta, Value};
use goofi_node::{
    Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs, ParamGroups,
    SlotDecl,
};

struct Psd;

impl Node for Psd {
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
        if signal.len() < 2 {
            return Ok(());
        }
        let fs = d.meta().sfreq.unwrap_or(1000.0) as f32;

        let (freqs, power) = goofi_dsp::psd_periodogram(&signal, fs);

        let buf: Vec<u8> = power.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut ch = BTreeMap::new();
        ch.insert(
            0usize,
            Arc::new(freqs.into_iter().map(|f| Coord::Num(f as f64)).collect::<Vec<_>>()),
        );
        let meta = Meta {
            channels: Channels(ch),
            ..Default::default()
        };
        let data = Data::from_array_bytes(DType::F32, vec![power.len()], buf, meta)
            .map_err(|e| e.to_string())?;
        out.set("psd", data);
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    ParamGroups::new()
}

fn make(_: &ParamGroups) -> Box<dyn Node> {
    Box::new(Psd)
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "psd",
    kind: SlotType::Array,
    length_preserving: false,
}];

inventory::submit! {
    NodeManifest {
        type_name: "PSD",
        category: "signal",
        doc: "One-sided power spectral density (Hann periodogram); freq axis in meta channels.",
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
    fn psd_emits_power_with_freq_axis() {
        let m = goofi_node::find("PSD").unwrap();
        let mut node = (m.make)(&(m.default_params)());

        let fs = 128.0f32;
        let n = 128usize;
        let f0 = 16.0f32;
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
        let d = outbuf.get("psd").unwrap().as_ref().unwrap();
        // frequency axis present and matching length
        let coords = &d.meta().channels.0[&0];
        if let Value::Array(s) = d.value() {
            assert_eq!(s.shape(), &[n / 2 + 1]);
            assert_eq!(coords.len(), n / 2 + 1);
            // peak power near f0
            let power: Vec<f32> = s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let peak = power
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let peak_hz = peak as f32 * fs / n as f32;
            assert!((peak_hz - f0).abs() <= 1.0, "peak at {peak_hz} Hz, want ~{f0}");
        } else {
            panic!("expected array");
        }
    }
}
