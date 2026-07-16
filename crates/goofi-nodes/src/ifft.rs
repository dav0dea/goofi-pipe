//! IFFT — reconstruct a real time-domain signal from a magnitude spectrum and
//! its phase via the inverse FFT. Ported from the Python `nodes/signal/ifft.py`:
//! builds the complex spectrum `spectrum * exp(i*phase)`, inverse-transforms it,
//! and emits the real part. The two inputs may differ in length; the shorter is
//! zero-padded to the longer (numpy `np.concatenate`) before transforming. Output
//! carries the phase input's metadata, matching the Python node.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Value};
use goofi_node::{
    Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs, ParamGroups,
    SlotDecl,
};

struct Ifft {
    // Kept across ticks so a repeated signal length never re-plans the FFT.
    planner: goofi_dsp::FftPlanner<f32>,
}

/// Read a slot's array as a flat f32 vector, or `None` unless it is an F32 array
/// (matches the repo's silent-no-op contract for absent/wrong-typed inputs).
fn as_f32(d: Option<&Data>) -> Option<Vec<f32>> {
    let Value::Array(store) = d?.value() else {
        return None;
    };
    if store.dtype() != DType::F32 {
        return None;
    }
    Some(
        store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

impl Node for Ifft {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        // Both inputs are required; a missing/wrong-typed one is a no-op no-emit.
        let (Some(spectrum), Some(phase_d)) = (as_f32(inp.get("spectrum")), inp.get("phase")) else {
            return Ok(());
        };
        let Some(phase) = as_f32(Some(phase_d)) else {
            return Ok(());
        };

        let recon = goofi_dsp::inverse_fft_real(&mut self.planner, &spectrum, &phase);
        let n = recon.len();
        if n == 0 {
            return Ok(());
        }

        let buf: Vec<u8> = recon.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Carry the phase input's meta (Python returns `phase.meta`). The output is
        // a 1-D length-`n` array; drop any channel axis whose length no longer fits
        // (zero-padding can change the length) so strict Data construction never
        // rejects a valid reconstruction.
        let mut meta = phase_d.meta().clone();
        meta.channels.0.retain(|&dim, coords| dim == 0 && coords.len() == n);
        let data = Data::from_array_bytes(DType::F32, vec![n], buf, meta).map_err(|e| e.to_string())?;
        out.set("reconstructed", data);
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    ParamGroups::new()
}

fn make(_: &ParamGroups) -> Box<dyn Node> {
    Box::new(Ifft {
        planner: goofi_dsp::FftPlanner::new(),
    })
}

static INPUTS: &[SlotDecl] = &[
    SlotDecl {
        name: "spectrum",
        kind: SlotType::Array,
        trigger_process: true,
    },
    SlotDecl {
        name: "phase",
        kind: SlotType::Array,
        trigger_process: true,
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "reconstructed",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "IFFT",
        category: "signal",
        doc: "Inverse FFT: reconstruct a real time signal from a magnitude spectrum + phase.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::f32::consts::PI;
    use std::sync::Arc;

    use goofi_core::{Channels, Coord, Data, DType, Meta, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs};
    use indexmap::IndexMap;

    fn arr(v: &[f32], meta: Meta) -> Data {
        let buf: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, vec![v.len()], buf, meta).unwrap()
    }

    fn vals(d: &Data) -> Vec<f32> {
        match d.value() {
            Value::Array(s) => s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => panic!("expected array"),
        }
    }

    fn run(spectrum: Option<Data>, phase: Option<Data>) -> Option<Data> {
        let m = goofi_node::find("IFFT").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("spectrum", spectrum);
        inmap.insert("phase", phase);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        outbuf.get("reconstructed").unwrap().clone()
    }

    #[test]
    fn reconstructs_cosine_from_mag_and_phase() {
        // A real cosine at bin k0 has a full DFT that is n/2 at bins k0 and n-k0
        // (real, so phase 0) and zero elsewhere. Feeding that magnitude/phase back
        // through the IFFT must recover cos(2*pi*k0*j/n) exactly.
        let n = 8usize;
        let k0 = 1usize;
        let mut mag = vec![0f32; n];
        mag[k0] = n as f32 / 2.0;
        mag[n - k0] = n as f32 / 2.0;
        let phase = vec![0f32; n];

        let out = run(Some(arr(&mag, Meta::empty())), Some(arr(&phase, Meta::empty())))
            .expect("both inputs present -> emits");
        if let Value::Array(s) = out.value() {
            assert_eq!(s.shape(), &[n], "reconstruction length matches the spectrum");
        }
        for (j, &g) in vals(&out).iter().enumerate() {
            let want = (2.0 * PI * k0 as f32 * j as f32 / n as f32).cos();
            assert!((g - want).abs() < 1e-5, "sample {j}: {g} != {want}");
        }
    }

    #[test]
    fn missing_or_wrong_typed_input_does_not_emit() {
        let real = arr(&[0.0, 0.0], Meta::empty());
        assert!(run(Some(real.clone()), None).is_none(), "no phase -> no emit");
        assert!(run(None, Some(real.clone())).is_none(), "no spectrum -> no emit");
        assert!(run(None, None).is_none(), "nothing -> no emit");

        // A non-F32 spectrum is a silent no-op (guarded, not a panic).
        let i32buf: Vec<u8> = [1i32, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let ints = Data::from_array_bytes(DType::I32, vec![2], i32buf, Meta::empty()).unwrap();
        assert!(run(Some(ints), Some(real)).is_none(), "non-f32 spectrum -> no emit");
    }

    #[test]
    fn carries_phase_meta_but_drops_stale_axis_when_padded() {
        // phase is shorter than spectrum -> zero-padded to length 4. Its 2-coord
        // freq axis no longer fits the length-4 output, so it is dropped; sfreq
        // (a scalar, length-agnostic) still carries through, as Python returns
        // `phase.meta` wholesale.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Num(0.0), Coord::Num(64.0)]));
        let phase_meta = Meta { sfreq: Some(128.0), channels: Channels(ch), ..Default::default() };
        let phase = arr(&[0.0, 0.0], phase_meta);
        let spectrum = arr(&[0.0, 4.0, 0.0, 0.0], Meta::empty());

        let out = run(Some(spectrum), Some(phase)).expect("emits");
        if let Value::Array(s) = out.value() {
            assert_eq!(s.shape(), &[4], "output length follows the longer input");
        }
        assert_eq!(out.meta().sfreq, Some(128.0), "scalar sfreq carried from phase");
        assert!(out.meta().channels.0.is_empty(), "2-coord axis dropped for length-4 output");
    }

    #[test]
    fn keeps_phase_freq_axis_when_length_matches() {
        // No padding: a length-4 phase axis fits the length-4 output and is kept
        // (faithful to Python attaching phase.meta unchanged).
        let mut ch = BTreeMap::new();
        ch.insert(
            0usize,
            Arc::new(vec![Coord::Num(0.0), Coord::Num(1.0), Coord::Num(2.0), Coord::Num(3.0)]),
        );
        let phase = arr(&[0.0f32; 4], Meta { channels: Channels(ch), ..Default::default() });
        let spectrum = arr(&[0.0f32; 4], Meta::empty());

        let out = run(Some(spectrum), Some(phase)).expect("emits");
        assert_eq!(
            out.meta().channels.0.get(&0).map(|c| c.len()),
            Some(4),
            "matching-length freq axis is preserved"
        );
    }
}
