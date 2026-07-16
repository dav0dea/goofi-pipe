//! TimeDelayEmbedding — reconstruct a 1-D signal's state space by stacking
//! delayed copies of itself. Each embedding dimension `i` is the signal rolled by
//! `int(i^exponent * delay)` samples; the leading `max_shift` samples (where the
//! circular roll would wrap the tail back to the front) are cropped off every copy
//! so the embedding never mixes wrapped, discontinuous samples. Ported from
//! `nodes/signal/timedelayembedding.py`. The roll is on axis 0 (the sample axis),
//! so the input's `dim0` coordinate labels move to the embedded output's `dim1`.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct TimeDelayEmbedding {
    delay: i64,
    unit: String,
    embedding_dimension: i64,
    moire: bool,
    exponent: f64,
}

impl Node for TimeDelayEmbedding {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("input_array") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }
        // Python rolls axis 0 on a 1-D signal; anything else isn't a valid input.
        let shape = store.shape();
        if shape.len() != 1 {
            return Ok(());
        }
        let n = shape[0];
        if n == 0 {
            return Ok(());
        }

        // embedding_dimension is Int(2,2,100); a below-minimum hand-set value is
        // an invalid param, so no-emit rather than build a degenerate embedding.
        let ed = self.embedding_dimension;
        if ed < 2 {
            return Ok(());
        }
        let edu = ed as usize;

        // "seconds" scales the delay by the frame's sampling rate (samples =
        // seconds * sfreq). A missing sfreq is a no-emit, NOT a raise (Python raises;
        // the Rust port degrades gracefully like the other guards).
        let delay: f64 = if self.unit == "seconds" {
            let Some(sfreq) = d.meta().sfreq else {
                return Ok(());
            };
            (self.delay as f64 * sfreq).trunc() // int(delay * sfreq)
        } else {
            self.delay as f64
        };
        let exponent = self.exponent;

        // max_shift = int((ed-1)^exponent * delay). The [max_shift:] crop drops the
        // wrap region of every rolled copy, so the embedding stays wrap-free. If the
        // crop would empty the array (or the shift is non-finite/huge), no-emit
        // instead of publishing an empty frame.
        let max_shift_f = ((ed - 1) as f64).powf(exponent) * delay;
        if !max_shift_f.is_finite() || max_shift_f < 0.0 || max_shift_f >= n as f64 {
            return Ok(());
        }
        let max_shift = max_shift_f.trunc() as usize;
        let cropped_len = n - max_shift; // >= 1 since max_shift < n

        let a: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // Per-dimension shift, dim 0 being the un-rolled original. Every shift is
        // <= max_shift (the base ed-1 is the largest, powf is monotonic for base>=1),
        // and max_shift < n, so within the cropped region `p - shift` never wraps and
        // stays a valid index — that is exactly what the crop guarantees.
        let shifts: Vec<usize> = (0..edu)
            .map(|dd| {
                if dd == 0 {
                    0
                } else {
                    ((dd as f64).powf(exponent) * delay).trunc() as usize
                }
            })
            .collect();

        // Non-moire: np.array(arrays) -> [ed, cropped_len] (row per dimension).
        // Moire: np.stack(arrays, axis=-1) -> [cropped_len, ed] (column per dimension).
        let mut buf: Vec<u8> = Vec::with_capacity(edu * cropped_len * 4);
        let out_shape = if self.moire {
            for j in 0..cropped_len {
                for &shift in &shifts {
                    buf.extend_from_slice(&a[max_shift + j - shift].to_le_bytes());
                }
            }
            vec![cropped_len, edu]
        } else {
            for &shift in &shifts {
                for j in 0..cropped_len {
                    buf.extend_from_slice(&a[max_shift + j - shift].to_le_bytes());
                }
            }
            vec![edu, cropped_len]
        };

        // The sample axis moved from dim0 to dim1, so its labels would follow it.
        // But cropping shortens that axis (its new length is out_shape[1]), so the
        // input's per-sample dim0 coords (length n) no longer describe it — carry
        // them to dim1 ONLY if they still fit, else drop the now-stale labels. This
        // fixes a latent bug the Python shares (it shifts unconditionally, which its
        // Data constructor then rejects with an AssertionError) — here it would fault
        // every tick on a coord-bearing input; dropping stale coords is correct by
        // construction. sfreq/index/extra always carry through.
        let mut meta = d.meta().clone();
        if let Some(coords) = meta.channels.0.remove(&0) {
            if coords.len() == out_shape[1] {
                meta.channels.0.insert(1, coords);
            }
        }

        let data =
            Data::from_array_bytes(DType::F32, out_shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("embedded_array", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("embedding", "delay") => {
                if let Some(x) = v.as_i64() {
                    self.delay = x;
                }
            }
            ("embedding", "unit") => {
                if let Some(s) = v.as_str() {
                    self.unit = s.to_string();
                }
            }
            ("embedding", "embedding_dimension") => {
                if let Some(x) = v.as_i64() {
                    self.embedding_dimension = x;
                }
            }
            ("embedding", "moire_embedding") => {
                if let Some(b) = v.as_bool() {
                    self.moire = b;
                }
            }
            ("embedding", "exponent") => {
                if let Some(x) = v.as_f64() {
                    self.exponent = x;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("delay".to_string(), Param::int(1, 1, 100));
    g.insert(
        "unit".to_string(),
        Param::Str {
            value: "samples".to_string(),
            options: Some(vec!["samples".to_string(), "seconds".to_string()]),
            refresh: None,
        },
    );
    g.insert("embedding_dimension".to_string(), Param::int(2, 2, 100));
    g.insert("moire_embedding".to_string(), Param::boolean(false));
    g.insert("exponent".to_string(), Param::float(1.0, 0.0, 10.0));
    let mut groups = ParamGroups::new();
    groups.insert("embedding".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let delay = param(p, "embedding", "delay").and_then(Param::as_i64).unwrap_or(1);
    let unit = param(p, "embedding", "unit")
        .and_then(Param::as_str)
        .unwrap_or("samples")
        .to_string();
    let embedding_dimension = param(p, "embedding", "embedding_dimension")
        .and_then(Param::as_i64)
        .unwrap_or(2);
    let moire = param(p, "embedding", "moire_embedding")
        .and_then(Param::as_bool)
        .unwrap_or(false);
    let exponent = param(p, "embedding", "exponent")
        .and_then(Param::as_f64)
        .unwrap_or(1.0);
    Box::new(TimeDelayEmbedding {
        delay,
        unit,
        embedding_dimension,
        moire,
        exponent,
    })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input_array",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "embedded_array",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "TimeDelayEmbedding",
        category: "signal",
        doc: "Time delay embedding: stack delayed copies of a 1-D signal to reconstruct its state space.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Channels, Coord, Data, DType, Meta, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Build the node, apply the given `embedding` params, feed one f32 frame, and
    /// return the output `Data` (None when the node guards / no-emits).
    fn run(params: &[(&str, Param)], shape: Vec<usize>, input: &[f32], meta: Meta) -> Option<Data> {
        let m = goofi_node::find("TimeDelayEmbedding").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        for (name, p) in params {
            node.on_param_changed(&ParamKey::new("embedding", *name), p).unwrap();
        }
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, shape, buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("input_array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        outbuf.get("embedded_array").unwrap().clone()
    }

    fn as_f32(d: &Data) -> (Vec<usize>, Vec<f32>) {
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn non_moire_matches_numpy() {
        // delay=1, ed=3, exponent=1 over [0..5): rows are the original and its two
        // rolls, each cropped by max_shift=2 -> [[2,3,4],[1,2,3],[0,1,2]].
        let d = run(
            &[("delay", Param::int(1, 1, 100)), ("embedding_dimension", Param::int(3, 2, 100))],
            vec![5],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            Meta::empty(),
        )
        .expect("emits");
        assert_eq!(as_f32(&d), (vec![3, 3], vec![2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn moire_stacks_on_last_axis() {
        // Same inputs but moire -> np.stack(axis=-1): shape [cropped_len, ed].
        let d = run(
            &[
                ("delay", Param::int(1, 1, 100)),
                ("embedding_dimension", Param::int(3, 2, 100)),
                ("moire_embedding", Param::boolean(true)),
            ],
            vec![5],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            Meta::empty(),
        )
        .expect("emits");
        assert_eq!(as_f32(&d), (vec![3, 3], vec![2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0]));
    }

    #[test]
    fn seconds_scales_delay_by_sfreq() {
        // unit=seconds, delay=1, sfreq=2 -> effective delay=int(1*2)=2, ed=2.
        // max_shift=2, rows [[2,3,4],[0,1,2]]. sfreq must carry through the meta.
        let mut meta = Meta::empty();
        meta.sfreq = Some(2.0);
        let d = run(
            &[("delay", Param::int(1, 1, 100)), ("unit", Param::str_free("seconds"))],
            vec![5],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            meta,
        )
        .expect("emits");
        assert_eq!(as_f32(&d), (vec![2, 3], vec![2.0, 3.0, 4.0, 0.0, 1.0, 2.0]));
        assert_eq!(d.meta().sfreq, Some(2.0), "sfreq carried through");
    }

    #[test]
    fn seconds_without_sfreq_no_emit() {
        // Python raises here; the port no-emits instead of stalling the graph.
        let d = run(
            &[("delay", Param::int(1, 1, 100)), ("unit", Param::str_free("seconds"))],
            vec![5],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            Meta::empty(),
        );
        assert!(d.is_none(), "no sfreq in 'seconds' mode -> no emit");
    }

    #[test]
    fn non_f32_input_no_emit() {
        let m = goofi_node::find("TimeDelayEmbedding").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let buf: Vec<u8> = [1i32, 2, 3, 4, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::I32, vec![5], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("input_array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("embedded_array").unwrap().is_none(), "non-f32 -> no emit");
    }

    #[test]
    fn non_1d_input_no_emit() {
        let d = run(
            &[("delay", Param::int(1, 1, 100))],
            vec![2, 3],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Meta::empty(),
        );
        assert!(d.is_none(), "2-D input -> no emit");
    }

    #[test]
    fn empty_after_crop_no_emit() {
        // n=3, ed=5, delay=1, exponent=1 -> max_shift=int(4*1)=4 >= 3 -> empty crop.
        let d = run(
            &[("delay", Param::int(1, 1, 100)), ("embedding_dimension", Param::int(5, 2, 100))],
            vec![3],
            &[0.0, 1.0, 2.0],
            Meta::empty(),
        );
        assert!(d.is_none(), "max_shift >= len -> no emit");
    }

    #[test]
    fn below_min_embedding_dimension_no_emit() {
        // embedding_dimension < 2 is below the param minimum -> invalid -> no emit.
        let d = run(
            &[("embedding_dimension", Param::int(1, 2, 100))],
            vec![5],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            Meta::empty(),
        );
        assert!(d.is_none(), "ed < 2 -> no emit");
    }

    #[test]
    fn dim0_channels_move_to_dim1() {
        // moire with ed==n keeps the output's dim1 length == n, so the input's dim0
        // sample labels validate on dim1 after the shift.
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("x".into()), Coord::Str("y".into()), Coord::Str("z".into())]));
        let meta = Meta { channels: Channels(ch), ..Default::default() };
        let d = run(
            &[
                ("delay", Param::int(1, 1, 100)),
                ("embedding_dimension", Param::int(3, 2, 100)),
                ("moire_embedding", Param::boolean(true)),
            ],
            vec![3],
            &[0.0, 1.0, 2.0],
            meta,
        )
        .expect("emits");
        // n=3, max_shift=2, cropped_len=1 -> moire shape [1, 3].
        assert_eq!(as_f32(&d), (vec![1, 3], vec![2.0, 1.0, 0.0]));
        assert!(!d.meta().channels.0.contains_key(&0), "dim0 coords removed");
        let coords = d.meta().channels.0.get(&1).expect("coords now on dim1");
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn stale_channels_are_dropped_not_faulted() {
        // Cropping shortens the sample axis, so length-n dim0 coords no longer fit
        // the output's dim1 (length cropped_len here). The node must DROP them and
        // still emit — not fault every tick (which shifting-unconditionally would,
        // the latent bug the Python shares). n=5, ed=3, delay=1 -> non-moire [3,3].
        let mut ch = BTreeMap::new();
        ch.insert(
            0usize,
            Arc::new((0..5).map(|i| Coord::Num(i as f64)).collect::<Vec<_>>()),
        );
        let meta = Meta { channels: Channels(ch), ..Default::default() };
        let d = run(
            &[("delay", Param::int(1, 1, 100)), ("embedding_dimension", Param::int(3, 2, 100))],
            vec![5],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            meta,
        )
        .expect("emits despite stale coords (no per-tick fault)");
        assert_eq!(as_f32(&d).0, vec![3, 3]);
        assert!(d.meta().channels.0.is_empty(), "stale sample coords dropped, not shifted onto a mismatched axis");
    }
}
