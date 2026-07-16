//! Compass — the turning of the difference vector (pole2 - pole1) read off as a
//! sequence of compass angles. For each adjacent pair of dimensions it takes the
//! heading `atan2(diff[i+1], diff[i])`, in degrees, normalized to [0, 360). Two
//! 1-D length-N inputs yield N-1 angles. Ported from `nodes/analysis/compass.py`;
//! the Python raises on a shape mismatch — here (matching the repo's silent-no-op
//! idiom) any missing/ill-typed/mismatched input simply emits nothing.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Meta, Value};
use goofi_node::{
    Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs, ParamGroups,
    SlotDecl,
};

struct Compass;

/// Decode a length-preserving f32 array store into a Vec.
fn as_f32(store: &goofi_core::ArrayStore) -> Vec<f32> {
    store
        .as_bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

impl Node for Compass {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        // Both poles are required; either absent -> no-op (Python returns None).
        let (Some(p1), Some(p2)) = (inp.get("pole1"), inp.get("pole2")) else {
            return Ok(());
        };
        let (Value::Array(s1), Value::Array(s2)) = (p1.value(), p2.value()) else {
            return Ok(());
        };
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            return Ok(());
        }
        // Python demands equal-length 1-D arrays (else ValueError); we no-op instead.
        if s1.shape().len() != 1 || s2.shape().len() != 1 || s1.shape()[0] != s2.shape()[0] {
            return Ok(());
        }
        let n = s1.shape()[0];

        let a = as_f32(s1);
        let b = as_f32(s2);
        // vector_diff = pole2 - pole1.
        let diff: Vec<f32> = b.iter().zip(&a).map(|(y, x)| y - x).collect();

        // One heading per adjacent pair -> N-1 angles; N<2 (incl. an empty input)
        // yields an empty angle list, which the constructor accepts as shape [0].
        let m = n.saturating_sub(1);
        let mut angles: Vec<f32> = Vec::with_capacity(m);
        for i in 0..m {
            // arctan2(y, x) with y = diff[i+1], x = diff[i]; degrees, then [0, 360).
            let deg = diff[i + 1].atan2(diff[i]).to_degrees();
            angles.push(deg.rem_euclid(360.0));
        }

        // Empty meta, faithful to the Python `{}` — the angles carry no axis/rate.
        let buf: Vec<u8> = angles.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = Data::from_array_bytes(DType::F32, vec![m], buf, Meta::empty())
            .map_err(|e| e.to_string())?;
        out.set("angles", data);
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    // No node-specific params (the engine layers on the universal `common` group).
    ParamGroups::new()
}

fn make(_p: &ParamGroups) -> Box<dyn Node> {
    Box::new(Compass)
}

static INPUTS: &[SlotDecl] = &[
    SlotDecl {
        name: "pole1",
        kind: SlotType::Array,
        trigger_process: true,
        multi: false,
    },
    SlotDecl {
        name: "pole2",
        kind: SlotType::Array,
        trigger_process: true,
        multi: false,
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "angles",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Compass",
        category: "analysis",
        doc: "Angles (deg, [0,360)) of the difference vector pole2 - pole1, per adjacent dimension pair.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Channels, Coord, Data, DType, Meta, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs};
    use indexmap::IndexMap;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Drive Compass on the two poles; returns the emitted `angles` Data (or `None`
    /// when the node no-ops).
    fn run(p1: Option<Data>, p2: Option<Data>) -> Option<Data> {
        let m = goofi_node::find("Compass").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("pole1", p1);
        inmap.insert("pole2", p2);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new())
            .unwrap();
        outbuf.get("angles").unwrap().as_ref().cloned()
    }

    /// A 1-D f32 frame with empty meta.
    fn frame(vals: &[f32]) -> Data {
        let buf: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, vec![vals.len()], buf, Meta::empty()).unwrap()
    }

    fn vals(d: &Data) -> (Vec<usize>, Vec<f32>) {
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn headings_of_a_simple_diff() {
        // pole1=[0,0,0], pole2=[1,1,0] -> diff=[1,1,0].
        // i=0: atan2(1,1)=45deg; i=1: atan2(0,1)=0deg. -> [45, 0], shape [2].
        let d = run(Some(frame(&[0.0, 0.0, 0.0])), Some(frame(&[1.0, 1.0, 0.0]))).expect("emits");
        let (shape, v) = vals(&d);
        assert_eq!(shape, vec![2]);
        assert!((v[0] - 45.0).abs() < 1e-3, "got {}", v[0]);
        assert!(v[1].abs() < 1e-3, "got {}", v[1]);
    }

    #[test]
    fn negative_heading_wraps_into_0_360() {
        // pole1=[0,0], pole2=[1,-1] -> diff=[1,-1]. atan2(-1,1) = -45deg -> 315.
        let d = run(Some(frame(&[0.0, 0.0])), Some(frame(&[1.0, -1.0]))).expect("emits");
        let (shape, v) = vals(&d);
        assert_eq!(shape, vec![1]);
        assert!((v[0] - 315.0).abs() < 1e-3, "got {}", v[0]);
    }

    #[test]
    fn covers_all_four_quadrants() {
        // diff = pole2 (pole1 all zero) = [1, 1, -1, -1, 1]; each heading is
        // atan2(diff[i+1], diff[i]): (1,1)=45, (-1,1)=315, (-1,-1)=225, (1,-1)=135.
        let d = run(Some(frame(&[0.0; 5])), Some(frame(&[1.0, 1.0, -1.0, -1.0, 1.0]))).expect("emits");
        let (shape, v) = vals(&d);
        assert_eq!(shape, vec![4]);
        for (got, want) in v.iter().zip([45.0, 315.0, 225.0, 135.0]) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn output_meta_is_empty() {
        // Even when the inputs carry sfreq/channels, the angle output is bare ({}).
        let buf: Vec<u8> = [0.0f32, 0.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Num(0.0), Coord::Num(1.0), Coord::Num(2.0)]));
        let meta = Meta { sfreq: Some(256.0), channels: Channels(ch), ..Default::default() };
        let p1 = Data::from_array_bytes(DType::F32, vec![3], buf.clone(), meta).unwrap();
        let p2 = frame(&[1.0, 1.0, 0.0]);
        let d = run(Some(p1), Some(p2)).expect("emits");
        assert_eq!(d.meta().sfreq, None, "sfreq not carried onto the angle output");
        assert!(d.meta().channels.0.is_empty(), "no channel axis on the angle output");
    }

    #[test]
    fn length_one_input_emits_empty_array() {
        // N=1 -> zero adjacent pairs -> empty (shape [0]) angle array, still emitted.
        let d = run(Some(frame(&[5.0])), Some(frame(&[7.0]))).expect("emits an empty array");
        let (shape, v) = vals(&d);
        assert_eq!(shape, vec![0]);
        assert!(v.is_empty());
    }

    #[test]
    fn empty_input_emits_empty_array() {
        // N=0 -> empty diff -> empty angle array (the constructor accepts shape [0]).
        let d = run(Some(frame(&[])), Some(frame(&[]))).expect("emits an empty array");
        assert_eq!(vals(&d).0, vec![0]);
    }

    #[test]
    fn no_emit_on_missing_pole1() {
        assert!(run(None, Some(frame(&[1.0, 2.0]))).is_none());
    }

    #[test]
    fn no_emit_on_missing_pole2() {
        assert!(run(Some(frame(&[1.0, 2.0])), None).is_none());
    }

    #[test]
    fn no_emit_on_length_mismatch() {
        // Python would raise ValueError; the port silently declines to emit.
        assert!(run(Some(frame(&[1.0, 2.0, 3.0])), Some(frame(&[1.0, 2.0]))).is_none());
    }

    #[test]
    fn no_emit_on_non_1d_input() {
        // A 2-D pole is not a valid 1-D vector -> no-op.
        let two_d = Data::from_array_bytes(
            DType::F32,
            vec![2, 2],
            [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
            Meta::empty(),
        )
        .unwrap();
        assert!(run(Some(two_d), Some(frame(&[1.0, 2.0, 3.0, 4.0]))).is_none());
    }

    #[test]
    fn no_emit_on_non_f32_dtype() {
        // A u8 pole carries the right shape but the wrong dtype -> silent no-op.
        let u8_pole = Data::from_array_bytes(DType::U8, vec![2], vec![1u8, 2], Meta::empty()).unwrap();
        assert!(run(Some(u8_pole), Some(frame(&[1.0, 2.0]))).is_none());
    }
}
