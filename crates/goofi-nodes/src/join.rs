//! Join — combine TWO float32 array inputs into one, either by concatenating them
//! along an existing axis or stacking them along a fresh dimension. A staple for
//! fusing sibling streams (e.g. two electrode montages into one channel axis, or
//! two feature vectors into a [2, N] batch). Ported from the Python
//! `nodes/array/join.py`; the join-axis channel labels are merged (concatenate)
//! and the higher axes shift up (stack) so coordinate labels stay aligned.
//!
//! Faithful deviations from the Python source, mandated by the Rust node contract:
//!   · Both inputs are required — a missing `a` or `b` is a no-op (`Ok(())`),
//!     rather than passing the lone present input through. Native slots deliver
//!     latest-wins, so once both have produced, every trigger emits.
//!   · Incompatible shapes are a no-op (`Ok(())`), not a raised error, mirroring
//!     how the other array nodes swallow malformed input instead of faulting.
//!   · Non-f32 input is a no-op — the native engine speaks f32 arrays.

use std::collections::BTreeMap;
use std::sync::Arc;

use goofi_core::SlotType;
use goofi_core::{Channels, Coord, Data, DType, Meta, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    Concatenate,
    Stack,
}

impl Method {
    fn parse(s: &str) -> Method {
        match s {
            "stack" => Method::Stack,
            _ => Method::Concatenate,
        }
    }
}

struct Join {
    method: Method,
    axis: i64,
}

/// Decode a contiguous little-endian f32 array store to a flat vec.
fn to_f32(store: &goofi_core::ArrayStore) -> Vec<f32> {
    store
        .as_bytes()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Carry sfreq like the Python (mirrors Operation): `a` stays authoritative when
/// both sides have it (they should match for a meaningful join); otherwise carry
/// whichever side provides it. `meta` already holds `a`'s sfreq via the clone, so
/// only the b-only case (a absent, b present) needs an explicit assignment.
fn carry_sfreq(meta: &mut Meta, a: &Meta, b: &Meta) {
    if a.sfreq.is_none() && b.sfreq.is_some() {
        meta.sfreq = b.sfreq;
    }
}

/// Normalize a concatenate axis into `[0, ndim)` (numpy negative-from-end); an
/// out-of-range axis yields `None` (a no-op tick).
fn norm_concat(axis: i64, ndim: usize) -> Option<usize> {
    let a = if axis < 0 { axis + ndim as i64 } else { axis };
    (a >= 0 && (a as usize) < ndim).then_some(a as usize)
}

/// Normalize a stack axis into `[0, ndim]` — stack inserts a NEW axis, so `ndim`
/// itself (append at the end) is valid. Unlike the Python, whose channel-rename
/// loop normalized a negative axis with `+ndim` (a latent inconsistency vs the
/// `+ndim+1` np.stack uses), this single normalization drives both the insertion
/// and the channel shift, keeping them consistent. The common axis >= 0 path is
/// identical to the Python.
fn norm_stack(axis: i64, ndim: usize) -> Option<usize> {
    let a = if axis < 0 { axis + ndim as i64 + 1 } else { axis };
    (a >= 0 && (a as usize) <= ndim).then_some(a as usize)
}

impl Node for Join {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        // Both inputs must be present and f32 arrays, else no emit.
        let (Some(da), Some(db)) = (inp.get("a"), inp.get("b")) else {
            return Ok(());
        };
        let (Value::Array(sa), Value::Array(sb)) = (da.value(), db.value()) else {
            return Ok(());
        };
        if sa.dtype() != DType::F32 || sb.dtype() != DType::F32 {
            return Ok(());
        }
        let ash = sa.shape();
        let bsh = sb.shape();
        let ndim = ash.len();
        let a_data = to_f32(sa);
        let b_data = to_f32(sb);

        let (out_shape, result, meta) = match self.method {
            Method::Concatenate => {
                let Some(ax) = norm_concat(self.axis, ndim) else {
                    return Ok(());
                };
                // All non-join axes must match (and the ranks): guard so a mismatch
                // is a clean no-op instead of a garbled interleave. The ndim check
                // short-circuits the per-axis compare, keeping bsh[d] in range.
                if bsh.len() != ndim || (0..ndim).any(|d| d != ax && ash[d] != bsh[d]) {
                    return Ok(());
                }
                // Row-major view as [outer, axis, inner]; for each outer block append
                // a's slab of `a_axis` then b's slab of `b_axis` along the join axis.
                let outer: usize = ash[..ax].iter().product();
                let inner: usize = ash[ax + 1..].iter().product();
                let a_axis = ash[ax];
                let b_axis = bsh[ax];
                let a_blk = a_axis * inner;
                let b_blk = b_axis * inner;
                let mut result = Vec::with_capacity((a_axis + b_axis) * outer * inner);
                for o in 0..outer {
                    result.extend_from_slice(&a_data[o * a_blk..o * a_blk + a_blk]);
                    result.extend_from_slice(&b_data[o * b_blk..o * b_blk + b_blk]);
                }
                let mut out_shape = ash.to_vec();
                out_shape[ax] = a_axis + b_axis;

                // result_meta = deepcopy(a.meta): a is authoritative for every axis
                // except the join axis, whose labels are the concatenation of both
                // sides' — but only when BOTH sides label it (else a's stale labels
                // stay and the length check faults, exactly as the Python asserts).
                let mut meta = da.meta().clone();
                if let (Some(ca), Some(cb)) = (
                    da.meta().channels.0.get(&ax),
                    db.meta().channels.0.get(&ax),
                ) {
                    let mut merged = Vec::with_capacity(ca.len() + cb.len());
                    merged.extend(ca.iter().cloned());
                    merged.extend(cb.iter().cloned());
                    meta.channels.0.insert(ax, Arc::new(merged));
                }
                carry_sfreq(&mut meta, da.meta(), db.meta());
                (out_shape, result, meta)
            }
            Method::Stack => {
                // Stack needs identical shapes (numpy would raise a shape-less error);
                // a mismatch is a no-op here.
                if ash != bsh {
                    return Ok(());
                }
                let Some(ax) = norm_stack(self.axis, ndim) else {
                    return Ok(());
                };
                // A fresh size-2 axis at `ax`: for each outer block emit a's inner
                // slab (new-axis index 0) then b's (index 1).
                let outer: usize = ash[..ax].iter().product();
                let inner: usize = ash[ax..].iter().product();
                let mut result = Vec::with_capacity(2 * a_data.len());
                for o in 0..outer {
                    result.extend_from_slice(&a_data[o * inner..o * inner + inner]);
                    result.extend_from_slice(&b_data[o * inner..o * inner + inner]);
                }
                let mut out_shape = ash.to_vec();
                out_shape.insert(ax, 2);

                // The new axis pushes every a-label at dim >= ax up by one; the new
                // axis itself is unlabeled. b's labels are dropped (deepcopy(a.meta)).
                let mut meta = da.meta().clone();
                let shifted: BTreeMap<usize, Arc<Vec<Coord>>> = da
                    .meta()
                    .channels
                    .0
                    .iter()
                    .map(|(&k, v)| (if k >= ax { k + 1 } else { k }, v.clone()))
                    .collect();
                meta.channels = Channels(shifted);
                carry_sfreq(&mut meta, da.meta(), db.meta());
                (out_shape, result, meta)
            }
        };

        let buf: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = Data::from_array_bytes(DType::F32, out_shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("join", "method") => {
                if let Some(s) = v.as_str() {
                    self.method = Method::parse(s);
                }
            }
            ("join", "axis") => {
                if let Some(x) = v.as_i64() {
                    self.axis = x;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert(
        "method".to_string(),
        Param::Str {
            value: "concatenate".to_string(),
            options: Some(vec!["concatenate".to_string(), "stack".to_string()]),
            refresh: None,
        },
    );
    g.insert("axis".to_string(), Param::int(0, -8, 8));
    let mut groups = ParamGroups::new();
    groups.insert("join".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let method = param(p, "join", "method")
        .and_then(Param::as_str)
        .map(Method::parse)
        .unwrap_or(Method::Concatenate);
    let axis = param(p, "join", "axis").and_then(Param::as_i64).unwrap_or(0);
    Box::new(Join { method, axis })
}

static INPUTS: &[SlotDecl] = &[
    SlotDecl {
        name: "a",
        kind: SlotType::Array,
        trigger_process: true,
    },
    SlotDecl {
        name: "b",
        kind: SlotType::Array,
        trigger_process: true,
    },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Join",
        category: "array",
        doc: "Combine two arrays by concatenating along an axis or stacking on a new one.",
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

    /// Build an f32 array Data with empty meta.
    fn arr(shape: Vec<usize>, data: &[f32]) -> Data {
        let buf: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, shape, buf, Meta::empty()).unwrap()
    }

    /// Drive Join once with the given method/axis and (optional) inputs; return the
    /// emitted Data, or None when the node produced nothing.
    fn join(method: &str, axis: i64, a: Option<Data>, b: Option<Data>) -> Option<Data> {
        let m = goofi_node::find("Join").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("join", "method"), &Param::str_free(method))
            .unwrap();
        node.on_param_changed(&ParamKey::new("join", "axis"), &Param::int(axis, -8, 8))
            .unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("a", a);
        inmap.insert("b", b);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new())
            .unwrap();
        outbuf.get("out").unwrap().clone()
    }

    fn shape_vals(d: &Data) -> (Vec<usize>, Vec<f32>) {
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn concatenate_1d() {
        let d = join("concatenate", 0, Some(arr(vec![2], &[1.0, 2.0])), Some(arr(vec![3], &[3.0, 4.0, 5.0]))).unwrap();
        assert_eq!(shape_vals(&d), (vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    }

    #[test]
    fn concatenate_2d_along_each_axis() {
        // axis 0: a [1,3] over b [2,3] -> [3,3] (row-major append of whole rows).
        let a = arr(vec![1, 3], &[1.0, 2.0, 3.0]);
        let b = arr(vec![2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let d = join("concatenate", 0, Some(a), Some(b)).unwrap();
        assert_eq!(shape_vals(&d), (vec![3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));

        // axis 1: a [2,2]=[[1,2],[3,4]] with b [2,1]=[[5],[6]] -> [[1,2,5],[3,4,6]].
        let a = arr(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = arr(vec![2, 1], &[5.0, 6.0]);
        let d = join("concatenate", 1, Some(a), Some(b)).unwrap();
        assert_eq!(shape_vals(&d), (vec![2, 3], vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]));
    }

    #[test]
    fn concatenate_negative_axis_counts_from_end() {
        // -1 on a 2-D array == axis 1.
        let a = arr(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = arr(vec![2, 1], &[5.0, 6.0]);
        let d = join("concatenate", -1, Some(a), Some(b)).unwrap();
        assert_eq!(shape_vals(&d), (vec![2, 3], vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]));
    }

    #[test]
    fn incompatible_nonjoin_axis_is_noop() {
        // Concatenating along axis 0, the non-join axis (dim1) differs: 3 vs 2.
        let a = arr(vec![2, 3], &[0.0; 6]);
        let b = arr(vec![2, 2], &[0.0; 4]);
        assert!(join("concatenate", 0, Some(a), Some(b)).is_none());
    }

    #[test]
    fn mismatched_rank_is_noop() {
        let a = arr(vec![4], &[0.0; 4]);
        let b = arr(vec![2, 2], &[0.0; 4]);
        assert!(join("concatenate", 0, Some(a), Some(b)).is_none());
    }

    #[test]
    fn out_of_range_axis_is_noop() {
        let a = arr(vec![2], &[1.0, 2.0]);
        let b = arr(vec![2], &[3.0, 4.0]);
        assert!(join("concatenate", 5, Some(a), Some(b)).is_none());
    }

    #[test]
    fn either_input_absent_is_noop() {
        let a = arr(vec![2], &[1.0, 2.0]);
        let b = arr(vec![2], &[3.0, 4.0]);
        assert!(join("concatenate", 0, Some(a.clone()), None).is_none(), "b absent");
        assert!(join("concatenate", 0, None, Some(b)).is_none(), "a absent");
        assert!(join("concatenate", 0, None, None).is_none(), "both absent");
    }

    #[test]
    fn non_f32_input_is_noop() {
        // b is a u8 array — the native engine only joins f32 arrays.
        let a = arr(vec![2], &[1.0, 2.0]);
        let b = Data::from_array_bytes(DType::U8, vec![2], vec![3u8, 4u8], Meta::empty()).unwrap();
        assert!(join("concatenate", 0, Some(a), Some(b)).is_none());
    }

    #[test]
    fn concatenate_merges_join_axis_channels() {
        // Both sides label the join axis (dim0): merged labels = a's ++ b's.
        let mut ca = BTreeMap::new();
        ca.insert(0usize, Arc::new(vec![Coord::Str("a".into()), Coord::Str("b".into())]));
        let a = Data::from_array_bytes(
            DType::F32,
            vec![2],
            [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
            Meta { channels: Channels(ca), ..Default::default() },
        )
        .unwrap();
        let mut cb = BTreeMap::new();
        cb.insert(0usize, Arc::new(vec![Coord::Str("c".into())]));
        let b = Data::from_array_bytes(
            DType::F32,
            vec![1],
            3.0f32.to_le_bytes().to_vec(),
            Meta { channels: Channels(cb), ..Default::default() },
        )
        .unwrap();

        let d = join("concatenate", 0, Some(a), Some(b)).unwrap();
        assert_eq!(shape_vals(&d).0, vec![3]);
        let coords = d.meta().channels.0.get(&0).expect("join-axis labels merged");
        assert_eq!(
            coords.as_ref(),
            &vec![Coord::Str("a".into()), Coord::Str("b".into()), Coord::Str("c".into())]
        );
    }

    #[test]
    fn concatenate_carries_a_side_nonjoin_channels() {
        // Concatenating along axis 0, a's dim1 labels survive on the untouched axis;
        // b's are ignored (a is authoritative for non-join axes).
        let mut ca = BTreeMap::new();
        ca.insert(1usize, Arc::new(vec![Coord::Str("p".into()), Coord::Str("q".into())]));
        let a = Data::from_array_bytes(
            DType::F32,
            vec![1, 2],
            [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
            Meta { channels: Channels(ca), ..Default::default() },
        )
        .unwrap();
        let b = arr(vec![1, 2], &[3.0, 4.0]);
        let d = join("concatenate", 0, Some(a), Some(b)).unwrap();
        assert_eq!(shape_vals(&d).0, vec![2, 2]);
        let coords = d.meta().channels.0.get(&1).expect("dim1 labels carried from a");
        assert_eq!(coords.as_ref(), &vec![Coord::Str("p".into()), Coord::Str("q".into())]);
    }

    #[test]
    fn sfreq_carry_rules() {
        let with_sfreq = |v: Option<f64>| {
            Data::from_array_bytes(
                DType::F32,
                vec![2],
                [1.0f32, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect(),
                Meta { sfreq: v, ..Default::default() },
            )
            .unwrap()
        };
        // both present -> a's; a-absent -> b's; neither -> None.
        let d = join("concatenate", 0, Some(with_sfreq(Some(250.0))), Some(with_sfreq(Some(500.0)))).unwrap();
        assert_eq!(d.meta().sfreq, Some(250.0), "a authoritative when both present");
        let d = join("concatenate", 0, Some(with_sfreq(None)), Some(with_sfreq(Some(500.0)))).unwrap();
        assert_eq!(d.meta().sfreq, Some(500.0), "carry b when a lacks sfreq");
        let d = join("concatenate", 0, Some(with_sfreq(None)), Some(with_sfreq(None))).unwrap();
        assert_eq!(d.meta().sfreq, None, "neither -> unset");
    }

    #[test]
    fn stack_new_axis_at_front() {
        // stack axis 0 of two [2] vectors -> [2,2] with a as row 0, b as row 1.
        let d = join("stack", 0, Some(arr(vec![2], &[1.0, 2.0])), Some(arr(vec![2], &[3.0, 4.0]))).unwrap();
        assert_eq!(shape_vals(&d), (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn stack_new_axis_interleaves() {
        // stack axis 1 of two [2] vectors -> [2,2] with a,b interleaved column-wise:
        // result[i] = [a[i], b[i]].
        let d = join("stack", 1, Some(arr(vec![2], &[1.0, 2.0])), Some(arr(vec![2], &[3.0, 4.0]))).unwrap();
        assert_eq!(shape_vals(&d), (vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]));
    }

    #[test]
    fn stack_requires_identical_shapes() {
        let a = arr(vec![2], &[1.0, 2.0]);
        let b = arr(vec![3], &[3.0, 4.0, 5.0]);
        assert!(join("stack", 0, Some(a), Some(b)).is_none());
    }

    #[test]
    fn stack_shifts_channels_up() {
        // a labels dim0; stacking at axis 0 inserts a new dim0, pushing a's labels to
        // dim1. The new axis is unlabeled.
        let mut ca = BTreeMap::new();
        ca.insert(0usize, Arc::new(vec![Coord::Str("x".into()), Coord::Str("y".into())]));
        let a = Data::from_array_bytes(
            DType::F32,
            vec![2],
            [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
            Meta { channels: Channels(ca), ..Default::default() },
        )
        .unwrap();
        let b = arr(vec![2], &[3.0, 4.0]);
        let d = join("stack", 0, Some(a), Some(b)).unwrap();
        assert_eq!(shape_vals(&d).0, vec![2, 2]);
        assert!(!d.meta().channels.0.contains_key(&0), "new axis 0 is unlabeled");
        let coords = d.meta().channels.0.get(&1).expect("a's labels shifted to dim1");
        assert_eq!(coords.as_ref(), &vec![Coord::Str("x".into()), Coord::Str("y".into())]);
    }
}
