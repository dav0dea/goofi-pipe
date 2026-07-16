//! Reduce — aggregate a float32 array along one axis (mean/median/min/max/std/
//! norm/sum), dropping that axis. A staple feature-extraction step (a signal or a
//! per-channel window collapsed to a scalar/vector). Ported from the Python
//! `nodes/array/reduce.py`; the reduced axis is removed from `meta["channels"]`
//! and higher axes shift down to keep coordinate labels aligned.

use std::collections::BTreeMap;

use goofi_core::SlotType;
use goofi_core::{Channels, Data, DType, Meta, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    Mean,
    Median,
    Min,
    Max,
    Std,
    Norm,
    Sum,
}

impl Method {
    fn parse(s: &str) -> Method {
        match s {
            "median" => Method::Median,
            "min" => Method::Min,
            "max" => Method::Max,
            "std" => Method::Std,
            "norm" => Method::Norm,
            "sum" => Method::Sum,
            _ => Method::Mean,
        }
    }

    /// Reduce a gathered slice of values along the axis to one scalar.
    fn apply(self, v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        let n = v.len() as f32;
        match self {
            Method::Sum => v.iter().sum(),
            Method::Mean => v.iter().sum::<f32>() / n,
            Method::Min => v.iter().copied().fold(f32::INFINITY, f32::min),
            Method::Max => v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            Method::Norm => v.iter().map(|x| x * x).sum::<f32>().sqrt(),
            Method::Std => {
                // Population standard deviation (numpy default ddof=0).
                let mean = v.iter().sum::<f32>() / n;
                (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt()
            }
            Method::Median => {
                let mut s = v.to_vec();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let m = s.len() / 2;
                if s.len() % 2 == 1 {
                    s[m]
                } else {
                    (s[m - 1] + s[m]) / 2.0
                }
            }
        }
    }
}

struct Reduce {
    method: Method,
    axis: i64,
}

impl Node for Reduce {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("array") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }
        let shape = store.shape();
        let ndim = shape.len();
        if ndim == 0 {
            return Ok(());
        }
        // Negative axes count from the end (numpy semantics); out-of-range is a no-op.
        let axis = if self.axis < 0 { self.axis + ndim as i64 } else { self.axis };
        if axis < 0 || axis as usize >= ndim {
            return Ok(());
        }
        let a = axis as usize;

        let data: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // View the row-major buffer as [outer, axis_len, inner]; reducing the middle
        // axis collapses it to [outer, inner].
        let axis_len = shape[a];
        let outer: usize = shape[..a].iter().product();
        let inner: usize = shape[a + 1..].iter().product();
        let mut result = vec![0f32; outer * inner];
        let mut gather = Vec::with_capacity(axis_len);
        for o in 0..outer {
            for i in 0..inner {
                gather.clear();
                for k in 0..axis_len {
                    gather.push(data[o * axis_len * inner + k * inner + i]);
                }
                result[o * inner + i] = self.method.apply(&gather);
            }
        }
        let out_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| (i != a).then_some(s))
            .collect();

        // Drop the reduced axis's channel labels; shift higher axes down one.
        let mut channels: BTreeMap<usize, _> = BTreeMap::new();
        for (&dim, coords) in d.meta().channels.0.iter() {
            match dim.cmp(&a) {
                std::cmp::Ordering::Less => {
                    channels.insert(dim, coords.clone());
                }
                std::cmp::Ordering::Greater => {
                    channels.insert(dim - 1, coords.clone());
                }
                std::cmp::Ordering::Equal => {}
            }
        }
        let meta = Meta {
            channels: Channels(channels),
            ..Default::default()
        };

        let buf: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = Data::from_array_bytes(DType::F32, out_shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("reduce", "method") => {
                if let Some(s) = v.as_str() {
                    self.method = Method::parse(s);
                }
            }
            ("reduce", "axis") => {
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
            value: "mean".to_string(),
            options: Some(
                ["mean", "median", "min", "max", "std", "norm", "sum"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            refresh: None,
        },
    );
    g.insert("axis".to_string(), Param::int(0, -8, 8));
    let mut groups = ParamGroups::new();
    groups.insert("reduce".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let method = param(p, "reduce", "method").and_then(Param::as_str).map(Method::parse).unwrap_or(Method::Mean);
    let axis = param(p, "reduce", "axis").and_then(Param::as_i64).unwrap_or(0);
    Box::new(Reduce { method, axis })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "array",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Reduce",
        category: "array",
        doc: "Aggregate an array along one axis (mean/median/min/max/std/norm/sum).",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Coord, Data, DType, Meta, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;
    use std::sync::Arc;

    fn reduce(method: &str, axis: i64, shape: Vec<usize>, input: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let m = goofi_node::find("Reduce").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("reduce", "method"), &goofi_core::Param::str_free(method)).unwrap();
        node.on_param_changed(&ParamKey::new("reduce", "axis"), &goofi_core::Param::int(axis, -8, 8)).unwrap();
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, shape, buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("out").unwrap().as_ref().unwrap().clone();
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn reduces_1d_by_each_method() {
        let v = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(reduce("sum", 0, vec![4], &v), (vec![1], vec![10.0]));
        assert_eq!(reduce("mean", 0, vec![4], &v), (vec![1], vec![2.5]));
        assert_eq!(reduce("min", 0, vec![4], &v), (vec![1], vec![1.0]));
        assert_eq!(reduce("max", 0, vec![4], &v), (vec![1], vec![4.0]));
        // std (population): sqrt(mean of squared deviations from 2.5) = sqrt(1.25).
        let (_, s) = reduce("std", 0, vec![4], &v);
        assert!((s[0] - 1.25f32.sqrt()).abs() < 1e-5);
        // norm (L2): sqrt(1+4+9+16) = sqrt(30).
        let (_, nrm) = reduce("norm", 0, vec![4], &v);
        assert!((nrm[0] - 30.0f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn median_even_and_odd() {
        assert_eq!(reduce("median", 0, vec![4], &[1.0, 3.0, 2.0, 4.0]), (vec![1], vec![2.5]));
        assert_eq!(reduce("median", 0, vec![3], &[5.0, 1.0, 3.0]), (vec![1], vec![3.0]));
    }

    #[test]
    fn reduces_2d_along_each_axis() {
        // shape [2,3]: [[1,2,3],[4,5,6]].
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // axis 0 -> mean over rows -> [2.5, 3.5, 4.5], shape [3].
        assert_eq!(reduce("mean", 0, vec![2, 3], &v), (vec![3], vec![2.5, 3.5, 4.5]));
        // axis 1 -> mean over cols -> [2, 5], shape [2].
        assert_eq!(reduce("mean", 1, vec![2, 3], &v), (vec![2], vec![2.0, 5.0]));
        // negative axis: -1 == last axis == axis 1.
        assert_eq!(reduce("mean", -1, vec![2, 3], &v), (vec![2], vec![2.0, 5.0]));
    }

    #[test]
    fn channels_drop_reduced_axis_and_shift() {
        // shape [2,3] with a coord list on dim1; reducing axis 0 must move it to dim0.
        let m = goofi_node::find("Reduce").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut ch = std::collections::BTreeMap::new();
        ch.insert(1usize, Arc::new(vec![Coord::Str("a".into()), Coord::Str("b".into()), Coord::Str("c".into())]));
        let meta = Meta { channels: goofi_core::Channels(ch), ..Default::default() };
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![2, 3], buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        // The dim1 coords are now on dim0 (the reduced axis 0 is gone).
        let coords = d.meta().channels.0.get(&0).expect("coords shifted to dim0");
        assert_eq!(coords.len(), 3);
        assert!(!d.meta().channels.0.contains_key(&1), "no dim1 after reduction to 1-D");
    }
}
