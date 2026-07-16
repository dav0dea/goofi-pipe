//! Transpose — permute the axes of a float32 array (e.g. `[channels, time]` ->
//! `[time, channels]` to feed a differently-oriented downstream node). The `axes`
//! param is a comma-separated permutation (output dim i sources input dim axes[i]);
//! a 1-D input becomes a `[1, N]` row. Channel coordinate lists follow the
//! permutation. Ported from the Python `nodes/array/transpose.py`.

use std::collections::BTreeMap;

use goofi_core::SlotType;
use goofi_core::{Channels, Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Transpose {
    axes: String,
}

/// Row-major strides for `shape` (stride[d] = product of the trailing dims).
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

impl Node for Transpose {
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
        let data: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // 1-D -> [1, N] row (data unchanged); the single dim0 moves to dim1.
        let (out_shape, out_data, axes): (Vec<usize>, Vec<f32>, Vec<usize>) = if ndim <= 1 {
            let n = shape.first().copied().unwrap_or(0);
            (vec![1, n], data, vec![1, 0])
        } else {
            // Parse + validate the axes permutation; a malformed one is a no-op.
            let axes: Vec<usize> = self
                .axes
                .split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect();
            let mut sorted = axes.clone();
            sorted.sort_unstable();
            if axes.len() != ndim || sorted != (0..ndim).collect::<Vec<_>>() {
                return Ok(()); // not a permutation of 0..ndim
            }
            let out_shape: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();
            let in_stride = strides(shape);
            let out_stride = strides(&out_shape);
            let nelem: usize = shape.iter().product();
            let mut buf = vec![0f32; nelem];
            for (out_lin, slot) in buf.iter_mut().enumerate() {
                let mut rem = out_lin;
                let mut in_lin = 0usize;
                for i in 0..ndim {
                    let oidx = rem / out_stride[i];
                    rem %= out_stride[i];
                    in_lin += oidx * in_stride[axes[i]];
                }
                *slot = data[in_lin];
            }
            (out_shape, buf, axes)
        };

        // Permute channels: output dim i sources input dim axes[i].
        let mut channels: BTreeMap<usize, _> = BTreeMap::new();
        for (new_dim, &old_dim) in axes.iter().enumerate() {
            if let Some(coords) = d.meta().channels.0.get(&old_dim) {
                channels.insert(new_dim, coords.clone());
            }
        }
        let mut meta = d.meta().clone();
        meta.channels = Channels(channels);

        let buf: Vec<u8> = out_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = Data::from_array_bytes(DType::F32, out_shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if key.group == "transpose" && key.name == "axes" {
            if let Some(s) = v.as_str() {
                self.axes = s.to_string();
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("axes".to_string(), Param::str_free("1,0"));
    let mut groups = ParamGroups::new();
    groups.insert("transpose".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let axes = param(p, "transpose", "axes").and_then(Param::as_str).unwrap_or("1,0").to_string();
    Box::new(Transpose { axes })
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
        type_name: "Transpose",
        category: "array",
        doc: "Permute array axes (comma-separated `axes`, e.g. \"1,0\").",
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

    fn transpose(axes: &str, shape: Vec<usize>, input: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let m = goofi_node::find("Transpose").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("transpose", "axes"), &goofi_core::Param::str_free(axes)).unwrap();
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, shape, buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        match outbuf.get("out").unwrap().as_ref().unwrap().value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn transposes_2d() {
        // [2,3] [[a,b,c],[d,e,f]] -> [3,2] [[a,d],[b,e],[c,f]].
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(transpose("1,0", vec![2, 3], &v), (vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    #[test]
    fn one_d_becomes_a_row() {
        assert_eq!(transpose("1,0", vec![3], &[7.0, 8.0, 9.0]), (vec![1, 3], vec![7.0, 8.0, 9.0]));
    }

    #[test]
    fn three_d_permutation() {
        // [2,1,3] with axes 2,0,1 -> [3,2,1]. Verify a couple of elements.
        let v: Vec<f32> = (0..6).map(|i| i as f32).collect(); // in[a,b,c] row-major over [2,1,3]
        let (shape, out) = transpose("2,0,1", vec![2, 1, 3], &v);
        assert_eq!(shape, vec![3, 2, 1]);
        // out[i,j,k] = in[j, k, i]; out[0,0,0]=in[0,0,0]=0; out[0,1,0]=in[1,0,0]=3; out[1,0,0]=in[0,0,1]=1.
        assert_eq!(out[0], 0.0);
        // out linear for [0,1,0] over shape[3,2,1]: strides [2,1,1] -> 0*2+1*1+0 = 1.
        assert_eq!(out[1], 3.0);
        // [1,0,0]: 1*2 = 2.
        assert_eq!(out[2], 1.0);
    }

    #[test]
    fn invalid_axes_is_a_noop() {
        // "0,0" is not a permutation of 0..2 -> node emits nothing.
        let m = goofi_node::find("Transpose").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("transpose", "axes"), &goofi_core::Param::str_free("0,0")).unwrap();
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![2, 2], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("out").unwrap().is_none(), "malformed axes must be a no-op");
    }

    #[test]
    fn channels_follow_the_permutation() {
        // [2,3] with dim0 channel labels; transpose 1,0 -> labels move to dim1.
        let m = goofi_node::find("Transpose").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut meta = Meta::empty();
        meta.channels.0.insert(0, Arc::new(vec![Coord::Str("Fz".into()), Coord::Str("Cz".into())]));
        let buf: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![2, 3], buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        assert!(!d.meta().channels.0.contains_key(&0), "dim0 labels moved off dim0");
        assert_eq!(d.meta().channels.0.get(&1).unwrap().len(), 2, "labels now on dim1");
    }
}
