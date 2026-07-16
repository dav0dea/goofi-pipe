//! Reshape — re-partition a float32 array's axes into a new shape parsed from a
//! comma-separated param (e.g. `"8,-1"`, `"-1"`), preserving the total element
//! count. Ported from the Python `nodes/array/reshape.py`. A row-major reshape is
//! a pure re-labelling of the same contiguous bytes, so the payload passes through
//! untouched. Per-axis channel names can't survive an arbitrary re-partition of
//! the axes, so they're dropped; every other meta field (sfreq/index/…) is kept.

use goofi_core::SlotType;
use goofi_core::{Channels, Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Reshape {
    /// The raw `"d0,d1,…"` target-shape string; parsed per tick so a param edit
    /// takes effect without re-instantiating the node.
    shape: String,
}

/// Parse a comma-separated shape string into signed dims (numpy allows one `-1`).
/// `int()` in Python strips surrounding whitespace, so we trim each segment to
/// match. Any non-integer segment makes the whole parse fail (`None`).
fn parse_shape(s: &str) -> Option<Vec<i64>> {
    s.split(',').map(|p| p.trim().parse::<i64>().ok()).collect()
}

/// Resolve a parsed shape against the input's element count, applying numpy's
/// reshape rules: at most one inferred (`-1`) dim, no other negatives, and the
/// product must equal `total`. Returns the concrete `usize` dims, or `None` when
/// the shape is incompatible (which the caller turns into a no-emit).
fn resolve_shape(dims: &[i64], total: usize) -> Option<Vec<usize>> {
    // numpy permits exactly one unknown dimension; more is ambiguous.
    if dims.iter().filter(|&&d| d == -1).count() > 1 {
        return None;
    }
    // -1 is the only legal negative; anything below it is invalid.
    if dims.iter().any(|&d| d < -1) {
        return None;
    }
    // Product of the explicitly-given dims (all `>= 0` here after the guards).
    let known: usize = dims
        .iter()
        .filter(|&&d| d != -1)
        .try_fold(1usize, |a, &d| a.checked_mul(d as usize))?;

    if dims.contains(&-1) {
        // Infer the missing dim so the element count is preserved. `known == 0`
        // would make the inference ambiguous (division by zero) — numpy rejects it.
        if known == 0 || total % known != 0 {
            return None;
        }
        let inferred = total / known;
        Some(dims.iter().map(|&d| if d == -1 { inferred } else { d as usize }).collect())
    } else {
        // No inference: the given shape must exactly account for every element.
        if known != total {
            return None;
        }
        Some(dims.iter().map(|&d| d as usize).collect())
    }
}

impl Node for Reshape {
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
        // A malformed param can't produce a valid array; Python raised here, the
        // Rust idiom is a silent no-emit (like a dtype mismatch) rather than surfacing
        // garbage downstream.
        let Some(dims) = parse_shape(&self.shape) else {
            return Ok(());
        };
        let total: usize = store.shape().iter().product();
        // Total element count must be preserved; an incompatible shape is a no-op.
        let Some(out_shape) = resolve_shape(&dims, total) else {
            return Ok(());
        };

        // A C-contiguous reshape moves no data — the row-major bytes are identical,
        // only the shape changes. Copy them through unchanged.
        let buf = store.as_bytes().to_vec();

        // Drop channel labels (the axes were re-partitioned so per-axis coords no
        // longer describe the data); keep sfreq/index/reduced/extra.
        let mut meta = d.meta().clone();
        meta.channels = Channels::default();

        let data = Data::from_array_bytes(DType::F32, out_shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if let ("reshape", "shape") = (key.group.as_str(), key.name.as_str()) {
            if let Some(s) = v.as_str() {
                self.shape = s.to_string();
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    // Free-form string (no options): a lone "-1" flattens to 1-D by default.
    g.insert("shape".to_string(), Param::str_free("-1"));
    let mut groups = ParamGroups::new();
    groups.insert("reshape".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let shape = param(p, "reshape", "shape")
        .and_then(Param::as_str)
        .unwrap_or("-1")
        .to_string();
    Box::new(Reshape { shape })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "array",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Reshape",
        category: "array",
        doc: "Reshape an array to a new shape (comma-separated dims; one -1 is inferred).",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Coord, Data, DType, Meta, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;
    use std::sync::Arc;

    /// Drive Reshape with a shape param over an f32 input; returns the output
    /// (shape, flat values), or `None` when the node emitted nothing.
    fn run(shape: &str, in_shape: Vec<usize>, input: &[f32]) -> Option<(Vec<usize>, Vec<f32>)> {
        let m = goofi_node::find("Reshape").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("reshape", "shape"), &Param::str_free(shape)).unwrap();
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, in_shape, buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let out = outbuf.get("out").unwrap().as_ref()?.clone();
        match out.value() {
            Value::Array(s) => Some((
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            )),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn reshapes_and_preserves_flat_values() {
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 1-D -> 2-D: shape changes, the row-major byte order is untouched.
        assert_eq!(run("2,3", vec![6], &v), Some((vec![2, 3], v.to_vec())));
        // 2-D -> 2-D likewise keeps the flat order.
        assert_eq!(run("3,2", vec![2, 3], &v), Some((vec![3, 2], v.to_vec())));
        // Identity reshape is a valid no-op reshape.
        assert_eq!(run("6", vec![6], &v), Some((vec![6], v.to_vec())));
    }

    #[test]
    fn infers_single_negative_one() {
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(run("2,-1", vec![6], &v), Some((vec![2, 3], v.to_vec())));
        assert_eq!(run("-1,3", vec![6], &v), Some((vec![2, 3], v.to_vec())));
        // A lone -1 flattens any input to 1-D.
        assert_eq!(run("-1", vec![2, 3], &v), Some((vec![6], v.to_vec())));
        // int() strips whitespace around each dim — match that leniency.
        assert_eq!(run(" 2 , -1 ", vec![6], &v), Some((vec![2, 3], v.to_vec())));
    }

    #[test]
    fn incompatible_or_malformed_shape_emits_nothing() {
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Product 8 != 6.
        assert_eq!(run("4,2", vec![6], &v), None);
        // 6 not divisible by 4 -> the -1 dim can't be inferred.
        assert_eq!(run("4,-1", vec![6], &v), None);
        // Two unknowns is ambiguous; numpy rejects it.
        assert_eq!(run("-1,-1", vec![6], &v), None);
        // A negative other than -1 is invalid.
        assert_eq!(run("-2,3", vec![6], &v), None);
        // Non-integer / empty param strings can't parse.
        assert_eq!(run("abc", vec![6], &v), None);
        assert_eq!(run("", vec![6], &v), None);
    }

    #[test]
    fn non_f32_and_missing_input_are_noops() {
        let m = goofi_node::find("Reshape").unwrap();
        let mut node = (m.make)(&(m.default_params)());

        // A missing/absent input slot emits nothing.
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", None);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("out").unwrap().is_none(), "no input -> no emit");

        // A non-f32 array is rejected by the native f32-only contract.
        let buf: Vec<u8> = [1i32, 2, 3, 4, 5, 6].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::I32, vec![6], buf, Meta::empty()).unwrap();
        let mut inmap2: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap2.insert("array", Some(frame));
        let inp2 = Inputs::new(&inmap2);
        let mut outbuf2 = m.output_buffer();
        node.process(&inp2, &mut Outputs::new(&mut outbuf2), &mut NodeCtx::new()).unwrap();
        assert!(outbuf2.get("out").unwrap().is_none(), "non-f32 -> no emit");
    }

    #[test]
    fn drops_channels_keeps_other_meta() {
        // shape [2,3] carrying dim1 coords + an sfreq; after a reshape the channel
        // labels must be gone while sfreq survives (deepcopy-then-pop semantics).
        let m = goofi_node::find("Reshape").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("reshape", "shape"), &Param::str_free("-1")).unwrap();
        let mut ch = std::collections::BTreeMap::new();
        ch.insert(1usize, Arc::new(vec![Coord::Str("a".into()), Coord::Str("b".into()), Coord::Str("c".into())]));
        let mut meta = Meta { channels: goofi_core::Channels(ch), ..Default::default() };
        meta.sfreq = Some(250.0);
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![2, 3], buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        assert!(d.meta().channels.0.is_empty(), "channels dropped on reshape");
        assert_eq!(d.meta().sfreq, Some(250.0), "sfreq preserved");
        match d.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[6]),
            _ => panic!("expected array"),
        }
    }
}
