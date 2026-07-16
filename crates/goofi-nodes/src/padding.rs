//! Padding — grow a float32 array along one axis by padding it, using one of
//! numpy's fill modes (constant/edge/reflect/wrap). A staple windowing step (pad
//! a short buffer up to a fixed length, mirror-extend a signal before an FFT).
//! Ported from the Python `nodes/array/padding.py`.
//!
//! The one axis grows by `pad_before + pad_after`; every other axis (and all its
//! `meta["channels"]` labels) is untouched. On the padded axis, existing channel
//! labels become stale, so — like the Python node — we splice in `pad_<i>`
//! placeholder names around the originals to keep the coord list length matching
//! the new axis length. `target_size` is a convenience override: when set it pads
//! at the *end* up to that length (and ignores the explicit before/after amounts),
//! matching the Python semantics exactly.

use std::sync::Arc;

use goofi_core::SlotType;
use goofi_core::{Coord, Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Constant,
    Edge,
    Reflect,
    Wrap,
}

impl Mode {
    /// The dropdown constrains the value to the four supported modes; an
    /// unrecognized string falls back to `constant` (numpy's most permissive
    /// mode) rather than erroring, mirroring `reduce.rs`'s parse.
    fn parse(s: &str) -> Mode {
        match s {
            "edge" => Mode::Edge,
            "reflect" => Mode::Reflect,
            "wrap" => Mode::Wrap,
            _ => Mode::Constant,
        }
    }
}

/// Map an out-of-range axis index to a valid one under the given mode. `src` is
/// the position along the *original* axis a padded output cell refers to (negative
/// for the before-pad region, `>= n` for the after-pad region). Never called for
/// `Constant` (that fill needs no source cell) nor when `n == 0`.
fn fold_index(mode: Mode, src: isize, n: usize) -> usize {
    match mode {
        // numpy 'edge': clamp to the nearest edge value.
        Mode::Edge => {
            if src < 0 {
                0
            } else if src as usize >= n {
                n - 1
            } else {
                src as usize
            }
        }
        // numpy 'wrap': periodic tiling (period n). Euclidean-style modulo keeps
        // the before-pad region (negative src) non-negative.
        Mode::Wrap => {
            let m = n as isize;
            (((src % m) + m) % m) as usize
        }
        // numpy 'reflect' (default, edge value NOT repeated): a triangle-wave fold
        // with period 2*(n-1). A length-1 axis has nothing to reflect, so it just
        // repeats the single value.
        Mode::Reflect => {
            if n == 1 {
                return 0;
            }
            let m = n as isize;
            let period = 2 * (m - 1);
            let mut r = src % period;
            if r < 0 {
                r += period;
            }
            if r < m {
                r as usize
            } else {
                (period - r) as usize
            }
        }
        Mode::Constant => 0, // unreachable: constant never indexes the source
    }
}

struct Padding {
    pad_before: i64,
    pad_after: i64,
    target_size: i64,
    axis: i64,
    mode: Mode,
    constant_value: f32,
}

impl Node for Padding {
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
        // Negative axes count from the end (numpy); out-of-range is invalid -> no emit
        // (Python would raise here — we guard instead, per the engine's contract).
        let axis = if self.axis < 0 { self.axis + ndim as i64 } else { self.axis };
        if axis < 0 || axis as usize >= ndim {
            return Ok(());
        }
        let a = axis as usize;
        // The UI caps these at >= 0, but a hand-edited patch could send junk; a
        // negative pad has no meaning -> treat as invalid.
        if self.pad_before < 0 || self.pad_after < 0 || self.target_size < 0 {
            return Ok(());
        }

        let n = shape[a];
        // `target_size` (when set) overrides the explicit amounts: pad only at the
        // end up to that length; a target <= current size is a no-op pad.
        let (pad_before, pad_after) = if self.target_size > 0 {
            let ts = self.target_size as usize;
            if ts > n {
                (0usize, ts - n)
            } else {
                (0usize, 0usize)
            }
        } else {
            (self.pad_before as usize, self.pad_after as usize)
        };

        // Reflect/edge/wrap need at least one source value along the axis; padding
        // an empty axis with them is undefined (numpy raises) -> no emit. Constant
        // is fine on an empty axis (it never reads a source cell).
        if n == 0 && self.mode != Mode::Constant && (pad_before > 0 || pad_after > 0) {
            return Ok(());
        }

        let out_axis = pad_before + n + pad_after;
        // Row-major view as [outer, axis, inner]; only the middle dim changes length.
        let outer: usize = shape[..a].iter().product();
        let inner: usize = shape[a + 1..].iter().product();

        let data: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let mut result = vec![0f32; outer * out_axis * inner];
        for o in 0..outer {
            for p in 0..out_axis {
                // Which original-axis cell this output row copies from (out of range
                // in the pad regions).
                let src = p as isize - pad_before as isize;
                let in_range = src >= 0 && (src as usize) < n;
                for i in 0..inner {
                    let val = if in_range {
                        data[o * n * inner + src as usize * inner + i]
                    } else if self.mode == Mode::Constant {
                        self.constant_value
                    } else {
                        data[o * n * inner + fold_index(self.mode, src, n) * inner + i]
                    };
                    result[o * out_axis * inner + p * inner + i] = val;
                }
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape[a] = out_axis;

        // Carry the whole meta through; only the padded axis's channel labels go
        // stale. Rewrite them as [pad_0.., originals.., pad_<L>..] so the coord
        // list still matches the grown axis (Python does the same). Other axes'
        // labels stay valid (their lengths are unchanged).
        let mut meta = d.meta().clone();
        if let Some(orig) = meta.channels.0.get(&a).cloned() {
            let orig_len = orig.len();
            let mut coords: Vec<Coord> = Vec::with_capacity(pad_before + orig_len + pad_after);
            for i in 0..pad_before {
                coords.push(Coord::Str(format!("pad_{i}").into()));
            }
            coords.extend(orig.iter().cloned());
            for i in 0..pad_after {
                coords.push(Coord::Str(format!("pad_{}", orig_len + i).into()));
            }
            meta.channels.0.insert(a, Arc::new(coords));
        }

        let buf: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = Data::from_array_bytes(DType::F32, out_shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("padding", "pad_before") => {
                if let Some(x) = v.as_i64() {
                    self.pad_before = x;
                }
            }
            ("padding", "pad_after") => {
                if let Some(x) = v.as_i64() {
                    self.pad_after = x;
                }
            }
            ("padding", "target_size") => {
                if let Some(x) = v.as_i64() {
                    self.target_size = x;
                }
            }
            ("padding", "axis") => {
                if let Some(x) = v.as_i64() {
                    self.axis = x;
                }
            }
            ("padding", "mode") => {
                if let Some(s) = v.as_str() {
                    self.mode = Mode::parse(s);
                }
            }
            ("padding", "constant_value") => {
                if let Some(x) = v.as_f64() {
                    self.constant_value = x as f32;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("pad_before".to_string(), Param::int(0, 0, 1000));
    g.insert("pad_after".to_string(), Param::int(0, 0, 1000));
    g.insert("target_size".to_string(), Param::int(0, 0, 10000));
    g.insert("axis".to_string(), Param::int(-1, -10, 10));
    g.insert(
        "mode".to_string(),
        Param::Str {
            value: "constant".to_string(),
            options: Some(
                ["constant", "edge", "reflect", "wrap"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            refresh: None,
        },
    );
    // Python declares FloatParam(0.0): default 0.0, slider range [0.0, 1.0]. The
    // range is only a UI hint — the engine stores whatever value is set.
    g.insert("constant_value".to_string(), Param::float(0.0, 0.0, 1.0));
    let mut groups = ParamGroups::new();
    groups.insert("padding".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let pad_before = param(p, "padding", "pad_before").and_then(Param::as_i64).unwrap_or(0);
    let pad_after = param(p, "padding", "pad_after").and_then(Param::as_i64).unwrap_or(0);
    let target_size = param(p, "padding", "target_size").and_then(Param::as_i64).unwrap_or(0);
    let axis = param(p, "padding", "axis").and_then(Param::as_i64).unwrap_or(-1);
    let mode = param(p, "padding", "mode").and_then(Param::as_str).map(Mode::parse).unwrap_or(Mode::Constant);
    let constant_value = param(p, "padding", "constant_value").and_then(Param::as_f64).unwrap_or(0.0) as f32;
    Box::new(Padding {
        pad_before,
        pad_after,
        target_size,
        axis,
        mode,
        constant_value,
    })
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
        type_name: "Padding",
        category: "array",
        doc: "Pad an array along one axis (constant/edge/reflect/wrap), or up to a target size.",
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

    /// Build the node, apply the given param edits, run one tick, return the raw
    /// output `Data` (None when the node emits nothing).
    fn run_data(params: &[(&str, Param)], input: Option<Data>) -> Option<Data> {
        let m = goofi_node::find("Padding").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        for (name, p) in params {
            node.on_param_changed(&ParamKey::new("padding", *name), p).unwrap();
        }
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", input);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        outbuf.get("out").unwrap().clone()
    }

    fn f32_frame(shape: Vec<usize>, input: &[f32]) -> Data {
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, shape, buf, Meta::empty()).unwrap()
    }

    /// Run and destructure the emitted array into (shape, values).
    fn run(params: &[(&str, Param)], shape: Vec<usize>, input: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let d = run_data(params, Some(f32_frame(shape, input))).expect("expected an emit");
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn constant_pads_before_and_after_1d() {
        // Default mode=constant, constant_value=0.0.
        let (shape, out) = run(
            &[("pad_before", Param::int(1, 0, 1000)), ("pad_after", Param::int(2, 0, 1000))],
            vec![3],
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(shape, vec![6]);
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn constant_value_is_honored() {
        // The [0,1] slider range is a UI hint; on_param_changed stores 9.0 verbatim.
        let (_, out) = run(
            &[("pad_before", Param::int(2, 0, 1000)), ("constant_value", Param::float(9.0, 0.0, 1.0))],
            vec![2],
            &[1.0, 2.0],
        );
        assert_eq!(out, vec![9.0, 9.0, 1.0, 2.0]);
    }

    #[test]
    fn edge_repeats_the_border_values() {
        // numpy 'edge': [1,1,1,2,3,3].
        let (_, out) = run(
            &[
                ("mode", Param::str_free("edge")),
                ("pad_before", Param::int(2, 0, 1000)),
                ("pad_after", Param::int(1, 0, 1000)),
            ],
            vec![3],
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(out, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn reflect_mirrors_without_repeating_the_edge() {
        // numpy 'reflect': [3,2,1,2,3,4,5,4,3] (edge value 1/5 not duplicated).
        let (_, out) = run(
            &[
                ("mode", Param::str_free("reflect")),
                ("pad_before", Param::int(2, 0, 1000)),
                ("pad_after", Param::int(2, 0, 1000)),
            ],
            vec![5],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        );
        assert_eq!(out, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn reflect_length_one_axis_just_repeats() {
        // Nothing to reflect off a single value; numpy fills with that value.
        let (_, out) = run(
            &[
                ("mode", Param::str_free("reflect")),
                ("pad_before", Param::int(2, 0, 1000)),
                ("pad_after", Param::int(1, 0, 1000)),
            ],
            vec![1],
            &[7.0],
        );
        assert_eq!(out, vec![7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn wrap_tiles_periodically() {
        // numpy 'wrap': end values pad the front, front values pad the end.
        let (_, out) = run(
            &[
                ("mode", Param::str_free("wrap")),
                ("pad_before", Param::int(2, 0, 1000)),
                ("pad_after", Param::int(2, 0, 1000)),
            ],
            vec![5],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        );
        assert_eq!(out, vec![4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]);
    }

    #[test]
    fn pads_2d_along_first_axis_via_negative_index() {
        // axis -2 == axis 0 on a [2,3]; constant pad adds zero rows top and bottom.
        let (shape, out) = run(
            &[
                ("axis", Param::int(-2, -10, 10)),
                ("pad_before", Param::int(1, 0, 1000)),
                ("pad_after", Param::int(1, 0, 1000)),
            ],
            vec![2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert_eq!(shape, vec![4, 3]);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pads_2d_along_last_axis_respecting_inner_stride() {
        // Default axis=-1 (last). Edge-pad each row on the right by 2.
        let (shape, out) = run(
            &[("mode", Param::str_free("edge")), ("pad_after", Param::int(2, 0, 1000))],
            vec![2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert_eq!(shape, vec![2, 5]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn target_size_overrides_amounts_and_pads_at_end() {
        // target 5 on a length-3 axis -> append 2; the explicit pad_before is ignored.
        let (shape, out) = run(
            &[("target_size", Param::int(5, 0, 10000)), ("pad_before", Param::int(7, 0, 1000))],
            vec![3],
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(shape, vec![5]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn target_size_at_or_below_current_is_a_noop() {
        let (shape, out) = run(
            &[("target_size", Param::int(2, 0, 10000)), ("pad_after", Param::int(5, 0, 1000))],
            vec![3],
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(shape, vec![3]);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn channels_on_padded_axis_get_placeholder_names() {
        // dim0 coords [a,b,c]; pad 1 before + 2 after -> [pad_0,a,b,c,pad_3,pad_4].
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("a".into()), Coord::Str("b".into()), Coord::Str("c".into())]));
        let meta = Meta { sfreq: Some(250.0), channels: Channels(ch), ..Default::default() };
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![3], buf, meta).unwrap();
        let d = run_data(
            &[
                ("axis", Param::int(0, -10, 10)),
                ("pad_before", Param::int(1, 0, 1000)),
                ("pad_after", Param::int(2, 0, 1000)),
            ],
            Some(frame),
        )
        .expect("emit");
        let coords = d.meta().channels.0.get(&0).expect("dim0 coords");
        let names: Vec<String> = coords
            .iter()
            .map(|c| match c {
                Coord::Str(s) => s.to_string(),
                Coord::Num(n) => n.to_string(),
            })
            .collect();
        assert_eq!(names, vec!["pad_0", "a", "b", "c", "pad_3", "pad_4"]);
        assert_eq!(d.meta().sfreq, Some(250.0), "unrelated meta carried through");
        match d.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[6], "coord list matches grown axis"),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn channels_on_other_axes_survive_unchanged() {
        // Coords live on dim1; padding dim0 must leave them (and their length) intact.
        let mut ch = BTreeMap::new();
        ch.insert(1usize, Arc::new(vec![Coord::Str("x".into()), Coord::Str("y".into()), Coord::Str("z".into())]));
        let meta = Meta { channels: Channels(ch), ..Default::default() };
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![2, 3], buf, meta).unwrap();
        let d = run_data(
            &[("axis", Param::int(0, -10, 10)), ("pad_after", Param::int(1, 0, 1000))],
            Some(frame),
        )
        .expect("emit");
        let dim1 = d.meta().channels.0.get(&1).expect("dim1 coords intact");
        assert_eq!(dim1.len(), 3);
        assert!(!d.meta().channels.0.contains_key(&0), "padded axis had no labels -> none added");
        match d.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[3, 3]),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn no_emit_on_invalid_or_absent_input() {
        // Missing input slot -> nothing.
        assert!(run_data(&[("pad_after", Param::int(2, 0, 1000))], None).is_none());
        // Non-f32 dtype -> native f32-only guard.
        let i32frame = Data::from_array_bytes(DType::I32, vec![2], vec![0u8; 8], Meta::empty()).unwrap();
        assert!(run_data(&[("pad_after", Param::int(2, 0, 1000))], Some(i32frame)).is_none());
        // Axis out of range -> guarded (Python would raise).
        assert!(run_data(
            &[("axis", Param::int(5, -10, 10)), ("pad_after", Param::int(1, 0, 1000))],
            Some(f32_frame(vec![3], &[1.0, 2.0, 3.0])),
        )
        .is_none());
    }

    #[test]
    fn zero_padding_passes_the_array_through() {
        // No amounts, no target -> shape and values unchanged (still emits).
        let (shape, out) = run(&[], vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
