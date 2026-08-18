//! The ViewSpec algebra: many viewers' constraints folded into ONE reduction per frame.
//!
//! Payload-free by construction — a spec states what a viewer CAN draw, never what it wants to
//! see — so the fold is pure and testable against a shape alone.

use goofi_view::*;

/// A shape-only test frame implementing the Seam B trait.
struct Frame {
    tag: u8,
    shape: Vec<usize>,
}
impl Frame {
    fn array(shape: &[usize]) -> Frame {
        Frame { tag: 0, shape: shape.to_vec() }
    }
    fn string() -> Frame {
        Frame { tag: 1, shape: vec![] }
    }
}
impl Reducible for Frame {
    fn dtype_tag(&self) -> u8 {
        self.tag
    }
    fn ndim(&self) -> usize {
        self.shape.len()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

fn line_1d(w: usize) -> ViewSpec {
    ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Le, 2)],
        dims: vec![],
        reduce: vec![AxisReduce { dim: -1, max: w, method: ReduceMethod::Envelope }],
    }
}

#[test]
fn dimcmp_holds_all_operators() {
    assert!(DimCmp::Lt.holds(1, 2) && !DimCmp::Lt.holds(2, 2));
    assert!(DimCmp::Le.holds(2, 2) && !DimCmp::Le.holds(3, 2));
    assert!(DimCmp::Eq.holds(2, 2) && !DimCmp::Eq.holds(1, 2));
    assert!(DimCmp::Ge.holds(2, 2) && !DimCmp::Ge.holds(1, 2));
    assert!(DimCmp::Gt.holds(3, 2) && !DimCmp::Gt.holds(2, 2));
}

#[test]
fn canon_dim_negative_and_out_of_range() {
    assert_eq!(canon_dim(-1, 3), Some(2));
    assert_eq!(canon_dim(0, 3), Some(0));
    assert_eq!(canon_dim(-3, 3), Some(0));
    assert_eq!(canon_dim(3, 3), None);
    assert_eq!(canon_dim(-4, 3), None);
}

#[test]
fn admits_dtype_gate() {
    let string_viewer = ViewSpec { dtype: ViewDtype::String, ndim: vec![], dims: vec![], reduce: vec![] };
    assert!(!string_viewer.admits(&Frame::array(&[10])), "string viewer rejects an array");
    assert!(string_viewer.admits(&Frame::string()), "string viewer admits a string on dtype alone");
    assert!(line_1d(150).admits(&Frame::array(&[1000])), "line admits a 1-D array");
}

#[test]
fn admits_ndim_and_per_dim_constraints() {
    // An image viewer: any ndim, but the last dim must be a channel count 1..=4.
    let image = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![],
        dims: vec![DimConstraint { dim: -1, cmp: DimCmp::Le, n: 4 }],
        reduce: vec![],
    };
    assert!(image.admits(&Frame::array(&[200, 100, 3])), "HxWx3 admitted");
    assert!(!image.admits(&Frame::array(&[200, 100, 5])), "HxWx5 rejected (>4 channels)");
    // ndim gate: a trajectory viewer needs exactly 2 dims.
    let traj = ViewSpec { dtype: ViewDtype::Array, ndim: vec![(DimCmp::Eq, 2)], dims: vec![], reduce: vec![] };
    assert!(traj.admits(&Frame::array(&[500, 3])));
    assert!(!traj.admits(&Frame::array(&[500])), "1-D rejected by ndim Eq 2");
}

#[test]
fn admits_ndim_list_expresses_a_bounded_range() {
    // The real image viewer's dim-count need is a RANGE — 2-D (H,W) or 3-D (H,W,C) —
    // which a single ndim comparison can't state. A list of ndim constraints (ALL must
    // hold) expresses `2 <= ndim <= 3`, dropping the viewer from a 1-D or 4-D merge.
    let image = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Ge, 2), (DimCmp::Le, 3)],
        dims: vec![],
        reduce: vec![],
    };
    assert!(image.admits(&Frame::array(&[200, 100])), "2-D HxW admitted");
    assert!(image.admits(&Frame::array(&[200, 100, 3])), "3-D HxWxC admitted");
    assert!(!image.admits(&Frame::array(&[4000])), "1-D rejected (below the range)");
    assert!(!image.admits(&Frame::array(&[2, 2, 2, 2])), "4-D rejected (above the range)");
}

#[test]
fn incompatible_viewer_drops_out_of_merge() {
    // A line viewer (compatible) + an image viewer that rejects a 1-D frame. Only the
    // line's reduction survives into the plan.
    let image = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Eq, 2)],
        dims: vec![],
        reduce: vec![AxisReduce { dim: 0, max: 100, method: ReduceMethod::Area }],
    };
    let plan = plan(&[line_1d(150), image], &Frame::array(&[4000]));
    assert_eq!(plan.axes, vec![PlannedAxis { dim: 0, max: 150, method: ReduceMethod::Envelope }]);
}

#[test]
fn golden_wire_shape_specs_deserialize_and_merge() {
    // G6: pin the merge against the EXACT JSON wire shape the frontend sends (viewSpecForKind
    // / the inband {op:"view"} specs). Guards serde compatibility AND the merge algebra
    // together: a line viewer wanting the last axis enveloped to 128, plus a channel viewer
    // subsampling dim 0 to 64, on a 2-D (channels × samples) frame.
    let wire = r#"[
        {"dtype":"array","ndim":[["le",2]],"dims":[],
         "reduce":[{"dim":0,"max":64,"method":"subsample"},
                   {"dim":-1,"max":128,"method":"envelope"}]},
        {"dtype":"array","ndim":[["le",2]],"dims":[],
         "reduce":[{"dim":-1,"max":256,"method":"envelope"}]}
    ]"#;
    let specs: Vec<ViewSpec> = serde_json::from_str(wire).expect("wire specs deserialize");
    // 8 channels × 4000 samples.
    let plan = plan(&specs, &Frame::array(&[8, 4000]));
    // Dim 0 (channels): only the first viewer touches it → subsample to 64 (but 8 < 64, so
    // that axis is admitted-but-won't-shrink; it still appears in the plan at max 64).
    // Dim 1 (samples): max(128, 256) = 256, envelope (richest).
    assert_eq!(
        plan.axes,
        vec![
            PlannedAxis { dim: 0, max: 64, method: ReduceMethod::Subsample },
            PlannedAxis { dim: 1, max: 256, method: ReduceMethod::Envelope },
        ]
    );
}

#[test]
fn merge_takes_largest_need_and_richest_method_per_dim() {
    // Two viewers on the same axis: max(max)=300, richest method = envelope (> subsample).
    let a = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![],
        dims: vec![],
        reduce: vec![AxisReduce { dim: -1, max: 150, method: ReduceMethod::Subsample }],
    };
    let b = line_1d(300); // envelope, max 300
    let plan = plan(&[a, b], &Frame::array(&[10_000]));
    assert_eq!(plan.axes, vec![PlannedAxis { dim: 0, max: 300, method: ReduceMethod::Envelope }]);
}

#[test]
fn cross_family_axis_conflict_degrades_to_subsample() {
    // A line viewer (envelope) and an image viewer (area/block-mean) co-view the SAME 2-D slot;
    // both admit a 2-D array, so their specs merge per-axis. envelope and area are DIFFERENT
    // families — envelope doubles an axis into interleaved [min,max] (uninterpretable as an
    // image), area block-means (destroys the exact samples a line viewer needs). "Richest wins"
    // is only valid WITHIN the line family (subsample<->envelope); across families the only
    // value/position-preserving reduction both can render is exact subsampling. So the conflict
    // must degrade to Subsample, not envelope. Otherwise the image viewer draws corrupt pixels.
    let line = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Le, 3)],
        dims: vec![],
        reduce: vec![AxisReduce { dim: 1, max: 500, method: ReduceMethod::Envelope }],
    };
    let image = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Ge, 2), (DimCmp::Le, 3)],
        dims: vec![],
        reduce: vec![AxisReduce { dim: 1, max: 300, method: ReduceMethod::Area }],
    };
    let plan = plan(&[line, image], &Frame::array(&[8, 4000]));
    assert_eq!(
        plan.axes,
        vec![PlannedAxis { dim: 1, max: 500, method: ReduceMethod::Subsample }],
        "cross-family (envelope vs area) must degrade to the common denominator: subsample"
    );
}

#[test]
fn cross_family_degradation_does_not_depend_on_spec_order() {
    // THREE co-viewers on one axis — two line panels and one image panel. The specs reach
    // `plan` in whatever order the reducer's per-connection map yields (mount order within a
    // socket, arbitrary across tabs), so the SAME multiset must always plan the same way.
    // A pairwise fold cannot deliver that: it forgets that an Area viewer was ever present
    // the moment the cross-family conflict degrades to Subsample.
    let line = |max: usize| ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Le, 3)],
        dims: vec![],
        reduce: vec![AxisReduce { dim: 1, max, method: ReduceMethod::Envelope }],
    };
    let image = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Ge, 2), (DimCmp::Le, 3)],
        dims: vec![],
        reduce: vec![AxisReduce { dim: 1, max: 300, method: ReduceMethod::Area }],
    };
    let expected = vec![PlannedAxis { dim: 1, max: 500, method: ReduceMethod::Subsample }];
    let frame = Frame::array(&[8, 4000]);
    for (label, specs) in [
        ("image last", vec![line(500), line(400), image.clone()]),
        ("image middle", vec![line(500), image.clone(), line(400)]),
        ("image first", vec![image.clone(), line(500), line(400)]),
    ] {
        assert_eq!(plan(&specs, &frame).axes, expected, "{label}: an Area co-viewer must survive the fold");
    }
}

#[test]
fn one_and_two_d_line_specs_collapse_on_a_1d_frame() {
    // A 2-D line spec ({0,rows,subsample},{-1,W,envelope}) on a 1-D frame: both axes
    // canonicalize to dim 0 → richest (envelope) wins, max = max(rows, W).
    let line_2d = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![(DimCmp::Le, 2)],
        dims: vec![],
        reduce: vec![
            AxisReduce { dim: 0, max: 32, method: ReduceMethod::Subsample },
            AxisReduce { dim: -1, max: 200, method: ReduceMethod::Envelope },
        ],
    };
    let plan = plan(&[line_2d], &Frame::array(&[5000]));
    assert_eq!(plan.axes, vec![PlannedAxis { dim: 0, max: 200, method: ReduceMethod::Envelope }]);
}

#[test]
fn image_area_reduction_preserves_aspect() {
    // A 400x100 frame reduced by an image viewer wanting <=200 (H) x <=100 (W). The H
    // axis needs 200/400=0.5; W needs 100/100=1.0; min=0.5 applied to both → 200x50.
    let image = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![],
        dims: vec![],
        reduce: vec![
            AxisReduce { dim: 0, max: 200, method: ReduceMethod::Area },
            AxisReduce { dim: 1, max: 100, method: ReduceMethod::Area },
        ],
    };
    let plan = plan(&[image], &Frame::array(&[400, 100]));
    assert_eq!(
        plan.axes,
        vec![
            PlannedAxis { dim: 0, max: 200, method: ReduceMethod::Area },
            PlannedAxis { dim: 1, max: 50, method: ReduceMethod::Area },
        ],
        "aspect preserved: both scaled by 0.5",
    );
}

#[test]
fn passthrough_when_no_reductions_requested() {
    let string_viewer = ViewSpec { dtype: ViewDtype::String, ndim: vec![], dims: vec![], reduce: vec![] };
    assert_eq!(plan(&[string_viewer], &Frame::string()), MergedViewSpec::default());
}

#[test]
fn viewspec_json_roundtrips() {
    // The /data inband contract: a viewer's spec serializes to JSON and back.
    let spec = line_1d(150);
    let json = serde_json::to_string(&spec).unwrap();
    assert_eq!(serde_json::from_str::<ViewSpec>(&json).unwrap(), spec);
}
