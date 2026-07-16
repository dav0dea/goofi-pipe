//! goofi-view — the shared, payload-free **ViewSpec algebra** for the data plane.
//!
//! A viewer publishes ONE [`ViewSpec`]: a *compatibility predicate* (what it can draw) plus
//! a *reduction request* (what it wants the drawable axes shrunk to). When several viewers
//! look at the same `(node, slot)` — a line panel and a thumbnail, two tabs — their specs
//! [`plan`]-merge into one [`MergedViewSpec`]: the largest need per dimension, with any
//! viewer incompatible with the actual frame dropping out. The backend then reduces the slot
//! **exactly once**, no matter how many viewers watch it.
//!
//! This crate is **`Data`-free**: every shape query goes through the [`Reducible`] trait
//! (the layering spec's Seam B), so the signal `Data` payload — and future audio/video
//! payloads — implement it without this crate depending on any pillar's payload type. The
//! reduction *kernels* (envelope/subsample/area over real bytes) live pillar-side behind the
//! same seam; here lives only the shape-level algebra.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The Data kind a viewer draws. Tag values match goofi-core's `Value::dtype_tag`
/// (0=array, 1=string, 2=table) — the wire contract — without depending on that crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewDtype {
    Array,
    String,
    Table,
}

impl ViewDtype {
    pub fn tag(self) -> u8 {
        match self {
            ViewDtype::Array => 0,
            ViewDtype::String => 1,
            ViewDtype::Table => 2,
        }
    }
}

/// Comparison operators for the dim-count and per-dim length constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimCmp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
}

impl DimCmp {
    /// `actual <op> n`.
    pub fn holds(self, actual: usize, n: usize) -> bool {
        match self {
            DimCmp::Lt => actual < n,
            DimCmp::Le => actual <= n,
            DimCmp::Eq => actual == n,
            DimCmp::Ge => actual >= n,
            DimCmp::Gt => actual > n,
        }
    }
}

/// A constraint on ONE dimension's length. `dim` may be negative (from the end);
/// canonicalized against the actual ndim at evaluation time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimConstraint {
    pub dim: i32,
    pub cmp: DimCmp,
    pub n: usize,
}

/// The per-axis reduction kernel a viewer asks for on a drawable axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduceMethod {
    Envelope,
    Subsample,
    Area,
}

impl ReduceMethod {
    /// Richest-wins ordering on a merge conflict: envelope > area > subsample. A richer
    /// method is a superset a poorer viewer can still render (envelope keeps a waveform's
    /// peaks; a subsample viewer can still draw the envelope's points).
    pub fn richness(self) -> u8 {
        match self {
            ReduceMethod::Envelope => 3,
            ReduceMethod::Area => 2,
            ReduceMethod::Subsample => 1,
        }
    }
}

/// Desired reduction of one axis to at most `max` entries via `method`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisReduce {
    pub dim: i32,
    pub max: usize,
    pub method: ReduceMethod,
}

/// One viewer's full declaration: what it can draw + what it wants reduced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewSpec {
    /// The Data kind this viewer draws. A frame of any other dtype ⇒ this spec does not
    /// admit it (contributes nothing to the merge).
    pub dtype: ViewDtype,
    /// Dim-count comparisons — ALL must hold to admit (empty ⇒ any ndim). A list, so a viewer
    /// can bound a RANGE (an image is `[(Ge, 2), (Le, 3)]`: 2-D H×W or 3-D H×W×C), which a
    /// single comparison can't state.
    #[serde(default)]
    pub ndim: Vec<(DimCmp, usize)>,
    /// Per-dim length comparisons (ALL must hold to admit).
    #[serde(default)]
    pub dims: Vec<DimConstraint>,
    /// Desired per-axis reductions (only applied when this spec admits the frame).
    #[serde(default)]
    pub reduce: Vec<AxisReduce>,
}

/// A payload frame the reducer can query for shape — the Seam B bridge to a pillar's payload
/// (signal `Data`, and future audio/video frames) without this crate depending on it.
pub trait Reducible {
    /// 0=array, 1=string, 2=table (matches the wire dtype tag).
    fn dtype_tag(&self) -> u8;
    /// Number of dimensions (0 for a non-array payload).
    fn ndim(&self) -> usize;
    /// The shape (empty for a non-array payload).
    fn shape(&self) -> &[usize];
}

/// Map a possibly-negative axis index to `0..ndim`, or `None` if out of range. An
/// out-of-range constraint fails admission — a viewer referencing an axis the frame does not
/// have cannot draw it.
pub fn canon_dim(dim: i32, ndim: usize) -> Option<usize> {
    let d = if dim < 0 { dim + ndim as i32 } else { dim };
    (d >= 0 && (d as usize) < ndim).then_some(d as usize)
}

impl ViewSpec {
    /// Whether this viewer can draw `frame` (⇒ its reductions join the merge). The single
    /// place "incompatible viewer drops out" is decided, evaluated against the ACTUAL frame.
    pub fn admits<R: Reducible + ?Sized>(&self, frame: &R) -> bool {
        // 1. dtype gate
        if frame.dtype_tag() != self.dtype.tag() {
            return false;
        }
        // 2. only arrays have dims; string/table admit on dtype alone
        if frame.dtype_tag() != ViewDtype::Array.tag() {
            return true;
        }
        let ndim = frame.ndim();
        // 3. dim-count comparisons (all must hold)
        for &(cmp, n) in &self.ndim {
            if !cmp.holds(ndim, n) {
                return false;
            }
        }
        // 4. per-dim length comparisons (negative dim from the end)
        for c in &self.dims {
            let Some(d) = canon_dim(c.dim, ndim) else {
                return false;
            };
            if !c.cmp.holds(frame.shape()[d], c.n) {
                return false;
            }
        }
        true
    }
}

/// One planned axis reduction (concrete non-negative dim, target `max`, kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedAxis {
    pub dim: usize,
    pub max: usize,
    pub method: ReduceMethod,
}

/// The merged reduction plan for ONE frame. Empty ⇒ passthrough (no reduction).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MergedViewSpec {
    pub axes: Vec<PlannedAxis>,
}

/// Merge N viewers' specs into ONE concrete plan for THIS frame:
///  1. drop every spec that does not admit the frame (incompatible viewer drops out),
///  2. canonicalize each surviving `AxisReduce.dim` against the actual ndim,
///  3. group by canonical dim → `max(max)`, richest method (largest-need-per-dim),
///  4. aspect-preserve: if ≥2 axes reduce via `Area` (an image's H,W), scale them by ONE
///     factor so a non-square source keeps its aspect ratio.
///
/// The per-axis fold — `(max(a.max, b.max), richest(a.method, b.method))` — is applied at
/// EVERY collision (multiple specs on one dim; a `-1` from one viewer and a `+1` from another
/// folding onto the same physical axis after canonicalization), in one place.
pub fn plan<R: Reducible + ?Sized>(specs: &[ViewSpec], frame: &R) -> MergedViewSpec {
    let ndim = frame.ndim();
    let mut order: Vec<usize> = Vec::new(); // first-seen dim order → stable output
    let mut folded: HashMap<usize, (usize, ReduceMethod)> = HashMap::new();
    for spec in specs {
        if !spec.admits(frame) {
            continue;
        }
        for r in &spec.reduce {
            let Some(d) = canon_dim(r.dim, ndim) else {
                continue;
            };
            match folded.get_mut(&d) {
                Some((mx, m)) => {
                    *mx = (*mx).max(r.max);
                    if r.method.richness() > m.richness() {
                        *m = r.method;
                    }
                }
                None => {
                    order.push(d);
                    folded.insert(d, (r.max, r.method));
                }
            }
        }
    }
    let mut axes: Vec<PlannedAxis> = order
        .iter()
        .map(|&d| {
            let (mx, m) = folded[&d];
            PlannedAxis { dim: d, max: mx, method: m }
        })
        .collect();
    aspect_preserve_area(&mut axes, frame.shape());
    MergedViewSpec { axes }
}

/// When ≥2 planned axes use `Area` and at least one genuinely needs to shrink, scale ALL of
/// them by the single smallest factor `min(max_d / shape_d)`, so a non-square image keeps its
/// aspect ratio instead of stretching. Axes that already fit are left untouched.
fn aspect_preserve_area(axes: &mut [PlannedAxis], shape: &[usize]) {
    let area: Vec<usize> = (0..axes.len())
        .filter(|&i| axes[i].method == ReduceMethod::Area && axes[i].dim < shape.len())
        .collect();
    if area.len() < 2 {
        return;
    }
    let factor = area
        .iter()
        .map(|&i| axes[i].max as f64 / shape[axes[i].dim].max(1) as f64)
        .fold(f64::INFINITY, f64::min);
    if !factor.is_finite() || factor >= 1.0 {
        return; // every area axis already fits — no reduction, nothing to preserve
    }
    for &i in &area {
        axes[i].max = ((shape[axes[i].dim] as f64 * factor).round() as usize).max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
