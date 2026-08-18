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

/// Which kernels the admitted viewers asked for on ONE axis. A SET, not a running pairwise
/// merge: set-union is commutative and associative, so the fold cannot depend on the order the
/// specs happen to arrive in. (A pairwise merge could not promise that — degrading a
/// cross-family conflict to Subsample erased the fact that an Area viewer had been seen, and a
/// third spec asking for Envelope then won the axis back.)
#[derive(Clone, Copy, Default)]
struct MethodSet {
    envelope: bool,
    subsample: bool,
    area: bool,
}

impl MethodSet {
    fn add(&mut self, m: ReduceMethod) {
        match m {
            ReduceMethod::Envelope => self.envelope = true,
            ReduceMethod::Subsample => self.subsample = true,
            ReduceMethod::Area => self.area = true,
        }
    }

    /// The ONE method that must serve every subscriber (the "reduce once, fan out to all"
    /// invariant). Within the LINE family (subsample<->envelope) the richer wins — envelope is a
    /// superset a subsample viewer can still draw. ACROSS families "richest" is meaningless:
    /// envelope doubles an axis into interleaved [min,max] (uninterpretable as an image), and
    /// area is a block-MEAN (destroys the exact samples a line/trajectory viewer needs). The only
    /// value- and position-preserving reduction BOTH an image and a line/trajectory viewer can
    /// render is exact subsampling, so a cross-family conflict degrades to Subsample — the safe
    /// common denominator.
    fn resolve(self) -> ReduceMethod {
        if self.area {
            if self.envelope || self.subsample {
                ReduceMethod::Subsample
            } else {
                ReduceMethod::Area
            }
        } else if self.envelope {
            ReduceMethod::Envelope
        } else {
            ReduceMethod::Subsample
        }
    }
}

/// Desired reduction of one axis to `max` BINS via `method`.
///
/// Bins, not output entries — the distinction matters for `Envelope`, which emits a (min, max)
/// pair per bin and so returns `2 * max` values. That is deliberate and is what the caller wants:
/// `capacity.ts` sends `max` = the viewer's pixel WIDTH, and a waveform renders one min/max pair
/// per pixel column. (This doc line used to say "at most `max` entries", which reads as a bug in
/// `envelope_axis`. It is not — halving the bin count to satisfy that wording would halve every
/// waveform's resolution.) `Subsample` and `Area` emit one value per bin, so for them bins and
/// entries coincide.
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
///  3. group by canonical dim → `max(max)`, union of the requested kernels (largest-need-per-dim),
///  4. aspect-preserve: if ≥2 axes reduce via `Area` (an image's H,W), scale them by ONE
///     factor so a non-square source keeps its aspect ratio.
///
/// The per-axis accumulation — `(max(max), set-union of the methods)` — happens at EVERY
/// collision (multiple specs on one dim; a `-1` from one viewer and a `+1` from another folding
/// onto the same physical axis after canonicalization), in one place, and both halves are
/// order-independent by construction. The kernel is chosen ONCE, by
/// [`MethodSet::resolve`], from the full set.
pub fn plan<R: Reducible + ?Sized>(specs: &[ViewSpec], frame: &R) -> MergedViewSpec {
    let ndim = frame.ndim();
    let mut order: Vec<usize> = Vec::new(); // first-seen dim order → stable output
    let mut folded: HashMap<usize, (usize, MethodSet)> = HashMap::new();
    for spec in specs {
        if !spec.admits(frame) {
            continue;
        }
        for r in &spec.reduce {
            let Some(d) = canon_dim(r.dim, ndim) else {
                continue;
            };
            let entry = folded.entry(d).or_insert_with(|| {
                order.push(d);
                (0, MethodSet::default())
            });
            entry.0 = entry.0.max(r.max);
            entry.1.add(r.method);
        }
    }
    let mut axes: Vec<PlannedAxis> = order
        .iter()
        .map(|&d| {
            let (mx, set) = folded[&d];
            PlannedAxis { dim: d, max: mx, method: set.resolve() }
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
