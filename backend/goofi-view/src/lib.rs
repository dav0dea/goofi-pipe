//! The shared, payload-free ViewSpec algebra: a viewer publishes what it can draw and what it
//! wants reduced, and N specs merge into ONE plan per frame.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The Data kind a viewer draws; the tags are goofi-core's wire dtype tags, restated to keep
/// this crate free of that dependency.
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

/// A constraint on ONE dimension's length; a negative `dim` counts from the end.
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

/// Which kernels the admitted viewers asked for on ONE axis. A set, not a running pairwise
/// merge, so the fold cannot depend on the order the specs arrive in.
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

    /// The ONE method that must serve every subscriber; a cross-family conflict degrades to
    /// Subsample, the only reduction both an image and a line viewer can draw.
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

/// Desired reduction of one axis to `max` BINS via `method`; `Envelope` emits a (min, max) pair
/// per bin, so it returns `2 * max` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisReduce {
    pub dim: i32,
    pub max: usize,
    pub method: ReduceMethod,
}

/// The bins an axis is capped to for a viewer that has declared nothing; Subsample because it is
/// the one kernel every viewer family can draw.
pub const UNDECLARED_MAX: usize = 512;

/// One viewer's full declaration: what it can draw + what it wants reduced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewSpec {
    /// The Data kind this viewer draws.
    pub dtype: ViewDtype,
    /// Dim-count comparisons — ALL must hold to admit (empty ⇒ any ndim).
    #[serde(default)]
    pub ndim: Vec<(DimCmp, usize)>,
    /// Per-dim length comparisons (ALL must hold to admit).
    #[serde(default)]
    pub dims: Vec<DimConstraint>,
    /// Desired per-axis reductions.
    #[serde(default)]
    pub reduce: Vec<AxisReduce>,
}

/// A payload frame the reducer can query for shape.
pub trait Reducible {
    /// 0=array, 1=string, 2=table (matches the wire dtype tag).
    fn dtype_tag(&self) -> u8;
    /// Number of dimensions (0 for a non-array payload).
    fn ndim(&self) -> usize;
    /// The shape (empty for a non-array payload).
    fn shape(&self) -> &[usize];
}

/// Map a possibly-negative axis index to `0..ndim`, or `None` if out of range.
pub fn canon_dim(dim: i32, ndim: usize) -> Option<usize> {
    let d = if dim < 0 { dim + ndim as i32 } else { dim };
    (d >= 0 && (d as usize) < ndim).then_some(d as usize)
}

impl ViewSpec {
    /// What stands in for a viewer that has declared nothing yet: a preview, never the full frame,
    /// so a slot nobody has sized cannot cost the stream's whole rate.
    pub fn undeclared() -> ViewSpec {
        let cap = |dim| AxisReduce { dim, max: UNDECLARED_MAX, method: ReduceMethod::Subsample };
        ViewSpec { dtype: ViewDtype::Array, ndim: Vec::new(), dims: Vec::new(), reduce: vec![cap(0), cap(-1)] }
    }

    /// Whether this viewer can draw `frame`, and so joins the merge.
    pub fn admits<R: Reducible + ?Sized>(&self, frame: &R) -> bool {
        if frame.dtype_tag() != self.dtype.tag() {
            return false;
        }
        if frame.dtype_tag() != ViewDtype::Array.tag() {
            return true;
        }
        let ndim = frame.ndim();
        for &(cmp, n) in &self.ndim {
            if !cmp.holds(ndim, n) {
                return false;
            }
        }
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

/// One planned axis reduction, with `dim` already canonical.
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

/// Merge N viewers' specs into ONE concrete plan for THIS frame: specs that do not admit the
/// frame drop out, and each canonical dim folds to `max(max)` plus the union of the kernels.
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

/// Scale every `Area` axis by one shared factor, so a non-square image keeps its aspect ratio.
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
        return;
    }
    for &i in &area {
        axes[i].max = ((shape[axes[i].dim] as f64 * factor).round() as usize).max(1);
    }
}
