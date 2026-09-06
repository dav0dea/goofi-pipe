//! Select — keep part of one axis, named three ways: by label, by position, or by the coordinate
//! range a labelled axis carries, which is how a spectrum is cut to a band.

use goofi_core::{resolve_axis, Coord, Data, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Select;

/// A glob with `*` standing for any run of characters.
fn matches(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else { return false };
    if !name.starts_with(first) {
        return false;
    }
    let mut at = first.len();
    let mut last = None;
    for part in parts {
        last = Some(part);
        if part.is_empty() {
            continue;
        }
        match name[at..].find(part) {
            Some(found) => at += found + part.len(),
            None => return false,
        }
    }
    match last {
        // No `*` at all: the whole name had to be the pattern.
        None => name.len() == at,
        Some(tail) => name.ends_with(tail) && name.len() >= at,
    }
}

/// The labels of `axis`, as the strings a pattern is matched against.
fn names(d: &Data, axis: usize, len: usize) -> Vec<String> {
    match d.meta().channels().get(axis).and_then(|a| a.coords.clone()) {
        Some(coords) => coords
            .iter()
            .map(|c| match c {
                Coord::Str(s) => s.to_string(),
                Coord::Num(n) => n.to_string(),
            })
            .collect(),
        None => (0..len).map(|i| i.to_string()).collect(),
    }
}

fn by_name(spec: &str, labels: &[String]) -> Vec<usize> {
    let mut picked = Vec::new();
    for pattern in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        for (i, name) in labels.iter().enumerate() {
            if matches(pattern, name) && !picked.contains(&i) {
                picked.push(i);
            }
        }
    }
    picked
}

fn by_index(spec: &str, len: usize) -> Result<Vec<usize>, String> {
    let mut picked = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let mut push = |i: usize| {
            if i < len && !picked.contains(&i) {
                picked.push(i);
            }
        };
        match part.split_once(':') {
            // A slice, in the reading every language shares: the end is not included.
            Some((from, to)) => {
                let from: usize = from.trim().parse().unwrap_or(0);
                let to: usize = to.trim().parse().unwrap_or(len);
                for i in from..to.min(len) {
                    push(i);
                }
            }
            None => push(part.parse().map_err(|_| format!("`{part}` is not a position"))?),
        }
    }
    Ok(picked)
}

fn by_range(spec: &str, coords: &[Coord]) -> Vec<usize> {
    let mut picked = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (from, to) = match part.split_once(':') {
            Some((a, b)) => (a.trim().parse().unwrap_or(f64::NEG_INFINITY), b.trim().parse().unwrap_or(f64::INFINITY)),
            None => {
                let one = part.parse().unwrap_or(f64::NAN);
                (one, one)
            }
        };
        for (i, c) in coords.iter().enumerate() {
            // Both ends are included: a band written 8:13 holds 13.
            if let Coord::Num(n) = c {
                if *n >= from && *n <= to && !picked.contains(&i) {
                    picked.push(i);
                }
            }
        }
    }
    picked
}

impl Node for Select {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        let a = d.assert_ndims().at_least(1)?;
        let shape = a.shape();
        let axis = resolve_axis(p.i64("select", "axis").unwrap_or(0), shape.len())?;
        let len = shape[axis];
        let mode = p.str("select", "mode").unwrap_or("name");
        let include = p.str("select", "include").unwrap_or("");
        let exclude = p.str("select", "exclude").unwrap_or("");

        let coords = d.meta().channels().get(axis).and_then(|x| x.coords.clone());
        let pick = |spec: &str| -> Result<Vec<usize>, String> {
            match mode {
                "index" => by_index(spec, len),
                "range" => Ok(by_range(spec, coords.as_deref().unwrap_or(&[]))),
                _ => Ok(by_name(spec, &names(d, axis, len))),
            }
        };
        let mut kept = if include.trim().is_empty() { (0..len).collect() } else { pick(include)? };
        if !exclude.trim().is_empty() {
            let dropped = pick(exclude)?;
            kept.retain(|i| !dropped.contains(i));
        }
        if kept.is_empty() {
            return Err("nothing is left after the selection".to_string().into());
        }

        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let src = a.as_bytes();
        let block = inner * 4;
        let mut buf = Vec::with_capacity(outer * kept.len() * block);
        for o in 0..outer {
            for &k in &kept {
                let from = (o * len + k) * block;
                buf.extend_from_slice(&src[from..from + block]);
            }
        }

        let mut shape_out = shape.to_vec();
        shape_out[axis] = kept.len();
        let mut meta = d.meta().keep(axis, &kept);
        if kept.len() == 1 && p.bool("select", "squeeze").unwrap_or(false) {
            shape_out.remove(axis);
            meta = d.meta().keep(axis, &kept).drop_axis(axis, shape.len());
        }
        out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "select",
        name: "axis",
        spec: ParamSpec::Int { default: 0, min: -8, max: 7 },
        expression: None,
        doc: Some("Which axis to cut, negative from the end. 0 is channels on a `[channels, time]` frame."),
    },
    ParamDecl {
        group: "select",
        name: "mode",
        spec: ParamSpec::Str { default: "name", options: &["name", "index", "range"], refresh: false },
        expression: None,
        doc: Some(
            "How the entries are named: by their labels, by their positions, or by the coordinate \
             range a labelled axis carries, which is how a spectrum is cut to a band.",
        ),
    },
    ParamDecl {
        group: "select",
        name: "include",
        spec: ParamSpec::Str { default: "", options: &[], refresh: false },
        expression: None,
        doc: Some(
            "What to keep, separated by commas: labels with `*` standing for anything, positions \
             like `0,2,4:8`, or ranges like `8:13` with both ends included. Empty keeps everything.",
        ),
    },
    ParamDecl {
        group: "select",
        name: "exclude",
        spec: ParamSpec::Str { default: "", options: &[], refresh: false },
        expression: None,
        doc: Some("What to drop from the kept set, written the same way. Empty drops nothing."),
    },
    ParamDecl {
        group: "select",
        name: "squeeze",
        spec: ParamSpec::Bool { default: false },
        expression: None,
        doc: Some("When one entry is left, remove the axis instead of leaving it one long."),
    },
];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Array }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform],
    doc: "Keep part of one axis.\n\
          Named by label, by position, or by the coordinate range it carries.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Select, MANIFEST);
