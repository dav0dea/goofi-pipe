//! The read ops an agent uses to see what it built: a scope as a diagram, a node as a page of
//! text. Pure reads — they clone what they need and format off the graph lock, never wait, never
//! sample, and never write a file.

use goofi_core::{MetaValue, Value as DataValue};
use goofi_engine::{Graph, Uid};
use serde_json::{json, Value};

/// A uid as a mermaid node id. Mermaid ids may not start with a digit, so every node id is the uid
/// with an `n` in front — and the raw uid rides the LABEL too, so a reader never has to know that.
fn mid(uid: Uid) -> String {
    format!("n{}", uid.to_hex())
}

/// A display name safe inside a mermaid `"…"` label.
fn label(name: &str) -> String {
    name.replace(['"', '\n'], "'")
}

/// The direct members of a scope — leaves and nested sub-patch facades. `None` is the root scope,
/// whose members are everything the `scope_of` tree does not place anywhere else.
fn members(g: &Graph, scope: Option<Uid>) -> Vec<Uid> {
    match scope {
        Some(s) => g.scope_members(s),
        None => g
            .node_uids()
            .into_iter()
            .chain(g.scope_uids())
            .filter(|u| g.scope_of(*u).is_none())
            .collect(),
    }
}

/// Which member of `scope` contains `uid` — `uid` itself when it is a direct member, else the
/// ancestor sub-patch facade it lives inside. `None` when it is somewhere else entirely. This is
/// what lets a flat leaf→leaf link draw as an edge onto a collapsed sub-patch.
fn member_in(g: &Graph, scope: Option<Uid>, uid: Uid) -> Option<Uid> {
    let mut at = uid;
    loop {
        if g.scope_of(at) == scope {
            return Some(at);
        }
        at = g.scope_of(at)?;
    }
}

/// A scope's full path from the root, as display names.
fn scope_path(g: &Graph, scope: Uid) -> String {
    let mut parts = vec![g.scope(scope).map_or("?".into(), |s| s.name.clone())];
    let mut at = scope;
    while let Some(parent) = g.scope_of(at) {
        parts.push(g.scope(parent).map_or("?".into(), |s| s.name.clone()));
        at = parent;
    }
    parts.reverse();
    parts.join("/")
}

/// A node's full path, so a whole-patch error listing says WHERE the node is.
fn node_path(g: &Graph, uid: Uid) -> String {
    let name = g.name(uid).map(str::to_string).unwrap_or_else(|| uid.to_hex());
    match g.scope_of(uid) {
        Some(s) => format!("{}/{name}", scope_path(g, s)),
        None => name,
    }
}

/// One second, one decimal — enough to tell a settling pipeline from a broken one, and no more
/// precision than a 2 Hz observer could act on.
fn age(g: &Graph, uid: Uid) -> String {
    match g.error_age(uid) {
        Some(d) => format!(" — for {:.1}s", d.as_secs_f64()),
        None => String::new(),
    }
}

/// `inspect_patch`: one scope drawn as a mermaid flowchart, under a header that identifies the
/// patch, followed by every standing error in the WHOLE patch with the age of each.
pub fn patch(
    g: &Graph,
    scope: Option<Uid>,
    save_path: Option<&str>,
    workspace: &str,
    dirty: bool,
) -> Result<String, String> {
    if let Some(s) = scope {
        if g.scope(s).is_none() {
            return Err(format!("inspect_patch: no sub-patch `{}`", s.to_hex()));
        }
    }
    let mut out = format!(
        "patch: {}\nworkspace: {workspace}\nunsaved changes: {}\nscope: {}\n",
        save_path.unwrap_or("(never saved)"),
        if dirty { "yes" } else { "no" },
        scope.map_or("root".to_string(), |s| format!("{} ({})", scope_path(g, s), s.to_hex())),
    );

    let member = members(g, scope);
    let stubs = scope.and_then(|s| g.scope(s)).map(|s| &s.stubs);
    if member.is_empty() && stubs.is_none_or(|s| s.is_empty()) {
        out.push_str("\n(no nodes)\n");
    } else {
        out.push_str("\n```mermaid\nflowchart LR\n");
        // Boundary ports first: their mermaid ids ARE their stub ids, verbatim, because that is
        // what `wire_boundary` and friends address them by.
        for (id, stub) in stubs.into_iter().flatten() {
            out.push_str(&format!(
                "  {id}([\"{} · {} {}\"])\n",
                label(&stub.name),
                stub.dir.name(),
                stub.dtype.name()
            ));
        }
        for &uid in &member {
            let hex = uid.to_hex();
            match g.scope(uid) {
                Some(s) => out.push_str(&format!("  {}[[\"{}<br/>{hex}\"]]\n", mid(uid), label(&s.name))),
                None => {
                    let warn = if g.last_error(uid).is_some() { "⚠ " } else { "" };
                    out.push_str(&format!(
                        "  {}[\"{warn}{}: {}<br/>{hex}\"]\n",
                        mid(uid),
                        label(g.name(uid).unwrap_or("?")),
                        g.type_name(uid).unwrap_or("?"),
                    ));
                }
            }
        }
        // The runtime links are always flat leaf→leaf, so each end is folded onto whichever member
        // of THIS scope contains it — that is how a wire into a collapsed sub-patch gets drawn.
        let mut edges: Vec<String> = Vec::new();
        for l in g.links_view() {
            let (Some(a), Some(b)) =
                (member_in(g, scope, l.node_out), member_in(g, scope, l.node_in))
            else {
                continue;
            };
            // Both ends folding onto ONE member means the wire is internal to a collapsed
            // sub-patch — a fact one level down. Unless the member IS both endpoints: a node wired
            // to its own input is a real self-loop at this level, and drawing it is the point.
            if a == b && (l.node_out != a || l.node_in != b) {
                continue;
            }
            let e = format!("  {} -- {}→{} --> {}\n", mid(a), l.slot_out, l.slot_in, mid(b));
            if !edges.contains(&e) {
                edges.push(e);
            }
        }
        // A port's own wire: the stub's inner side, which is not a flat link and so is not above.
        for (id, stub) in stubs.into_iter().flatten() {
            let Some((inner, slot)) = &stub.inner else { continue };
            let Some(m) = member_in(g, scope, *inner) else { continue };
            edges.push(match stub.dir {
                goofi_engine::subpatch::Dir::In => format!("  {id} -- {slot} --> {}\n", mid(m)),
                goofi_engine::subpatch::Dir::Out => format!("  {} -- {slot} --> {id}\n", mid(m)),
            });
        }
        out.extend(edges);
        out.push_str("```\n\nuids: a node's uid is its mermaid id without the leading `n`; a \
                      boundary port's id is its mermaid id verbatim.\n");
    }

    let errored: Vec<Uid> = g.node_uids().into_iter().filter(|u| g.last_error(*u).is_some()).collect();
    out.push_str("\nerrors (whole patch):\n");
    if errored.is_empty() {
        out.push_str("  none\n");
    }
    for uid in errored {
        out.push_str(&format!(
            "  ⚠ {} ({}): {}{}\n",
            node_path(g, uid),
            uid.to_hex(),
            g.last_error(uid).unwrap_or(""),
            age(g, uid),
        ));
    }
    Ok(out)
}

/// One param as the goldened inline form — the one an agent reads and then feeds straight back
/// into `update_param` (`group.name` + value) or `set_expression` (the source + enabled).
fn param_line(p: &goofi_core::Param, expr: Option<&goofi_engine::ExprInfo>) -> String {
    use goofi_core::Param as P;
    // Value and declared type in ONE match, so a variant cannot be described by one arm and
    // rendered by another.
    let (value, ty) = match p {
        P::Float { value, vmin, vmax } => (format!("{value}"), format!("float {vmin}..{vmax}")),
        P::Int { value, vmin, vmax } => (format!("{value}"), format!("int {vmin}..{vmax}")),
        P::Bool { value } => (format!("{value}"), "bool".to_string()),
        P::Trigger { fired } => (format!("{fired}"), "trigger".to_string()),
        P::Str { value, options: Some(o), .. } => {
            (format!("\"{value}\""), format!("string one of [{}]", o.join(", ")))
        }
        P::Str { value, .. } => (format!("\"{value}\""), "string".to_string()),
    };
    match expr {
        // A bound param shows its SOURCE and what it currently evaluates to, which is what
        // `set_expression` takes back; the declared range belongs to the literal it replaced.
        Some(e) => format!(
            "expr: {} → {value} ({}){}",
            e.source,
            if e.enabled { "on" } else { "off" },
            e.error.as_ref().map(|e| format!(" [error: {e}]")).unwrap_or_default(),
        ),
        None => format!("{value} ({ty})"),
    }
}

/// The value-health line for one emitted frame: what it is, how much of it is a real number, and
/// what scale it is on. This is how a NaN, a silent all-zeros or a wrong unit becomes visible
/// without writing a single byte to disk.
///
/// The range is over the FINITE values deliberately: that a non-finite is present is already said
/// by `finite=`, while a range that reads `nan` would hide the scale, which is the other half of
/// the question. `finite=` is one bucket, so it does not separate a NaN (0/0) from an overflow
/// (∞) — the next question, and one an agent asks with `inspect_node slot=…` on the producer.
///
/// ONE streaming pass, deliberately: a 4K RGB frame is 25 M elements and this runs under the graph
/// lock the tick thread needs — materializing it, twice, cost 317 ms and 189 MB of that lock.
fn health(d: &goofi_core::Data) -> String {
    match d.value() {
        DataValue::Array(a) => {
            let (mut n, mut finite) = (0usize, 0usize);
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for c in a.as_bytes().chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                n += 1;
                if v.is_finite() {
                    (finite, lo, hi) = (finite + 1, lo.min(v), hi.max(v));
                }
            }
            let shape: Vec<String> = a.shape().iter().map(|d| d.to_string()).collect();
            let range =
                if finite == 0 { "range=none".to_string() } else { format!("range=[{lo},{hi}]") };
            format!("f32[{}] finite={finite}/{n} {range}", shape.join(","))
        }
        DataValue::Str(s) => format!("string len={}", s.chars().count()),
        DataValue::Table(t) => format!("table keys={}", t.len()),
    }
}

/// A meta value in one short line; `None` for the unset builtins so they do not fill the page.
fn meta_val(v: &MetaValue) -> Option<String> {
    Some(match v {
        MetaValue::Null => return None,
        MetaValue::Bool(b) => b.to_string(),
        MetaValue::Int(i) => i.to_string(),
        MetaValue::Uint(u) => u.to_string(),
        MetaValue::Float(f) => f.to_string(),
        MetaValue::Str(s) => s.clone(),
        MetaValue::List(l) => format!("[{}]", l.iter().filter_map(meta_val).collect::<Vec<_>>().join(", ")),
        MetaValue::Map(m) => format!("{{{}}}", m.keys().cloned().collect::<Vec<_>>().join(", ")),
        MetaValue::Bytes(b) => format!("<{} bytes>", b.len()),
        MetaValue::Axes(a) if a.is_empty() => return None,
        MetaValue::Axes(a) => format!("{} labelled dims", a.0.len()),
    })
}

/// `inspect_node`: the cheap peek — what the node is, what its params say, what its outputs are
/// actually carrying, and whether it is erroring.
pub fn node(
    g: &Graph,
    uid: Uid,
    slot: Option<&str>,
    want_params: bool,
    want_meta: bool,
    want_error: bool,
) -> Result<String, String> {
    let manifest = g.manifest(uid).ok_or_else(|| format!("inspect_node: no node `{}`", uid.to_hex()))?;
    let tier = match manifest.isolation {
        goofi_node::Isolation::InProcess => "in-process",
        goofi_node::Isolation::Subprocess => "subprocess",
    };
    let mut out = format!(
        "{}: {} (uid {}, {tier}, stage {})\n",
        g.name(uid).unwrap_or("?"),
        manifest.type_name,
        uid.to_hex(),
        g.node_stage(uid),
    );
    if let Some(s) = slot {
        if !manifest.outputs.iter().any(|o| o.name == s) {
            return Err(format!(
                "inspect_node: `{}` has no output slot `{s}` (it has: {})",
                manifest.type_name,
                manifest.outputs.iter().map(|o| o.name).collect::<Vec<_>>().join(", "),
            ));
        }
    }

    if want_params {
        out.push_str("\nparams:\n");
        for (group, names) in g.params(uid).into_iter().flatten() {
            for (name, p) in names {
                let expr = g.param_expression(uid, group, name);
                out.push_str(&format!("  {group}.{name} = {}\n", param_line(p, expr.as_ref())));
            }
        }
    }

    out.push_str("\noutputs:\n");
    if manifest.outputs.is_empty() {
        out.push_str("  (none)\n");
    }
    for o in manifest.outputs.iter().filter(|o| slot.is_none_or(|s| s == o.name)) {
        match g.latest_frame(uid, o.name) {
            None => out.push_str(&format!("  {}: {} — nothing emitted yet\n", o.name, o.kind.name())),
            Some(d) => {
                out.push_str(&format!("  {}: {}\n", o.name, health(&d)));
                if want_meta {
                    let meta: Vec<String> =
                        d.meta().iter().filter_map(|(k, v)| meta_val(v).map(|v| format!("{k}={v}"))).collect();
                    if !meta.is_empty() {
                        out.push_str(&format!("    meta: {}\n", meta.join(", ")));
                    }
                }
            }
        }
    }

    if want_error {
        out.push_str(&match g.last_error(uid) {
            Some(e) => format!("\nerror: {e}{}\n", age(g, uid)),
            None => "\nerror: none\n".to_string(),
        });
    }
    Ok(out)
}

/// `list_globals`: what an expression can read and `set_global` can write.
pub fn globals(g: &Graph) -> Value {
    let entries: Vec<Value> = g
        .globals()
        .entries()
        .map(|(name, v, system)| {
            json!({
                "name": name,
                "type": v.type_tag(),
                "value": goofi_engine::global_to_json(v)["value"],
                "system": system,
            })
        })
        .collect();
    json!({ "globals": entries })
}

/// `read_node_source`: a node type's text where it has one, and its provenance either way. A
/// native type is compiled in — there is nothing to read and nothing to edit, which the reply
/// says rather than leaving the caller to infer from a null.
pub fn node_source(g: &Graph, ty: &str, dirs: &[(std::path::PathBuf, &str)]) -> Result<Value, String> {
    let native = goofi_node::find(ty);
    let manifest = native
        .or_else(|| g.dyn_type_manifests().into_iter().find(|m| m.type_name == ty))
        .ok_or_else(|| format!("read_node_source: no node type `{ty}` (see list_nodes)"))?;
    let mut info = crate::schemas::node_type_info(
        manifest,
        if g.is_patch_type(ty) { "patch" } else { "builtin" },
    );
    // The file a discovered node came from: the type name is its CamelCased stem, and the scan
    // walks exactly these directories, so re-deriving the path here needs no second registry.
    let found = dirs.iter().find_map(|(dir, provenance)| {
        let entries = std::fs::read_dir(dir).ok()?;
        let path = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.extension().is_some_and(|e| e == "py")
                    && p.file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| goofi_node::discover::camel(s) == ty)
            })?;
        Some((path, *provenance))
    });
    info["language"] = json!(if native.is_some() { "rust" } else { "python" });
    info["tier"] = json!(match (native.is_some(), manifest.isolation) {
        (true, _) => "native",
        (_, goofi_node::Isolation::Subprocess) => "subprocess",
        _ => "in-process",
    });
    info["provenance"] = json!(match &found {
        Some((_, p)) => *p,
        None => "compiled in — no source file; copy a python node into the patch workspace to modify one",
    });
    info["path"] =
        found.as_ref().map(|(p, _)| json!(goofi_core::path::to_slash(p))).unwrap_or(Value::Null);
    info["source"] = found
        .as_ref()
        .and_then(|(p, _)| std::fs::read_to_string(p).ok())
        .map(Value::String)
        .unwrap_or(Value::Null);
    Ok(info)
}

// ---------------------------------------------------------------------------
// The arrangement — the flat layout read back as something a caller can navigate
// ---------------------------------------------------------------------------

use goofi_engine::layout::{Entry, Layout};

/// One entry's line in the tree, and its children under it. A split names its axis, a panel its
/// type and binding; both carry the share of their parent they take, which is the number a caller
/// adjusts with `page_resize_split`.
fn layout_line(l: &Layout, id: &str, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    match l.get(id) {
        Some(Entry::Page { name, .. }) => out.push_str(&format!("{pad}page `{name}`  [{id}]\n")),
        Some(Entry::Split { size, axis, .. }) => {
            out.push_str(&format!("{pad}{} split {size:.2}  [{id}]\n", axis.name()))
        }
        Some(Entry::Panel { size, panel_type, state, .. }) => {
            let bound = state.get("node").and_then(|v| v.as_str()).map(|n| format!(" → {n}"));
            out.push_str(&format!(
                "{pad}{panel_type}{} {size:.2}  [{id}]\n",
                bound.unwrap_or_default()
            ))
        }
        None => {}
    }
    for c in l.children(id) {
        layout_line(l, &c, depth + 1, out);
    }
}

/// The arrangement as an indented tree — the ONE layout read. `page` narrows it to a single page,
/// the way `inspect_patch {scope}` narrows the graph: a caller working on one page pays for one
/// page, and a caller that needs the split id to name as a move target reads them all.
pub fn layout_tree(l: &Layout, page: Option<&str>) -> String {
    let mut out = String::from(
        "The editor arrangement. Pages are addressed by NAME; panels and splits by the id in []. \
         The number on each entry is its share of its parent — what page_resize_split sets.\n\n",
    );
    let pages = match page {
        Some(p) => vec![p.to_string()],
        None => l.pages(),
    };
    for p in pages {
        layout_line(l, &p, 0, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{Data, Meta, SlotType};
    use goofi_engine::subpatch::Dir;
    use goofi_node::{Inputs, NodeCtx, NodeManifest, NodeResult, Outputs, Params};

    /// A node that always fails, so a fixture can hold a red node.
    struct Boom;
    impl goofi_node::Node for Boom {
        fn process(&mut self, _i: &Inputs<'_>, _o: &mut Outputs<'_>, _c: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
            Err("the sensor is unplugged".into())
        }
    }
    static BOOM_OUT: &[goofi_node::OutputDecl] =
        &[goofi_node::OutputDecl { name: "out", kind: SlotType::Array }];
    static BOOM: NodeManifest = NodeManifest {
        type_name: "Boom",
        category: "test",
        doc: "always fails",
        inputs: &[],
        outputs: BOOM_OUT,
        params: &[],
        isolation: goofi_node::Isolation::InProcess,
        producer: true,
        factory: || Box::new(Boom),
    };

    /// A patch with a sub-patch (holding a Buffer behind an input boundary port), a top-level
    /// source wired through that port, and a node that is erroring.
    fn fixture() -> (Graph, Uid) {
        let mut g = Graph::new();
        g.register_dyn_type(&BOOM, Box::new(|_| Box::new(Boom)));
        let osc = g.add_node("Oscillator", None).unwrap();
        let _boom = g.add_node("Boom", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();
        let scope = g.group_nodes(&[buf], [40.0, 10.0]).unwrap();
        let bnd = g.add_boundary(scope, Dir::In, SlotType::Array, [0.0, 0.0]).unwrap();
        g.set_stub_inner(scope, &bnd, Some((buf, "data".into()))).unwrap();
        g.add_link(osc, "out", buf, "data").unwrap();
        g.tick();
        (g, scope)
    }

    /// Ages are wall-clock, so the golden pins everything BUT the number — while still requiring
    /// that an age is there at all (a line without one keeps its text and fails the compare).
    fn ages_out(s: &str) -> String {
        s.split_inclusive('\n')
            .map(|l| match l.split_once(" — for ") {
                Some((head, tail)) => format!("{head} — for <age>{}", &tail[tail.len() - 1..]),
                None => l.to_string(),
            })
            .collect()
    }

    #[test]
    fn inspect_patch_draws_the_root_scope_and_the_whole_patchs_errors() {
        let (g, _) = fixture();
        let out = patch(&g, None, Some("/patches/demo.gfi"), "/tmp/goofi-abc/workspace", true).unwrap();
        assert_eq!(
            ages_out(&out),
            "\
patch: /patches/demo.gfi
workspace: /tmp/goofi-abc/workspace
unsaved changes: yes
scope: root

```mermaid
flowchart LR
  n000000000001[\"oscillator0: Oscillator<br/>000000000001\"]
  n000000000002[\"⚠ boom0: Boom<br/>000000000002\"]
  n000000000004[[\"subpatch4<br/>000000000004\"]]
  n000000000001 -- out→data --> n000000000004
```

uids: a node's uid is its mermaid id without the leading `n`; a boundary port's id is its mermaid id verbatim.

errors (whole patch):
  ⚠ boom0 (000000000002): the sensor is unplugged — for <age>
"
        );
    }

    #[test]
    fn inspect_patch_draws_a_sub_patchs_boundary_ports_by_their_own_ids() {
        let (g, scope) = fixture();
        let out = patch(&g, Some(scope), None, "/tmp/w", false).unwrap();
        assert_eq!(
            ages_out(&out),
            "\
patch: (never saved)
workspace: /tmp/w
unsaved changes: no
scope: subpatch4 (000000000004)

```mermaid
flowchart LR
  in0([\"in0 · in ARRAY\"])
  n000000000003[\"buffer0: Buffer<br/>000000000003\"]
  in0 -- data --> n000000000003
```

uids: a node's uid is its mermaid id without the leading `n`; a boundary port's id is its mermaid id verbatim.

errors (whole patch):
  ⚠ boom0 (000000000002): the sensor is unplugged — for <age>
"
        );
    }

    #[test]
    fn a_wire_inside_a_collapsed_sub_patch_is_not_drawn_as_a_self_loop_on_its_facade() {
        // Both ends of an internal wire fold onto the same facade. Drawing it would put a loop on
        // the sub-patch that states nothing about the scope being read — the wire is a fact one
        // level down, and `inspect_patch {scope}` is where it belongs.
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        g.add_link(a, "out", b, "data").unwrap();
        g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();
        let out = patch(&g, None, None, "/tmp/w", false).unwrap();
        assert!(out.contains("[["), "the facade is drawn: {out}");
        assert!(!out.contains("-->"), "…with no edge at all at this level: {out}");
    }

    #[test]
    fn a_node_wired_to_itself_keeps_its_edge() {
        // The fold above collapses an internal wire onto one facade, which is why `a == b` is
        // skipped — but a node wired to its OWN input folds onto itself for the honest reason.
        // `add_link` accepts that and the engine tolerates the cycle, so the diagram must show it:
        // an agent debugging a feedback loop would otherwise read that the wire is not there.
        let mut g = Graph::new();
        let buf = g.add_node("Buffer", None).unwrap();
        g.add_link(buf, "out", buf, "data").unwrap();
        let out = patch(&g, None, None, "/tmp/w", false).unwrap();
        assert!(out.contains(&format!("{0} -- out→data --> {0}\n", mid(buf))), "{out}");
    }

    #[test]
    fn an_empty_scope_says_so_rather_than_drawing_an_empty_diagram() {
        let mut g = Graph::new();
        let out = patch(&g, None, None, "/tmp/w", false).unwrap();
        assert!(out.contains("(no nodes)"), "{out}");
        assert!(!out.contains("mermaid"), "no diagram for an empty scope: {out}");
        assert!(out.contains("errors (whole patch):\n  none"), "{out}");
        // A scope uid that names nothing is refused rather than drawn as empty.
        let n = g.add_node("Oscillator", None).unwrap();
        assert!(patch(&g, Some(n), None, "/tmp/w", false).is_err(), "a leaf is not a scope");
    }

    #[test]
    fn a_frame_with_a_nan_reads_as_unhealthy_while_keeping_its_scale() {
        let vals: Vec<f32> = vec![0.5, f32::NAN, -0.25, 0.98];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d = Data::array_f32(vec![2, 2], bytes, Meta::empty()).unwrap();
        assert_eq!(health(&d), "f32[2,2] finite=3/4 range=[-0.25,0.98]");

        // An all-NaN frame has no scale to report, and says that instead of inventing one.
        let nans: Vec<u8> = [f32::NAN; 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let d = Data::array_f32(vec![2], nans, Meta::empty()).unwrap();
        assert_eq!(health(&d), "f32[2] finite=0/2 range=none");

        // An overflow is non-finite too, and it must not drag the range out to infinity — the
        // scale of the values that ARE real is the whole point of reporting one.
        let mixed: Vec<u8> =
            [f32::INFINITY, 2.0, f32::NEG_INFINITY, -1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let d = Data::array_f32(vec![4], mixed, Meta::empty()).unwrap();
        assert_eq!(health(&d), "f32[4] finite=2/4 range=[-1,2]");

        // An empty frame: no elements to fold, and so no scale — the degenerate case a running
        // min/max has to answer without inventing `inf` as the range.
        let d = Data::array_f32(vec![0], Vec::new(), Meta::empty()).unwrap();
        assert_eq!(health(&d), "f32[0] finite=0/0 range=none");
    }

    #[test]
    fn inspect_node_reports_params_output_health_and_the_error() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        g.set_expression(osc, "oscillator", "amplitude", "globals.default_ufreq / 30", true, false)
            .unwrap();
        // The oscillator emits by wall clock, so a frame needs time to pass, not just a tick.
        for _ in 0..50 {
            g.tick();
            if g.latest_frame(osc, "out").is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let out = node(&g, osc, None, true, true, true).unwrap();
        assert!(
            out.starts_with(&format!("oscillator0: Oscillator (uid {}, in-process, stage ready)", osc.to_hex())),
            "{out}"
        );
        // The goldened inline param format, round-trippable into update_param…
        assert!(out.contains("  oscillator.frequency = 1 (float 0..100)"), "{out}");
        assert!(
            out.contains("  common.frequency_mode = \"updates-per-second\" (string one of [updates-per-second, "),
            "{out}"
        );
        // …and into set_expression, for a param that is bound instead of literal. This binding
        // cannot compile (no evaluator in this crate), which is itself worth showing inline.
        assert!(
            out.contains("  oscillator.amplitude = expr: globals.default_ufreq / 30 → 1 (on) [error: "),
            "{out}"
        );
        assert!(out.contains("  out: f32["), "the value-health line: {out}");
        assert!(out.contains("    meta: sfreq=250"), "the frame's meta: {out}");
        // A node's own error is age-annotated here too, for the same settling-vs-broken reason.
        assert!(out.contains("\nerror: no expression evaluator available — for 0."), "{out}");

        // A node with nothing wrong says so plainly.
        let idle = g.add_node("Buffer", None).unwrap();
        assert!(node(&g, idle, None, false, false, true).unwrap().ends_with("error: none\n"));

        // The flags actually gate their sections.
        let bare = node(&g, osc, None, false, false, false).unwrap();
        assert!(!bare.contains("params:") && !bare.contains("meta:") && !bare.contains("error:"), "{bare}");

        // An unknown slot is refused by naming the ones that exist.
        let err = node(&g, osc, Some("psd"), true, true, true).unwrap_err();
        assert!(err.contains("no output slot `psd`") && err.contains("out"), "{err}");
    }

    #[test]
    fn list_globals_names_the_system_globals_an_expression_can_read() {
        let g = Graph::new();
        let v = globals(&g);
        let first = &v["globals"][0];
        assert_eq!(first["name"], json!("default_ufreq"));
        assert_eq!(first["type"], json!("float"));
        assert_eq!(first["value"], json!(30.0));
        assert_eq!(first["system"], json!(true));
    }

    #[test]
    fn read_node_source_says_a_native_type_has_no_file_to_edit() {
        let g = Graph::new();
        let v = node_source(&g, "Oscillator", &[]).unwrap();
        assert_eq!(v["language"], json!("rust"));
        assert_eq!(v["tier"], json!("native"));
        assert_eq!(v["source"], Value::Null);
        assert!(v["provenance"].as_str().unwrap().contains("copy a python node"), "{v}");
        // The manifest a caller needs instead comes along.
        assert_eq!(v["output_slots"]["out"], json!("ARRAY"));
        assert!(node_source(&g, "Nope", &[]).unwrap_err().contains("no node type `Nope`"));
    }

    #[test]
    fn read_node_source_finds_a_discovered_types_file_by_re_deriving_its_name() {
        // The half a caller actually wants: the text to edit, the path to write it back to, and
        // which tree it came from. The path is re-derived from the type name rather than recorded,
        // so this pins the derivation — a scan stores no path for it to disagree with.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("boom.py"), "class Boom:\n    pass\n").unwrap();
        let mut g = Graph::new();
        g.register_dyn_type(&BOOM, Box::new(|_| Box::new(Boom)));

        let v = node_source(&g, "Boom", &[(dir.path().to_path_buf(), "patch")]).unwrap();
        assert_eq!(v["provenance"], json!("patch"), "{v}");
        assert_eq!(v["path"], json!(goofi_core::path::to_slash(&dir.path().join("boom.py"))), "{v}");
        assert_eq!(v["source"], json!("class Boom:\n    pass\n"), "{v}");
        assert_eq!(v["language"], json!("python"), "{v}");

        // A tree that holds no file of that name leaves the discovery half empty and SAYS so,
        // rather than handing back a bare null the caller has to interpret.
        let v = node_source(&g, "Boom", &[(dir.path().join("nope"), "patch")]).unwrap();
        assert_eq!(v["source"], Value::Null);
        assert!(v["provenance"].as_str().unwrap().contains("compiled in"), "{v}");
    }
}
