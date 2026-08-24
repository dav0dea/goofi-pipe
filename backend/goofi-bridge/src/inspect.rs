//! The read ops an agent uses to see what it built: a scope as a diagram, a node as a page of text.

use goofi_engine::{Graph, Uid};
use serde_json::{json, Value};

/// A uid as a mermaid node id: mermaid ids may not start with a digit, hence the leading `n`.
fn mid(uid: Uid) -> String {
    format!("n{}", uid.to_hex())
}

/// A display name safe inside a mermaid `"…"` label.
fn label(name: &str) -> String {
    name.replace(['"', '\n'], "'")
}

/// The direct members of a scope; `None` is the root, which holds everything `scope_of` places
/// nowhere else.
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

/// Which member of `scope` contains `uid` — itself, or the ancestor facade it lives inside.
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

/// A node's full path, so an error listing says where the node is.
fn node_path(g: &Graph, uid: Uid) -> String {
    let name = g.name(uid).map(str::to_string).unwrap_or_else(|| uid.to_hex());
    match g.scope_of(uid) {
        Some(s) => format!("{}/{name}", scope_path(g, s)),
        None => name,
    }
}

/// How long an error has stood, to one decimal of a second.
fn age(g: &Graph, uid: Uid) -> String {
    match g.error_age(uid) {
        Some(d) => format!(" — for {:.1}s", d.as_secs_f64()),
        None => String::new(),
    }
}

/// `inspect_patch`: ONE scope drawn as a mermaid flowchart, under a header that identifies the
/// patch. Scope-wide and nothing more — the patch's standing errors are `get_patch`'s.
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
        for (id, stub) in stubs.into_iter().flatten() {
            out.push_str(&format!(
                "  {}([\"{}: {}<br/>{}\"])\n",
                mid(*id),
                label(&stub.name),
                goofi_engine::subpatch::boundary_type_name(stub.dir, stub.dtype),
                id.to_hex(),
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
        // Runtime links are flat leaf→leaf, so each end folds onto the member of THIS scope that
        // contains it.
        let mut edges: Vec<String> = Vec::new();
        for l in g.links_view() {
            let (Some(a), Some(b)) =
                (member_in(g, scope, l.node_out), member_in(g, scope, l.node_in))
            else {
                continue;
            };
            // Both ends on ONE member is a wire internal to a collapsed sub-patch — unless the
            // member IS both endpoints, which is a real self-loop at this level.
            if a == b && (l.node_out != a || l.node_in != b) {
                continue;
            }
            let e = format!("  {} -- {}→{} --> {}\n", mid(a), l.slot_out, l.slot_in, mid(b));
            if !edges.contains(&e) {
                edges.push(e);
            }
        }
        // The stub's inner side is not a flat link, so it is not in the loop above.
        for (id, stub) in stubs.into_iter().flatten() {
            let Some((inner, slot)) = &stub.inner else { continue };
            let Some(m) = member_in(g, scope, *inner) else { continue };
            edges.push(match stub.dir {
                goofi_engine::subpatch::Dir::In => format!("  {} -- {slot} --> {}\n", mid(*id), mid(m)),
                goofi_engine::subpatch::Dir::Out => format!("  {} -- {slot} --> {}\n", mid(m), mid(*id)),
            });
        }
        out.extend(edges);
        out.push_str("```\n\nuids: a uid is its mermaid id without the leading `n`.\n");
    }

    Ok(out)
}

/// Every standing error in the patch, with the age of each — what `get_patch` answers, because a
/// patch's health is the patch's, not a scope's. `inspect_patch` drew it under whichever scope was
/// asked for, so the same list arrived again under each.
pub fn errors(g: &Graph) -> Vec<Value> {
    g.node_uids()
        .into_iter()
        .filter(|u| g.last_error(*u).is_some())
        .map(|uid| {
            serde_json::json!({
                "node": uid.to_hex(),
                "path": node_path(g, uid),
                "error": g.last_error(uid).unwrap_or(""),
                "standing": g.error_age(uid).map(|d| d.as_secs_f64()),
            })
        })
        .collect()
}

/// One param in the inline form an agent feeds straight back into `edit_node`.
fn param_line(p: &goofi_core::Param, expr: Option<&goofi_engine::ExprInfo>) -> String {
    use goofi_core::Param as P;
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
        Some(e) => format!(
            "expr: {} → {value} ({}){}",
            e.source,
            if e.enabled { "on" } else { "off" },
            e.error.as_ref().map(|e| format!(" [error: {e}]")).unwrap_or_default(),
        ),
        None => format!("{value} ({ty})"),
    }
}

/// The node a slot's frames really come from: itself for a leaf, and for a port — which relays
/// rather than runs — whatever is behind it. `slot` names a facade's port by uid.
fn behind(g: &Graph, uid: Uid, slot: &str) -> Uid {
    match g.stream(uid, slot) {
        Some(goofi_engine::Stream::At(leaf, _)) => leaf,
        _ => uid,
    }
}

/// `inspect_node`: what the node is, what its params say, which output slots it has and whether it
/// is emitting on them. The frames themselves are only on `/data/<node>/<slot>`.
pub fn node(
    g: &Graph,
    uid: Uid,
    slot: Option<&str>,
    want_params: bool,
    want_error: bool,
) -> Result<String, String> {
    let type_name = g.node_type(uid)
        .ok_or_else(|| format!("inspect_node: no node `{}`", uid.to_hex()))?;
    // A port and a facade never run, so they wear no tier and reach no stage; everything else a
    // read says about a node, they answer.
    let runtime = match g.manifest(uid) {
        Some(m) => format!(
            ", {}, stage {}",
            match m.isolation {
                goofi_node::Isolation::InProcess => "in-process",
                goofi_node::Isolation::Subprocess => "subprocess",
            },
            g.node_stage(uid),
        ),
        None => String::new(),
    };
    let mut out =
        format!("{}: {type_name} (uid {}{runtime})\n", g.name(uid).unwrap_or("?"), uid.to_hex());
    // `(key, label, dtype)`: a facade keys its slots by port uid so a rename cannot break a wire,
    // and carries the port's display name beside it — so nothing here re-derives a label.
    let outputs = crate::vocab::output_slots(g, uid);
    if let Some(s) = slot {
        if !outputs.iter().any(|(key, label, _)| label == s || key == s) {
            return Err(format!(
                "inspect_node: `{type_name}` has no output slot `{s}` (it has: {})",
                outputs.iter().map(|(_, l, _)| l.as_str()).collect::<Vec<_>>().join(", "),
            ));
        }
    }

    if want_params {
        out.push_str("\nparams:\n");
        for (group, names) in g.params(uid).iter().flat_map(|p| p.iter()) {
            for (name, p) in names {
                let expr = g.param_expression(uid, group, name);
                out.push_str(&format!("  {group}.{name} = {}\n", param_line(p, expr.as_ref())));
            }
        }
    }

    out.push_str("\noutputs:\n");
    if outputs.is_empty() {
        out.push_str("  (none)\n");
    }
    for (key, name, kind) in outputs.iter().filter(|(k, l, _)| slot.is_none_or(|s| s == k || s == l)) {
        // `ufreq` measures how often a node RUNS, so it is read off whichever node the frames
        // really come from — itself for a leaf, the node behind it for a port.
        let rate = match g.node_ufreq(behind(g, uid, key)) {
            Some(hz) => format!("emitting at {hz:.1} Hz"),
            None => "nothing emitted yet".to_string(),
        };
        out.push_str(&format!("  {name}: {kind} — {rate}\n"));
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

/// `list_nodes {type}`: a node type's text where it has one, and its provenance either way.
pub fn node_source(g: &Graph, ty: &str, dirs: &[(std::path::PathBuf, &str)]) -> Result<Value, String> {
    let native = goofi_node::find(ty);
    let manifest = native
        .or_else(|| g.dyn_type_manifests().into_iter().find(|m| m.type_name == ty))
        .ok_or_else(|| format!("list_nodes: no node type `{ty}`"))?;
    let mut info = crate::schemas::node_type_info(
        manifest,
        if g.is_patch_type(ty) { "patch" } else { "builtin" },
    );
    // The type name is its file's CamelCased stem, so the path re-derives without a registry.
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

use goofi_engine::layout::{Layout, Node};

/// One node's line in the arrangement tree, and its children under it.
fn layout_line(n: &Node, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    match n {
        Node::Split { id, size, axis, children } => {
            out.push_str(&format!("{pad}{} split {size:.2}  [{id}]\n", axis.name()));
            for c in children {
                layout_line(c, depth + 1, out);
            }
        }
        Node::Panel { id, size, panel_type, state } => {
            let bound = state.get("node").and_then(|v| v.as_str()).map(|b| format!(" → {b}"));
            out.push_str(&format!("{pad}{panel_type}{} {size:.2}  [{id}]\n", bound.unwrap_or_default()))
        }
    }
}

pub fn layout_tree(l: &Layout, tab: Option<&str>) -> String {
    let mut out = String::from(
        "The editor arrangement. Every entry — tab, split and panel — is addressed by the id in []. \
         The number on each entry is its share of its parent — what edit_panel's `fractions` sets.\n\n",
    );
    let tabs = match tab {
        Some(t) => vec![t.to_string()],
        None => l.tabs(),
    };
    for t in tabs {
        let Some(name) = l.name_of(&t) else { continue };
        out.push_str(&format!("tab `{name}`  [{t}]\n"));
        if let Some(root) = l.root_of(&t).and_then(|r| l.node(&r).cloned()) {
            layout_line(&root, 1, &mut out);
        }
    }
    out
}
