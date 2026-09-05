//! The read ops an agent uses to see what it built: a scope as a diagram, a node as a page of text.

use std::path::{Path, PathBuf};

use goofi_graph::{Graph, Uid};
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
        None => g.all_uids().into_iter().filter(|u| g.scope_of(*u).is_none()).collect(),
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
    let mut parts = vec![g.name(scope).unwrap_or("?").to_string()];
    let mut at = scope;
    while let Some(parent) = g.scope_of(at) {
        parts.push(g.name(parent).unwrap_or("?").to_string());
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

/// `nodes inspect`: ONE scope drawn as a mermaid flowchart. Scope-wide and nothing more — the
/// patch's identity and standing errors are `session status`'s.
pub fn patch(g: &Graph, scope: Option<Uid>) -> Result<String, String> {
    if let Some(s) = scope {
        if !g.is_facade(s) {
            return Err(format!("nodes inspect: no sub-patch `{}`", s.to_hex()));
        }
    }
    let mut out = format!(
        "scope: {}\n",
        scope.map_or("root".to_string(), |s| format!("{} ({})", scope_path(g, s), s.to_hex())),
    );

    let member = members(g, scope);
    if member.is_empty() {
        out.push_str("\n(no nodes)\n");
    } else {
        out.push_str("\n```mermaid\nflowchart LR\n");
        // ONE loop: a port, a facade and a leaf are all members, and only the SHAPE they are drawn
        // in differs — which is the one distinction a diagram is allowed to make.
        for &uid in &member {
            let hex = uid.to_hex();
            let name = label(g.name(uid).unwrap_or("?"));
            let ty = g.node_type(uid).unwrap_or_else(|| "?".into());
            let warn = if g.last_error(uid).is_some() { "⚠ " } else { "" };
            if g.stub(uid).is_some() {
                out.push_str(&format!("  {}([\"{warn}{name}: {ty}<br/>{hex}\"])\n", mid(uid)));
            } else if g.is_facade(uid) {
                out.push_str(&format!("  {}[[\"{warn}{name}<br/>{hex}\"]]\n", mid(uid)));
            } else {
                out.push_str(&format!("  {}[\"{warn}{name}: {ty}<br/>{hex}\"]\n", mid(uid)));
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
        out.extend(edges);
        out.push_str("```\n\nuids: a uid is its mermaid id without the leading `n`.\n");
    }

    Ok(out)
}

/// Every standing error in the patch, with the age of each — what `session status` answers:
/// a patch's health is the patch's, not a scope's.
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

fn param_line(p: &goofi_core::Param, source: Option<&goofi_graph::SourceInfo>) -> String {
    use goofi_core::Param as P;
    let (value, ty) = match p {
        P::Float { value, vmin, vmax } => (format!("{value}"), format!("float {vmin}..{vmax}")),
        P::Int { value, vmin, vmax } => (format!("{value}"), format!("int {vmin}..{vmax}")),
        P::Bool { value } => (format!("{value}"), "bool".to_string()),
        P::Str { value, options: Some(o), .. } => {
            (format!("\"{value}\""), format!("string one of [{}]", o.join(", ")))
        }
        P::Str { value, .. } => (format!("\"{value}\""), "string".to_string()),
        P::Pulse => ("—".to_string(), "pulse".to_string()),
    };
    let error = |s: &goofi_graph::SourceInfo| {
        s.error.as_ref().map(|e| format!(" [error: {e}]")).unwrap_or_default()
    };
    match source {
        Some(s) if s.state.mode == goofi_graph::Mode::Expression => {
            format!("expr: {} → {value}{}", s.state.expression, error(s))
        }
        Some(s) if s.state.mode == goofi_graph::Mode::Reference => {
            format!("ref: {} → {value}{}", s.state.reference, error(s))
        }
        _ => format!("{value} ({ty})"),
    }
}

/// The node a slot's frames really come from: itself for a leaf, and for a port — which relays
/// rather than runs — whatever is behind it. `slot` names a facade's port by uid.
fn behind(g: &Graph, uid: Uid, slot: &str) -> Uid {
    match g.stream(uid, slot) {
        Some(goofi_graph::Stream::At(leaf, _)) => leaf,
        _ => uid,
    }
}

/// `node state`: what the node is, what its params say, which output slots it has and whether it
/// is emitting on them. The frames themselves are only on `/data/<node>/<slot>`.
pub fn node(
    g: &Graph,
    uid: Uid,
    slot: Option<&str>,
    want_params: bool,
    want_error: bool,
) -> Result<String, String> {
    let type_name = g.node_type(uid)
        .ok_or_else(|| format!("node state: no node `{}`", uid.to_hex()))?;
    // A port and a facade never run, so they wear no tier and reach no stage; everything else a
    // read says about a node, they answer.
    let runtime = match g.node_tier(uid) {
        Some(tier) => format!(", {}, stage {}", tier.wire(), g.node_stage(uid)),
        None => String::new(),
    };
    let mut out =
        format!("{}: {type_name} (uid {}{runtime})\n", g.name(uid).unwrap_or("?"), uid.to_hex());
    // `(key, label, dtype)`: a facade keys its slots by port uid so a rename cannot break a wire,
    // and carries the port's display name beside it — so nothing here re-derives a label.
    let outputs = crate::vocab::output_slots(g, uid);
    if let Some(s) = slot {
        crate::vocab::check_slot(g, "node state", uid, s)?;
    }

    if want_params {
        out.push_str("\nparams:\n");
        for (group, names) in g.params(uid).iter().flat_map(|p| p.iter()) {
            for (name, p) in names {
                let source = g.param_source(uid, group, name);
                let mut shown = p.clone();
                if let (goofi_core::Param::Str { options, .. }, Some(live)) =
                    (&mut shown, g.refreshed_options(uid, group, name))
                {
                    *options = Some(live.to_vec());
                }
                out.push_str(&format!("  {group}.{name} = {}\n", param_line(&shown, source.as_ref())));
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

/// `global list`: what an expression can read and the global writes can set.
pub fn globals(g: &Graph) -> Value {
    let entries: Vec<Value> = g
        .globals()
        .entries()
        .map(|(name, v, system, locked)| {
            let mut e = goofi_graph::global_to_json(v);
            e["name"] = json!(name);
            e["system"] = json!(system);
            e["locked"] = json!(locked);
            e
        })
        .collect();
    json!({ "globals": entries })
}

/// `library get`: a node type's text where it has one, and its provenance either way.
pub fn node_source(g: &Graph, ty: &str, mount: &Path, roots: &[PathBuf]) -> Result<Value, String> {
    let (engine, entry) = g.resolve_type(ty).map_err(|e| format!("library get: {e}"))?;
    let ty = &goofi_node::qualify(engine, entry.manifest.type_name);
    let mut info = crate::schemas::node_type_info(g, engine, entry.manifest, crate::schemas::source_of(g, ty));
    // `.rev()` is load-bearing: `rescan` scans the roots forwards and lets each overwrite the
    // last, so a first-match search walks them backwards.
    let workspace: Vec<PathBuf> = g.engine_ids().into_iter().map(|id| mount.join(goofi_node::folder_of(id))).collect();
    let dirs = workspace
        .into_iter()
        .filter(|_| g.is_patch_type(ty))
        .map(|d| (d, "patch"))
        .chain(roots.iter().rev().map(|d| (d.clone(), "shipped")));
    // The file names the type, so the path re-derives without a registry; the registry says only
    // whether the patch's folder is where it lives.
    let found = dirs.into_iter().find_map(|(dir, provenance)| {
        let entries = std::fs::read_dir(dir).ok()?;
        let path = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| goofi_node::type_name_of(p).as_deref() == Some(goofi_node::bare(ty)) && goofi_node::engine_of(p).as_deref() == Some(engine))?;
        Some((path, provenance))
    });
    let tier = g.type_tier(ty);
    info["language"] = json!(tier.map(goofi_node::Isolation::language));
    info["tier"] = json!(tier.map(goofi_node::Isolation::wire));
    info["provenance"] = json!(match &found {
        Some((_, p)) => *p,
        None => "no source file under any node root",
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

use goofi_graph::layout::{Layout, Node};

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
         The number on each entry is its share of its parent — what `layout split edit` sets.\n\n",
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
