//! One function per registry row — the HANDLER its `ops.rs` row names. The row's kind is the
//! contract each function is held to: a Read touches nothing; a Write routes every mutation
//! through the command history and leaves the re-mirror and the dirty decision to `call`'s
//! shared tail; an Effect owns its consequences — re-mirror, events and dirtiness — itself.

use super::*;

pub(crate) fn list_dir(
    _state: &AppState,
    payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    // Served WITHOUT the graph mutex: it walks the filesystem, which under the lock would stall
    // the status-drain worker.
    Ok(fsbrowse::list_dir(payload.get("path").and_then(|v| v.as_str())))
}

pub(crate) fn get_state(
    state: &AppState,
    _payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    Ok(state.doc.lock().unwrap().to_json())
}

// The harness ops touch no graph state: they fork and signal children, and the roster converges
// through `harness_changed` rather than by making a caller wait.

pub(crate) fn list_harnesses(
    state: &AppState,
    _payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    state.harnesses.refresh_in_background(state.events.clone());
    Ok(state.harnesses.roster())
}

pub(crate) fn spawn_harness(
    state: &AppState,
    payload: &Value,
    _session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let h = payload
        .get("harness")
        .and_then(|v| v.as_str())
        .ok_or("spawn_harness: missing harness")?;
    let id = state.harnesses.spawn(
        h,
        &state.mount(),
        &state.mcp_url(),
        &term::parent_env(),
        state.events.clone(),
    )?;
    events.push(event("harness_changed", state.harnesses.roster()));
    Ok(json!({ "instance_id": id }))
}

pub(crate) fn stop_harness(
    state: &AppState,
    payload: &Value,
    _session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    state
        .harnesses
        .stop(payload.get("instance").and_then(|v| v.as_str()).ok_or("stop_harness: missing instance")?)?;
    events.push(event("harness_changed", state.harnesses.roster()));
    Ok(json!({ "ok": true }))
}

/// Several writes as ONE undo step. Each step is a whole [`AppState::call`]: it locks, mirrors
/// the document and broadcasts exactly as it would alone.
pub(crate) fn compound(
    state: &AppState,
    payload: &Value,
    session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let steps = payload
        .get("ops")
        .and_then(|v| v.as_array())
        .ok_or("compound: `ops` is a list of {op, payload}")?
        .clone();
    for (i, step) in steps.iter().enumerate() {
        let name = step.get("op").and_then(|v| v.as_str());
        // A step must be an undoable write — a Write row — or a rollback could not take it back.
        if !name.and_then(ops::find).is_some_and(|o| o.handler.is_write()) {
            return Err(format!(
                "compound: step {i} `{}` is not a step — a step is one undoable write",
                name.unwrap_or("")
            ));
        }
    }
    // The redo run is cleared UP FRONT so no step's own clearing can shift the mark.
    let from = {
        let mut h = state.history.lock().unwrap();
        h.clear_redo(session);
        h.len()
    };
    let mut results = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let name = step["op"].as_str().unwrap_or_default().to_string();
        let arg = step.get("payload").cloned().unwrap_or_else(|| json!({}));
        match state.call(&name, arg, session) {
            Ok(r) => results.push(r),
            Err(e) => {
                // A compound is a UNIT, so a refused step takes back the ones that landed.
                let mut g = state.graph.lock().unwrap();
                state.history.lock().unwrap().rollback(&mut g, session, from);
                drop(g);
                resync_and_broadcast(state);
                return Err(format!("compound: step {i} `{name}` was refused: {e}"));
            }
        }
    }
    state.history.lock().unwrap().coalesce(session, from);
    resync_and_broadcast(state);
    events.extend(state.set_dirty(true));
    Ok(json!({ "results": results }))
}

/// The catalog, or ONE entry of it in full. A type's source and provenance are the same entry
/// with the file behind it read, so they are the same op narrowed.
pub(crate) fn list_nodes(
    state: &AppState,
    payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mount = state.mount();
    let g = state.graph.lock().unwrap();
    match payload.get("type").and_then(|v| v.as_str()) {
        None => Ok(json!({ "types": schemas::catalog_types(&g) })),
        Some(ty) => {
            // `.rev()` is load-bearing: `rescan` scans the shipped list forwards and lets each
            // directory overwrite the last, so a first-match search walks it backwards.
            let dirs: Vec<(PathBuf, &str)> = [(mount.join("nodes"), "patch")]
                .into_iter()
                .chain(state.system_nodes.iter().rev().map(|d| (d.clone(), "shipped")))
                .collect();
            inspect::node_source(&g, ty, &dirs)
        }
    }
}

/// Explicit, never watched: an agent calls it after writing a node file.
pub(crate) fn rescan_nodes(
    state: &AppState,
    _payload: &Value,
    _session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let result = {
        let mut g = state.graph.lock().unwrap();
        let (diff, _) = rescan(state, &mut g, &state.mount());
        restart_changed(&mut g, &diff);
        events.push(node_types_event(&g));
        json!({ "added": diff.added, "changed": diff.changed, "removed": diff.removed })
    };
    resync_and_broadcast(state);
    Ok(result)
}

pub(crate) fn copy_nodes(
    state: &AppState,
    payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    let uids = parse_uid_list(payload, "nodes")?;
    Ok(json!({ "doc": g.fragment(&g.subtree_of(&uids)) }))
}

pub(crate) fn paste_nodes(
    state: &AppState,
    payload: &Value,
    session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let doc = payload.get("doc").ok_or("paste_nodes: missing doc")?;
    let offset = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
    let scope = match payload.get("inst_id").filter(|v| !v.is_null()) {
        Some(v) => Some(v.as_str().and_then(Uid::from_hex).ok_or("paste_nodes: malformed inst_id")?),
        None => None,
    };
    let (cmd, rename) = g.import_fragment(doc, scope, offset)?;
    state.history.lock().unwrap().apply(&mut g, session, cmd)?;
    for uid in rename.values() {
        events.push(event("node_added", json!({ "uid": uid })));
    }
    Ok(json!({ "rename": rename }))
}

pub(crate) fn add_node(
    state: &AppState,
    payload: &Value,
    session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let ty = payload
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or("add_node: missing type")?
        .to_string();
    // A CHOSEN uid and name, so a caller reconstructing a known graph keeps its uid-keyed
    // bindings. Not the undo path, which is manager-owned.
    let restore = payload.get("member_uid").and_then(|v| v.as_str()).and_then(Uid::from_hex);
    let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
    // A chosen name that collides — or that an expression could not read as an attribute — is
    // refused here, so a caller told nothing cannot get a node under a name it never asked for.
    if !name.is_empty() {
        if g.name_taken(&name, None) {
            return Err(format!("add_node: the name `{name}` is taken"));
        }
        if !goofi_core::globals::is_valid_identifier(&name) {
            return Err(format!(
                "add_node: `{name}` is not a legal name: {}",
                goofi_engine::NAME_RULE
            ));
        }
    }
    let pos = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
    // Never silently rooted on a bad `inst_id`: the canvas draws only the entered scope, so a
    // rooted node would be invisible exactly where the user placed it.
    let scope = match payload.get("inst_id").filter(|v| !v.is_null()) {
        Some(v) => Some(v.as_str().and_then(Uid::from_hex).ok_or("add_node: malformed inst_id")?),
        None => None,
    };
    // Inline params are applied AFTER: RemoveNode's inverse captures the LIVE node, so an
    // undo→redo restores them without threading them through the command.
    let cmd = goofi_engine::Command::AddNode {
        type_name: ty,
        pos,
        uid: restore,
        name: (!name.is_empty()).then_some(name),
        params: None,
        exprs: vec![],
        viewers: None,
        scope,
    };
    let uid = match state.history.lock().unwrap().apply(&mut g, session, cmd)? {
        goofi_engine::Outcome::Uid(u) => u,
        _ => return Err("add_node: no uid returned".into()),
    };
    // Applied UNDER THE GRAPH LOCK, so the node is born configured before the resync mirrors it
    // into the doc.
    if let Some(params) = payload.get("params").filter(|v| !v.is_null()) {
        for cmd in parse_params_bag(&g, uid, params).map_err(|e| format!("add_node: {e}"))? {
            cmd.execute(&mut g).map_err(|e| format!("add_node: {e}"))?;
        }
    }
    // A bare uid: the node itself arrives via the doc mirror.
    events.push(event("node_added", json!({ "uid": uid.to_hex() })));
    // The REPLY answers a caller with no doc replica: the minted name, the slots to wire, and
    // the params as BORN. Read off the GRAPH, not a manifest: a facade and a port have none, and
    // each answers the one slot vocabulary the wiring ops judge against.
    let slots = |v: Vec<(String, String, goofi_core::SlotType)>| {
        Value::Object(v.into_iter().map(|(k, _, t)| (k, json!(t.name()))).collect())
    };
    Ok(json!({
        "uid": uid.to_hex(),
        "name": g.name(uid).unwrap_or_default(),
        "input_slots": slots(g.input_slots(uid)),
        "output_slots": slots(g.output_slots(uid)),
        "params": g.params(uid).map(|p| schemas::param_value_map(&p)).unwrap_or_else(|| json!({})),
    }))
}

pub(crate) fn remove_node(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let uid = parse_uid(payload, "node")?;
    // The command is idempotent, so a uid naming nothing succeeds; the reply says which of the
    // two happened.
    let existed = g.exists(uid);
    let cmd = goofi_engine::Command::RemoveNode { uid };
    state.history.lock().unwrap().apply(&mut g, session, cmd)?;
    Ok(json!({ "removed": existed }))
}

/// Recovery, not an edit, so it is NOT routed through the command history: the client records no
/// `graph_cmd` for a restart and the two stacks must stay 1:1.
pub(crate) fn restart_node(
    state: &AppState,
    payload: &Value,
    _session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    {
        let mut g = state.graph.lock().unwrap();
        let uid = parse_uid(payload, "node")?;
        g.restart_node(uid)?;
        // Pushed at once, so the red border lifts on the click rather than on the sweep.
        events.push(param_state_update(&g, uid));
    }
    resync_and_broadcast(state);
    Ok(json!({ "ok": true }))
}

pub(crate) fn add_link(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let (a, so, b, si) = parse_link(payload)?;
    let (a, so) = wirable_endpoint(&g, a, &so, "node_out")?;
    let (b, si) = wirable_endpoint(&g, b, &si, "node_in")?;
    state.history.lock().unwrap().apply(
        &mut g,
        session,
        goofi_engine::Command::AddLink {
            node_out: a,
            slot_out: so.clone(),
            node_in: b,
            slot_in: si.clone(),
        },
    )?;
    // The wire AS MADE, not as named: a boundary endpoint resolves to its inner leaf, and the
    // agreed dtype gates the next link to this output.
    let dtype = vocab::output_slots(&g, a)
        .into_iter()
        .find(|(key, _, _)| *key == so)
        .map(|(_, _, dtype)| dtype);
    Ok(json!({
        "node_out": a.to_hex(), "slot_out": so,
        "node_in": b.to_hex(), "slot_in": si,
        "dtype": dtype,
    }))
}

pub(crate) fn remove_link(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let (a, so, b, si) = parse_link(payload)?;
    let (a, so) = resolve_link_endpoint(&g, a, &so);
    let (b, si) = resolve_link_endpoint(&g, b, &si);
    // Idempotent for the same reason `remove_node` is, and answered the same way.
    let existed = g.has_link(a, &so, b, &si);
    state.history.lock().unwrap().apply(
        &mut g,
        session,
        goofi_engine::Command::RemoveLink { node_out: a, slot_out: so, node_in: b, slot_in: si },
    )?;
    Ok(json!({ "removed": existed }))
}

/// NOT a command: options are runtime-only, so there is nothing to undo. They do not ride this
/// reply either — the hook runs on the node's own thread.
pub(crate) fn refresh_param(
    state: &AppState,
    payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    {
        let mut g = state.graph.lock().unwrap();
        let uid = parse_uid(payload, "node")?;
        let group = parse_str(payload, "group")?.to_string();
        let name = parse_str(payload, "name")?.to_string();
        g.refresh_param(uid, &group, &name)?;
    }
    resync_and_broadcast(state);
    Ok(json!({ "ok": true }))
}

pub(crate) fn edit_node(
    state: &AppState,
    payload: &Value,
    session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let uid = parse_uid(payload, "node")?;
    let name = payload.get("name").and_then(|v| v.as_str()).map(str::to_string);
    // The rename command tolerates a collision as a no-op so a stale replay converges; the
    // user-facing error therefore belongs here, at the forward RPC.
    if let Some(n) = &name {
        if g.name_taken(n, Some(uid)) {
            return Err(format!("edit_node: the name `{n}` is taken"));
        }
    }
    // A display name is read as an ATTRIBUTE in an expression, so it has to be one — which also
    // covers the quote and backslash that would break the source.
    if name.as_deref().is_some_and(|n| !goofi_core::globals::is_valid_identifier(n)) {
        return Err(format!(
            "edit_node: `{}` is not a legal name: {}",
            name.unwrap_or_default(),
            goofi_engine::NAME_RULE
        ));
    }
    let pos = payload
        .get("pos")
        .filter(|v| !v.is_null())
        .map(|v| parse_pos(v).ok_or("edit_node: pos is [x, y]"))
        .transpose()?;
    // Viewers MERGE key by key, so only the slots named move; the command then sets the whole
    // blob, which is what makes its inverse exact. The PATCH is what is checked — a stale slot
    // already stored is inert, and refusing it would block every later edit on a node whose file
    // changed its slots.
    let viewers = match payload.get("viewers").filter(|v| !v.is_null()) {
        Some(patch) => {
            vocab::check_viewers(&g, uid, patch)?;
            let mut whole = g.viewers(uid).cloned().ok_or("edit_node: no such node")?;
            merge_json(&mut whole, patch);
            Some(whole)
        }
        None => None,
    };
    let params = payload.get("params").filter(|v| !v.is_null());
    if name.is_none() && pos.is_none() && viewers.is_none() && params.is_none() {
        return Err("edit_node: give a name, pos, params or viewers".into());
    }

    // ONE command, so one undo step covers whatever the call carried: the node's own fields,
    // then a param edit each.
    let mut cmds = Vec::new();
    if name.is_some() || pos.is_some() || viewers.is_some() {
        cmds.push(goofi_engine::Command::EditNode { uid, name, pos, viewers });
    }
    let mut touched: Vec<(String, String)> = Vec::new();
    if let Some(params) = params {
        for cmd in parse_params_bag(&g, uid, params).map_err(|e| format!("edit_node: {e}"))? {
            if let goofi_engine::Command::EditParam { group, name, .. } = &cmd {
                touched.push((group.clone(), name.clone()));
            }
            cmds.push(cmd);
        }
    }
    let out = state.history.lock().unwrap().apply(
        &mut g,
        session,
        if cmds.len() == 1 { cmds.pop().unwrap() } else { goofi_engine::Command::Compound(cmds) },
    )?;
    // The runtime `expression_error` is doc-invisible, so echo the descriptors — for this node,
    // and for every referrer a rename rewrote.
    if !touched.is_empty() {
        events.push(param_state_update(&g, uid));
    }
    if let goofi_engine::Outcome::Nodes(referrers) = out {
        for r in referrers {
            events.push(param_state_update(&g, r));
        }
    }
    // Every param touched AS STORED: a literal is coerced to its declared type, and a binding
    // that does not compile is stored WITH its error.
    let mut out = serde_json::Map::new();
    for (group, name) in touched {
        let entry = json!({
            "value": g.params(uid)
                .and_then(|p| goofi_node::param(&p, &group, &name).cloned())
                .map(|p| goofi_engine::param_value_json(&p, true)),
            "error": g.param_expression(uid, &group, &name).and_then(|e| e.error),
        });
        out.entry(group).or_insert_with(|| json!({}))
            .as_object_mut().unwrap()
            .insert(name, entry);
    }
    Ok(json!({ "params": Value::Object(out) }))
}

/// Where THIS client is looking: not a doc root, so it neither drags a peer nor raises the
/// unsaved dot, but it still rides the `.gfi` and `hello`.
pub(crate) fn set_viewpoint(
    state: &AppState,
    payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    {
        let mut g = state.graph.lock().unwrap();
        g.set_viewpoint(payload.get("viewpoint").cloned().unwrap_or(Value::Null));
    }
    resync_and_broadcast(state);
    Ok(json!({ "ok": true }))
}

pub(crate) fn inspect_layout(
    state: &AppState,
    payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    let tab = payload.get("tab").and_then(|v| v.as_str()).map(str::to_string);
    Ok(json!({ "text": inspect::layout_tree(g.arrangement(), tab.as_deref()) }))
}

/// One entry's FIELDS, whichever kind it is: a tab wears a name, a panel a type and a state, a
/// split its shares. Several in one call is one Compound, so one undo step.
pub(crate) fn edit_panel(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let panel = parse_str(payload, "panel")?.to_string();
    let name = payload.get("name").and_then(|v| v.as_str());
    let ty = payload.get("type").and_then(|v| v.as_str()).map(str::to_string);
    let panel_state = payload.get("state").cloned().filter(|v| !v.is_null());
    let fractions = match payload.get("fractions").filter(|v| !v.is_null()) {
        // A non-numeric entry becomes NaN, which the planner refuses beside a zero or a negative
        // one — so "is this a fraction" is answered in one place.
        Some(v) => Some(
            v.as_array()
                .ok_or("edit_panel: fractions is a list of numbers")?
                .iter()
                .map(|x| x.as_f64().unwrap_or(f64::NAN))
                .collect::<Vec<f64>>(),
        ),
        None => None,
    };
    if name.is_none() && ty.is_none() && panel_state.is_none() && fractions.is_none() {
        return Err("edit_panel: give a name, type, state or fractions".into());
    }

    let mut writes: Vec<goofi_engine::layout::Write> = Vec::new();
    if let Some(n) = name {
        writes.extend(g.arrangement().rename_tab(&panel, n)?);
    }
    if ty.is_some() || panel_state.is_some() {
        // A panel bound to a node that is not there renders empty and explains nothing.
        let named = panel_state
            .as_ref()
            .and_then(|s| s.get("node"))
            .and_then(|v| v.as_str())
            .filter(|n| !n.is_empty());
        if let Some(node) = named {
            if !bindable_node(&g, node) {
                return Err(format!("edit_panel: no node `{node}` in this patch"));
            }
        }
        // The slot is checked against the node this write LEAVES the panel bound to: its own, or
        // the one already stored, since a state write merges.
        let bound = named
            .or_else(|| {
                g.arrangement()
                    .panel_state(&panel)
                    .and_then(|s| s.get("node"))
                    .and_then(|v| v.as_str())
            })
            .and_then(Uid::from_hex);
        vocab::check_panel(&g, ty.as_deref(), panel_state.as_ref(), bound)?;
        writes.extend(g.arrangement().set_panel(&panel, ty.as_deref(), panel_state)?);
    }

    let mut cmds: Vec<goofi_engine::Command> = Vec::new();
    if !writes.is_empty() {
        cmds.push(goofi_engine::Command::LayoutContents { writes });
    }
    if let Some(fractions) = fractions {
        // Planned here only so a bad split or a wrong fraction count answers teachably; the
        // command re-plans it under this same lock.
        g.arrangement().resize_split(&panel, &fractions)?;
        cmds.push(goofi_engine::Command::LayoutResizeSplit { split: panel.clone(), fractions });
    }
    let cmd = if cmds.len() == 1 {
        cmds.pop().expect("length checked")
    } else {
        goofi_engine::Command::Compound(cmds)
    };
    apply_layout(state, &mut g, session, cmd)
}

/// ONE op per drag gesture: a drop is one undo step, and peers never see an arrangement that was
/// not on somebody's screen. Placement is ONE grammar: `panel` says what (an existing entry, or
/// a fresh one when absent), the rest says where. Splitting it into a birth op and a move op
/// spelled the "where" twice, and `add_tab {subtree}` was already both.
pub(crate) fn place_panel(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    const OP: &str = "place_panel";
    let mut g = state.graph.lock().unwrap();
    let panel = payload.get("panel").and_then(|v| v.as_str()).map(str::to_string);
    let to = payload.get("to").and_then(|v| v.as_str()).map(str::to_string);
    let index = payload.get("index").and_then(|v| v.as_u64()).map(|i| i as usize);
    let side = match payload.get("direction").filter(|v| !v.is_null()) {
        Some(_) => Some(parse_side(payload, OP)?),
        None => None,
    };
    let ratio = payload.get("ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
    let name = payload.get("name").and_then(|v| v.as_str());
    // A tab already has a tab, so a `to`-less place is a reorder rather than a wrap. The id says
    // which — as it does for `edit_panel` and `remove_panel`.
    let is_tab = panel.as_deref().is_some_and(|p| g.arrangement().tab_index(p).is_some());
    let (plan, placed) = match (panel.as_deref(), to.as_deref(), side) {
        (Some(p), None, _) if is_tab => {
            let at = index.ok_or(format!("{OP}: give a `to`, an `index`, or both"))?;
            g.arrangement().reorder_tab(p, at)?;
            let cmd = goofi_engine::Command::LayoutReorderTab { tab: p.to_string(), to_index: at };
            let text = apply_layout(state, &mut g, session, cmd)?;
            let tab = p.to_string();
            return Ok(json!({ "id": tab, "tab": tab, "text": text["text"] }));
        }
        // Onto a tab of its own — a fresh panel, or an existing subtree wrapped, which is the
        // drag onto the tab bar.
        (p, None, _) => {
            let (plan, tab) = g.arrangement().add_tab(name, index, p)?;
            // A tab built AROUND an existing subtree is a MOVE, so its undo gives the subtree
            // back; one born with a fresh panel inverts by closing.
            let cmd = match p {
                Some(root) => goofi_engine::Command::LayoutMove {
                    plan: Some(plan), root: root.to_string(), home: None },
                None => goofi_engine::Command::LayoutBirth { plan, born: tab.clone() },
            };
            let text = apply_layout(state, &mut g, session, cmd)?;
            // The root panel's id, which a caller cannot otherwise know.
            let id = p.map(str::to_string)
                .unwrap_or_else(|| g.arrangement().root_of(&tab).unwrap_or_default());
            return Ok(json!({ "id": id, "tab": tab, "text": text["text"] }));
        }
        // Beside a target, dividing it — the drop on a panel's edge, or a split.
        (Some(p), Some(target), Some(side)) => {
            (g.arrangement().insert_at_panel(p, target, side, ratio)?, p.to_string())
        }
        // A fresh panel divides its target; `to` cannot mean "inside that split" here, because
        // there is nothing yet to put inside one. So the side simply defaults, as `split_panel`
        // always did.
        (None, Some(target), _) => {
            let side = side.unwrap_or(goofi_engine::layout::Side::Right);
            let (plan, fresh) = g.arrangement().split_panel(target, side, ratio)?;
            let cmd = goofi_engine::Command::LayoutBirth { plan, born: fresh.clone() };
            let text = apply_layout(state, &mut g, session, cmd)?;
            let tab = g.arrangement().tab_of(&fresh).unwrap_or_default();
            return Ok(json!({ "id": fresh, "tab": tab, "text": text["text"] }));
        }
        // Inside a split, at an index — the drop into a container that exists. There is nothing
        // for a FRESH panel to take space from, so it needs a direction.
        (Some(p), Some(parent), None) => {
            (g.arrangement().move_subtree(p, parent, index.unwrap_or(0))?, p.to_string())
        }
    };
    let cmd = goofi_engine::Command::LayoutMove { plan: Some(plan), root: placed.clone(), home: None };
    let text = apply_layout(state, &mut g, session, cmd)?;
    let tab = g.arrangement().tab_of(&placed).unwrap_or_default();
    Ok(json!({ "id": placed, "tab": tab, "text": text["text"] }))
}

pub(crate) fn remove_panel(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let panel = parse_str(payload, "panel")?.to_string();
    // A tab is closed whole; anything else is closed with promote. Planned here only so a bad id
    // answers teachably: `LayoutClose` re-plans it under this same lock, and DEGRADES rather
    // than errors.
    match g.arrangement().tab_index(&panel) {
        Some(_) => g.arrangement().remove_tab(&panel)?,
        None => g.arrangement().remove_subtree(&panel)?,
    };
    apply_layout(state, &mut g, session, goofi_engine::Command::LayoutClose { born: panel })
}

pub(crate) fn set_global(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let name = parse_str(payload, "name")?.to_string();
    let held = g.globals().get(&name).map(goofi_engine::global_to_json);
    // NO value is a delete, so removing a global is the absence of one rather than an op of its
    // own.
    let Some(val) = payload.get("value").filter(|v| !v.is_null()) else {
        if held.is_none() {
            return Err(format!("set_global: no such global `{name}`"));
        }
        state.history.lock().unwrap().apply(
            &mut g,
            session,
            goofi_engine::Command::EditGlobal { name, value: None, at: None },
        )?;
        return Ok(json!({ "removed": true }));
    };
    // Every expression reading a global depends on its TYPE, so re-typing one through a value
    // edit would break the reference rather than the call.
    let held_ty = held.as_ref().map(|h| h["type"].as_str().unwrap_or_default().to_string());
    let ty = match (payload.get("type").and_then(|v| v.as_str()), &held_ty) {
        (Some(t), Some(h)) if t != h => {
            return Err(format!(
                "set_global: `{name}` is a {h} — remove it and set it again to re-type it"
            ))
        }
        (Some(t), _) => t.to_string(),
        (None, Some(h)) => h.clone(),
        (None, None) => return Err(format!("set_global: `{name}` is new — give its `type`")),
    };
    let value = goofi_engine::global_from_json(&json!({ "value": val, "type": ty }))
        .ok_or_else(|| format!("set_global: `{val}` is not a {ty}"))?;
    state.history.lock().unwrap().apply(
        &mut g,
        session,
        goofi_engine::Command::EditGlobal { name, value: Some(value.clone()), at: None },
    )?;
    // As STORED: the conversion is type-directed, so a fraction into an int rounds.
    Ok(json!({ "value": goofi_engine::global_to_json(&value)["value"] }))
}

pub(crate) fn group_nodes(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let uids = parse_uid_list(payload, "members")?;
    let pos = payload.get("pos").and_then(parse_pos).unwrap_or([0.0, 0.0]);
    let out = state.history.lock().unwrap().apply(
        &mut g,
        session,
        goofi_engine::Command::Group { members: uids, pos, restore: None },
    )?;
    let inst = match out {
        goofi_engine::Outcome::Uid(u) => u,
        _ => return Err("group_nodes: no scope uid returned".into()),
    };
    Ok(json!({ "inst_id": inst.to_hex() }))
}

pub(crate) fn expand_instance(
    state: &AppState,
    payload: &Value,
    session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let inst = parse_uid(payload, "inst_id")?;
    state
        .history
        .lock()
        .unwrap()
        .apply(&mut g, session, goofi_engine::Command::Expand { scope: inst })?;
    Ok(json!({ "ok": true }))
}

pub(crate) fn inspect_patch(
    state: &AppState,
    payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    // The header carries the workspace walk, so it is taken BEFORE the graph lock: no filesystem
    // walk may run while the status-drain worker waits on that lock.
    let dirty = state.is_dirty();
    let workspace = state.mount();
    let save_path = state.save_path();
    let g = state.graph.lock().unwrap();
    let scope = match payload.get("scope").filter(|v| !v.is_null()) {
        Some(v) => {
            Some(v.as_str().and_then(Uid::from_hex).ok_or("inspect_patch: malformed scope")?)
        }
        None => None,
    };
    let text =
        inspect::patch(&g, scope, save_path.as_deref(), &goofi_core::path::to_slash(&workspace), dirty)?;
    Ok(json!({ "text": text }))
}

pub(crate) fn inspect_node(
    state: &AppState,
    payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    let uid = parse_uid(payload, "node")?;
    let want = |k: &str| payload.get(k).and_then(|v| v.as_bool()).unwrap_or(true);
    let slot = payload.get("slot").and_then(|v| v.as_str());
    let text = inspect::node(&g, uid, slot, want("params"), want("error"))?;
    Ok(json!({ "text": text }))
}

pub(crate) fn list_globals(
    state: &AppState,
    _payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    Ok(inspect::globals(&g))
}

/// The open patch's identity AND its health. The error list was drawn under every
/// `inspect_patch`, whichever scope was asked for, so it arrived again under each.
pub(crate) fn get_patch(
    state: &AppState,
    _payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    // The walks (dirty, mount) run before the graph lock, as everywhere.
    let save_path = state.save_path();
    let workspace = goofi_core::path::to_slash(&state.mount());
    let dirty = state.is_dirty();
    let g = state.graph.lock().unwrap();
    Ok(json!({
        "save_path": save_path,
        "workspace": workspace,
        "dirty": dirty,
        "errors": inspect::errors(&g),
    }))
}

pub(crate) fn serialize(
    state: &AppState,
    _payload: &Value,
    _session: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    Ok(json!({ "yaml": g.serialize() }))
}

/// The mount is a per-run temp directory under a random name, so asking is the only way a client
/// or a harness can find it.
pub(crate) fn save(
    state: &AppState,
    payload: &Value,
    _session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    // Expand `~` exactly as the browser does — the two must agree on what a path means. A save
    // writes a file or it is malformed.
    let path = payload
        .get("path")
        .and_then(|v| v.as_str())
        .map(fsbrowse::resolve)
        .ok_or("save: missing path")?;
    let mount = state.mount();
    // Sampled BEFORE the pack: baselining after would call a file written during the zip packed
    // either way, which is the direction that LOSES an edit.
    let packed = goofi_engine::archive::fingerprint(&mount);
    save_archive(std::path::Path::new(&path), &g.serialize(), &mount)?;
    // Announced UNCONDITIONALLY, not on the flag's transition: a patch dirtied solely by a file
    // in the mount leaves the flag already false, so no transition comes.
    *state.workspace_baseline.lock().unwrap() = packed;
    state.set_dirty(false);
    events.push(event("unsaved_changes", json!({ "unsaved_changes": false })));
    // The patch's home, stored ONLY on success and announced as well as stored: an
    // already-connected peer gets no new snapshot to read it from.
    *state.save_path.lock().unwrap() = Some(path.clone());
    events.push(event("save_path_changed", json!({ "save_path": &path })));
    Ok(json!({ "path": path }))
}

/// One arm for every source, so nothing after the read can drift between them.
pub(crate) fn load(
    state: &AppState,
    payload: &Value,
    _session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let result = {
        let mut g = state.graph.lock().unwrap();
        // Every source mounts FRESH, and the live mount is swapped only once the manifest has
        // parsed, so a refused load leaves the open patch untouched on both planes.
        let fresh = new_mount();
        let (content, from_path) =
            stage_load(&fresh, payload).inspect_err(|_| remove_mount(&fresh))?;
        // ORDER is load-bearing: the types the patch SHIPS are registered before the manifest
        // resolves, or the unknown-type gate fires on the nodes the archive brought.
        rescan(state, &mut g, &fresh);
        // Parse BEFORE anything is announced or committed.
        if let Err(e) = g.load_doc(&content) {
            // Refused, so the registry the scan above swapped is re-derived from the mount that
            // is still live.
            rescan(state, &mut g, &state.mount());
            remove_mount(&fresh);
            return Err(e);
        }
        // Commit, now that nothing left can fail: the loaded patch's workspace becomes the live
        // one, and the replaced mount goes with the harnesses spawned into it.
        let replaced = std::mem::replace(&mut *state.mount.lock().unwrap(), fresh);
        state.retire_mount(&replaced);
        events.push(event("harness_changed", state.harnesses.roster()));
        // `read_gfi` restores no mtimes, so without a baseline taken HERE a patch would be dirty
        // from the moment it finished loading.
        *state.workspace_baseline.lock().unwrap() =
            goofi_engine::archive::fingerprint(&state.mount());
        // A load fully resets the session: there is nothing to undo across it.
        state.history.lock().unwrap().clear();
        events.extend(state.set_dirty(false));
        // NONE for an inline load and for `new`, neither with a file behind it: an inherited
        // path would aim the next silent Save at an unrelated `.gfi`.
        *state.save_path.lock().unwrap() = from_path.clone();
        events.push(event(
            "graph_replaced",
            schemas::snapshot(&g, &state.instance_id, false, false, from_path.as_deref(),
                              state.harnesses.roster()),
        ));
        // The patch brought its own node types, which `graph_replaced` does not carry.
        events.push(node_types_event(&g));
        if let Some(path) = from_path {
            events.push(event("save_path_changed", json!({ "save_path": path })));
        }
        // A stored arrangement this model admits but cannot render falls back to the default, so
        // the reply says so rather than leaving the change unexplained.
        json!({ "ok": true, "layout_warning": g.arrangement_warning() })
    };
    resync_and_broadcast(state);
    Ok(result)
}

pub(crate) fn undo(
    state: &AppState,
    _payload: &Value,
    session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let result = {
        let mut g = state.graph.lock().unwrap();
        let mut hist = state.history.lock().unwrap();
        let changed = hist.undo(&mut g, session)?;
        json!({ "changed": changed, "can_undo": hist.can_undo(session), "can_redo": hist.can_redo(session) })
    };
    resync_and_broadcast(state);
    events.extend(state.set_dirty(true));
    Ok(result)
}

pub(crate) fn redo(
    state: &AppState,
    _payload: &Value,
    session: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let result = {
        let mut g = state.graph.lock().unwrap();
        let mut hist = state.history.lock().unwrap();
        let changed = hist.redo(&mut g, session)?;
        json!({ "changed": changed, "can_undo": hist.can_undo(session), "can_redo": hist.can_redo(session) })
    };
    resync_and_broadcast(state);
    events.extend(state.set_dirty(true));
    Ok(result)
}
