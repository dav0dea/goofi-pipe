//! One function per registry row — the HANDLER its `ops.rs` row names. The row's kind is the
//! contract each function is held to: a Read touches nothing; a Write routes every mutation
//! through the command history and leaves the re-mirror and the dirty decision to `call`'s
//! shared tail; an Effect owns its consequences — re-mirror, events and dirtiness — itself.

use super::*;
use crate::schemas::Detail;

/// A bool arg with the default it takes when the caller names none.
fn flag(payload: &Value, name: &str, default: bool) -> bool {
    payload.get(name).and_then(Value::as_bool).unwrap_or(default)
}

/// The granularity a catalog read asks for: an index unless the caller says `--full`.
fn detail(payload: &Value, name: &str) -> Detail {
    match flag(payload, name, false) {
        true => Detail::Full,
        false => Detail::Index,
    }
}

pub(crate) fn dir_list(
    _state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    // Served WITHOUT the graph mutex: it walks the filesystem, which under the lock would stall
    // the status-drain worker.
    Ok(fsbrowse::list_dir(payload.get("path").and_then(|v| v.as_str()), flag(payload, "hidden", false)))
}

pub(crate) fn session_state(
    state: &AppState,
    _payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    Ok(state.doc.lock().unwrap().to_json())
}

// The harness ops touch no graph state: they fork and signal children, and the roster converges
// through `harness_changed` rather than by making a caller wait.

pub(crate) fn agent_list(
    state: &AppState,
    _payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    Ok(state.harnesses.roster(&goofi_core::home::agents()))
}

pub(crate) fn agent_start(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let h = payload
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or("agent start: missing name")?;
    // The mount lock is held ACROSS the spawn, so a concurrent load's swap-and-delete cannot
    // take the workspace out from under the child's cwd.
    let id = {
        let mount = state.mount.lock().unwrap();
        state.harnesses.spawn(
            h,
            &mount,
            &state.instance_id,
            &term::parent_env(),
            state.events.clone(),
            state.history.clone(),
        )?
    };
    events.push(event("harness_changed", state.harnesses.roster(&goofi_core::home::agents())));
    Ok(json!({ "instance_id": id }))
}

pub(crate) fn agent_stop(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let id =
        payload.get("instance").and_then(|v| v.as_str()).ok_or("agent stop: missing instance")?;
    // The stopped shell's undo stack is dropped by the REAPER, where the actor really dies.
    state.harnesses.stop(id)?;
    events.push(event("harness_changed", state.harnesses.roster(&goofi_core::home::agents())));
    Ok(json!({ "ok": true }))
}

/// Several steps as ONE undo step, decided from SETTLED state: the handlers run directly, so no
/// step re-mirrors or dirties on its own — the batch does each exactly once when it settles, and
/// viewers never see an intermediate document.
pub(crate) fn compound(
    state: &AppState,
    payload: &Value,
    actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let steps = payload
        .get("ops")
        .and_then(|v| v.as_array())
        .ok_or("compound: `ops` is a list of {op, payload}")?
        .clone();
    // Every row is resolved BEFORE anything lands: a Read rides for its result, a Write can be
    // taken back, and an Effect owns consequences a rollback cannot reach — refused whole.
    let mut resolved = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let name = step.get("op").and_then(|v| v.as_str()).unwrap_or_default();
        let op = state
            .find_op(name)
            .ok_or_else(|| format!("compound: step {i}: unknown op `{name}`"))?;
        if !(op.handler.is_read() || op.handler.is_write()) {
            return Err(format!(
                "compound: step {i} `{name}` is not a step — a read or an undoable write \
                 rides a batch; an effect runs as the only command"
            ));
        }
        resolved.push((op, step.get("payload").cloned().unwrap_or_else(|| json!({}))));
    }
    // A batch of reads alone never reaches the history: a Read is a no-op for undo, and the
    // actor's redo run must survive it.
    let writes = resolved.iter().any(|(op, _)| op.handler.is_write());
    // The redo run is cleared UP FRONT, and the batch is a thread-local STAMP on the entries the
    // steps make — never a position, which a peer's removal could shift. No step touches the
    // dirty flag — only this settle does — so a refusal has nothing to restore.
    let batch = writes.then(|| {
        state.history.lock().unwrap().clear_redo(actor);
        // Held on the GRAPH, because the drain is another thread: without the hold, its
        // settle can deliver this compound's intermediates between two steps.
        state.graph.lock().unwrap().hold_settle();
        goofi_graph::open_batch()
    });
    let mut results = Vec::with_capacity(resolved.len());
    for (i, (op, arg)) in resolved.iter().enumerate() {
        match op.handler.run(state, arg, actor, events) {
            Ok(r) => results.push(r),
            Err(e) => {
                // A compound is a UNIT, so a refused step takes back the ones that landed —
                // and the events they queued name state the correction just took away.
                if let Some(batch) = &batch {
                    let mut g = state.graph.lock().unwrap();
                    state.history.lock().unwrap().rollback(&mut g, batch.id());
                    g.release_settle();
                    drop(g);
                    events.clear();
                    resync_and_broadcast(state);
                }
                return Err(format!("compound: step {i} `{}` was refused: {e}", op.name));
            }
        }
    }
    if let Some(batch) = &batch {
        state.history.lock().unwrap().coalesce(actor, batch.id());
        state.graph.lock().unwrap().release_settle();
        resync_and_broadcast(state);
        events.extend(state.set_dirty(true));
    }
    // The steps' own replies, in order — a BARE list, the shape every batch door answers.
    Ok(Value::Array(results))
}

/// The library as an INDEX; `--full` answers the palette a client builds every node from.
pub(crate) fn library_list(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    Ok(json!({ "types": schemas::catalog_types(&g, detail(payload, "full")) }))
}

/// ONE library entry: the palette entry with its provenance, and `--source` the file behind it.
pub(crate) fn library_get(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let ty = parse_str(payload, "type")?;
    let mount = state.mount();
    let source = flag(payload, "source", false);
    inspect::node_source(&state.graph.lock().unwrap(), ty, &mount, &state.roots, source)
}

/// Explicit, never watched: an agent calls it after writing a node file.
pub(crate) fn library_refresh(
    state: &AppState,
    _payload: &Value,
    _actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    prebuild(state, &state.mount());
    let result = {
        let mut g = state.graph.lock().unwrap();
        let (diff, _) = rescan(state, &mut g, &state.mount());
        restart_changed(&mut g, &diff);
        events.push(event("node_types", json!({ "types": schemas::catalog_types(&g, Detail::Full) })));
        json!({ "added": diff.added, "changed": diff.changed, "removed": diff.removed })
    };
    resync_and_broadcast(state);
    Ok(result)
}

pub(crate) fn nodes_copy(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    let uids = parse_uid_list(&g, payload, "nodes")?;
    Ok(json!({ "doc": g.fragment(&g.subtree_of(&uids)) }))
}

pub(crate) fn nodes_paste(
    state: &AppState,
    payload: &Value,
    actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let doc = payload.get("doc").ok_or("nodes paste: missing doc")?;
    let offset = payload
        .get("pos")
        .filter(|v| !v.is_null())
        .map(|v| parse_pos(v).ok_or("nodes paste: pos is [x, y]"))
        .transpose()?
        .unwrap_or([0.0, 0.0]);
    let scope = parse_uid_opt(&g, payload, "inst_id", "nodes paste")?;
    let (cmd, rename) = g.import_fragment(doc, scope, offset)?;
    state.history.lock().unwrap().apply(&mut g, actor, cmd)?;
    for uid in rename.values() {
        events.push(event("node_added", json!({ "uid": uid })));
    }
    Ok(json!({ "rename": rename }))
}

pub(crate) fn node_add(
    state: &AppState,
    payload: &Value,
    actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let ty = payload
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or("node add: missing type")?
        .to_string();
    // A CHOSEN uid and name, so a caller reconstructing a known graph keeps its uid-keyed
    // bindings. Not the undo path, which is manager-owned.
    let restore = payload.get("member_uid").and_then(|v| v.as_str()).and_then(Uid::from_hex);
    let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
    // A chosen name that collides — or that an expression could not read as an attribute — is
    // refused here, so a caller told nothing cannot get a node under a name it never asked for.
    if !name.is_empty() {
        if g.name_taken(&name, None) {
            return Err(format!("node add: the name `{name}` is taken"));
        }
        if !goofi_core::globals::is_valid_name(&name) {
            return Err(format!(
                "node add: `{name}` is not a legal name: {}",
                goofi_core::globals::NAME_RULE
            ));
        }
    }
    let pos = payload
        .get("pos")
        .filter(|v| !v.is_null())
        .map(|v| parse_pos(v).ok_or("node add: pos is [x, y]"))
        .transpose()?
        .unwrap_or([0.0, 0.0]);
    // Never silently rooted on a bad `inst_id`: the canvas draws only the entered scope, so a
    // rooted node would be invisible exactly where the user placed it.
    let scope = parse_uid_opt(&g, payload, "inst_id", "node add")?;
    // Inline params are applied AFTER: RemoveNode's inverse captures the LIVE node, so an
    // undo→redo restores them without threading them through the command.
    let cmd = goofi_graph::Command::AddNode {
        type_name: ty,
        pos,
        uid: restore,
        name: (!name.is_empty()).then_some(name),
        params: None,
        sources: vec![],
        viewers: None,
        scope,
    };
    let uid = match state.history.lock().unwrap().apply(&mut g, actor, cmd)? {
        goofi_graph::Outcome::Uid(u) => u,
        _ => return Err("node add: no uid returned".into()),
    };
    // Applied UNDER THE GRAPH LOCK, so the node is born configured before the resync mirrors it
    // into the doc.
    if let Some(entries) = payload.get("param").filter(|v| !v.is_null()) {
        let bag = param_entries_bag(entries).map_err(|e| format!("node add: {e}"))?;
        for cmd in goofi_graph::param_commands(&g, uid, &bag).map_err(|e| format!("node add: {e}"))? {
            cmd.execute(&mut g).map_err(|e| format!("node add: {e}"))?;
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
        "name": named(&g, uid),
        "uid": uid.to_hex(),
        "input_slots": slots(g.input_slots(uid)),
        "output_slots": slots(g.output_slots(uid)),
        "params": g.params(uid).map(|p| schemas::param_value_map(&p)).unwrap_or_else(|| json!({})),
    }))
}

pub(crate) fn node_remove(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let uid = parse_uid(&g, payload, "node")?;
    // The command is idempotent, so a uid naming nothing succeeds; the reply says which of the
    // two happened.
    let existed = g.exists(uid);
    let cmd = goofi_graph::Command::RemoveNode { uid };
    state.history.lock().unwrap().apply(&mut g, actor, cmd)?;
    Ok(json!({ "removed": existed }))
}

/// Recovery, not an edit, so it is NOT routed through the command history: the client records no
/// `graph_cmd` for a restart and the two stacks must stay 1:1.
pub(crate) fn node_restart(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    {
        let mut g = state.graph.lock().unwrap();
        let uid = parse_uid(&g, payload, "node")?;
        g.restart_node(uid)?;
        // Pushed at once, so the red border lifts on the click rather than on the sweep.
        events.push(param_state_update(&g, uid, &[]));
    }
    resync_and_broadcast(state);
    Ok(json!({ "ok": true }))
}

/// Neither an edit nor recovery: a window on the machine goofi runs on, opened or closed.
pub(crate) fn node_editor(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let action = {
        let mut g = state.graph.lock().unwrap();
        let uid = parse_uid(&g, payload, "node")?;
        let show = payload.get("show").and_then(Value::as_bool).unwrap_or(true);
        g.node_editor(uid, show)?
    };
    Ok(json!({ "changed": action()? }))
}

pub(crate) fn link_add(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let (a, so, b, si) = parse_link(&g, payload, "link add")?;
    let (a, so) = wirable_endpoint(&g, a, &so, "from")?;
    let (b, si) = wirable_endpoint(&g, b, &si, "to")?;
    state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::AddLink {
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
        "from": format!("{}/{so}", named(&g, a)),
        "to": format!("{}/{si}", named(&g, b)),
        "dtype": dtype,
    }))
}

pub(crate) fn link_remove(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let (a, so, b, si) = parse_link(&g, payload, "link remove")?;
    let (a, so) = g.normalise(a, &so);
    let (b, si) = g.normalise(b, &si);
    // Idempotent for the same reason `remove_node` is, and answered the same way.
    let existed = g.has_link(a, &so, b, &si);
    state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::RemoveLink { node_out: a, slot_out: so, node_in: b, slot_in: si },
    )?;
    Ok(json!({ "removed": existed }))
}

/// NOT a command: options are runtime-only, so there is nothing to undo. They do not ride this
/// reply either — the hook runs on the node's own thread.
pub(crate) fn node_param_refresh(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    {
        let mut g = state.graph.lock().unwrap();
        let uid = parse_uid(&g, payload, "node")?;
        let (group, name) = parse_param_addr(payload, "node param refresh")?;
        g.refresh_param(uid, &group, &name)?;
    }
    resync_and_broadcast(state);
    Ok(json!({ "ok": true }))
}

/// NOT a command either: a pulse holds no state, so there is nothing to undo. The node acts on it
/// on its own thread, so this reply says only that the request was dispatched.
pub(crate) fn node_param_pulse(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let uid = parse_uid(&g, payload, "node")?;
    let (group, name) = parse_param_addr(payload, "node param pulse")?;
    g.pulse_param(uid, &group, &name)?;
    Ok(json!({ "ok": true }))
}

/// The joined `param_addr` — `group/param`, split on the FIRST `/` — one spelling on both surfaces.
fn parse_param_addr(payload: &Value, op: &str) -> Result<(String, String), String> {
    let addr = payload
        .get("param")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{op}: missing param"))?;
    let (group, name) = addr
        .split_once('/')
        .ok_or_else(|| format!("{op}: `{addr}` is not `group/param`"))?;
    Ok((group.to_string(), name.to_string()))
}

/// `--param` entries `{name: "group/param", …fields}`, folded into the `{group: {param: fields}}`
/// bag the engine path reads.
fn param_entries_bag(entries: &Value) -> Result<Value, String> {
    let list = entries.as_array().ok_or("`param` is a list of entries")?;
    let mut bag = serde_json::Map::new();
    for e in list {
        let mut o = e
            .as_object()
            .cloned()
            .ok_or(r#"a param entry is {"name": "group/param", …}"#)?;
        let addr = match o.remove("name") {
            Some(Value::String(s)) => s,
            _ => return Err(r#"a param entry names its param: {"name": "group/param", …}"#.into()),
        };
        let (group, name) =
            addr.split_once('/').ok_or_else(|| format!("`{addr}` is not `group/param`"))?;
        let taken = bag
            .entry(group.to_string())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .unwrap()
            .insert(name.to_string(), Value::Object(o));
        if taken.is_some() {
            return Err(format!("`{addr}` is named twice"));
        }
    }
    Ok(Value::Object(bag))
}

pub(crate) fn node_snapshot(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    // The address resolves exactly as a viewer's does: a facade or a port names the stream
    // BEHIND it, and one with nothing behind it yet is the unwired state, never an error.
    let key = {
        let g = state.graph.lock().unwrap();
        let (uid, slot) = parse_endpoint(&g, payload, "node snapshot", "output")?;
        if !g.exists(uid) {
            return Err(format!("node snapshot: no node {}", named(&g, uid)));
        }
        let slot = vocab::resolve_slot(&g, "node snapshot", uid, &slot)?;
        stream_behind(&g, uid, &slot)
    };
    let Some(key) = key else {
        return Ok(json!({
            "frame": null,
            "reason": "nothing is behind this port yet — wire its inside, then ask again",
        }));
    };
    let raw = flag(payload, "raw", false);
    match state.reducers.latest(key) {
        Some(d) => Ok(frame_json(&d, raw)),
        None => Ok(json!({
            "frame": null,
            "reason": "nothing cached for this slot yet — its feed is now open, so ask again \
                       after the node's next emit",
        })),
    }
}

/// A frame as the snapshot answers it: an ARRAY as its shape and range unless `raw` asks for the
/// numbers, STRING as its text, TABLE recursing.
fn frame_json(d: &goofi_core::Data, raw: bool) -> Value {
    let meta = meta_json(d.meta());
    match d.value() {
        goofi_core::Value::Array(s) if raw => {
            use base64::Engine;
            let npy = base64::engine::general_purpose::STANDARD.encode(npy_bytes(s));
            json!({ "meta": meta, "npy_b64": npy })
        }
        goofi_core::Value::Array(s) => json!({ "meta": meta, "shape": s.shape(), "range": range_json(s) }),
        goofi_core::Value::Str(s) => json!({ "meta": meta, "value": &**s }),
        goofi_core::Value::Table(t) => json!({ "meta": meta,
            "value": Value::Object(t.iter().map(|(k, v)| (k.clone(), frame_json(v, raw))).collect()) }),
    }
}

/// What an array holds, in three numbers — enough to tell silence from signal without the frame.
fn range_json(s: &goofi_core::ArrayStore) -> Value {
    let mut n = 0u64;
    let (mut min, mut max, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    for chunk in s.as_bytes().chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
        (min, max, sum, n) = (min.min(v), max.max(v), sum + v, n + 1);
    }
    match n {
        0 => Value::Null,
        n => json!({ "min": min, "max": max, "mean": sum / n as f64 }),
    }
}

/// NPY v1.0: goofi arrays are row-major, little-endian f32 by construction.
fn npy_bytes(s: &goofi_core::ArrayStore) -> Vec<u8> {
    let shape: String = s.shape().iter().map(|d| format!("{d},")).collect();
    let mut h =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape}), }}").into_bytes();
    h.resize(h.len() + (64 - (10 + h.len() + 1) % 64) % 64, b' ');
    h.push(b'\n');
    let mut out = Vec::with_capacity(10 + h.len() + s.as_bytes().len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(s.as_bytes());
    out
}

fn meta_json(m: &goofi_core::Meta) -> Value {
    Value::Object(
        m.iter().filter_map(|(k, v)| Some((k.clone(), meta_value_json(v)?))).collect(),
    )
}

/// `Bytes` carries no JSON form and is dropped; `Axes` becomes the `{dimN: [...]}` channels map.
fn meta_value_json(v: &goofi_core::MetaValue) -> Option<Value> {
    use goofi_core::MetaValue as M;
    Some(match v {
        M::Null => Value::Null,
        M::Bool(b) => json!(b),
        M::Int(i) => json!(i),
        M::Uint(u) => json!(u),
        M::Float(f) => json!(f),
        M::Str(s) => json!(s),
        M::List(l) => Value::Array(l.iter().filter_map(meta_value_json).collect()),
        M::Map(m) => {
            Value::Object(m.iter().filter_map(|(k, v)| Some((k.clone(), meta_value_json(v)?))).collect())
        }
        M::Bytes(_) => return None,
        M::Axes(a) => {
            let dims: serde_json::Map<String, Value> = a
                .dims()
                .map(|(dim, coords)| {
                    (dim, Value::Array(coords.iter().map(|c| match c {
                        goofi_core::Coord::Num(n) => json!(n),
                        goofi_core::Coord::Str(s) => json!(&**s),
                    }).collect()))
                })
                .collect();
            match dims.is_empty() {
                true => return None,
                false => Value::Object(dims),
            }
        }
    })
}

pub(crate) fn node_param_edit(
    state: &AppState,
    payload: &Value,
    actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let uid = parse_uid(&g, payload, "node")?;
    let (group, name) = parse_param_addr(payload, "node param edit")?;
    let mut entry = serde_json::Map::new();
    for key in ["value", "expression", "reference", "mode", "triggers"] {
        if let Some(v) = payload.get(key).filter(|v| !v.is_null()) {
            entry.insert(key.into(), v.clone());
        }
    }
    let bag = json!({ &group: { &name: entry } });
    let cmd = goofi_graph::param_commands(&g, uid, &bag)
        .map_err(|e| format!("node param edit: {e}"))?
        .pop()
        .ok_or("node param edit: nothing to change")?;
    state.history.lock().unwrap().apply(&mut g, actor, cmd)?;
    // The runtime `error` is doc-invisible, so echo the descriptor.
    events.push(param_state_update(&g, uid, &[]));
    Ok(json!({
        "value": g.params(uid)
            .and_then(|p| goofi_node::param(&p, &group, &name).cloned())
            .map(|p| goofi_graph::param_value_json(&p)),
        "error": g.param_source(uid, &group, &name).and_then(|s| s.error),
    }))
}

pub(crate) fn node_edit(
    state: &AppState,
    payload: &Value,
    actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let uid = parse_uid(&g, payload, "node")?;
    let name = payload.get("name").and_then(|v| v.as_str()).map(str::to_string);
    // The rename command tolerates a collision as a no-op so a stale replay converges; the
    // user-facing error therefore belongs here, at the forward RPC.
    if let Some(n) = &name {
        if g.name_taken(n, Some(uid)) {
            return Err(format!("node edit: the name `{n}` is taken"));
        }
    }
    if name.as_deref().is_some_and(|n| !goofi_core::globals::is_valid_name(n)) {
        return Err(format!(
            "node edit: `{}` is not a legal name: {}",
            name.unwrap_or_default(),
            goofi_core::globals::NAME_RULE
        ));
    }
    let pos = payload
        .get("pos")
        .filter(|v| !v.is_null())
        .map(|v| parse_pos(v).ok_or("node edit: pos is [x, y]"))
        .transpose()?;
    // Viewer entries MERGE slot by slot, so only the slots named move; the command then sets the
    // whole blob, which is what makes its inverse exact. The PATCH is what is checked — a stale
    // slot already stored is inert, and refusing it would block every later edit on a node whose
    // file changed its slots.
    let viewers = match payload.get("viewer").filter(|v| !v.is_null()) {
        Some(entries) => {
            let patch = viewer_entries_patch(entries)?;
            vocab::check_viewers(&g, uid, &patch)?;
            let mut whole = g.viewers(uid).cloned().ok_or("node edit: no such node")?;
            crate::doc::apply_merge(&mut whole, &Value::Object(patch));
            Some(whole)
        }
        None => None,
    };
    if name.is_none() && pos.is_none() && viewers.is_none() {
        return Err("node edit: give a name, pos or viewer".into());
    }
    let out = state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::EditNode { uid, name, pos, viewers },
    )?;
    // The runtime `error` is doc-invisible, so echo every referrer a rename rewrote.
    if let goofi_graph::Outcome::Nodes(referrers) = out {
        for r in referrers {
            events.push(param_state_update(&g, r, &[]));
        }
    }
    Ok(json!({ "ok": true }))
}

/// `--viewer` entries `{slot, …view}` — or `{slot, clear: true}` — as the merge patch the stored
/// blob takes, where clearing is the patch's `null`.
fn viewer_entries_patch(entries: &Value) -> Result<serde_json::Map<String, Value>, String> {
    let list = entries.as_array().ok_or("node edit: `viewer` is a list of entries")?;
    let mut patch = serde_json::Map::new();
    for e in list {
        let mut o = e
            .as_object()
            .cloned()
            .ok_or(r#"node edit: a viewer entry is {"slot", …} or {"slot", "clear": true}"#)?;
        let slot = match o.remove("slot") {
            Some(Value::String(s)) => s,
            _ => return Err("node edit: a viewer entry names its slot".into()),
        };
        let taken = match o.remove("clear") {
            Some(Value::Bool(true)) => patch.insert(slot.clone(), Value::Null),
            None => patch.insert(slot.clone(), Value::Object(o)),
            _ => return Err("node edit: `clear` is only ever true — omit it to set".into()),
        };
        if taken.is_some() {
            return Err(format!("node edit: slot `{slot}` is named twice"));
        }
    }
    Ok(patch)
}

/// Where THIS client is looking: not a doc root, so it neither drags a peer nor raises the
/// unsaved dot, but it still rides the `.gfi` and `hello`.
pub(crate) fn layout_viewpoint_edit(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    {
        let mut g = state.graph.lock().unwrap();
        g.set_viewpoint(payload.get("value").cloned().unwrap_or(Value::Null));
    }
    resync_and_broadcast(state);
    Ok(json!({ "ok": true }))
}

pub(crate) fn layout_inspect(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    let tab = payload.get("tab").and_then(|v| v.as_str()).map(str::to_string);
    Ok(json!({ "text": inspect::layout_tree(&g, tab.as_deref()) }))
}

/// Relabel a TAB — refused for any other kind of id, because an `edit` op edits ONE kind.
pub(crate) fn layout_tab_edit(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let tab = parse_str(payload, "tab")?.to_string();
    let name = parse_str(payload, "name")?;
    let writes = g.arrangement().rename_tab(&tab, name)?;
    apply_layout(state, &mut g, actor, goofi_graph::Command::LayoutContents { writes })
}

/// Edit a PANEL's content: its type, its state, or both — one call, one undo.
pub(crate) fn layout_panel_edit(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let panel = parse_str(payload, "panel")?.to_string();
    let ty = payload.get("type").and_then(|v| v.as_str()).map(str::to_string);
    let panel_state = payload.get("state").cloned().filter(|v| !v.is_null());
    if ty.is_none() && panel_state.is_none() {
        return Err("layout panel edit: give a type, a state, or both".into());
    }
    // A panel bound to a node that is not there renders empty and explains nothing.
    let named = panel_state
        .as_ref()
        .and_then(|s| s.get("node"))
        .and_then(|v| v.as_str())
        .filter(|n| !n.is_empty());
    if let Some(node) = named {
        if !bindable_node(&g, node) {
            return Err(format!("layout panel edit: no node `{node}` in this patch"));
        }
    }
    // The slot is checked against the node this write LEAVES the panel bound to: its own, or
    // the one already stored, since a state write merges.
    let bound = named
        .or_else(|| {
            g.arrangement().panel_state(&panel).and_then(|s| s.get("node")).and_then(|v| v.as_str())
        })
        .and_then(Uid::from_hex);
    vocab::check_panel(&g, ty.as_deref(), panel_state.as_ref(), bound)?;
    // A control panel is born naming a group of its own, `control0`, `control1`, … — the first
    // that nothing holds — so no panel ever waits on a name.
    let panel_state = match (ty.as_deref(), panel_state) {
        (Some("control"), state) if state.as_ref().and_then(|s| s.get("group")).is_none() => {
            let taken = g.arrangement().control_panels();
            let fresh = (0..)
                .map(|n| format!("control{n}"))
                .find(|c| !g.globals().has_group(c) && !taken.iter().any(|(_, held)| held == c))
                .expect("the integers do not run out");
            let mut s = state.and_then(|s| s.as_object().cloned()).unwrap_or_default();
            s.insert("group".into(), json!(fresh));
            Some(Value::Object(s))
        }
        (_, state) => state,
    };
    let writes = g.arrangement().set_panel(&panel, ty.as_deref(), panel_state)?;
    apply_layout(state, &mut g, actor, goofi_graph::Command::LayoutContents { writes })
}

/// Set the shares of ALL of a SPLIT's children at once — what a resize drag commits.
pub(crate) fn layout_split_edit(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let split = parse_str(payload, "split")?.to_string();
    // A non-numeric entry becomes NaN, which the planner refuses beside a zero or a negative
    // one — so "is this a fraction" is answered in one place.
    let fractions: Vec<f64> = payload
        .get("fraction")
        .and_then(|v| v.as_array())
        .ok_or("layout split edit: `fraction` is a list of numbers")?
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect();
    // Planned here only so a bad split or a wrong fraction count answers teachably; the command
    // re-plans it under this same lock.
    g.arrangement().resize_split(&split, &fractions)?;
    apply_layout(state, &mut g, actor,
                 goofi_graph::Command::LayoutResizeSplit { split, fractions })
}

/// A fresh empty panel: beside a target, or on a new tab of its own.
pub(crate) fn layout_panel_add(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    const OP: &str = "layout panel add";
    let mut g = state.graph.lock().unwrap();
    let beside = payload.get("beside").and_then(|v| v.as_str());
    let ratio = payload.get("ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
    match beside {
        // Beside a target, dividing it — the drop on a panel's edge.
        Some(target) => {
            let side = parse_side(payload, OP)?;
            let (plan, fresh) = g.arrangement().split_panel(target, side, ratio)?;
            let cmd = goofi_graph::Command::LayoutBirth { plan, born: fresh.clone() };
            let text = apply_layout(state, &mut g, actor, cmd)?;
            let tab = g.arrangement().tab_of(&fresh).unwrap_or_default();
            Ok(json!({ "id": fresh, "tab": tab, "text": text["text"] }))
        }
        // On a tab of its own, at `index` in the strip, labelled `name` or minted.
        None => {
            let name = payload.get("name").and_then(|v| v.as_str());
            let index = payload.get("index").and_then(|v| v.as_u64()).map(|i| i as usize);
            let (plan, tab) = g.arrangement().add_tab(name, index, None)?;
            let cmd = goofi_graph::Command::LayoutBirth { plan, born: tab.clone() };
            let text = apply_layout(state, &mut g, actor, cmd)?;
            // The root panel's id, which a caller cannot otherwise know.
            let id = g.arrangement().root_of(&tab).unwrap_or_default();
            Ok(json!({ "id": id, "tab": tab, "text": text["text"] }))
        }
    }
}

/// Move a layout entry — a panel, a subtree or a tab; ONE op per drag gesture, so a drop is one
/// undo step and peers never see an arrangement that was not on somebody's screen.
pub(crate) fn layout_move(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    const OP: &str = "layout move";
    let mut g = state.graph.lock().unwrap();
    let entry = parse_str(payload, "entry")?.to_string();
    let beside = payload.get("beside").and_then(|v| v.as_str()).map(str::to_string);
    let within = payload.get("in").and_then(|v| v.as_str()).map(str::to_string);
    let index = payload.get("index").and_then(|v| v.as_u64()).map(|i| i as usize);
    let ratio = payload.get("ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
    // A tab already has a tab, so a destination-less move is a reorder rather than a wrap. The
    // id says which — as it does for the edit trio and `layout remove`.
    let is_tab = g.arrangement().tab_index(&entry).is_some();
    let (plan, placed) = match (beside.as_deref(), within.as_deref()) {
        (Some(_), Some(_)) => {
            return Err(format!("{OP}: `--beside` and `--in` are two destinations — give one"))
        }
        // Beside a target, dividing it; the side defaults right, as a birth's does.
        (Some(target), None) => {
            let side = parse_side(payload, OP)?;
            (g.arrangement().insert_at_panel(&entry, target, side, ratio)?, entry.clone())
        }
        // Inside a split, at an index — the drop into a container that exists.
        (None, Some(parent)) => {
            (g.arrangement().move_subtree(&entry, parent, index.unwrap_or(0))?, entry.clone())
        }
        (None, None) if is_tab => {
            let at = index.ok_or(format!("{OP}: a tab moves to an `--index` in the strip"))?;
            g.arrangement().reorder_tab(&entry, at)?;
            let cmd = goofi_graph::Command::LayoutReorderTab { tab: entry.clone(), to_index: at };
            let text = apply_layout(state, &mut g, actor, cmd)?;
            return Ok(json!({ "id": entry, "tab": entry, "text": text["text"] }));
        }
        // Onto a tab of its own — the drag onto the tab bar. A tab built AROUND an existing
        // subtree is a MOVE, so its undo gives the subtree back.
        (None, None) => {
            let name = payload.get("name").and_then(|v| v.as_str());
            let (plan, tab) = g.arrangement().add_tab(name, index, Some(&entry))?;
            let cmd = goofi_graph::Command::LayoutMove {
                plan: Some(plan), root: entry.clone(), home: None };
            let text = apply_layout(state, &mut g, actor, cmd)?;
            return Ok(json!({ "id": entry, "tab": tab, "text": text["text"] }));
        }
    };
    let cmd = goofi_graph::Command::LayoutMove { plan: Some(plan), root: placed.clone(), home: None };
    let text = apply_layout(state, &mut g, actor, cmd)?;
    let tab = g.arrangement().tab_of(&placed).unwrap_or_default();
    Ok(json!({ "id": placed, "tab": tab, "text": text["text"] }))
}

pub(crate) fn layout_remove(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let panel = parse_str(payload, "entry")?.to_string();
    // A tab is closed whole; anything else is closed with promote. Planned here only so a bad id
    // answers teachably: `LayoutClose` re-plans it under this same lock, and DEGRADES rather
    // than errors.
    match g.arrangement().tab_index(&panel) {
        Some(_) => g.arrangement().remove_tab(&panel)?,
        None => g.arrangement().remove_subtree(&panel)?,
    };
    apply_layout(state, &mut g, actor, goofi_graph::Command::LayoutClose { born: panel })
}

/// The `control` payload: absent leaves the record alone, null clears it, an object sets it.
fn parse_control(payload: &Value) -> Result<Option<Option<goofi_core::globals::Control>>, String> {
    match payload.get("control") {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(v) => serde_json::from_value(v.clone())
            .map(|c| Some(Some(c)))
            .map_err(|e| format!("control: {e}")),
    }
}

/// `cmds`, lifted past the group's config lock for their one command: a control panel owns its
/// group, so its door edits a locked group and leaves it locked — the undo of the whole is one step.
fn through_the_lock(g: &goofi_graph::Graph, group: &str, cmds: Vec<goofi_graph::Command>) -> goofi_graph::Command {
    use goofi_graph::Command;
    let held = g.globals().group_lock(group);
    if !held.config {
        return Command::Compound(cmds);
    }
    let mut all = vec![Command::LockGlobalGroup {
        group: group.to_string(),
        lock: goofi_core::globals::Lock { config: false, value: held.value },
    }];
    all.extend(cmds);
    all.push(Command::LockGlobalGroup { group: group.to_string(), lock: held });
    Command::Compound(all)
}

fn parse_kind(v: &Value) -> Result<goofi_core::globals::ControlKind, String> {
    serde_json::from_value(v.clone()).map_err(|_| {
        let kinds = goofi_core::globals::ControlKind::ALL.iter().map(|k| k.as_str()).collect::<Vec<_>>().join("/");
        format!("`kind` is one of {kinds}, not `{v}`")
    })
}

/// A control panel's element, addressed `group` + `element`, as the `control` ops read it.
fn element_of(g: &goofi_graph::Graph, op: &str, payload: &Value) -> Result<(String, String, String), String> {
    let group = parse_str(payload, "group")?.to_string();
    let element = parse_str(payload, "element")?.to_string();
    let name = format!("{group}.{element}");
    if g.globals().control(&name).is_none() {
        return Err(format!("{op}: no element `{element}` in control panel `{group}`"));
    }
    Ok((group, element, name))
}

fn element_json(g: &goofi_graph::Graph, name: &str, value: &goofi_core::globals::GlobalValue) -> Value {
    let mut e = goofi_graph::global_to_json(value);
    e["name"] = json!(name);
    e["element"] = json!(name.split_once('.').map(|(_, el)| el).unwrap_or(name));
    e["lock"] = serde_json::to_value(g.globals().lock_of(name)).expect("a plain record");
    e["control"] = serde_json::to_value(g.globals().control(name)).expect("a plain record");
    if let Some(s) = g.globals().source(name) {
        e["source"] = serde_json::to_value(s).expect("a plain record");
    }
    e
}

pub(crate) fn control_list(
    state: &AppState,
    _payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    let panels: Vec<Value> =
        g.arrangement().control_panels().into_iter().map(|(panel, group)| json!({ "panel": panel, "group": group })).collect();
    let mut groups = serde_json::Map::new();
    let named: Vec<String> = g.arrangement().control_panels().into_iter().map(|(_, group)| group).collect();
    let mut order: Vec<String> = named.clone();
    for (name, _, _, control, _) in g.globals().entries() {
        if let (Some(_), Some((group, _))) = (control, name.split_once('.')) {
            if !order.iter().any(|o| o == group) {
                order.push(group.to_string());
            }
        }
    }
    for group in order {
        let elements: Vec<Value> = g
            .globals()
            .entries()
            .filter(|(name, _, _, control, _)| control.is_some() && name.split_once('.').is_some_and(|(gr, _)| gr == group))
            .map(|(name, value, ..)| element_json(&g, name, value))
            .collect();
        groups.insert(group.clone(), json!({ "lock": g.globals().group_lock(&group), "elements": elements }));
    }
    Ok(json!({ "panels": panels, "groups": groups }))
}

/// Bear a widget: the manager mints its name and its cell where none is given.
pub(crate) fn control_add(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    use goofi_core::globals::{free_cell, is_valid_identifier};
    let mut g = state.graph.lock().unwrap();
    let group = parse_str(payload, "group")?.to_string();
    if !is_valid_identifier(&group) {
        return Err(format!("control add: invalid group `{group}`: {}", goofi_core::globals::GLOBAL_NAME_RULE));
    }
    let kind = parse_kind(payload.get("kind").ok_or("control add: missing kind")?).map_err(|e| format!("control add: {e}"))?;
    let element = match payload.get("element").and_then(|v| v.as_str()) {
        Some(e) => e.to_string(),
        None => (0..)
            .map(|n| format!("{}{n}", kind.as_str()))
            .find(|e| g.globals().get(&format!("{group}.{e}")).is_none())
            .expect("the integers do not run out"),
    };
    let name = format!("{group}.{element}");
    if g.globals().get(&name).is_some() {
        return Err(format!("control add: `{name}` already exists — `control edit` changes it"));
    }
    let born = kind.born_value();
    let value = match payload.get("value").filter(|v| !v.is_null()) {
        Some(v) => goofi_graph::global_from_json(&json!({ "value": v, "type": born.type_name() }))
            .ok_or_else(|| format!("control add: `{v}` is not a {}", born.type_name()))?,
        None => born,
    };
    let (bw, bh) = kind.born_box();
    let num = |key: &str, or: f64| payload.get(key).and_then(|v| v.as_f64()).unwrap_or(or);
    let (w, h) = (num("w", bw), num("h", bh));
    let (x, y) = match (payload.get("x").and_then(|v| v.as_f64()), payload.get("y").and_then(|v| v.as_f64())) {
        (Some(x), Some(y)) => (x, y),
        _ => {
            let taken: Vec<(f64, f64, f64, f64)> = g
                .globals()
                .entries()
                .filter(|(n, ..)| n.split_once('.').is_some_and(|(gr, _)| gr == group))
                .filter_map(|(_, _, _, c, _)| c.map(|c| (c.x, c.y, c.w, c.h)))
                .collect();
            free_cell(&taken, w, h)
        }
    };
    let mut record = json!({ "kind": kind, "x": x, "y": y, "w": w, "h": h });
    for key in ["min", "max", "step", "options"] {
        if let Some(v) = payload.get(key).filter(|v| !v.is_null()) {
            record[key] = v.clone();
        }
    }
    if matches!(value, goofi_core::globals::GlobalValue::Float(_)) {
        for (key, or) in [("min", json!(0.0)), ("max", json!(1.0)), ("step", json!(0.01))] {
            record.as_object_mut().unwrap().entry(key).or_insert(or);
        }
    }
    let control: goofi_core::globals::Control = serde_json::from_value(record.clone()).map_err(|e| format!("control add: {e}"))?;
    let cmd = through_the_lock(
        &g,
        &group,
        vec![goofi_graph::Command::EditGlobal { name: name.clone(), value: Some(value), at: None, control: Some(Some(control)) }],
    );
    state.history.lock().unwrap().apply(&mut g, actor, cmd)?;
    Ok(json!({ "name": name, "control": record }))
}

pub(crate) fn control_edit(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let (group, _, name) = element_of(&g, "control edit", payload)?;
    let mut cmds = Vec::new();
    let mut target = name.clone();
    if let Some(to) = payload.get("name").and_then(|v| v.as_str()) {
        target = format!("{group}.{to}");
        if target != name {
            cmds.push(goofi_graph::Command::RenameGlobal { from: name.clone(), to: target.clone() });
        }
    }
    let mut record = serde_json::to_value(g.globals().control(&name)).expect("a plain record");
    let mut touched = false;
    for key in ["kind", "min", "max", "step", "options", "x", "y", "w", "h"] {
        if let Some(v) = payload.get(key) {
            if key == "kind" {
                parse_kind(v).map_err(|e| format!("control edit: {e}"))?;
            }
            record[key] = v.clone();
            touched = true;
        }
    }
    if touched {
        let control: goofi_core::globals::Control = serde_json::from_value(record).map_err(|e| format!("control edit: {e}"))?;
        // The value it holds rides along under the new name: `None` would be a remove.
        let value = g.globals().get(&name).cloned();
        cmds.push(goofi_graph::Command::EditGlobal { name: target.clone(), value, at: None, control: Some(Some(control)) });
    }
    if cmds.is_empty() {
        return Err("control edit: nothing to change — give a name, a kind, a range, options or a cell".into());
    }
    let cmd = through_the_lock(&g, &group, cmds);
    state.history.lock().unwrap().apply(&mut g, actor, cmd)?;
    Ok(json!({ "name": target }))
}

pub(crate) fn control_remove(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let (group, _, name) = element_of(&g, "control remove", payload)?;
    let cmd = through_the_lock(&g, &group, vec![goofi_graph::Command::EditGlobal { name, value: None, at: None, control: None }]);
    state.history.lock().unwrap().apply(&mut g, actor, cmd)?;
    Ok(json!({ "removed": true }))
}

pub(crate) fn control_source(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let (group, _, name) = element_of(&g, "control source", payload)?;
    let reference = parse_str(payload, "reference")?.trim().to_string();
    let index = match payload.get("index") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().map(|i| i as usize).ok_or_else(|| format!("control source: `index` is a whole number, not `{v}`"))?),
    };
    let source = (!reference.is_empty()).then_some(goofi_core::globals::GlobalSource { reference, index });
    let cmd = through_the_lock(&g, &group, vec![goofi_graph::Command::SourceGlobal { name, source: source.clone() }]);
    state.history.lock().unwrap().apply(&mut g, actor, cmd)?;
    Ok(json!({ "source": source }))
}

/// Create a global. Every expression reading one depends on its TYPE, so the type is declared
/// at birth and immutable after — re-typing is a remove and an add.
pub(crate) fn global_add(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let name = parse_str(payload, "name")?.to_string();
    if g.globals().get(&name).is_some() {
        return Err(format!("global entry add: `{name}` already exists — `global entry edit` changes it"));
    }
    let ty = parse_str(payload, "type")?;
    let val = payload.get("value").filter(|v| !v.is_null()).ok_or("global entry add: missing value")?;
    let value = goofi_graph::global_from_json(&json!({ "value": val, "type": ty }))
        .ok_or_else(|| format!("global entry add: `{val}` is not a {ty}"))?;
    let control = parse_control(payload)?;
    state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::EditGlobal { name, value: Some(value.clone()), at: None, control },
    )?;
    // As STORED: the conversion is type-directed, so a fraction into an int rounds.
    Ok(json!({ "value": goofi_graph::global_to_json(&value)["value"] }))
}

pub(crate) fn global_edit(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let name = parse_str(payload, "name")?.to_string();
    let held = g.globals().get(&name).map(goofi_graph::global_to_json);
    let Some(held) = held else {
        return Err(format!("global entry edit: no global `{name}` — `global entry add` creates one"));
    };
    let ty = held["type"].as_str().unwrap_or_default().to_string();
    let control = parse_control(payload)?;
    let val = payload.get("value").filter(|v| !v.is_null());
    // A control-only edit is what the panel sends when it moves a widget, so the value is optional
    // once a `control` is given.
    let val = match (val, &control) {
        (Some(v), _) => v,
        (None, Some(_)) => &held["value"],
        (None, None) => return Err("global entry edit: missing value".to_string()),
    };
    let value = goofi_graph::global_from_json(&json!({ "value": val, "type": ty }))
        .ok_or_else(|| format!("global entry edit: `{val}` is not a {ty}"))?;
    state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::EditGlobal { name, value: Some(value.clone()), at: None, control },
    )?;
    Ok(json!({ "value": goofi_graph::global_to_json(&value)["value"] }))
}

pub(crate) fn global_remove(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let name = parse_str(payload, "name")?.to_string();
    if g.globals().get(&name).is_none() {
        return Err(format!("global entry remove: no global `{name}`"));
    }
    state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::EditGlobal { name, value: None, at: None, control: None },
    )?;
    Ok(json!({ "removed": true }))
}

pub(crate) fn global_rename(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let from = parse_str(payload, "name")?.to_string();
    let to = parse_str(payload, "to")?.to_string();
    state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::RenameGlobal { from, to: to.clone() },
    )?;
    Ok(json!({ "name": to }))
}

/// The lock a payload asks for over `held`: an axis it does not name keeps what it has.
fn parse_lock(payload: &Value, held: goofi_core::globals::Lock) -> Result<goofi_core::globals::Lock, String> {
    let axis = |key: &str, have: bool| match payload.get(key) {
        None | Some(Value::Null) => Ok(have),
        Some(Value::Bool(b)) => Ok(*b),
        Some(other) => Err(format!("`{key}` is a bool, not `{other}`")),
    };
    Ok(goofi_core::globals::Lock { config: axis("config", held.config)?, value: axis("value", held.value)? })
}

pub(crate) fn global_lock(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let name = parse_str(payload, "name")?.to_string();
    if g.globals().get(&name).is_none() {
        return Err(format!("global entry lock: no global `{name}`"));
    }
    let held = g.globals().entries().find(|(n, ..)| *n == name).map(|(_, _, l, ..)| l).unwrap_or_default();
    let lock = parse_lock(payload, held).map_err(|e| format!("global entry lock: {e}"))?;
    state.history.lock().unwrap().apply(&mut g, actor, goofi_graph::Command::LockGlobal { name, lock })?;
    Ok(json!({ "lock": lock }))
}

pub(crate) fn global_source(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let name = parse_str(payload, "name")?.to_string();
    if g.globals().get(&name).is_none() {
        return Err(format!("global entry source: no global `{name}`"));
    }
    let reference = parse_str(payload, "reference")?.trim().to_string();
    let index = match payload.get("index") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64().map(|i| i as usize).ok_or_else(|| format!("global entry source: `index` is a whole number, not `{v}`"))?,
        ),
    };
    let source = (!reference.is_empty()).then_some(goofi_core::globals::GlobalSource { reference, index });
    state.history.lock().unwrap().apply(&mut g, actor, goofi_graph::Command::SourceGlobal { name, source: source.clone() })?;
    Ok(json!({ "source": source }))
}

pub(crate) fn global_group_lock(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let group = parse_str(payload, "group")?.to_string();
    let lock = parse_lock(payload, g.globals().group_lock(&group)).map_err(|e| format!("global group lock: {e}"))?;
    state.history.lock().unwrap().apply(&mut g, actor, goofi_graph::Command::LockGlobalGroup { group, lock })?;
    Ok(json!({ "lock": lock }))
}

pub(crate) fn global_group_rename(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let from = parse_str(payload, "from")?.to_string();
    let to = parse_str(payload, "to")?.to_string();
    state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::RenameGlobalGroup { from, to: to.clone() },
    )?;
    Ok(json!({ "group": to }))
}

pub(crate) fn nodes_group(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let uids = parse_uid_list(&g, payload, "nodes")?;
    let pos = payload
        .get("pos")
        .filter(|v| !v.is_null())
        .map(|v| parse_pos(v).ok_or("nodes group: pos is [x, y]"))
        .transpose()?
        .unwrap_or([0.0, 0.0]);
    let out = state.history.lock().unwrap().apply(
        &mut g,
        actor,
        goofi_graph::Command::Group { members: uids, pos, restore: None },
    )?;
    let inst = match out {
        goofi_graph::Outcome::Uid(u) => u,
        _ => return Err("nodes group: no scope uid returned".into()),
    };
    Ok(json!({ "name": named(&g, inst), "inst_id": inst.to_hex() }))
}

pub(crate) fn nodes_ungroup(
    state: &AppState,
    payload: &Value,
    actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    let inst = parse_uid(&g, payload, "subpatch")?;
    state
        .history
        .lock()
        .unwrap()
        .apply(&mut g, actor, goofi_graph::Command::Expand { scope: inst })?;
    Ok(json!({ "ok": true }))
}

pub(crate) fn nodes_inspect(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    let scope = parse_uid_opt(&g, payload, "scope", "nodes inspect")?;
    Ok(json!({ "text": inspect::patch(&g, scope)? }))
}

pub(crate) fn node_state(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    let uid = parse_uid(&g, payload, "node")?;
    let want = |k: &str| payload.get(k).and_then(|v| v.as_bool()).unwrap_or(true);
    let slot = payload.get("slot").and_then(|v| v.as_str());
    let text = inspect::node(&g, uid, slot, want("params"), want("error"))?;
    Ok(json!({ "text": text }))
}

pub(crate) fn global_list(
    state: &AppState,
    _payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    Ok(inspect::globals(&g))
}

/// The open patch's identity AND its health.
pub(crate) fn session_status(
    state: &AppState,
    _payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    // The walks (dirty, mount) run before the graph lock, as everywhere.
    let save_path = state.save_path();
    let workspace = goofi_core::path::to_slash(&state.mount());
    let dirty = state.is_dirty();
    let mut g = state.graph.lock().unwrap();
    let errors = inspect::errors(&g);
    // A demo registers no audio engine, and a machine with no adapter no graphics one. Status is
    // a READ: it answers what is there.
    let audio = crate::try_audio_engine(&mut g).map(|a| a.status());
    let graphics = crate::try_graphics_engine(&mut g).map(|a| a.status());
    Ok(json!({
        // The id is what the session-file probe verifies: a listener that answers with another
        // id — or none — is not this session.
        "instance_id": &*state.instance_id,
        "save_path": save_path,
        "workspace": workspace,
        "dirty": dirty,
        "errors": errors,
        // The timing door: what the clock is doing, read by hand on a device before it is trusted.
        // Null where there is no audio engine to ask — a demo runs the signal plane alone.
        "audio": audio.map(|a| json!({
            "clock": a.clock,
            "device": a.device,
            "rate": a.rate,
            "channels": a.channels,
            "callbacks": a.callbacks,
            "xruns": a.xruns,
            "render_max_us": a.render_max_us,
        })),
        // The same door for the GPU: which adapter answered, and how much of a tick it took.
        "graphics": graphics.map(|a| json!({
            "clock": a.clock,
            "adapter": a.adapter,
            "backend": a.backend,
            "frames": a.frames,
            "stages": a.stages,
            "tick_max_us": a.tick_max_us,
        })),
    }))
}

pub(crate) fn session_manifest(
    state: &AppState,
    _payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let g = state.graph.lock().unwrap();
    Ok(json!({ "yaml": g.serialize() }))
}

/// The mount is a per-run temp directory under a random name, so asking is the only way a client
/// or a harness can find it.
pub(crate) fn session_save(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let mut g = state.graph.lock().unwrap();
    // Expand `~` exactly as the browser does — the two must agree on what a path means. No path
    // means the patch's HOME, and a patch that never had one is refused rather than guessed at.
    let path = match payload.get("path").and_then(|v| v.as_str()) {
        Some(p) => fsbrowse::resolve(p),
        None => state.save_path().ok_or(
            "session save: this patch has no home yet — give a path")?,
    };
    let mount = state.mount();
    // Sampled BEFORE the pack: baselining after would call a file written during the zip packed
    // either way, which is the direction that LOSES an edit.
    g.persist();
    let packed = goofi_graph::archive::fingerprint(&mount);
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

/// The core every patch replacement shares, so nothing after the read can drift between the
/// sources: a `.gfi`, an inline manifest, or nothing at all — the empty patch.
fn load_patch(state: &AppState, payload: &Value) -> Result<Value, String> {
    // Read OFF the graph lock, as the hello does: the roster's config half is a disk read.
    let agents = goofi_core::home::agents();
    // Every source mounts FRESH, and the live mount is swapped only once the manifest has parsed,
    // so a refused load leaves the open patch untouched on both planes. Staged and built off the
    // lock: the archive's own Rust nodes may take seconds to build.
    let fresh = new_mount();
    let (content, from_path) = stage_load(&fresh, payload).inspect_err(|_| remove_mount(&fresh))?;
    prebuild(state, &fresh);
    let result = {
        let mut g = state.graph.lock().unwrap();
        // ORDER is load-bearing: the types the patch SHIPS are registered before the manifest
        // resolves, or the unknown-type gate fires on the nodes the archive brought.
        rescan(state, &mut g, &fresh);
        // Parse BEFORE anything is announced or committed.
        if let Err(e) = g.load_doc(&content, &fresh) {
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
        // Sent under the lock, not queued for the dispatcher: the status worker's next stage delta
        // needs this lock, so nothing it says can overtake the snapshot it is a delta over.
        let _ = state.events.send(event("harness_changed", state.harnesses.roster(&agents)));
        // `read_gfi` restores no mtimes, so without a baseline taken HERE a patch would be dirty
        // from the moment it finished loading.
        *state.workspace_baseline.lock().unwrap() =
            goofi_graph::archive::fingerprint(&state.mount());
        // A load fully resets the session: there is nothing to undo across it.
        state.history.lock().unwrap().clear();
        if let Some(e) = state.set_dirty(false) {
            let _ = state.events.send(e);
        }
        // NONE for an inline load and for `session new`, neither with a file behind it: an
        // inherited path would aim the next silent save at an unrelated `.gfi`.
        *state.save_path.lock().unwrap() = from_path.clone();
        let _ = state.events.send(event(
            "graph_replaced",
            schemas::snapshot(&g, &state.instance_id, false, false, from_path.as_deref(),
                              state.harnesses.roster(&agents), state.mode.demo),
        ));
        // The patch brought its own node types, which `graph_replaced` does not carry.
        let _ = state.events.send(event("node_types", json!({ "types": schemas::catalog_types(&g, Detail::Full) })));
        if let Some(path) = from_path {
            let _ = state.events.send(event("save_path_changed", json!({ "save_path": path })));
        }
        // A stored arrangement this model admits but cannot render falls back to the default, so
        // the reply says so rather than leaving the change unexplained.
        json!({ "ok": true, "layout_warning": g.arrangement_warning() })
    };
    resync_and_broadcast(state);
    Ok(result)
}

pub(crate) fn session_load(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    // A source is REQUIRED: no bare word may be the destructive New. `session new` is explicit.
    let has_source = payload.get("path").and_then(|v| v.as_str()).is_some_and(|p| !p.is_empty())
        || payload.get("content").and_then(|v| v.as_str()).is_some();
    if !has_source {
        return Err("session load: give a `path` or `--content` — `session new` opens the empty patch".into());
    }
    load_patch(state, payload)
}

pub(crate) fn session_new(
    state: &AppState,
    _payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    load_patch(state, &json!({}))
}

pub(crate) fn undo(
    state: &AppState,
    _payload: &Value,
    actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let result = {
        let mut g = state.graph.lock().unwrap();
        let mut hist = state.history.lock().unwrap();
        let changed = hist.undo(&mut g, actor)?;
        json!({ "changed": changed, "can_undo": hist.can_undo(actor), "can_redo": hist.can_redo(actor) })
    };
    resync_and_broadcast(state);
    // Only a flip that CHANGED something raises the dot: an empty stack is not an edit.
    if result["changed"] == json!(true) {
        events.extend(state.set_dirty(true));
    }
    Ok(result)
}

pub(crate) fn redo(
    state: &AppState,
    _payload: &Value,
    actor: &str,
    events: &mut Vec<String>,
) -> Result<Value, String> {
    let result = {
        let mut g = state.graph.lock().unwrap();
        let mut hist = state.history.lock().unwrap();
        let changed = hist.redo(&mut g, actor)?;
        json!({ "changed": changed, "can_undo": hist.can_undo(actor), "can_redo": hist.can_redo(actor) })
    };
    resync_and_broadcast(state);
    // Only a flip that CHANGED something raises the dot: an empty stack is not an edit.
    if result["changed"] == json!(true) {
        events.extend(state.set_dirty(true));
    }
    Ok(result)
}

/// The registry itself, as data a caller derives a whole client from.
pub(crate) fn op_list(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let doc = flag(payload, "doc", false);
    let ops: Vec<Value> = state
        .ops()
        .iter()
        .map(|o| {
            let mut row = json!({ "op": o.name, "args": o.args, "kind": o.handler.kind_name() });
            if doc {
                row["positional"] = json!(o.positional);
                row["doc"] = json!(o.doc());
                row["result"] = json!(o.result);
            }
            row
        })
        .collect();
    Ok(json!({ "ops": ops }))
}

pub(crate) fn op_complete(
    state: &AppState,
    payload: &Value,
    _actor: &str,
    _events: &mut Vec<String>,
) -> Result<Value, String> {
    let line = payload.get("line").and_then(Value::as_str).unwrap_or_default();
    let rows: Vec<String> = crate::phrase::complete(state.ops(), Some(state), line)
        .into_iter()
        .map(|(word, doc)| format!("{word}\t{doc}"))
        .collect();
    Ok(json!({ "text": rows.join("\n") }))
}
