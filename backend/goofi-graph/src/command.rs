//! Patch commands with exact inverses — the manager's undo/redo unit.

use crate::{Graph, Uid};
use goofi_core::globals::GlobalValue;
use goofi_core::Param;

use crate::Mode;
use goofi_node::{param, ParamGroups};

/// What a command produced, for the caller. Kept serde-free so the engine needs no JSON dep.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// A plain success (`{ ok: true }` on the wire).
    Ok,
    /// A minted/affected uid — `add_node`/`group`/a boundary add return the node/scope/stub uid.
    Uid(Uid),
    /// Nodes the command touched that need a runtime echo — a rename's rewritten referrers, so
    /// the bridge re-broadcasts their params.
    Nodes(Vec<Uid>),
}

/// A param's source record as [`Command::EditParam`] carries it: the mode, and the expression and
/// reference it retains whatever the mode. Nothing retained and a constant mode is no record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceState {
    pub mode: Mode,
    pub expression: String,
    pub reference: String,
    pub triggers: bool,
}

impl SourceState {
    pub fn is_empty(&self) -> bool {
        self.mode == Mode::Constant && self.expression.is_empty() && self.reference.is_empty()
    }
}

/// The captured state to recreate a scope EXACTLY — the inverse of [`Command::Expand`], which
/// restores the exact scope id rather than minting a fresh one. Its PORTS are not in here: they
/// are nodes, so they come back as the `AddNode` children beside this command.
#[derive(Clone, Debug, PartialEq)]
pub struct ScopeRestore {
    pub scope_id: Uid,
    pub name: String,
    /// The scope's parent, captured explicitly (not derived from members) so an EMPTY scope — a
    /// sub-patch whose members were all deleted — restores at the right place. `None` = ROOT.
    pub parent: Option<Uid>,
}

/// One semantic patch edit. Every variant has an exact inverse (see [`Command::execute`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Executed in order; its inverse is the children's inverses in REVERSE order.
    Compound(Vec<Command>),
    AddNode {
        type_name: String,
        pos: [f64; 2],
        /// `Some` restores a specific uid (undo/redo, so links + panels reconnect); `None` mints one.
        uid: Option<Uid>,
        name: Option<String>,
        /// `Some` restores captured params (a `RemoveNode` inverse); `None` uses the type's defaults.
        params: Option<ParamGroups>,
        /// Captured source records `(group, name, state)` to re-apply. Empty for a user add.
        sources: Vec<(String, String, SourceState)>,
        /// Captured viewer view-state blob to restore; `None` for a user add (defaults to empty).
        viewers: Option<serde_json::Value>,
        /// The scope to create the node INSIDE (`None` = ROOT). A PORT's membership rides HERE and
        /// nowhere else: one cannot be created without a scope. Every other kind is placed by a
        /// [`Command::SetScope`] child, after every uid the capture names exists.
        scope: Option<Uid>,
    },
    RemoveNode {
        uid: Uid,
    },
    AddLink {
        node_out: Uid,
        slot_out: String,
        node_in: Uid,
        slot_in: String,
    },
    RemoveLink {
        node_out: Uid,
        slot_out: String,
        node_in: Uid,
        slot_in: String,
    },
    /// Edit a node's mutable identity — its display `name` and/or its `pos`. A `None` field is left
    /// untouched; the inverse restores whichever fields were set.
    EditNode {
        uid: Uid,
        name: Option<String>,
        pos: Option<[f64; 2]>,
        /// The node's per-slot viewer state, WHOLE. The caller merges; this sets, so the inverse is
        /// the blob it replaced and a replay cannot half-apply.
        viewers: Option<serde_json::Value>,
    },
    /// Edit a param — its literal `value` and/or its source record. A `None` field is left
    /// untouched; the inverse restores whichever were set.
    EditParam {
        uid: Uid,
        group: String,
        name: String,
        value: Option<Param>,
        source: Option<SourceState>,
    },
    /// Add / edit / remove a global: `Some(value)` upserts, `None` removes. `at` is the ordered
    /// slot to re-add at — only a delete's captured inverse carries one, since order is observable.
    EditGlobal {
        name: String,
        value: Option<GlobalValue>,
        at: Option<usize>,
    },
    /// Move a tab to a position in the strip. Its CONTENT is a position, so it cannot ride
    /// [`Command::LayoutContents`]; it inverts as another reorder, aimed at where the tab is now.
    LayoutReorderTab {
        tab: crate::layout::Id,
        to_index: usize,
    },
    /// Set a split's children's shares. A GEOMETRY rather than a thing an entry holds, so it
    /// inverts as itself, against the shares the split carries at flip time.
    LayoutResizeSplit {
        split: crate::layout::Id,
        fractions: Vec<f64>,
    },
    /// A layout op that BIRTHS `born`. Its inverse is [`Command::LayoutClose`], planned at undo
    /// time: restoring the displaced slots would delete a wrapper a PEER has since built on.
    LayoutBirth {
        plan: crate::layout::Layout,
        born: crate::layout::Id,
    },
    /// The inverse of [`Command::LayoutBirth`]. Never a user op: a forward close must refuse
    /// teachably, where this must DEGRADE to a no-op when a peer has already closed it.
    LayoutClose {
        born: crate::layout::Id,
    },
    /// The inverse of [`Command::LayoutClose`]. It puts the closed subtree's own entries back and
    /// RE-PLANS where its root belongs, never restoring the slots the close's promote rewrote.
    LayoutRevive {
        dead: crate::layout::Dead,
        born: crate::layout::Id,
        /// Where `born` sat before the close. `None` for a tab, which is put back by strip index.
        home: Option<crate::layout::Home>,
    },
    /// A layout op that MOVES a subtree. Its inverse is RE-PLANNED like a birth's: another move,
    /// back to wherever `home` still lives.
    LayoutMove {
        /// The forward plan, when this is the user's own op; `None` on an inverse, which is planned
        /// from `home` against the arrangement as it stands at flip time.
        plan: Option<crate::layout::Layout>,
        root: crate::layout::Id,
        /// Where `root` sat before — captured by [`Command::execute`], so a forward carries `None`.
        home: Option<crate::layout::Home>,
    },
    /// A layout op that edits what entries HOLD, leaving where they sit alone. Its inverse reads
    /// each slot at flip time rather than restoring the whole entry.
    LayoutContents {
        writes: Vec<crate::layout::Write>,
    },
    /// Re-parent a node or scope into `scope` (`None` = ROOT). Used inside a delete's inverse to
    /// restore a member back INSIDE its scope.
    SetScope {
        uid: Uid,
        scope: Option<Uid>,
    },

    /// Group `members` into a new sub-patch scope at `pos`. `restore` is `None` for a user group
    /// and `Some` recreates an exact scope (the inverse of `Expand`). Returns the scope uid.
    Group {
        members: Vec<Uid>,
        pos: [f64; 2],
        restore: Option<ScopeRestore>,
    },
    /// Expand (dissolve) a scope back into its parent. Inverse = the `Group` that recreates it.
    Expand {
        scope: Uid,
    },
}

impl Command {
    /// What a FRESH caller must satisfy, checked in [`CommandHistory::apply`] ONLY, so `flip` keeps
    /// its tolerance. `Compound` is absent: its later children need a graph its earlier ones have
    /// not built yet.
    fn precondition(&self, g: &Graph) -> Result<(), String> {
        match self {
            Command::Expand { scope } => {
                g.is_facade(*scope).then_some(()).ok_or_else(|| format!("no sub-patch {}", scope.to_hex()))
            }
            // Never silently rooted on a scope that is not there: the canvas draws one scope, so a
            // node placed in another is invisible exactly where the caller put it.
            Command::AddNode { scope: Some(s), .. } => {
                g.is_facade(*s).then_some(()).ok_or_else(|| format!("node add: no such scope {s}"))
            }
            // A collapsed sub-patch facade is editable here (name/pos), so either kind counts.
            Command::EditNode { uid, .. } => {
                (g.contains(*uid) || g.is_facade(*uid) || g.stub(*uid).is_some())
                    .then_some(())
                    .ok_or_else(|| format!("no node, sub-patch or port {}", uid.to_hex()))
            }
            // Stricter than `EditNode`: a scope facade has no params to edit.
            Command::EditParam { uid, .. } => {
                g.contains(*uid).then_some(()).ok_or_else(|| format!("no node {}", uid.to_hex()))
            }
            // RemoveNode/RemoveLink stay tolerant ON PURPOSE: removing something already gone is
            // not a caller error. AddLink is validated at dispatch by `wirable_endpoint`.
            _ => Ok(()),
        }
    }

    /// Apply this command to `g`, returning its result and the exact inverse command.
    pub fn execute(self, g: &mut Graph) -> Result<(Outcome, Command), String> {
        match self {
            Command::Compound(cmds) => {
                let mut inverses = Vec::with_capacity(cmds.len());
                let mut last = Outcome::Ok;
                // Nodes ACCUMULATE where the other outcomes overwrite: they are a list of runtime
                // echoes owed, and a later child returning nothing does not cancel an earlier one.
                let mut echoes: Vec<Uid> = Vec::new();
                for c in cmds {
                    match c.execute(g) {
                        Ok((res, inv)) => {
                            match res {
                                Outcome::Nodes(ns) => echoes.extend(ns),
                                other => last = other,
                            }
                            inverses.push(inv);
                        }
                        // A Compound is a restoration UNIT, so abandoning one half-applied leaves
                        // a graph mutation no client is told about. Unwind what landed, newest first.
                        Err(e) => {
                            for inv in inverses.into_iter().rev() {
                                // Best-effort by necessity: stopping the unwind on an inverse that
                                // will not re-apply leaves strictly more wreckage than finishing.
                                let _ = inv.execute(g);
                            }
                            return Err(e);
                        }
                    }
                }
                inverses.reverse(); // undo the children back-to-front
                let out = if echoes.is_empty() { last } else { Outcome::Nodes(echoes) };
                Ok((out, Command::Compound(inverses)))
            }

            Command::AddNode { type_name, pos, uid, name, params, sources, viewers, scope } => {
                // A peer dissolved the scope this restore names. Tolerated HERE, because a replay
                // that errors wedges the actor's stack for good; the fresh caller is refused by
                // this command's precondition instead.
                if scope.is_some_and(|s| !g.is_facade(s)) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                // Idempotent: the uid is already present (a redo racing another client's add) —
                // reuse it, and re-place it only when a scope was ASKED for, since an
                // unconditional re-parent to ROOT would yank an already-placed node out.
                let u = match uid.filter(|u| g.exists(*u)) {
                    Some(u) => {
                        if let Some(s) = scope {
                            g.reparent(u, Some(s))?;
                        }
                        u
                    }
                    None => g.create_node(&type_name, uid, name.as_deref().unwrap_or(""), params, scope)?,
                };
                let _ = g.set_node_pos(u, pos);
                // Re-apply captured source records and viewer state; a user add carries none.
                for (group, name, s) in &sources {
                    let _ = g.set_source(u, group, name, s.clone());
                }
                if let Some(v) = viewers {
                    let _ = g.set_node_viewers(u, v);
                }
                Ok((Outcome::Uid(u), Command::RemoveNode { uid: u }))
            }

            Command::RemoveNode { uid } => {
                // Handles a plain leaf, a sub-patch member (leaf or nested scope), OR a top-level
                // instance — nothing live at this uid is the idempotent no-op.
                if !g.exists(uid) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                let (inverse, gone) = capture_subtree_restore(g, uid);
                // A panel bound to a uid this delete takes renders empty, so the binding goes with
                // the node — HERE, inside the one command, so it is one undo step.
                let unbind = g.arrangement().unbind(&gone);
                // By KIND, not by where it sits: a port is taken off its scope, a facade tears
                // down its subtree, and a leaf is a single-node removal, member or not.
                if let Some((scope, _)) = g.stub(uid) {
                    g.remove_stub(scope, uid);
                } else if g.is_facade(uid) {
                    g.remove_instance(uid)?;
                } else {
                    g.remove_node(uid)?;
                }
                if unbind.is_empty() {
                    return Ok((Outcome::Ok, inverse));
                }
                // Re-binding runs AFTER the nodes are back, which `Compound` replays in order.
                let (_, rebind) = Command::LayoutContents { writes: unbind }.execute(g)?;
                Ok((Outcome::Ok, Command::Compound(vec![inverse, rebind])))
            }

            Command::AddLink { node_out, slot_out, node_in, slot_in } => {
                // An endpoint is gone, so the wire cannot exist and restoring it is a no-op.
                // Without this a concurrent delete would error through `flip`, wedging the actor's stack.
                if !g.wirable(node_out) || !g.wirable(node_in) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                // Idempotent: the exact wire already exists, so its inverse must be one too — a
                // bare RemoveLink would DESTROY the pre-existing wire on undo.
                if g.has_link(node_out, &slot_out, node_in, &slot_in) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                // A single-input connect EVICTS a prior wire, so capture the displaced one for the
                // inverse. A multi input appends, and reconnecting the same wire displaces nothing.
                let displaced = g
                    .single_input_source(node_in, &slot_in)
                    .filter(|(o, s)| !(*o == node_out && *s == slot_out));
                g.add_link(node_out, &slot_out, node_in, &slot_in)?;
                let remove_new = Command::RemoveLink {
                    node_out,
                    slot_out: slot_out.clone(),
                    node_in,
                    slot_in: slot_in.clone(),
                };
                let inverse = match displaced {
                    Some((dout, dslot)) => Command::Compound(vec![
                        remove_new,
                        Command::AddLink { node_out: dout, slot_out: dslot.to_string(), node_in, slot_in },
                    ]),
                    None => remove_new,
                };
                Ok((Outcome::Ok, inverse))
            }

            Command::RemoveLink { node_out, slot_out, node_in, slot_in } => {
                // The wire is already gone. This guard is what lets two clients' undo of a
                // connect converge instead of wedging one of the stacks.
                if g.remove_link(node_out, &slot_out, node_in, &slot_in).is_err() {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                Ok((Outcome::Ok, Command::AddLink { node_out, slot_out, node_in, slot_in }))
            }

            Command::EditNode { uid, name, pos, viewers } => {
                // A node, a scope facade or a boundary port; only a vanished uid is the no-op.
                if !g.contains(uid) && !g.is_facade(uid) && g.stub(uid).is_none() {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: it is gone
                }
                let old_pos = pos.map(|_| g.pos(uid).unwrap_or([0.0, 0.0]));
                // A rename rewrites `nd('old')` → `nd('new')` in referring expressions; report the
                // touched referrers so the bridge re-broadcasts their runtime-enriched descriptors.
                let mut referrers = Vec::new();
                // Capture the pre-rename name only if the rename lands: a peer may have reclaimed
                // the target name since. A collision reaching here is always a stale replay.
                let inv_name = match &name {
                    None => None,
                    Some(n) => {
                        let old = g.name(uid).unwrap_or("").to_string();
                        match g.rename_node(uid, n) {
                            Ok(touched) => {
                                referrers = touched;
                                Some(old)
                            }
                            Err(_) => None, // collision → no-op; the inverse touches no name
                        }
                    }
                };
                if let Some(p) = pos {
                    g.set_node_pos(uid, p)?;
                }
                // A scope facade has no viewer state; a node and a port both do.
                let old_viewers = match &viewers {
                    Some(_) => g.viewers(uid).cloned(),
                    None => None,
                };
                if let Some(v) = viewers {
                    g.set_node_viewers(uid, v)?;
                }
                let out = if referrers.is_empty() { Outcome::Ok } else { Outcome::Nodes(referrers) };
                Ok((out, Command::EditNode { uid, name: inv_name, pos: old_pos, viewers: old_viewers }))
            }

            Command::EditParam { uid, group, name, value, source } => {
                if !g.contains(uid) {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: node gone
                }
                let old_value = match &value {
                    Some(_) => Some(
                        g.params(uid)
                            .and_then(|p| param(&p, &group, &name).cloned())
                            .ok_or_else(|| format!("edit_param: no param {group}.{name} on {}", uid.to_hex()))?,
                    ),
                    None => None,
                };
                // Captured when the caller names a source, and ALSO for a bare literal over a driven
                // param, since a literal write switches the mode the inverse owes back.
                let held = g.source_state_of(uid, &group, &name);
                let driven = held.as_ref().is_some_and(|s| s.mode != Mode::Constant);
                let old_source =
                    (source.is_some() || (value.is_some() && driven)).then(|| held.unwrap_or_default());
                // Literal FIRST, then source, and the order is load-bearing: a literal write switches
                // the mode to constant, so an `EditParam` carrying both would set and then undo it.
                if let Some(v) = value {
                    g.update_param(uid, &group, &name, v)?;
                }
                if let Some(s) = &source {
                    g.set_source(uid, &group, &name, s.clone())?;
                }
                Ok((Outcome::Ok, Command::EditParam { uid, group, name, value: old_value, source: old_source }))
            }

            Command::EditGlobal { name, value, at } => {
                let old = g.globals().get(&name).cloned();
                // A delete's inverse re-adds at the removed index; add/edit inverses carry no slot.
                let inv_at = if value.is_none() { g.globals().index_of(&name) } else { None };
                g.apply_global_change(&name, value, at)?;
                Ok((Outcome::Ok, Command::EditGlobal { name, value: old, at: inv_at }))
            }

            Command::LayoutReorderTab { tab, to_index } => {
                // Read BEFORE the move, so the inverse names where the tab is standing right now —
                // and degrade when a peer has closed it, as every other layout inverse does.
                let Some(from) = g.arrangement().tab_index(&tab) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                let Ok(writes) = g.arrangement().reorder_tab(&tab, to_index) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                g.arrangement_mut().apply(writes);
                Ok((Outcome::Ok, Command::LayoutReorderTab { tab, to_index: from }))
            }

            Command::LayoutBirth { plan, born } => {
                g.arrangement_mut().apply(plan);
                Ok((Outcome::Ok, Command::LayoutClose { born }))
            }

            Command::LayoutClose { born } => {
                // A tab is closed whole; anything else is closed with promote — the SAME planners
                // the forward ops call, so there is one algebra rather than two.
                let plan = match g.arrangement().tab_index(&born) {
                    Some(_) => g.arrangement().remove_tab(&born),
                    None => g.arrangement().remove_subtree(&born),
                };
                // The subtree itself and where its root sat — the two things its revive needs,
                // captured before anything moves. The slots the promote rewrote are NOT among them.
                let (Ok(next), Some(dead)) = (plan, g.arrangement().dead_subtree(&born)) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                let home = g.arrangement().home_of(&born);
                g.arrangement_mut().apply(next);
                Ok((Outcome::Ok, Command::LayoutRevive { dead, born, home }))
            }

            Command::LayoutRevive { dead, born, home } => {
                let Ok(next) = g.arrangement().revive(&dead, home.as_ref()) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                g.arrangement_mut().apply(next);
                Ok((Outcome::Ok, Command::LayoutClose { born }))
            }

            Command::LayoutResizeSplit { split, fractions } => {
                let Some(from) = g.arrangement().fractions(&split) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                let Ok(next) = g.arrangement().resize_split(&split, &fractions) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                g.arrangement_mut().apply(next);
                Ok((Outcome::Ok, Command::LayoutResizeSplit { split, fractions: from }))
            }

            Command::LayoutMove { plan, root, home } => {
                // Captured BEFORE anything moves, because the inverse is "put it back where it is
                // standing right now" — planned then, against the arrangement of that moment.
                let back = g.arrangement().home_of(&root);
                let plan = match (plan, &home) {
                    (Some(p), _) => Some(p),
                    (None, Some(h)) => g.arrangement().re_home(&root, h).ok(),
                    (None, None) => None,
                };
                // A stale replay — a peer closed or carried it off first. Degrade to a no-op like
                // `LayoutClose`: an `Err` inside `flip` wedges that actor's undo stack.
                let (Some(plan), Some(back)) = (plan, back) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                g.arrangement_mut().apply(plan);
                Ok((Outcome::Ok, Command::LayoutMove { plan: None, root, home: Some(back) }))
            }

            Command::LayoutContents { writes } => {
                // What those entries hold RIGHT NOW, landed the same way, so the pair is closed
                // under inversion. An id that has gone contributes nothing, in both directions.
                let back = writes
                    .iter()
                    .filter_map(|(id, _)| Some((id.clone(), g.arrangement().contents(id)?)))
                    .collect();
                g.arrangement_mut().set_contents(&writes);
                Ok((Outcome::Ok, Command::LayoutContents { writes: back }))
            }

            Command::SetScope { uid, scope } => {
                // Idempotent: the uid is gone (a redo racing a delete) → no-op.
                if !g.exists(uid) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                // The destination scope was dissolved. `SetScope` is never a user RPC, so this is
                // always a stale replay and the restored member simply lands at ROOT.
                if scope.is_some_and(|s| !g.is_facade(s)) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                let old = g.reparent(uid, scope)?;
                Ok((Outcome::Ok, Command::SetScope { uid, scope: old }))
            }

            Command::Group { members, pos, restore } => {
                // `minted` collects any stub `group_nodes` must add to a PRE-EXISTING nested member
                // — a side effect on a scope OUTSIDE the new one, which Expand alone would not undo.
                let mut minted: Vec<(Uid, Uid)> = Vec::new();
                let scope = match restore {
                    None => g.group_nodes_capturing(&members, pos, &mut minted)?,
                    // Idempotent: the exact scope is already live (a redo racing another client) —
                    // reuse it; otherwise recreate it uid-stable.
                    Some(r) if g.is_facade(r.scope_id) => r.scope_id,
                    Some(r) => g.restore_scope(r.scope_id, r.name, pos, &members, r.parent)?,
                };
                // Inverse: expand the new scope, then remove each minted port so group→undo is
                // exact (redo re-adds them before re-grouping, since Compound reverses child inverses).
                let inverse = if minted.is_empty() {
                    Command::Expand { scope }
                } else {
                    let mut cmds = vec![Command::Expand { scope }];
                    cmds.extend(minted.into_iter().map(|(_, id)| Command::RemoveNode { uid: id }));
                    Command::Compound(cmds)
                };
                Ok((Outcome::Uid(scope), inverse))
            }

            Command::Expand { scope } => {
                if !g.is_facade(scope) {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: already expanded/gone
                }
                // Capture the scope verbatim BEFORE dissolving, so the inverse re-groups it exactly.
                let name = g.name(scope).unwrap_or("").to_string();
                let spos = g.pos(scope).unwrap_or([0.0, 0.0]);
                // A facade wears viewers on its out ports like any node, and `restore_scope` builds
                // a bare record — so the blob rides back as the ordinary edit that sets one.
                let seen = g.viewers(scope).filter(|v| v.as_object().is_some_and(|m| !m.is_empty())).cloned();
                // Its ports come back as the NODES they are, and their cables as the links they
                // are — after the facade, which is what a port needs to be a port of.
                let ports: Vec<Command> = g
                    .ports_of(scope)
                    .into_iter()
                    .map(|id| Command::AddNode {
                        type_name: g.node_type(id).unwrap_or("").to_string(),
                        pos: g.pos(id).unwrap_or([0.0, 0.0]),
                        uid: Some(id),
                        name: g.name(id).map(str::to_string),
                        params: None,
                        sources: vec![],
                        viewers: g.viewers(id).cloned(),
                        scope: Some(scope),
                    })
                    .collect();
                let cables: Vec<Command> = g
                    .links_view()
                    .into_iter()
                    .filter(|l| g.stub(l.node_in).is_some_and(|(s, _)| s == scope)
                        || g.stub(l.node_out).is_some_and(|(s, _)| s == scope))
                    .map(|l| Command::AddLink {
                        node_out: l.node_out,
                        slot_out: l.slot_out.to_string(),
                        node_in: l.node_in,
                        slot_in: l.slot_in.to_string(),
                    })
                    .collect();
                let sparent = g.scope_of(scope); // the scope's parent, captured before it dissolves
                let members = g.scope_members(scope);
                let spliced = g.expand_instance(scope)?;
                // The wall's removal JOINED each crossing cable's two halves; putting the wall back
                // means taking those joins out, then restoring both halves against the ports.
                // Ordered so the REVERSE is legal too, since that reverse is the redo: the joins go
                // before the wall comes back, and come back after it goes again.
                let mut inverse: Vec<Command> = spliced
                    .into_iter()
                    .map(|(a, so, b, si)| Command::RemoveLink {
                        node_out: a,
                        slot_out: so.to_string(),
                        node_in: b,
                        slot_in: si.to_string(),
                    })
                    .collect();
                inverse.push(Command::Group {
                    members,
                    pos: spos,
                    restore: Some(ScopeRestore { scope_id: scope, name, parent: sparent }),
                });
                inverse.extend(seen.map(|v| Command::EditNode {
                    uid: scope,
                    name: None,
                    pos: None,
                    viewers: Some(v),
                }));
                inverse.extend(ports);
                inverse.extend(cables);
                Ok((Outcome::Ok, Command::Compound(inverse)))
            }


        }
    }
}

/// A per-ACTOR undo/redo history over one shared [`Graph`]. An entry holds ONE toggle, and
/// executing it returns the next — so an entry ping-pongs and stays uid-stable. Scoped by actor,
/// so one client's timeline is independent of another's.
#[derive(Default)]
pub struct CommandHistory {
    entries: Vec<HistoryEntry>,
}

struct HistoryEntry {
    /// The command that flips this entry's state: its inverse when applied, its forward when undone.
    toggle: Command,
    actor: String,
    undone: bool,
    /// The batch whose step made this entry — a compound settles by this stamp, never by a
    /// position another thread's removal can shift.
    batch: Option<u64>,
}

std::thread_local! {
    static OPEN_BATCH: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

static NEXT_BATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The batch stamp is thread-local because a compound's steps run the ordinary write arms on the
/// compound's own thread — so a peer's entry, even under the SAME actor, can never carry it.
pub struct BatchScope {
    id: u64,
}

pub fn open_batch() -> BatchScope {
    let id = NEXT_BATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    OPEN_BATCH.set(Some(id));
    BatchScope { id }
}

impl BatchScope {
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for BatchScope {
    fn drop(&mut self) {
        OPEN_BATCH.set(None);
    }
}

impl CommandHistory {
    pub fn new() -> CommandHistory {
        CommandHistory::default()
    }

    /// Execute `cmd` against `g`, record its inverse tagged with `actor`, and return the outcome.
    /// A new command clears THIS actor's redo run, never another actor's.
    pub fn apply(&mut self, g: &mut Graph, actor: &str, cmd: Command) -> Result<Outcome, String> {
        // The fresh-caller gate. `flip` deliberately does NOT call this — see `Command::precondition`.
        cmd.precondition(g)?;
        let (outcome, inverse) = cmd.execute(g)?;
        // Record EVERY successful command, a forward no-op included: the client records one entry
        // per mutating RPC, so skipping one here desyncs the stacks and a later undo flips wrong.
        self.entries.retain(|e| !(e.actor == actor && e.undone));
        self.entries.push(HistoryEntry {
            toggle: inverse,
            actor: actor.to_string(),
            undone: false,
            batch: OPEN_BATCH.get(),
        });
        Ok(outcome)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop this actor's redo run, as an `apply` would. A compound clears it once up front, so
    /// no step of its own can move the mark under it.
    pub fn clear_redo(&mut self, actor: &str) {
        self.entries.retain(|e| !(e.actor == actor && e.undone));
    }

    /// Drop an actor's WHOLE stack — a stack's lifetime follows its actor, so a stopped agent's
    /// history goes with it. The graph keeps every change; only the way back is gone.
    pub fn drop_actor(&mut self, actor: &str) {
        self.entries.retain(|e| e.actor != actor);
    }

    /// Fold everything `batch`'s steps added into ONE entry, so a compound RPC is a single undo
    /// step. A peer's entry that landed in between is left exactly where it is.
    pub fn coalesce(&mut self, actor: &str, batch: u64) {
        let mine: Vec<usize> =
            (0..self.entries.len()).filter(|&i| self.entries[i].batch == Some(batch)).collect();
        if mine.len() < 2 {
            return;
        }
        // Newest first: each toggle is an inverse, and a Compound applies its children in order.
        let toggle =
            Command::Compound(mine.iter().rev().map(|&i| self.entries.remove(i).toggle).collect());
        self.entries.push(HistoryEntry {
            toggle,
            actor: actor.to_string(),
            undone: false,
            batch: None,
        });
    }

    /// Undo and DISCARD everything `batch`'s steps added — what a compound does when a later
    /// step is refused, so a failed call leaves no redo run either.
    pub fn rollback(&mut self, g: &mut Graph, batch: u64) {
        let mine: Vec<usize> =
            (0..self.entries.len()).filter(|&i| self.entries[i].batch == Some(batch)).collect();
        for &i in mine.iter().rev() {
            // Best-effort by necessity, exactly as `Compound`'s own unwind is.
            let _ = self.entries.remove(i).toggle.execute(g);
        }
    }

    /// Drop the entire history (every actor's entries). Loading a patch fully resets the
    /// session — there is nothing to undo across a load — so the manager clears here.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Undo the actor's most-recent applied command. `Ok(false)` if it has nothing to undo.
    pub fn undo(&mut self, g: &mut Graph, actor: &str) -> Result<bool, String> {
        let Some(idx) = self.entries.iter().rposition(|e| e.actor == actor && !e.undone) else {
            return Ok(false);
        };
        self.flip(g, idx, true)
    }

    /// Redo the actor's most-recently-undone command. `Ok(false)` if it has nothing to redo.
    pub fn redo(&mut self, g: &mut Graph, actor: &str) -> Result<bool, String> {
        let Some(idx) = self.entries.iter().position(|e| e.actor == actor && e.undone) else {
            return Ok(false);
        };
        self.flip(g, idx, false)
    }

    fn flip(&mut self, g: &mut Graph, idx: usize, undone: bool) -> Result<bool, String> {
        let (_out, next) = self.entries[idx].toggle.clone().execute(g)?;
        self.entries[idx].toggle = next;
        self.entries[idx].undone = undone;
        Ok(true)
    }

    pub fn can_undo(&self, actor: &str) -> bool {
        self.entries.iter().any(|e| e.actor == actor && !e.undone)
    }

    pub fn can_redo(&self, actor: &str) -> bool {
        self.entries.iter().any(|e| e.actor == actor && e.undone)
    }
}

/// Capture the exact inverse to restore the subtree rooted at `root`, BEFORE the caller removes it.
/// The Compound recreates every node, scope, membership, pruned stub and touching link, uid-stable.
fn capture_subtree_restore(g: &Graph, root: Uid) -> (Command, std::collections::HashSet<Uid>) {
    // Where the restored top returns to: `None` = ROOT (a top-level instance / leaf).
    let orig_parent = g.scope_of(root);

    // Discovery order, so a facade always precedes the members that name it. Ports are held apart
    // only because a port is a port OF a scope: it takes its scope at birth, so it is created after
    // every facade rather than before.
    let mut members: Vec<Uid> = Vec::new();
    let mut ports: Vec<Uid> = Vec::new();
    let mut stack = vec![root];
    while let Some(u) = stack.pop() {
        if g.stub(u).is_some() {
            ports.push(u);
            continue;
        }
        members.push(u);
        if g.is_facade(u) {
            stack.extend(g.scope_members(u));
        }
    }

    let mut cmds: Vec<Command> = Vec::new();

    // ONE loop: every member of any kind is recreated at ROOT, uid-stable, with the full persisted
    // state its kind carries — a leaf's params and expression bindings, and everyone's viewers.
    // Membership is restored by the `SetScope` children below, not here — see the field's doc.
    for &u in members.iter().chain(&ports) {
        let sources = g.param_sources(u);
        cmds.push(Command::AddNode {
            type_name: g.node_type(u).unwrap_or("").to_string(),
            pos: g.pos(u).unwrap_or([0.0, 0.0]),
            uid: Some(u),
            name: g.name(u).map(str::to_string),
            params: g.params(u).map(|p| (*p).clone()),
            sources,
            viewers: g.viewers(u).filter(|v| v.as_object().is_some_and(|m| !m.is_empty())).cloned(),
            scope: g.stub(u).map(|(s, _)| s),
        });
    }

    // Membership, once every uid exists. A PORT is not here: its scope rode its `AddNode`, because
    // a port cannot be created without one, and one owner is the rule. The root's own is last, so a
    // member delete puts the top back INSIDE its enclosing scope; a top-level one lands at ROOT.
    for &u in &members {
        if u == root {
            continue;
        }
        cmds.push(Command::SetScope { uid: u, scope: g.scope_of(u) });
    }
    if orig_parent.is_some() {
        cmds.push(Command::SetScope { uid: root, scope: orig_parent });
    }

    // Every link touching the subtree, after all endpoints exist — an enclosing port's wire
    // included, since its other end is in here. The set is handed back too: it is what a panel
    // bound to one of these uids has to stop naming, so it holds ONLY what is going.
    let subtree: std::collections::HashSet<Uid> = members.iter().chain(&ports).copied().collect();
    for l in g.links_view() {
        if subtree.contains(&l.node_out) || subtree.contains(&l.node_in) {
            cmds.push(Command::AddLink {
                node_out: l.node_out,
                slot_out: l.slot_out.to_string(),
                node_in: l.node_in,
                slot_in: l.slot_in.to_string(),
            });
        }
    }

    (Command::Compound(cmds), subtree)
}
