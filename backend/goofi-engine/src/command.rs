//! Patch commands with exact inverses — the manager's undo/redo unit.
//!
//! [`Command::execute`] mutates the [`Graph`] and returns `(outcome, inverse)`, and the inverse
//! itself returns the forward again — so redo is executing what undo returned, and an entry
//! ping-pongs rather than being rebuilt.
//!
//! The surface is deliberately minimal: add/remove node, add/remove link, and one `Edit*` op per
//! target covering every field of it. Loading a patch is NOT a command — a load resets the
//! session, so there is nothing to undo across it.
//!
//! **Tolerance belongs to replay.** An edit or remove against an already-absent target is a benign
//! no-op, so two clients undoing the same change converge. What a FRESH caller must satisfy is
//! [`Command::precondition`], checked in `apply` and never in `flip`.

use crate::subpatch::{self, Dir, Stub, StubId};
use crate::{Graph, Uid};
use goofi_core::globals::GlobalValue;
use goofi_core::Param;
use goofi_node::{param, ParamGroups};
use indexmap::IndexMap;

/// What a command produced, for the caller (the RPC reply). Kept serde-free so the engine needs no
/// JSON dep — the bridge maps it to the wire.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// A plain success (`{ ok: true }` on the wire).
    Ok,
    /// A minted/affected uid — `add_node`/`group` return the node/scope uid.
    Uid(Uid),
    /// A minted/restored stub id — `add_stub` returns the stub's id (`in0`/`out0`).
    StubId(StubId),
    /// Nodes the command touched that need a runtime echo — `EditNode`'s rename returns the
    /// referrers whose `nd()` expressions were rewritten, so the bridge re-broadcasts their params.
    Nodes(Vec<Uid>),
}

/// A param's expression binding, as carried by [`Command::EditParam`]. An empty `source` clears the
/// binding (unbinds back to the literal); a non-empty one (re)binds.
#[derive(Clone, Debug, PartialEq)]
pub struct ExprState {
    pub source: String,
    pub enabled: bool,
    pub triggers: bool,
}

/// The captured state to recreate a scope EXACTLY — carried by [`Command::Group`]'s restore form
/// (the inverse of [`Command::Expand`]). Redo-of-group / undo-of-expand restores the exact scope id
/// + stubs (uid-stable), never a freshly-minted one.
#[derive(Clone, Debug, PartialEq)]
pub struct ScopeRestore {
    pub scope_id: Uid,
    pub name: String,
    pub stubs: IndexMap<StubId, Stub>,
    /// The scope's parent, captured explicitly (not derived from members) so an EMPTY scope — a
    /// sub-patch whose members were all deleted — restores at the right place. `None` = ROOT.
    pub parent: Option<Uid>,
    /// Parent-scope stubs `Expand` re-pointed away from this scope (`(parent, stub_id, old_inner)`),
    /// so the Group inverse re-points them back exactly. Empty for a delete-undo (which prunes, not
    /// re-points) — see [`Graph::parent_stubs_referencing`].
    pub parent_stubs: Vec<subpatch::ParentStub>,
}

/// One semantic patch edit. Every variant has an exact inverse (see [`Command::execute`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Executed in order; its inverse is the children's inverses in REVERSE order. Also how a
    /// multi-step edit (e.g. a global rename = add-new + remove-old) is expressed as one undo step.
    Compound(Vec<Command>),
    AddNode {
        type_name: String,
        pos: [f64; 2],
        /// `Some` restores a specific uid (undo/redo, so links + panels reconnect); `None` mints one.
        uid: Option<Uid>,
        name: Option<String>,
        /// `Some` restores captured params (a `RemoveNode` inverse); `None` uses the type's defaults.
        params: Option<ParamGroups>,
        /// Captured expression bindings `(group, name, binding)` to re-apply — restores a node's
        /// live-driven params on a `RemoveNode` inverse. Empty for a user add.
        exprs: Vec<(String, String, ExprState)>,
        /// Captured viewer view-state blob to restore; `None` for a user add (defaults to empty).
        viewers: Option<serde_json::Value>,
        /// The scope to create the node INSIDE (`None` = ROOT). NOT used by a `RemoveNode`
        /// inverse, which restores membership with its own [`Command::SetScope`] child — so a
        /// restore leaves this `None` and nothing re-parents a node already placed.
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
    },
    /// Edit a param — its literal `value` and/or its expression binding. A `None` field is left
    /// untouched; the inverse restores whichever were set.
    EditParam {
        uid: Uid,
        group: String,
        name: String,
        value: Option<Param>,
        expr: Option<ExprState>,
    },
    /// Add / edit / remove a global: `Some(value)` upserts, `None` removes. A rename is two of these
    /// (add-new then remove-old), composed into one undo step via [`Command::Compound`]. `at` is the
    /// ordered slot to re-add at — `None` for every user-issued mutation (an add appends); `Some(i)`
    /// only on a delete's captured inverse, so undo restores the global to its original position
    /// (order is observable via `.gfi`/mirror/panel) rather than the tail.
    EditGlobal {
        name: String,
        value: Option<GlobalValue>,
        at: Option<usize>,
    },
    /// Write one flat-arrangement entry: `Some` upserts it, `None` removes it. A layout op plans its
    /// change as a set of these and composes them with [`Command::Compound`], which is why the whole
    /// tree algebra needs no command of its own and undo/redo of a split is undo/redo of a map write.
    EditLayoutEntry {
        id: crate::layout::Id,
        entry: Option<crate::layout::Entry>,
    },
    /// A layout op that BIRTHS `born`. Its inverse is NOT the slots the writes displaced: it is
    /// [`Command::LayoutClose`], planned at undo time. Restoring the slots would delete a wrapper a
    /// PEER has since hung a panel off — a lost update no graph command can make. Close-with-promote
    /// already knows how to hand a split's survivors to its parent, so borrowing it makes a foreign
    /// undo non-destructive by construction rather than by a guard.
    LayoutBirth {
        writes: Vec<crate::layout::Write>,
        born: crate::layout::Id,
    },
    /// The inverse of [`Command::LayoutBirth`]. Never a user op: a forward close must refuse
    /// teachably, where this must DEGRADE to a no-op when a peer has already closed it.
    LayoutClose {
        born: crate::layout::Id,
    },
    /// The inverse of [`Command::LayoutClose`]. It puts the closed subtree's own entries back —
    /// dead ids, referenced by nothing — and then RE-PLANS where its root belongs. Restoring the
    /// slots the close's promote rewrote is precisely what it exists not to do.
    LayoutRevive {
        dead: Vec<(crate::layout::Id, crate::layout::Entry)>,
        born: crate::layout::Id,
        /// Where `born` sat before the close. `None` for a tab, which is put back by strip index.
        home: Option<crate::layout::Home>,
    },
    /// A layout op that MOVES a subtree. Its inverse is RE-PLANNED like a birth's, not restored:
    /// another move, back to wherever `home` still lives. Restoring the slots a move displaced puts
    /// back the split the move promoted away — on top of whatever a peer has since built in its
    /// place, which leaves that tab two roots and the peer's panel in the one nothing renders.
    LayoutMove {
        /// The forward plan, when this is the user's own op; `None` on an inverse, which is planned
        /// from `home` against the arrangement as it stands at flip time.
        writes: Option<Vec<crate::layout::Write>>,
        root: crate::layout::Id,
        /// Where `root` sat before — captured by [`Command::execute`], so a forward carries `None`.
        home: Option<crate::layout::Home>,
    },
    /// A layout op that edits what entries HOLD, leaving where they sit alone. Its inverse lands
    /// the same way, reading each slot at flip time — restoring the whole entry instead puts back
    /// the `order` a peer's adjacent split has since taken.
    LayoutContents {
        writes: Vec<crate::layout::Write>,
    },
    /// Re-parent a node or scope into `scope` (`None` = ROOT). The one membership move — used inside
    /// a delete's inverse to restore a member back INSIDE its scope. Inverse re-parents to the old
    /// scope.
    SetScope {
        uid: Uid,
        scope: Option<Uid>,
    },

    // ── Structural sub-patch commands (uid-stable on the flat scope model) ─────────
    /// Group `members` into a new sub-patch scope at `pos`. `restore` is `None` for a user group
    /// (mints the scope + derives crossing stubs); `Some` recreates an exact scope (the inverse of
    /// `Expand`). Returns the scope uid.
    Group {
        members: Vec<Uid>,
        pos: [f64; 2],
        restore: Option<ScopeRestore>,
    },
    /// Expand (dissolve) a scope back into its parent. Inverse = the `Group` that recreates it.
    Expand {
        scope: Uid,
    },
    /// Add a boundary stub to a scope. `restore` is `None` for a user add (mints an unwired stub);
    /// `Some((id, stub))` recreates an exact captured stub (the inverse of `RemoveStub`).
    AddStub {
        scope: Uid,
        dir: Dir,
        dtype: goofi_core::SlotType,
        pos: [f64; 2],
        restore: Option<(StubId, Stub)>,
    },
    /// Remove a stub. Inverse = `AddStub` restoring the captured stub.
    RemoveStub {
        scope: Uid,
        stub_id: StubId,
    },
    /// Wire (`Some`) or unwire (`None`) a stub's inner target. `dtype` is `None` on a user wire (the
    /// dtype is resolved from the wired slot); the inverse carries the old dtype so unwire restores
    /// the exact pre-wire advertised type. Inverse restores the old inner + dtype.
    WireStub {
        scope: Uid,
        stub_id: StubId,
        inner: subpatch::StubInner,
        dtype: Option<goofi_core::SlotType>,
    },
    /// Edit a stub's display name and/or pill pos. A `None` field is left untouched; the inverse
    /// restores whichever were set.
    EditStub {
        scope: Uid,
        stub_id: StubId,
        name: Option<String>,
        pos: Option<[f64; 2]>,
    },
}

impl Command {
    /// What a FRESH caller must satisfy, checked in [`CommandHistory::apply`] ONLY — so `flip`
    /// keeps its tolerance and convergence is unchanged by construction. A first-hand RPC earns
    /// none of replay's benefit of the doubt: an `{ok: true}` for a target that is not there
    /// asserts a state the patch is not in, and every later decision is taken against it.
    ///
    /// `Compound` is deliberately absent: its later children are validated against a graph its
    /// earlier children have not built yet, so checking the PRE state would refuse legal restores.
    fn precondition(&self, g: &Graph) -> Result<(), String> {
        let stub = |scope: Uid, id: &str| -> Result<(), String> {
            let s = g.scope(scope).ok_or_else(|| format!("no sub-patch {}", scope.to_hex()))?;
            s.stubs
                .contains_key(id)
                .then_some(())
                .ok_or_else(|| format!("sub-patch {} has no boundary `{id}`", scope.to_hex()))
        };
        match self {
            Command::WireStub { scope, stub_id, inner, .. } => {
                stub(*scope, stub_id)?;
                match inner {
                    Some(target) => g.stub_wire_dtype(*scope, stub_id, target).map(|_| ()),
                    None => Ok(()), // an unwire always applies once the stub is known to exist
                }
            }
            Command::RemoveStub { scope, stub_id } | Command::EditStub { scope, stub_id, .. } => {
                stub(*scope, stub_id)
            }
            Command::Expand { scope } => {
                g.scope(*scope).map(|_| ()).ok_or_else(|| format!("no sub-patch {}", scope.to_hex()))
            }
            // A collapsed sub-patch facade is editable here (name/pos), so either kind counts.
            Command::EditNode { uid, .. } => (g.contains(*uid) || g.scope(*uid).is_some())
                .then_some(())
                .ok_or_else(|| format!("no node or sub-patch {}", uid.to_hex())),
            // Stricter than `EditNode`: a scope facade has no params to edit.
            Command::EditParam { uid, .. } => {
                g.contains(*uid).then_some(()).ok_or_else(|| format!("no node {}", uid.to_hex()))
            }
            // RemoveNode/RemoveLink stay tolerant ON PURPOSE: removing something already gone is
            // not a caller error, and both report `{removed: false}` so the caller is told the
            // truth without one. AddLink is validated at the dispatch boundary by
            // `wirable_endpoint`, which says far more than a generic existence check could.
            _ => Ok(()),
        }
    }

    /// Apply this command to `g`, returning its result and the exact inverse command.
    pub fn execute(self, g: &mut Graph) -> Result<(Outcome, Command), String> {
        match self {
            Command::Compound(cmds) => {
                let mut inverses = Vec::with_capacity(cmds.len());
                let mut last = Outcome::Ok;
                for c in cmds {
                    match c.execute(g) {
                        Ok((res, inv)) => {
                            last = res;
                            inverses.push(inv);
                        }
                        // A Compound is a restoration UNIT, and the bridge gates its re-mirror on
                        // `is_ok()` — so abandoning one half-applied leaves a graph mutation no
                        // client is told about. Unwind what landed, newest first, with the exact
                        // inverses those children just handed back.
                        Err(e) => {
                            for inv in inverses.into_iter().rev() {
                                // Best-effort by necessity: each inverse was minted moments ago
                                // against this same graph, and stopping the unwind on one that
                                // will not re-apply leaves strictly more wreckage than finishing.
                                let _ = inv.execute(g);
                            }
                            return Err(e);
                        }
                    }
                }
                inverses.reverse(); // undo the children back-to-front
                Ok((last, Command::Compound(inverses)))
            }

            Command::AddNode { type_name, pos, uid, name, params, exprs, viewers, scope } => {
                // Validate the destination BEFORE mutating anything: an add that cannot honour its
                // scope must leave the graph exactly as it found it, else the caller is told the
                // command failed while a stray node stays behind and the CRDT mirror disagrees.
                if let Some(s) = scope {
                    if g.scope(s).is_none() {
                        return Err(format!("add_node: no such scope {s}"));
                    }
                }
                let u = match uid {
                    // Idempotent: the uid is already present (a redo racing another client's add) — reuse it.
                    Some(u) if g.contains(u) => u,
                    Some(u) => g.add_node_at(&type_name, params, u, name.as_deref().unwrap_or(""))?,
                    None => {
                        let u = g.add_node(&type_name, None)?;
                        if let Some(n) = &name {
                            let _ = g.rename_node(u, n);
                        }
                        u
                    }
                };
                // Only when a scope was ASKED for: the idempotent branch above can hand back a
                // node already placed, so an unconditional re-parent to ROOT would yank a live
                // member out of its sub-patch.
                if let Some(s) = scope {
                    g.reparent(u, Some(s))?;
                }
                let _ = g.set_node_pos(u, pos);
                // Re-apply captured expression bindings + viewer state (a RemoveNode inverse restores
                // them; a user add carries none). Bindings are separate node state from param values.
                for (group, name, e) in &exprs {
                    let _ = g.set_expression(u, group, name, &e.source, e.enabled, e.triggers);
                }
                if let Some(v) = viewers {
                    let _ = g.set_node_viewers(u, v);
                }
                Ok((Outcome::Uid(u), Command::RemoveNode { uid: u }))
            }

            Command::RemoveNode { uid } => {
                // Handles a plain leaf, a sub-patch member (leaf or nested scope), OR a top-level
                // instance — nothing live at this uid is the idempotent no-op.
                if !g.contains(uid) && g.scope(uid).is_none() {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                let (inverse, gone) = capture_subtree_restore(g, uid);
                // A panel bound to a uid this delete takes renders empty and explains nothing, so
                // the binding goes with the node — HERE, inside the one command, so it is one undo
                // step and a peer's replica never holds a panel naming a node that is not there.
                let unbind = g.arrangement().unbind(&gone);
                // A scope MEMBER routes through remove_member (prunes the enclosing scope's stubs); a
                // top-level scope tears down its subtree; a plain leaf is a single-node removal.
                if g.scope_of(uid).is_some() {
                    g.remove_member(uid)?;
                } else if g.scope(uid).is_some() {
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
                // Without this, a concurrent delete would error through `flip` — wedging the
                // session AND leaving the Compound's earlier child applied but unbroadcast.
                if !g.contains(node_out) || !g.contains(node_in) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                // Idempotent: the exact wire already exists → the forward `add_link` is a silent
                // no-op, so its inverse must be one too. A bare RemoveLink would DESTROY the
                // pre-existing wire on undo (the inverse of a no-op is not a mutation).
                if g.has_link(node_out, &slot_out, node_in, &slot_in) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                // A single-input connect EVICTS a prior wire on that input — capture the displaced
                // wire so the inverse RESTORES it (else undo-of-reconnect leaves the input empty). A
                // multi input appends (nothing displaced); reconnecting the same wire displaces nothing.
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

            Command::EditNode { uid, name, pos } => {
                // A node OR a scope facade (collapsed instance) is editable here; only a truly
                // vanished uid is the idempotent no-op (a redo racing a delete).
                if !g.contains(uid) && g.scope(uid).is_none() {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: node/scope gone
                }
                let old_pos = pos.map(|_| g.pos(uid).unwrap_or([0.0, 0.0]));
                // A rename rewrites `nd('old')` → `nd('new')` in referring expressions; report the
                // touched referrers so the bridge re-broadcasts their runtime-enriched descriptors.
                let mut referrers = Vec::new();
                // Capture the pre-rename name only if the rename lands: a peer may have reclaimed
                // the target name since this toggle was recorded. A forward rename is pre-validated
                // at the bridge, so a collision reaching here is always a stale replay.
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
                let out = if referrers.is_empty() { Outcome::Ok } else { Outcome::Nodes(referrers) };
                Ok((out, Command::EditNode { uid, name: inv_name, pos: old_pos }))
            }

            Command::EditParam { uid, group, name, value, expr } => {
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
                // Captured when the caller names an `expr` — and ALSO when it names only a literal
                // over a param that is bound, because §3.4 makes a literal write an unbind. The
                // binding is part of what this edit destroys, so it is part of what the inverse
                // owes; without this, undo hands back the number and the expression stays gone.
                let bound = g.param_expression(uid, &group, &name);
                let old_expr = (expr.is_some() || (value.is_some() && bound.is_some())).then(|| {
                    bound
                        .map(|e| ExprState { source: e.source, enabled: e.enabled, triggers: e.triggers_process })
                        .unwrap_or(ExprState { source: String::new(), enabled: false, triggers: false })
                });
                // Literal FIRST, then binding, and the order is load-bearing: §3.4 makes a
                // literal write an UNBIND, so an `EditParam` carrying both would otherwise bind and
                // then immediately undo it.
                if let Some(v) = value {
                    g.update_param(uid, &group, &name, v)?;
                }
                if let Some(e) = &expr {
                    g.set_expression(uid, &group, &name, &e.source, e.enabled, e.triggers)?;
                }
                Ok((Outcome::Ok, Command::EditParam { uid, group, name, value: old_value, expr: old_expr }))
            }

            Command::EditGlobal { name, value, at } => {
                let old = g.globals().get(&name).cloned();
                let old_index = g.globals().index_of(&name);
                let was_delete = value.is_none();
                match (&value, at) {
                    // Re-add at a captured slot (the inverse of a delete/rename) — preserve order.
                    (Some(v), Some(i)) if !g.globals().contains(&name) => {
                        g.insert_global_at(&name, v.clone(), i)?;
                    }
                    _ => g.apply_global_change(&name, value)?,
                }
                // A delete's inverse re-adds at the removed index; add/edit inverses carry no slot.
                let inv_at = if was_delete { old_index } else { None };
                Ok((Outcome::Ok, Command::EditGlobal { name, value: old, at: inv_at }))
            }

            Command::EditLayoutEntry { id, entry } => {
                // A map slot swap, so the inverse is simply what was there — including `None`, which
                // is how the inverse of an add is a remove. Nothing here can fail, which is what lets
                // a planner validate the whole op up front and hand over a Compound that lands whole.
                let old = match entry {
                    Some(e) => g.arrangement_mut().insert(id.clone(), e),
                    None => g.arrangement_mut().remove(&id),
                };
                Ok((Outcome::Ok, Command::EditLayoutEntry { id, entry: old }))
            }

            Command::LayoutBirth { writes, born } => {
                g.arrangement_mut().apply(writes);
                Ok((Outcome::Ok, Command::LayoutClose { born }))
            }

            Command::LayoutClose { born } => {
                // A tab is closed whole (its panels are its own); anything else is closed with
                // promote — the SAME planners the forward ops call, so there is one algebra, not a
                // second subtly-different removal living in the inverse.
                let plan = match g.arrangement().get(&born) {
                    Some(crate::layout::Entry::Tab { .. }) => g.arrangement().remove_tab(&born),
                    _ => g.arrangement().remove_subtree(&born),
                };
                let Ok(writes) = plan else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                // The subtree's own entries and where its root sat — the two things its revive needs,
                // captured before anything moves. The slots the promote rewrote are NOT among them.
                let dead = g.arrangement().dead_subtree(&born);
                let home = g.arrangement().home_of(&born);
                g.arrangement_mut().apply(writes);
                Ok((Outcome::Ok, Command::LayoutRevive { dead, born, home }))
            }

            Command::LayoutRevive { dead, born, home } => {
                let Ok(writes) = g.arrangement().revive(&dead, &born, home.as_ref()) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                g.arrangement_mut().apply(writes);
                Ok((Outcome::Ok, Command::LayoutClose { born }))
            }

            Command::LayoutMove { writes, root, home } => {
                // Captured BEFORE anything moves, because the inverse is "put it back where it is
                // standing right now" — planned then, against the arrangement of that moment.
                let back = g.arrangement().home_of(&root);
                let plan = match (writes, &home) {
                    (Some(w), _) => Some(w),
                    (None, Some(h)) => g.arrangement().re_home(&root, h).ok(),
                    (None, None) => None,
                };
                // A stale replay — a peer closed or carried it off first. Degrade to a no-op like
                // `LayoutClose`: an `Err` inside `flip` wedges that session's undo stack.
                let (Some(plan), Some(back)) = (plan, back) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                };
                g.arrangement_mut().apply(plan);
                Ok((Outcome::Ok, Command::LayoutMove { writes: None, root, home: Some(back) }))
            }

            Command::LayoutContents { writes } => {
                let plan = g.arrangement().set_contents(&writes);
                // What those slots hold RIGHT NOW, which is what the inverse lands — and it lands it
                // the same way, so the pair is closed under inversion and a redo re-plans too.
                let back = plan
                    .iter()
                    .map(|(id, _)| (id.clone(), g.arrangement().get(id).cloned()))
                    .collect();
                g.arrangement_mut().apply(plan);
                Ok((Outcome::Ok, Command::LayoutContents { writes: back }))
            }

            Command::SetScope { uid, scope } => {
                // Idempotent: the uid is gone (a redo racing a delete) → no-op.
                if !g.contains(uid) && g.scope(uid).is_none() {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                // The destination scope was dissolved. `SetScope` is never a user RPC — only the
                // membership-restoring child of a `RemoveNode` inverse — so this is always a stale
                // replay, and the restored member simply lands at ROOT.
                if scope.is_some_and(|s| g.scope(s).is_none()) {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                let old = g.reparent(uid, scope)?;
                Ok((Outcome::Ok, Command::SetScope { uid, scope: old }))
            }

            Command::Group { members, pos, restore } => {
                // `minted` collects any stub group_nodes must add to a PRE-EXISTING nested member to
                // re-expose an orphaned crossing link — a side effect on a scope OUTSIDE the new one,
                // which Expand alone would not undo. Only a fresh group (restore=None) can mint.
                let mut minted: Vec<(Uid, StubId)> = Vec::new();
                let scope = match restore {
                    None => g.group_nodes_capturing(&members, pos, &mut minted)?,
                    Some(r) => {
                        // Idempotent: the exact scope is already live (a redo racing another client) —
                        // reuse it; otherwise recreate it uid-stable.
                        let scope = if g.scope(r.scope_id).is_some() {
                            r.scope_id
                        } else {
                            g.restore_scope(r.scope_id, r.name, pos, &members, r.stubs, r.parent)?
                        };
                        // The exact reversal of `expand_instance`, written onto the stub with NO
                        // validation: this restores a known-good captured state, which may
                        // legitimately name a nested scope.
                        for (p, sid, inner) in r.parent_stubs {
                            if let Some(st) = g.stub_mut(p, &sid) {
                                st.inner = inner;
                            }
                        }
                        scope
                    }
                };
                // Inverse: expand the new scope, then RemoveStub each minted port so group→undo is
                // exact (redo re-adds them before re-grouping, since Compound reverses child inverses).
                let inverse = if minted.is_empty() {
                    Command::Expand { scope }
                } else {
                    let mut cmds = vec![Command::Expand { scope }];
                    cmds.extend(
                        minted.into_iter().map(|(mscope, id)| Command::RemoveStub { scope: mscope, stub_id: id }),
                    );
                    Command::Compound(cmds)
                };
                Ok((Outcome::Uid(scope), inverse))
            }

            Command::Expand { scope } => {
                let Some(s) = g.scope(scope) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: already expanded/gone
                };
                // Capture the scope verbatim BEFORE dissolving, so the inverse re-groups it exactly.
                let name = s.name.clone();
                let spos = s.pos;
                let stubs = s.stubs.clone();
                let sparent = g.scope_of(scope); // the scope's parent, captured before it dissolves
                // Parent stubs expand_instance is about to re-point — captured BEFORE, so the Group
                // inverse re-points them back exactly.
                let parent_stubs = g.parent_stubs_referencing(scope);
                let members = g.scope_members(scope);
                g.expand_instance(scope)?;
                Ok((
                    Outcome::Ok,
                    Command::Group {
                        members,
                        pos: spos,
                        restore: Some(ScopeRestore { scope_id: scope, name, stubs, parent: sparent, parent_stubs }),
                    },
                ))
            }

            Command::AddStub { scope, dir, dtype, pos, restore } => {
                let id = match restore {
                    None => g.add_boundary(scope, dir, dtype, pos)?,
                    // Idempotent: the scope was dissolved (a concurrent expand) → the restore is moot,
                    // like every sibling structural inverse. (A user add to a missing scope still errors
                    // via `add_boundary` above — that is a caller mistake, not a redo race.)
                    Some(_) if g.scope(scope).is_none() => {
                        return Ok((Outcome::Ok, Command::Compound(vec![])));
                    }
                    Some((id, stub)) => {
                        if let Some(stubs) = g.stubs_mut(scope) {
                            stubs.insert(id.clone(), stub);
                        }
                        id
                    }
                };
                Ok((Outcome::StubId(id.clone()), Command::RemoveStub { scope, stub_id: id }))
            }

            Command::RemoveStub { scope, stub_id } => {
                let Some(stub) = g.scope(scope).and_then(|s| s.stubs.get(&stub_id).cloned()) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: already gone
                };
                // External flat links stay valid leaf->leaf links — they never referenced the
                // stub at runtime — so they are left in place.
                if let Some(stubs) = g.stubs_mut(scope) {
                    stubs.shift_remove(&stub_id);
                }
                let (dir, dtype, pos) = (stub.dir, stub.dtype, stub.pos);
                Ok((Outcome::Ok, Command::AddStub { scope, dir, dtype, pos, restore: Some((stub_id, stub)) }))
            }

            Command::WireStub { scope, stub_id, inner, dtype } => {
                let Some(st) = g.scope(scope).and_then(|s| s.stubs.get(&stub_id)) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: stub gone
                };
                // Capture BOTH sides wiring mutates — inner and the resolved dtype — so the inverse
                // restores the exact pre-wire state (unwire alone would leave the wired slot's dtype).
                let old_inner = st.inner.clone();
                let old_dtype = st.dtype;
                match inner {
                    // A wire can stop being applicable under a peer edit — the target is no longer
                    // a member, or another stub already exposes that slot. `set_stub_inner`
                    // validates before mutating, so a refused attempt leaves the stub untouched.
                    Some(target) => {
                        if g.set_stub_inner(scope, &stub_id, Some(target)).is_err() {
                            return Ok((Outcome::Ok, Command::Compound(vec![])));
                        }
                    }
                    // An unwire always applies (the stub exists — checked above).
                    None => g.set_stub_inner(scope, &stub_id, None)?,
                }
                // The inverse path forces the captured dtype back: wiring resolves a stub's dtype
                // from the inner slot, so without this an unwired pill would keep the wired slot's
                // type instead of its own provisional one.
                if let Some(dt) = dtype {
                    if let Some(st) = g.stub_mut(scope, &stub_id) {
                        st.dtype = dt;
                    }
                }
                Ok((Outcome::Ok, Command::WireStub { scope, stub_id, inner: old_inner, dtype: Some(old_dtype) }))
            }

            Command::EditStub { scope, stub_id, name, pos } => {
                let Some(st) = g.scope(scope).and_then(|s| s.stubs.get(&stub_id)) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: stub gone
                };
                let old_name = name.as_ref().map(|_| st.name.clone());
                let old_pos = pos.map(|_| st.pos);
                // One handle for the read above and both writes. The `StubId` never changes, so a
                // rename leaves every external wire intact.
                if let Some(st) = g.stub_mut(scope, &stub_id) {
                    if let Some(n) = &name {
                        st.name = n.clone();
                    }
                    if let Some(p) = pos {
                        st.pos = p;
                    }
                }
                Ok((Outcome::Ok, Command::EditStub { scope, stub_id, name: old_name, pos: old_pos }))
            }
        }
    }
}

/// A per-session undo/redo history over one shared [`Graph`]. Each entry holds ONE toggle plus the
/// session that issued it, and executing a toggle returns the next one — so an entry ping-pongs and
/// stays **uid-stable**: redo restores the very uid the undo removed, never a fresh one. Scoped by
/// session, so one client's timeline is independent of another's over the one shared history.
#[derive(Default)]
pub struct CommandHistory {
    entries: Vec<HistoryEntry>,
}

struct HistoryEntry {
    /// The command that flips this entry's state: its inverse when applied, its forward when undone.
    toggle: Command,
    session: String,
    undone: bool,
}

impl CommandHistory {
    pub fn new() -> CommandHistory {
        CommandHistory::default()
    }

    /// Execute `cmd` against `g`, record its inverse tagged with `session`, and return the outcome.
    /// A new command clears THIS session's redo run (its trailing undone entries) — a fresh edit
    /// invalidates that session's redo future, but never another session's.
    pub fn apply(&mut self, g: &mut Graph, session: &str, cmd: Command) -> Result<Outcome, String> {
        // The fresh-caller gate. `flip` deliberately does NOT call this — see `Command::precondition`.
        cmd.precondition(g)?;
        let (outcome, inverse) = cmd.execute(g)?;
        // Record EVERY successful command, a forward no-op included. The client records exactly
        // one entry per successful mutating RPC, unconditionally, so the two stacks must stay 1:1
        // — skipping a no-op here desyncs them and a later undo flips the WRONG entry.
        self.entries.retain(|e| !(e.session == session && e.undone));
        self.entries.push(HistoryEntry { toggle: inverse, session: session.to_string(), undone: false });
        Ok(outcome)
    }

    /// Drop the entire history (every session's entries). Loading a patch fully resets the
    /// session — there is nothing to undo across a `load_text` — so the manager clears here.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Undo the session's most-recent applied command. `Ok(false)` if it has nothing to undo.
    pub fn undo(&mut self, g: &mut Graph, session: &str) -> Result<bool, String> {
        let Some(idx) = self.entries.iter().rposition(|e| e.session == session && !e.undone) else {
            return Ok(false);
        };
        self.flip(g, idx, true)
    }

    /// Redo the session's most-recently-undone command. `Ok(false)` if it has nothing to redo.
    pub fn redo(&mut self, g: &mut Graph, session: &str) -> Result<bool, String> {
        let Some(idx) = self.entries.iter().position(|e| e.session == session && e.undone) else {
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

    pub fn can_undo(&self, session: &str) -> bool {
        self.entries.iter().any(|e| e.session == session && !e.undone)
    }

    pub fn can_redo(&self, session: &str) -> bool {
        self.entries.iter().any(|e| e.session == session && e.undone)
    }
}

/// Capture the exact inverse to restore the subtree rooted at `root` — a plain leaf, a scope member
/// (leaf or nested scope), or a top-level instance — BEFORE the caller removes it. The Compound
/// recreates every node (uid + params), every scope (innermost-first), the deleted top's membership,
/// any enclosing-scope stub the removal will prune, and every link touching the subtree — uid-stable.
fn capture_subtree_restore(g: &Graph, root: Uid) -> (Command, std::collections::HashSet<Uid>) {
    // Where the restored top returns to: `None` = ROOT (a top-level instance / leaf).
    let orig_parent = g.scope_of(root);

    // Walk the subtree (root + all descendants), splitting live nodes from scopes.
    let mut leaves: Vec<Uid> = Vec::new();
    let mut scopes: Vec<Uid> = Vec::new(); // discovery order (root first)
    let mut stack = vec![root];
    while let Some(u) = stack.pop() {
        if g.scope(u).is_some() {
            scopes.push(u);
            stack.extend(g.scope_members(u));
        } else {
            leaves.push(u);
        }
    }

    let mut cmds: Vec<Command> = Vec::new();

    // 1. Recreate every leaf (any depth) at ROOT, uid-stable, with its FULL persisted state —
    //    params, expression bindings, and viewer view-state (not just literal param values).
    for &u in &leaves {
        let exprs = g
            .param_bindings(u)
            .into_iter()
            .map(|(group, name, source, enabled, triggers)| {
                (group, name, ExprState { source, enabled, triggers })
            })
            .collect();
        let viewers = g.viewers(u).filter(|v| v.as_object().is_some_and(|m| !m.is_empty())).cloned();
        cmds.push(Command::AddNode {
            type_name: g.type_name(u).unwrap_or("").to_string(),
            pos: g.pos(u).unwrap_or([0.0, 0.0]),
            uid: Some(u),
            name: g.name(u).map(str::to_string),
            params: g.params(u).map(|p| (*p).clone()),
            exprs,
            viewers,
            // Membership is restored by the SetScope child below, not here — see the field's doc.
            scope: None,
        });
    }

    // 2. Recreate every scope INNERMOST-FIRST (a nested scope must exist before its parent groups
    //    it). `scopes` is root-first, so reverse ⇒ deepest-first. Each carries its captured parent,
    //    so it lands in place (a nested scope re-nests, an EMPTY scope restores without a []-Group
    //    choking on `common_parent`).
    for &s in scopes.iter().rev() {
        cmds.push(Command::Group {
            members: g.scope_members(s),
            pos: g.pos(s).unwrap_or([0.0, 0.0]),
            restore: Some(ScopeRestore {
                scope_id: s,
                name: g.name(s).unwrap_or("").to_string(),
                stubs: g.scope(s).map(|sc| sc.stubs.clone()).unwrap_or_default(),
                parent: g.scope_of(s),
                parent_stubs: vec![], // a delete-undo prunes enclosing stubs (AddStub, below), never re-points
            }),
        });
    }

    // 3. Move the restored top back INSIDE its enclosing scope (a member delete); a top-level
    //    instance already lands at ROOT.
    if orig_parent.is_some() {
        cmds.push(Command::SetScope { uid: root, scope: orig_parent });
    }

    // 4. Re-add any enclosing-scope stub the removal will prune (a stub whose inner named `root`).
    if let Some(parent) = orig_parent {
        if let Some(psc) = g.scope(parent) {
            for (id, st) in &psc.stubs {
                if st.inner.as_ref().map(|(u, _)| *u == root).unwrap_or(false) {
                    cmds.push(Command::AddStub {
                        scope: parent,
                        dir: st.dir,
                        dtype: st.dtype,
                        pos: st.pos,
                        restore: Some((id.clone(), st.clone())),
                    });
                }
            }
        }
    }

    // 5. Recreate every link touching the subtree (internal + crossing), after all endpoints exist.
    // The set is handed back too: it is exactly the uids the delete takes off the canvas, which is
    // what a panel bound to one of them has to stop naming.
    let subtree: std::collections::HashSet<Uid> = leaves.iter().chain(scopes.iter()).copied().collect();
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

