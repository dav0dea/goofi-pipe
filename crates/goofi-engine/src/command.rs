//! Semantic patch commands with exact inverses — the manager's undo/redo unit.
//!
//! Each [`Command::execute`] mutates the [`Graph`] and returns `(outcome, inverse)`: the inverse is a
//! fully-formed `Command` that, executed, restores the pre-state — and itself returns the forward
//! again, so redo is just executing what undo returned. The manager records inverses in a per-session
//! [`CommandHistory`]; undo/redo are `execute(inverse)` / `execute(forward)`.
//!
//! The surface is deliberately minimal — add/remove node, add/remove link, and one `Edit*` op per
//! target (node / param / global) that covers every field of that target. Loading a patch is NOT a
//! command: a load resets the whole session, so there is nothing to undo across it.
//!
//! Inverses are **idempotent** where multi-client convergence needs it: an edit/remove against an
//! already-absent node is a benign no-op, so two clients undoing the same change converge.

use crate::subpatch::{Dir, Stub, StubId};
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
    pub parent_stubs: Vec<(Uid, StubId, Option<(Uid, String)>)>,
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
        /// The sub-patch scope to create the node INSIDE (`None` = ROOT). Set when the user adds
        /// while an editor has entered a scope. Deliberately NOT used by a `RemoveNode` inverse:
        /// that restores membership with its own [`Command::SetScope`] child (see
        /// [`capture_subtree_restore`]), so a restore leaves this `None` and nothing re-parents a
        /// node the idempotent-reuse branch found already placed.
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
        inner: Option<(Uid, String)>,
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
    /// Apply this command to `g`, returning its result and the exact inverse command.
    pub fn execute(self, g: &mut Graph) -> Result<(Outcome, Command), String> {
        match self {
            Command::Compound(cmds) => {
                let mut inverses = Vec::with_capacity(cmds.len());
                let mut last = Outcome::Ok;
                for c in cmds {
                    let (res, inv) = c.execute(g)?;
                    last = res;
                    inverses.push(inv);
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
                // Register membership through the one validated re-parent seam — `scope_of` is the
                // SSOT for parentage. ONLY when a scope was actually asked for: a `RemoveNode`
                // inverse restores membership with its own `SetScope` child, and the idempotent
                // branch above can hand back a node that is already placed, so an unconditional
                // re-parent to ROOT would yank a live member out of its sub-patch.
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
                let inverse = capture_subtree_restore(g, uid);
                // A scope MEMBER routes through remove_member (prunes the enclosing scope's stubs); a
                // top-level scope tears down its subtree; a plain leaf is a single-node removal.
                if g.scope_of(uid).is_some() {
                    g.remove_member(uid)?;
                } else if g.scope(uid).is_some() {
                    g.remove_instance(uid)?;
                } else {
                    g.remove_node(uid)?;
                }
                Ok((Outcome::Ok, inverse))
            }

            Command::AddLink { node_out, slot_out, node_in, slot_in } => {
                // Idempotent: an endpoint node is gone (a peer deleted it) → the wire cannot exist,
                // so restoring it is a benign no-op. AddLink is the RemoveLink inverse AND the
                // trailing child of the RemoveNode-inverse Compound, so without this a concurrent
                // endpoint delete would error through flip() — wedging the session AND leaving the
                // earlier Compound child (a restored node) applied but unbroadcast (a phantom). A
                // forward user add_link to a missing node is impossible from a well-formed client.
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
                // Idempotent: the wire is already gone (a peer removed it, or undo/redo racing a
                // concurrent delete) → benign no-op. Erroring here would propagate through flip()
                // and permanently wedge the session's undo stack (undo keeps re-selecting the
                // un-flippable entry). RemoveLink is the inverse of every user add_link, so this
                // guard is what lets multi-client undo of a connect converge.
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
                // Capture the pre-rename name only if the rename actually lands. A concurrent peer
                // may have reclaimed the target name since this toggle was recorded (undo of a
                // rename after another client minted a node onto the freed base name). Tolerate that
                // collision as a recoverable no-op — propagating Err would flow through flip() and
                // permanently wedge the session's undo stack (undo re-selects the un-flippable
                // entry forever). A FORWARD user rename is pre-validated at the bridge, so a
                // collision reaching this arm is always a stale-toggle replay.
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
                            .and_then(|p| param(p, &group, &name))
                            .cloned()
                            .ok_or_else(|| format!("edit_param: no param {group}.{name} on {}", uid.to_hex()))?,
                    ),
                    None => None,
                };
                let old_expr = expr.as_ref().map(|_| {
                    g.param_expression(uid, &group, &name)
                        .map(|e| ExprState { source: e.source, enabled: e.enabled, triggers: e.triggers_process })
                        .unwrap_or(ExprState { source: String::new(), enabled: false, triggers: false })
                });
                // Literal and binding are independent slots on the param, so order is immaterial.
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

            Command::SetScope { uid, scope } => {
                // Idempotent: the uid is gone (a redo racing a delete) → no-op.
                if !g.contains(uid) && g.scope(uid).is_none() {
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                // Idempotent: the DESTINATION scope was dissolved (a peer's expand/delete racing this
                // toggle). SetScope is never a user RPC — it exists only as the membership-restoring
                // child of a `RemoveNode` inverse — so a missing destination is always a stale replay.
                // Erroring would propagate through flip() and permanently wedge the session's undo
                // stack, AND strand the Compound's already-executed AddNode live but unbroadcast (the
                // bridge gates its re-mirror on `result.is_ok()`). Degrade like the sibling arms: the
                // restored member simply lands at ROOT.
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
                        // Re-point parent stubs Expand re-pointed away — the exact reversal of
                        // expand_instance (empty for a delete-undo, which prunes rather than re-points).
                        for (p, sid, inner) in r.parent_stubs {
                            g.restore_stub_inner(p, &sid, inner);
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
                        g.insert_stub(scope, id.clone(), stub)?;
                        id
                    }
                };
                Ok((Outcome::StubId(id.clone()), Command::RemoveStub { scope, stub_id: id }))
            }

            Command::RemoveStub { scope, stub_id } => {
                let Some(stub) = g.scope(scope).and_then(|s| s.stubs.get(&stub_id).cloned()) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: already gone
                };
                g.remove_boundary(scope, &stub_id)?;
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
                    // A wire (inner=Some) can become non-applicable under a concurrent peer edit — the
                    // target is no longer a member, or another stub already exposes that inner slot.
                    // Tolerate it as a recoverable no-op (like the stub-gone guard) instead of erroring
                    // through flip() and wedging. set_stub_inner validates before mutating, so a failed
                    // attempt leaves the stub untouched.
                    Some(target) => {
                        if g.set_stub_inner(scope, &stub_id, Some(target)).is_err() {
                            return Ok((Outcome::Ok, Command::Compound(vec![])));
                        }
                    }
                    // An unwire always applies (the stub exists — checked above).
                    None => g.set_stub_inner(scope, &stub_id, None)?,
                }
                if let Some(dt) = dtype {
                    g.set_stub_dtype(scope, &stub_id, dt)?; // inverse path: force the captured dtype back
                }
                Ok((Outcome::Ok, Command::WireStub { scope, stub_id, inner: old_inner, dtype: Some(old_dtype) }))
            }

            Command::EditStub { scope, stub_id, name, pos } => {
                let Some(st) = g.scope(scope).and_then(|s| s.stubs.get(&stub_id)) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: stub gone
                };
                let old_name = name.as_ref().map(|_| st.name.clone());
                let old_pos = pos.map(|_| st.pos);
                if let Some(n) = &name {
                    g.rename_boundary(scope, &stub_id, n)?;
                }
                if let Some(p) = pos {
                    g.set_boundary_pos(scope, &stub_id, p)?;
                }
                Ok((Outcome::Ok, Command::EditStub { scope, stub_id, name: old_name, pos: old_pos }))
            }
        }
    }
}

/// A per-session undo/redo history over a shared [`Graph`]. Each entry holds a single `toggle`
/// command — the command that flips the entry's applied/undone state — plus the session that issued
/// it. Executing a toggle returns the NEXT toggle (its own inverse), so an entry ping-pongs
/// forward↔inverse and stays **uid-stable**: an `AddNode`'s first execution mints uid X, and undo
/// captures the uid-stable restore (via `RemoveNode{X}`'s inverse), so redo restores X — never a
/// fresh uid. Undo/redo are scoped to a session, so one client's timeline is independent of another's
/// over the single shared history (multi-client). Layout is NOT here — it stays client-local.
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
        let (outcome, inverse) = cmd.execute(g)?;
        // Record EVERY successful command — including a forward no-op (an idempotent guard fired,
        // yielding an empty-Compound inverse). The delegating client records exactly one graph_cmd
        // per successful mutation RPC, UNCONDITIONALLY, and its undo delegates back to this history;
        // so the two stacks must stay 1:1. Skipping the no-op here would desync them — a later undo
        // would flip the WRONG (earlier) entry. A no-op's toggle is an empty Compound, so undoing/
        // redoing it is itself a benign no-op; and a fresh command clearing this session's redo run
        // mirrors the client clearing its own redo on any recorded action.
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
fn capture_subtree_restore(g: &Graph, root: Uid) -> Command {
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
            params: g.params(u).cloned(),
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

    Command::Compound(cmds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn freq(uid: Uid, g: &Graph) -> Option<Param> {
        g.params(uid).and_then(|p| param(p, "common", "max_frequency")).cloned()
    }

    /// An `AddNode` for a fresh user add, parameterised only by where it lands.
    fn add_into(scope: Option<Uid>) -> Command {
        Command::AddNode {
            type_name: "Oscillator".into(),
            pos: [1.0, 2.0],
            uid: None,
            name: None,
            params: None,
            exprs: vec![],
            viewers: None,
            scope,
        }
    }

    #[test]
    fn add_node_round_trips_with_its_inverse() {
        let mut g = Graph::new();
        let (res, inverse) = add_into(None).execute(&mut g).unwrap();
        let Outcome::Uid(uid) = res else { panic!("add_node returns a uid") };
        assert!(g.contains(uid), "node added");
        assert_eq!(g.pos(uid), Some([1.0, 2.0]));

        let (_r, forward) = inverse.execute(&mut g).unwrap();
        assert!(!g.contains(uid), "inverse removed the node");
        forward.execute(&mut g).unwrap();
        assert!(g.contains(uid), "redo restored the same uid");
    }

    #[test]
    fn add_node_into_an_existing_scope_registers_membership() {
        // A node added while the user has ENTERED a sub-patch belongs to THAT scope, not to ROOT.
        // `scope_of` is the single source of truth for parentage, so its three readers must agree.
        let mut g = Graph::new();
        let sibling = g.add_node("Buffer", None).unwrap();
        let scope = g.group_nodes(&[sibling], [0.0, 0.0]).unwrap();

        let (res, _inv) = add_into(Some(scope)).execute(&mut g).unwrap();
        let Outcome::Uid(added) = res else { panic!("add_node returns a uid") };

        assert_eq!(g.scope_of(added), Some(scope), "scope_of names the entered scope");
        assert!(g.scope_members(scope).contains(&added), "the scope lists it as a direct member");
        let root: Vec<Uid> = g.node_uids().into_iter().filter(|u| g.scope_of(*u).is_none()).collect();
        assert!(!root.contains(&added), "and it is NOT a root node; root = {root:?}");
    }

    #[test]
    fn add_node_into_an_unknown_scope_errors_before_creating_anything() {
        // Validated BEFORE the mutation: an add that cannot honour its scope leaves the graph
        // untouched. Creating the node first and only then failing would leave a stray node at ROOT
        // — the graph and its CRDT mirror disagreeing on a command the caller was told had failed.
        let mut g = Graph::new();
        let before = g.node_uids().len();

        let err = add_into(Uid::from_hex("deadbeef")).execute(&mut g).unwrap_err();
        assert!(err.contains("scope"), "the error names the scope; got {err}");
        assert_eq!(g.node_uids().len(), before, "no node was created by the refused add");
    }

    #[test]
    fn remove_node_inverse_restores_uid_params_and_links() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();
        g.add_link(osc, "out", buf, "data").unwrap();
        g.update_param(osc, "common", "max_frequency", Param::float(42.0, 0.0, 100.0)).unwrap();
        let before = freq(osc, &g);

        let (_res, inverse) = Command::RemoveNode { uid: osc }.execute(&mut g).unwrap();
        assert!(!g.contains(osc), "node removed");
        assert_eq!(g.links_view().len(), 0, "its link went with it");

        inverse.execute(&mut g).unwrap();
        assert!(g.contains(osc), "node restored under the same uid");
        assert_eq!(freq(osc, &g), before, "params restored exactly");
        assert_eq!(g.links_view().len(), 1, "link restored");
    }

    #[test]
    fn remove_instance_is_undoable_restoring_the_whole_subtree() {
        // Deleting a collapsed sub-patch instance tears down its subtree (members + internal links +
        // stubs). RemoveNode must capture it all so the inverse restores the scope, its members, and
        // their links UID-STABLY — and a redo re-deletes cleanly.
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        g.add_link(a, "out", b, "data").unwrap();
        let scope = g.group_nodes(&[a, b], [5.0, 6.0]).unwrap(); // a→b stays internal (both inside)

        let (_r, inverse) = Command::RemoveNode { uid: scope }.execute(&mut g).unwrap();
        assert!(g.scope(scope).is_none(), "scope gone");
        assert!(!g.contains(a) && !g.contains(b), "members torn down");
        assert_eq!(g.links_view().len(), 0, "internal link went with them");

        let (_r2, redo) = inverse.execute(&mut g).unwrap();
        assert!(g.scope(scope).is_some(), "scope restored under the same uid");
        assert!(g.contains(a) && g.contains(b), "members restored uid-stable");
        assert_eq!(g.scope_members(scope).len(), 2, "both members back inside the scope");
        assert_eq!(g.scope_of(a), Some(scope), "a re-parented into the scope");
        assert_eq!(g.links_view().len(), 1, "internal link restored");
        assert_eq!(g.pos(scope), Some([5.0, 6.0]), "scope pos restored");

        // Redo re-deletes the whole subtree.
        redo.execute(&mut g).unwrap();
        assert!(g.scope(scope).is_none() && !g.contains(a) && !g.contains(b), "redo re-deleted the subtree");
    }

    #[test]
    fn remove_member_leaf_is_undoable_restoring_membership() {
        // Deleting a LEAF inside a sub-patch must restore it back INSIDE the scope (not orphaned at
        // ROOT) on undo, with its internal link.
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        g.add_link(a, "out", b, "data").unwrap();
        let scope = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();

        let (_r, inverse) = Command::RemoveNode { uid: b }.execute(&mut g).unwrap();
        assert!(!g.contains(b), "member leaf removed");
        assert_eq!(g.scope_of(a), Some(scope), "sibling a still a member");
        assert_eq!(g.links_view().len(), 0, "the internal link went with b");

        inverse.execute(&mut g).unwrap();
        assert!(g.contains(b), "member restored");
        assert_eq!(g.scope_of(b), Some(scope), "restored back INSIDE its scope, not at ROOT");
        assert_eq!(g.links_view().len(), 1, "internal link a→b restored");
    }

    #[test]
    fn remove_nested_instance_member_is_undoable() {
        // Deleting a nested sub-patch (a scope that is a MEMBER of another scope) restores the nested
        // subtree back inside its parent scope.
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        let inner = g.group_nodes(&[b], [0.0, 0.0]).unwrap(); // inner scope holds b
        let outer = g.group_nodes(&[a, inner], [1.0, 1.0]).unwrap(); // outer holds a + inner
        assert_eq!(g.scope_of(inner), Some(outer), "inner nested in outer");

        let (_r, inverse) = Command::RemoveNode { uid: inner }.execute(&mut g).unwrap();
        assert!(g.scope(inner).is_none() && !g.contains(b), "nested subtree gone");
        assert_eq!(g.scope_of(a), Some(outer), "outer's other member survives");

        inverse.execute(&mut g).unwrap();
        assert!(g.scope(inner).is_some() && g.contains(b), "nested subtree restored");
        assert_eq!(g.scope_of(inner), Some(outer), "inner re-nested in outer");
        assert_eq!(g.scope_of(b), Some(inner), "b back inside inner");
    }

    #[test]
    fn remove_node_inverse_restores_expression_bindings_and_viewers() {
        // params carry only the literal value — a node's expression BINDINGS and viewer view-state
        // are separate node state. The RemoveNode inverse must restore them too (else delete→undo
        // silently freezes a live-driven param to a literal and blanks its viewer).
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        g.set_expression(osc, "common", "max_frequency", "globals.default_ufreq", true, false).unwrap();
        g.set_node_viewers(osc, serde_json::json!({ "out": { "kind": "line" } })).unwrap();
        assert!(g.param_expression(osc, "common", "max_frequency").is_some(), "bound before delete");

        let (_r, inverse) = Command::RemoveNode { uid: osc }.execute(&mut g).unwrap();
        assert!(!g.contains(osc), "node removed");

        inverse.execute(&mut g).unwrap();
        let binding = g.param_expression(osc, "common", "max_frequency");
        assert!(binding.is_some(), "expression binding restored on undo");
        assert_eq!(binding.unwrap().source, "globals.default_ufreq", "binding source restored");
        assert_eq!(
            g.viewers(osc),
            Some(&serde_json::json!({ "out": { "kind": "line" } })),
            "viewer view-state restored"
        );
    }

    #[test]
    fn remove_an_empty_instance_is_undoable() {
        // A sub-patch whose members were all deleted is a live EMPTY scope. Deleting it must still
        // undo — the restore's []-member Group must not choke on `common_parent([])`.
        let mut g = Graph::new();
        let b = g.add_node("Buffer", None).unwrap();
        let scope = g.group_nodes(&[b], [1.0, 2.0]).unwrap();
        Command::RemoveNode { uid: b }.execute(&mut g).unwrap(); // empty the scope
        assert!(g.scope(scope).is_some() && g.scope_members(scope).is_empty(), "live empty scope");

        let (_r, inverse) = Command::RemoveNode { uid: scope }.execute(&mut g).unwrap();
        assert!(g.scope(scope).is_none(), "empty scope deleted");
        inverse.execute(&mut g).unwrap(); // must NOT error 'group: empty selection'
        assert!(g.scope(scope).is_some() && g.scope_members(scope).is_empty(), "empty scope restored");
        assert_eq!(g.pos(scope), Some([1.0, 2.0]), "pos restored");
    }

    #[test]
    fn remove_instance_with_an_empty_nested_scope_is_undoable() {
        // A deleted subtree can contain an empty nested scope; the whole delete must still undo.
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        let inner = g.group_nodes(&[b], [0.0, 0.0]).unwrap();
        let outer = g.group_nodes(&[a, inner], [0.0, 0.0]).unwrap();
        Command::RemoveNode { uid: b }.execute(&mut g).unwrap(); // empty the inner scope
        assert!(g.scope(inner).is_some() && g.scope_members(inner).is_empty(), "live empty inner scope");

        let (_r, inverse) = Command::RemoveNode { uid: outer }.execute(&mut g).unwrap();
        assert!(g.scope(outer).is_none() && g.scope(inner).is_none(), "subtree gone");
        inverse.execute(&mut g).unwrap();
        assert!(g.scope(outer).is_some() && g.scope(inner).is_some(), "both scopes restored");
        assert_eq!(g.scope_of(inner), Some(outer), "inner re-nested in outer");
        assert!(g.scope_members(inner).is_empty(), "inner restored empty");
        assert_eq!(g.scope_of(a), Some(outer), "a restored in outer");
    }

    #[test]
    fn remove_node_is_idempotent_when_already_gone() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        g.remove_node(osc).unwrap();
        let (res, inverse) = Command::RemoveNode { uid: osc }.execute(&mut g).unwrap();
        assert_eq!(res, Outcome::Ok);
        assert_eq!(inverse, Command::Compound(vec![]));
    }

    #[test]
    fn add_link_inverse_restores_a_displaced_single_input_wire() {
        // Connecting a second source to an occupied SINGLE input evicts the first wire. The inverse
        // must RESTORE the displaced wire (not just remove the new one), else undo-of-reconnect
        // leaves the input empty. (This is the manager's job now that the client delegates undo.)
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();
        g.add_link(a, "out", buf, "data").unwrap(); // a → buf.data occupies the input

        let (_r, inverse) =
            Command::AddLink { node_out: b, slot_out: "out".into(), node_in: buf, slot_in: "data".into() }
                .execute(&mut g)
                .unwrap();
        assert_eq!(g.links_view().len(), 1, "b displaced a — still one wire");
        assert!(g.links_view().iter().any(|l| l.node_out == b), "b now feeds buf");

        inverse.execute(&mut g).unwrap();
        assert_eq!(g.links_view().len(), 1, "one wire after undo");
        assert!(
            g.links_view().iter().any(|l| l.node_out == a && l.node_in == buf),
            "the displaced wire a → buf was restored"
        );
        assert!(!g.links_view().iter().any(|l| l.node_out == b), "the new wire was removed");
    }

    #[test]
    fn add_and_remove_link_are_inverses() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();

        let (_r, inverse) = Command::AddLink {
            node_out: osc,
            slot_out: "out".into(),
            node_in: buf,
            slot_in: "data".into(),
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(g.links_view().len(), 1, "link added");
        inverse.execute(&mut g).unwrap();
        assert_eq!(g.links_view().len(), 0, "inverse removed the link");
    }

    #[test]
    fn add_link_is_idempotent_when_an_endpoint_node_is_gone() {
        // AddLink is both the RemoveLink inverse AND the trailing child of the RemoveNode-inverse
        // Compound (restore the node, then re-add its links). A concurrent peer that deleted the
        // OTHER endpoint makes the wire un-creatable — this must be a benign no-op, not an Err that
        // both wedges the session AND (mid-Compound) leaves a phantom restored node with no broadcast.
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();
        g.remove_node(buf).unwrap(); // the peer deleted the input endpoint

        let (res, inverse) =
            Command::AddLink { node_out: osc, slot_out: "out".into(), node_in: buf, slot_in: "data".into() }
                .execute(&mut g)
                .unwrap();
        assert!(matches!(res, Outcome::Ok));
        assert!(matches!(inverse, Command::Compound(ref v) if v.is_empty()), "endpoint gone → no-op inverse");
        assert_eq!(g.links_view().len(), 0, "no phantom wire created");
    }

    #[test]
    fn remove_link_is_idempotent_when_the_wire_is_already_gone() {
        // A peer may drop a wire between one client's add_link and that client's undo (whose inverse
        // is RemoveLink). Executing RemoveLink on an absent wire must be a benign no-op — not an Err
        // that propagates through flip() and permanently wedges the session's undo stack.
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();

        let (res, inverse) = Command::RemoveLink {
            node_out: osc,
            slot_out: "out".into(),
            node_in: buf,
            slot_in: "data".into(),
        }
        .execute(&mut g)
        .unwrap();
        assert!(matches!(res, Outcome::Ok));
        assert!(matches!(inverse, Command::Compound(ref v) if v.is_empty()), "absent wire → no-op inverse");
    }

    #[test]
    fn add_link_of_an_existing_wire_records_a_noop_inverse() {
        // add_link is a silent no-op when the exact wire already exists. Its recorded inverse must
        // therefore ALSO be a no-op — a bare RemoveLink would DESTROY the pre-existing wire on undo.
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();
        g.add_link(osc, "out", buf, "data").unwrap(); // wire already present

        let (_r, inverse) = Command::AddLink {
            node_out: osc,
            slot_out: "out".into(),
            node_in: buf,
            slot_in: "data".into(),
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(g.links_view().len(), 1, "still one wire (the forward add was a no-op)");

        inverse.execute(&mut g).unwrap();
        assert_eq!(g.links_view().len(), 1, "undo of a no-op add must NOT destroy the pre-existing wire");
    }

    #[test]
    fn edit_node_name_and_pos_round_trip() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        g.set_node_pos(osc, [3.0, 4.0]).unwrap();
        let old_name = g.name(osc).unwrap().to_string();

        let (_r, inverse) = Command::EditNode { uid: osc, name: Some("renamed".into()), pos: Some([10.0, 20.0]) }
            .execute(&mut g)
            .unwrap();
        assert_eq!(g.name(osc), Some("renamed"));
        assert_eq!(g.pos(osc), Some([10.0, 20.0]));

        inverse.execute(&mut g).unwrap();
        assert_eq!(g.name(osc), Some(old_name.as_str()), "name restored");
        assert_eq!(g.pos(osc), Some([3.0, 4.0]), "pos restored");
    }

    #[test]
    fn edit_node_leaves_an_untouched_field_alone() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let name = g.name(osc).unwrap().to_string();

        // Only pos set → the inverse only carries pos; name is never touched.
        let (_r, inverse) = Command::EditNode { uid: osc, name: None, pos: Some([9.0, 9.0]) }.execute(&mut g).unwrap();
        assert_eq!(inverse, Command::EditNode { uid: osc, name: None, pos: Some([0.0, 0.0]) });
        assert_eq!(g.name(osc), Some(name.as_str()), "name untouched");
    }

    #[test]
    fn edit_node_moves_and_renames_a_scope_facade() {
        // A collapsed sub-patch instance is a scope, not a live node. EditNode must move + rename
        // its facade (`scopes[uid]`) uniformly, and its inverse must restore both — this is the
        // instance-drag / instance-rename path the client drives through the same set_node_pos /
        // rename_node seam as a plain node.
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let scope = g.group_nodes(&[a], [1.0, 2.0]).unwrap();
        let old_name = g.name(scope).unwrap().to_string();
        assert_eq!(g.pos(scope), Some([1.0, 2.0]), "facade starts at the group pos");

        let (_r, inverse) =
            Command::EditNode { uid: scope, name: Some("myscope".into()), pos: Some([50.0, 60.0]) }
                .execute(&mut g)
                .unwrap();
        assert_eq!(g.name(scope), Some("myscope"), "scope renamed");
        assert_eq!(g.pos(scope), Some([50.0, 60.0]), "scope facade moved");

        inverse.execute(&mut g).unwrap();
        assert_eq!(g.name(scope), Some(old_name.as_str()), "scope name restored");
        assert_eq!(g.pos(scope), Some([1.0, 2.0]), "scope pos restored");
    }

    #[test]
    fn edit_node_rename_inverse_no_ops_on_a_collision_instead_of_erroring() {
        // Multi-client: client-1 renames A "oscillator0" → "myosc" (its inverse restores
        // "oscillator0"). A peer then mints a node that reclaims the freed "oscillator0". Replaying
        // the inverse rename now collides — it MUST be a recoverable no-op (Ok), not an Err that
        // propagates through flip() and wedges the whole session's undo stack.
        let mut g = Graph::new();
        let a = g.add_node("Oscillator", None).unwrap();
        assert_eq!(g.name(a), Some("oscillator0"));

        let (_r, inverse) =
            Command::EditNode { uid: a, name: Some("myosc".into()), pos: None }.execute(&mut g).unwrap();
        assert_eq!(g.name(a), Some("myosc"));

        let b = g.add_node("Oscillator", None).unwrap();
        assert_eq!(g.name(b), Some("oscillator0"), "peer reclaimed the freed base name");

        // The inverse rename collides with the peer's node — recoverable no-op, not Err.
        let (_r2, _redo) = inverse.execute(&mut g).expect("inverse-rename collision must not error");
        assert_eq!(g.name(a), Some("myosc"), "A keeps its current name; the undo did not wedge");
        assert_eq!(g.name(b), Some("oscillator0"), "the peer's node is untouched");
    }

    #[test]
    fn a_stale_rename_undo_does_not_wedge_the_session_stack() {
        // The session-level guarantee behind the fix above: a rename whose inverse can no longer
        // apply must not block the entries BENEATH it. After the colliding rename-undo, the earlier
        // step is still undoable.
        let mut g = Graph::new();
        let mut h = CommandHistory::new();
        let buf = g.add_node("Buffer", None).unwrap();
        let a = g.add_node("Oscillator", None).unwrap();
        h.apply(&mut g, "s1", Command::EditNode { uid: buf, name: Some("mybuf".into()), pos: None }).unwrap();
        h.apply(&mut g, "s1", Command::EditNode { uid: a, name: Some("myosc".into()), pos: None }).unwrap();

        // A peer reclaims "oscillator0" (now free since A → "myosc").
        let _peer = g.add_node("Oscillator", None).unwrap();

        // Undo the rename (its inverse collides) → Ok, not Err; then the earlier step is still undoable.
        assert!(h.undo(&mut g, "s1").unwrap(), "colliding rename-undo returns Ok(true), not Err");
        assert!(h.undo(&mut g, "s1").unwrap(), "the earlier step is still undoable — the stack is not wedged");
        assert_eq!(g.name(buf), Some("buffer0"), "the earlier rename was undone");
    }

    #[test]
    fn edit_param_value_round_trips() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let before = freq(osc, &g).unwrap();

        let (_r, inverse) = Command::EditParam {
            uid: osc,
            group: "common".into(),
            name: "max_frequency".into(),
            value: Some(Param::float(7.0, 0.0, 100.0)),
            expr: None,
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(freq(osc, &g), Some(Param::float(7.0, 0.0, 100.0)), "value set");
        inverse.execute(&mut g).unwrap();
        assert_eq!(freq(osc, &g), Some(before), "inverse restored the old value");
    }

    #[test]
    fn edit_param_expression_inverse_clears_a_fresh_binding() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        assert!(g.param_expression(osc, "common", "max_frequency").is_none());

        let (_r, inverse) = Command::EditParam {
            uid: osc,
            group: "common".into(),
            name: "max_frequency".into(),
            value: None,
            expr: Some(ExprState { source: "globals.default_ufreq".into(), enabled: true, triggers: false }),
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(
            g.param_expression(osc, "common", "max_frequency").map(|e| e.source),
            Some("globals.default_ufreq".into())
        );
        inverse.execute(&mut g).unwrap();
        assert!(g.param_expression(osc, "common", "max_frequency").is_none(), "inverse cleared the binding");
    }

    #[test]
    fn edit_global_covers_add_edit_remove_with_inverses() {
        let mut g = Graph::new();
        // Add a fresh user global; the inverse removes it (old was absent).
        let (_r, undo_add) = Command::EditGlobal { name: "subj".into(), value: Some(GlobalValue::Str("P01".into())), at: None }
            .execute(&mut g)
            .unwrap();
        assert_eq!(g.globals().get("subj"), Some(&GlobalValue::Str("P01".into())));

        // Edit it; the inverse restores the prior value.
        let (_r, undo_edit) = Command::EditGlobal { name: "subj".into(), value: Some(GlobalValue::Str("P02".into())), at: None }
            .execute(&mut g)
            .unwrap();
        undo_edit.execute(&mut g).unwrap();
        assert_eq!(g.globals().get("subj"), Some(&GlobalValue::Str("P01".into())), "edit undone");

        // Undo the add → the global is gone again.
        undo_add.execute(&mut g).unwrap();
        assert_eq!(g.globals().get("subj"), None, "add undone");
    }

    #[test]
    fn a_global_rename_is_two_edit_globals_as_one_compound_step() {
        // Rename = add-new + remove-old, composed so it undoes as ONE step.
        let mut g = Graph::new();
        g.apply_global_change("subj", Some(GlobalValue::Int(7))).unwrap();

        let rename = Command::Compound(vec![
            Command::EditGlobal { name: "subject".into(), value: Some(GlobalValue::Int(7)), at: None },
            Command::EditGlobal { name: "subj".into(), value: None, at: None },
        ]);
        let (_r, inverse) = rename.execute(&mut g).unwrap();
        assert_eq!(g.globals().get("subj"), None);
        assert_eq!(g.globals().get("subject"), Some(&GlobalValue::Int(7)), "renamed");

        inverse.execute(&mut g).unwrap();
        assert_eq!(g.globals().get("subject"), None);
        assert_eq!(g.globals().get("subj"), Some(&GlobalValue::Int(7)), "one undo restores the old name + value");
    }

    /// The ordered list of global names — the projection that feeds the `.gfi`, the CRDT mirror, and
    /// the Globals panel (so its order is observable and must survive an undo).
    fn global_order(g: &Graph) -> Vec<String> {
        g.globals().entries().map(|(k, _, _)| k.to_string()).collect()
    }

    #[test]
    fn edit_global_delete_undo_restores_the_ordered_position() {
        // Deleting a MIDDLE global and undoing must return it to its original slot, not append it at
        // the tail (globals are ordered; the order is observable).
        let mut g = Graph::new();
        g.apply_global_change("a", Some(GlobalValue::Int(1))).unwrap();
        g.apply_global_change("gain", Some(GlobalValue::Int(2))).unwrap();
        g.apply_global_change("b", Some(GlobalValue::Int(3))).unwrap();
        let before = global_order(&g);

        let (_r, inverse) =
            Command::EditGlobal { name: "gain".into(), value: None, at: None }.execute(&mut g).unwrap();
        assert!(!g.globals().contains("gain"), "deleted");

        inverse.execute(&mut g).unwrap();
        assert_eq!(global_order(&g), before, "gain restored to its original slot, not the tail");
    }

    #[test]
    fn rename_global_undo_restores_the_ordered_position() {
        // A rename is the bridge's Compound [add-new, remove-old]. Undoing it must put the old name
        // back at its original slot (the remove-old inverse captures the removed index).
        let mut g = Graph::new();
        g.apply_global_change("a", Some(GlobalValue::Int(1))).unwrap();
        g.apply_global_change("gain", Some(GlobalValue::Int(2))).unwrap();
        g.apply_global_change("b", Some(GlobalValue::Int(3))).unwrap();
        let before = global_order(&g);

        let (_r, inverse) = Command::Compound(vec![
            Command::EditGlobal { name: "gain2".into(), value: Some(GlobalValue::Int(2)), at: None },
            Command::EditGlobal { name: "gain".into(), value: None, at: None },
        ])
        .execute(&mut g)
        .unwrap();
        assert!(g.globals().contains("gain2") && !g.globals().contains("gain"), "renamed");

        inverse.execute(&mut g).unwrap();
        assert_eq!(global_order(&g), before, "undo restores gain to its original slot");
    }

    // ── CommandHistory ───────────────────────────────────────────────────────────────────────────

    fn add_node(name: &str) -> Command {
        Command::AddNode {
            type_name: name.into(),
            pos: [0.0, 0.0],
            uid: None,
            name: None,
            params: None,
            exprs: vec![],
            viewers: None,
            scope: None,
        }
    }

    #[test]
    fn history_undo_redo_is_uid_stable() {
        // The whole reason for the toggle model: redo must restore the SAME uid (a checkpoint could
        // not). AddNode mints uid X; undo captures the uid-stable restore; redo brings X back.
        let mut g = Graph::new();
        let mut h = CommandHistory::new();
        let Outcome::Uid(uid) = h.apply(&mut g, "s1", add_node("Oscillator")).unwrap() else {
            panic!("add_node returns a uid")
        };
        assert!(g.contains(uid));

        assert!(h.undo(&mut g, "s1").unwrap());
        assert!(!g.contains(uid), "undo removed the node");
        assert!(h.redo(&mut g, "s1").unwrap());
        assert!(g.contains(uid), "redo restored the SAME uid");
        assert!(h.undo(&mut g, "s1").unwrap());
        assert!(!g.contains(uid), "undo again still works after a redo");
    }

    #[test]
    fn history_undo_is_scoped_per_session() {
        let mut g = Graph::new();
        let mut h = CommandHistory::new();
        let Outcome::Uid(a) = h.apply(&mut g, "s1", add_node("Oscillator")).unwrap() else { unreachable!() };
        let Outcome::Uid(b) = h.apply(&mut g, "s2", add_node("Buffer")).unwrap() else { unreachable!() };

        assert!(h.undo(&mut g, "s1").unwrap());
        assert!(!g.contains(a) && g.contains(b), "only s1's node undone");
        assert!(!h.undo(&mut g, "s1").unwrap(), "s1 has nothing left to undo");

        assert!(h.undo(&mut g, "s2").unwrap());
        assert!(!g.contains(b));
    }

    #[test]
    fn history_new_command_clears_that_sessions_redo() {
        let mut g = Graph::new();
        let mut h = CommandHistory::new();
        h.apply(&mut g, "s1", add_node("Oscillator")).unwrap();
        h.undo(&mut g, "s1").unwrap();
        assert!(h.can_redo("s1"), "an undone command is redoable");

        h.apply(&mut g, "s1", add_node("Buffer")).unwrap();
        assert!(!h.can_redo("s1"), "the new command cleared the redo run");
        assert!(!h.redo(&mut g, "s1").unwrap());
    }

    #[test]
    fn history_reports_can_undo_can_redo() {
        let mut g = Graph::new();
        let mut h = CommandHistory::new();
        assert!(!h.can_undo("s1") && !h.can_redo("s1"));
        h.apply(&mut g, "s1", add_node("Oscillator")).unwrap();
        assert!(h.can_undo("s1") && !h.can_redo("s1"));
        h.undo(&mut g, "s1").unwrap();
        assert!(!h.can_undo("s1") && h.can_redo("s1"));
    }

    #[test]
    fn history_apply_records_even_a_noop_to_stay_aligned_with_the_client() {
        // The delegating client records ONE graph_cmd per successful mutation RPC, UNCONDITIONALLY,
        // and its undo delegates back to this history — so the client↔manager stacks must stay 1:1.
        // A forward no-op (an idempotent guard fired, e.g. removing an already-absent wire) therefore
        // STILL records an entry here, matching the client's record; skipping it would desync the two
        // stacks so a later undo flips the wrong entry. Its toggle is an empty Compound, so undoing/
        // redoing it is itself a benign no-op.
        let mut g = Graph::new();
        let mut h = CommandHistory::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();

        assert!(!h.can_undo("s1"));
        h.apply(
            &mut g,
            "s1",
            Command::RemoveLink { node_out: osc, slot_out: "out".into(), node_in: buf, slot_in: "data".into() },
        )
        .unwrap();
        assert!(h.can_undo("s1"), "a no-op still records an entry (1:1 with the client's graph_cmd)");

        // Undoing the no-op entry succeeds and is a benign no-op (empty-Compound toggle), then redoable.
        assert!(h.undo(&mut g, "s1").unwrap(), "undo of the no-op entry succeeds");
        assert!(h.can_redo("s1"), "and it is redoable");
    }

    // ── structural commands (flat scope model) ────────────────────────────────────

    /// osc → buf → sink; group [osc, buf] cuts buf→sink into one Out stub.
    fn grouped_chain() -> (Graph, Uid, Uid, Uid) {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let buf = g.add_node("Buffer", None).unwrap();
        let sink = g.add_node("Buffer", None).unwrap();
        g.add_link(osc, "out", buf, "data").unwrap();
        g.add_link(buf, "out", sink, "data").unwrap();
        (g, osc, buf, sink)
    }

    #[test]
    fn group_undo_expands_and_redo_regroups_uid_stable() {
        let (mut g, osc, buf, _sink) = grouped_chain();
        let nodes_before = g.node_uids();
        let links_before = g.links_view().len();

        let (res, undo) = Command::Group { members: vec![osc, buf], pos: [5.0, 6.0], restore: None }
            .execute(&mut g)
            .unwrap();
        let Outcome::Uid(scope) = res else { panic!("group returns the scope uid") };
        assert_eq!(g.scope_of(osc), Some(scope), "osc grouped");
        let stub_count = g.scope(scope).unwrap().stubs.len();
        assert_eq!(stub_count, 1, "one Out stub for the buf→sink cut");

        // Undo → expand: members back at ROOT, scope gone, flat graph identical.
        let (_r, redo) = undo.execute(&mut g).unwrap();
        assert!(g.scope(scope).is_none(), "scope dissolved");
        assert_eq!(g.scope_of(osc), None, "osc back at ROOT");
        assert_eq!(g.node_uids(), nodes_before, "leaf uids unchanged");
        assert_eq!(g.links_view().len(), links_before, "flat links intact");

        // Redo → regroup at the SAME scope uid with the SAME stubs.
        let (res2, _u2) = redo.execute(&mut g).unwrap();
        assert_eq!(res2, Outcome::Uid(scope), "redo restored the SAME scope uid");
        assert_eq!(g.scope_of(osc), Some(scope), "osc regrouped under the same scope");
        assert_eq!(g.scope(scope).unwrap().stubs.len(), stub_count, "stubs restored verbatim");
    }

    #[test]
    fn group_undo_un_mints_the_stub_it_added_to_a_nested_member() {
        // Round-1's group_nodes MINTS a re-exposing stub on a nested member when a crossing link's
        // port was previously removed. Group's inverse (Expand) must UN-MINT it — else group→undo
        // leaves a spurious boundary port resurrected on the nested member (not an exact inverse).
        let mut g = Graph::new();
        let a = g.add_node("_TestEcho", None).unwrap();
        let x = g.add_node("_TestEcho", None).unwrap();
        g.add_link(a, "out", x, "in").unwrap();
        let s = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        g.remove_boundary(s, "out0").unwrap(); // drop the port; the flat link a.out→x.in survives
        assert!(g.scope(s).unwrap().stubs.is_empty(), "s starts with no port");

        // Group [s] — group_nodes re-mints a stub on s to expose the orphaned crossing link.
        let (_r, inverse) =
            Command::Group { members: vec![s], pos: [1.0, 1.0], restore: None }.execute(&mut g).unwrap();
        assert!(!g.scope(s).unwrap().stubs.is_empty(), "group re-minted a port on s");

        // Undo — dissolves the new scope AND un-mints the stub the group added to s.
        inverse.execute(&mut g).unwrap();
        assert!(
            g.scope(s).unwrap().stubs.is_empty(),
            "group→undo removed the re-minted port (an exact inverse, no resurrected boundary)"
        );
    }

    #[test]
    fn expand_command_undo_regroups_exactly() {
        let (mut g, osc, buf, _sink) = grouped_chain();
        let scope = g.group_nodes(&[osc, buf], [1.0, 2.0]).unwrap();
        let stub_count = g.scope(scope).unwrap().stubs.len();

        let (_res, undo) = Command::Expand { scope }.execute(&mut g).unwrap();
        assert!(g.scope(scope).is_none(), "expanded");
        assert_eq!(g.scope_of(osc), None);

        undo.execute(&mut g).unwrap(); // undo-of-expand = re-group exactly
        assert_eq!(g.scope_of(osc), Some(scope), "re-grouped under the same scope uid");
        assert_eq!(g.scope(scope).unwrap().stubs.len(), stub_count, "stubs restored");
    }

    #[test]
    fn expand_is_idempotent_when_the_scope_is_gone() {
        let (mut g, osc, buf, _sink) = grouped_chain();
        let scope = g.group_nodes(&[osc, buf], [0.0, 0.0]).unwrap();
        g.expand_instance(scope).unwrap();
        let (res, inv) = Command::Expand { scope }.execute(&mut g).unwrap();
        assert_eq!(res, Outcome::Ok);
        assert_eq!(inv, Command::Compound(vec![]), "no-op inverse when already expanded");
    }

    #[test]
    fn add_stub_and_remove_stub_are_inverses() {
        let (mut g, osc, buf, _sink) = grouped_chain();
        let scope = g.group_nodes(&[buf], [0.0, 0.0]).unwrap(); // osc→buf cut + buf→sink cut auto-stub
        let _ = osc;
        let before = g.scope(scope).unwrap().stubs.len();

        let (_res, undo) = Command::AddStub {
            scope,
            dir: Dir::In,
            dtype: goofi_core::SlotType::Array,
            pos: [3.0, 4.0],
            restore: None,
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(g.scope(scope).unwrap().stubs.len(), before + 1, "stub added");

        let (_r, redo) = undo.execute(&mut g).unwrap();
        assert_eq!(g.scope(scope).unwrap().stubs.len(), before, "inverse removed it");
        redo.execute(&mut g).unwrap();
        assert_eq!(g.scope(scope).unwrap().stubs.len(), before + 1, "redo restored the stub");
    }

    #[test]
    fn wire_stub_inverse_restores_the_old_inner_and_dtype() {
        let mut g = Graph::new();
        let a = g.add_node("Buffer", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        g.add_link(a, "out", b, "data").unwrap();
        let scope = g.group_nodes(&[a], [0.0, 0.0]).unwrap(); // auto out stub on a.out
        // A provisional STRING In stub — wiring to a's ARRAY input slot will coerce dtype to Array.
        let stub = g.add_boundary(scope, Dir::In, goofi_core::SlotType::String, [0.0, 0.0]).unwrap();
        assert!(g.scope(scope).unwrap().stubs[&stub].inner.is_none(), "born unwired");
        assert_eq!(g.scope(scope).unwrap().stubs[&stub].dtype, goofi_core::SlotType::String, "provisional dtype");

        let (_res, undo) = Command::WireStub {
            scope,
            stub_id: stub.clone(),
            inner: Some((a, "data".to_string())),
            dtype: None,
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(g.resolve_stub(scope, &stub), Some((a, "data".to_string())), "wired");
        assert_eq!(g.scope(scope).unwrap().stubs[&stub].dtype, goofi_core::SlotType::Array, "dtype resolved to the wired slot's");

        undo.execute(&mut g).unwrap(); // inverse restores the OLD inner (None) AND the OLD dtype (String)
        assert!(g.scope(scope).unwrap().stubs[&stub].inner.is_none(), "unwired back to the old state");
        assert_eq!(
            g.scope(scope).unwrap().stubs[&stub].dtype,
            goofi_core::SlotType::String,
            "dtype restored to the pre-wire provisional (not left as the wired Array)"
        );
    }

    #[test]
    fn wire_stub_tolerates_a_non_applicable_redo() {
        // Multi-client: after undoing a wire (sa unwired), a peer wires a DIFFERENT stub (sb) to the
        // same inner slot. Replaying sa's wire (a redo) is now non-applicable — that inner slot is
        // already exposed by sb. It must be a recoverable no-op, not an Err that wedges the session.
        let mut g = Graph::new();
        let l = g.add_node("_TestEcho", None).unwrap();
        let s = g.group_nodes(&[l], [0.0, 0.0]).unwrap();
        let sa = g.add_boundary(s, Dir::Out, goofi_core::SlotType::Array, [0.0, 0.0]).unwrap();
        let sb = g.add_boundary(s, Dir::Out, goofi_core::SlotType::Array, [0.0, 0.0]).unwrap();

        // A peer wires sb → (l, out); that inner slot is now exposed.
        g.set_stub_inner(s, &sb, Some((l, "out".to_string()))).unwrap();

        let (res, inverse) = Command::WireStub {
            scope: s,
            stub_id: sa.clone(),
            inner: Some((l, "out".to_string())),
            dtype: None,
        }
        .execute(&mut g)
        .unwrap();
        assert!(matches!(res, Outcome::Ok));
        assert!(matches!(inverse, Command::Compound(ref v) if v.is_empty()), "non-applicable wire → no-op");
        assert!(g.scope(s).unwrap().stubs[&sa].inner.is_none(), "sa left untouched (still unwired)");
    }

    #[test]
    fn add_stub_restore_is_idempotent_when_the_scope_is_gone() {
        // Multi-client: undo-of-add records an AddStub{restore}; a concurrent expand dissolves the
        // scope; replaying that AddStub{restore} must be a benign no-op, not an Err (redo would stick).
        let (mut g, _osc, buf, _sink) = grouped_chain();
        let scope = g.group_nodes(&[buf], [0.0, 0.0]).unwrap();
        let stub = g.add_boundary(scope, Dir::In, goofi_core::SlotType::Array, [0.0, 0.0]).unwrap();
        let captured = g.scope(scope).unwrap().stubs[&stub].clone();
        g.expand_instance(scope).unwrap(); // scope dissolved out from under the pending restore

        let (res, inv) = Command::AddStub {
            scope,
            dir: Dir::In,
            dtype: goofi_core::SlotType::Array,
            pos: [0.0, 0.0],
            restore: Some((stub, captured)),
        }
        .execute(&mut g)
        .unwrap(); // must NOT Err
        assert_eq!(res, Outcome::Ok);
        assert_eq!(inv, Command::Compound(vec![]), "no-op inverse when the scope is gone");
    }

    #[test]
    fn edit_stub_round_trips_name_and_pos() {
        let (mut g, _osc, buf, _sink) = grouped_chain();
        let scope = g.group_nodes(&[buf], [0.0, 0.0]).unwrap();
        let stub = g.scope(scope).unwrap().stubs.keys().next().unwrap().clone();
        let (old_name, old_pos) = {
            let st = &g.scope(scope).unwrap().stubs[&stub];
            (st.name.clone(), st.pos)
        };

        let (_res, undo) = Command::EditStub {
            scope,
            stub_id: stub.clone(),
            name: Some("wave".into()),
            pos: Some([9.0, 9.0]),
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(g.scope(scope).unwrap().stubs[&stub].name, "wave");
        assert_eq!(g.scope(scope).unwrap().stubs[&stub].pos, [9.0, 9.0]);

        undo.execute(&mut g).unwrap();
        assert_eq!(g.scope(scope).unwrap().stubs[&stub].name, old_name, "name restored");
        assert_eq!(g.scope(scope).unwrap().stubs[&stub].pos, old_pos, "pos restored");
    }

    #[test]
    fn group_expand_interleave_in_one_session_history() {
        // Structural + leaf commands share one session's history and undo in order.
        let (mut g, osc, buf, _sink) = grouped_chain();
        let mut h = CommandHistory::new();
        let scope = match h
            .apply(&mut g, "s1", Command::Group { members: vec![osc, buf], pos: [0.0, 0.0], restore: None })
            .unwrap()
        {
            Outcome::Uid(u) => u,
            _ => panic!("group returns a uid"),
        };
        h.apply(&mut g, "s1", add_node("Oscillator")).unwrap(); // a later leaf edit
        assert!(g.scope(scope).is_some());

        h.undo(&mut g, "s1").unwrap(); // undo the leaf add
        assert!(g.scope(scope).is_some(), "group still stands");
        h.undo(&mut g, "s1").unwrap(); // undo the group → expand
        assert!(g.scope(scope).is_none(), "group undone");
        h.redo(&mut g, "s1").unwrap(); // redo the group → same uid
        assert!(g.scope(scope).is_some(), "group redone at the same scope uid");
    }

    #[test]
    fn a_member_restore_whose_scope_a_peer_dissolved_does_not_wedge_the_session_stack() {
        // Multi-tab: session A deletes a scope MEMBER — its inverse is
        // `Compound[AddNode{member}, SetScope{member, Some(scope)}]`. Session B then dissolves that
        // scope. A's undo must degrade gracefully (the member returns at ROOT) rather than Err
        // through flip(), which would leave the entry applied — undo re-selecting it forever — and
        // leave the Compound's already-executed AddNode live but unbroadcast (the bridge gates
        // `resync_and_broadcast` on `result.is_ok()`).
        let mut g = Graph::new();
        let mut h = CommandHistory::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let b = g.add_node("Buffer", None).unwrap();
        let scope = g.group_nodes(&[a, b], [0.0, 0.0]).unwrap();

        let earlier = g.add_node("Oscillator", None).unwrap();
        h.apply(&mut g, "A", Command::EditNode { uid: earlier, name: Some("keep".into()), pos: None }).unwrap();
        h.apply(&mut g, "A", Command::RemoveNode { uid: a }).unwrap();
        h.apply(&mut g, "B", Command::Expand { scope }).unwrap();
        assert!(g.scope(scope).is_none(), "the peer dissolved the scope");

        assert!(h.undo(&mut g, "A").unwrap(), "the stale member-restore returns Ok(true), not Err");
        assert!(g.contains(a), "the member is restored");
        assert_eq!(g.scope_of(a), None, "its scope is gone, so it lands at ROOT");
        assert!(h.undo(&mut g, "A").unwrap(), "the earlier step is still undoable — the stack is not wedged");
        assert_eq!(g.name(earlier), Some("oscillator1"), "the earlier rename was undone");
    }

    #[test]
    fn a_scope_restore_whose_parent_a_peer_dissolved_lands_at_root() {
        // The sibling hole the same race opens on the structural side: a nested scope's delete-undo
        // carries its captured parent through `Group{restore}` → `restore_scope`. If the peer
        // dissolved that parent meanwhile, writing the captured parent verbatim installs a
        // dangling-parent orphan — a scope whose `scope_of` names a scope that no longer exists,
        // which nothing else in the graph can reach or clean up.
        let mut g = Graph::new();
        let mut h = CommandHistory::new();
        let a = g.add_node("Oscillator", None).unwrap();
        let inner = g.group_nodes(&[a], [0.0, 0.0]).unwrap();
        let outer = g.group_nodes(&[inner], [1.0, 1.0]).unwrap();

        h.apply(&mut g, "A", Command::RemoveNode { uid: inner }).unwrap(); // delete the nested scope
        h.apply(&mut g, "B", Command::Expand { scope: outer }).unwrap(); // peer dissolves its parent

        assert!(h.undo(&mut g, "A").unwrap(), "the stale scope-restore returns Ok(true)");
        assert!(g.scope(inner).is_some(), "the nested scope is restored");
        assert_eq!(g.scope_of(inner), None, "its parent is gone, so it lands at ROOT — not dangling");
    }
}
