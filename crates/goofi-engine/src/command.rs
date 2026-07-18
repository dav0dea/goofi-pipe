//! Semantic patch commands with exact inverses — the manager's undo/redo unit.
//!
//! Each [`Command::execute`] mutates the [`Graph`] and returns `(outcome, inverse)`: the inverse is a
//! fully-formed `Command` that, executed, restores the pre-state — and itself returns the forward
//! again, so redo is just executing what undo returned. The manager records inverses in a per-session
//! history; undo/redo are `execute(inverse)` / `execute(forward)`.
//!
//! Inverses are **idempotent** where multi-client convergence needs it: a remove against an already-
//! absent node is a benign no-op (so two clients undoing the same creation converge instead of
//! erroring). The pre-state an inverse needs is captured at execute time (e.g. `RemoveNode` returns a
//! `Compound` that re-adds the node with its uid + params, then re-adds its links).

use crate::{Graph, Uid};
use goofi_core::globals::GlobalValue;
use goofi_core::Param;
use goofi_node::{param, ParamGroups};

/// What a command produced, for the caller (the RPC reply). Kept serde-free so the engine needs no
/// JSON dep — the bridge maps it to the wire.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// A plain success (`{ ok: true }` on the wire).
    Ok,
    /// A minted/affected uid — `add_node` returns the node uid, etc.
    Uid(Uid),
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
    SetParam {
        uid: Uid,
        group: String,
        name: String,
        value: Param,
    },
    RenameNode {
        uid: Uid,
        name: String,
    },
    MoveNode {
        uid: Uid,
        pos: [f64; 2],
    },
    SetExpression {
        uid: Uid,
        group: String,
        name: String,
        source: String,
        enabled: bool,
        triggers: bool,
    },
    /// Upsert (`Some`) or remove (`None`) a global — one command covers add / edit / delete.
    SetGlobal {
        name: String,
        value: Option<GlobalValue>,
    },
    RenameGlobal {
        from: String,
        to: String,
    },
    /// Restore the whole graph from a `.gfi` serialization; its inverse is the graph as it was
    /// before. The wholesale-`load` inverse. NOT used for group/expand/etc.: `load_doc` re-mints
    /// uids (an idmap), which would invalidate other history entries' uid references — a load, by
    /// contrast, resets the session's history, so re-minting is harmless there. Uid-stable clean
    /// inverses for the structural ops (group↔expand, boundaries, share) are a focused follow-up.
    Checkpoint {
        yaml: String,
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

            Command::AddNode { type_name, pos, uid, name, params } => {
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
                let _ = g.set_node_pos(u, pos);
                Ok((Outcome::Uid(u), Command::RemoveNode { uid: u }))
            }

            Command::RemoveNode { uid } => {
                if !g.contains(uid) {
                    // Idempotent: already gone → no-op, and a no-op inverse (nothing to restore).
                    return Ok((Outcome::Ok, Command::Compound(vec![])));
                }
                let inverse = capture_restore(g, uid);
                g.remove_node(uid)?;
                Ok((Outcome::Ok, inverse))
            }

            Command::AddLink { node_out, slot_out, node_in, slot_in } => {
                g.add_link(node_out, &slot_out, node_in, &slot_in)?;
                Ok((Outcome::Ok, Command::RemoveLink { node_out, slot_out, node_in, slot_in }))
            }

            Command::RemoveLink { node_out, slot_out, node_in, slot_in } => {
                g.remove_link(node_out, &slot_out, node_in, &slot_in)?;
                Ok((Outcome::Ok, Command::AddLink { node_out, slot_out, node_in, slot_in }))
            }

            Command::SetParam { uid, group, name, value } => {
                let old = g
                    .params(uid)
                    .and_then(|p| param(p, &group, &name))
                    .cloned()
                    .ok_or_else(|| format!("set_param: no param {group}.{name} on {}", uid.to_hex()))?;
                g.update_param(uid, &group, &name, value)?;
                Ok((Outcome::Ok, Command::SetParam { uid, group, name, value: old }))
            }

            Command::RenameNode { uid, name } => {
                if !g.contains(uid) {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: node gone
                }
                let old = g.name(uid).unwrap_or("").to_string();
                g.rename_node(uid, &name)?;
                Ok((Outcome::Ok, Command::RenameNode { uid, name: old }))
            }

            Command::MoveNode { uid, pos } => {
                let Some(old) = g.pos(uid) else {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: node gone
                };
                g.set_node_pos(uid, pos)?;
                Ok((Outcome::Ok, Command::MoveNode { uid, pos: old }))
            }

            Command::SetExpression { uid, group, name, source, enabled, triggers } => {
                if !g.contains(uid) {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: node gone
                }
                let old = g.param_expression(uid, &group, &name);
                g.set_expression(uid, &group, &name, &source, enabled, triggers)?;
                let inverse = match old {
                    Some(e) => Command::SetExpression {
                        uid,
                        group,
                        name,
                        source: e.source,
                        enabled: e.enabled,
                        triggers: e.triggers_process,
                    },
                    // No prior binding → the inverse is a clear (empty source unbinds).
                    None => Command::SetExpression { uid, group, name, source: String::new(), enabled: false, triggers: false },
                };
                Ok((Outcome::Ok, inverse))
            }

            Command::SetGlobal { name, value } => {
                let old = g.globals().get(&name).cloned();
                g.apply_global_change(&name, value)?;
                Ok((Outcome::Ok, Command::SetGlobal { name, value: old }))
            }

            Command::RenameGlobal { from, to } => {
                let Some(value) = g.globals().get(&from).cloned() else {
                    return Ok((Outcome::Ok, Command::Compound(vec![]))); // idempotent: nothing to rename
                };
                g.apply_global_change(&from, None)?;
                g.apply_global_change(&to, Some(value))?;
                Ok((Outcome::Ok, Command::RenameGlobal { from: to, to: from }))
            }

            Command::Checkpoint { yaml } => {
                let before = g.serialize();
                g.load_doc(&yaml)?;
                Ok((Outcome::Ok, Command::Checkpoint { yaml: before }))
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
        self.entries.retain(|e| !(e.session == session && e.undone));
        self.entries.push(HistoryEntry { toggle: inverse, session: session.to_string(), undone: false });
        Ok(outcome)
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

/// Build the inverse of removing `uid`, captured BEFORE removal: re-add the node (same uid + name +
/// params) then re-add every link that touched it (so a downstream endpoint reconnects).
fn capture_restore(g: &Graph, uid: Uid) -> Command {
    let type_name = g.type_name(uid).unwrap_or("").to_string();
    let name = g.name(uid).map(str::to_string);
    let pos = g.pos(uid).unwrap_or([0.0, 0.0]);
    let params = g.params(uid).cloned();
    let mut cmds = vec![Command::AddNode { type_name, pos, uid: Some(uid), name, params }];
    for l in g.links_view() {
        if l.node_out == uid || l.node_in == uid {
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

    #[test]
    fn add_node_round_trips_with_its_inverse() {
        let mut g = Graph::new();
        let (res, inverse) = Command::AddNode {
            type_name: "Oscillator".into(),
            pos: [1.0, 2.0],
            uid: None,
            name: None,
            params: None,
        }
        .execute(&mut g)
        .unwrap();
        let Outcome::Uid(uid) = res else { panic!("add_node returns a uid") };
        assert!(g.contains(uid), "node added");
        assert_eq!(g.pos(uid), Some([1.0, 2.0]));

        // Undo (the inverse) removes it; the forward it returns re-adds it (redo).
        let (_r, forward) = inverse.execute(&mut g).unwrap();
        assert!(!g.contains(uid), "inverse removed the node");
        forward.execute(&mut g).unwrap();
        assert!(g.contains(uid), "redo restored the same uid");
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
    fn remove_node_is_idempotent_when_already_gone() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        g.remove_node(osc).unwrap();
        // Removing an already-absent node converges: no error, no-op inverse.
        let (res, inverse) = Command::RemoveNode { uid: osc }.execute(&mut g).unwrap();
        assert_eq!(res, Outcome::Ok);
        assert_eq!(inverse, Command::Compound(vec![]));
    }

    #[test]
    fn set_param_round_trips() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let before = freq(osc, &g).unwrap();

        let (_r, inverse) = Command::SetParam {
            uid: osc,
            group: "common".into(),
            name: "max_frequency".into(),
            value: Param::float(7.0, 0.0, 100.0),
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(freq(osc, &g), Some(Param::float(7.0, 0.0, 100.0)), "value set");

        inverse.execute(&mut g).unwrap();
        assert_eq!(freq(osc, &g), Some(before), "inverse restored the old value");
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
    fn rename_node_round_trips() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        let before = g.name(osc).unwrap().to_string();

        let (_r, inverse) = Command::RenameNode { uid: osc, name: "renamed".into() }.execute(&mut g).unwrap();
        assert_eq!(g.name(osc), Some("renamed"));
        inverse.execute(&mut g).unwrap();
        assert_eq!(g.name(osc), Some(before.as_str()), "inverse restores the old name");
    }

    #[test]
    fn move_node_round_trips() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        g.set_node_pos(osc, [3.0, 4.0]).unwrap();

        let (_r, inverse) = Command::MoveNode { uid: osc, pos: [10.0, 20.0] }.execute(&mut g).unwrap();
        assert_eq!(g.pos(osc), Some([10.0, 20.0]));
        inverse.execute(&mut g).unwrap();
        assert_eq!(g.pos(osc), Some([3.0, 4.0]), "inverse restores the old position");
    }

    #[test]
    fn set_expression_inverse_clears_a_freshly_bound_param() {
        let mut g = Graph::new();
        let osc = g.add_node("Oscillator", None).unwrap();
        assert!(g.param_expression(osc, "common", "max_frequency").is_none());

        let (_r, inverse) = Command::SetExpression {
            uid: osc,
            group: "common".into(),
            name: "max_frequency".into(),
            source: "globals.default_ufreq".into(),
            enabled: true,
            triggers: false,
        }
        .execute(&mut g)
        .unwrap();
        assert_eq!(g.param_expression(osc, "common", "max_frequency").map(|e| e.source), Some("globals.default_ufreq".into()));

        inverse.execute(&mut g).unwrap();
        assert!(g.param_expression(osc, "common", "max_frequency").is_none(), "inverse cleared the binding");
    }

    #[test]
    fn set_global_covers_add_edit_remove_with_inverses() {
        let mut g = Graph::new();
        // Add a fresh user global; the inverse removes it (old was absent).
        let (_r, undo_add) = Command::SetGlobal { name: "subj".into(), value: Some(GlobalValue::Str("P01".into())) }
            .execute(&mut g)
            .unwrap();
        assert_eq!(g.globals().get("subj"), Some(&GlobalValue::Str("P01".into())));

        // Edit it; the inverse restores the prior value.
        let (_r, undo_edit) = Command::SetGlobal { name: "subj".into(), value: Some(GlobalValue::Str("P02".into())) }
            .execute(&mut g)
            .unwrap();
        undo_edit.execute(&mut g).unwrap();
        assert_eq!(g.globals().get("subj"), Some(&GlobalValue::Str("P01".into())), "edit undone");

        // Undo the add → the global is gone again.
        undo_add.execute(&mut g).unwrap();
        assert_eq!(g.globals().get("subj"), None, "add undone");
    }

    #[test]
    fn checkpoint_restores_a_serialized_graph_and_its_inverse_restores_the_prior() {
        let mut g = Graph::new();
        g.add_node("Oscillator", None).unwrap();
        let state_a = g.serialize(); // one node
        g.add_node("Buffer", None).unwrap(); // two nodes
        assert_eq!(g.node_uids().len(), 2);

        // Checkpoint to state A → one node; its inverse restores state B (two nodes).
        let (_r, back_to_b) = Command::Checkpoint { yaml: state_a }.execute(&mut g).unwrap();
        assert_eq!(g.node_uids().len(), 1, "restored to the one-node snapshot");
        back_to_b.execute(&mut g).unwrap();
        assert_eq!(g.node_uids().len(), 2, "inverse restored the two-node state");
    }

    #[test]
    fn rename_global_round_trips() {
        let mut g = Graph::new();
        g.apply_global_change("subj", Some(GlobalValue::Int(7))).unwrap();

        let (_r, inverse) = Command::RenameGlobal { from: "subj".into(), to: "subject".into() }.execute(&mut g).unwrap();
        assert_eq!(g.globals().get("subj"), None);
        assert_eq!(g.globals().get("subject"), Some(&GlobalValue::Int(7)), "value moved to the new name");

        inverse.execute(&mut g).unwrap();
        assert_eq!(g.globals().get("subject"), None);
        assert_eq!(g.globals().get("subj"), Some(&GlobalValue::Int(7)), "inverse restores the old name + value");
    }

    // ── CommandHistory ───────────────────────────────────────────────────────────────────────────

    fn add_node(name: &str) -> Command {
        Command::AddNode { type_name: name.into(), pos: [0.0, 0.0], uid: None, name: None, params: None }
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

        // s1's undo touches only s1's node; s2's is untouched.
        assert!(h.undo(&mut g, "s1").unwrap());
        assert!(!g.contains(a) && g.contains(b), "only s1's node undone");
        assert!(!h.undo(&mut g, "s1").unwrap(), "s1 has nothing left to undo");

        // s2 undoes its own.
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

        // A fresh command invalidates the redo future.
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
}
