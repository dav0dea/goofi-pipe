//! Flat sub-patch scopes — a purely organizational overlay over the flat node graph.
//!
//! Sub-patches are facades, not a separate runtime. Nodes live in one flat set (`Graph.nodes`);
//! a [`Scope`] references member uids (via the Graph's `scope_of` tree index) and holds In/Out
//! [`Stub`]s (boundary ports). The runtime is flat — `Graph.links` is ALWAYS leaf→leaf — so a stub
//! stores only its CHILD (inner) side; the parent side *is* the flat links (an Out stub fans out to
//! N consumers as N links, for free). [`resolve_stub`] walks a stub's `inner` chain-to-leaf (through
//! nested scopes) to the single physical leaf it exposes — the resolution the data plane and
//! link-authoring perform. No defs, no sharing, no materialize: nodes never move or re-mint, so
//! group/expand are pure reference moves and undo is uid-stable by construction.

use indexmap::IndexMap;

use crate::Uid;
use goofi_core::SlotType;

/// Stable boundary key inside a scope (`"in0"`, `"out0"`). Never re-minted on rename, so external
/// wires (which resolve through it to a flat leaf link) survive a relabel.
pub type StubId = String;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    In,
    Out,
}

impl Dir {
    pub fn name(self) -> &'static str {
        match self {
            Dir::In => "in",
            Dir::Out => "out",
        }
    }
}

/// A boundary port on a scope: a naming indirection over an inner member slot. Stores ONLY its
/// child side — `inner` is the concrete inner member+slot it exposes: a leaf `(leaf_uid, slot)`,
/// or a nested scope's `(facade_uid, StubId)` when the member is itself a sub-patch. `None` = an
/// unwired pill (a present-but-dangling port). The parent side is not stored here — it is the flat
/// leaf→leaf links that resolve through this stub.
#[derive(Clone, Debug, PartialEq)]
pub struct Stub {
    pub dir: Dir,
    /// The port's advertised dtype — the wired inner slot's type (provisional until wired).
    pub dtype: SlotType,
    /// `(inner member uid, inner slot-or-StubId)`; `None` = UNWIRED.
    pub inner: Option<(Uid, String)>,
    /// Pill position inside the entered view.
    pub pos: [f64; 2],
    /// Renameable display label (defaults `in0`/`out0`).
    pub name: String,
}

/// A sub-patch scope: organizational metadata for a set of member nodes. Membership + parentage
/// live in the Graph's `scope_of` index (the one tree SSOT — absent ⇒ ROOT); a `Scope` holds only
/// its own display name, the canvas position of its collapsed facade node, and its boundary stubs.
/// The scope's uid (its key in `Graph.scopes`) doubles as the facade node's uid.
#[derive(Clone, Debug, PartialEq)]
pub struct Scope {
    pub name: String,
    pub pos: [f64; 2],
    pub stubs: IndexMap<StubId, Stub>,
}

impl Scope {
    pub fn new(name: String, pos: [f64; 2]) -> Scope {
        Scope { name, pos, stubs: IndexMap::new() }
    }
}

/// Chain-to-leaf: resolve `(scope_uid, stub_id)` to the single physical inner leaf `(uid, slot)` it
/// exposes, walking through nested-scope stubs. `None` if the stub (or any stub in its chain) is
/// unwired, or the ids don't resolve. This is the resolution the data plane performs before
/// subscribing, and that link authoring uses to store a boundary wire as a flat leaf→leaf link.
pub fn resolve_stub(scopes: &IndexMap<Uid, Scope>, scope_uid: Uid, stub_id: &str) -> Option<(Uid, String)> {
    let scope = scopes.get(&scope_uid)?;
    let (inner_uid, inner_slot) = scope.stubs.get(stub_id)?.inner.as_ref()?;
    if scopes.contains_key(inner_uid) {
        // The inner member is itself a nested sub-patch — recurse through its exposing stub.
        resolve_stub(scopes, *inner_uid, inner_slot)
    } else {
        Some((*inner_uid, inner_slot.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out_stub(inner_uid: Uid, inner_slot: &str) -> Stub {
        Stub {
            dir: Dir::Out,
            dtype: SlotType::Array,
            inner: Some((inner_uid, inner_slot.to_string())),
            pos: [0.0, 0.0],
            name: "out0".to_string(),
        }
    }

    /// A scope (uid 10) exposing leaf 2's `out` slot as `out0`.
    fn one_scope() -> IndexMap<Uid, Scope> {
        let mut scopes = IndexMap::new();
        let mut s = Scope::new("subpatch0".to_string(), [0.0, 0.0]);
        s.stubs.insert("out0".to_string(), out_stub(Uid(2), "out"));
        scopes.insert(Uid(10), s);
        scopes
    }

    #[test]
    fn resolve_stub_reaches_the_inner_leaf() {
        let scopes = one_scope();
        assert_eq!(
            resolve_stub(&scopes, Uid(10), "out0"),
            Some((Uid(2), "out".to_string())),
            "output stub → leaf 2's out slot",
        );
        assert_eq!(resolve_stub(&scopes, Uid(10), "nope"), None, "unknown stub → None");
        assert_eq!(resolve_stub(&scopes, Uid(99), "out0"), None, "unknown scope → None");
    }

    #[test]
    fn resolve_stub_walks_two_deep_nesting() {
        // outer scope 20 exposes child scope 10's `out0` as its own `out0`; child 10 exposes leaf 2.
        let mut scopes = one_scope();
        let mut outer = Scope::new("outer0".to_string(), [0.0, 0.0]);
        outer.stubs.insert("out0".to_string(), out_stub(Uid(10), "out0"));
        scopes.insert(Uid(20), outer);
        assert_eq!(
            resolve_stub(&scopes, Uid(20), "out0"),
            Some((Uid(2), "out".to_string())),
            "chain walks through the nested scope to the deepest leaf",
        );
    }

    #[test]
    fn unwired_stub_resolves_to_none() {
        let mut scopes = one_scope();
        scopes.get_mut(&Uid(10)).unwrap().stubs.get_mut("out0").unwrap().inner = None;
        assert_eq!(resolve_stub(&scopes, Uid(10), "out0"), None);
    }
}
