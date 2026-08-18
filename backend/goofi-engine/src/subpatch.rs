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

/// Chain-to-leaf: resolve `(scope_uid, stub_id)` to the single physical inner leaf `(uid, slot)` it
/// exposes, walking through nested-scope stubs. `None` if the stub (or any stub in its chain) is
/// unwired, or the ids don't resolve. This is the resolution the data plane performs before
/// subscribing, and that link authoring uses to store a boundary wire as a flat leaf→leaf link.
///
/// The walk carries a visited set because it CANNOT assume the chain is acyclic: a `.gfi` is
/// hand-editable and `reload_scopes` admits whatever stub graph it finds, so a stub pointing back
/// into its own chain is representable persisted state. Recursing on it does not raise a
/// recoverable panic — a Rust stack overflow aborts the process, taking every node thread and every
/// connected client with it, and `/data/<scope>/<stub>` reaches here from an ordinary subscribe.
/// A cycle is malformed, so it answers `None`, the same answer an unknown stub already gives.
pub fn resolve_stub(scopes: &IndexMap<Uid, Scope>, scope_uid: Uid, stub_id: &str) -> Option<(Uid, String)> {
    let mut seen: Vec<(Uid, &str)> = Vec::new();
    let (mut scope_uid, mut stub_id) = (scope_uid, stub_id);
    loop {
        if seen.contains(&(scope_uid, stub_id)) {
            return None;
        }
        seen.push((scope_uid, stub_id));
        let (inner_uid, inner_slot) = scopes.get(&scope_uid)?.stubs.get(stub_id)?.inner.as_ref()?;
        if !scopes.contains_key(inner_uid) {
            return Some((*inner_uid, inner_slot.clone()));
        }
        // The inner member is itself a nested sub-patch — continue through its exposing stub.
        scope_uid = *inner_uid;
        stub_id = inner_slot.as_str();
    }
}

