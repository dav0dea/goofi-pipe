//! Flat sub-patch scopes — a purely organizational overlay over the flat node graph.
//! Nodes live in one flat set; a scope references member uids and holds boundary stubs.

use indexmap::IndexMap;

use crate::Uid;
use goofi_core::SlotType;

/// What a stub points at: `(inner member uid, inner slot)`. `None` = UNWIRED. On a nested scope
/// member the slot names that scope's own stub, spelled as its uid hex.
pub type StubInner = Option<(Uid, String)>;

/// One parent-scope stub and where it pointed — `(parent scope, stub, inner)`.
pub type ParentStub = (Uid, Uid, StubInner);

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

/// The type name a sub-patch facade wears. Not in the palette — `group_nodes` is what makes one.
pub const SCOPE_TYPE: &str = "SubPatch";

/// The one slot a boundary port carries. An In port FEEDS a member, so it wears an output; an Out
/// port drains one.
pub const BOUNDARY_SLOT: &str = "value";

/// The six boundary types: a port, one per direction per dtype. This table is the only place a
/// type name and the `(dir, dtype)` behind it are related.
pub const BOUNDARY_TYPES: &[(&str, Dir, SlotType)] = &[
    ("InArray", Dir::In, SlotType::Array),
    ("InString", Dir::In, SlotType::String),
    ("InTable", Dir::In, SlotType::Table),
    ("OutArray", Dir::Out, SlotType::Array),
    ("OutString", Dir::Out, SlotType::String),
    ("OutTable", Dir::Out, SlotType::Table),
];

/// The `(dir, dtype)` a boundary type name stands for, or `None` for any other type.
pub fn boundary_type(name: &str) -> Option<(Dir, SlotType)> {
    BOUNDARY_TYPES.iter().find(|(n, _, _)| *n == name).map(|(_, d, t)| (*d, *t))
}

/// The boundary type name a live port wears — the inverse of [`boundary_type`].
pub fn boundary_type_name(dir: Dir, dtype: SlotType) -> &'static str {
    BOUNDARY_TYPES
        .iter()
        .find(|(_, d, t)| *d == dir && *t == dtype)
        .map(|(n, _, _)| *n)
        .expect("the table covers every dir/dtype pair")
}

/// A boundary port on a scope: a naming indirection over an inner member slot, child side only.
/// It is addressed by a `Uid` from the graph's one counter, so an op names it exactly as it names
/// a node, and the facade slot it backs is spelled by that uid's hex.
#[derive(Clone, Debug, PartialEq)]
pub struct Stub {
    pub dir: Dir,
    /// The port's advertised dtype — the wired inner slot's type (provisional until wired).
    pub dtype: SlotType,
    pub pos: [f64; 2],
    pub name: String,
    /// Per-slot viewer view-state, as a node carries it. An IN port wears an output, so it is the
    /// one a viewer can attach to; an OUT port's stays empty.
    pub viewers: serde_json::Value,
}

impl Stub {
    pub fn new(dir: Dir, dtype: SlotType, pos: [f64; 2], name: String) -> Self {
        Stub { dir, dtype, pos, name, viewers: serde_json::json!({}) }
    }
}

/// A sub-patch scope: its display name, facade position, boundary stubs and the facade's own
/// per-slot viewer state. Membership lives in the Graph's `scope_of` index.
#[derive(Clone, Debug, PartialEq)]
pub struct Scope {
    pub name: String,
    pub pos: [f64; 2],
    pub stubs: IndexMap<Uid, Stub>,
    /// Per-slot viewer view-state, as a node and a port carry it — the facade draws its OUT ports
    /// as output slots, so each of them is a slot a viewer attaches to.
    pub viewers: serde_json::Value,
}

impl Scope {
    pub fn new(name: String, pos: [f64; 2], stubs: IndexMap<Uid, Stub>) -> Self {
        Scope { name, pos, stubs, viewers: serde_json::json!({}) }
    }
}
