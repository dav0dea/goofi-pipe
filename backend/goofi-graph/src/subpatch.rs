//! What a sub-patch IS, stated in the few things its nature forces. A facade and a boundary port
//! are ordinary node records in the graph's ONE map; membership is `scope_of`'s. What is left here
//! is the vocabulary: which way a port faces, what a boundary type is called, and the one slot.

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

/// The boundary types: a port, one per direction per dtype. This table is the only place a
/// type name and the `(dir, dtype)` behind it are related.
pub const BOUNDARY_TYPES: &[(&str, Dir, SlotType)] = &[
    ("InArray", Dir::In, SlotType::Array),
    ("InString", Dir::In, SlotType::String),
    ("InTable", Dir::In, SlotType::Table),
    ("InAudio", Dir::In, SlotType::Audio),
    ("OutArray", Dir::Out, SlotType::Array),
    ("OutString", Dir::Out, SlotType::String),
    ("OutTable", Dir::Out, SlotType::Table),
    ("OutAudio", Dir::Out, SlotType::Audio),
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

/// A boundary port's own nature: which way it faces and what it carries. Everything else about it
/// — its name, position, viewer state, membership — it wears as the node it is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Port {
    pub dir: Dir,
    /// The port's dtype, fixed by its type at birth: a port relays, so nothing re-types it.
    pub dtype: SlotType,
}

