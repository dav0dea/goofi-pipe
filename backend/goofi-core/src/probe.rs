//! The node-introspection probe schema — defined once, so producer and consumer cannot drift.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Introspection {
    pub gil_safe: bool,
    #[serde(default)]
    pub doc: String,
    /// The palette category the node declares; absent, the scan's default for its folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Whether the node makes frames on its own schedule rather than in answer to an input.
    /// `serde(default)`: an older wheel emits no key, and a parse failure greys out every node.
    #[serde(default)]
    pub producer: bool,
    pub inputs: Vec<Slot>,
    pub outputs: Vec<OutSlot>,
    pub params: Vec<Param>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slot {
    pub name: String,
    pub kind: String,
    pub trigger: bool,
    pub multi: bool,
    /// Whether the engine refuses to tick the node while this slot's last-store is empty.
    /// `serde(default)`: an older wheel emits no key, and a parse failure greys out every node.
    #[serde(default)]
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutSlot {
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub group: String,
    pub name: String,
    /// Help text for the UI tooltip.
    /// `serde(default)`: an older wheel emits no key, and a parse failure greys out every node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    /// A default expression binding, live from birth — `me.params.…`, `globals.…` and the rest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    #[serde(flatten)]
    pub spec: ParamSpec,
}

/// The kind-specific fields, tagged by `kind` — exhaustive, so neither side has an unknown kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ParamSpec {
    Int { default: i64, min: i64, max: i64 },
    Float { default: f64, min: f64, max: f64 },
    Bool { default: bool },
    Str {
        default: String,
        #[serde(default)]
        options: Vec<String>,
        #[serde(default)]
        refresh: bool,
    },
}
