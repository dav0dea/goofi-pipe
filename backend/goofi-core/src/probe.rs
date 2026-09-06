//! The node-introspection probe schema — defined once, so producer and consumer cannot drift.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Introspection {
    /// `serde(default)`: a node file whose language has no GIL — a `.wgsl` — says nothing.
    #[serde(default)]
    pub gil_safe: bool,
    #[serde(default)]
    pub doc: String,
    /// The palette tags the node declares, from the one closed vocabulary.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Whether the node makes frames on its own schedule rather than in answer to an input.
    /// `serde(default)`: an older wheel emits no key, and a parse failure greys out every node.
    #[serde(default)]
    pub producer: bool,
    /// Whether the node reads its input as the PREVIOUS tick left it: the one kind of node a
    /// loop may close through. `serde(default)`: only a scheduled engine asks.
    #[serde(default)]
    pub feedback: bool,
    #[serde(default)]
    pub inputs: Vec<Slot>,
    #[serde(default)]
    pub outputs: Vec<OutSlot>,
    #[serde(default)]
    pub params: Vec<Param>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slot {
    pub name: String,
    pub kind: String,
    /// `serde(default)`: the signal-plane flags a scheduled engine's node file never states.
    #[serde(default)]
    pub trigger: bool,
    #[serde(default)]
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
    Pulse {},
}
