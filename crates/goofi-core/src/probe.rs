//! The node-introspection probe schema: the JSON `goofi.introspect` emits and the discoverer
//! parses. Defined once (serde) so the producer (`goofi-pymod`) and consumer (`goofi-node`)
//! can't drift, and neither hand-rolls (de)serialization.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Introspection {
    pub gil_safe: bool,
    #[serde(default)]
    pub doc: String,
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
    #[serde(flatten)]
    pub spec: ParamSpec,
}

/// The kind-specific fields, tagged by `kind` (`{"kind":"int",…}`) — exhaustive, so neither
/// side has an unknown-kind path.
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
