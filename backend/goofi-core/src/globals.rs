//! Patch-scoped globals — named typed scalars shared across a patch.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A patch global's value — a typed scalar. The serde shape is the `{type, value}` of the `.gfi`
/// and the doc: the tag is what preserves float-vs-int through JSON's whole-float normalization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "lowercase")]
pub enum GlobalValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    #[serde(rename = "string")]
    Str(String),
}

impl GlobalValue {
    /// Coerce to `template`'s variant, so an existing global's declared type stays stable on set.
    fn coerced_like(self, template: &GlobalValue) -> GlobalValue {
        use GlobalValue as G;
        match (template, self) {
            (G::Float(_), G::Int(v)) => G::Float(v as f64),
            (G::Float(_), G::Bool(v)) => G::Float(if v { 1.0 } else { 0.0 }),
            (G::Float(_), G::Str(_)) => G::Float(0.0),
            (G::Int(_), G::Float(v)) => G::Int(v.round() as i64),
            (G::Int(_), G::Bool(v)) => G::Int(v.into()),
            (G::Int(_), G::Str(_)) => G::Int(0),
            (G::Bool(_), G::Float(_) | G::Int(_) | G::Str(_)) => G::Bool(false),
            (G::Str(_), G::Float(v)) => G::Str(v.to_string()),
            (G::Str(_), G::Int(v)) => G::Str(v.to_string()),
            (G::Str(_), G::Bool(v)) => G::Str(v.to_string()),
            (_, same) => same,
        }
    }
}

/// A code-owned system global: never deletable or renamable. A LOCKED one is not editable
/// either — its value is the machine's, re-derived at every reassert, and a `.gfi` never
/// carries it.
pub struct GlobalDef {
    pub name: &'static str,
    pub value: fn() -> GlobalValue,
    pub doc: &'static str,
    pub locked: bool,
}

pub static SYSTEM_GLOBALS: &[GlobalDef] = &[
    GlobalDef {
        name: "default_ufreq",
        value: || GlobalValue::Float(30.0),
        doc: "Default update rate (Hz) for producer nodes that have not overridden it.",
        locked: false,
    },
    GlobalDef {
        name: "goofi_home",
        value: || GlobalValue::Str(crate::path::to_slash(&crate::home::dir())),
        doc: "The .goofi folder, where goofi keeps its own files. The machine says where it is.",
        locked: true,
    },
];

/// Python's keywords, plus goofi's own namespace token `globals`. A regex reads each as an
/// identifier and a parser does not, so a name that is one cannot be an attribute — which is the
/// position every name here is read in.
const RESERVED: &[&str] = &[
    "globals", "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
    "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if",
    "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
];

/// A legal name in the ONE expression namespace: `[A-Za-z_][A-Za-z0-9_]*` and not reserved.
///
/// Every name an expression can spell is held to this, because an expression reads one as an
/// ATTRIBUTE — `globals.gain`, and a sub-patch's slot in `nd('chain').drain`. A name Python cannot
/// parse there breaks every reference to it and takes the rewrite with it: the next rename has no
/// `nd('<old>')` left to follow, so the damage cannot be undone by renaming back.
pub fn is_valid_identifier(name: &str) -> bool {
    if RESERVED.contains(&name) {
        return false;
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// What a node or slot name has to be, said once — it is the tail of every refusal about one.
pub const NAME_RULE: &str =
    "a letter then letters or digits, and not a Python keyword — an expression reads a name as an attribute, and a reference spells `node.slot`";

/// A legal node or slot name: `[A-Za-z][A-Za-z0-9]*` and not reserved. Narrower than a global's
/// identifier so that `node.slot` needs no quoting anywhere it is spelled.
pub fn is_valid_name(name: &str) -> bool {
    if RESERVED.contains(&name) {
        return false;
    }
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic()) && chars.all(|c| c.is_ascii_alphanumeric())
}

/// The authoritative globals map. System globals may be edited but never removed, and the
/// insertion order is observable (the panel, the `.gfi` and the mirror all read it).
#[derive(Clone)]
pub struct GlobalStore {
    values: IndexMap<String, GlobalValue>,
    system: std::collections::HashSet<String>,
    locked: std::collections::HashSet<String>,
}

impl Default for GlobalStore {
    fn default() -> GlobalStore {
        GlobalStore::new()
    }
}

impl GlobalStore {
    pub fn new() -> GlobalStore {
        let mut s = GlobalStore {
            values: IndexMap::new(),
            system: std::collections::HashSet::new(),
            locked: std::collections::HashSet::new(),
        };
        s.reassert_system();
        s
    }

    /// Back-fill any missing system global with its default — on construction and after a load.
    /// A LOCKED one is overwritten instead: its value is this machine's, never a file's.
    pub fn reassert_system(&mut self) {
        for def in SYSTEM_GLOBALS {
            if def.locked {
                self.values.insert(def.name.to_string(), (def.value)());
                self.locked.insert(def.name.to_string());
            } else {
                self.values.entry(def.name.to_string()).or_insert_with(def.value);
            }
            self.system.insert(def.name.to_string());
        }
    }

    pub fn get(&self, name: &str) -> Option<&GlobalValue> {
        self.values.get(name)
    }
    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// Every global in order, tagged with whether it is a system global and whether it is locked.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &GlobalValue, bool, bool)> {
        self.values
            .iter()
            .map(|(k, v)| (k.as_str(), v, self.system.contains(k), self.locked.contains(k)))
    }

    /// Set an EXISTING global, coercing to its declared type; errors when it does not exist.
    pub fn set(&mut self, name: &str, value: GlobalValue) -> Result<(), String> {
        if self.locked.contains(name) {
            return Err(format!("global `{name}` is read-only: its value is the machine's"));
        }
        match self.values.get(name) {
            Some(existing) => {
                let coerced = value.coerced_like(existing);
                self.values.insert(name.to_string(), coerced);
                Ok(())
            }
            None => Err(format!("no such global `{name}`")),
        }
    }

    /// Add a NEW user global, at ordered position `at` (clamped) when given — the re-add a
    /// delete/rename undo needs. Errors on an invalid name or a collision.
    pub fn add(&mut self, name: &str, value: GlobalValue, at: Option<usize>) -> Result<(), String> {
        if !is_valid_identifier(name) {
            return Err(format!("invalid global name `{name}`"));
        }
        if self.values.contains_key(name) {
            return Err(format!("global `{name}` already exists"));
        }
        let at = at.unwrap_or(usize::MAX).min(self.values.len());
        self.values.shift_insert(at, name.to_string(), value);
        Ok(())
    }

    /// Ordered position of `name` — a delete's inverse captures it to re-add at the original slot.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.values.get_index_of(name)
    }

    /// Remove a USER global; errors when it is a system global or absent.
    pub fn remove(&mut self, name: &str) -> Result<(), String> {
        if self.system.contains(name) {
            return Err(format!("cannot delete system global `{name}`"));
        }
        if self.values.shift_remove(name).is_none() {
            return Err(format!("no such global `{name}`"));
        }
        Ok(())
    }

    /// Apply one change: `Some(v)` sets or adds (a NEW global lands at `at`), `None` removes.
    pub fn apply_change(
        &mut self,
        name: &str,
        value: Option<GlobalValue>,
        at: Option<usize>,
    ) -> Result<(), String> {
        match value {
            Some(v) if self.values.contains_key(name) => self.set(name, v),
            Some(v) => self.add(name, v, at),
            None => self.remove(name),
        }
    }
}
