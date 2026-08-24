//! Patch-scoped globals — named typed scalars shared across a patch.

use std::sync::Arc;

use indexmap::IndexMap;

/// A patch global's value — a typed scalar.
#[derive(Clone, Debug, PartialEq)]
pub enum GlobalValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
}

impl GlobalValue {
    /// Numeric view (`Int`/`Bool` widen; `Str` is `None`).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GlobalValue::Float(v) => Some(*v),
            GlobalValue::Int(v) => Some(*v as f64),
            GlobalValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            GlobalValue::Str(_) => None,
        }
    }
    /// Integer view (`Float` rounds to nearest; `Str` is `None`).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            GlobalValue::Int(v) => Some(*v),
            GlobalValue::Float(v) => Some(v.round() as i64),
            GlobalValue::Bool(v) => Some(*v as i64),
            GlobalValue::Str(_) => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GlobalValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GlobalValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn type_tag(&self) -> &'static str {
        match self {
            GlobalValue::Float(_) => "float",
            GlobalValue::Int(_) => "int",
            GlobalValue::Bool(_) => "bool",
            GlobalValue::Str(_) => "string",
        }
    }
    fn display_string(&self) -> String {
        match self {
            GlobalValue::Float(v) => v.to_string(),
            GlobalValue::Int(v) => v.to_string(),
            GlobalValue::Bool(v) => v.to_string(),
            GlobalValue::Str(s) => s.clone(),
        }
    }
    /// Coerce to `template`'s variant, so an existing global's declared type stays stable on set.
    fn coerced_like(self, template: &GlobalValue) -> GlobalValue {
        match template {
            GlobalValue::Float(_) => GlobalValue::Float(self.as_f64().unwrap_or(0.0)),
            GlobalValue::Int(_) => GlobalValue::Int(self.as_i64().unwrap_or(0)),
            GlobalValue::Bool(_) => GlobalValue::Bool(self.as_bool().unwrap_or(false)),
            GlobalValue::Str(_) => GlobalValue::Str(self.display_string()),
        }
    }
}

/// A code-owned system global: editable, but never deletable or renamable.
pub struct GlobalDef {
    pub name: &'static str,
    pub default: GlobalValue,
    pub doc: &'static str,
}

pub static SYSTEM_GLOBALS: &[GlobalDef] = &[GlobalDef {
    name: "default_ufreq",
    default: GlobalValue::Float(30.0),
    doc: "Default update rate (Hz) for producer nodes that have not overridden it.",
}];

/// Python's keywords. A regex reads one as an identifier and a parser does not, so a name that is
/// one cannot be an attribute — which is the position both namespaces are read in.
const KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

/// A legal Python identifier: `[A-Za-z_][A-Za-z0-9_]*` and not a keyword.
///
/// Every name an expression can spell is held to this, because an expression reads one as an
/// ATTRIBUTE — `globals.gain`, and a sub-patch's slot in `nd('chain').drain`. A name Python cannot
/// parse there breaks every reference to it and takes the rewrite with it: the next rename has no
/// `nd('<old>')` left to follow, so the damage cannot be undone by renaming back.
pub fn is_valid_identifier(name: &str) -> bool {
    if KEYWORDS.contains(&name) {
        return false;
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// …and never the reserved namespace token `globals`, which is the one extra a global carries.
pub fn is_valid_global_name(name: &str) -> bool {
    name != "globals" && is_valid_identifier(name)
}

/// An immutable, cheaply-cloned view of the patch globals, for node setup/process and eval.
#[derive(Clone, Debug, Default)]
pub struct GlobalsSnapshot {
    map: Arc<IndexMap<String, GlobalValue>>,
}

impl GlobalsSnapshot {
    pub fn new(map: IndexMap<String, GlobalValue>) -> GlobalsSnapshot {
        GlobalsSnapshot { map: Arc::new(map) }
    }
    pub fn f64(&self, name: &str) -> Option<f64> {
        self.map.get(name)?.as_f64()
    }
    pub fn i64(&self, name: &str) -> Option<i64> {
        self.map.get(name)?.as_i64()
    }
    pub fn bool(&self, name: &str) -> Option<bool> {
        self.map.get(name)?.as_bool()
    }
    pub fn str(&self, name: &str) -> Option<&str> {
        self.map.get(name)?.as_str()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &GlobalValue)> {
        self.map.iter()
    }
}

/// The authoritative globals map. System globals may be edited but never removed, and the
/// insertion order is observable (the panel, the `.gfi` and the mirror all read it).
#[derive(Clone)]
pub struct GlobalStore {
    values: IndexMap<String, GlobalValue>,
    system: std::collections::HashSet<String>,
}

impl Default for GlobalStore {
    fn default() -> GlobalStore {
        GlobalStore::new()
    }
}

impl GlobalStore {
    pub fn new() -> GlobalStore {
        let mut s = GlobalStore { values: IndexMap::new(), system: std::collections::HashSet::new() };
        s.reassert_system();
        s
    }

    /// Back-fill any missing system global with its default — on construction and after a load.
    pub fn reassert_system(&mut self) {
        for def in SYSTEM_GLOBALS {
            self.values.entry(def.name.to_string()).or_insert_with(|| def.default.clone());
            self.system.insert(def.name.to_string());
        }
    }

    pub fn is_system(&self, name: &str) -> bool {
        self.system.contains(name)
    }
    pub fn get(&self, name: &str) -> Option<&GlobalValue> {
        self.values.get(name)
    }
    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }
    pub fn snapshot(&self) -> GlobalsSnapshot {
        GlobalsSnapshot::new(self.values.clone())
    }

    /// Every global in order, tagged with whether it is a system global.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &GlobalValue, bool)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v, self.system.contains(k)))
    }

    /// Set an EXISTING global, coercing to its declared type; errors when it does not exist.
    pub fn set(&mut self, name: &str, value: GlobalValue) -> Result<(), String> {
        match self.values.get(name) {
            Some(existing) => {
                let coerced = value.coerced_like(existing);
                self.values.insert(name.to_string(), coerced);
                Ok(())
            }
            None => Err(format!("no such global `{name}`")),
        }
    }

    /// Add a NEW user global; errors on an invalid name or a collision.
    pub fn add(&mut self, name: &str, value: GlobalValue) -> Result<(), String> {
        if !is_valid_global_name(name) {
            return Err(format!("invalid global name `{name}`"));
        }
        if self.values.contains_key(name) {
            return Err(format!("global `{name}` already exists"));
        }
        self.values.insert(name.to_string(), value);
        Ok(())
    }

    /// Ordered position of `name` — a delete's inverse captures it to re-add at the original slot.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.values.get_index_of(name)
    }

    /// Add a NEW user global at position `at` (clamped) — the re-add a delete/rename undo needs.
    pub fn add_at(&mut self, name: &str, value: GlobalValue, at: usize) -> Result<(), String> {
        if !is_valid_global_name(name) {
            return Err(format!("invalid global name `{name}`"));
        }
        if self.values.contains_key(name) {
            return Err(format!("global `{name}` already exists"));
        }
        self.values.shift_insert(at.min(self.values.len()), name.to_string(), value);
        Ok(())
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

    /// Apply one mirrored client change: `Some(v)` sets or adds, `None` removes.
    pub fn apply_change(&mut self, name: &str, value: Option<GlobalValue>) -> Result<(), String> {
        match value {
            Some(v) if self.values.contains_key(name) => self.set(name, v),
            Some(v) => self.add(name, v),
            None => self.remove(name),
        }
    }
}
