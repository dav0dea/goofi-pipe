//! Patch-scoped **globals** — named typed scalars shared across a patch. See
//! `docs/superpowers/specs/2026-07-17-globals-panel-design.md`.
//!
//! - [`GlobalValue`] — a typed scalar (`float | int | bool | string`).
//! - [`GlobalsSnapshot`] — a cheap read-only view handed to node setup/process (`NodeCtx.globals`)
//!   and expression eval (`EvalCtx.globals`).
//! - [`GlobalStore`] — the engine's authoritative map with the system-set invariant (system globals
//!   are editable but never deletable/renamable; the store re-asserts them so a delete reappears).
//! - [`SYSTEM_GLOBALS`] — the code-owned system global definitions (v1: `default_ufreq = 30`).

use std::sync::Arc;

use indexmap::IndexMap;

/// A patch global's value — a typed scalar mirroring the scalar kinds of `Param` without its widget
/// metadata. The panel edits it; expressions and node setup/process read it.
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
    /// The wire/panel type tag — `"float" | "int" | "bool" | "string"`.
    pub fn type_tag(&self) -> &'static str {
        match self {
            GlobalValue::Float(_) => "float",
            GlobalValue::Int(_) => "int",
            GlobalValue::Bool(_) => "bool",
            GlobalValue::Str(_) => "string",
        }
    }
    /// A plain-text rendering, used when coercing an incoming value into a `Str`-typed slot.
    fn display_string(&self) -> String {
        match self {
            GlobalValue::Float(v) => v.to_string(),
            GlobalValue::Int(v) => v.to_string(),
            GlobalValue::Bool(v) => v.to_string(),
            GlobalValue::Str(s) => s.clone(),
        }
    }
    /// Coerce `self` to `template`'s variant — used when SETTING an existing global so its declared
    /// type stays stable (a `default_ufreq` Float stays Float even if the doc sends `30`/`"30"`).
    fn coerced_like(self, template: &GlobalValue) -> GlobalValue {
        match template {
            GlobalValue::Float(_) => GlobalValue::Float(self.as_f64().unwrap_or(0.0)),
            GlobalValue::Int(_) => GlobalValue::Int(self.as_i64().unwrap_or(0)),
            GlobalValue::Bool(_) => GlobalValue::Bool(self.as_bool().unwrap_or(false)),
            GlobalValue::Str(_) => GlobalValue::Str(self.display_string()),
        }
    }
}

/// A code-owned system global: identity + default + panel doc. System globals are editable but never
/// deletable or renamable; the engine re-asserts them so a delete reappears.
pub struct GlobalDef {
    pub name: &'static str,
    pub default: GlobalValue,
    pub doc: &'static str,
}

/// The v1 system global set — exactly `default_ufreq = 30` (the producer reference rate).
pub static SYSTEM_GLOBALS: &[GlobalDef] = &[GlobalDef {
    name: "default_ufreq",
    default: GlobalValue::Float(30.0),
    doc: "Default update rate (Hz) for producer nodes that have not overridden it.",
}];

/// Whether `name` is a legal global identifier: `[A-Za-z_][A-Za-z0-9_]*`, non-empty, and not the
/// reserved namespace token `globals` (so `globals.<name>` never shadows the namespace itself).
pub fn is_valid_global_name(name: &str) -> bool {
    if name == "globals" {
        return false;
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// An immutable, cheaply-cloned view of the patch globals, handed to node setup/process
/// (`NodeCtx.globals`) and expression eval (`EvalCtx.globals`). Backed by an `Arc` so passing it into
/// every tick's `NodeCtx` is a refcount bump, not a map clone.
#[derive(Clone, Debug, Default)]
pub struct GlobalsSnapshot {
    map: Arc<IndexMap<String, GlobalValue>>,
}

impl GlobalsSnapshot {
    pub fn new(map: IndexMap<String, GlobalValue>) -> GlobalsSnapshot {
        GlobalsSnapshot { map: Arc::new(map) }
    }
    // Typed scalar reads — the node-authoring surface (`ctx.globals.f64("default_ufreq")`), mirroring
    // the four `Params<'a>` readers. `iter()` is the eval-namespace escape hatch (expr injection).
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

/// The engine's authoritative globals map + the system-set invariant. System globals are seeded on
/// construction (and re-asserted after a load); they may be edited but never removed. User globals
/// may be added, edited, and removed. (Rename is not a store op — it arrives as a remove + add
/// through the CRDT diff, since a CRDT map can't rename a key in place.) Insertion order (system
/// first, then user in creation order) is preserved for a stable panel.
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
    /// A fresh store with the system globals seeded to their defaults.
    pub fn new() -> GlobalStore {
        let mut s = GlobalStore { values: IndexMap::new(), system: std::collections::HashSet::new() };
        s.reassert_system();
        s
    }

    /// Back-fill any missing system global with its default and mark it system — called on
    /// construction and after a `.gfi` load, so system globals are always present (an older patch
    /// missing one, or a system global added since, is filled) and can never be orphaned.
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
    /// A read-only snapshot for eval / node context.
    pub fn snapshot(&self) -> GlobalsSnapshot {
        GlobalsSnapshot::new(self.values.clone())
    }

    /// Every global in order, tagged with whether it is a system global — for mirroring + `.gfi`.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &GlobalValue, bool)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v, self.system.contains(k)))
    }

    /// Set an EXISTING global's value, coercing to its declared type so system/user types stay
    /// stable. Errors if the global does not exist (use [`Self::add`] for a new one).
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

    /// Add a NEW user global. Errors on an invalid name or a collision (system or user).
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

    /// Current ordered position of `name`, if present. The order is observable (it feeds the `.gfi`,
    /// the CRDT mirror, and the Globals panel), so a delete's inverse captures this to re-add the
    /// global at its original slot rather than the tail.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.values.get_index_of(name)
    }

    /// Add a NEW user global at ordered position `at` (clamped to the current length) — the
    /// position-preserving re-add used by a delete/rename undo. Same validation as [`Self::add`].
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

    /// Remove a USER global. Errors if the global is system (protected) or absent.
    pub fn remove(&mut self, name: &str) -> Result<(), String> {
        if self.system.contains(name) {
            return Err(format!("cannot delete system global `{name}`"));
        }
        if self.values.shift_remove(name).is_none() {
            return Err(format!("no such global `{name}`"));
        }
        Ok(())
    }

    /// Apply one mirrored client change: `Some(v)` sets an existing global or adds a new user one;
    /// `None` removes (rejected for system globals). The single entry point the manager calls per
    /// changed doc leaf, so add/edit/delete/rename (delete+add) all route through here uniformly.
    pub fn apply_change(&mut self, name: &str, value: Option<GlobalValue>) -> Result<(), String> {
        match value {
            Some(v) if self.values.contains_key(name) => self.set(name, v),
            Some(v) => self.add(name, v),
            None => self.remove(name),
        }
    }
}
