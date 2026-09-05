//! `GraphDoc` — goofi's control-plane document, and the deltas that keep a browser replica equal
//! to it.
//!
//! A delta is an RFC 7386 merge patch, which is exact only while the document has no null leaf,
//! because merge patch spends `null` on "delete this key".

use serde_json::{Map, Value};

/// The roots, in the order a fresh document declares them.
const ROOTS: [&str; 4] = ["nodes", "links", "globals", "arrangement"];

/// The RFC 7386 merge patch that turns `before` into `after`, or `None` when they are equal.
pub fn merge_patch(before: &Value, after: &Value) -> Option<Value> {
    match (before, after) {
        (Value::Object(b), Value::Object(a)) => {
            let mut patch = Map::new();
            for (k, av) in a {
                match b.get(k) {
                    Some(bv) => {
                        if let Some(sub) = merge_patch(bv, av) {
                            patch.insert(k.clone(), sub);
                        }
                    }
                    None => {
                        patch.insert(k.clone(), av.clone());
                    }
                }
            }
            for k in b.keys() {
                if !a.contains_key(k) {
                    patch.insert(k.clone(), Value::Null);
                }
            }
            (!patch.is_empty()).then_some(Value::Object(patch))
        }
        _ => (before != after).then(|| after.clone()),
    }
}

/// Apply an RFC 7386 merge patch in place: `null` removes a key, an object merges into an object,
/// anything else replaces.
pub fn apply_merge(target: &mut Value, patch: &Value) {
    let Value::Object(p) = patch else {
        *target = patch.clone();
        return;
    };
    if !target.is_object() {
        *target = Value::Object(Map::new());
    }
    let Value::Object(t) = target else { unreachable!("just made it an object") };
    for (k, pv) in p {
        if pv.is_null() {
            t.shift_remove(k);
        } else {
            apply_merge(t.entry(k.clone()).or_insert(Value::Null), pv);
        }
    }
}

/// What a replica did with a patch.
#[derive(Debug, PartialEq, Eq)]
pub enum Patch {
    Applied,
    Stale,
    /// This replica is missing everything between `at` and `from`, and can only be re-seeded.
    Gap { from: u64, at: u64 },
}

/// The control-plane document, and the version every change advances.
pub struct GraphDoc {
    state: Value,
    version: u64,
}

impl GraphDoc {
    /// An empty document: the five roots present and empty, at version 0.
    pub fn new() -> GraphDoc {
        let mut state = Map::new();
        for root in ROOTS {
            state.insert(root.to_string(), if root == "links" { Value::Array(vec![]) } else { Value::Object(Map::new()) });
        }
        GraphDoc { state: Value::Object(state), version: 0 }
    }

    /// The whole document as plain JSON.
    pub fn to_json(&self) -> Value {
        self.state.clone()
    }

    /// The version this document is at.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Take the document to `target`, answering the patch that gets a replica there, or `None`
    /// when nothing changed; the version advances only on a real change.
    pub fn reconcile_root(&mut self, target: &Value) -> Option<Value> {
        let patch = merge_patch(&self.state, target)?;
        self.state = target.clone();
        self.version += 1;
        Some(patch)
    }

    /// Apply a patch a peer produced. A result already held is stale and skipped; one reaching
    /// forward of this replica is a gap and refused.
    pub fn apply_patch(&mut self, from: u64, to: u64, patch: &Value) -> Patch {
        if to <= self.version {
            return Patch::Stale;
        }
        if from != self.version {
            return Patch::Gap { from, at: self.version };
        }
        apply_merge(&mut self.state, patch);
        self.version = to;
        Patch::Applied
    }

    /// Adopt a peer's whole document.
    pub fn reset_to(&mut self, version: u64, state: Value) {
        self.state = state;
        self.version = version;
    }

    /// Read the JSON value at a path, or `None` if any segment is missing.
    pub fn read_at(&self, path: &[&str]) -> Option<Value> {
        let mut cur = &self.state;
        for seg in path {
            cur = cur.get(seg)?;
        }
        Some(cur.clone())
    }

    /// The keys of one root map.
    fn root_keys(&self, root: &str) -> Vec<String> {
        self.state
            .get(root)
            .and_then(Value::as_object)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn node_ids(&self) -> Vec<String> {
        self.root_keys("nodes")
    }

}

impl Default for GraphDoc {
    fn default() -> GraphDoc {
        GraphDoc::new()
    }
}
