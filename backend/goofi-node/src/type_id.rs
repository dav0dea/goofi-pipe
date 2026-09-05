//! The `engine:Name` type id. A structural type has no engine and stays bare.

pub const SEP: char = ':';

pub fn qualify(engine: &str, name: &str) -> String {
    format!("{engine}{SEP}{name}")
}

/// `(Some(engine), name)` for a qualified id, `(None, id)` for a bare one.
pub fn split(id: &str) -> (Option<&str>, &str) {
    match id.split_once(SEP) {
        Some((engine, name)) => (Some(engine), name),
        None => (None, id),
    }
}

pub fn bare(id: &str) -> &str {
    split(id).1
}
