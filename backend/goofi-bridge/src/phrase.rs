//! The phrase layer: ONE parser from a command line to an op call, and one renderer from a
//! result to the text a caller reads. Every transport that speaks lines shares it — the MCP tool
//! and, next, `/exec` and the CLI — so the surfaces cannot drift.

use serde_json::{json, Map, Value};

use crate::ops::{self, Op};

/// A command line as bash would hand it to argv — same words, same quoting.
pub fn split(line: &str) -> Result<Vec<String>, String> {
    shell_words::split(line).map_err(|e| format!("{e}"))
}

/// The op a word sequence names: the FIRST complete registered phrase, walking left to right.
/// Prefix-freedom over the registry makes the first match the only possible one, so min-length
/// and longest-match coincide. Answers the op and how many words its phrase consumed.
pub fn resolve(words: &[String]) -> Result<(&'static Op, usize), String> {
    for n in 1..=words.len() {
        if let Some(op) = ops::find(&words[..n].join(" ")) {
            return Ok((op, n));
        }
    }
    let line = words.join(" ");
    let first = words.first().map(String::as_str).unwrap_or_default();
    let near: Vec<&str> =
        ops::REGISTRY.iter().map(|o| o.name).filter(|n| n.split(' ').next() == Some(first)).collect();
    match near.is_empty() {
        true => Err(format!("unknown op `{line}` — `op list` answers every op this server speaks")),
        false => Err(format!("unknown op `{line}` — under `{first}`: {}", near.join(", "))),
    }
}

/// One line, parsed against the registry: the phrase, then every argument as a flag the op's own
/// schema types. Answers the op and the payload the socket envelope would carry — one payload
/// shape, whichever surface spelled it.
pub fn parse(line: &str) -> Result<(&'static Op, Value), String> {
    let words = split(line)?;
    if words.is_empty() {
        return Err("empty command".into());
    }
    let (op, used) = resolve(&words)?;
    let payload = parse_flags(op, &words[used..])?;
    Ok((op, payload))
}

/// The arguments an op offers, for a refusal that teaches: leading positionals as `<name>`,
/// then `--name <type>`, `!` marking required.
fn usage(op: &Op) -> String {
    let spelled: Vec<String> = op
        .args()
        .enumerate()
        .map(|(i, (name, ty, req))| match (i < op.positional, ty, req) {
            (true, _, true) => format!("<{name}>"),
            (true, _, false) => format!("[{name}]"),
            (false, "bool", _) => format!("--[no-]{name}"),
            (false, ty, true) => format!("--{name} <{ty}>!"),
            (false, ty, false) => format!("--{name} <{ty}>"),
        })
        .collect();
    match spelled.is_empty() {
        true => format!("`{}` takes no arguments", op.name),
        false => format!("`{}` takes: {}", op.name, spelled.join(" ")),
    }
}

/// One raw flag value, typed by the schema. A list type `T[]` is typed by its item.
fn typed(op: &Op, key: &str, ty: &str, raw: String) -> Result<Value, String> {
    match ty {
        "int" => raw.parse::<i64>().map(Value::from).map_err(|_| {
            format!("{}: `--{key}` takes an integer, not `{raw}`", op.name)
        }),
        "float" => raw.parse::<f64>().map(Value::from).map_err(|_| {
            format!("{}: `--{key}` takes a number, not `{raw}`", op.name)
        }),
        "float2" => {
            let parsed = raw.split_once(',').and_then(|(x, y)| {
                Some(json!([x.trim().parse::<f64>().ok()?, y.trim().parse::<f64>().ok()?]))
            });
            parsed.ok_or_else(|| format!("{}: `--{key}` takes `x,y`, not `{raw}`", op.name))
        }
        "json" => serde_json::from_str(&raw)
            .map_err(|e| format!("{}: `--{key}` takes JSON — {e}", op.name)),
        // `any`: JSON when it parses, otherwise the bare string — `--value 2.5` and `--value hi`.
        "any" => Ok(serde_json::from_str(&raw).unwrap_or(Value::String(raw))),
        // uid, string, param_addr and the rest ride as the string they are.
        _ => Ok(Value::String(raw)),
    }
}

/// Flags → payload, mechanically from the args schema: a bool is `--x` / `--no-x` and sends its
/// key only when given; a list-typed arg repeats; every other flag's value is the NEXT word,
/// whatever it looks like, or inline via `--flag=value`.
fn parse_flags(op: &'static Op, words: &[String]) -> Result<Value, String> {
    let decls: Vec<(&str, &str, bool)> = op.args().collect();
    let mut payload = Map::new();
    let mut i = 0;
    // Leading bare words fill the op's positional args, in declaration order; a list-typed
    // positional is variadic and takes every bare word up to the first flag. Each positional
    // stays reachable as a flag too, so the sugar never hides a spelling.
    for (name, ty, _) in decls.iter().take(op.positional) {
        if words.get(i).is_none_or(|w| w.starts_with("--")) {
            break;
        }
        match ty.strip_suffix("[]") {
            Some(item) => {
                let mut list = Vec::new();
                while let Some(w) = words.get(i).filter(|w| !w.starts_with("--")) {
                    list.push(typed(op, name, item, w.clone())?);
                    i += 1;
                }
                payload.insert(name.to_string(), Value::Array(list));
            }
            None => {
                payload.insert(name.to_string(), typed(op, name, ty, words[i].clone())?);
                i += 1;
            }
        }
    }
    while i < words.len() {
        let word = &words[i];
        i += 1;
        let Some(flag) = word.strip_prefix("--") else {
            return Err(format!("{}: unexpected `{word}` — {}", op.name, usage(op)));
        };
        let (mut key, inline) = match flag.split_once('=') {
            Some((k, v)) => (k, Some(v.to_string())),
            None => (flag, None),
        };
        let negated = match key.strip_prefix("no-") {
            // `--no-x` only where `x` is a declared bool: an op's own `no-…` name stays reachable.
            Some(bare) if decls.iter().any(|(n, t, _)| *n == bare && *t == "bool") => {
                key = bare;
                true
            }
            _ => false,
        };
        let Some((name, ty, _)) = decls.iter().find(|(n, ..)| *n == key) else {
            return Err(format!("{}: no flag `--{key}` — {}", op.name, usage(op)));
        };
        if *ty == "bool" {
            if inline.is_some() {
                return Err(format!("{}: `--{key}` is a flag and takes no value", op.name));
            }
            payload.insert(name.to_string(), Value::Bool(!negated));
            continue;
        }
        let raw = match inline {
            Some(v) => v,
            None => {
                let Some(next) = words.get(i) else {
                    return Err(format!("{}: `--{key}` needs a value — {}", op.name, usage(op)));
                };
                i += 1;
                next.clone()
            }
        };
        match ty.strip_suffix("[]") {
            Some(item) => {
                let v = typed(op, key, item, raw)?;
                match payload.entry(name.to_string()).or_insert_with(|| json!([])) {
                    Value::Array(list) => list.push(v),
                    _ => unreachable!("a list flag only ever inserts an array"),
                }
            }
            None => {
                if payload.insert(name.to_string(), typed(op, key, ty, raw)?).is_some() {
                    return Err(format!("{}: `{key}` was given twice", op.name));
                }
            }
        }
    }
    for (name, _, required) in &decls {
        if *required && !payload.contains_key(*name) {
            return Err(format!("{}: `--{name}` is required — {}", op.name, usage(op)));
        }
    }
    Ok(Value::Object(payload))
}

/// An op's answer as the text a caller reads: prose and bare strings verbatim, everything else
/// pretty-printed.
pub fn render(result: &Value) -> String {
    if let Value::String(s) = result {
        return s.clone();
    }
    match result.as_object() {
        Some(o) if o.len() == 1 => match o.get("text").and_then(|t| t.as_str()) {
            Some(t) => t.to_string(),
            None => serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string()),
        },
        _ => serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string()),
    }
}
