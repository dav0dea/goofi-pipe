//! The phrase layer: ONE parser from a command line to an op call, and one renderer from a
//! result to the text a caller reads. Every transport that speaks lines shares it — the MCP tool
//! and, next, `/exec` and the CLI — so the surfaces cannot drift.

use serde_json::{json, Map, Value};

use crate::ops::{Entry, Op, TREE};

/// A command line as bash would hand it to argv — same words, same quoting.
pub fn split(line: &str) -> Result<Vec<String>, String> {
    shell_words::split(line).map_err(|e| format!("{e}"))
}

/// Where a word-by-word descent of the phrase tree ended — the one state machine under
/// [`resolve`] and [`complete`].
enum Stop {
    /// The words named a leaf: its full phrase, and how many words it took.
    Op(String, usize),
    /// The words ran out inside a group: what can come next, and the phrase so far.
    Children(&'static [Entry], String),
    /// A word matched nothing at its level; the phrase walked before the miss.
    Unknown(String),
}

fn walk(words: &[String]) -> Stop {
    let mut children = TREE;
    let mut prefix = String::new();
    for (i, word) in words.iter().enumerate() {
        match children.iter().find(|e| word_of(e) == word) {
            Some(Entry::Group(w, _, kids)) => {
                prefix.push_str(w);
                prefix.push(' ');
                children = kids;
            }
            Some(Entry::Leaf(op)) => return Stop::Op(format!("{prefix}{}", op.name), i + 1),
            None => return Stop::Unknown(prefix),
        }
    }
    Stop::Children(children, prefix)
}

/// The op a word sequence names, against the rows THIS server serves. Answers the op and how many
/// words its phrase consumed. A refusal teaches: the served phrases under the words that DID
/// match — and a phrase the tree spells that no row serves is the headless mode by construction,
/// since layout is the one subtree a server withholds.
pub fn resolve<'a>(ops: &[&'a Op], words: &[String]) -> Result<(&'a Op, usize), String> {
    let line = words.join(" ");
    let prefix = match walk(words) {
        Stop::Op(name, used) => match ops.iter().find(|o| o.name == name) {
            Some(op) => return Ok((op, used)),
            None => name,
        },
        Stop::Children(_, prefix) | Stop::Unknown(prefix) => prefix,
    };
    let near: Vec<&str> = match prefix.is_empty() {
        true => Vec::new(),
        false => ops.iter().map(|o| o.name).filter(|n| n.starts_with(&prefix)).collect(),
    };
    Err(match (near.is_empty(), prefix.is_empty()) {
        // The tree spells the phrase and no row serves it: the withheld layout subtree.
        (true, false) => {
            format!("unknown op `{line}` — this server is headless, and the layout ops are not served")
        }
        (true, true) => {
            format!("unknown op `{line}` — `op list` answers every op this server speaks")
        }
        _ => format!("unknown op `{line}` — under `{}`: {}", prefix.trim_end(), near.join(", ")),
    })
}

/// One line, parsed against the registry: the phrase, then every argument as a flag the op's own
/// schema types. Answers the op and the payload the socket envelope would carry — one payload
/// shape, whichever surface spelled it.
pub fn parse<'a>(ops: &[&'a Op], line: &str) -> Result<(&'a Op, Value), String> {
    let words = split(line)?;
    if words.is_empty() {
        return Err("empty command".into());
    }
    let (op, used) = resolve(ops, &words)?;
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
fn parse_flags(op: &Op, words: &[String]) -> Result<Value, String> {
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
/// pretty-printed. A result carrying NPY is DATA, not text — the text form points at the pipe,
/// and only the CLI's own renderer writes the bytes.
pub fn render(result: &Value) -> String {
    if let Value::String(s) = result {
        return s.clone();
    }
    if let Some(b64) = result.get("npy_b64").and_then(Value::as_str) {
        return format!(
            "an ARRAY frame as NPY — {} base64 bytes in `npy_b64`, beside its `meta`",
            b64.len()
        );
    }
    match result.as_object() {
        Some(o) if o.len() == 1 => match o.get("text").and_then(|t| t.as_str()) {
            Some(t) => t.to_string(),
            None => serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string()),
        },
        _ => serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string()),
    }
}

/// One or several command lines, run as every line transport runs them — `goofi_exec`, `/exec`
/// and `goofi -` share this door. A single line answers help when it asks for it, else executes
/// directly; several lines run as ONE batch. A parse refusal names its command index.
pub fn exec_lines(
    state: &crate::AppState,
    lines: &[String],
    actor: &str,
) -> Result<Vec<Value>, String> {
    if lines.is_empty() {
        return Err("`commands` is a non-empty list of command lines".into());
    }
    if let [line] = lines {
        let words = split(line).map_err(|e| format!("command 0: {e}"))?;
        if let Some(text) = help(state.ops(), &words) {
            return Ok(vec![json!({ "text": text })]);
        }
        let (op, payload) = parse(state.ops(), line).map_err(|e| format!("command 0: {e}"))?;
        return state.call(op.name, payload, actor).map(|r| vec![r]);
    }
    let mut steps = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let (op, payload) =
            parse(state.ops(), line).map_err(|e| format!("command {i}: {e}"))?;
        steps.push(json!({ "op": op.name, "payload": payload }));
    }
    match state.call("compound", json!({ "ops": steps }), actor)? {
        Value::Array(results) => Ok(results),
        other => Ok(vec![other]),
    }
}

/// The help door: `help [words…]` or `<words…> --help`. `None` when the line is not asking.
pub fn help(ops: &[&Op], words: &[String]) -> Option<String> {
    let target: Vec<String> = match words.first().map(String::as_str) {
        Some("help") => words[1..].to_vec(),
        _ if words.iter().any(|w| w == "--help") => {
            words.iter().filter(|w| *w != "--help").cloned().collect()
        }
        _ => return None,
    };
    if target.is_empty() {
        return Some(top_help(ops));
    }
    if let Ok((op, _)) = resolve(ops, &target) {
        return Some(op_help(op));
    }
    let first = target[0].as_str();
    let near: Vec<String> = ops
        .iter()
        .filter(|o| o.name.split(' ').next() == Some(first))
        .map(|o| format!("  {}", usage(o)))
        .chain(
            crate::ops::RESERVED
                .iter()
                .filter(|r| r.split(' ').next() == Some(first) && r.contains(' '))
                .map(|r| format!("  `{r}` — the client's own door")),
        )
        .collect();
    match near.is_empty() {
        true => Some(format!("nothing under `{}` — {}", target.join(" "), top_help(ops))),
        false => Some(format!("under `{first}`:\n{}", near.join("\n"))),
    }
}

fn top_help(ops: &[&Op]) -> String {
    let mut groups: Vec<String> = Vec::new();
    let mut bare: Vec<&str> = Vec::new();
    for e in TREE {
        match e {
            Entry::Group(word, doc, _) if served(ops, "", e) => {
                groups.push(format!("  {word} — {doc}"));
            }
            Entry::Group(..) => {}
            Entry::Leaf(op) => bare.push(op.name),
        }
    }
    format!(
        "goofi speaks noun-first phrases. `help <group>` lists one group,\n\
         `<phrase> --help` explains one op, and `op list` answers the whole registry as data.\n\
         groups:\n{}\n\
         subjectless: {}.\n\
         reserved words: {} — and `goofi -` runs stdin lines as one batch.",
        groups.join("\n"),
        bare.join(", "),
        crate::ops::RESERVED.join(", "),
    )
}

/// Whether anything under `entry` (at the phrase position `prefix`) is in the served set — how a
/// headless server's listings drop the layout group without a second spelling of the mode.
fn served(ops: &[&Op], prefix: &str, entry: &Entry) -> bool {
    match entry {
        Entry::Leaf(op) => {
            let name = format!("{prefix}{}", op.name);
            ops.iter().any(|o| o.name == name)
        }
        Entry::Group(word, _, _) => {
            let below = format!("{prefix}{word} ");
            ops.iter().any(|o| o.name.starts_with(&below))
        }
    }
}

/// An entry's completion doc: a group's own line, an op's first line capped to a candidate row.
fn doc_line(entry: &Entry) -> String {
    let line = match entry {
        Entry::Group(_, doc, _) => doc,
        Entry::Leaf(op) => op.doc.lines().next().unwrap_or_default(),
    };
    match line.char_indices().nth(100) {
        Some((cut, _)) => {
            let end = line[..cut].rfind(' ').unwrap_or(cut);
            format!("{}…", &line[..end])
        }
        None => line.to_string(),
    }
}

/// What can come NEXT on a partial command line, as `(word, doc)` candidates — the completion
/// read behind `op complete`. The line's trailing word, unless the line ends in whitespace, is a
/// partial that filters. `state` is where the LIVE candidates come from — a `uid` offers the
/// patch's own nodes — and `None` (the offline client) still answers everything static.
pub fn complete(
    ops: &[&Op],
    state: Option<&crate::AppState>,
    line: &str,
) -> Vec<(String, String)> {
    let Ok(words) = shell_words::split(line) else { return Vec::new() };
    let (done, partial) = match line.ends_with(char::is_whitespace) || words.is_empty() {
        true => (&words[..], String::new()),
        false => (&words[..words.len() - 1], words[words.len() - 1].clone()),
    };
    match walk(done) {
        Stop::Op(name, used) => match ops.iter().find(|o| o.name == name) {
            Some(op) => complete_args(op, &done[used..], &partial, state),
            None => Vec::new(),
        },
        Stop::Children(children, prefix) => children
            .iter()
            .filter(|e| word_of(e).starts_with(&partial) && served(ops, &prefix, e))
            .map(|e| (word_of(e).to_string(), doc_line(e)))
            .collect(),
        Stop::Unknown(_) => Vec::new(),
    }
}

fn word_of(entry: &Entry) -> &'static str {
    match entry {
        Entry::Group(word, ..) => word,
        Entry::Leaf(op) => op.name,
    }
}

/// Candidates once the op is named: a value for the flag just given, then the next positional's
/// values beside the flags not yet spent.
fn complete_args(
    op: &Op,
    given: &[String],
    partial: &str,
    state: Option<&crate::AppState>,
) -> Vec<(String, String)> {
    let decls: Vec<(&str, &str, bool)> = op.args().collect();
    if let Some(key) = given.last().and_then(|w| w.strip_prefix("--")) {
        let key = key.strip_prefix("no-").unwrap_or(key);
        if let Some((_, ty, _)) = decls.iter().find(|(n, t, _)| *n == key && *t != "bool") {
            return values_for(ty, state, partial);
        }
    }
    let mut out = Vec::new();
    let bare = given.iter().take_while(|w| !w.starts_with("--")).count();
    if !partial.starts_with("--") {
        if let Some((_, ty, _)) = decls.get(bare).filter(|_| bare < op.positional) {
            out.extend(values_for(ty, state, partial));
        }
    }
    for (name, ty, required) in &decls {
        let spent = *ty != "bool"
            && !ty.ends_with("[]")
            && given.iter().any(|w| {
                w.strip_prefix("--")
                    .is_some_and(|f| f.split('=').next() == Some(*name))
            });
        if spent {
            continue;
        }
        let doc = match (ty.strip_suffix("[]"), *ty, required) {
            (_, "bool", _) => "flag; --no- negates".to_string(),
            (Some(item), _, true) => format!("<{item}>, repeats, required"),
            (Some(item), _, false) => format!("<{item}>, repeats"),
            (None, ty, true) => format!("<{ty}>, required"),
            (None, ty, false) => format!("<{ty}>"),
        };
        let flag = format!("--{name}");
        if flag.starts_with(partial) {
            out.push((flag, doc));
        }
    }
    out
}

/// The live values a typed position can offer. Closed vocabularies answer always; the patch's own
/// names need the server.
fn values_for(ty: &str, state: Option<&crate::AppState>, partial: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = match ty.trim_end_matches("[]") {
        "panel_type" => crate::vocab::PANEL_TYPES
            .iter()
            .map(|p| (p.id.to_string(), p.doc.to_string()))
            .collect(),
        "bool" => ["true", "false"].iter().map(|w| (w.to_string(), String::new())).collect(),
        "uid" | "endpoint" => {
            let Some(state) = state else { return Vec::new() };
            let doc = state.doc.lock().unwrap().to_json();
            let Some(nodes) = doc.get("nodes").and_then(Value::as_object) else {
                return Vec::new();
            };
            nodes
                .iter()
                .map(|(uid, n)| {
                    let name = n["name"].as_str().unwrap_or_default();
                    let kind = n["type"].as_str().unwrap_or_default();
                    (uid.clone(), format!("{name} ({kind})"))
                })
                .collect()
        }
        _ => Vec::new(),
    };
    out.retain(|(w, _)| w.starts_with(partial));
    out.sort();
    out
}

fn op_help(op: &Op) -> String {
    format!("{}\n\n{}\n\nanswers: {}", usage(op), op.doc(), op.result)
}
