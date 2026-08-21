//! The expression rewrite (spec §5.3): `nd('lfo').out.mean() * globals.gain` becomes
//! `__v0.mean() * __v1` plus the variable map the graph resolves.

use goofi_node::ExprError;

/// One variable of a rewritten expression, before the graph resolves it to a service or a value.
#[derive(Clone, Debug, PartialEq)]
pub enum VarRef {
    /// `nd('name')` or `nd('name').slot` — a producer's output.
    Node { var: String, name: String, slot: Option<String> },
    Global { var: String, key: String },
}

impl VarRef {
    pub fn var(&self) -> &str {
        match self {
            VarRef::Node { var, .. } | VarRef::Global { var, .. } => var,
        }
    }
}

/// What a term refers to. Two terms with equal targets share one variable, and one mailbox.
#[derive(Clone, PartialEq)]
enum Target {
    Node { name: String, slot: Option<String> },
    Global { key: String },
}

impl Target {
    fn into_ref(self, var: String) -> VarRef {
        match self {
            Target::Node { name, slot } => VarRef::Node { var, name, slot },
            Target::Global { key } => VarRef::Global { var, key },
        }
    }
}

/// One span of the source to replace, and what it refers to.
struct Term {
    start: usize,
    end: usize,
    target: Target,
}

/// Rewrite every `nd(..)` and `globals.*` term into a generated variable, answering the rewritten
/// source and the variables it names, in first-seen order.
///
/// A trailing `.identifier` is read as the output SLOT unless it is a method call (`.mean()`),
/// which is left in the rewritten source for the evaluator.
pub fn rewrite(source: &str) -> Result<(String, Vec<VarRef>), ExprError> {
    let mut terms: Vec<Term> = Vec::new();
    for call in goofi_node::scan_nd_calls(source) {
        // `nd('')` names nothing however it is spelled, so this precedes the unclosed-call check.
        if call.name.is_empty() {
            return Err(ExprError("nd() needs a node name".to_string()));
        }
        // An unclosed call is left verbatim, so the binding reports a NameError rather than being
        // quietly rewired.
        let Some(end) = call.end else { continue };
        let (end, slot) = match slot_after(source, end) {
            Some((slot, at)) => (at, Some(slot.to_string())),
            None => (end, None),
        };
        terms.push(Term { start: call.start, end, target: Target::Node { name: call.name.to_string(), slot } });
    }
    for read in goofi_node::scan_globals(source) {
        terms.push(Term {
            start: read.start,
            end: read.end,
            target: Target::Global { key: read.name.to_string() },
        });
    }
    let terms = merge(terms);

    let mut vars: Vec<VarRef> = Vec::new();
    let mut targets: Vec<Target> = Vec::new();
    let mut out = String::with_capacity(source.len());
    let mut cursor = 0;
    for term in terms {
        // Two spellings of one reference share ONE variable: `nd('a') + nd('a')` subscribes once.
        let var = match targets.iter().position(|t| *t == term.target) {
            Some(at) => vars[at].var().to_string(),
            None => {
                let var = format!("__v{}", vars.len());
                vars.push(term.target.clone().into_ref(var.clone()));
                targets.push(term.target);
                var
            }
        };
        out.push_str(&source[cursor..term.start]);
        out.push_str(&var);
        cursor = term.end;
    }
    out.push_str(&source[cursor..]);
    Ok((out, vars))
}

/// The two scans' terms as ONE ascending, non-overlapping list — the only shape the splice can
/// consume. An unsorted or nested term slices backwards, which panics under the graph mutex.
fn merge(mut terms: Vec<Term>) -> Vec<Term> {
    terms.sort_by_key(|t| t.start);
    let mut merged: Vec<Term> = Vec::with_capacity(terms.len());
    for term in terms {
        if merged.last().is_none_or(|prev| prev.end <= term.start) {
            merged.push(term);
        }
    }
    merged
}

/// The output slot named after a closed `nd(..)` call, and where it ends. `None` when the next
/// thing is a method call, or not an attribute at all.
fn slot_after(source: &str, end: usize) -> Option<(&str, usize)> {
    let b = source.as_bytes();
    if b.get(end) != Some(&b'.') {
        return None;
    }
    let start = end + 1;
    let mut at = start;
    while at < b.len() && is_ident(b[at]) {
        at += 1;
    }
    if at == start || b[start].is_ascii_digit() {
        return None;
    }
    // `.mean()` is the evaluator's to run; whitespace counts, since `nd('x').mean ()` is a call.
    let mut after = at;
    while after < b.len() && (b[after] as char).is_whitespace() {
        after += 1;
    }
    if b.get(after) == Some(&b'(') {
        return None;
    }
    Some((&source[start..at], at))
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

