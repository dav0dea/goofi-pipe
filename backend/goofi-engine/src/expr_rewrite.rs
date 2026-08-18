//! The expression rewrite (spec §5.3): `nd('lfo').out.mean() * globals.gain` becomes
//! `__v0.mean() * __v1` plus the variable map the graph resolves.
//!
//! **The AUTHORED source is the SSOT.** It is what the `.gfi` stores, what the doc shows, what a
//! rename edits and what the inspector round-trips. The rewritten source and its [`VarRef`] list are
//! DERIVED — recomputed whenever the authored source or the graph's names change, shipped to the
//! node, and never stored as the thing a user edits. Getting that backwards makes a rename silently
//! stop following: `rewrite_nd_refs` edits `nd('old')`, and there is no `nd('old')` in `__v0`.
//!
//! The scan itself is [`goofi_node::scan_nd_calls`], shared with the rename rewriter — see
//! [`rewrite`] for why that sharing is load-bearing rather than incidental.

use goofi_node::ExprError;

/// One variable of a rewritten expression, before the graph resolves it to a service or a value.
#[derive(Clone, Debug, PartialEq)]
pub enum VarRef {
    /// `nd('name')` or `nd('name').slot` — a producer's output.
    Node { var: String, name: String, slot: Option<String> },
    /// `globals.key`.
    Global { var: String, key: String },
}

impl VarRef {
    pub fn var(&self) -> &str {
        match self {
            VarRef::Node { var, .. } | VarRef::Global { var, .. } => var,
        }
    }
}

/// What a term refers to — the variable map's content, before a variable name is minted for it.
/// Two terms with equal targets share one variable, because they share one mailbox.
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
/// **Why the `nd()` scan is shared with the rename rewriter rather than reimplemented.** Both
/// answer the same question — "where are the `nd()` calls?" — and only differ in which *span* of
/// each call they want: a rename replaces the NAME LITERAL between the quotes and must leave every
/// other byte alone, while this replaces the WHOLE TERM including the call's parentheses and an
/// optional `.slot`. Two scanners with two word-boundary rules would drift, and the drift is
/// invisible: a source one of them declines to see is a rename that silently stops following, or a
/// reference that never becomes a variable. So [`goofi_node::scan_nd_calls`] reports both spans of
/// each call and each consumer takes the one it needs — one rule about what an `nd()` call IS.
///
/// A trailing `.identifier` is read as the output SLOT unless it is a method call (`.mean()`), which
/// is left in the rewritten source for the evaluator. An attribute that is neither — `nd('a').T` —
/// is read as a slot and the graph then reports that the producer has no such output, which is a
/// visible error rather than a silently wrong reference.
pub fn rewrite(source: &str) -> Result<(String, Vec<VarRef>), ExprError> {
    let mut terms: Vec<Term> = Vec::new();
    for call in goofi_node::scan_nd_calls(source) {
        // Checked BEFORE the call is required to close: `nd('')` names nothing however it is
        // spelled, and an unclosed one is not the more forgivable of the two.
        if call.name.is_empty() {
            return Err(ExprError("nd() needs a node name".to_string()));
        }
        // A call whose `)` the scanner never found is not a term this can span. Left verbatim: the
        // rewritten source then names `nd`, which no longer exists in the eval namespace, so the
        // binding reports a NameError instead of being quietly rewired.
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
        // Two spellings of the same reference share ONE variable, because they share one mailbox:
        // `nd('a') + nd('a')` subscribes once.
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
/// consume, and the reason it needs no bounds check of its own.
///
/// Both properties are earned here rather than assumed, because both were violated by ordinary user
/// input. Ascending: the scans run one after the other, so `globals.g * nd('a')` arrives with its
/// terms in the wrong order and an unsorted splice slices `source[19..0]`. Non-overlapping: the
/// `globals.` scan is a byte scan and does not know what a string literal is, so
/// `nd('globals.gain')` yields a `globals` term INSIDE the `nd` term — and a nested term is not a
/// term at all, it is part of the node name. A slice run backwards PANICS, and this runs under the
/// graph mutex a `set_expression` RPC holds, so the panic poisons the whole control plane.
///
/// Dropping the nested term is the answer rather than refusing the source: `nd('globals.gain')`
/// names a node called `globals.gain`, which is a legal display name.
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
    // `.mean()` is the evaluator's to run; `.out` is ours to resolve. Whitespace before the `(`
    // counts, because `nd('x').mean ()` is one call in Python.
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

