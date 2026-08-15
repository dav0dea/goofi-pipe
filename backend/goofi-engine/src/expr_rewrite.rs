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
        // A call whose `)` the scanner never found is not a term this can span. Left verbatim: the
        // rewritten source then names `nd`, which no longer exists in the eval namespace, so the
        // binding reports a NameError instead of being quietly rewired.
        let Some(end) = call.end else { continue };
        if call.name.is_empty() {
            return Err(ExprError("nd() needs a node name".to_string()));
        }
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
    // Source order, so `__v0` is the first term a reader sees — and so the splice below can walk
    // the terms once.
    terms.sort_by_key(|t| t.start);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scanner_spans_the_whole_reference_not_just_the_name() {
        let (src, vars) = rewrite("nd('lfo').out.mean() * globals.gain").unwrap();
        assert_eq!(src, "__v0.mean() * __v1", "the .out slot is consumed, .mean() is not");
        assert_eq!(vars.len(), 2);
        assert_eq!(
            vars,
            [
                VarRef::Node {
                    var: "__v0".to_string(),
                    name: "lfo".to_string(),
                    slot: Some("out".to_string()),
                },
                VarRef::Global { var: "__v1".to_string(), key: "gain".to_string() },
            ],
        );
    }

    #[test]
    fn a_bare_reference_on_a_single_output_node_still_resolves() {
        let (src, vars) = rewrite("nd('lfo') + 1").unwrap();
        assert_eq!(src, "__v0 + 1");
        assert!(matches!(&vars[0], VarRef::Node { slot: None, .. }));
    }

    #[test]
    fn a_word_containing_nd_is_not_a_reference() {
        // The existing scan_nd_calls guards this with a word boundary; the new scanner must too, or
        // `round(x)` and `.rfind(` become references. `myglobals` is the same rule on the other
        // namespace — one that fires would rewrite a bare attribute chain into a variable.
        let (src, vars) = rewrite("round(t) + grand + myglobals.gain").unwrap();
        assert_eq!(src, "round(t) + grand + myglobals.gain");
        assert!(vars.is_empty());
    }

    #[test]
    fn two_spellings_of_one_reference_share_a_variable() {
        // One reference is one mailbox. Minting a second variable would subscribe the node twice to
        // the same producer and give the evaluator two names for one value.
        let (src, vars) = rewrite("nd('a') + nd( \"a\" ) * globals.g - globals.g").unwrap();
        assert_eq!(src, "__v0 + __v0 * __v1 - __v1");
        assert_eq!(vars.len(), 2, "one per distinct reference");

        // …and a DIFFERENT slot on the same node is a different reference, because it is a
        // different producer output.
        let (_, two) = rewrite("nd('a').x + nd('a').y").unwrap();
        assert_eq!(two.len(), 2);
    }

    #[test]
    fn the_rewritten_source_keeps_everything_that_is_not_a_reference() {
        // The rewrite is a splice, not a re-render: a quote style, spacing, a comment or a nested
        // call the evaluator needs must survive byte-for-byte.
        let (src, vars) = rewrite("np.clip(nd('lfo') * 2.5, 0, globals.max_v)  # keep").unwrap();
        assert_eq!(src, "np.clip(__v0 * 2.5, 0, __v1)  # keep");
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn an_expression_with_no_references_is_handed_back_unchanged() {
        // A pure-`t` expression is legal and common, and it must not gain a variable — §2.1: a
        // binding with no variables never triggers.
        let (src, vars) = rewrite("t * 2 + 1").unwrap();
        assert_eq!(src, "t * 2 + 1");
        assert!(vars.is_empty());
    }

    #[test]
    fn a_nameless_reference_is_refused_rather_than_left_unresolvable() {
        // `nd('')` names nothing, so there is no producer it could ever resolve to — saying so at
        // bind time is the honest answer, where a variable would sit permanently Missing.
        assert!(rewrite("nd('') + 1").is_err());
    }

    #[test]
    fn a_method_call_is_never_read_as_a_slot() {
        // The whole distinction the term scanner exists for. `.mean` looks exactly like a slot
        // until the `(` — including with a space before it, which Python allows.
        let (src, _) = rewrite("nd('a').mean() + nd('b').sum ()").unwrap();
        assert_eq!(src, "__v0.mean() + __v1.sum ()");
    }
}
