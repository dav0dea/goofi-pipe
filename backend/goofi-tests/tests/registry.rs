//! The op registry and the word vocabularies — two tables that generate three consumers each, and
//! the guards that keep the four in step.
//!
//! `dispatch` was a string-keyed match with no way to say "this arm is missing", and `op` was a free
//! string at scattered call sites. The registry closed both directions; these guards are what stop
//! it re-opening. They are table checks over public API, so they belong beside the suite rather than
//! inside the crate they judge.

use std::collections::HashSet;

use goofi_bridge::ops::{find, typescript, Surface, MCP_PREFIX, REGISTRY};
use goofi_bridge::vocab;
use goofi_tests::{j, Goofi};

/// The argument types the schema DSL admits. A type outside this set is a typo.
const ARG_TYPES: &[&str] = &[
    "uid", "string", "float", "int", "bool", "float2", "json", "panel_type", "uid[]",
    "string[]", "float[]",
];

/// A generated file, kept honest. On drift it is REWRITTEN and the test fails once, so the fix is
/// to re-run and commit rather than to hand-transcribe a table.
fn regenerated(rel: &str, want: String) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    if std::fs::read_to_string(&path).ok().as_deref() != Some(want.as_str()) {
        std::fs::write(&path, &want).expect("rewriting the generated file");
        panic!("{rel} was stale; it has been regenerated — review and commit it");
    }
}

/// A name outside `[a-z0-9_]+`, or one long enough to push `mcp__goofi__<name>` past 64
/// characters, makes Claude and OpenAI reject the ENTIRE tool list with a 400 — every tool,
/// not just the offending one. So this is a build-stopping invariant, not a lint.
#[test]
fn every_op_name_fits_the_mcp_budget() {
    for op in REGISTRY {
        assert!(
            op.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                && !op.name.is_empty(),
            "`{}` is not [a-z0-9_]+",
            op.name
        );
        assert!(
            MCP_PREFIX.len() + op.name.len() <= 64,
            "`{}{}` is {} characters — over the 64 a tool name may have",
            MCP_PREFIX,
            op.name,
            MCP_PREFIX.len() + op.name.len(),
        );
    }
}

/// The params schema is a string, so a typo in it is only a fact at read time. Parse every
/// row here instead, where a malformed one stops the build.
#[test]
fn every_row_declares_a_well_formed_schema() {
    for op in REGISTRY {
        assert_eq!(
            op.args().count(),
            op.args.split_whitespace().count(),
            "`{}` has an argument with no `name:type`: {:?}",
            op.name,
            op.args
        );
        for (arg, ty, _) in op.args() {
            assert!(ARG_TYPES.contains(&ty), "`{}`'s `{arg}` has unknown type `{ty}`", op.name);
        }
        assert!(!op.doc.is_empty() && !op.result.is_empty(), "`{}` is undocumented", op.name);
    }
    // The `!` itself has to reach the parse, or Task 4 would advertise every argument as
    // optional and a model would omit the one the op cannot run without.
    let add: Vec<_> = find("add_node").expect("add_node is registered").args().collect();
    assert_eq!(add[0], ("type", "string", true));
    assert_eq!(add[1], ("pos", "float2", false));
}

/// A caller that has to guess a vocabulary word gets it wrong (`params` for `parameters`), and
/// the guess used to be answered `{ok: true}`. So the description ENUMERATES both vocabularies,
/// and it does it by expansion rather than by a hand-copied list — which would be the very
/// duplication `vocab.rs` exists to remove.
#[test]
fn an_op_that_takes_a_vocabulary_word_names_the_set_in_its_own_description() {
    let doc = find("page_set_panel").expect("page_set_panel is registered").doc();
    for word in ["parameters", "node-editor", "viewer", "line", "trajectory", "topomap"] {
        assert!(doc.contains(word), "`{word}` is not offered by page_set_panel's doc: {doc}");
    }
    // The same vocabulary, one door over: a node's stored per-slot view names a kind too, and
    // the manager refuses a guess at it — so the description has to offer the choices here as
    // well, or the refusal is the only teacher.
    let doc = find("set_node_viewers").expect("set_node_viewers is registered").doc();
    for word in ["line", "topomap", "table"] {
        assert!(doc.contains(word), "`{word}` is not offered by set_node_viewers's doc: {doc}");
    }
    for op in REGISTRY {
        let doc = op.doc();
        assert!(
            !doc.contains("{panel_types}") && !doc.contains("{viewer_kinds}"),
            "`{}` has an unexpanded placeholder — a model would read it verbatim",
            op.name
        );
    }
}

/// Agents set `triggers: true` on every expression they bound. The tool description is the ONLY
/// text they read — [`crate::mcp::tools`] projects `doc` + `result`, and the input schema carries
/// no per-argument description — and this doc named NEITHER boolean, so both read as one "turn
/// the expression on" gesture; and `enabled` genuinely does have to be true. The description now
/// states each flag's default and what `triggers` costs. This is where that stays true.
#[test]
fn set_expressions_description_states_both_flags_defaults() {
    let doc = find("set_expression").expect("set_expression is registered").doc();
    for phrase in ["`enabled` defaults false", "`triggers` defaults false", "enabled: true"] {
        assert!(doc.contains(phrase), "set_expression's doc does not say {phrase:?}: {doc}");
    }
}

/// Uniqueness matters twice over: two rows of one name would give the MCP tool list a
/// duplicate (a 400, like a bad name) and make `find` silently prefer the first.
#[test]
fn op_names_are_unique() {
    let mut seen = HashSet::new();
    for op in REGISTRY {
        assert!(seen.insert(op.name), "`{}` is declared twice", op.name);
    }
}

/// `surface` is the one column with a SAFETY consequence, and Task 4 generates the agent's
/// whole tool list from it — so it is pinned as a set, not as a property. Every name here
/// either replaces the patch an agent is working inside (and, for the three that share the
/// `load` arm, its undo history with it), is the human file browser's half of that door, or is
/// a harness op: an agent that could spawn or kill a harness could spawn itself a peer, or
/// terminate the very process it is speaking through (user, 2026-08-10).
/// Adding a row to this list is a decision; the test is where it gets made deliberately.
#[test]
fn only_the_self_terminating_and_file_browser_ops_are_kept_off_the_agent_surface() {
    let control_only: Vec<&str> =
        REGISTRY.iter().filter(|o| o.surface == Surface::ControlOnly).map(|o| o.name).collect();
    assert_eq!(
        control_only,
        [
            "list_dir",
            "set_viewpoint",
            "serialize",
            "save",
            "load_text",
            "load",
            "new",
            "list_harnesses",
            "spawn_harness",
            "stop_harness"
        ]
    );
}

/// The other half of the coverage claim. A row without a dispatch arm falls through to the
/// match's catch-all and answers `unknown op` — an op the palette, the MCP tool list and the
/// frontend's `OpName` union all advertise and nothing can actually call. (The converse — an
/// arm without a row — needs no test: the gate in `dispatch` refuses the op before the match
/// is reached, so such an arm is unreachable rather than silently live.)
#[test]
fn every_registry_op_has_a_dispatch_arm() {
    let g = Goofi::new();
    for op in REGISTRY {
        // Called with no arguments, so most answer a refusal — the one answer that must not appear
        // is the catch-all's, which is what a row with no arm falls through to.
        if let Err(e) = g.try_call(op.name, j!({})) {
            assert!(!e.contains(&format!("unknown op `{}`", op.name)),
                    "`{}` is in the registry but dispatch has no arm for it: {e}", op.name);
        }
    }
}

/// The generated frontend union, kept honest. On drift the file is REWRITTEN and the test
/// fails once, so the fix is to re-run and commit rather than to hand-transcribe a list.
#[test]
fn the_frontend_op_union_is_generated_from_the_registry() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../frontend/src/lib/api/ops.ts");
    let want = typescript();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(want.as_str()) {
        std::fs::write(&path, &want).expect("rewriting the generated op union");
        panic!("{} was stale; it has been regenerated — review and commit it", path.display());
    }
}

// --------------------------------------------------------------------------
// The word vocabularies
// --------------------------------------------------------------------------

/// The generated frontend module, kept honest.
#[test]
fn the_frontend_vocabulary_is_generated_from_the_registry() {
    regenerated("frontend/src/lib/api/vocab.ts", vocab::typescript());
}

/// The generator emits TypeScript string literals with no escaping, so a word carrying a quote
/// or a newline would emit a file that does not parse — caught here rather than by `npm run
/// check` a commit later.
#[test]
fn every_word_is_safe_to_emit_and_unique() {
    let mut seen = HashSet::new();
    for (id, doc) in vocab::PANEL_TYPES.iter().map(|p| (p.id, p.doc))
        .chain(vocab::VIEWER_KINDS.iter().map(|k| (k.id, k.doc)))
    {
        assert!(seen.insert(id), "`{id}` is declared twice");
        assert!(!doc.is_empty(), "`{id}` is undocumented");
        for s in [id, doc] {
            assert!(!s.contains('\'') && !s.contains('\\') && !s.contains('\n'), "unquotable: {s}");
        }
    }
}

/// The engine mints panel entries of its own — the default page's, and the empty one a split
/// births. Both name a type as a bare string, so this is where the vocabulary and the layout
/// engine are held to the same table. (The frontend cannot drift from them at all: it reads
/// both constants out of the generated module.)
#[test]
fn the_types_the_layout_engine_mints_are_in_the_vocabulary() {
    for ty in [goofi_engine::layout::DEFAULT_PANEL_TYPE, goofi_engine::layout::EMPTY_PANEL_TYPE] {
        assert!(vocab::panel_type(ty).is_some(), "`{ty}` is not a declared panel type");
    }
}

/// A kind's ViewSpec has to accept everything the component draws, or a frame the viewer WOULD
/// render is filtered out of the merge and never arrives.
#[test]
fn what_a_kind_accepts_covers_what_it_draws() {
    for k in vocab::VIEWER_KINDS {
        if let vocab::Draws::Array { draws, accepts } = k.draws {
            assert!(draws.0 <= draws.1 && accepts.0 <= accepts.1, "`{}` has an empty range", k.id);
            assert!(
                accepts.0 <= draws.0 && accepts.1 >= draws.1,
                "`{}` draws {draws:?} but its ViewSpec only accepts {accepts:?}",
                k.id
            );
        }
    }
}
