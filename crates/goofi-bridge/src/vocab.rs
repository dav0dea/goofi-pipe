//! The panel-type and viewer-kind vocabularies — the one place each word is declared.
//!
//! Sub-project A made the manager authoritative for layout, but the words that give a panel entry
//! its MEANING stayed in the frontend: panel types were registered in `panels/register.ts`, viewer
//! kinds were a TypeScript union in `viewers/kind.ts`. The manager wrote both into an entry as free
//! strings it could not check, so `page_set_panel {type: "params"}` — a plausible guess for
//! `parameters` — answered `{ok: true}` and left the panel in an "Unknown panel type" state (found
//! by the user driving a real agent, 2026-08-10). An incomplete migration, not a missing guard.
//!
//! So the vocabulary and its metadata are declared HERE and the TypeScript is GENERATED from it,
//! the way [`crate::ops`] generates the frontend's `OpName` union: a checked-in file plus a drift
//! test, no build machinery. Three consumers follow from one table — the manager's refusal, the
//! agent's tool schema, and the frontend's registration.
//!
//! **What deliberately does NOT live here: the behaviour keyed off a word.** Which Svelte component
//! renders a panel type, and whether a particular array actually draws, are code over client state
//! — moving them would be inventing an abstraction, not removing a duplicate. What collapses is the
//! duplicated *lists* and the hand-maintained agreements between them.

use goofi_engine::layout::{DEFAULT_PANEL_TYPE, EMPTY_PANEL_TYPE};
use serde_json::Value;

/// One panel type — what a layout entry's `panel_type` may say, and what `page_set_panel {type}`
/// may name.
pub struct PanelType {
    /// The layout key. Stored in the `.gfi`, so renaming one is a migration, not an edit.
    pub id: &'static str,
    /// The header label. The frontend's, but declared here because it is a fact about the word
    /// rather than about the component that happens to draw it.
    pub title: &'static str,
    /// A name from the frontend's one icon set. Typed as `IconName` on the generated side, so a
    /// glyph that does not exist is a `npm run check` error rather than an empty box at runtime.
    pub icon: &'static str,
    /// Whether a node can be BOUND to this panel (`state.node`). The editor already gates its
    /// drop target on this; declaring it here is what lets the manager refuse a bind the panel
    /// would silently ignore, and lets the agent read which types take a node at all.
    pub accepts_node: bool,
    /// What the panel shows, for a caller choosing one. Short: every one of these rides in
    /// `page_set_panel`'s tool description, which a model provider truncates at 2 KB.
    pub doc: &'static str,
}

/// What a viewer kind can be handed. The two questions — which dtype it serves and how many
/// dimensions it takes — are one enum so that "a STRING viewer that draws 2-D arrays" cannot be
/// written down.
pub enum Draws {
    /// An ARRAY kind. `draws` is the dimension range the component actually renders; `accepts` is
    /// the (equal or wider) range its ViewSpec declares compatible, so a frame it will only
    /// SUMMARISE still arrives reduced instead of at full size. `line` is the one kind where they
    /// differ, and that difference is exactly why both are stated: `capacity.ts` and `kind.ts` each
    /// carried one of them, with a comment claiming they mirrored each other.
    Array { draws: (u8, u8), accepts: (u8, u8) },
    /// A kind PINNED to a non-array dtype: a slot of that dtype always resolves to this kind, and
    /// the dimension question does not arise.
    Pinned(&'static str),
}

/// One viewer kind — what a viewer panel's `state.kind` and a node's stored per-slot view may say.
pub struct ViewerKind {
    pub id: &'static str,
    pub draws: Draws,
    /// What the kind shows, for a caller choosing one. Same 2 KB budget as [`PanelType::doc`].
    pub doc: &'static str,
}

impl ViewerKind {
    /// The `Data` dtype this kind serves, in the spelling the wire uses.
    pub fn dtype(&self) -> &'static str {
        match self.draws {
            Draws::Array { .. } => "ARRAY",
            Draws::Pinned(d) => d,
        }
    }
}

/// Declaration order is the order the frontend registers them in, which is the order the panel
/// menu lists them — so this table is also the menu. `empty` leads because it is the framework's
/// own placeholder, registered before any app panel and filtered out of the choice grid.
pub static PANEL_TYPES: &[PanelType] = &[
    PanelType { id: "empty", title: "Empty", icon: "square-dashed", accepts_node: false,
                doc: "a placeholder with no content yet — what a fresh split births" },
    PanelType { id: "node-editor", title: "Node Editor", icon: "workflow", accepts_node: false,
                doc: "the patch canvas — nodes, wires and sub-patches" },
    PanelType { id: "parameters", title: "Parameters", icon: "sliders-horizontal", accepts_node: true,
                doc: "the parameters of one node, with ranges and expression bindings" },
    PanelType { id: "viewer", title: "Viewer", icon: "activity", accepts_node: true,
                doc: "live frames from one output slot, drawn by `state.kind`" },
    PanelType { id: "metadata", title: "Metadata", icon: "info", accepts_node: true,
                doc: "frame metadata from one output slot (sfreq, channels, shape)" },
    PanelType { id: "console", title: "Console", icon: "terminal", accepts_node: true,
                doc: "the patch log; a bound node filters it to that node" },
    PanelType { id: "globals", title: "Globals", icon: "globe", accepts_node: false,
                doc: "the patch globals, which any expression can read" },
    PanelType { id: "agent", title: "Agent", icon: "bot", accepts_node: false,
                doc: "a terminal on an agent harness, running in the patch workspace" },
];

/// The four ARRAY kinds first, in the order the viewer's type dropdown offers them; the two
/// dtype-pinned kinds after, which no dropdown offers because the slot's dtype chooses them.
pub static VIEWER_KINDS: &[ViewerKind] = &[
    ViewerKind { id: "line", draws: Draws::Array { draws: (0, 2), accepts: (0, 3) },
                 doc: "a time plot: one series (1-D) or one per channel (C, N)" },
    ViewerKind { id: "image", draws: Draws::Array { draws: (2, 3), accepts: (2, 3) },
                 doc: "a bitmap: (H, W), or (H, W, C) for 1–4 channels" },
    ViewerKind { id: "trajectory", draws: Draws::Array { draws: (2, 2), accepts: (2, 2) },
                 doc: "a phase portrait over pairs of rows of a (D, N) frame" },
    ViewerKind { id: "topomap", draws: Draws::Array { draws: (1, 1), accepts: (1, 1) },
                 doc: "a scalp map of one scalar per channel" },
    ViewerKind { id: "string", draws: Draws::Pinned("STRING"), doc: "the text of a STRING slot" },
    ViewerKind { id: "table", draws: Draws::Pinned("TABLE"), doc: "the rows of a TABLE slot" },
];

/// The row for `id`, if the panel type exists.
pub fn panel_type(id: &str) -> Option<&'static PanelType> {
    PANEL_TYPES.iter().find(|p| p.id == id)
}

/// Every panel type's id — the JSON-Schema `enum` an agent's tool list carries, and the set a
/// refusal names.
pub fn panel_type_ids() -> Vec<&'static str> {
    PANEL_TYPES.iter().map(|p| p.id).collect()
}

/// Every viewer kind's id.
pub fn viewer_kind_ids() -> Vec<&'static str> {
    VIEWER_KINDS.iter().map(|k| k.id).collect()
}

/// The vocabularies as prose, for the tool description a model reads BEFORE it calls. The teachable
/// refusal is the fallback; this is the mechanism, because a caller that never guesses never has to
/// be corrected. Each entry is `id — doc`, which is what makes it a choice rather than a list.
fn described(entries: impl Iterator<Item = (&'static str, &'static str)>) -> String {
    entries.map(|(id, doc)| format!("{id} ({doc})")).collect::<Vec<_>>().join("; ")
}
pub fn panel_types_help() -> String {
    described(PANEL_TYPES.iter().map(|p| (p.id, p.doc)))
}
pub fn viewer_kinds_help() -> String {
    described(VIEWER_KINDS.iter().map(|k| (k.id, k.doc)))
}

/// The frontend's vocabulary module, generated from the tables above. Checked into the tree (see
/// [`tests::the_frontend_vocabulary_is_generated_from_the_registry`]) for the same reason
/// [`crate::ops::typescript`] is: small, reviewable in a diff, and no new build machinery.
pub fn typescript() -> String {
    let panel_ids = PANEL_TYPES.iter().map(|p| format!("\n\t| '{}'", p.id)).collect::<String>();
    let kind_ids = VIEWER_KINDS.iter().map(|k| format!("\n\t| '{}'", k.id)).collect::<String>();
    let panels = PANEL_TYPES
        .iter()
        .map(|p| {
            format!(
                "\t{{ id: '{}', title: '{}', icon: '{}', acceptsNode: {}, doc: '{}' }},\n",
                p.id, p.title, p.icon, p.accepts_node, p.doc
            )
        })
        .collect::<String>();
    let dims = |r: Option<(u8, u8)>| match r {
        Some((lo, hi)) => format!("[{lo}, {hi}]"),
        None => "null".into(),
    };
    let kinds = VIEWER_KINDS
        .iter()
        .map(|k| {
            let (draws, accepts) = match k.draws {
                Draws::Array { draws, accepts } => (Some(draws), Some(accepts)),
                Draws::Pinned(_) => (None, None),
            };
            format!(
                "\t{{ id: '{}', dtype: '{}', draws: {}, accepts: {}, doc: '{}' }},\n",
                k.id,
                k.dtype(),
                dims(draws),
                dims(accepts),
                k.doc
            )
        })
        .collect::<String>();
    format!(
        "// GENERATED from crates/goofi-bridge/src/vocab.rs — do not edit by hand.\n\
         // Panel types and viewer kinds are declared ONCE, in the manager: naming one that is not\n\
         // in the table is a type error here and a teachable refusal there. The BEHAVIOUR keyed off\n\
         // a word stays client-side — which component renders a panel type (`panels/register.ts`),\n\
         // and whether a particular array draws (`viewers/kind.ts`). Regenerate by running\n\
         // `cargo test -p goofi-bridge`, which rewrites this file when it drifts.\n\
         import type {{ IconName }} from '$lib/ui';\n\
         \n\
         export type PanelTypeId ={panel_ids};\n\
         \n\
         /** The panel type a brand-new page starts with. */\n\
         export const DEFAULT_PANEL_TYPE = '{DEFAULT_PANEL_TYPE}';\n\
         /** The placeholder a split births, whose in-panel grid chooses its content. */\n\
         export const EMPTY_PANEL_TYPE = '{EMPTY_PANEL_TYPE}';\n\
         \n\
         export type ViewerKind ={kind_ids};\n\
         \n\
         export interface PanelTypeInfo {{\n\
         \treadonly id: PanelTypeId;\n\
         \treadonly title: string;\n\
         \treadonly icon: IconName;\n\
         \t/** Whether a node dragged onto this panel binds to it (`state.node`). */\n\
         \treadonly acceptsNode: boolean;\n\
         \treadonly doc: string;\n\
         }}\n\
         \n\
         export interface ViewerKindInfo {{\n\
         \treadonly id: ViewerKind;\n\
         \t/** The slot dtype this kind serves. A non-ARRAY dtype PINS its kind: a STRING slot is\n\
         \t * always drawn by the string viewer, whatever kind was stored. */\n\
         \treadonly dtype: 'ARRAY' | 'STRING' | 'TABLE';\n\
         \t/** The dimension range the component actually renders — null for a pinned kind. */\n\
         \treadonly draws: readonly [number, number] | null;\n\
         \t/** The dimension range its ViewSpec declares compatible: equal or wider than `draws`,\n\
         \t * so a frame the viewer only summarises still arrives reduced. */\n\
         \treadonly accepts: readonly [number, number] | null;\n\
         \treadonly doc: string;\n\
         }}\n\
         \n\
         export const PANEL_TYPES: readonly PanelTypeInfo[] = [\n{panels}];\n\
         \n\
         export const VIEWER_KINDS: readonly ViewerKindInfo[] = [\n{kinds}];\n"
    )
}

/// A `{name: dtype}` map of a node's OUTPUT slots — a leaf's declared outputs, or a collapsed
/// sub-patch's outward boundary ports, which is what a panel bound to an instance actually reads.
/// One answer for both, so a caller's slot cannot be valid on one projection and not the other.
pub fn output_slots(g: &goofi_engine::Graph, uid: goofi_engine::Uid) -> Vec<(String, &'static str)> {
    if let Some(scope) = g.scope(uid) {
        return scope
            .stubs
            .iter()
            .filter(|(_, s)| s.dir == goofi_engine::subpatch::Dir::Out)
            .map(|(id, s)| (id.clone(), s.dtype.name()))
            .collect();
    }
    g.manifest(uid)
        .map(|m| m.outputs.iter().map(|o| (o.name.to_string(), o.kind.name())).collect())
        .unwrap_or_default()
}

/// Check one word against a vocabulary, refusing with the whole set. Naming the set is the point:
/// an agent that guessed can only correct itself if the refusal says what it could have meant.
fn check(op: &str, field: &str, word: &str, valid: Vec<&'static str>) -> Result<(), String> {
    match valid.contains(&word) {
        true => Ok(()),
        false => Err(format!("{op}: no {field} `{word}` — this app has: {}", valid.join(", "))),
    }
}

/// Check one output slot name against the node it is addressed on, refusing with the ones that
/// exist. Shared by the two ops that carry a slot name inside an opaque bag.
fn check_slot(g: &goofi_engine::Graph, op: &str, uid: goofi_engine::Uid, slot: &str) -> Result<(), String> {
    let slots = output_slots(g, uid);
    if slots.iter().any(|(name, _)| name == slot) {
        return Ok(());
    }
    let have: Vec<&str> = slots.iter().map(|(n, _)| n.as_str()).collect();
    Err(format!("{op}: node `{}` has no output slot `{slot}` — it has: {}", uid.to_hex(), have.join(", ")))
}

/// Validate a `set_node_viewers` bag: `{slot: {kind, settings, collapsed}}`. The same two free
/// strings [`check_panel`] refuses, one door over — a node's stored per-slot view carries the very
/// vocabulary a viewer panel's `state.kind` does, and the manager kept this bag opaque, so a guess
/// was answered `{ok: true}` and the editor drew something else.
///
/// A uid naming no node is left alone: the engine's own write refuses it, and by name.
pub fn check_viewers(
    g: &goofi_engine::Graph,
    uid: goofi_engine::Uid,
    viewers: &Value,
) -> Result<(), String> {
    const OP: &str = "set_node_viewers";
    let bag = viewers
        .as_object()
        .ok_or_else(|| format!("{OP}: viewers is a {{slot: {{kind, settings, collapsed}}}} map"))?;
    if g.manifest(uid).is_none() {
        return Ok(());
    }
    for (slot, view) in bag {
        check_slot(g, OP, uid, slot)?;
        if let Some(kind) = view.get("kind").and_then(Value::as_str).filter(|k| !k.is_empty()) {
            check(OP, "viewer kind", kind, viewer_kind_ids())?;
        }
    }
    Ok(())
}

/// Validate a `page_set_panel` write against the vocabularies and the node it binds, BEFORE the
/// layout is planned. Every one of these is a write that would otherwise succeed and mean something
/// other than what was asked: an unknown type renders "Unknown panel type", an unknown kind falls
/// back to the line plot, a slot the node does not have renders the panel's empty state, and a node
/// bound to a panel that does not take one is simply ignored.
///
/// `bound` is the node the panel ends up bound to — this write's, or the one already stored, since
/// a state write MERGES.
pub fn check_panel(
    g: &goofi_engine::Graph,
    ty: Option<&str>,
    state: Option<&Value>,
    bound: Option<goofi_engine::Uid>,
) -> Result<(), String> {
    const OP: &str = "page_set_panel";
    if let Some(t) = ty {
        check(OP, "panel type", t, panel_type_ids())?;
    }
    let key = |k: &str| state.and_then(|s| s.get(k)).and_then(Value::as_str).filter(|v| !v.is_empty());
    if let Some(kind) = key("kind") {
        check(OP, "viewer kind", kind, viewer_kind_ids())?;
    }
    if let (Some(slot), Some(uid)) = (key("slot"), bound) {
        check_slot(g, OP, uid, slot)?;
    }
    // A bind is refused against the type this write LEAVES the panel with, which is this write's
    // when it names one — a combined `{type, state}` is one act.
    if key("node").is_some() {
        if let Some(t) = ty.and_then(panel_type).filter(|t| !t.accepts_node) {
            return Err(format!("{OP}: a `{}` panel does not bind a node", t.id));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generated frontend module, kept honest. On drift the file is REWRITTEN and the test
    /// fails once, so the fix is to re-run and commit rather than to hand-transcribe a table.
    #[test]
    fn the_frontend_vocabulary_is_generated_from_the_registry() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../frontend/src/lib/api/vocab.ts");
        let want = typescript();
        if std::fs::read_to_string(&path).ok().as_deref() != Some(want.as_str()) {
            std::fs::write(&path, &want).expect("rewriting the generated vocabulary");
            panic!("{} was stale; it has been regenerated — review and commit it", path.display());
        }
    }

    /// The generator emits TypeScript string literals with no escaping, so a word carrying a quote
    /// or a newline would emit a file that does not parse — caught here rather than by `npm run
    /// check` a commit later.
    #[test]
    fn every_word_is_safe_to_emit_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for (id, doc) in PANEL_TYPES.iter().map(|p| (p.id, p.doc))
            .chain(VIEWER_KINDS.iter().map(|k| (k.id, k.doc)))
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
        for ty in [DEFAULT_PANEL_TYPE, EMPTY_PANEL_TYPE] {
            assert!(panel_type(ty).is_some(), "`{ty}` is not a declared panel type");
        }
    }

    /// A kind's ViewSpec has to accept everything the component draws, or a frame the viewer WOULD
    /// render is filtered out of the merge and never arrives.
    #[test]
    fn what_a_kind_accepts_covers_what_it_draws() {
        for k in VIEWER_KINDS {
            if let Draws::Array { draws, accepts } = k.draws {
                assert!(draws.0 <= draws.1 && accepts.0 <= accepts.1, "`{}` has an empty range", k.id);
                assert!(
                    accepts.0 <= draws.0 && accepts.1 >= draws.1,
                    "`{}` draws {draws:?} but its ViewSpec only accepts {accepts:?}",
                    k.id
                );
            }
        }
    }
}
