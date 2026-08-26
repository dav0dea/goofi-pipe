//! The panel-type and viewer-kind vocabularies — the one place each word is declared, and the
//! source the frontend's module is generated from. The BEHAVIOUR keyed off a word stays client-side.

use goofi_engine::layout::{DEFAULT_PANEL_TYPE, EMPTY_PANEL_TYPE};
use goofi_engine::subpatch::{Dir, BOUNDARY_SLOT, BOUNDARY_TYPES, SCOPE_TYPE};
use serde_json::{json, Value};

/// One panel type — what a layout entry's `panel_type` may say.
pub struct PanelType {
    /// The layout key. Stored in the `.gfi`, so renaming one is a migration, not an edit.
    pub id: &'static str,
    pub title: &'static str,
    /// A name from the frontend's one icon set, typed as `IconName` on the generated side.
    pub icon: &'static str,
    /// Whether a node can be BOUND to this panel (`state.node`).
    pub accepts_node: bool,
    /// What the panel shows. Short: these ride in `edit_panel`'s tool description, which a model
    /// provider truncates at 2 KB.
    pub doc: &'static str,
}

/// What a viewer kind can be handed: dtype and dimensions in one enum, so "a STRING viewer that
/// draws 2-D arrays" cannot be written down.
pub enum Draws {
    /// `draws` is what the component renders; `accepts` is the equal-or-wider range its ViewSpec
    /// declares, so a frame it only SUMMARISES still arrives reduced.
    Array { draws: (u8, u8), accepts: (u8, u8) },
    /// A kind PINNED to a non-array dtype: that dtype's slots always resolve to this kind.
    Pinned(&'static str),
}

/// One viewer kind — what a viewer panel's `state.kind` and a node's stored per-slot view may say.
pub struct ViewerKind {
    pub id: &'static str,
    pub draws: Draws,
    /// What the kind shows. Same 2 KB budget as [`PanelType::doc`].
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

/// Declaration order is the order the panel menu lists them in; `empty` must lead, because the
/// framework registers its placeholder before any app panel.
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

/// The ARRAY kinds first, in the order the viewer's dropdown offers them; the pinned ones after.
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

/// Every panel type's id — the JSON-Schema `enum` an agent's tool list carries.
pub fn panel_type_ids() -> Vec<&'static str> {
    PANEL_TYPES.iter().map(|p| p.id).collect()
}

/// Every viewer kind's id.
pub fn viewer_kind_ids() -> Vec<&'static str> {
    VIEWER_KINDS.iter().map(|k| k.id).collect()
}

/// The vocabularies as prose, for the tool description a model reads BEFORE it calls.
fn described(entries: impl Iterator<Item = (&'static str, &'static str)>) -> String {
    entries.map(|(id, doc)| format!("{id} ({doc})")).collect::<Vec<_>>().join("; ")
}
pub fn panel_types_help() -> String {
    described(PANEL_TYPES.iter().map(|p| (p.id, p.doc)))
}
pub fn viewer_kinds_help() -> String {
    described(VIEWER_KINDS.iter().map(|k| (k.id, k.doc)))
}

/// The frontend's vocabulary module, generated from the tables above and checked into the tree.
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
    let boundaries = BOUNDARY_TYPES
        .iter()
        .map(|(name, dir, dtype)| {
            format!("\t{{ type: '{name}', dir: '{}', dtype: '{}' }},\n", dir.name(), dtype.name())
        })
        .collect::<String>();
    format!(
        "// GENERATED from backend/goofi-bridge/src/vocab.rs — do not edit by hand.\n\
         // Panel types and viewer kinds are declared ONCE, in the manager: naming one that is not\n\
         // in the table is a type error here and a teachable refusal there. The BEHAVIOUR keyed off\n\
         // a word stays client-side — which component renders a panel type (`panels/register.ts`),\n\
         // and whether a particular array draws (`viewers/kind.ts`). Regenerate by running\n\
         // `cargo test -p goofi-bridge`, which rewrites this file when it drifts.\n\
         import type {{ IconName }} from '$lib/ui';\n\
         \n\
         export type PanelTypeId ={panel_ids};\n\
         \n\
         /** The panel type a brand-new tab starts with. */\n\
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
         export const VIEWER_KINDS: readonly ViewerKindInfo[] = [\n{kinds}];\n\
         \n\
         /** The type a sub-patch facade wears in the document. It is not in the palette — grouping\n\
          * is what makes one. */\n\
         export const SCOPE_TYPE = '{SCOPE_TYPE}';\n\
         \n\
         /** The one slot a boundary port carries. */\n\
         export const BOUNDARY_SLOT = '{BOUNDARY_SLOT}';\n\
         \n\
         export interface BoundaryTypeInfo {{\n\
         \treadonly type: string;\n\
         \t/** An `in` port FEEDS the sub-patch, so it wears an output and is a link's SOURCE. */\n\
         \treadonly dir: 'in' | 'out';\n\
         \treadonly dtype: 'ARRAY' | 'STRING' | 'TABLE';\n\
         }}\n\
         \n\
         /** The six boundary port types: a port's direction and dtype ARE its type. */\n\
         export const BOUNDARY_TYPES: readonly BoundaryTypeInfo[] = [\n{boundaries}];\n\
         \n\
         export const boundaryType = (type: string): BoundaryTypeInfo | undefined =>\n\
         \tBOUNDARY_TYPES.find((b) => b.type === type);\n"
    )
}

/// The six as catalog entries, so a palette and `list_nodes` see one vocabulary of node types.
pub fn boundary_catalog() -> Vec<(String, String, Value)> {
    BOUNDARY_TYPES
        .iter()
        .map(|(name, dir, dtype)| {
            let slot = json!({ BOUNDARY_SLOT: dtype.name() });
            let (inputs, outputs) = match dir {
                Dir::In => (json!({}), slot),
                Dir::Out => (slot, json!({})),
            };
            (
                "boundary".to_string(),
                name.to_string(),
                json!({
                    "type": name,
                    "source": "builtin",
                    "pillar": "signal",
                    "category": "boundary",
                    "doc": format!("Sub-patch {} ({})", dir.name(), dtype.name().to_lowercase()),
                    "available": true,
                    "missing_deps": [],
                    "input_slots": inputs,
                    "input_multi": [],
                    "output_slots": outputs,
                    "params": {},
                }),
            )
        })
        .collect()
}

/// A node's OUTPUT slots as `(key, label, dtype-name)`. The graph owns this — which slots a thing
/// exposes is a fact about the graph, not a vocabulary — so this is the one read, widened.
pub fn output_slots(g: &goofi_engine::Graph, uid: goofi_engine::Uid) -> Vec<(String, String, &'static str)> {
    g.output_slots(uid).into_iter().map(|(k, l, d)| (k, l, d.name())).collect()
}

/// Check one word against a vocabulary, refusing with the whole set.
fn check(op: &str, field: &str, word: &str, valid: Vec<&'static str>) -> Result<(), String> {
    match valid.contains(&word) {
        true => Ok(()),
        false => Err(format!("{op}: no {field} `{word}` — this app has: {}", valid.join(", "))),
    }
}

/// Resolve a slot word — key or display label — to the KEY, refusing by naming the real ones.
pub fn resolve_slot(
    g: &goofi_engine::Graph,
    op: &str,
    uid: goofi_engine::Uid,
    slot: &str,
) -> Result<String, String> {
    let slots = output_slots(g, uid);
    if let Some((key, _, _)) = slots.iter().find(|(key, label, _)| key == slot || label == slot) {
        return Ok(key.clone());
    }
    let have: Vec<&str> = slots.iter().map(|(_, l, _)| l.as_str()).collect();
    Err(format!("{op}: node `{}` has no output slot `{slot}` — it has: {}", uid.to_hex(), have.join(", ")))
}

fn check_slot(g: &goofi_engine::Graph, op: &str, uid: goofi_engine::Uid, slot: &str) -> Result<(), String> {
    resolve_slot(g, op, uid, slot).map(|_| ())
}

/// Validate a `node edit` viewer patch — `--viewer` entries already folded to `{slot: view}`. A
/// uid naming no node is left alone, because the engine's own write refuses it by name.
pub fn check_viewers(
    g: &goofi_engine::Graph,
    uid: goofi_engine::Uid,
    viewers: &serde_json::Map<String, Value>,
) -> Result<(), String> {
    const OP: &str = "node edit";
    if g.node_type(uid).is_none() {
        return Ok(());
    }
    for (slot, view) in viewers {
        check_slot(g, OP, uid, slot)?;
        if let Some(kind) = view.get("kind").and_then(Value::as_str).filter(|k| !k.is_empty()) {
            check(OP, "viewer kind", kind, viewer_kind_ids())?;
        }
    }
    Ok(())
}

/// Validate an `edit_panel` write against the vocabularies and the node it binds, BEFORE the layout
/// is planned. `bound` is the node the panel ENDS UP bound to, since a state write merges.
pub fn check_panel(
    g: &goofi_engine::Graph,
    ty: Option<&str>,
    state: Option<&Value>,
    bound: Option<goofi_engine::Uid>,
) -> Result<(), String> {
    const OP: &str = "layout panel edit";
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
    // Refused against the type this write LEAVES the panel with: a `{type, state}` pair is one act.
    if key("node").is_some() {
        if let Some(t) = ty.and_then(panel_type).filter(|t| !t.accepts_node) {
            return Err(format!("{OP}: a `{}` panel does not bind a node", t.id));
        }
    }
    Ok(())
}
