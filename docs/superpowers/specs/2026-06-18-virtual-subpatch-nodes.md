# Spec: sub-patches as virtual nodes

2026-06-18, branch `feat/persistence-subpatch`. Prerequisite for the In/Out
authoring work. Driven by Phil's directive:

> node positioning doesn't get mirrored across sub-patches. And sub-patches don't
> open a param window. They should behave exactly as other nodes — virtual nodes
> (no node class instantiated, but everything else behaves the same). Reuse all
> node layers to simplify. make-unique/duplicate should sit in the param window of
> the sub-patch node, not in the node itself. The param window should also indicate
> which other sub-patch nodes are sync'd/mirrored with the selected one.

## Decisions (from the Understand workflow, 6 parallel readers)

1. **Single seam for "virtual node": `graph.nodeByName()` resolves instances.**
   When a name is a sub-patch instance id, `nodeByName` synthesizes a
   `NodeInstanceInfo` carrying a `subpatch` marker (kind, def_id, siblings,
   memberCount, interface, members). This makes every lookup-driven node layer —
   `selection.selectedNode`, the slide-in `InspectorOverlay`, the standalone
   Parameters panel via `NodeLinkedPanel`, `ParamPanel` — work for sub-patches
   with **no change to those files**. Instances are NOT merged into the `nodes`
   array (that was the high-blast-radius option: it breaks clone/group/Ctrl+A/
   viewer-seed, all of which iterate `nodes`). Lookup resolves; iteration stays
   real-only.

2. **`ParamPanel` branches on `node.subpatch`** → renders the new
   `SubPatchInspector` instead of param groups; `InspectorOverlay` skips the
   Metadata + error sections for a sub-patch. `node_moved` handler must check
   `instances[name]` **before** `nodeByName` (else the synth node shadows the
   real instance-pos update and the group node snaps back).

3. **`SubPatchInspector.svelte`** (new): header (glyph + kind + id + member
   count); unique → "Duplicate as shared"; shared → "Make unique" + a
   **"Mirrored with"** list of sibling instance ids (click → select that
   sibling); plus "Expand (dissolve)". Calls `graph()` (duplicateShared/
   makeUnique/expandInstance) + `selection()` directly — no callback threading,
   so it works in both the inline inspector and the standalone Parameters panel.
   **Siblings are computed frontend-side** from `instances` by matching `def_id`
   — no backend/snapshot change needed.

4. **`SubPatchNode.svelte`** loses the make-unique/duplicate buttons (→ inspector);
   keeps the enter (⮕) + expand (⤢) header buttons and the distinct group visual.

5. **Member position mirror across shared siblings** (backend): new
   `Manager.set_node_pos(name, pos) -> list[str]` mirrors the `update_param`
   pattern — set the node's `gui_kwargs["pos"]`; if it's a member of a *shared*
   instance, also write the pos into the definition's member record and into every
   sibling instance's corresponding member; return all changed node names. The
   bridge `set_node_pos` op routes real-node moves through it and broadcasts
   `node_moved` for each returned name. Member pos thus lives in the definition
   (shared truth, like params); build_v2/_expand need no change — all siblings
   carry identical pos, so the envelope round-trips the same layout. The group
   node's own pos stays per-instance (independent placements).

## Out of scope (noted)
- Delete-key on a sub-patch (still a swallowed no-op) — separate follow-up.
- The In/Out authoring nodes — the next task, unblocked by this.
