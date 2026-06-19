# Per-viewer-instance view state

**Date:** 2026-06-18
**Status:** Approved (brainstorm)
**Branch:** `feat/persistence-subpatch`

## Problem

Viewer **type** (`kind`) and **settings** are stored globally keyed by
`(node, slot)` in two parallel runtime maps (`viewerState.svelte.ts` →
`kinds`, `viewerSettings.svelte.ts` → `store`). The in-canvas inline viewer on
a node body and *every* docked Viewer panel pointed at that slot read and write
the same key, so they are locked together: set the node's inline viewer to
`image` and a Viewer panel on the same slot also flips to `image` (and shares
its settings). The current code documents this as intentional ("stay in
lock-step") — we are reversing it.

The user wants each viewer to be an **independent instance** — its own kind and
settings — while still **sharing one underlying data stream** (we do not
duplicate the backend→browser stream).

This is a clean rewrite of the view-state layer. There is **no back-compat**
requirement: we are not constrained by the previous iteration's formats or
sub-ideal decisions.

## Goals

- The inline node viewer and each docked Viewer panel are **separate viewer
  instances** with independent `kind` + `settings`.
- Multiple Viewer panels on the same `(node, slot)` are independent of each
  other and of the inline viewer.
- All instances of a slot **share a single data stream** (one backend
  `_SlotMux`, one refcounted frontend WebSocket) — unchanged.
- The inline viewer and the panel viewer are implemented as the **same viewer**,
  not two divergent components. The only thing that differs is *where a given
  instance's view state is persisted* — dictated by where that viewer
  physically lives.
- Each instance's view state round-trips into the saved `.gfi`.

## Non-goals (explicitly deferred / out of scope)

- **Data reduction.** There is currently no backend reduction at all — frames
  are sent full-size, verbatim (`bridge/data.py:on_frame`). The per-axis
  max-wins reduction + in-browser re-reduction is a separate, later effort. No
  backend changes in this work.
- Reworking the data plane, codec, or `frames.ts` stream sharing (already
  correct).
- Migrating old `.gfi` files (clean rewrite; old viewer state is not carried
  over).

## Architecture

Split the two concerns currently fused on the `(node, slot)` key:

| Concern | Identity | Shared? | Owner |
|---|---|---|---|
| **Data source** | `(node, slot, dtype)` | yes — one stream | `frames.ts` refcount ↔ backend `_SlotMux` (unchanged) |
| **View state** (`kind` + `settings`) | per viewer **instance** | no — independent | the placement that mounts the viewer |

### The `ViewBinding` seam

One small interface is the single point of variation between instances:

```ts
interface ViewBinding {
  readonly kind: ViewerKind;        // resolved (defaults applied)
  readonly settings: SettingsMap;   // resolved (defaults + overrides)
  setKind(kind: ViewerKind): void;
  setSetting(key: string, value: SettingValue): void;
}
```

`kind`/`settings` are getters so the binding stays reactive to its backing
store. The shared viewer components consume a `ViewBinding`; they contain **no**
inline-vs-panel branching and never read a global store directly.

### Pure resolution helpers (single source of truth)

Extracted so both bindings resolve identically:

- `resolveKind(dtype, storedKind): ViewerKind` — `STRING`→`string`,
  `TABLE`→`table`, otherwise `storedKind ?? 'line'`.
- `resolveSettings(kind, overrides): SettingsMap` —
  `{ ...defaultSettings(kind), ...overrides }`.

### Binding-driven viewer components

- **`ViewerFeed`** — props `{ node, slot, dtype, binding }`. Subscribes frames
  by `(node, slot)` (the shared stream); renders via `ViewerSurface` using
  `binding.kind` / `binding.settings`.
- **`ViewerControls`** — props `{ dtype, binding }`. The ARRAY kind dropdown
  (`binding.setKind`) + the settings cog.
- **`ViewerSettingsMenu`** — props `{ binding }`. Reads `binding.kind` /
  `binding.settings`; writes via `binding.setSetting`. The `pushNodeViewers`
  call currently buried here moves out — persistence is the binding's job.

### Two placements, two bindings, same viewer

- **Inline** (`SlotViewer`, exactly one per `(node, slot)` on the node body):
  builds an inline binding backed by a single runtime **inline-view store**
  keyed by `(node, slot)`; its setters persist to `node.viewers[slot]` (the
  node record, alongside the collapse flag), so the inline view travels with the
  node. `SlotViewer` keeps its own collapse chrome.
- **Panel** (`ViewerPanel`, keyed by `panelId`, many allowed per slot): builds a
  panel binding backed by the panel's layout-state blob
  `{ node, slot, kind, settings }`; its setters call `setState`, persisting
  per-panel in the layout/`.gfi` (the same mechanism the editor's sub-patch path
  uses). `ViewerPanel` keeps its own node-link + slot-picker chrome.

Both placements render the identical `ViewerControls` + `ViewerFeed` against a
`ViewBinding`. "Inline" and "panel" differ only in chrome and in which
container their binding reads/writes.

### Runtime view-state layer cleanup

Freed from prior decisions, the two parallel inline maps (`kinds` and
`settings`) are merged into **one** inline-view store keyed by `(node, slot)`
holding `{ kind?, settings }` — matching the `ViewBinding`'s single `{kind,
settings}` unit. This store remains separate from the node *data* record (so a
node state-update/snapshot replacement never clobbers live view edits), seeded
from `node.viewers[slot]` on load and pushed back (debounced) via
`pushNodeViewers`. The panel's view state lives only in its layout-state blob.

## Data flow

```
                 ┌─────────── backend ───────────┐
                 │  one _SlotMux per (uid, slot)  │
                 └───────────────┬────────────────┘
                                 │ binary frames (full-size; reduction later)
                 ┌───────────────▼────────────────┐
                 │ frames.ts: ONE refcounted WS    │
                 │ per (node, slot)                │
                 └───┬───────────────────────┬─────┘
        ViewerFeed   │                       │   ViewerFeed
   (inline binding)  │                       │  (panel binding)
   ┌─────────────────▼───┐           ┌───────▼──────────────────┐
   │ SlotViewer          │           │ ViewerPanel              │
   │ kind/settings →     │           │ kind/settings →          │
   │ inline-view store   │           │ panel layout state       │
   │ → node.viewers[slot]│           │ → .gfi layout            │
   └─────────────────────┘           └──────────────────────────┘
        independent view state            independent view state
                       (one shared stream feeds both)
```

## Unchanged behaviors

- **Expanded/collapsed** is inherently inline-only (`ui.isSlotExpanded`,
  persisted in `node.viewers[slot].collapsed`); panels always render their feed.
  No change.
- **Agent surface** (`commands.setViewerKind` / `setViewerSetting(node, slot,…)`)
  targets the node's **inline** instance.
- **Stream sharing**: `frames.ts` refcount and the backend `_SlotMux` are
  untouched.

## Testing

**Unit (frontend):**
- `resolveKind` (dtype forcing + fallback) and `resolveSettings` (defaults +
  overrides merge).

**e2e (`e2e/`):**
- ARRAY node: set the inline viewer to `image`; open a Viewer panel on the same
  node/slot → the panel viewer is still `line` (independent default); flip the
  panel to `image` and a setting → the inline viewer is unaffected, and
  vice-versa.
- Two Viewer panels on the same slot hold independent kind/settings.
- Both the inline viewer and a panel render simultaneously from a **single**
  data stream (one WS for the slot).
- Save → reload preserves the inline viewer's kind/settings (via the node) and
  each panel's kind/settings (via the layout) independently.

## Files touched

- `frontend/src/lib/viewers/viewerState.svelte.ts` /
  `viewerSettings.svelte.ts` — merge into one inline-view store; extract
  `resolveKind` / `resolveSettings`; drop the "shared with panel" framing.
- `frontend/src/lib/viewers/viewBinding.ts` *(new)* — `ViewBinding` type +
  `inlineBinding(node, slot, dtype)` / `panelBinding(getState, setState, dtype)`
  factories.
- `frontend/src/lib/viewers/ViewerControls.svelte`,
  `ViewerSettingsMenu.svelte`, `ViewerFeed.svelte` — binding-driven props.
- `frontend/src/lib/editor/SlotViewer.svelte` — build + pass the inline binding.
- `frontend/src/lib/panels/ViewerPanel.svelte` — hold `{kind, settings}` in
  panel state; build + pass the panel binding.
- `frontend/src/lib/stores/graph.svelte.ts` — inline-view seed/push adjusted to
  the merged store (no behavioral change to `node.viewers` round-trip).

No backend changes.
