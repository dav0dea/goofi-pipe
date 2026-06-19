# goofi-pipe Undo/Redo — Ground-Up Redesign — Design Spec

> Status: **draft for review**. This is a design spec, not an implementation.
> The terminal step after approval is `writing-plans`, then test-driven
> implementation phase by phase.

---

## Decision log (authoritative — overrides any contradicting text below)

1. **There is no undo/redo to repair.** A code-verified sweep (every branch,
   stash, reflog, the Python backend, the whole frontend) found **zero**
   undo/redo in `goofi-pipe`. The "previously working" version lived in
   **goofi3**, where it was *backend-tracked* (manager owned `_undo`/`_redo`
   stacks; the browser sent `{type:"undo"}` and got `undo.applied` back). This
   spec is a **ground-up design** for the current architecture, using goofi3 as
   a principles reference only (no code ported).
2. **One frontend-owned, unified history stack** spanning the graph and layout
   domains on a single timeline. `Ctrl+Z` reverses the last action regardless of
   domain. The backend never learns that "undo" exists — undo issues normal
   forward/inverse RPCs and waits for the same echo events any user action does.
3. **One undo entry per mutation primitive** (user's call). No time-window
   coalescing. A paste of N nodes = N+ entries; a 5-node delete = 5 entries. An
   op that is a single backend RPC (group, expand, load) is one entry. A node
   drag already emits exactly one `set_node_pos` RPC at drag-stop, so it is
   naturally one entry.
4. **Never auto-clear** the history on user actions. To keep cross-load history
   coherent, a patch load (`load` / `load_text` / `graph_replaced`) is itself
   recorded as a `load_patch` action whose payload is before/after patch YAML +
   before/after workspace layout. Undoing past a load restores the prior patch
   wholesale.
5. **The one exception to (4):** a new backend session (manager `instance_id`
   changes — a kill/relaunch) **hard-resets** the stack, because the entire
   authoritative world was replaced out-of-band and replaying old RPCs is
   meaningless. Wired into the existing fresh-session detection
   (`graph.svelte.ts:83`, `_onWholesaleLoad` `:98`).
6. **Tracked layout ops:** panel `split` / `close` / `resize` / move
   (drag-drop within a workspace and onto the tab bar) / `setPanelType`; tab
   `add` / `close` / `duplicate` / `rename` / `reorder`; and node→panel link
   binding. **Not tracked:** `toggleMaximize`, `setActive`, `selectTab`,
   internal `setPanelState` (editor zoom/selection inside a panel),
   `enteredPath` navigation.
7. **Navigation is never tracked, only restored for highlight.** Each action
   captures a `NavContext` (workspace, panel, `enteredPath`, selection). On
   undo/redo we first restore that context, then apply the mutation, then
   center/flash the change.
8. **Minimal backend footprint.** The core design requires **no backend
   changes**: identity is restored via the *reused display name* (verified
   `manager.py:336–348`), displaced links are captured frontend-side before the
   RPC, and group/expand round-trip via the existing `inst_id` / `restored`
   return values. A short list of **optional, additive** bridge changes
   (§7) is gated behind Phase-3 test evidence; `member_uid` preservation and a
   `restore_node` op are **deferred** unless tests prove them necessary.
9. **TDD is mandatory** (Iron Law: no production code without a failing test
   first). The architecture exposes testable seams — pure executors with
   injected deps, a `FakeControl` that records RPCs and emits synthetic events,
   pure NavContext + layout-snapshot helpers — so units run without a live
   backend.
10. **Failed undo/redo is atomic-or-nothing.** If a replay RPC rejects (backend
    offline, name re-taken, node gone), the action is restored to its stack, a
    toast is shown, and the user can retry. Undo-of-delete onto a now-occupied
    name fails gracefully (the backend raises `KeyError`, verified
    `manager.py:83`) — it does **not** silently rename in the MVP.

---

## 0. Overview & current-state finding

### What exists today (verified)

- **Graph domain is strictly server-authoritative.** Every `GraphStore`
  mutation method (`addNode`, `removeNode`, `addLink`, `removeLink`,
  `updateParam`, `setExpression`, `setNodePos`, `groupNodes`, `expandInstance`,
  `duplicateShared`, `makeUnique`, `add/wire/removeBoundary`, `setBoundaryPos`,
  `load`, `loadText`) is an `async` RPC wrapper that **never writes state**
  (`graph.svelte.ts:310–467`). Local state is written **only** in `_handle(ev)`
  when the backend broadcasts an event (`:148–295`). `control.ts` `call()`
  resolves only when the backend returns a result, rejects on a backend error.
- **Layout domain is frontend-authoritative, in-memory.** All workspace ops are
  pure tree transforms returning a new `WorkspaceState`
  (`workspace.svelte.ts`, `model.ts`), pushed to the backend as one **debounced
  blob** (`AppShell.svelte` ~400 ms via `g.setLayout`). `serialize()` =
  `$state.snapshot`.
- **Navigation/focus is ephemeral, per-panel.** `enteredPath` (sub-patch depth)
  lives per editor panel and is persisted to `panelState.subpatchPath`;
  selection is keyed per `panelId` (`selection.svelte.ts`); `ws.activePanelId`
  and `state.activeWorkspaceId` track focus.

### Why a frontend-owned stack is correct

Server-authoritative does **not** block frontend undo: undo replays normal
forward/inverse RPCs and lets the echo events reconcile, exactly like a user
action. All navigation/focus context the user wants for highlighting lives
frontend-side, which is precisely where the action records it. The backend stays
a dumb, authoritative executor.

### The three hard problems and how they dissolve (verified)

| Problem | Resolution | Evidence |
|---|---|---|
| Node identity on undo-of-delete | The **display name is reused** (`oscillator0` frees and returns); the frontend references nodes **by display name** (links + panel bindings). Re-add with `name`+`params`+`membership` (all already in the `node_added` payload) restores everything the frontend sees. A fresh transport `node_id`/`member_uid` is invisible in-session. | `manager.py:336–348`; `schemas.py:describe_node_instance`; bridge `add_node` already accepts `name`+`params` (`control.py:151–172`) |
| Displaced links (single-source rule) | `add_link` tears down the occupied input's wire and **broadcasts `link_removed`**; the frontend can also read `isInputConnected(node_in, slot_in)` **before** the add. Captured purely frontend-side. | `manager.py:456–468`, `graph.svelte.ts:567–569` |
| Param prior-value / batch partial-failure / shared-sibling mirror | Snapshot the prior value from the store before the RPC; record one entry per primitive; for shared members issue **one** inverse RPC and let the backend re-mirror it. | `graph.svelte.ts:250–256`, `manager.py:1115–1123` |

---

## Glossary — identity & id model

| Term | Meaning | Reuse semantics |
|---|---|---|
| **display name** | User-facing node label and graph key (e.g. `oscillator0`). Links and panel bindings reference this. | **Reused** once freed (`manager.py:336–348`). The undo system's primary identity. |
| **member_uid** | Backend stable persistent identity; indexes `_refs_by_uid`; matters for sub-patch strict-mirror and save fidelity. | Never reused; minted fresh on add. **Not in the `node_added` payload today**; not required for in-session undo. |
| **node_id** | Transport id `name-<uuid8>` embedded in iceoryx2 service names. | Always fresh (must never collide). Invisible to the frontend. |
| **instance_id** | Sub-patch group-node id (e.g. `subpatch0`). | Regenerated on each group/expand cycle — an action that stores one must remap it after redo. |
| **panel_id** | Workspace layout-node id (`panel-N`). | Regenerated on layout hydrate (`reseedIds`). |
| **bnd_id** | Boundary (In/Out pill) id within an instance. | Auto-incremented; may differ on re-add (read it back from the response/snapshot). |
| **manager instance_id** | The backend session id carried in every snapshot. | Changes on backend restart → triggers the §6 hard reset. |

---

## 1. Core model — `HistoryStore` & `Action`

New file: `frontend/src/lib/stores/history.svelte.ts`.

### 1.1 Action — discriminated union, typed payloads (not closures)

Actions are plain, inspectable, future-serializable data keyed by `kind`. They
carry the minimal delta to replay (`forward`) or reverse (`inverse`) the change,
plus the `NavContext` (§5) where it happened. Closures are deliberately avoided
so actions can be inspected (history panel, agent surface) and, later,
persisted.

```typescript
export type ActionDomain = 'graph' | 'layout';

export interface BaseAction {
  kind: string;
  label: string;            // human label for the undo button/tooltip ("Add Oscillator")
  domain: ActionDomain;
  context: NavContext;      // where it happened — restored for highlight (§5)
}

// ---- graph domain (replayed as RPCs) ----------------------------------
export type GraphAction =
  | BaseAction & { kind: 'add_node'; domain: 'graph';
      payload: { type: string; category: string; pos: [number, number];
                 instId?: string; assignedName?: string /* filled after RPC */ } }
  | BaseAction & { kind: 'remove_node'; domain: 'graph';
      payload: { name: string; node: NodeInstanceInfo; links: LinkInfo[];
                 membership: { instance: string; local_name: string } | null;
                 boundPanels: Array<{ panelId: string; state: unknown }> } }
  | BaseAction & { kind: 'add_link'; domain: 'graph';
      payload: { link: LinkInfo; displaced: LinkInfo | null } }
  | BaseAction & { kind: 'remove_link'; domain: 'graph';
      payload: { link: LinkInfo } }
  | BaseAction & { kind: 'update_param'; domain: 'graph';
      payload: { node: string; group: string; name: string;
                 oldValue: unknown; newValue: unknown } }
  | BaseAction & { kind: 'set_expression'; domain: 'graph';
      payload: { node: string; group: string; name: string;
                 oldExpr: ExprState; newExpr: ExprState } }
  | BaseAction & { kind: 'set_node_pos'; domain: 'graph';
      payload: { name: string; oldPos: [number, number]; newPos: [number, number] } }
  | BaseAction & { kind: 'group_nodes'; domain: 'graph';
      payload: { members: string[]; instId: string; pos?: [number, number] } }
  | BaseAction & { kind: 'expand_instance'; domain: 'graph';
      payload: { instId: string; restoredMembers: string[];
                 interface: Record<string, SubPatchPort> } }
  | BaseAction & { kind: 'duplicate_shared'; domain: 'graph';
      payload: { instId: string; newInstId: string; wasUnique: boolean; pos?: [number, number] } }
  | BaseAction & { kind: 'make_unique'; domain: 'graph';
      payload: { instId: string; defIdBefore: string | null } }
  | BaseAction & { kind: 'add_boundary'; domain: 'graph';
      payload: { instId: string; bndId: string; dir: 'in' | 'out'; dtype: string; pos: [number, number] } }
  | BaseAction & { kind: 'wire_boundary'; domain: 'graph';
      payload: { instId: string; bndId: string;
                 oldInner: { node: string | null; slot: string | null };
                 newInner: { node: string | null; slot: string | null } } }
  | BaseAction & { kind: 'remove_boundary'; domain: 'graph';
      payload: { instId: string; bndId: string; port: SubPatchPort } }
  | BaseAction & { kind: 'set_boundary_pos'; domain: 'graph';
      payload: { instId: string; bndId: string; oldPos: [number, number]; newPos: [number, number] } }
  | BaseAction & { kind: 'load_patch'; domain: 'graph';
      payload: { beforeYaml: string; afterYaml: string;
                 beforeLayout: WorkspaceState | null; afterLayout: WorkspaceState | null;
                 instanceId: string } };

// ---- layout domain (replayed as WorkspaceState snapshot restores) -----
export type LayoutAction = BaseAction & {
  domain: 'layout';
  kind:
    | 'split_panel' | 'close_panel' | 'resize_split' | 'move_panel'
    | 'set_panel_type' | 'link_node_to_panel'
    | 'add_tab' | 'close_tab' | 'duplicate_tab' | 'rename_tab' | 'reorder_tab';
  payload: { before: WorkspaceState; after: WorkspaceState };
};

export type Action = GraphAction | LayoutAction;

export interface ExprState {
  expression: string | null; enabled: boolean;
  triggers_process: boolean; autoeval: boolean;
}
```

> **Layout uses whole-`WorkspaceState` before/after snapshots**, not per-op
> deltas. The trees are small, `$state.snapshot` is cheap, and restore is a
> single assignment — far simpler and more robust than inverting tree edits. We
> keep the op-specific fields (`panelId`, `direction`, …) **out** of replay and
> only in `label`/debug, because the snapshot is the source of truth.

### 1.2 Executors — pure, dependency-injected

An executor is a pair of pure functions for one `kind`, taking injected deps so
units run against fakes:

```typescript
export interface ExecutorDeps {
  control: Control;        // the WS RPC client (real or FakeControl)
  graph: GraphStore;
  workspace: WorkspaceStore;
}

export interface Executor<A extends Action = Action> {
  /** Re-apply (redo). May mutate the action in place to record fresh ids
   *  (e.g. a re-grouped instId) so the next undo targets the right thing. */
  forward(action: A, deps: ExecutorDeps): Promise<void>;
  /** Reverse (undo). */
  inverse(action: A, deps: ExecutorDeps): Promise<void>;
}

export const executors: Record<string, Executor> = { /* one per kind */ };
```

Graph executors call `deps.control.call(...)` **directly** (not the recording
`GraphStore` wrappers of §2.1) and return when the RPC resolves; state reconciles
via the echo event. Layout executors call `deps.workspace.restore(before|after)`
— a **new** thin method that assigns `state` *without* `reseedIds`/migration (so
undo restores ids exactly, unlike `hydrate`). Because both replay paths are
non-recording by construction, no new action is pushed during undo/redo;
`history().suspend` is the belt-and-suspenders guard for any executor that
chooses to reuse a recording store method instead.

### 1.3 HistoryStore API

```typescript
class HistoryStore {
  // reactive for the TopBar
  canUndo = $state(false);
  canRedo = $state(false);
  undoLabel = $state<string | null>(null);   // next undo's label, for tooltip
  redoLabel = $state<string | null>(null);

  /** Record a completed action. No-op while suspended. Clears the redo stack. */
  record(action: Action): void;

  /** Replay the top action's inverse, move it to redo. Restores NavContext
   *  first, then applies the inverse, then highlights. Atomic: on failure the
   *  action stays on the undo stack and an error is surfaced (§6.3). */
  async undo(): Promise<void>;
  async redo(): Promise<void>;

  /** Run fn with recording disabled (used by executors during replay so
   *  inverse RPCs don't push new actions). Reentrant (depth-counted). */
  suspend<T>(fn: () => T): T;
  get isSuspended(): boolean;

  /** Hard reset — only on a new backend session (§6.2). */
  reset(): void;
}

export function history(): HistoryStore;  // lazy singleton, like graph()/workspace()
```

`record` derives `canUndo/canRedo/undoLabel/redoLabel` from the stacks. The
stacks themselves are private; tests assert through the public API only (Iron
Law — never inspect privates).

---

## 2. Recording integration (call-site templates)

Recording happens at the **store-method layer** — the universal choke point that
keyboard handlers, paste/duplicate, the agent façade, and canvas gestures all
funnel through — behind `history().suspend` so replays don't re-record.

### 2.1 Graph template

Each `GraphStore` mutation gains a thin recording wrapper. Pattern: **capture
pre-state synchronously from the store, await the RPC, then record** (for adds,
the assigned name comes from the RPC result; the echo event updates state
independently and recording does not wait for it).

```typescript
// graph.svelte.ts — illustrative for removeNode
async removeNode(name: string): Promise<void> {
  if (!history().isSuspended) {
    const node = this.nodeByName(name);
    const links = this.links.filter(l => l.node_in === name || l.node_out === name)
                            .map(l => ({ ...l }));
    const boundPanels = workspace().panelsBoundTo(name);   // {panelId, state}[] (new helper, §5.4)
    if (node) history().record({
      kind: 'remove_node', domain: 'graph', label: `Delete ${name}`,
      context: captureNavContext(),
      payload: { name, node: structuredClone($state.snapshot(node)), links,
                 membership: node.membership ?? null, boundPanels },
    });
  }
  await getControl().call('remove_node', { name });   // unchanged RPC
}
```

For `addNode`, record **after** the `await` so the backend-assigned name is
known:

```typescript
async addNode(type, category, pos, instId?): Promise<string> {
  const name = (await getControl().call<string>('add_node', { type, category, pos, inst_id: instId })) ?? '';
  if (name && !history().isSuspended) history().record({
    kind: 'add_node', domain: 'graph', label: `Add ${type}`,
    context: captureNavContext(),
    payload: { type, category, pos, instId, assignedName: name },
  });
  return name;
}
```

> Composite helpers (`instantiateNodes`, `cloneNodes`, `removeNodes`) loop these
> primitives, so per decision-log (3) they naturally produce **N entries**.
> They wrap their loop in nothing special — each inner call records itself.

### 2.2 Layout template

Each tracked `WorkspaceStore` method snapshots `before`, mutates, snapshots
`after`, and records — unless suspended:

```typescript
// workspace.svelte.ts — illustrative for split
split(panelId, direction, placeBefore = false, fraction = 0.5, newType?): void {
  const before = this.serialize();
  /* ...existing split logic... */
  if (!history().isSuspended)
    history().record({ kind: 'split_panel', domain: 'layout', label: 'Split panel',
                       context: captureNavContext(),
                       payload: { before, after: this.serialize() } });
}
```

Undo of a layout action calls `history().suspend(() => workspace().restore(before))`
where `restore(s)` assigns `this.state = s` and resets ephemeral
`maximizedPanelId`. The existing debounced `set_layout` push (`AppShell`) then
syncs the backend — harmless and correct.

---

## 3. Graph-domain executors (inverse/forward reference)

One history entry per primitive. The backend never sees "undo" — only normal
RPCs. **Pre-state is read from the authoritative store before the RPC.**

| Op | kind | Pre-state captured | Inverse RPC(s) | Forward RPC(s) | Notes |
|---|---|---|---|---|---|
| addNode | `add_node` | call args; assigned name from RPC result | `remove_node(name)` | `add_node(type,category,pos,inst_id)` → record new name | display name reused on redo |
| removeNode | `remove_node` | node snapshot + its links + membership + bound panels | `add_node(name,params,…)` then `add_link(×links)` then restore bound panels | `remove_node(name)` | see §3.1 |
| addLink | `add_link` | `displaced = isInputConnected? links.find(...)` **before** add | `remove_link(link)` then `add_link(displaced)` if any | `add_link(link)` | single-source rule |
| removeLink | `remove_link` | link record | `add_link(link)` | `remove_link(link)` | — |
| updateParam | `update_param` | `oldValue` from `node.params[group][name]` | `update_param(old)` | `update_param(new)` | shared member: backend re-mirrors |
| setExpression | `set_expression` | prior `ExprState` | `set_expression(old)` | `set_expression(new)` | — |
| setNodePos | `set_node_pos` | `oldPos` from `node.pos` | `set_node_pos(old)` | `set_node_pos(new)` | one RPC per drag at drag-stop |
| groupNodes | `group_nodes` | member names | `expand_instance(instId)` | `group_nodes(members)` → **remap `instId`** | inverse needs only the id |
| expandInstance | `expand_instance` | `restored` names (from RPC), interface | `group_nodes(restored)` → **remap `instId`** | `expand_instance(instId)` | uses existing return |
| duplicateShared | `duplicate_shared` | `wasUnique` (`!instances[id].def_id`) | `remove_instance(newInstId)` (+ `make_unique(instId)` if `wasUnique`) | `duplicate_shared(instId)` → record `newInstId` | — |
| makeUnique | `make_unique` | `defIdBefore` | `duplicate_shared(instId)` if it was shared; else no-op | `make_unique(instId)` | — |
| addBoundary | `add_boundary` | dir/dtype/pos; `bndId` from result | `remove_boundary(instId,bndId)` | `add_boundary(...)` → record `bndId` | id may change on redo |
| wireBoundary | `wire_boundary` | prior `(inner_node,inner_slot)` from `instances[id].interface[bndId]` | `wire_boundary(old)` | `wire_boundary(new)` | backend re-splices |
| removeBoundary | `remove_boundary` | full `SubPatchPort` (+ external links via re-splice on re-add) | `add_boundary` then `wire_boundary(port.inner)` if wired | `remove_boundary(...)` | re-read new `bndId` |
| setBoundaryPos | `set_boundary_pos` | `oldPos` | `set_boundary_pos(old)` | `set_boundary_pos(new)` | — |
| load (any) | `load_patch` | before YAML+layout; after from event | `load_text(beforeYaml)` + restore layout | `load_text(afterYaml)` + restore layout | §6.1 |

Verified call sites: `graph.svelte.ts:310–467`; backend inverses
`manager.py` add_node `:310`, remove_node `:385`, add_link `:432–468`,
group_nodes `:601`, expand_instance `:739`, update_param `:1094`.

### 3.1 removeNode — the cross-domain case

`remove_node` cascades on the backend (drops touching links) and, frontend-side,
`_handle('node_removed')` calls `workspace().clearNodeRefs(name)`, **emptying any
panel bound to the node** (`graph.svelte.ts:227`, `model.ts:264–280`). So the
inverse must restore three things, in order:

1. **The node** — `add_node` with the snapshot's `type`, `category`, `params`,
   `pos`, and `membership` (a unique sub-patch member re-binds via the
   `membership` kwarg the bridge can pass; a *shared* member can't be deleted at
   all — `manager.py:396`). Identity = the **same display name** (snapshot
   `name`). Members go back via `add_member_node` so the namespaced name and
   membership map are rebuilt.
2. **Its links** — `add_link` for each captured link (endpoints are display
   names, valid again once the node is back).
3. **Its panel bindings** — re-apply each `boundPanels[i].state` via
   `workspace().setPanelState(panelId, state)` (suspended, so it doesn't record
   a layout action).

Pre-state for (3) is captured by a new pure helper
`workspace().panelsBoundTo(name)` that scans every workspace root for panels
whose `linkedNodeName(state) === name` (same predicate as `clearNodeRef`).

### 3.2 Identity, collisions, shared mirroring (hard cases)

- **Name re-take on undo-of-delete.** `add_node(force_name=True)` **raises
  `KeyError` if the name is occupied** (`manager.py:83`) — it does not
  auto-increment. MVP behavior: the inverse RPC rejects → undo fails atomically
  (action restored to stack + toast: *"Can't undo delete: name 'oscillator0' is
  in use"*). Auto-rename-and-remap is **deferred** (§11).
- **instId remap on group/expand redo.** Redoing a group/expand mints a fresh
  `instId`. The executor's `forward` writes the new id back into
  `action.payload.instId` so the next inverse targets it. Same for
  `duplicate_shared`/`add_boundary` ids.
- **Shared-member param/pos.** The backend mirrors an edit to all siblings
  (`manager.py:1115–1123`, `:1154–1165`). The inverse issues **one** RPC on the
  edited node; the backend re-mirrors the revert. The action stores only the
  primary edit.

---

## 4. Layout-domain executors (snapshot model)

Tracked ops (decision-log 6) each record `{ before, after }` full
`WorkspaceState`. **Undo** = restore `before`; **redo** = restore `after`; both
via `workspace().restore(state)` under `suspend`. The debounced `set_layout`
push follows naturally.

| Op (`workspace.svelte.ts`) | kind | Tracked? |
|---|---|---|
| `split` `:131` | `split_panel` | ✅ |
| `close` `:146` | `close_panel` | ✅ |
| `resize` `:155` | `resize_split` | ✅ |
| `dropOnPanel` `:304`, `dropPanelOnTabBar` `:326` | `move_panel` | ✅ |
| `setType` `:159` | `set_panel_type` | ✅ |
| `linkNodeToPanel` `:174` | `link_node_to_panel` | ✅ (user drag-binding only; distinguish from internal `setPanelState`) |
| `addTab` `:208`, `closeTab` `:250`, `duplicateTab` `:235`, `renameTab` `:225`, `reorderTab` `:345` | `add_tab` / `close_tab` / `duplicate_tab` / `rename_tab` / `reorder_tab` | ✅ |
| `toggleMaximize` `:194`, `setActive` `:168`, `selectTab` `:217` | — | ❌ navigation/ephemeral |
| `setPanelState` `:164` (internal editor zoom/selection) | — | ❌ not a user layout edit |
| `hydrate` / `reset` / `serialize` | — | ❌ lifecycle/read |

> **`linkNodeToPanel` vs `setPanelState`.** Both end up calling `setPanelState`,
> but only the *user drag-binding* records a `link_node_to_panel` action. The
> recording wrapper lives on `linkNodeToPanel` specifically; the generic
> `setPanelState` (called by panel content to persist its own zoom/scroll/
> selection) never records.
>
> **Ephemeral fields are not in the snapshot.** `maximizedPanelId` and
> `activePanelId` are restored via `NavContext`, not via the `WorkspaceState`
> snapshot (they aren't part of it). Internal panel state (a node editor's zoom)
> is **not** restored by layout undo — documented limitation (§11).

---

## 5. Navigation/focus context & highlight-on-undo

### 5.1 The token

```typescript
export interface NavContext {
  activeWorkspaceId: string;
  activePanelId: string | null;
  /** per editor panel: the sub-patch instance-id stack (root → deepest). */
  enteredPath: Record<string, string[]>;
  /** selection at record time, per panel. */
  selection: Record<string, { nodes: string[]; edges: string[] }>;
}
```

Viewport (pan/zoom) is **excluded** — on restore we `fitView`/center on the
affected node instead of restoring exact pan/zoom.

### 5.2 Capture & restore (pure helpers)

New file `frontend/src/lib/workspace/navContext.ts`:

- `captureNavContext(): NavContext` — read `workspace` + `selection` +
  per-panel `enteredPath`.
- `restoreNavContext(ctx): Promise<void>` — the ordered sequence:
  1. `workspace().selectTab(ctx.activeWorkspaceId)` (suspended).
  2. `workspace().setActive(ctx.activePanelId)`, `selection().activeEditorId = …`.
  3. Drive each editor's `enteredPath` to match via `enterInstance`/`exitToDepth`
     (best-effort — see §5.3).
  4. *(caller then applies the inverse/forward mutation)*
  5. Set selection to the affected node(s)/edge(s).
  6. `fitView`/center + a brief **highlight pulse** (~600 ms CSS class) on the
     changed node/param (mirrors goofi3's param-highlight affordance).

The undo/redo flow is therefore: **restore context (1–3) → apply mutation →
highlight (5–6).**

### 5.3 Multi-panel & best-effort fallback

- If `ctx.activePanelId` no longer exists (panel closed/retyped), fall back to
  any node-editor panel, else the active panel.
- If a recorded `enteredPath` references a sub-patch that no longer exists, pop
  to the nearest valid depth.
- After undoing past a `load_patch` (§6.1), `enteredPath` is **reset to root**
  for all panels — the prior patch's sub-patches may not exist; the user
  re-navigates if desired (documented).

### 5.4 Cross-domain coupling

`remove_node`'s inverse restores panel bindings (§3.1) using the `boundPanels`
snapshot, so undoing a delete reinstates the Parameters/Viewer/Metadata panels
that were emptied. `clearNodeRefs` itself is **never** its own history entry —
it's a side-effect folded into the `remove_node` action.

---

## 6. History lifecycle

### 6.1 Loads are undoable checkpoints (never-clear, made safe)

`load` / `load_text`, and the resulting `graph_replaced`, record one
`load_patch` action:

- **before:** `await graph().serialize()` (current YAML) + `workspace().serialize()`,
  captured **before** issuing the load RPC.
- **after:** the new YAML (`serialize()` after the `graph_replaced` snapshot is
  applied) + the hydrated layout.
- **inverse:** `load_text(beforeYaml)` + `workspace().hydrate(beforeLayout)`;
  reset `enteredPath` to root.
- **forward:** `load_text(afterYaml)` + hydrate `afterLayout`.

This keeps one coherent timeline across loads; nothing dangles.

### 6.2 The one hard reset: backend restart

`_replaceSnapshot` already detects a changed manager `instance_id`
(`graph.svelte.ts:83`) and `_onWholesaleLoad` runs for a fresh session
(`:98`). We call `history().reset()` there **only** when `instance_id` changed
(not on a same-session reconnect, not on an in-session load). This is the single
exception to never-clear (decision-log 5): the authoritative world is gone, so
replaying old RPCs is invalid.

### 6.3 Error recovery (atomic-or-nothing)

`undo()`/`redo()` wrap the executor replay in try/catch:

- On RPC rejection (`control.ts` `call()` rejects on a backend error), **re-push
  the action onto the stack it came from**, leave the other stack unchanged, and
  surface a toast. The user can retry or continue.
- Specifically handled: undo-of-delete onto a re-taken name (`KeyError`), undo of
  an op whose target node/instance vanished (e.g. removed by a later action that
  was itself undone out of order — guarded by the LIFO ordering, but defended
  anyway).

---

## 7. Backend / bridge changes — minimal & gated

**The core design needs no backend changes.** The items below are **optional,
additive, backward-compatible robustness**, each gated behind a Phase-3 test
that demonstrates the frontend-only path is insufficient. If the tests pass
without them, they are dropped.

| # | Change | File | Why it might help | Gate |
|---|---|---|---|---|
| O1 | Thread `member_uid` through bridge `add_node` (and `add_member_node`) | `control.py:165–172`, `:156` | Preserve stable identity across undo for **save fidelity** (not in-session correctness). **Also requires** adding `member_uid` to `describe_node_instance` so the frontend can capture it (`schemas.py:80`). | Only if a save-after-undo round-trip test shows identity drift that matters. |
| O2 | Return `displaced_link` in `add_link` response | `control.py:182–186` | Confirmation of the frontend's pre-RPC capture. | Only if frontend pre-capture proves racy under the single-source rule. |
| O3 | Return `member_renames` / `spliced_links` in `group_nodes` response | `control.py:289–296` | Avoid diffing the `subpatch_changed` snapshot. | Likely unnecessary: group's inverse is `expand_instance(instId)`; expand's inverse is `group_nodes(restored)` using the value expand **already returns**. |
| — | `restore_node` RPC | — | Full-fidelity node restore | **Deferred.** Only if Phase-3 proves `add_node(name, params, membership)` can't restore a deleted node's needed state (e.g. expression config not present in the params snapshot). §3 assumes it can; the test decides. |

What `add_node` already restores vs. not (informs the gate): restores params
(incl. expression source/flags per `params` serialization), `pos`, `membership`;
does **not** restore viewer state (frontend-owned, separately tracked) or
runtime/transport state (re-established fresh — correct).

---

## 8. Keybindings, TopBar & agent surface

- **Global keybindings** in `AppShell.onKeydown` (`:125`): `Ctrl/Cmd+Z` → undo,
  `Ctrl/Cmd+Shift+Z` and `Ctrl+Y` → redo. Gated against `INPUT/TEXTAREA/SELECT`
  focus and the `ExpressionModal` (add `ui().modalOpen` guard if absent).
  Coexists with existing `Ctrl+S/O` and per-panel `Ctrl+A/C/V/D/G` — undo/redo
  are global and fire regardless of active panel; panel shortcuts still require
  an active panel.
- **TopBar** (`TopBar.svelte`): undo/redo buttons, `disabled={!history().canUndo}` /
  `canRedo`, tooltips showing `undoLabel`/`redoLabel`.
- **Agent surface:** add `undo()` / `redo()` to `agent/commands.ts` and
  `canUndo` / `canRedo` / `historyLength` / stack labels to `agent/query.ts`, so
  Playwright e2e and the future AI panel can drive and assert history through
  `window.goofi`.
- **Future (not MVP):** a History panel as a new workspace panel type listing
  entries (cheap given actions are inspectable data).

---

## 9. TDD strategy & phased plan

**Iron Law:** no production code without a failing test first. Tests verify
*behavior through real data paths*, never a mock's internals. The `FakeControl`
is dumb — it records RPCs and **only emits events when the test explicitly
tells it to**, so a test must drive the event stream and assert on resulting
store state (not "an RPC was recorded").

### 9.1 Testable seams

1. **`HistoryStore`** — pure, synchronous, no I/O. Public-API tests only.
2. **Executors** — factory/DI: `(deps) => Executor`. Tests inject fakes.
3. **`FakeControl`** (`frontend/src/lib/test/fakeControl.ts`, new) — implements
   `Control` (`call`, `on`, `onConnect`); `call` records and returns a resolved
   promise; `emit(event)` synchronously fans out to listeners; `recordedCalls()`
   for inspection. Does **not** auto-synthesize echo events.
4. **NavContext helpers** — pure capture; deterministic restore.
5. **Layout snapshot/restore** — pure over `WorkspaceState` (reuse
   `model.test.ts` patterns).

### 9.2 Phases (each RED → GREEN → REFACTOR)

**Phase 1 — HistoryStore core (unit).** `history.test.ts`: push→canUndo;
undo pops & moves to redo; redo; `suspend` blocks recording; `suspend` returns
resume; `reset` clears; canUndo/canRedo empties; `record` clears redo stack.

**Phase 2 — simple graph executors (unit + FakeControl).**
`graph.executor.test.ts`: add/remove node, add/remove link, move, param,
expression — each "RPC then `emit(event)` updates the store"; then history
round-trips. Keystone: **"undo of removeNode re-adds the node with the same
display name and restores its links"** (add `oscillator0`, add another, wire,
remove `oscillator0`, undo, assert the `add_node` RPC carries `oscillator0`,
`emit('node_added')`+`emit('link_added')`, assert graph matches pre-removal).
Plus **"suspend blocks recording during a 5-node batch"**.

**Phase 3 — composite + sub-patch executors + (gated) backend (unit +
integration + pytest).** group/expand round-trip (uses `inst_id`/`restored`,
no backend change); duplicate_shared/make_unique; boundaries; the cross-domain
**"undo of removeNode restores its bound panels"**. Backend tests in
`tests/test_undo_support.py` run **only for whichever O1–O3 the gates select** —
e.g. if O2 is needed: *adding a link to an occupied input returns
`displaced_link` and tears down the prior link*. (Correction to the draft: a
node is placed in an instance via **`add_member_node`/`membership`**, not via
`member_uid`; the member_uid test, if O1 is taken, asserts `_refs_by_uid[uid]`
identity only.)

**Phase 4 — layout executors (unit).** `layout.executor.test.ts`: split records
`{before,after}`; **"undo of closePanel restores the panel with its id"**;
move/resize/tab ops; `setPanelState`-internal does **not** record;
`linkNodeToPanel` does.

**Phase 5 — NavContext + cross-domain + load_patch (unit + e2e).**
`navContext.test.ts`: capture returns per-panel `enteredPath`; restore re-enters
sub-patches and restores focus. `graph.executor.test.ts`: load records a
`load_patch` entry; **"undo past load_patch restores prior patch + resets
enteredPath to root"**. First e2e via `window.goofi`.

**Phase 6 — keybindings, TopBar, agent surface, highlight, polish (unit +
e2e).** `Ctrl+Z`/`Ctrl+Shift+Z` invoke undo/redo; TopBar disabled states;
highlight pulse appears and fades. e2e (`e2e/test_undo_redo.py`):
add/remove node, link, group/expand, layout split/close, load_patch, and
**"backend restart clears the stack"**.

### 9.3 Verification gates

After each phase the named tests pass with pristine output. Backend restart
resets; undo/redo work across graph + layout on one stack; the stress patch
still runs; `pytest tests/` stays green.

---

## 10. Top risks

1. **Async reconciliation timing.** Undo issues an inverse RPC and returns when
   it resolves, but store state updates on the echo event. The highlight step
   must run after the echo (await the relevant event, or a microtask) or center
   on a node that isn't placed yet. *Mitigation:* `restoreNavContext` step 5–6
   waits for the expected event kind (a small `awaitEvent(pred, timeout)` helper)
   before highlighting.
2. **instId/bndId churn on redo.** Forgetting to remap a regenerated id after
   redo breaks the next undo. *Mitigation:* executors write fresh ids back into
   the action; Phase-3 tests cover a group→undo→redo→undo cycle.
3. **Layout snapshot vs. live debounce.** A layout undo assigns `state`, then the
   `AppShell` effect pushes `set_layout`. Ensure the push isn't mistaken for a
   new user edit (it isn't — only `WorkspaceStore` methods record, and restore is
   suspended).
4. **Name collisions on undo-of-delete.** Graceful-fail is acceptable but
   surprising; surface a clear toast. Auto-rename+remap is the deferred upgrade.
5. **Expression/viewer state fidelity.** If a deleted node's expression config
   isn't fully in the params snapshot, undo loses it → triggers gate O1/`restore_node`.
   Phase-3 test decides.

## 11. Deferred / out of scope (recorded, not chosen)

- Auto-rename + link-remap when undo-of-delete hits a name collision.
- `member_uid` preservation / `restore_node` RPC unless a gate (§7) fires.
- History persistence across browser reload or `localStorage` (the stack is
  in-memory; a reload starts fresh — the user did not request persistence).
- Restoring a node editor's internal pan/zoom and a panel's internal scroll on
  layout undo.
- Time-window coalescing (explicitly rejected — one entry per primitive).
- A dedicated History panel UI (cheap to add later; not MVP).
- Stack-size pruning (unbounded for now; revisit if memory shows growth).
