# Per-viewer-instance view state — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every viewer (the inline node viewer and each docked Viewer panel) an independent instance with its own kind + settings, while all instances of a slot share one data stream.

**Architecture:** Split "data source" `(node, slot, dtype)` (shared, unchanged) from "view state" `kind`+`settings` (per-instance). A single `ViewBinding` interface is the only point of variation: the shared viewer components consume a binding; the inline placement backs it with a node-scoped store (persists to `node.viewers`), each panel backs it with its own layout state. Clean rewrite of the view-state layer — no back-compat.

**Tech Stack:** Svelte 5 runes, TypeScript (strict), Vitest-style not present → frontend unit tests run via `npx vitest` if configured else pure-function tests live in `*.test.ts` run by the project's test runner; Playwright (`e2e/`, gitignored) driven through `window.goofi.commands`.

## Global Constraints

- TypeScript strict; no `any` in app code (codec layer excepted).
- Do **not** touch the data plane: `frames.ts` refcount, `data.ts`, `bridge/data.py`, `codec`, backend `_SlotMux` stay as-is. One stream per `(node, slot)`.
- No backend changes at all in this work.
- No `.gfi` migration / back-compat: old viewer state need not be honored.
- Reduction is out of scope (deferred).
- Persistence by placement: inline → `node.viewers[slot]` (with the collapse flag); panel → that panel's layout-state blob.
- Delete anything left unused by the rewrite (old stores, dead exports/imports).
- After implementation: a code-review pass and a code-simplify pass (subagents), fixing findings.

---

## Verification commands (used throughout)

- Typecheck: `cd frontend && npx svelte-check --threshold error` → expect `0 ERRORS`.
- Build: `cd frontend && npm run build` → expect `✔ done`.
- Frontend unit (pure helpers): `cd frontend && npx vitest run src/lib/viewers/<file>.test.ts` (if vitest absent, see Task 1 Step 0).
- e2e (run from repo root; clean SHM first): `rm -f /dev/shm/iox2_* 2>/dev/null; .venv/bin/python -m pytest e2e/test_viewers.py -q`

---

## Task 1: Pure resolution helpers

Extract the dtype-forcing and defaults-merge logic so both bindings resolve identically. These are the only pure-unit-testable pieces.

**Files:**
- Modify: `frontend/src/lib/viewers/kind.ts` (add `resolveKind`)
- Modify: `frontend/src/lib/viewers/settingsSchema.ts` (add `resolveSettings`)
- Test: `frontend/src/lib/viewers/viewResolve.test.ts` (new)

**Interfaces:**
- Produces:
  - `resolveKind(dtype: string | null, stored: ViewerKind | undefined): ViewerKind`
  - `resolveSettings(kind: ViewerKind, overrides: SettingsMap | undefined): SettingsMap`

- [ ] **Step 0: Confirm the test runner.**
Run: `cd frontend && cat package.json | grep -A2 '"scripts"' && ls vitest.config.* 2>/dev/null`
If no vitest, run `npx vitest --version` (it's a dev dep of SvelteKit projects). If truly unavailable, write the tests anyway as `*.test.ts` and run with `npx vitest run`; if that fails, fold the assertions into a tiny `node --test` script — but vitest is expected.

- [ ] **Step 1: Write failing tests.**

```ts
// frontend/src/lib/viewers/viewResolve.test.ts
import { describe, it, expect } from 'vitest';
import { resolveKind } from './kind';
import { resolveSettings } from './settingsSchema';

describe('resolveKind', () => {
  it('forces string/table viewers by dtype regardless of stored kind', () => {
    expect(resolveKind('STRING', 'image')).toBe('string');
    expect(resolveKind('TABLE', 'line')).toBe('table');
  });
  it('uses the stored kind for ARRAY, falling back to line', () => {
    expect(resolveKind('ARRAY', 'image')).toBe('image');
    expect(resolveKind('ARRAY', undefined)).toBe('line');
  });
  it('falls back to line for null dtype', () => {
    expect(resolveKind(null, undefined)).toBe('line');
  });
});

describe('resolveSettings', () => {
  it('merges overrides over the kind defaults', () => {
    const merged = resolveSettings('line', { logY: true });
    expect(merged.logY).toBe(true);
    // a default key from the line schema is still present
    expect(Object.keys(merged).length).toBeGreaterThan(1);
  });
  it('returns pure defaults when overrides are absent', () => {
    expect(resolveSettings('image', undefined)).toEqual(resolveSettings('image', {}));
  });
});
```

- [ ] **Step 2: Run, verify FAIL.**
Run: `cd frontend && npx vitest run src/lib/viewers/viewResolve.test.ts`
Expected: FAIL (`resolveKind`/`resolveSettings` not exported).

- [ ] **Step 3: Implement `resolveKind` in `kind.ts`.**
Append (uses the existing `ViewerKind` type already in this file):

```ts
/** The viewer kind to actually use: STRING/TABLE slots force their dedicated
 * viewer; ARRAY (and anything else) uses the stored kind, defaulting to line. */
export function resolveKind(dtype: string | null, stored: ViewerKind | undefined): ViewerKind {
	if (dtype === 'STRING') return 'string';
	if (dtype === 'TABLE') return 'table';
	return stored ?? 'line';
}
```

- [ ] **Step 4: Implement `resolveSettings` in `settingsSchema.ts`.**
It already exports `defaultSettings(kind)` and `SettingValue`. Add a `SettingsMap` re-export if not present, then:

```ts
export type SettingsMap = Record<string, SettingValue>;

/** Resolved settings for a kind: its declared defaults with the explicit
 * overrides applied on top. */
export function resolveSettings(kind: ViewerKind, overrides: SettingsMap | undefined): SettingsMap {
	return { ...defaultSettings(kind), ...(overrides ?? {}) };
}
```
(If `ViewerKind` isn't imported in `settingsSchema.ts`, add `import type { ViewerKind } from './kind';`. If `SettingsMap` is already defined in `viewerSettings.svelte.ts`, that definition is removed in Task 2 — `settingsSchema.ts` becomes its home.)

- [ ] **Step 5: Run, verify PASS.**
Run: `cd frontend && npx vitest run src/lib/viewers/viewResolve.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit.**
```bash
git add frontend/src/lib/viewers/kind.ts frontend/src/lib/viewers/settingsSchema.ts frontend/src/lib/viewers/viewResolve.test.ts
git commit -m "feat(viewers): pure resolveKind/resolveSettings helpers"
```

---

## Task 2: Merged inline-view store (replaces the two parallel maps)

Consolidate `viewerState.svelte.ts` (kind map) + `viewerSettings.svelte.ts` (settings map) into one node-scoped inline-view store holding `{kind?, settings}`, and repoint `graph.svelte.ts` + `commands.ts`. This is the inline placement's backing store.

**Files:**
- Create: `frontend/src/lib/viewers/inlineView.svelte.ts`
- Delete: `frontend/src/lib/viewers/viewerState.svelte.ts`, `frontend/src/lib/viewers/viewerSettings.svelte.ts`
- Modify: `frontend/src/lib/stores/graph.svelte.ts`, `frontend/src/lib/agent/commands.ts`

**Interfaces:**
- Consumes: `SettingsMap` (now from `settingsSchema.ts`), `ViewerKind`, `resolveKind`, `resolveSettings` (Task 1).
- Produces (module `inlineView.svelte.ts`):
  - `rawInlineView(node: string, slot: string): { kind?: ViewerKind; settings: SettingsMap }`
  - `setInlineKind(node: string, slot: string, kind: ViewerKind): void`
  - `setInlineSetting(node: string, slot: string, key: string, value: SettingValue): void`
  - `seedInlineView(node: string, slot: string, view: { kind?: ViewerKind; settings?: SettingsMap } | undefined): void`
  - `forgetInlineView(node: string): void`

- [ ] **Step 1: Create the merged store.**

```ts
// frontend/src/lib/viewers/inlineView.svelte.ts
/**
 * Per-(node, slot) INLINE viewer view-state: the kind + settings of the viewer
 * shown on the node body. One entry per slot; seeded from node.viewers on load,
 * pushed back (debounced) via graph.pushNodeViewers. Kept separate from the node
 * DATA record so a node state-update/snapshot replacement never clobbers live
 * view edits. Docked Viewer PANELS do NOT use this — each panel owns its view
 * state in its own layout state (see viewBinding.panelBinding).
 */
import type { ViewerKind } from './kind';
import type { SettingValue, SettingsMap } from './settingsSchema';

interface InlineView {
	kind?: ViewerKind;
	settings: SettingsMap;
}

function key(node: string, slot: string): string {
	return `${node}|${slot}`;
}

const store = $state<Record<string, InlineView>>({});

/** Raw stored view for a slot (no defaults applied). */
export function rawInlineView(node: string, slot: string): InlineView {
	return store[key(node, slot)] ?? { settings: {} };
}

export function setInlineKind(node: string, slot: string, kind: ViewerKind): void {
	const id = key(node, slot);
	store[id] = { ...(store[id] ?? { settings: {} }), kind };
}

export function setInlineSetting(node: string, slot: string, k: string, value: SettingValue): void {
	const id = key(node, slot);
	const cur = store[id] ?? { settings: {} };
	store[id] = { ...cur, settings: { ...cur.settings, [k]: value } };
}

/** Seed a slot's inline view from a restored patch (no-op when empty). */
export function seedInlineView(
	node: string,
	slot: string,
	view: { kind?: ViewerKind; settings?: SettingsMap } | undefined
): void {
	if (!view) return;
	const hasKind = view.kind != null;
	const hasSettings = view.settings && Object.keys(view.settings).length > 0;
	if (!hasKind && !hasSettings) return;
	store[key(node, slot)] = { kind: view.kind, settings: { ...(view.settings ?? {}) } };
}

/** Drop every slot's inline view for a node that no longer exists. */
export function forgetInlineView(node: string): void {
	const prefix = `${node}|`;
	for (const k of Object.keys(store)) if (k.startsWith(prefix)) delete store[k];
}
```

- [ ] **Step 2: Repoint `graph.svelte.ts`.**
Replace the two imports (lines ~26-27):
```ts
import { seedInlineView, forgetInlineView, rawInlineView } from '$lib/viewers/inlineView.svelte';
import { resolveKind } from '$lib/viewers/kind';
import type { SettingsMap } from '$lib/viewers/settingsSchema';
```
In `_replaceSnapshot` (the `forgetViewerKinds`/`forgetViewerSettings` loop, ~68-69) replace both with:
```ts
			forgetInlineView(old.name);
```
In `_seedNodeViewerState` (~110-117) replace the two `seed*` calls with:
```ts
		for (const slot of slots) {
			const v = node.viewers?.[slot];
			seedInlineView(node.name, slot, {
				kind: v?.kind as ViewerKind | undefined,
				settings: v?.settings as SettingsMap | undefined
			});
		}
```
In `pushNodeViewers` (~135-137) replace the per-slot record build with:
```ts
				const view = rawInlineView(node, slot);
				viewers[slot] = {
					collapsed: !ui().isSlotExpanded(node, slot),
					kind: resolveKind(n.output_slots[slot], view.kind),
					settings: view.settings
				};
```
In the `subpatch_changed` handler (~183-184) replace the two `forget*` calls with `forgetInlineView(old.name);`.
In the `node_removed` handler (~222-223) replace the two `forget*` calls with `forgetInlineView(ev.payload.name);`.

- [ ] **Step 3: Repoint `commands.ts`.**
Replace imports (lines 16-17):
```ts
import { setInlineKind, setInlineSetting } from '$lib/viewers/inlineView.svelte';
```
Update the two command bodies (~100-106):
```ts
	setViewerKind: (node: string, slot: string, kind: ViewerKind): void => {
		setInlineKind(node, slot, kind);
		graph().pushNodeViewers(node);
	},
	setViewerSetting: (node: string, slot: string, key: string, value: boolean | number | string): void => {
		setInlineSetting(node, slot, key, value);
		graph().pushNodeViewers(node);
	},
```

- [ ] **Step 4: Delete the old modules.**
```bash
rm frontend/src/lib/viewers/viewerState.svelte.ts frontend/src/lib/viewers/viewerSettings.svelte.ts
```

- [ ] **Step 5: Typecheck (expected to surface the remaining consumers).**
Run: `cd frontend && npx svelte-check --threshold error 2>&1 | tail -20`
Expected: errors ONLY in `ViewerControls.svelte`, `ViewerSettingsMenu.svelte`, `ViewerFeed.svelte`, `SlotViewer.svelte`, `ViewerPanel.svelte` (fixed in Tasks 4-6). No errors in `graph.svelte.ts` / `commands.ts`. If others surface (e.g. `grep -rn "viewerState.svelte\|viewerSettings.svelte" src/`), repoint them here.

- [ ] **Step 6: Commit (WIP — components still broken; that's expected mid-refactor).**
```bash
git add -A
git commit -m "refactor(viewers): merge inline kind+settings into one node-scoped store"
```

---

## Task 3: ViewBinding + factories

The single seam. Pure-ish module: the type plus two factories.

**Files:**
- Create: `frontend/src/lib/viewers/viewBinding.ts`
- Test: `frontend/src/lib/viewers/viewBinding.test.ts`

**Interfaces:**
- Consumes: `resolveKind`, `resolveSettings`, `rawInlineView`/`setInlineKind`/`setInlineSetting`, `graph().pushNodeViewers`, `asStateObject`.
- Produces:
  - `interface ViewBinding { readonly kind: ViewerKind; readonly settings: SettingsMap; setKind(kind: ViewerKind): void; setSetting(key: string, value: SettingValue): void; }`
  - `inlineBinding(node: string, slot: string, dtype: string | null): ViewBinding`
  - `panelBinding(getState: () => unknown, setState: (s: unknown) => void, dtype: string | null): ViewBinding`

- [ ] **Step 1: Write failing tests (panel binding is store-free → unit-testable).**

```ts
// frontend/src/lib/viewers/viewBinding.test.ts
import { describe, it, expect } from 'vitest';
import { panelBinding } from './viewBinding';

describe('panelBinding', () => {
  it('resolves kind/settings from the panel state and writes back via setState', () => {
    let state: Record<string, unknown> = { node: 'n', slot: 's' };
    const b = panelBinding(() => state, (s) => { state = s as Record<string, unknown>; }, 'ARRAY');
    expect(b.kind).toBe('line');           // default
    b.setKind('image');
    expect(state.kind).toBe('image');
    expect(b.kind).toBe('image');          // re-resolves from updated state
    b.setSetting('colormap', 'viridis');
    expect((state.settings as Record<string, unknown>).colormap).toBe('viridis');
    expect(b.settings.colormap).toBe('viridis');
  });
  it('forces string viewer for STRING dtype regardless of stored kind', () => {
    const state = { kind: 'image' };
    const b = panelBinding(() => state, () => {}, 'STRING');
    expect(b.kind).toBe('string');
  });
});
```

- [ ] **Step 2: Run, verify FAIL.**
Run: `cd frontend && npx vitest run src/lib/viewers/viewBinding.test.ts`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement.**

```ts
// frontend/src/lib/viewers/viewBinding.ts
/**
 * One viewer, many instances: a ViewBinding is the per-instance view state
 * (kind + settings) behind a viewer. The shared viewer components consume a
 * binding and never read a store directly, so the inline node viewer and every
 * docked panel are the same component differing only in where their binding
 * persists. The data source (node, slot, dtype) is separate and shared.
 */
import { resolveKind } from './kind';
import { resolveSettings, type SettingValue, type SettingsMap } from './settingsSchema';
import type { ViewerKind } from './kind';
import {
	rawInlineView,
	setInlineKind,
	setInlineSetting
} from './inlineView.svelte';
import { graph } from '$lib/stores/graph.svelte';
import { asStateObject } from '$lib/workspace/panelState';

export interface ViewBinding {
	readonly kind: ViewerKind;
	readonly settings: SettingsMap;
	setKind(kind: ViewerKind): void;
	setSetting(key: string, value: SettingValue): void;
}

/** Inline viewer (node body): backed by the node-scoped inline-view store;
 * mutations persist into node.viewers via pushNodeViewers. */
export function inlineBinding(node: string, slot: string, dtype: string | null): ViewBinding {
	return {
		get kind() {
			return resolveKind(dtype, rawInlineView(node, slot).kind);
		},
		get settings() {
			return resolveSettings(this.kind, rawInlineView(node, slot).settings);
		},
		setKind(kind) {
			setInlineKind(node, slot, kind);
			graph().pushNodeViewers(node);
		},
		setSetting(key, value) {
			setInlineSetting(node, slot, key, value);
			graph().pushNodeViewers(node);
		}
	};
}

/** Docked Viewer panel: backed by the panel's own layout state blob. */
export function panelBinding(
	getState: () => unknown,
	setState: (s: unknown) => void,
	dtype: string | null
): ViewBinding {
	const raw = () => asStateObject(getState());
	return {
		get kind() {
			return resolveKind(dtype, raw().kind as ViewerKind | undefined);
		},
		get settings() {
			return resolveSettings(this.kind, (raw().settings as SettingsMap) ?? {});
		},
		setKind(kind) {
			setState({ ...raw(), kind });
		},
		setSetting(key, value) {
			setState({ ...raw(), settings: { ...((raw().settings as SettingsMap) ?? {}), [key]: value } });
		}
	};
}
```

- [ ] **Step 4: Run, verify PASS.**
Run: `cd frontend && npx vitest run src/lib/viewers/viewBinding.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit.**
```bash
git add frontend/src/lib/viewers/viewBinding.ts frontend/src/lib/viewers/viewBinding.test.ts
git commit -m "feat(viewers): ViewBinding seam + inline/panel factories"
```

---

## Task 4: Make the shared viewer components binding-driven

Strip the `(node,slot)` store reads out of `ViewerControls`, `ViewerSettingsMenu`, `ViewerFeed`; they take a `ViewBinding`.

**Files:**
- Modify: `frontend/src/lib/viewers/ViewerControls.svelte`, `ViewerSettingsMenu.svelte`, `ViewerFeed.svelte`

**Interfaces:**
- Consumes: `ViewBinding` (Task 3).
- Produces (new prop shapes):
  - `ViewerControls`: `{ dtype: string; binding: ViewBinding }`
  - `ViewerSettingsMenu`: `{ binding: ViewBinding }`
  - `ViewerFeed`: `{ node: string; slot: string | null; dtype: string | null; binding: ViewBinding }`

- [ ] **Step 1: `ViewerControls.svelte` — script.** Replace the `<script>` body with:

```svelte
<script lang="ts">
	import ViewerSettingsMenu from './ViewerSettingsMenu.svelte';
	import { ARRAY_KINDS, type ViewerKind } from './kind';
	import type { ViewBinding } from './viewBinding';

	let { dtype, binding }: { dtype: string; binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);

	function onKindChange(e: Event): void {
		e.stopPropagation();
		binding.setKind((e.currentTarget as HTMLSelectElement).value as ViewerKind);
	}
</script>
```
Then in the template change `<ViewerSettingsMenu {node} {slot} {kind} />` to `<ViewerSettingsMenu {binding} />`. (The `<select class="kind">` block is unchanged; it already binds `value={kind}` + `onchange={onKindChange}`.)

- [ ] **Step 2: `ViewerSettingsMenu.svelte` — script.** Replace the prop + store usage:

```svelte
<script lang="ts">
	import { settingsSchemaFor, type SettingDescriptor, type SettingValue } from './settingsSchema';
	import type { ViewBinding } from './viewBinding';
	import { portal } from '$lib/workspace/portal';

	let { binding }: { binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);
	const groups = $derived(settingsSchemaFor(kind));
	const settings = $derived(binding.settings);

	let open = $state(false);
	let anchor = $state<{ x: number; y: number }>({ x: 0, y: 0 });
	let collapsed = $state<Record<string, boolean>>({});

	const MENU_W = 212;
	// ... keep toggle(), visible() unchanged ...
	function set(key: string, value: SettingValue): void {
		binding.setSetting(key, value);
	}
	// ... keep toggleGroup() unchanged ...
</script>
```
Remove the `viewerSettings`/`setViewerSetting` and `graph` imports and the `node`/`slot` props. Keep the rest of the component (template + the `toggle`/`visible`/`toggleGroup` functions) verbatim.

- [ ] **Step 3: `ViewerFeed.svelte` — script.** Replace the props + kind/settings derivation:

```svelte
<script lang="ts">
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import ViewerSurface from './ViewerSurface.svelte';
	import type { ViewBinding } from './viewBinding';
	import { onMount } from 'svelte';

	let {
		node,
		slot,
		dtype,
		binding
	}: { node: string; slot: string | null; dtype: string | null; binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);
	const settings = $derived(binding.settings);
	// ... rest (frame state, IntersectionObserver, subscribe effect, template) unchanged ...
</script>
```
`dtype` stays a prop (kept for parity / future), but is no longer read for kind. Remove the `viewerKind`/`viewerSettings` imports.

- [ ] **Step 4: Typecheck the three components compile against the new contract.**
Run: `cd frontend && npx svelte-check --threshold error 2>&1 | tail -20`
Expected: errors now ONLY in `SlotViewer.svelte` and `ViewerPanel.svelte` (the callers, fixed next).

- [ ] **Step 5: Commit.**
```bash
git add frontend/src/lib/viewers/ViewerControls.svelte frontend/src/lib/viewers/ViewerSettingsMenu.svelte frontend/src/lib/viewers/ViewerFeed.svelte
git commit -m "refactor(viewers): ViewerControls/SettingsMenu/Feed are binding-driven"
```

---

## Task 5: Wire the inline host (`SlotViewer`)

Provide the inline binding; move collapse persistence to the mutation site; delete the old "settled" push effect.

**Files:**
- Modify: `frontend/src/lib/editor/SlotViewer.svelte`

**Interfaces:**
- Consumes: `inlineBinding` (Task 3), `ViewerControls`/`ViewerFeed` (Task 4), `ui` (collapse), `graph().pushNodeViewers`.

- [ ] **Step 1: Rewrite `SlotViewer.svelte` `<script>`.**

```svelte
<script lang="ts">
	import ViewerFeed from './ViewerFeed.svelte';
	import ViewerControls from './ViewerControls.svelte';
	import { inlineBinding } from './viewBinding';
	import { ui } from '$lib/stores/ui.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { dtypeColor } from '$lib/editor/categoryColor';

	type Props = { node: string; slot: string; dtype: string };
	const { node, slot, dtype }: Props = $props();

	const uiStore = ui();
	const binding = $derived(inlineBinding(node, slot, dtype));
	const kind = $derived(binding.kind);
	const expanded = $derived(uiStore.isSlotExpanded(node, slot));

	function onSlotClick(e: MouseEvent): void {
		e.stopPropagation();
		ui().requestSlotClick({ node, slot, dtype, side: 'source', clientX: e.clientX, clientY: e.clientY });
	}
	function toggleExpanded(e?: Event): void {
		e?.stopPropagation();
		uiStore.toggleSlotExpanded(node, slot);
		graph().pushNodeViewers(node); // persist collapse at the mutation site
	}
	function stopSelect(e: Event): void {
		e.stopPropagation();
	}
</script>
```
Note paths: this file lives in `editor/` but `viewBinding.ts` is in `viewers/` → import is `'$lib/viewers/viewBinding'`. Adjust the import accordingly (the snippet's `'./viewBinding'` is wrong for this directory).

Correct imports for `SlotViewer.svelte`:
```ts
	import ViewerFeed from '$lib/viewers/ViewerFeed.svelte';
	import ViewerControls from '$lib/viewers/ViewerControls.svelte';
	import { inlineBinding } from '$lib/viewers/viewBinding';
```
(Keep whatever import style the file already uses for ViewerFeed/ViewerControls — it currently imports them; just add `inlineBinding` and drop the old `viewerKind`/`rawViewerSettings` imports and the `settled` `$effect`.)

- [ ] **Step 2: Update the template usages.**
- `<ViewerControls {node} {slot} {dtype} />` → `<ViewerControls {dtype} {binding} />`
- `<ViewerFeed {node} {slot} {dtype} />` → `<ViewerFeed {node} {slot} {dtype} {binding} />`
- Delete the `let settled = false; $effect(() => { ... pushNodeViewers ... })` block entirely (persistence now happens in `toggleExpanded` and in the binding's setters).

- [ ] **Step 3: Typecheck.**
Run: `cd frontend && npx svelte-check --threshold error 2>&1 | tail -10`
Expected: errors only in `ViewerPanel.svelte` now.

- [ ] **Step 4: Commit.**
```bash
git add frontend/src/lib/editor/SlotViewer.svelte
git commit -m "refactor(viewers): SlotViewer drives the inline binding"
```

---

## Task 6: Wire the panel host (`ViewerPanel`)

Each panel owns its `{kind, settings}` in layout state via a panel binding.

**Files:**
- Modify: `frontend/src/lib/panels/ViewerPanel.svelte`

**Interfaces:**
- Consumes: `panelBinding` (Task 3), `ViewerControls`/`ViewerFeed` (Task 4).

- [ ] **Step 1: Rewrite `ViewerPanel.svelte` `<script>` + snippets.**

```svelte
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ViewerFeed from '$lib/viewers/ViewerFeed.svelte';
	import ViewerControls from '$lib/viewers/ViewerControls.svelte';
	import { panelBinding } from '$lib/viewers/viewBinding';
	import { asStateObject } from '$lib/workspace/panelState';

	interface ViewerState {
		node?: string | null;
		slot?: string | null;
	}

	let props: PanelProps = $props();

	function st(): ViewerState {
		return asStateObject(props.state) as ViewerState;
	}
	function curSlot(node: NodeInstanceInfo): string | null {
		const cur = st();
		const names = Object.keys(node.output_slots);
		return cur.slot && node.output_slots[cur.slot] ? cur.slot : (names[0] ?? null);
	}
	function pick(slot: string): void {
		props.setState({ ...st(), slot });
	}
</script>

<NodeLinkedPanel {...props} label="data">
	{#snippet controls(node)}
		{@const slot = curSlot(node)}
		{@const dtype = slot ? node.output_slots[slot] : null}
		{@const binding = panelBinding(() => props.state, props.setState, dtype)}
		<select
			class="slot-pick"
			value={slot ?? ''}
			onchange={(e) => pick(e.currentTarget.value)}
			data-testid="viewer-slot"
		>
			{#each Object.entries(node.output_slots) as [name, dt] (name)}
				<option value={name}>{name} · {dt.toLowerCase()}</option>
			{/each}
		</select>
		{#if slot && dtype}
			<ViewerControls {dtype} {binding} />
		{/if}
	{/snippet}

	{#snippet content(node)}
		{@const slot = curSlot(node)}
		{@const dtype = slot ? node.output_slots[slot] : null}
		{@const binding = panelBinding(() => props.state, props.setState, dtype)}
		<div class="vp-body"><ViewerFeed node={node.name} {slot} {dtype} {binding} /></div>
	{/snippet}
</NodeLinkedPanel>
```
(Keep the existing `<style>` unchanged. Update the file's top comment to say each panel holds its own independent view state.)

- [ ] **Step 2: Typecheck + build clean.**
Run: `cd frontend && npx svelte-check --threshold error 2>&1 | tail -5 && npm run build 2>&1 | tail -2`
Expected: `0 ERRORS` and `✔ done`.

- [ ] **Step 3: Sanity — existing viewer e2e still passes (inline path unbroken).**
Run: `rm -f /dev/shm/iox2_* 2>/dev/null; .venv/bin/python -m pytest e2e/test_viewers.py -q 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 4: Commit.**
```bash
git add frontend/src/lib/panels/ViewerPanel.svelte
git commit -m "refactor(viewers): each Viewer panel owns its view state (panel binding)"
```

---

## Task 7: e2e — independence + persistence

**Files:**
- Create/Modify: `e2e/test_viewer_independence.py` (gitignored; drive via `window.goofi.commands` + DOM panel ids)

**Interfaces:**
- Consumes: conftest `page` fixture, `open_menu_and_pick`, `assert_no_console_errors`; agent surface `window.goofi.commands.{addNode,bindNodeToPanel,setPanelType}` and `query.graph()`.

- [ ] **Step 1: Write the independence test.**

```python
# e2e/test_viewer_independence.py
from playwright.sync_api import Page, expect
from .conftest import open_menu_and_pick, assert_no_console_errors


def _add_viewer_panel_bound_to(page: Page, node: str) -> str:
    """Split a second panel, make it a Viewer bound to `node`. Returns its id."""
    pid = page.evaluate(
        """([node]) => {
            const ws = window.goofi; // façade
            return null; // placeholder; replaced below
        }""",
        [node],
    )
    return pid


def test_inline_and_panel_viewer_kinds_are_independent(page: Page):
    open_menu_and_pick(page, "Oscillator")  # ARRAY output 'out'; inline viewer present
    node = page.evaluate("() => window.goofi.query.graph().nodes[0].name")

    # A Viewer panel bound to the same node/slot, created deterministically.
    panel_id = page.evaluate(
        """([node]) => {
            const c = window.goofi.commands;
            const ws = window.goofi; // see commands surface
            const pid = c.addTab ? null : null; return pid;
        }""",
        [node],
    )
    # NOTE: implement the deterministic split+bind in Step 2 once the exact
    # command names are confirmed; see Step 2.
    assert_no_console_errors(page)
```
This is a stub — Step 2 finalizes the setup once command names are confirmed.

- [ ] **Step 2: Finalize deterministic panel setup.**
Confirm the agent surface verbs:
Run: `cd frontend && grep -n "addTab\|setPanelType\|bindNodeToPanel\|split" src/lib/agent/commands.ts`
Use the confirmed verbs. If `commands` lacks a split, create the viewer panel by splitting via the workspace store directly through the façade, or fall back to the menu helpers `_split_via_menu`/`_set_content` (see `e2e/test_header_tabs.py`) to make panel index 1 a `Viewer`, then read its id from the DOM (`page.locator('[data-panel-type="viewer"]').get_attribute('data-panel-id')`) and call `window.goofi.commands.bindNodeToPanel(panelId, node)`. Final test body:

```python
def test_inline_and_panel_viewer_kinds_are_independent(page: Page):
    open_menu_and_pick(page, "Oscillator")
    node = page.evaluate("() => window.goofi.query.graph().nodes[0].name")

    # Set the INLINE viewer to image.
    page.locator(".svelte-flow__node select.kind").first.select_option("image")

    # Create a Viewer panel bound to the same node (deterministic via façade).
    from .conftest import split_panel_to_viewer  # add this helper in conftest if missing
    panel_id = split_panel_to_viewer(page)
    page.evaluate("([pid, n]) => window.goofi.commands.bindNodeToPanel(pid, n)", [panel_id, node])

    panel_kind = page.locator('[data-panel-type="viewer"] select.kind')
    expect(panel_kind).to_have_value("line")          # independent default, NOT image

    # Flip the panel to image + a setting; the inline viewer must NOT change.
    panel_kind.select_option("trajectory")
    expect(page.locator(".svelte-flow__node select.kind").first).to_have_value("image")
    expect(panel_kind).to_have_value("trajectory")
    assert_no_console_errors(page)
```
If a `split_panel_to_viewer` helper doesn't exist, implement it in `e2e/conftest.py` using the same menu-driven split as `test_header_tabs.py` and return the new panel's `data-panel-id`.

- [ ] **Step 3: Run the independence test.**
Run: `rm -f /dev/shm/iox2_* 2>/dev/null; .venv/bin/python -m pytest e2e/test_viewer_independence.py -q 2>&1 | tail -15`
Expected: PASS. If `select.kind` resolves to >1 element for the inline locator (e.g. multiple slots), scope to the node header's first viewer.

- [ ] **Step 4: Write the save/reload persistence test.**
Group the two assertions: after setting inline=image and panel=trajectory, save to a tmp `.gfi` and reload (mirror `e2e/test_subpatch_ui.py::test_entered_subpatch_survives_save_reload` for the save/load FsBrowser steps), then assert the inline `select.kind` is `image` and the panel `select.kind` is `trajectory`.

```python
def test_viewer_kinds_survive_save_reload(page: Page, tmp_path):
    # ... open_menu_and_pick Oscillator; set inline image; create+bind viewer panel; set panel trajectory ...
    # ... Save via [data-testid=topbar-save] → FsBrowser → filename "views" ...
    # ... assert saved file exists ...
    # ... Load via [data-testid=topbar-load] → FsBrowser → pick "views.gfi" ...
    expect(page.locator(".svelte-flow__node select.kind").first).to_have_value("image")
    expect(page.locator('[data-panel-type="viewer"] select.kind')).to_have_value("trajectory")
    assert_no_console_errors(page)
```
(Write the elided steps out in full following the save/reload pattern referenced above.)

- [ ] **Step 5: Run both e2e tests.**
Run: `rm -f /dev/shm/iox2_* 2>/dev/null; .venv/bin/python -m pytest e2e/test_viewer_independence.py -q 2>&1 | tail -15`
Expected: both PASS. (e2e is gitignored — no commit needed for the test files, but commit any `conftest.py` helper.)

- [ ] **Step 6: Commit conftest helper if added.**
```bash
git add e2e/conftest.py 2>/dev/null && git commit -m "test(e2e): viewer-panel split helper" || echo "no conftest change"
```

---

## Task 8: Cleanup, review, simplify

Rigorous structural close-out (per the request).

**Files:** whole viewer subsystem touched above.

- [ ] **Step 1: Dead-reference sweep.**
Run: `cd frontend && grep -rn "viewerState.svelte\|viewerSettings.svelte\|viewerKind\|setViewerKind\|rawViewerSettings\|seedViewerKind\|seedViewerSettings\|forgetViewerKinds\|forgetViewerSettings" src/`
Expected: NO matches (all replaced by `inlineView`/`resolveKind`/binding). Fix any stragglers. Confirm `git status` shows the two old store files deleted.

- [ ] **Step 2: Typecheck + build + full viewer/editor e2e.**
Run: `cd frontend && npx svelte-check --threshold error 2>&1 | tail -3 && npm run build 2>&1 | tail -2`
Run: `rm -f /dev/shm/iox2_* 2>/dev/null; .venv/bin/python -m pytest e2e/test_viewers.py e2e/test_viewer_independence.py -q 2>&1 | tail -5`
Expected: `0 ERRORS`, `✔ done`, all e2e pass.

- [ ] **Step 3: Code-review pass.**
Dispatch a `code-reviewer` subagent over the diff (`git diff main..HEAD -- frontend/src/lib/viewers frontend/src/lib/editor/SlotViewer.svelte frontend/src/lib/panels/ViewerPanel.svelte frontend/src/lib/stores/graph.svelte.ts`). Focus: any remaining `(node,slot)`-shared view state; binding reactivity correctness (getters under `$derived`); panel state not leaking across panels; persistence at every mutation site (kind, setting, collapse). Fix confirmed findings.

- [ ] **Step 4: Simplify pass.**
Dispatch a `code-simplifier` subagent over the same set. Goal: remove indirection the rewrite made redundant, tighten the binding factories and component props, confirm no duplicated resolve logic. Apply safe simplifications; re-run Step 2 after.

- [ ] **Step 5: Final commit.**
```bash
git add -A
git commit -m "refactor(viewers): cleanup + review/simplify pass for per-instance view state"
```

---

## Self-review (against the spec)

- **Spec coverage:** separate instances (Tasks 4-6) ✓; shared stream untouched (Global Constraints; no `frames.ts`/`data.py` edits) ✓; unified component + `ViewBinding` (Tasks 3-6) ✓; pure `resolveKind`/`resolveSettings` (Task 1) ✓; merged inline store (Task 2) ✓; inline→node.viewers, panel→layout (Tasks 2,3,6) ✓; collapse stays inline (Task 5) ✓; agent surface targets inline (Task 2 Step 3) ✓; tests incl. save/reload (Task 7) ✓; deletions/cleanup (Tasks 2,8) ✓; no backend changes (Global Constraints) ✓; no migration (Global Constraints) ✓.
- **Placeholder scan:** Task 7 Step 1 is an explicit stub finalized in Step 2 (deterministic setup depends on confirming the façade's split/bind verbs at execution time) — acceptable because the finalization step gives the concrete fallback (menu-driven split + `bindNodeToPanel` + DOM panel id).
- **Type consistency:** `ViewBinding` (kind/settings getters + setKind/setSetting) used identically in Tasks 3-6; `rawInlineView`/`setInlineKind`/`setInlineSetting`/`seedInlineView`/`forgetInlineView` names consistent across Tasks 2-3; `SettingsMap` home moved to `settingsSchema.ts` (Task 1) and imported from there everywhere after.
