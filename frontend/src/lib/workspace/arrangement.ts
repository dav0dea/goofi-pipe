/**
 * The flat arrangement, and the tree the panel system draws from it.
 *
 * The manager holds the workspace as `goofi_engine::layout::Layout`: every page, split and panel is
 * ONE id-keyed entry naming its `parent` and its `order` among that parent's children, mirrored
 * verbatim as the fifth CRDT doc root. Flat is what lets the arrangement ride the same reconciler
 * the graph does — and it is why the tree is rebuilt HERE, at render time, rather than shipped.
 *
 * Pure functions over plain records: no Svelte, no Yjs. The reactive replica
 * (`workspace.svelte.ts`) is the only layer that holds state, and the client never writes an entry
 * — every mutation is a layout command over `/control`.
 */
import {
	DEFAULT_PANEL_TYPE,
	EMPTY_PANEL_TYPE,
	type Direction,
	type LayoutNode,
	type Workspace
} from './model';

/** One arrangement entry, exactly as `Layout::to_json` writes it. `state` rides as a JSON STRING
 * leaf (the CRDT reconciler erases nested arrays and a viewer's settings can hold one). */
export interface ArrangementEntry {
	kind: 'page' | 'split' | 'panel';
	order: number;
	name?: string;
	parent?: string;
	size?: number;
	axis?: string;
	panel_type?: string;
	state?: string;
}

/** The whole arrangement, by id. Entries only — the manager's id counter rides the same doc root
 * under `#seq` and is filtered out where the root is read (`arrangementEntries`). */
export type Arrangement = Record<string, ArrangementEntry>;

/** The page an entry sits on, or null when the id is unknown / unreachable. */
export function pageOf(arr: Arrangement, id: string): string | null {
	let cur: string | undefined = id;
	// Bounded by the entry count: a cycle would be refused by the manager's own loader, so this
	// walks a tree, not a graph.
	for (let i = 0; cur !== undefined && i <= Object.keys(arr).length; i++) {
		const e: ArrangementEntry | undefined = arr[cur];
		if (!e) return null;
		if (e.kind === 'page') return cur;
		cur = e.parent;
	}
	return null;
}

/** A parent's children, in `order`. */
export function childIds(arr: Arrangement, parent: string): string[] {
	return Object.keys(arr)
		.filter((id) => arr[id].parent === parent)
		.sort((a, b) => arr[a].order - arr[b].order);
}

/** The first panel inside `id`, in document order — `id` itself when it is one. A drag names a
 * SUBTREE (a dragged tab names its page's root split), and this is the panel the user is working in
 * once it lands. Null when the id is unknown. */
export function firstPanelIn(arr: Arrangement, id: string): string | null {
	const e = arr[id];
	if (!e) return null;
	if (e.kind === 'panel') return id;
	for (const k of childIds(arr, id)) {
		const p = firstPanelIn(arr, k);
		if (p !== null) return p;
	}
	return null;
}

/** A split's children's shares, in child order — the baseline a resize drag adjusts and commits. */
export function splitFractions(arr: Arrangement, split: string): number[] {
	return childIds(arr, split).map((id) => arr[id].size ?? 0);
}

/** Parse a panel's opaque bag out of its JSON string leaf. */
function panelState(raw: string | undefined): unknown {
	if (typeof raw !== 'string') return undefined;
	try {
		const v: unknown = JSON.parse(raw);
		return v === null ? undefined : v;
	} catch {
		return undefined;
	}
}

/** The arrangement the panel system falls back to before the replica has pulled from the manager.
 * Ids are the manager's own first-mint spelling, so the pre-sync frame and the synced one draw the
 * same thing and nothing re-keys under the user. */
function unsyncedDefault(): Workspace[] {
	return [
		{
			id: 'page-1',
			name: 'Tab 1',
			root: { kind: 'panel', id: 'panel-2', panelType: DEFAULT_PANEL_TYPE, state: undefined }
		}
	];
}

/** Rebuild the render tree: one `Workspace` per page, in tab-strip order. A page whose root is
 * missing is dropped rather than drawn as a hole. */
export function buildWorkspaces(arr: Arrangement): Workspace[] {
	const build = (id: string): LayoutNode | null => {
		const e = arr[id];
		if (!e) return null;
		if (e.kind === 'panel') {
			return {
				kind: 'panel',
				id,
				panelType: e.panel_type ?? EMPTY_PANEL_TYPE,
				state: panelState(e.state)
			};
		}
		const kids = childIds(arr, id);
		const children: LayoutNode[] = [];
		const sizes: number[] = [];
		for (const k of kids) {
			const child = build(k);
			if (!child) continue;
			children.push(child);
			sizes.push(arr[k].size ?? 0);
		}
		if (children.length === 0) return null;
		const direction: Direction = e.axis === 'column' ? 'column' : 'row';
		return { kind: 'split', id, direction, children, sizes };
	};

	const pages = Object.keys(arr)
		.filter((id) => arr[id].kind === 'page')
		.sort((a, b) => arr[a].order - arr[b].order);
	const out: Workspace[] = [];
	for (const id of pages) {
		const root = build(childIds(arr, id)[0] ?? '');
		if (root) out.push({ id, name: arr[id].name ?? '', root });
	}
	return out.length > 0 ? out : unsyncedDefault();
}
