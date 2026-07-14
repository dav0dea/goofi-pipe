/**
 * Workspace layout model — the pure, framework-agnostic core of the panel
 * system.
 *
 * A workspace tab is a recursive tree. Leaves (`PanelNode`) host one
 * registered panel type; internal nodes (`SplitNode`) divide their space
 * between N children along one axis, with fractional `sizes` that sum to 1.
 *
 * Everything here is a pure function over plain data: no Svelte, no DOM, no
 * IDs pulled from the environment beyond a deterministic counter. That keeps
 * the tree algebra unit-testable in isolation — the reactive store
 * (`workspace.svelte.ts`) is the only layer that holds state and reacts.
 */

import { linkedNodeName, withLinkedNode } from './panelState';

/** Smallest fraction a single child may shrink to, so a panel can always be
 * grabbed again after an aggressive resize. */
export const MIN_FRACTION = 0.05;

/** `row` = children laid out left→right (vertical splitters between them).
 *  `column` = children stacked top→bottom (horizontal splitters). Matches CSS
 *  `flex-direction` so the renderer can map direction → layout directly. */
export type Direction = 'row' | 'column';

export interface PanelNode {
	kind: 'panel';
	id: string;
	/** Registry key of the content shown here (e.g. `'node-editor'`). */
	panelType: string;
	/** Opaque per-panel state, persisted with the layout (e.g. a Viewer
	 * panel's chosen node/slot). The framework never inspects it. */
	state?: unknown;
}

export interface SplitNode {
	kind: 'split';
	id: string;
	direction: Direction;
	children: LayoutNode[];
	/** One fraction per child, same length as `children`, summing to ~1. */
	sizes: number[];
}

export type LayoutNode = PanelNode | SplitNode;

export interface Workspace {
	id: string;
	name: string;
	root: LayoutNode;
}

export interface WorkspaceState {
	workspaces: Workspace[];
	activeWorkspaceId: string;
}

// ---------------------------------------------------------------------------
// IDs — a monotonic counter, namespaced by prefix. Uniqueness only needs to
// hold within a running session; `reseedIds` bumps the counter past any ids
// hydrated from a saved layout so freshly-created nodes never collide.
// ---------------------------------------------------------------------------

let _seq = 0;

export function uid(prefix: string): string {
	_seq += 1;
	return `${prefix}-${_seq}`;
}

/** After hydrating a saved layout, advance the counter past every `*-<n>` id
 * already present so new ids stay unique. */
export function reseedIds(state: WorkspaceState): void {
	let max = _seq;
	const scan = (node: LayoutNode): void => {
		const m = /-(\d+)$/.exec(node.id);
		if (m) max = Math.max(max, parseInt(m[1], 10));
		if (node.kind === 'split') node.children.forEach(scan);
	};
	for (const ws of state.workspaces) scan(ws.root);
	_seq = max;
}

export function makePanel(panelType: string, state?: unknown): PanelNode {
	return { kind: 'panel', id: uid('panel'), panelType, state };
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

export function findPanel(root: LayoutNode, panelId: string): PanelNode | null {
	if (root.kind === 'panel') return root.id === panelId ? root : null;
	for (const c of root.children) {
		const f = findPanel(c, panelId);
		if (f) return f;
	}
	return null;
}

export function collectPanels(root: LayoutNode): PanelNode[] {
	if (root.kind === 'panel') return [root];
	return root.children.flatMap(collectPanels);
}

export function countPanels(root: LayoutNode): number {
	return collectPanels(root).length;
}

/** Id of the first panel in document order — used to pick a sensible active
 * panel after a close. */
export function firstPanelId(root: LayoutNode): string {
	return collectPanels(root)[0]?.id ?? '';
}

interface ParentRef {
	parent: SplitNode;
	index: number;
}

export function findParent(root: LayoutNode, nodeId: string): ParentRef | null {
	if (root.kind === 'panel') return null;
	for (let i = 0; i < root.children.length; i++) {
		if (root.children[i].id === nodeId) return { parent: root, index: i };
		const deeper = findParent(root.children[i], nodeId);
		if (deeper) return deeper;
	}
	return null;
}

// ---------------------------------------------------------------------------
// Immutable transforms — every op returns a new tree (structural sharing on
// the unchanged subtrees) so a plain store assignment triggers reactivity.
// ---------------------------------------------------------------------------

/** Replace the node whose id is `nodeId` with `fn(node)`, rebuilding only the
 * path to it. Returns the same reference if nothing matched. */
function transform(root: LayoutNode, nodeId: string, fn: (n: LayoutNode) => LayoutNode): LayoutNode {
	if (root.id === nodeId) return fn(root);
	if (root.kind === 'panel') return root;
	let changed = false;
	const children = root.children.map((c) => {
		const nc = transform(c, nodeId, fn);
		if (nc !== c) changed = true;
		return nc;
	});
	return changed ? { ...root, children } : root;
}

export function setPanelType(root: LayoutNode, panelId: string, panelType: string): LayoutNode {
	// Switching type discards the old type's per-panel state — a new type can't
	// interpret it.
	return transform(root, panelId, (n) => (n.kind === 'panel' ? { ...n, panelType, state: undefined } : n));
}

export function setPanelState(root: LayoutNode, panelId: string, state: unknown): LayoutNode {
	return transform(root, panelId, (n) => (n.kind === 'panel' ? { ...n, state } : n));
}

/** Adjust the boundary between children `dividerIndex` and `dividerIndex+1` of
 * the given split by `delta` (a fraction). Clamps both neighbours to
 * MIN_FRACTION, pushing the overflow onto the other side. */
export function resizeSplit(
	root: LayoutNode,
	splitId: string,
	dividerIndex: number,
	delta: number
): LayoutNode {
	return transform(root, splitId, (n) => {
		if (n.kind !== 'split') return n;
		const i = dividerIndex;
		const j = dividerIndex + 1;
		if (i < 0 || j >= n.sizes.length) return n;
		let a = n.sizes[i] + delta;
		let b = n.sizes[j] - delta;
		if (a < MIN_FRACTION) {
			b -= MIN_FRACTION - a;
			a = MIN_FRACTION;
		}
		if (b < MIN_FRACTION) {
			a -= MIN_FRACTION - b;
			b = MIN_FRACTION;
		}
		const sizes = n.sizes.slice();
		sizes[i] = a;
		sizes[j] = b;
		return { ...n, sizes };
	});
}

/**
 * Insert an arbitrary `node` (a panel or a whole subtree) adjacent to the
 * target panel, splitting along `direction`.
 *
 * - If the target's parent split already runs along `direction`, the node is
 *   inserted as an adjacent sibling and the target's size slice is halved.
 * - Otherwise the target is wrapped in a fresh split with two equal children.
 *
 * `placeBefore` puts the inserted node before the target (left / top).
 * `fraction` is the inserted node's share of the target's current slot
 * (0.5 = even split); the target keeps the rest.
 */
export function insertNodeAtPanel(
	root: LayoutNode,
	targetPanelId: string,
	direction: Direction,
	placeBefore: boolean,
	node: LayoutNode,
	fraction = 0.5
): LayoutNode {
	const target = findPanel(root, targetPanelId);
	if (!target) return root;
	const parent = findParent(root, targetPanelId);
	const f = Math.max(0.05, Math.min(0.95, fraction));

	if (parent && parent.parent.direction === direction) {
		return transform(root, parent.parent.id, (n) => {
			if (n.kind !== 'split') return n;
			const idx = n.children.findIndex((c) => c.id === targetPanelId);
			if (idx < 0) return n;
			const slot = n.sizes[idx];
			const newSize = slot * f;
			const keepSize = slot - newSize;
			const children = n.children.slice();
			const sizes = n.sizes.slice();
			children.splice(placeBefore ? idx : idx + 1, 0, node);
			// Sizes must line up with the new child order at this position.
			sizes.splice(idx, 1, ...(placeBefore ? [newSize, keepSize] : [keepSize, newSize]));
			return { ...n, children, sizes };
		});
	}

	const wrap: SplitNode = {
		kind: 'split',
		id: uid('split'),
		direction,
		children: placeBefore ? [node, target] : [target, node],
		sizes: placeBefore ? [f, 1 - f] : [1 - f, f]
	};
	return transform(root, targetPanelId, () => wrap);
}

/**
 * Split `panelId` along `direction`, inserting a new panel of `newType`.
 * Returns the new tree and the id of the created panel.
 */
export function splitPanel(
	root: LayoutNode,
	panelId: string,
	direction: Direction,
	placeBefore: boolean,
	newType: string,
	fraction = 0.5
): { root: LayoutNode; newPanelId: string } {
	if (!findPanel(root, panelId)) return { root, newPanelId: '' };
	const newPanel = makePanel(newType);
	return {
		root: insertNodeAtPanel(root, panelId, direction, placeBefore, newPanel, fraction),
		newPanelId: newPanel.id
	};
}

/** Clear any panel whose state links the named node (`state.node === name`),
 * so deleting a node empties the Parameters / Viewer / Metadata panels bound to
 * it. Returns the same tree if nothing referenced it. */
export function clearNodeRef(root: LayoutNode, nodeName: string): LayoutNode {
	const visit = (n: LayoutNode): LayoutNode => {
		if (n.kind === 'panel') {
			return linkedNodeName(n.state) === nodeName
				? { ...n, state: withLinkedNode(n.state, null) }
				: n;
		}
		let changed = false;
		const children = n.children.map((c) => {
			const nc = visit(c);
			if (nc !== c) changed = true;
			return nc;
		});
		return changed ? { ...n, children } : n;
	};
	return visit(root);
}

function normalize(sizes: number[]): number[] {
	const total = sizes.reduce((a, b) => a + b, 0) || 1;
	return sizes.map((s) => s / total);
}

/**
 * Remove `panelId`, handing its space to its siblings (proportionally). If the
 * parent split is left with a single child, that child is promoted in the
 * parent's place. Returns null if the panel is the only one in the tree
 * (the last panel can't be closed).
 */
export function closePanel(root: LayoutNode, panelId: string): LayoutNode | null {
	if (countPanels(root) <= 1) return null;
	const parent = findParent(root, panelId);
	if (!parent) return null;
	const split = parent.parent;
	const idx = parent.index;
	const removed = split.sizes[idx];

	const children = split.children.filter((_, i) => i !== idx);
	const kept = split.sizes.filter((_, i) => i !== idx);
	const total = kept.reduce((a, b) => a + b, 0) || 1;
	const sizes = normalize(kept.map((s) => s + removed * (s / total)));

	const replacement: LayoutNode =
		children.length === 1 ? children[0] : { ...split, children, sizes };

	return transform(root, split.id, () => replacement);
}

/**
 * Remove `panelId` and return both the reduced tree and the removed panel, so
 * the panel can be re-inserted elsewhere (drag-to-reposition). `root` is null
 * when the panel was the entire tree.
 */
export function extractPanel(
	root: LayoutNode,
	panelId: string
): { root: LayoutNode | null; removed: PanelNode | null } {
	if (root.kind === 'panel') {
		return root.id === panelId ? { root: null, removed: root } : { root, removed: null };
	}
	const removed = findPanel(root, panelId);
	if (!removed) return { root, removed: null };
	// root is a split with >= 2 panels, so closePanel won't refuse.
	return { root: closePanel(root, panelId), removed };
}

// ---------------------------------------------------------------------------
// Defaults & deep clone (for tab duplication and hydration safety)
// ---------------------------------------------------------------------------

/** The panel type a brand-new workspace / tab starts with. */
export const DEFAULT_PANEL_TYPE = 'node-editor';

/** The placeholder a split births: an empty panel whose in-panel buttons let
 * the user choose its content, so a split never assumes a type. */
export const EMPTY_PANEL_TYPE = 'empty';

export function makeWorkspace(name: string, panelType: string = DEFAULT_PANEL_TYPE): Workspace {
	return { id: uid('ws'), name, root: makePanel(panelType) };
}

export function defaultWorkspaceState(): WorkspaceState {
	const ws = makeWorkspace('Layout');
	return { workspaces: [ws], activeWorkspaceId: ws.id };
}

/** Deep clone a subtree with fresh ids — used when duplicating a tab so the
 * copy shares no node identities with the original. */
export function cloneWithNewIds(node: LayoutNode): LayoutNode {
	if (node.kind === 'panel') {
		return { kind: 'panel', id: uid('panel'), panelType: node.panelType, state: node.state };
	}
	return {
		kind: 'split',
		id: uid('split'),
		direction: node.direction,
		children: node.children.map(cloneWithNewIds),
		sizes: node.sizes.slice()
	};
}
