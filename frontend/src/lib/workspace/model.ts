/**
 * Workspace layout model — the pure, framework-agnostic shapes the panel system
 * renders, and the queries over them.
 *
 * A workspace tab is a recursive tree. Leaves (`PanelNode`) host one registered
 * panel type; internal nodes (`SplitNode`) divide their space between N children
 * along one axis, with fractional `sizes` that sum to 1.
 *
 * The tree itself is no longer authored here: the manager holds the arrangement
 * FLAT and id-keyed (the fifth CRDT doc root) and `arrangement.ts` rebuilds this
 * shape from it at render time. What remains is the vocabulary, the read-only
 * queries every component uses, and the one piece of geometry a client must own
 * — the pixel floor a splitter drag is clamped to, which only the renderer can
 * measure.
 */

/** Smallest fraction a single child may shrink to, so a panel can always be
 * grabbed again after an aggressive resize. */
export const MIN_FRACTION = 0.05;

/**
 * Smallest PIXEL width/height a single child may shrink to (D-R10). A fraction alone is not a
 * floor: 5% of a 390px phone is a 19.5px sliver, while 5% of a 4K window is 192px — the same rule
 * meaning two entirely different things. This is the size at which a panel still reads as a panel
 * (a header with its controls, and something under it), and it is a desktop fix as much as a phone
 * one — a narrow desktop window collapses a panel exactly the same way.
 */
export const MIN_PANEL_PX = 120;

/** `row` = children laid out left→right (vertical splitters between them).
 *  `column` = children stacked top→bottom (horizontal splitters). Matches CSS
 *  `flex-direction` so the renderer can map direction → layout directly. */
export type Direction = 'row' | 'column';

export interface PanelNode {
	kind: 'panel';
	id: string;
	/** Registry key of the content shown here (e.g. `'node-editor'`). */
	panelType: string;
	/** Opaque per-panel state, held by the manager (e.g. a Viewer panel's chosen
	 * node/slot). The framework never inspects it. */
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

/** The panel type a brand-new workspace / tab starts with, and the placeholder a split births —
 * both re-exported from the manager's generated vocabulary rather than restated. The MANAGER mints
 * these entries (`goofi_engine::layout`), so a client copy of either was a value that could quietly
 * stop matching what actually arrives in the doc. */
export { DEFAULT_PANEL_TYPE, EMPTY_PANEL_TYPE } from '$lib/api/vocab';

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
// Resize geometry — the one thing the manager cannot plan
// ---------------------------------------------------------------------------

/**
 * Move the boundary between children `dividerIndex` and `dividerIndex+1` by `delta` (a fraction),
 * clamping both neighbours to the floor and pushing the overflow onto the other side. The whole
 * drag is drawn from this and lands as ONE `resize_split`.
 *
 * `containerPx` is the split's measured size along its axis, which is what turns MIN_FRACTION into
 * a real MIN_PANEL_PX floor (D-R10) — a rendering-time fact the manager has no way to know. 0 /
 * omitted means "unmeasured" and keeps the fraction floor alone.
 */
export function resizeFractions(
	sizes: number[],
	dividerIndex: number,
	delta: number,
	containerPx = 0
): number[] {
	const i = dividerIndex;
	const j = dividerIndex + 1;
	if (i < 0 || j >= sizes.length) return sizes;
	let a = sizes[i] + delta;
	let b = sizes[j] - delta;
	// The pair's total is invariant under a boundary move, so half of it is the largest floor both
	// sides can satisfy — below 2 × MIN_PANEL_PX the seam simply locks to the middle rather than
	// driving one neighbour negative.
	const share = sizes[i] + sizes[j];
	const floor =
		containerPx > 0
			? Math.min(share / 2, Math.max(MIN_FRACTION, MIN_PANEL_PX / containerPx))
			: MIN_FRACTION;
	if (a < floor) {
		b -= floor - a;
		a = floor;
	}
	if (b < floor) {
		a -= floor - b;
		b = floor;
	}
	const next = sizes.slice();
	next[i] = a;
	next[j] = b;
	return next;
}
