import { describe, it, expect } from 'vitest';
import {
	makePanel,
	splitPanel,
	closePanel,
	resizeSplit,
	setPanelType,
	setPanelState,
	findPanel,
	countPanels,
	collectPanels,
	cloneWithNewIds,
	MIN_FRACTION,
	type LayoutNode,
	type SplitNode
} from './model';

const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);

describe('splitPanel', () => {
	it('wraps a lone panel in a split with two equal children', () => {
		const root = makePanel('node-editor');
		const { root: next, newPanelId } = splitPanel(root, root.id, 'row', false, 'empty');
		expect(next.kind).toBe('split');
		const split = next as SplitNode;
		expect(split.direction).toBe('row');
		expect(split.children).toHaveLength(2);
		expect(split.sizes).toEqual([0.5, 0.5]);
		// original panel first, new panel second (placeBefore = false)
		expect(split.children[0].id).toBe(root.id);
		expect(split.children[1].id).toBe(newPanelId);
	});

	it('respects placeBefore by inserting the new panel first', () => {
		const root = makePanel('node-editor');
		const { root: next, newPanelId } = splitPanel(root, root.id, 'column', true, 'empty');
		const split = next as SplitNode;
		expect(split.children[0].id).toBe(newPanelId);
		expect(split.children[1].id).toBe(root.id);
	});

	it('inserts an adjacent sibling when the parent already runs the same direction', () => {
		const a = makePanel('a');
		// root: row [a, b]
		const step1 = splitPanel(a, a.id, 'row', false, 'b');
		const b = (step1.root as SplitNode).children[1] as LayoutNode;
		// split `b` again along the same row direction → flat 3-way split, no nesting
		const step2 = splitPanel(step1.root, b.id, 'row', false, 'c');
		const split = step2.root as SplitNode;
		expect(split.kind).toBe('split');
		expect(split.children).toHaveLength(3);
		expect(split.children.every((c) => c.kind === 'panel')).toBe(true);
		expect(sum(split.sizes)).toBeCloseTo(1, 6);
		// b's old 0.5 slice halved into two 0.25 slices
		expect(split.sizes[0]).toBeCloseTo(0.5, 6);
		expect(split.sizes[1]).toBeCloseTo(0.25, 6);
		expect(split.sizes[2]).toBeCloseTo(0.25, 6);
	});

	it('nests a new split when splitting along the perpendicular direction', () => {
		const a = makePanel('a');
		const step1 = splitPanel(a, a.id, 'row', false, 'b'); // row [a, b]
		const step2 = splitPanel(step1.root, a.id, 'column', false, 'c'); // a → column [a, c]
		const outer = step2.root as SplitNode;
		expect(outer.direction).toBe('row');
		expect(outer.children[0].kind).toBe('split');
		const inner = outer.children[0] as SplitNode;
		expect(inner.direction).toBe('column');
		expect(inner.children).toHaveLength(2);
	});
});

describe('closePanel', () => {
	it('refuses to close the last panel', () => {
		const root = makePanel('only');
		expect(closePanel(root, root.id)).toBeNull();
	});

	it('promotes the sibling when a 2-way split loses a child', () => {
		const a = makePanel('a');
		const { root, newPanelId } = splitPanel(a, a.id, 'row', false, 'b');
		const next = closePanel(root, newPanelId);
		expect(next).not.toBeNull();
		// split collapses back to the surviving panel
		expect(next!.kind).toBe('panel');
		expect(next!.id).toBe(a.id);
	});

	it('redistributes freed space proportionally in an N-way split', () => {
		const a = makePanel('a');
		const s1 = splitPanel(a, a.id, 'row', false, 'b');
		const b = (s1.root as SplitNode).children[1];
		const s2 = splitPanel(s1.root, b.id, 'row', false, 'c'); // sizes ~ [0.5, 0.25, 0.25]
		const mid = (s2.root as SplitNode).children[1].id;
		const next = closePanel(s2.root, mid) as SplitNode;
		expect(next.children).toHaveLength(2);
		expect(sum(next.sizes)).toBeCloseTo(1, 6);
		expect(countPanels(next)).toBe(2);
	});
});

describe('resizeSplit', () => {
	it('moves the boundary between two children', () => {
		const a = makePanel('a');
		const { root } = splitPanel(a, a.id, 'row', false, 'b');
		const split = root as SplitNode;
		const next = resizeSplit(root, split.id, 0, 0.2) as SplitNode;
		expect(next.sizes[0]).toBeCloseTo(0.7, 6);
		expect(next.sizes[1]).toBeCloseTo(0.3, 6);
		expect(sum(next.sizes)).toBeCloseTo(1, 6);
	});

	it('clamps a child to MIN_FRACTION instead of collapsing it', () => {
		const a = makePanel('a');
		const { root } = splitPanel(a, a.id, 'row', false, 'b');
		const split = root as SplitNode;
		const next = resizeSplit(root, split.id, 0, 0.9) as SplitNode; // would push child 1 negative
		expect(next.sizes[1]).toBeCloseTo(MIN_FRACTION, 6);
		expect(next.sizes[0]).toBeCloseTo(1 - MIN_FRACTION, 6);
	});
});

describe('panel content ops', () => {
	it('setPanelType swaps content and resets state', () => {
		const root = makePanel('a', { foo: 1 });
		const next = setPanelType(root, root.id, 'b');
		const p = findPanel(next, root.id)!;
		expect(p.panelType).toBe('b');
		expect(p.state).toBeUndefined();
	});

	it('setPanelState updates opaque state', () => {
		const root = makePanel('viewer');
		const next = setPanelState(root, root.id, { slot: 'out' });
		expect(findPanel(next, root.id)!.state).toEqual({ slot: 'out' });
	});
});

describe('cloneWithNewIds', () => {
	it('produces a structurally identical tree with no shared ids', () => {
		const a = makePanel('a');
		const { root } = splitPanel(a, a.id, 'row', false, 'b');
		const clone = cloneWithNewIds(root);
		const origIds = new Set(collectPanels(root).map((p) => p.id));
		const cloneIds = collectPanels(clone).map((p) => p.id);
		expect(cloneIds).toHaveLength(2);
		expect(cloneIds.every((id) => !origIds.has(id))).toBe(true);
		expect((clone as SplitNode).direction).toBe((root as SplitNode).direction);
	});
});

describe('immutability', () => {
	it('does not mutate the input tree', () => {
		const a = makePanel('a');
		const { root } = splitPanel(a, a.id, 'row', false, 'b');
		const before = JSON.stringify(root);
		splitPanel(root, a.id, 'column', false, 'c');
		closePanel(root, a.id);
		expect(JSON.stringify(root)).toBe(before);
	});
});
