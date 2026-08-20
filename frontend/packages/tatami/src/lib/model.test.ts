import { describe, it, expect } from 'vitest';
import {
	findPanel,
	findParent,
	countPanels,
	collectPanels,
	firstPanelId,
	resizeFractions,
	MIN_FRACTION,
	MIN_PANEL_PX,
	type LayoutNode,
	type SplitNode
} from './model';

/**
 * What is left of the model once the manager owns the tree: the render vocabulary, the queries
 * every component reads it with, and the one piece of geometry a client must own — the pixel floor
 * a splitter drag is clamped to (D-R10), which only the renderer can measure. The tree ALGEBRA
 * (split-or-wrap, close-with-promote, renormalize) moved to `goofi_engine::layout` and is tested
 * there against the same fixtures it was ported from; the reconstruction from the flat arrangement
 * is `arrangement.test.ts`.
 */

const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);

function tree(): SplitNode {
	return {
		kind: 'split',
		id: 'split-1',
		direction: 'row',
		sizes: [0.5, 0.5],
		children: [
			{ kind: 'panel', id: 'panel-a', panelType: 'node-editor' },
			{
				kind: 'split',
				id: 'split-2',
				direction: 'column',
				sizes: [0.5, 0.5],
				children: [
					{ kind: 'panel', id: 'panel-b', panelType: 'console' },
					{ kind: 'panel', id: 'panel-c', panelType: 'viewer' }
				]
			}
		]
	};
}

describe('queries', () => {
	it('finds a panel at any depth, and misses cleanly', () => {
		expect(findPanel(tree(), 'panel-c')?.panelType).toBe('viewer');
		expect(findPanel(tree(), 'nope')).toBeNull();
	});

	it('collects and counts panels in document order', () => {
		expect(collectPanels(tree()).map((p) => p.id)).toEqual(['panel-a', 'panel-b', 'panel-c']);
		expect(countPanels(tree())).toBe(3);
		expect(firstPanelId(tree())).toBe('panel-a');
	});

	it('names a node’s parent split and its index — the sibling test a drag-join runs', () => {
		const p = findParent(tree(), 'panel-c');
		expect(p?.parent.id).toBe('split-2');
		expect(p?.index).toBe(1);
		const leaf: LayoutNode = { kind: 'panel', id: 'lone', panelType: 'empty' };
		expect(findParent(leaf, 'lone'), 'a root has no parent').toBeNull();
	});
});

describe('resizeFractions', () => {
	it('moves the boundary between two children', () => {
		const next = resizeFractions([0.5, 0.5], 0, 0.2);
		expect(next[0]).toBeCloseTo(0.7, 6);
		expect(next[1]).toBeCloseTo(0.3, 6);
		expect(sum(next)).toBeCloseTo(1, 6);
	});

	it('leaves the shares alone when the divider is not there', () => {
		expect(resizeFractions([0.5, 0.5], 1, 0.2)).toEqual([0.5, 0.5]);
		expect(resizeFractions([0.5, 0.5], -1, 0.2)).toEqual([0.5, 0.5]);
	});

	it('clamps a child to MIN_FRACTION instead of collapsing it', () => {
		const next = resizeFractions([0.5, 0.5], 0, 0.9); // would push child 1 negative
		expect(next[1]).toBeCloseTo(MIN_FRACTION, 6);
		expect(next[0]).toBeCloseTo(1 - MIN_FRACTION, 6);
	});

	it('touches only the pair either side of the divider', () => {
		const next = resizeFractions([0.4, 0.4, 0.2], 0, 0.1);
		expect(next[2], 'a third child keeps its slice').toBeCloseTo(0.2, 6);
		expect(sum(next)).toBeCloseTo(1, 6);
	});

	/* D-R10. A fraction floor is not a floor: 5% of a 390px phone is a 19.5px panel, and 5% of a
	   4K window is 192px — the same rule producing an unusable sliver at one size and a generous
	   minimum at another. The split's measured px is what makes the floor mean something. This is a
	   desktop fix too: a narrow desktop window collapses a panel exactly the same way. It stays on
	   the CLIENT because the manager cannot know it — it plans in fractions and never sees a pixel. */
	describe('the pixel floor', () => {
		it('floors a child at MIN_PANEL_PX of the split it lives in', () => {
			const next = resizeFractions([0.5, 0.5], 0, 0.9, 800);
			expect(next[1] * 800, 'not 5% of 800 = 40px').toBeCloseTo(MIN_PANEL_PX, 6);
			expect(sum(next)).toBeCloseTo(1, 6);
		});

		it('still allows every size above the floor', () => {
			expect(resizeFractions([0.5, 0.5], 0, 0.2, 800)[0]).toBeCloseTo(0.7, 6);
		});

		it('never floors past an even split, so a tiny container stays draggable-to-centre', () => {
			// Below 2 × MIN_PANEL_PX the floor cannot be honoured on both sides; capping it at half
			// the pair's share keeps the arithmetic sane (and the sizes positive) instead of
			// producing a negative neighbour.
			const next = resizeFractions([0.5, 0.5], 0, 0.9, 200);
			expect(next[0]).toBeCloseTo(0.5, 6);
			expect(next[1]).toBeCloseTo(0.5, 6);
		});

		it('falls back to the fraction floor when the container size is unknown', () => {
			expect(resizeFractions([0.5, 0.5], 0, 0.9)[1]).toBeCloseTo(MIN_FRACTION, 6);
		});
	});
});
