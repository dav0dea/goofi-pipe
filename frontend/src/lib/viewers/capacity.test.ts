import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';
import { viewSpecForKind, foldViewSpecs, CAP_FLOOR, type ViewSpec } from './capacity';

/** The SAME fixture tests/test_viewspec_fold.py folds with the Python
 * `fold_viewspecs`; folding it here pins the cross-language contract (the TS fold
 * and the manager-side Python fold can't drift). */
const VIEWSPEC_GOLDEN = JSON.parse(
	readFileSync(fileURLToPath(new URL('../../../../tests/viewspec_golden.json', import.meta.url)), 'utf-8')
) as { name: string; input: ViewSpec[]; output: ViewSpec }[];

describe('viewSpecForKind', () => {
	it('line → channel subsample + sample envelope sized to width', () => {
		const spec = viewSpecForKind('line', 1600, 300, 7);
		expect(spec.version).toBe(7);
		expect(spec.axes).toEqual([
			{ axis: 0, max: 300, method: 'subsample' },
			{ axis: -1, max: 1600, method: 'envelope' }
		]);
	});

	it('image → area on both pixel axes', () => {
		const spec = viewSpecForKind('image', 1280, 720);
		expect(spec.axes).toEqual([
			{ axis: 0, max: 720, method: 'area' },
			{ axis: 1, max: 1280, method: 'area' }
		]);
	});

	it('trajectory → subsample the point axis', () => {
		const spec = viewSpecForKind('trajectory', 800, 800);
		expect(spec.axes).toEqual([{ axis: 0, max: 800, method: 'subsample' }]);
	});

	it('string/table/topomap → no reduction', () => {
		expect(viewSpecForKind('string', 100, 100).axes).toEqual([]);
		expect(viewSpecForKind('table', 100, 100).axes).toEqual([]);
		expect(viewSpecForKind('topomap', 100, 100).axes).toEqual([]);
	});

	it('clamps degenerate (0-px / collapsed) sizes to the floor', () => {
		const spec = viewSpecForKind('line', 0, 0);
		expect(spec.axes[1].max).toBe(CAP_FLOOR);
		expect(spec.axes[0].max).toBe(CAP_FLOOR);
	});
});

describe('foldViewSpecs', () => {
	it('richest-wins per axis: max() of max, richest method, max() of version', () => {
		const folded = foldViewSpecs([
			{ axes: [{ axis: -1, max: 800, method: 'subsample' }], version: 1 },
			{ axes: [{ axis: -1, max: 2000, method: 'envelope' }], version: 5 }
		]);
		expect(folded.axes).toEqual([{ axis: -1, max: 2000, method: 'envelope' }]);
		expect(folded.version).toBe(5);
	});

	it('keeps distinct axes independent and sorted', () => {
		const folded = foldViewSpecs([
			{ axes: [{ axis: 1, max: 500, method: 'envelope' }], version: 0 },
			{ axes: [{ axis: 0, max: 8, method: 'subsample' }], version: 0 }
		]);
		expect(folded.axes).toEqual([
			{ axis: 0, max: 8, method: 'subsample' },
			{ axis: 1, max: 500, method: 'envelope' }
		]);
	});

	it('empty input folds to no reduction', () => {
		expect(foldViewSpecs([])).toEqual({ axes: [], version: 0 });
	});
});

describe('foldViewSpecs cross-language conformance (py fold_viewspecs)', () => {
	for (const c of VIEWSPEC_GOLDEN) {
		it(c.name, () => {
			expect(foldViewSpecs(c.input)).toEqual(c.output);
		});
	}
});
