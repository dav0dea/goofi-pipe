import { describe, it, expect } from 'vitest';
import { refNodes, refSlots, splitReference, wantedDtype } from './refs';
import type { ExprCatalogue } from './catalogue';

const cat: ExprCatalogue = {
	nodes: [
		{ name: 'osc', slots: [{ name: 'out', dtype: 'ARRAY' }], params: [] },
		{
			name: 'tagger',
			slots: [
				{ name: 'label', dtype: 'STRING' },
				{ name: 'count', dtype: 'ARRAY' }
			],
			params: []
		},
		{ name: 'sink', slots: [], params: [] }
	],
	globals: []
};

describe('the reference picker offers only what the param may reference', () => {
	it('types by the param: a string reads a STRING output, everything else an ARRAY one', () => {
		expect(wantedDtype('string')).toBe('STRING');
		for (const t of ['float', 'int', 'bool', 'unknown']) expect(wantedDtype(t)).toBe('ARRAY');
	});

	it('lists the nodes with at least one output of the kind, naming those outputs', () => {
		expect(refNodes(cat, 'ARRAY')).toEqual([
			{ label: 'osc', detail: 'out' },
			{ label: 'tagger', detail: 'count' }
		]);
		expect(refNodes(cat, 'STRING')).toEqual([{ label: 'tagger', detail: 'label' }]);
	});

	it("lists one node's outputs of the kind, and nothing for an unknown node", () => {
		expect(refSlots(cat, 'tagger', 'ARRAY')).toEqual([{ label: 'count', detail: 'ARRAY' }]);
		expect(refSlots(cat, 'nope', 'ARRAY')).toEqual([]);
	});

	it('splits node.slot at its one dot, and reads nothing into a malformed value', () => {
		expect(splitReference('osc.out')).toEqual(['osc', 'out']);
		expect(splitReference('osc')).toEqual(['', '']);
		expect(splitReference(null)).toEqual(['', '']);
	});
});
