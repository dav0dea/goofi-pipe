import { describe, expect, it } from 'vitest';
import { matchParams, scoreParam, type ParamGroups } from './paramSearch';
import type { ParamDescriptor } from '$lib/api/types';

function param(doc: string | null = null): ParamDescriptor {
	return {
		type: 'float',
		value: 0,
		default: 0,
		vmin: 0,
		vmax: 1,
		doc,
		refreshable: false,
		mode: 'constant',
		expression: null,
		reference: null,
		triggers: false,
		error: null
	};
}

const params: ParamGroups = {
	osc: { tune: param(), model: param('Which oscillator model.') },
	vcf1: { cutoff: param(), filterCutoff: param(), resonance: param() },
	env1: { attack: param(), decay: param('The fall to the sustain level.') }
};

describe('scoreParam', () => {
	it('ranks an exact name over a prefix, a word start and a substring', () => {
		expect(scoreParam('cutoff', 'vcf1', '', 'cutoff')).toBe(6);
		expect(scoreParam('cutoffAmount', 'vcf1', '', 'cutoff')).toBe(5);
		expect(scoreParam('filterCutoff', 'vcf1', '', 'cutoff')).toBe(4);
		expect(scoreParam('precutoffx', 'vcf1', '', 'cutoff')).toBe(3);
	});

	it('falls back to the family, then the doc', () => {
		expect(scoreParam('tune', 'osc', '', 'osc')).toBe(2);
		expect(scoreParam('decay', 'env1', 'The fall to the sustain level.', 'sustain')).toBe(1);
		expect(scoreParam('tune', 'osc', '', 'nothing')).toBe(0);
	});
});

describe('matchParams', () => {
	it('matches nothing for an empty query, so the caller keeps its tabs', () => {
		expect(matchParams(params, '')).toEqual([]);
		expect(matchParams(params, '   ')).toEqual([]);
	});

	it('reaches across families and carries each hit its own group', () => {
		const hits = matchParams(params, 'cutoff');
		expect(hits.map((h) => h.name)).toEqual(['cutoff', 'filterCutoff']);
		expect(hits.every((h) => h.group === 'vcf1')).toBe(true);
	});

	it('finds a family by name', () => {
		expect(matchParams(params, 'env1').map((h) => h.name)).toEqual(['attack', 'decay']);
	});

	it('is case insensitive and finds a camelCase word start', () => {
		expect(matchParams(params, 'CUTOFF').map((h) => h.name)).toEqual(['cutoff', 'filterCutoff']);
	});

	it('orders a better hit first across families', () => {
		const hits = matchParams({ a: { tune: param() }, osc: { detune: param() } }, 'tune');
		expect(hits.map((h) => h.name)).toEqual(['tune', 'detune']);
	});
});
