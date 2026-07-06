import { describe, it, expect } from 'vitest';
import { rankNodeTypes } from './nodeSearch';
import type { NodeTypeInfo } from '$lib/api/control';

function node(type: string, category: string, doc = ''): NodeTypeInfo {
	return {
		type,
		category,
		doc,
		input_slots: {},
		output_slots: {},
		params: {},
		available: true,
		dynamic: false,
		missing_deps: []
	};
}

const order = (types: NodeTypeInfo[], q: string): string[] =>
	rankNodeTypes(types, q).map((t) => t.type);

describe('rankNodeTypes', () => {
	it('ranks name matches above a docstring-only match (the "osc" case)', () => {
		const types = [
			node('Kuramoto', 'inputs', 'Coupled Kuramoto oscillators producing phase signals.'),
			node('Oscillator', 'inputs', 'Generate a periodic waveform.'),
			node('OSCOut', 'outputs', 'Send values over Open Sound Control.')
		];
		const result = order(types, 'osc');
		// Both name hits come before the doc-only hit.
		expect(result.indexOf('Oscillator')).toBeLessThan(result.indexOf('Kuramoto'));
		expect(result.indexOf('OSCOut')).toBeLessThan(result.indexOf('Kuramoto'));
		expect(result[result.length - 1]).toBe('Kuramoto');
	});

	it('orders exact name above prefix', () => {
		const types = [node('FFTWindow', 'signal', ''), node('FFT', 'signal', '')];
		expect(order(types, 'fft')).toEqual(['FFT', 'FFTWindow']);
	});

	it('orders prefix > word-start > plain substring within the name', () => {
		// Synthetic names isolate each tier (real CamelCase makes most inner
		// capitals word starts, so they rarely fall to the substring tier):
		//   abcdef  → prefix          (starts with "abc")
		//   x_abc   → word start      ("abc" begins after the "_" separator)
		//   xabcx   → plain substring ("abc" buried in a lowercase run)
		const types = [node('xabcx', 'misc', ''), node('x_abc', 'misc', ''), node('abcdef', 'misc', '')];
		expect(order(types, 'abc')).toEqual(['abcdef', 'x_abc', 'xabcx']);
	});

	it('matches CamelCase / acronym word starts (e.g. "out" → *Out)', () => {
		const types = [
			node('Buffer', 'signal', 'buffers things, optionally writing output'), // doc-only
			node('AudioOut', 'outputs', ''), // word start "Out"
			node('OSCOut', 'outputs', '') // word start "Out"
		];
		const result = order(types, 'out');
		expect(result.indexOf('AudioOut')).toBeLessThan(result.indexOf('Buffer'));
		expect(result.indexOf('OSCOut')).toBeLessThan(result.indexOf('Buffer'));
	});

	it('ranks a category match above a docstring-only match', () => {
		const types = [
			node('Foo', 'misc', 'an array helper'), // doc has "array"
			node('Bar', 'array', 'does things') // category "array"
		];
		expect(order(types, 'array')).toEqual(['Bar', 'Foo']);
	});

	it('breaks ties by shorter name, then alphabetically', () => {
		const types = [node('Oscillator', 'inputs', ''), node('OSCOut', 'outputs', '')];
		// Both are name-prefix matches at index 0 → shorter wins.
		expect(order(types, 'osc')).toEqual(['OSCOut', 'Oscillator']);
	});

	it('drops non-matches and keeps an empty query untouched', () => {
		const types = [node('Oscillator', 'inputs', ''), node('Buffer', 'signal', '')];
		expect(order(types, 'osc')).toEqual(['Oscillator']);
		expect(order(types, '   ')).toEqual(['Oscillator', 'Buffer']); // whitespace == empty
	});
});
