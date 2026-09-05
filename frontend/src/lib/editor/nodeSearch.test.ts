import { describe, it, expect } from 'vitest';
import { rankNodeTypes } from './nodeSearch';
import type { NodeTypeInfo } from '$lib/api/control';
import { typeInfo } from '$lib/test/typeInfo';

const node = (type: string, tags: NodeTypeInfo['tags'], doc = '') =>
	typeInfo({ type, tags, doc });

const order = (types: NodeTypeInfo[], q: string): string[] =>
	rankNodeTypes(types, q).map((t) => t.type);

describe('rankNodeTypes', () => {
	it('ranks name matches above a docstring-only match (the "osc" case)', () => {
		const types = [
			node('Kuramoto', [], 'Coupled Kuramoto oscillators producing phase signals.'),
			node('Oscillator', [], 'Generate a periodic waveform.'),
			node('OSCOut', [], 'Send values over Open Sound Control.')
		];
		const result = order(types, 'osc');
		// Both name hits come before the doc-only hit.
		expect(result.indexOf('Oscillator')).toBeLessThan(result.indexOf('Kuramoto'));
		expect(result.indexOf('OSCOut')).toBeLessThan(result.indexOf('Kuramoto'));
		expect(result[result.length - 1]).toBe('Kuramoto');
	});

	it('orders exact name above prefix', () => {
		const types = [node('FFTWindow', [], ''), node('FFT', [], '')];
		expect(order(types, 'fft')).toEqual(['FFT', 'FFTWindow']);
	});

	it('orders prefix > word-start > plain substring within the name', () => {
		// Synthetic names isolate each tier (real CamelCase makes most inner
		// capitals word starts, so they rarely fall to the substring tier):
		//   abcdef  → prefix          (starts with "abc")
		//   x_abc   → word start      ("abc" begins after the "_" separator)
		//   xabcx   → plain substring ("abc" buried in a lowercase run)
		const types = [node('xabcx', [], ''), node('x_abc', [], ''), node('abcdef', [], '')];
		expect(order(types, 'abc')).toEqual(['abcdef', 'x_abc', 'xabcx']);
	});

	it('matches CamelCase / acronym word starts (e.g. "out" → *Out)', () => {
		const types = [
			node('Buffer', [], 'buffers things, optionally writing output'), // doc-only
			node('AudioOut', [], ''), // word start "Out"
			node('OSCOut', [], '') // word start "Out"
		];
		const result = order(types, 'out');
		expect(result.indexOf('AudioOut')).toBeLessThan(result.indexOf('Buffer'));
		expect(result.indexOf('OSCOut')).toBeLessThan(result.indexOf('Buffer'));
	});

	it('ranks a tag match above a docstring-only match', () => {
		const types = [
			node('Foo', [], 'an eeg helper'), // doc has "eeg"
			node('Bar', ['eeg'], 'does things') // tagged "eeg"
		];
		expect(order(types, 'eeg')).toEqual(['Bar', 'Foo']);
	});

	it('breaks ties by shorter name, then alphabetically', () => {
		const types = [node('Oscillator', [], ''), node('OSCOut', [], '')];
		// Both are name-prefix matches at index 0 → shorter wins.
		expect(order(types, 'osc')).toEqual(['OSCOut', 'Oscillator']);
	});

	it('ranks on the bare name, and the engine matches nothing', () => {
		const types = [
			node('signal:Filter', [], ''),
			node('audio:FilterB', [], ''),
			node('signal:FilterA', [], '')
		];
		// Exact bare name first, then the two prefixes alphabetically — the engine decides no tier,
		// no match position and no tie.
		expect(order(types, 'filter')).toEqual(['signal:Filter', 'signal:FilterA', 'audio:FilterB']);
		// The engine names a palette tab, never a node.
		expect(order(types, 'signal')).toEqual([]);
	});

	it('drops non-matches and keeps an empty query untouched', () => {
		const types = [node('Oscillator', [], ''), node('Buffer', [], '')];
		expect(order(types, 'osc')).toEqual(['Oscillator']);
		expect(order(types, '   ')).toEqual(['Oscillator', 'Buffer']); // whitespace == empty
	});
});
