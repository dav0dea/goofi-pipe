import { describe, it, expect } from 'vitest';
import { metaEntries, formatMetaValue, metaPreview } from './metaFormat';

describe('metaEntries', () => {
	it('returns top-level entries in insertion order', () => {
		expect(metaEntries({ b: 1, a: 2 })).toEqual([
			['b', 1],
			['a', 2]
		]);
	});
	it('is empty for null/undefined/non-object', () => {
		expect(metaEntries(null)).toEqual([]);
		expect(metaEntries(undefined)).toEqual([]);
	});
	it('hides __*__-namespaced internal-marker keys', () => {
		expect(metaEntries({ sfreq: 250, __internal__: { stats: {} } })).toEqual([['sfreq', 250]]);
	});
});

describe('formatMetaValue', () => {
	it('renders scalars as plain text (no quotes)', () => {
		expect(formatMetaValue(250)).toBe('250');
		expect(formatMetaValue('EEG')).toBe('EEG');
		expect(formatMetaValue(true)).toBe('true');
		expect(formatMetaValue(null)).toBe('null');
	});

	it('renders a list inline on a single line (the DOM word-wraps it to fill width)', () => {
		expect(formatMetaValue([1, 2, 3])).toBe('[1, 2, 3]');
		expect(formatMetaValue(['Fz', 'Cz', 'Pz'])).toBe('[Fz, Cz, Pz]');
		expect(formatMetaValue([])).toBe('[]');
	});

	it('keeps lists inline even nested inside an object', () => {
		expect(formatMetaValue({ ch_names: ['Fz', 'Cz'], sfreq: 250 })).toBe('ch_names: [Fz, Cz]\nsfreq: 250');
	});

	it('renders nested dicts as indented multi-line text', () => {
		expect(formatMetaValue({ info: { highpass: 0.1, lowpass: 40 } })).toBe(
			'info:\n  highpass: 0.1\n  lowpass: 40'
		);
		expect(formatMetaValue({})).toBe('{}');
	});

	it('renders a list of dicts inline (it is still a list)', () => {
		expect(formatMetaValue([{ a: 1 }, { a: 2 }])).toBe('[{a: 1}, {a: 2}]');
	});

	it('formats bigint without throwing (msgpack int64)', () => {
		expect(formatMetaValue(10n)).toBe('10');
		expect(formatMetaValue([1n, 2n])).toBe('[1, 2]');
	});

	it('caps very long lists so a huge coord array never builds a megabyte string', () => {
		const big = Array.from({ length: 5000 }, (_, i) => i);
		const out = formatMetaValue(big);
		expect(out.startsWith('[0, 1, 2,')).toBe(true);
		expect(out).toContain('… (+4800 more)'); // 5000 - 200 cap
		expect(out.length).toBeLessThan(2000);
	});

	it('keeps full precision — the two-decimal cap is the header line only', () => {
		expect(formatMetaValue(3.14159)).toBe('3.14159');
		expect(formatMetaValue([1.23456, 2.5])).toBe('[1.23456, 2.5]');
		expect(formatMetaValue({ highpass: 0.00001 })).toBe('highpass: 0.00001');
	});

	it('renders a typed array (msgpack bin → Uint8Array) like a plain list, not an indexed object', () => {
		expect(formatMetaValue(new Uint8Array([1, 2, 3]))).toBe('[1, 2, 3]');
		expect(formatMetaValue({ buf: new Float32Array([1.5, 2.5]) })).toBe('buf: [1.5, 2.5]');
		expect(metaPreview(new Uint8Array([1, 2, 3]))).toBe('[3]');
	});
});

describe('metaPreview', () => {
	it('summarizes containers by size and shows scalars directly', () => {
		expect(metaPreview([1, 2, 3])).toBe('[3]');
		expect(metaPreview({ a: 1, b: 2 })).toBe('{2}');
		expect(metaPreview(250)).toBe('250');
		expect(metaPreview('hi')).toBe('hi');
	});
	it('truncates a long scalar preview', () => {
		expect(metaPreview('x'.repeat(100)).endsWith('…')).toBe(true);
	});

	it('caps a number at two decimals, trailing zeros trimmed', () => {
		expect(metaPreview(3.14159)).toBe('3.14');
		expect(metaPreview(3)).toBe('3');
		expect(metaPreview(3.5)).toBe('3.5');
		expect(metaPreview(-2.71828)).toBe('-2.72');
		expect(metaPreview(333.333333)).toBe('333.33');
		expect(metaPreview(1000)).toBe('1000');
	});

	it('renders a magnitude that rounds away as plain 0, never a signed zero', () => {
		expect(metaPreview(-0.00001)).toBe('0');
		expect(metaPreview(0.001)).toBe('0');
		expect(metaPreview(-0)).toBe('0');
		expect(metaPreview(0)).toBe('0');
	});

	it('leaves a magnitude extreme in the exponential form JS already chose', () => {
		expect(metaPreview(1e-7)).toBe('1e-7');
		expect(metaPreview(-1.23456e-9)).toBe('-1.23456e-9');
		expect(metaPreview(1e21)).toBe('1e+21');
	});

	it('caps only numbers — a string, a bool, a bigint and a non-finite are untouched', () => {
		expect(metaPreview('3.14159')).toBe('3.14159');
		expect(metaPreview(true)).toBe('true');
		expect(metaPreview(10n)).toBe('10');
		expect(metaPreview(NaN)).toBe('NaN');
		expect(metaPreview(Infinity)).toBe('Infinity');
		expect(metaPreview(null)).toBe('null');
	});

	it('never reaches a container’s elements — a container previews as its size', () => {
		expect(metaPreview([1.23456, 2.5])).toBe('[2]');
		expect(metaPreview({ highpass: 0.123456 })).toBe('{1}');
	});
});

import { reconstructMeta } from './metaFormat';

describe('reconstructMeta (Option C reduction-aware inspector)', () => {
	it('restores original shape + drops envelope co-reduced coord + hides reduced', () => {
		const reducedMeta = {
			shape: [2000],
			dtype: '<f4',
			channels: { dim0: new Array(2000).fill(0) }, // co-reduced artifact
			reduced: { '0': { orig_len: 10000, method: 'envelope' } },
			sfreq: 250
		};
		const out = reconstructMeta(reducedMeta);
		expect(out.shape).toEqual([10000]); // true original length
		expect('reduced' in out).toBe(false); // artifact hidden
		expect((out.channels as Record<string, unknown>).dim0).toBeUndefined(); // dropped
		expect(out.sfreq).toBe(250); // untouched
	});

	it('restores carried orig_coord for a subsampled axis', () => {
		const out = reconstructMeta({
			shape: [3, 100],
			channels: { dim0: ['a', 'c', 'f'], dim1: [] },
			reduced: {
				'0': { orig_len: 6, method: 'subsample', orig_coord: ['a', 'b', 'c', 'd', 'e', 'f'] }
			}
		});
		expect(out.shape).toEqual([6, 100]);
		expect((out.channels as Record<string, unknown>).dim0).toEqual(['a', 'b', 'c', 'd', 'e', 'f']);
	});

	it('returns meta unchanged when there is no reduction', () => {
		const m = { shape: [128], sfreq: 250, channels: {} };
		expect(reconstructMeta(m)).toBe(m);
	});
});
