import { describe, expect, it } from 'vitest';
import type { DataFrame } from '$lib/codec/decode';
import { leafSummary, tableChildren } from './tableTree';

const str = (s: string): DataFrame => ({ dtype: 'STRING', data: s, meta: {} });
const scalar = (n: number): DataFrame => ({
	dtype: 'ARRAY',
	data: { dtype: '<f4', shape: [1], values: [n] },
	meta: {}
});
const arr = (shape: number[]): DataFrame => ({
	dtype: 'ARRAY',
	data: { dtype: '<f4', shape, values: [] },
	meta: {}
});
const table = (fields: Record<string, DataFrame>): DataFrame => ({
	dtype: 'TABLE',
	data: fields,
	meta: {}
});

describe('tableChildren', () => {
	it('returns entries for a TABLE, empty for a leaf', () => {
		expect(tableChildren(table({ a: str('x'), b: scalar(1) })).map(([k]) => k)).toEqual(['a', 'b']);
		expect(tableChildren(str('x'))).toEqual([]);
	});
});

describe('leafSummary', () => {
	it('formats scalars to the given decimals', () => {
		expect(leafSummary(scalar(3.14159), 2)).toBe('3.14');
	});
	it('summarizes a multi-d array by shape', () => {
		expect(leafSummary(arr([4, 8]), 3)).toBe('array[4×8]');
	});
	it('truncates a string', () => {
		expect(leafSummary(str('hello world'), 3)).toBe('hello world');
	});
	it('counts fields for a nested table', () => {
		expect(leafSummary(table({ a: str('x'), b: str('y') }), 3)).toBe('{2 fields}');
	});
});
