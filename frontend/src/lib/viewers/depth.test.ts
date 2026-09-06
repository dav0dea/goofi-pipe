import { describe, expect, it } from 'vitest';
import { isU8, sampleRange, toUnit } from './depth';

describe('the 8-bit viewer hop', () => {
	it('names the dtype', () => {
		expect(isU8('|u1')).toBe(true);
		expect(isU8('<f4')).toBe(false);
	});

	it('reads the range the texels span off the meta', () => {
		expect(sampleRange({ reduced: { depth: { lo: -2, hi: 2 } } })).toEqual([-2, 2]);
	});

	it('answers null when the frame carries no range', () => {
		expect(sampleRange({})).toBeNull();
		expect(sampleRange(undefined)).toBeNull();
		expect(sampleRange({ reduced: { orig_len: 8 } })).toBeNull();
	});

	it('maps a texel back into the frame’s units', () => {
		expect(toUnit(0, [-2, 2])).toBe(-2);
		expect(toUnit(255, [-2, 2])).toBe(2);
		expect(toUnit(0, [0, 1])).toBe(0);
		expect(toUnit(255, [0, 1])).toBe(1);
	});
});
