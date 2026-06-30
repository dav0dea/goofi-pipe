import { describe, it, expect } from 'vitest';
import { formatTick } from './format';

describe('formatTick — compact numeric label ladder', () => {
	it('renders zero plainly', () => {
		expect(formatTick(0)).toBe('0');
	});

	it('uses exponential below 0.01 and at/above 10000', () => {
		expect(formatTick(0.009)).toBe('9.0e-3');
		expect(formatTick(10000)).toBe('1.0e+4');
		expect(formatTick(9999)).toBe('9999'); // just under the upper cutoff → fixed(0)
	});

	it('drops decimals at/above 100', () => {
		expect(formatTick(100)).toBe('100');
		expect(formatTick(99.5)).toBe('99.50'); // below 100 → fixed(2)
	});

	it('keeps 2 decimals in [1, 100) and 3 below 1', () => {
		expect(formatTick(1)).toBe('1.00');
		expect(formatTick(0.5)).toBe('0.500');
	});

	it('returns empty string for non-finite values', () => {
		expect(formatTick(NaN)).toBe('');
		expect(formatTick(Infinity)).toBe('');
		expect(formatTick(-Infinity)).toBe('');
	});
});
