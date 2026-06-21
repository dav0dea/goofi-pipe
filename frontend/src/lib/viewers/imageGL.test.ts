import { describe, it, expect } from 'vitest';
import { glSupports } from './imageGL';

// The GL path handles uint8 RGB/RGBA (HD video) + uint8/float grayscale; float
// RGB, gray+alpha, and non-uint8 integer dtypes (u2/i2/u4/i4/i8/u8) fall back to
// the 2D path (wrong texture type / BigInt arrays would crash texImage2D).
describe('glSupports', () => {
	it('handles grayscale uint8 + float', () => {
		expect(glSupports(1, '|u1')).toBe(true);
		expect(glSupports(1, '<f4')).toBe(true);
		expect(glSupports(1, '<f8')).toBe(true);
	});
	it('handles uint8 RGB but not float RGB', () => {
		expect(glSupports(3, '|u1')).toBe(true);
		expect(glSupports(3, '<f4')).toBe(false);
	});
	it('handles RGBA uint8 + float', () => {
		expect(glSupports(4, '|u1')).toBe(true);
		expect(glSupports(4, '<f4')).toBe(true);
	});
	it('falls back for gray+alpha (c===2)', () => {
		expect(glSupports(2, '|u1')).toBe(false);
	});
	it('rejects non-uint8 integer dtypes (would crash or mis-normalize)', () => {
		expect(glSupports(1, '<u2')).toBe(false);
		expect(glSupports(1, '<i2')).toBe(false);
		expect(glSupports(1, '<u4')).toBe(false);
		expect(glSupports(1, '<i4')).toBe(false);
		expect(glSupports(1, '<i8')).toBe(false); // BigInt64Array
		expect(glSupports(3, '<u2')).toBe(false);
	});
});
