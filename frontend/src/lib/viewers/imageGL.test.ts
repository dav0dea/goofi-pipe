import { describe, it, expect } from 'vitest';
import { glSupports } from './imageGL';

// The GL path handles uint8 RGB/RGBA (HD video) + uint8/float grayscale; float
// RGB and gray+alpha fall back to the 2D path (RGB32F isn't core-guaranteed).
describe('glSupports', () => {
	it('handles grayscale (uint8 + float)', () => {
		expect(glSupports(1, false)).toBe(true);
		expect(glSupports(1, true)).toBe(true);
	});
	it('handles uint8 RGB but not float RGB', () => {
		expect(glSupports(3, false)).toBe(true);
		expect(glSupports(3, true)).toBe(false);
	});
	it('handles RGBA (uint8 + float)', () => {
		expect(glSupports(4, false)).toBe(true);
		expect(glSupports(4, true)).toBe(true);
	});
	it('falls back for gray+alpha (c===2)', () => {
		expect(glSupports(2, false)).toBe(false);
		expect(glSupports(2, true)).toBe(false);
	});
});
