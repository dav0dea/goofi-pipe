import { describe, it, expect } from 'vitest';
import { glSupports } from './imageGL';

// The wire is f32-only. The GL path handles float grayscale (R32F) and float RGBA
// (RGBA32F); RGB and gray+alpha fall back to the 2D path.
describe('glSupports', () => {
	it('handles float grayscale and float RGBA', () => {
		expect(glSupports(1, '<f4')).toBe(true);
		expect(glSupports(4, '<f4')).toBe(true);
	});
	it('falls back for RGB (RGB32F is not core-guaranteed) and gray+alpha', () => {
		expect(glSupports(3, '<f4')).toBe(false);
		expect(glSupports(2, '<f4')).toBe(false);
	});
	it('rejects any non-float dtype — the encoder can only emit f32', () => {
		expect(glSupports(1, '|u1')).toBe(false);
		expect(glSupports(4, '<i8')).toBe(false);
	});
});
