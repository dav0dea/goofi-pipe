import { describe, it, expect } from 'vitest';
import { glSupports } from './imageGL';

// The wire is f32-only. The GL path handles float grayscale (R32F), RGB (RGB32F) and
// RGBA (RGBA32F); only gray+alpha falls back to the 2D path.
describe('glSupports', () => {
	it('handles float grayscale and float RGBA', () => {
		expect(glSupports(1, '<f4')).toBe(true);
		expect(glSupports(4, '<f4')).toBe(true);
	});
	it('handles float RGB — video must not fall to the per-pixel 2D path', () => {
		// RGB32F is a required TEXTURE format in WebGL2; it is only non-renderable and
		// non-filterable, and the renderer already picks NEAREST without float-linear.
		expect(glSupports(3, '<f4')).toBe(true);
	});
	it('falls back for gray+alpha (c === 2)', () => {
		expect(glSupports(2, '<f4')).toBe(false);
	});
	it('rejects any non-float dtype — the encoder can only emit f32', () => {
		expect(glSupports(1, '|u1')).toBe(false);
		expect(glSupports(4, '<i8')).toBe(false);
	});
});
