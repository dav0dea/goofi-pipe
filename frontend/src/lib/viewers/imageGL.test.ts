import { describe, it, expect } from 'vitest';
import { glSupports } from './imageGL';

// The wire is f32, plus the reducer's 8-bit hop for an image. The GL path handles grayscale
// (R32F/R8), RGB (RGB32F/RGB8) and RGBA; only gray+alpha falls back to the 2D path.
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
	it('handles the 8-bit hop, which is what an image viewer asks for', () => {
		expect(glSupports(1, '|u1')).toBe(true);
		expect(glSupports(3, '|u1')).toBe(true);
		expect(glSupports(4, '|u1')).toBe(true);
		expect(glSupports(2, '|u1')).toBe(false);
	});
	it('rejects every other dtype — nothing else reaches a viewer', () => {
		expect(glSupports(4, '<i8')).toBe(false);
		expect(glSupports(1, '<u2')).toBe(false);
	});
});
