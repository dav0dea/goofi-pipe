import { describe, it, expect } from 'vitest';
import { resolveKind, isRenderable } from './kind';
import { resolveSettings } from './settingsSchema';

describe('resolveKind', () => {
	it('forces string/table viewers by dtype regardless of stored kind', () => {
		expect(resolveKind('STRING', 'image')).toBe('string');
		expect(resolveKind('TABLE', 'line')).toBe('table');
	});
	it('uses the stored kind for ARRAY, falling back to line', () => {
		expect(resolveKind('ARRAY', 'image')).toBe('image');
		expect(resolveKind('ARRAY', undefined)).toBe('line');
	});
	it('falls back to line for null dtype', () => {
		expect(resolveKind(null, undefined)).toBe('line');
	});
});

describe('isRenderable', () => {
	const spec = (...shape: number[]) => ({ dtype: '<f4', shape, values: new Float32Array(shape.reduce((a, b) => a * b, 1)) });

	it('line draws 1-D and 2-D (C,N) — and nothing higher', () => {
		expect(isRenderable('line', spec(128))).toBe(true);
		expect(isRenderable('line', spec(4, 128))).toBe(true);
		// ArrayViewer.pushData returns without touching uPlot above 2-D, so a 3-D frame on the
		// DEFAULT viewer kind would sit blank (or frozen on the last 2-D frame) forever. It must
		// take the same HighDimFallback a 4-D frame already gets.
		expect(isRenderable('line', spec(4, 4, 3))).toBe(false);
		expect(isRenderable('line', spec(2, 4, 4, 3))).toBe(false);
	});

	it('a non-array frame is always renderable by its own dedicated viewer', () => {
		expect(isRenderable('string', null)).toBe(true);
	});
});

describe('resolveSettings', () => {
	it('merges overrides over the kind defaults', () => {
		const merged = resolveSettings('line', { logY: true });
		expect(merged.logY).toBe(true);
		// a default key from the line schema is still present alongside the override
		expect(Object.keys(merged).length).toBeGreaterThan(1);
	});
	it('returns pure defaults when overrides are absent', () => {
		expect(resolveSettings('image', undefined)).toEqual(resolveSettings('image', {}));
	});
	it('image defaults to keeping aspect ratio; stretch is an opt-in override', () => {
		expect(resolveSettings('image', {}).stretch).toBe(false);
		expect(resolveSettings('image', { stretch: true }).stretch).toBe(true);
	});
});
