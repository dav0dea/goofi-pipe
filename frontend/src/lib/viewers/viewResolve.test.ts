import { describe, it, expect } from 'vitest';
import { resolveKind } from './kind';
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
