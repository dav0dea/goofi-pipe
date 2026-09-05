import { describe, it, expect } from 'vitest';
import { bareName, engineColor, engineOf, familyColor } from './typeId';

describe('the engine:Name type id', () => {
	it('splits a qualified id', () => {
		expect(bareName('audio:Filter')).toBe('Filter');
		expect(engineOf('audio:Filter')).toBe('audio');
	});
	it('leaves a structural type bare', () => {
		expect(bareName('SubPatch')).toBe('SubPatch');
		expect(engineOf('SubPatch')).toBeNull();
	});
	it('names one ink per engine, and falls back where there is no token', () => {
		expect(engineColor('audio:Filter')).toBe('var(--engine-audio, var(--text-muted))');
		expect(engineColor('signal:Filter')).toBe('var(--engine-signal, var(--text-muted))');
		expect(engineColor('SubPatch')).toBe('var(--text-muted)');
	});
	it('inks a plugin format the same way, off the palette tab rather than the engine', () => {
		expect(familyColor('vst')).toBe('var(--engine-vst, var(--text-muted))');
		expect(familyColor(null)).toBe('var(--text-muted)');
	});
});
