import { describe, it, expect } from 'vitest';
import { bareName, engineOf } from './typeId';

describe('the engine:Name type id', () => {
	it('splits a qualified id', () => {
		expect(bareName('audio:Filter')).toBe('Filter');
		expect(engineOf('audio:Filter')).toBe('audio');
	});
	it('leaves a structural type bare', () => {
		expect(bareName('SubPatch')).toBe('SubPatch');
		expect(engineOf('SubPatch')).toBeNull();
	});
});
