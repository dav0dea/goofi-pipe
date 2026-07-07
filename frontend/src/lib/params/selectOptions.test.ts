import { describe, it, expect } from 'vitest';
import { selectOptions } from './selectOptions';

describe('selectOptions — a string dropdown never renders blank for its own value', () => {
	it('prepends the active value when it is absent from the options', () => {
		// A saved LSL source / audio device that fell out of the live option list
		// (options are structural config, not persisted) must still show selected.
		expect(selectOptions(['goofi'], 'muse-1234')).toEqual(['muse-1234', 'goofi']);
	});

	it('leaves the options unchanged when the value is already present', () => {
		expect(selectOptions(['a', 'b'], 'b')).toEqual(['a', 'b']);
	});

	it('does not prepend an empty value', () => {
		expect(selectOptions(['a', 'b'], '')).toEqual(['a', 'b']);
	});
});
