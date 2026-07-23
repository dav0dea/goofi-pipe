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

	it('renders the active value alone when a scan came back empty', () => {
		// A refreshable picker whose scan found nothing (no devices, no LSL streams) still has
		// to render — and stay refreshable — rather than collapsing to a blank dropdown.
		expect(selectOptions([], 'mic')).toEqual(['mic']);
		expect(selectOptions([], '')).toEqual([]);
	});
});
