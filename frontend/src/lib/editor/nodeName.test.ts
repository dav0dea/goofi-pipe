import { describe, expect, it } from 'vitest';
import { joinNodeName, splitNodeName } from './nodeName';

describe('splitNodeName', () => {
	it('a top-level node has no path; the whole name is editable', () => {
		expect(splitNodeName('oscillator0')).toEqual({ path: '', base: 'oscillator0' });
	});

	it('a sub-patch member splits on the trailing :: into faint path + editable base', () => {
		expect(splitNodeName('subpatch0::filter0')).toEqual({ path: 'subpatch0::', base: 'filter0' });
	});

	it('a nested member keeps the full path faint and only the final segment editable', () => {
		expect(splitNodeName('a::b::c')).toEqual({ path: 'a::b::', base: 'c' });
	});

	it('tolerates an empty name', () => {
		expect(splitNodeName('')).toEqual({ path: '', base: '' });
	});
});

describe('joinNodeName', () => {
	it('recombines a path and an edited base into a full qualified name', () => {
		expect(joinNodeName('subpatch0::', 'myfilter')).toBe('subpatch0::myfilter');
	});

	it('a bare base with no path is returned unchanged', () => {
		expect(joinNodeName('', 'oscillator0')).toBe('oscillator0');
	});

	it('round-trips: split then join reproduces the original', () => {
		for (const name of ['osc0', 'subpatch0::filter0', 'a::b::c']) {
			const { path, base } = splitNodeName(name);
			expect(joinNodeName(path, base)).toBe(name);
		}
	});
});
