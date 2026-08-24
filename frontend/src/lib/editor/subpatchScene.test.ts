import { describe, it, expect } from 'vitest';
import { ROOT_ID, childrenOfScope, drawEndpoint } from './subpatchScene';

// The document's own shape: every record names the scope it sits in, and a facade and a port are
// records like any other. outer > [a, inner, oout]; inner > [leaf, iout]; 'top' is a plain
// top-level node. `leaf.out` reaches the top through the port chain iout → oout.
function scopes(): Map<string, string> {
	return new Map([
		['top', ROOT_ID],
		['outer', ROOT_ID],
		['a', 'outer'],
		['inner', 'outer'],
		['oout', 'outer'],
		['leaf', 'inner'],
		['iout', 'inner']
	]);
}

describe('childrenOfScope', () => {
	it('returns only the DIRECT members of a scope — ports and nested facades included', () => {
		const idx = scopes();
		expect(childrenOfScope(ROOT_ID, idx)).toEqual(['top', 'outer']);
		expect(childrenOfScope('outer', idx)).toEqual(['a', 'inner', 'oout']);
		expect(childrenOfScope('inner', idx)).toEqual(['leaf', 'iout']);
	});
});

describe('drawEndpoint', () => {
	it('is the identity for a direct member of the entered scope', () => {
		const idx = scopes();
		expect(drawEndpoint('top', 'out', ROOT_ID, idx)).toEqual({ node: 'top', handle: 'out' });
		expect(drawEndpoint('leaf', 'out', 'inner', idx)).toEqual({ node: 'leaf', handle: 'out' });
		expect(drawEndpoint('oout', 'value', 'outer', idx)).toEqual({ node: 'oout', handle: 'value' });
	});

	it('climbs one level: a port draws as the slot of the facade that holds it', () => {
		const idx = scopes();
		expect(drawEndpoint('iout', 'value', 'outer', idx)).toEqual({ node: 'inner', handle: 'iout' });
		expect(drawEndpoint('oout', 'value', ROOT_ID, idx)).toEqual({ node: 'outer', handle: 'oout' });
	});

	it('collapses both ends of a wholly-inner link onto one facade, so the caller hides it', () => {
		const idx = scopes();
		expect(drawEndpoint('leaf', 'out', ROOT_ID, idx)?.node).toBe('outer');
		expect(drawEndpoint('iout', 'value', ROOT_ID, idx)?.node).toBe('outer');
	});

	it('returns null when the endpoint is outside the entered subtree', () => {
		const idx = scopes();
		expect(drawEndpoint('top', 'out', 'inner', idx)).toBe(null);
		expect(drawEndpoint('oout', 'value', 'inner', idx)).toBe(null);
		expect(drawEndpoint('nobody', 'out', ROOT_ID, idx)).toBe(null);
	});
});
