import { describe, expect, it } from 'vitest';
import { NO_NODE, nodePickList } from './nodeChoices';

const set = [
	{ uid: 'u-osc10', name: 'oscillator10' },
	{ uid: 'u-osc2', name: 'oscillator2' },
	{ uid: 'u-buf', name: 'buffer0' }
];

describe('the node a panel binds, as a list of choices', () => {
	it('lists every live node, the empty choice first and the rest by display name', () => {
		const l = nodePickList(set, null, 'No node');
		expect(l.options).toEqual([NO_NODE, 'u-buf', 'u-osc2', 'u-osc10']);
		expect(l.labels[NO_NODE]).toBe('No node');
	});

	it('commits the UID and shows the display name, so a rename cannot rebind the panel', () => {
		const l = nodePickList(set, 'u-buf', 'No node');
		expect(l.value).toBe('u-buf');
		expect(l.labels['u-buf']).toBe('buffer0');
	});

	/* The engine unbinds a panel when its node is removed, but the panel's own state is a REPLICA of
	   a doc write that has not landed yet — and a `.gfi` can name a node the patch no longer has.
	   Either way the picker must read as unbound, never as a raw uid: `Select` keeps a truthy value
	   that is not among its options by PREPENDING it, so an unguarded stale uid would draw itself. */
	it('reads a binding whose node is gone as unbound, never as a raw uid', () => {
		const l = nodePickList(set, 'u-deleted', 'No node');
		expect(l.value).toBe(NO_NODE);
		expect(l.options).not.toContain('u-deleted');
	});

	it('takes the empty choice’s wording from the panel, because a console filters and a viewer binds', () => {
		expect(nodePickList(set, null, 'All nodes').labels[NO_NODE]).toBe('All nodes');
	});

	it('survives a patch with no nodes at all — the empty choice is still a choice', () => {
		const l = nodePickList([], 'u-buf', 'No node');
		expect(l.options).toEqual([NO_NODE]);
		expect(l.value).toBe(NO_NODE);
	});
});
