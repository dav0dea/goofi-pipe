import { describe, it, expect } from 'vitest';
import { serializeClipboard, parseClipboard, fragmentCentre, type GraphFragment } from './clipboard';

/** The shape `copy_nodes` answers with: the `.gfi`'s own `{nodes, links}`, keyed by uid. */
function fragment(): GraphFragment {
	return {
		nodes: {
			uidA: { pos: [0, 0] },
			uidB: { pos: [100, 40] }
		},
		links: [{ node_out: 'uidA', slot_out: 'out', node_in: 'uidB', slot_in: 'in' }]
	};
}

describe('clipboard — the payload is the manager’s own fragment, carried verbatim', () => {
	it('round-trips a fragment and refuses anything that is not one', () => {
		const clip = serializeClipboard(fragment());
		const back = parseClipboard(JSON.stringify(clip));
		// Verbatim: the frontend never re-shapes what it will hand straight back to `paste_nodes`,
		// so a record it does not understand — a facade, a port — survives a copy and a paste.
		expect(back?.doc).toEqual(fragment());

		expect(parseClipboard('not json')).toBeNull();
		expect(parseClipboard(JSON.stringify({ nodes: {} })), 'no version marker').toBeNull();
		expect(
			parseClipboard(JSON.stringify({ __goofi_clip__: 1, doc: fragment() })),
			'an older payload shape is refused rather than half-read'
		).toBeNull();
		expect(parseClipboard(JSON.stringify({ __goofi_clip__: 2, doc: {} })), 'no nodes map').toBeNull();
	});

	it('centres a fragment on its ROOTS, so a paste anchors where the user is looking', () => {
		expect(fragmentCentre(fragment())).toEqual([50, 20]);
		// A member is drawn in its sub-patch's own space, so its position is not on this canvas and
		// must not pull the anchor — a facade at the origin holding a member far out would otherwise
		// paste halfway to the member, off screen.
		expect(
			fragmentCentre({ nodes: { f: { pos: [10, 10] }, m: { pos: [9000, 9000], scope: 'f' } } })
		).toEqual([10, 10]);
		// A record with no position reads as the origin, and an empty fragment has no centre to find.
		expect(fragmentCentre({ nodes: { a: {} } })).toEqual([0, 0]);
		expect(fragmentCentre({ nodes: {} })).toEqual([0, 0]);
	});
});
