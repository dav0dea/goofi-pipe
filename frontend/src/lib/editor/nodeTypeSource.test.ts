import { describe, it, expect } from 'vitest';
import { nodeTypeSource } from './nodeTypeSource';
import { typeInfo as ty } from '$lib/test/typeInfo';

describe('nodeTypeSource — the one word a palette row carries', () => {
	it('names the patch a node came with, so a patch-local node is distinguishable', () => {
		expect(nodeTypeSource(ty({ source: 'patch' }))).toBe('this patch');
		expect(nodeTypeSource(ty({ source: 'builtin' }))).toBe('builtin');
	});

	// Categories are gone from the menu (the user asked for a flat list), and `unavailable` was one
	// of them. The word has to survive that removal on the row itself: greyed-and-unclickable alone
	// does not say WHY, and a node that cannot load must never read as a node that goofi ignored.
	it('says unavailable before it says where the file lives', () => {
		expect(nodeTypeSource(ty({ available: false, source: 'patch' }))).toBe('unavailable');
		expect(nodeTypeSource(ty({ available: false, source: 'builtin' }))).toBe('unavailable');
	});
});
