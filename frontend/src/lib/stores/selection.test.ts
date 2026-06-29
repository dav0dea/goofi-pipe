import { describe, it, expect, beforeEach } from 'vitest';
import { selection } from './selection.svelte';

describe('selection write is a no-op when the selection is unchanged', () => {
	beforeEach(() => selection().forgetAll());

	it('re-selecting the already-selected node does not replace the panel selection', () => {
		const sel = selection();
		sel.clickNode('p', 'n1', false);
		const set1 = sel.nodes('p');

		// Re-selecting the SAME sole node (what a drag-start mousedown does) must not
		// allocate a new selection object — a fresh object retriggers the editor's
		// flowNodes effect mid-drag and Svelte Flow's onnodedragstart never fires.
		sel.clickNode('p', 'n1', false);

		expect(sel.nodes('p')).toBe(set1); // same reference => write was skipped
	});

	it('a genuine selection change still replaces the selection', () => {
		const sel = selection();
		sel.clickNode('p', 'n1', false);
		const set1 = sel.nodes('p');
		sel.clickNode('p', 'n2', false);
		expect(sel.nodes('p')).not.toBe(set1);
		expect([...sel.nodes('p')]).toEqual(['n2']);
	});
});
