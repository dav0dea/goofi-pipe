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

/* Multi-select mode (D-R4). The user asked for a MODE, not a gesture: "one can enable
   multi-select where tapping on a node then selects it". A phone has no shift, ctrl or meta, so
   without it a touch user can select exactly one node — and group / delete-many / copy-many are
   all multi-node commands.
   The mode is folded in HERE rather than OR-ed in at the click site, so what "additive" means has
   one definition and a future caller cannot forget the mode. */
describe('multi-select mode', () => {
	beforeEach(() => {
		selection().forgetAll();
		if (selection().multiSelect) selection().toggleMultiSelect();
	});

	it('is off by default', () => {
		expect(selection().multiSelect).toBe(false);
	});

	it('makes a plain node click additive', () => {
		const sel = selection();
		sel.clickNode('p', 'n1', false);
		sel.toggleMultiSelect();
		sel.clickNode('p', 'n2', false);
		expect([...sel.nodes('p')].sort()).toEqual(['n1', 'n2']);
	});

	it('still toggles a node back off, exactly as shift-click does', () => {
		const sel = selection();
		sel.toggleMultiSelect();
		sel.clickNode('p', 'n1', false);
		sel.clickNode('p', 'n2', false);
		sel.clickNode('p', 'n1', false);
		expect([...sel.nodes('p')]).toEqual(['n2']);
	});

	it('keeps a plain edge click from wiping the node selection', () => {
		const sel = selection();
		sel.clickNode('p', 'n1', false);
		sel.toggleMultiSelect();
		sel.clickEdge('p', 'e1', false);
		expect([...sel.nodes('p')]).toEqual(['n1']);
		expect([...sel.edges('p')]).toEqual(['e1']);
	});

	it('restores replace-on-click when switched off', () => {
		const sel = selection();
		sel.toggleMultiSelect();
		sel.clickNode('p', 'n1', false);
		sel.toggleMultiSelect();
		sel.clickNode('p', 'n2', false);
		expect([...sel.nodes('p')]).toEqual(['n2']);
	});

	it('survives forgetAll — it is a session mode, not per-panel state', () => {
		const sel = selection();
		sel.toggleMultiSelect();
		sel.forgetAll();
		expect(sel.multiSelect).toBe(true);
	});
});
