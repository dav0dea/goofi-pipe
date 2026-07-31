import { describe, it, expect } from 'vitest';
import { UIStore } from './ui.svelte';

// The editor standdown (undo/redo stands down while an in-panel fx editor owns the keyboard) is
// REF-COUNTED, not a shared boolean: each open editor registers a stable id, and the standdown lifts
// only when the LAST one unregisters. This pins the store side of inspector fixes #2B and #3.
describe('UIStore editor standdown (ref-counted modalOpen)', () => {
	it('is closed with no editors registered', () => {
		const ui = new UIStore();
		expect(ui.modalOpen).toBe(false);
	});

	it('opens while any editor is registered', () => {
		const ui = new UIStore();
		ui.openEditor('a');
		expect(ui.modalOpen).toBe(true);
	});

	it('stays open until the LAST editor closes — collapsing one never lifts another (#2B)', () => {
		const ui = new UIStore();
		ui.openEditor('a');
		ui.openEditor('b');
		ui.closeEditor('a');
		expect(ui.modalOpen).toBe(true); // b still holds the standdown
		ui.closeEditor('b');
		expect(ui.modalOpen).toBe(false);
	});

	it('register/unregister is idempotent (double open counts once, extra close underflows nothing)', () => {
		const ui = new UIStore();
		ui.openEditor('a');
		ui.openEditor('a');
		ui.closeEditor('a');
		expect(ui.modalOpen).toBe(false);
		ui.closeEditor('a');
		expect(ui.modalOpen).toBe(false);
	});
});

// Slot expand state is stored only for slots a writer has actually touched; an untouched slot
// answers the default. Every writer stores a real boolean, so absent and false are distinguishable
// by the stored value alone — which is what lets the lookup be a plain `?? true`.
describe('UIStore slot expand state', () => {
	it('an untouched slot defaults to expanded', () => {
		const ui = new UIStore();
		expect(ui.isSlotExpanded('n1', 'out')).toBe(true);
	});

	it('an explicit false is honoured, not mistaken for absent', () => {
		const ui = new UIStore();
		ui.setSlotExpanded('n1', 'out', false);
		expect(ui.isSlotExpanded('n1', 'out')).toBe(false);
		ui.setSlotExpanded('n1', 'out', true);
		expect(ui.isSlotExpanded('n1', 'out')).toBe(true);
	});

	it('toggle flips from the default and back, and is per-slot', () => {
		const ui = new UIStore();
		ui.toggleSlotExpanded('n1', 'out');
		expect(ui.isSlotExpanded('n1', 'out')).toBe(false);
		expect(ui.isSlotExpanded('n1', 'other'), 'a sibling slot is untouched').toBe(true);
		ui.toggleSlotExpanded('n1', 'out');
		expect(ui.isSlotExpanded('n1', 'out')).toBe(true);
	});

	it('a seeded node applies each saved collapsed flag, and forget drops them again', () => {
		const ui = new UIStore();
		ui.seedNodeViewers('n1', ['a', 'b'], { a: { collapsed: true } });
		expect([ui.isSlotExpanded('n1', 'a'), ui.isSlotExpanded('n1', 'b')]).toEqual([false, true]);
		ui.forget('n1');
		expect(ui.isSlotExpanded('n1', 'a'), 'back to the default once forgotten').toBe(true);
	});
});
