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
