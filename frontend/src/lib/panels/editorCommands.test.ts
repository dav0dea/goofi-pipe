import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
	activeOrOnlyEditor,
	editorFor,
	editorAt,
	registerEditor,
	unregisterEditor,
	type EditorCommands
} from './editorCommands';

function stub(): EditorCommands {
	return {
		openAddMenu: vi.fn(),
		fitView: vi.fn(),
		focusNode: vi.fn(),
		selectAll: vi.fn(),
		clearSelection: vi.fn(),
		deleteSelection: vi.fn(),
		groupSelection: vi.fn(),
		copySelection: vi.fn(),
		pasteClipboard: vi.fn(),
		duplicateSelection: vi.fn(),
		hasSelection: () => false
	};
}

describe('editorAt — the strict lookup the app header addresses through', () => {
	beforeEach(() => {
		unregisterEditor('a');
		unregisterEditor('b');
	});

	it('resolves a registered editor', () => {
		const a = stub();
		registerEditor('a', a);
		expect(editorAt('a')).toBe(a);
	});

	/* The header is app-global chrome and several node editors can be open, so a command that
	   deletes or groups must not land in an editor the user is not looking at. `editorFor`'s
	   "first registered" fallback is right for the error chip (which is showing you a node and
	   just needs somewhere to show it) and wrong here — the menu row disables instead. */
	it('does NOT fall back to some other editor', () => {
		registerEditor('a', stub());
		expect(editorAt('b')).toBeNull();
		expect(editorFor('b'), 'the lenient lookup still falls back, for its own callers').not.toBeNull();
	});

	it('resolves nothing for a null panel id', () => {
		registerEditor('a', stub());
		expect(editorAt(null)).toBeNull();
	});

	it('stops resolving an editor that unmounted', () => {
		registerEditor('a', stub());
		unregisterEditor('a');
		expect(editorAt('a')).toBeNull();
	});
});

/* The header's rows all resolve through the ACTIVE editor id, and that id can name a panel that is
   no longer an editor: `hydrate` focuses the loaded layout's first panel (and `forgetAll` nulls the
   id), `selectTab` lands on a tab whose first panel is a console, and `navContext` writes back
   whatever panel an undo re-oriented to. With one editor on screen every canvas row then went dead
   — Select all and Paste on `!ed`, and Delete/Group/Copy/Duplicate on the `!has` that follows from
   it — and the only recovery was a pointerdown somewhere in the editor, which nothing tells you. */
describe('activeOrOnlyEditor — the header, when the active id has gone stale', () => {
	beforeEach(() => {
		unregisterEditor('a');
		unregisterEditor('b');
	});

	it('prefers the active editor when the id resolves', () => {
		const a = stub();
		const b = stub();
		registerEditor('a', a);
		registerEditor('b', b);
		expect(activeOrOnlyEditor('a')).toBe(a);
	});

	it('resolves the ONE open editor when the id does not', () => {
		const a = stub();
		registerEditor('a', a);
		expect(activeOrOnlyEditor('stale')).toBe(a);
		expect(activeOrOnlyEditor(null)).toBe(a);
	});

	it('still refuses to guess between two', () => {
		registerEditor('a', stub());
		registerEditor('b', stub());
		expect(activeOrOnlyEditor('stale')).toBeNull();
	});

	it('resolves nothing with no editor open', () => {
		expect(activeOrOnlyEditor('a')).toBeNull();
	});
});
