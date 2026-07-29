import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
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
