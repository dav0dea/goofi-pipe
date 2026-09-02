import { describe, it, expect } from 'vitest';
import { EditorState } from '@codemirror/state';
import { singleLineExpression } from './singleLine';

/* D-X1 makes the INLINE param field a CodeMirror editor too, so there is one editor component rather
 * than two implementations. Its contract is that the document stays one line and the field never
 * grows.
 *
 * Enter is not this extension's job — the keymap binds it to COMMIT, so a typed Enter never reaches
 * the document. This is the backstop for every other way a newline gets in: a paste, a drop, a
 * programmatic set. Flattened to a space rather than deleted, because joining `nd('a')` and `+ 1`
 * with nothing would silently change the expression's meaning.
 */

const state = (doc: string, singleLine = true) =>
	EditorState.create({ doc, extensions: singleLine ? [singleLineExpression] : [] });

describe('single-line mode rejects newlines', () => {
	it('flattens a pasted multi-line expression to one line', () => {
		const tr = state("nd('a')").update({ changes: { from: 7, insert: "\n+ 1\n* 2" } });
		expect(tr.state.doc.lines).toBe(1);
		expect(tr.state.doc.toString()).toBe("nd('a') + 1 * 2");
	});

	// The control: without the extension the same transaction lands three lines, so the assertion
	// above is measuring the extension rather than something CodeMirror does anyway.
	it('is the extension doing it, not the editor', () => {
		const tr = state("nd('a')", false).update({ changes: { from: 7, insert: "\n+ 1\n* 2" } });
		expect(tr.state.doc.lines).toBe(3);
	});

	// A newline is one character and so is the space replacing it, so the caret the transaction asked
	// for is still the caret it gets — no off-by-one after a paste.
	it('keeps the selection the transaction asked for', () => {
		const tr = state('').update({ changes: { from: 0, insert: 'a\nb' }, selection: { anchor: 3 } });
		expect(tr.state.doc.toString()).toBe('a b');
		expect(tr.state.selection.main.head).toBe(3);
	});

	it('leaves a single-line change completely alone', () => {
		const tr = state('a').update({ changes: { from: 1, insert: ' + 1' } });
		expect(tr.state.doc.toString()).toBe('a + 1');
	});

	// Replacing a range, not just inserting: the flattening must rebuild every change in the set.
	it('flattens a replacement of existing text', () => {
		const tr = state('abc').update({ changes: { from: 1, to: 2, insert: 'X\nY' } });
		expect(tr.state.doc.toString()).toBe('aX Yc');
	});
});
