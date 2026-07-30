/**
 * The single-line half of D-X1 — the inline param field is the SAME editor component as the expanded
 * one, in a mode where the document is one line and stays one line.
 *
 * A typed Enter never gets here: the keymap binds it to COMMIT (D-X6), which is what the plain
 * `TextInput` did. This is the backstop for every other way a line break arrives — a paste, a drop,
 * a programmatic set — and it FLATTENS rather than drops, because joining `nd('a')` to `+ 1` with
 * nothing would silently change the expression instead of refusing the newline.
 */
import { EditorState, type ChangeSpec } from '@codemirror/state';

export const singleLineExpression = EditorState.transactionFilter.of((tr) => {
	if (!tr.docChanged || tr.newDoc.lines === 1) return tr;
	const changes: ChangeSpec[] = [];
	tr.changes.iterChanges((fromA, toA, _fromB, _toB, inserted) => {
		changes.push({
			from: fromA,
			to: toA,
			// `sliceString`'s third argument is the line separator, so this is the flatten — and a `\n`
			// and the space replacing it are both one character, which is why the transaction's own
			// selection survives unmapped.
			insert: inserted.lines > 1 ? inserted.sliceString(0, inserted.length, ' ') : inserted
		});
	});
	return { changes, selection: tr.newSelection };
});
