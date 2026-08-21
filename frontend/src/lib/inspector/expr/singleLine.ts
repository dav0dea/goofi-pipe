/** Keeps the inline expression editor one line: a pasted or programmatic newline flattens to a space. */
import { EditorState, type ChangeSpec } from '@codemirror/state';

export const singleLineExpression = EditorState.transactionFilter.of((tr) => {
	if (!tr.docChanged || tr.newDoc.lines === 1) return tr;
	const changes: ChangeSpec[] = [];
	tr.changes.iterChanges((fromA, toA, _fromB, _toB, inserted) => {
		changes.push({
			from: fromA,
			to: toA,
			// The space replacing the `\n` is the same length, so the transaction's selection still maps.
			insert: inserted.lines > 1 ? inserted.sliceString(0, inserted.length, ' ') : inserted
		});
	});
	return { changes, selection: tr.newSelection };
});
