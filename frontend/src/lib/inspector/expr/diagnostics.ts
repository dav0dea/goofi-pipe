/** A param's `expression_error` as an inline CodeMirror diagnostic, anchored to the line the message names. */
import type { Text } from '@codemirror/state';
import type { Diagnostic } from '@codemirror/lint';

/** `line N` as Python spells it in a SyntaxError's `str()`. */
const LINE = /\bline (\d+)/;

export function expressionDiagnostics(error: string | null, doc: Text): Diagnostic[] {
	if (!error || !error.trim()) return [];
	const whole = { from: 0, to: doc.length, severity: 'error' as const, message: error };
	const m = LINE.exec(error);
	if (!m) return [whole];
	// A stale error can name a line the document no longer has.
	const line = doc.line(Math.min(Math.max(Number(m[1]), 1), doc.lines));
	// A zero-width anchor draws an invisible squiggle.
	return line.from === line.to ? [whole] : [{ ...whole, from: line.from, to: line.to }];
}
