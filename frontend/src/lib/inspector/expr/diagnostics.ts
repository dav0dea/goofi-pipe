/**
 * `expression_error` → an inline CodeMirror diagnostic (D-X3).
 *
 * The descriptor already carries the backend's last compile/eval failure for the param
 * (`api/types.ts`), pushed after each eval, so there is nothing to fetch — only a position to decide.
 *
 * The message decides it. A Python `SyntaxError` renders as `… (<goofi-expr>, line N)`, so the
 * diagnostic anchors to that line. Everything else — a `KeyError` for a node that is gone, the
 * `_NdProxy._bare()` ambiguity raise, a `TypeError` from the arithmetic — carries no position
 * whatsoever, and the honest answer there is the whole document: a squiggle under a guessed token is
 * worse than one under all of it, because it asserts a location the message never claimed.
 *
 * The message goes through verbatim, so the squiggle's hover and the always-visible error row below
 * the editor are the same string (CLAUDE.md forbids information that lives only behind a hover — the
 * squiggle is the position, the row is the text).
 */
import type { Text } from '@codemirror/state';
import type { Diagnostic } from '@codemirror/lint';

/** `line N` as Python spells it in a SyntaxError's `str()`. */
const LINE = /\bline (\d+)/;

export function expressionDiagnostics(error: string | null, doc: Text): Diagnostic[] {
	if (!error || !error.trim()) return [];
	const whole = { from: 0, to: doc.length, severity: 'error' as const, message: error };
	const m = LINE.exec(error);
	if (!m) return [whole];
	// Clamp: a stale error can name a line the document no longer has (the descriptor is pushed
	// asynchronously, so an edit can shorten the doc before the fixed value arrives).
	const line = doc.line(Math.min(Math.max(Number(m[1]), 1), doc.lines));
	// A zero-width anchor draws an invisible squiggle, which reports nothing at all.
	return line.from === line.to ? [whole] : [{ ...whole, from: line.from, to: line.to }];
}
