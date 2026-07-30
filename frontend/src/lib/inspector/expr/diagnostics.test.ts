import { describe, it, expect } from 'vitest';
import { Text } from '@codemirror/state';
import { expressionDiagnostics } from './diagnostics';

/* D-X3: the backend's `expression_error` becomes an inline diagnostic. The only judgement in the
 * mapping is WHERE, and the rule is that the message decides: a Python SyntaxError carries
 * `line N`, so anchor there; anything else (a KeyError, the nd() ambiguity raise) carries no
 * position at all, so span the whole document. Never guess a narrower range than the message
 * supports — a squiggle under the wrong token is worse than one under all of it. */

const doc = (...lines: string[]) => Text.of(lines);

describe('a positioned SyntaxError anchors to its line', () => {
	it('spans the named line, not the document', () => {
		const d = expressionDiagnostics(
			'SyntaxError: invalid syntax (<goofi-expr>, line 2)',
			doc('a = 1', 'b +', 'c')
		);
		expect(d).toHaveLength(1);
		expect([d[0].from, d[0].to], 'line 2 is offsets 6..9').toEqual([6, 9]);
		expect(d[0].severity).toBe('error');
	});

	it('carries the message through verbatim, so the row and the squiggle agree', () => {
		const msg = 'SyntaxError: unexpected EOF while parsing (<goofi-expr>, line 1)';
		expect(expressionDiagnostics(msg, doc('1 +'))[0].message).toBe(msg);
	});

	// A stale error can outlive the edit that fixed it: the descriptor is pushed asynchronously, so
	// the document can already be shorter than the line the message names.
	it('clamps a line number past the end of the document', () => {
		const d = expressionDiagnostics('SyntaxError: bad (<goofi-expr>, line 9)', doc('1 +'));
		expect([d[0].from, d[0].to]).toEqual([0, 3]);
	});

	// An anchor with no width would render an invisible squiggle, which is the same as reporting
	// nothing — so an empty line falls back to the whole document.
	it('falls back to the document when the named line is empty', () => {
		const d = expressionDiagnostics('SyntaxError: bad (<goofi-expr>, line 2)', doc('nd(', ''));
		expect([d[0].from, d[0].to]).toEqual([0, 4]);
	});
});

describe('a positionless error spans the document', () => {
	// The middle one is `expr.rs:37` verbatim — the raise D-X10's completion detail exists to prevent.
	for (const msg of [
		"KeyError: 'oscillator0'",
		"ValueError: nd('spectrum0') is ambiguous: it has multiple outputs; use nd('spectrum0').slot",
		'TypeError: unsupported operand type(s)'
	]) {
		it(`spans it all: ${msg.split(':')[0]}`, () => {
			const d = expressionDiagnostics(msg, doc("nd('spectrum0') * 2"));
			expect(d).toHaveLength(1);
			expect([d[0].from, d[0].to]).toEqual([0, 19]);
		});
	}

	it('still reports on an empty document', () => {
		const d = expressionDiagnostics('KeyError: x', doc(''));
		expect(d).toHaveLength(1);
		expect([d[0].from, d[0].to]).toEqual([0, 0]);
	});
});

describe('nothing to report clears the diagnostics', () => {
	it('null and blank both yield none', () => {
		expect(expressionDiagnostics(null, doc('1'))).toEqual([]);
		expect(expressionDiagnostics('   ', doc('1'))).toEqual([]);
	});
});
