import { describe, it, expect } from 'vitest';
import { EditorState } from '@codemirror/state';
import { python } from '@codemirror/lang-python';
import { exprContext, entriesFor, type ExprContext } from './complete';
import type { ExprCatalogue } from './catalogue';

/* The completion source, driven exactly the way the running editor drives it — a real Python parse
 * plus a cursor offset — against a FAKE catalogue. That is the whole testability seam (spec §3): the
 * component cannot mount in vitest, but the decision "what does this cursor position mean" is pure
 * and is where every interesting case lives.
 *
 * The cases are chosen to be the ones a REGEX would get wrong, because using the Lezer tree instead
 * of a regex is the design decision under test: a nested call, a binary operand, a second argument,
 * a string that belongs to some other callee, a comment, and an attribute one level past a slot.
 */

const CAT: ExprCatalogue = {
	nodes: [
		{ name: 'oscillator0', slots: [{ name: 'out', dtype: 'array' }] },
		{ name: 'buffer0', slots: [{ name: 'out', dtype: 'array' }] },
		{
			name: 'spectrum0',
			slots: [
				{ name: 'psd', dtype: 'array' },
				{ name: 'freqs', dtype: 'array' }
			]
		}
	],
	globals: [
		{ name: 'default_ufreq', type: 'float' },
		{ name: 'gain', type: 'float' }
	]
};

/** The parsed state the editor would have. `python()` is the same extension the editor loads. */
function at(doc: string, pos = doc.length): ExprContext | null {
	return exprContext(EditorState.create({ doc, extensions: [python()] }), pos);
}

/** What the popup would be offered at this cursor — `[]` when the goofi source stands down. */
function labels(doc: string, pos = doc.length): string[] {
	const c = at(doc, pos);
	return c ? entriesFor(c, CAT).map((o) => o.label) : [];
}

describe('nd() node names', () => {
	it('offers every node name inside the first argument', () => {
		expect(labels("nd('")).toEqual(['oscillator0', 'buffer0', 'spectrum0']);
	});

	it('reads a double-quoted argument the same way', () => {
		expect(labels('nd("')).toEqual(['oscillator0', 'buffer0', 'spectrum0']);
	});

	// Partially typed: the source still offers the whole list and lets CodeMirror filter it, but the
	// replaced range must cover what is already typed or accepting would duplicate it.
	it('replaces the whole partially-typed name, not just the caret', () => {
		const c = at("nd('osc");
		expect(c?.kind).toBe('node');
		expect(c && [c.from, c.to], 'from = after the quote, to = end of the literal').toEqual([4, 7]);
	});

	// A terminated literal has its whole CONTENT replaced — `to` is the closing quote, not the caret,
	// so accepting `buffer0` with the caret two characters in does not leave the tail behind.
	it('replaces the whole literal when the caret sits inside a closed one', () => {
		const c = at("nd('oscillator0')", 5);
		expect(c?.kind).toBe('node');
		expect(c && [c.from, c.to], 'the literal’s content, not [caret, caret]').toEqual([4, 15]);
	});

	// The two cases a regex reads as one: `nd('` nested inside another call, and `nd('` as the right
	// operand of an expression. The tree resolves the enclosing CallExpression either way.
	it('works in a nested call and in a binary operand', () => {
		expect(labels("max(nd('a"), 'nested inside max(…)').toContain('oscillator0');
		expect(labels("nd('osc').out.data.mean() + nd('"), 'the second nd() in a sum').toContain(
			'buffer0'
		);
	});

	// Only the FIRST argument names a node — `nd` takes one, and a regex looking for `nd('…` would
	// happily complete the second.
	it('stands down on a later argument', () => {
		expect(labels("nd('a', '")).toEqual([]);
	});

	it('stands down inside some other callee’s string, and inside a comment', () => {
		expect(labels("open('"), 'a string argument to something else').toEqual([]);
		expect(labels("# nd('"), 'a comment').toEqual([]);
		expect(labels("'plain string'", 6), 'a bare string literal').toEqual([]);
	});
});

// D-X10: the completer knows each node's arity from the store, so it can say what the engine will do
// with the bare form — `_NdProxy._bare()` raises "is ambiguous" for a multi-output node — and steer
// to `.slot` instead of leaving the user to discover the raise at eval time.
describe('output arity is surfaced, not left to the eval error (D-X10)', () => {
	const byLabel = (doc: string) => {
		const c = at(doc)!;
		return new Map(entriesFor(c, CAT).map((o) => [o.label, o]));
	};

	it('a single-output node says it may be used bare', () => {
		const e = byLabel("nd('").get('oscillator0')!;
		expect(e.detail).toMatch(/bare/);
		expect(e.detail).not.toMatch(/ambiguous/);
	});

	it('a multi-output node says the bare form is ambiguous', () => {
		const e = byLabel("nd('").get('spectrum0')!;
		expect(e.detail).toMatch(/2 outputs/);
		expect(e.detail).toMatch(/ambiguous/);
	});

	// "Prefer the `.slot` form" as behaviour rather than prose: accepting finishes the call, and for a
	// multi-output node it leaves the caret on a dot so the slot list is the next thing that opens.
	it('accepting finishes the call, and a multi-output node lands on the dot', () => {
		expect(byLabel("nd('").get('spectrum0')!.apply, 'steers to .slot').toBe("spectrum0').");
		expect(byLabel("nd('").get('oscillator0')!.apply, 'usable bare — just close the call').toBe(
			"oscillator0')"
		);
	});

	// The literal can be closed while the CALL is not (the user typed both quotes and stopped), and
	// the two are tracked separately because only the unwritten characters may be added.
	it('closes only what is not written yet', () => {
		const c = at("nd('a'", 5)!;
		expect(new Map(entriesFor(c, CAT).map((o) => [o.label, o])).get('spectrum0')!.apply).toBe(
			'spectrum0).'
		);
	});

	// Both already written: inserting either would corrupt the call, and there is nowhere to put the
	// steering dot — the `)` is past the caret — so the plain name is the whole answer.
	it('inserts the bare name into an already-finished call', () => {
		const c = at("nd('osc')", 5)!;
		const e = new Map(entriesFor(c, CAT).map((o) => [o.label, o])).get('spectrum0')!;
		expect(e.apply).toBeUndefined();
	});
});

describe('nd(…). output slots', () => {
	it('offers THAT node’s slots after the dot', () => {
		expect(labels("nd('spectrum0').")).toEqual(['psd', 'freqs']);
		expect(labels("nd('oscillator0').")).toEqual(['out']);
	});

	it('carries the slot’s dtype as the detail', () => {
		const c = at("nd('spectrum0').")!;
		expect(entriesFor(c, CAT)[0].detail).toBe('array');
	});

	it('replaces a partially-typed slot name', () => {
		const c = at("nd('spectrum0').p");
		expect(c?.kind).toBe('slot');
		expect(c && [c.from, c.to]).toEqual([16, 17]);
		expect(labels("nd('spectrum0').p")).toEqual(['psd', 'freqs']);
	});

	// One level further out is a slot's OWN attribute (`.data`, `.mean`) — the engine's, not ours.
	it('stands down one level past the slot', () => {
		expect(labels("nd('oscillator0').out.")).toEqual([]);
	});

	it('stands down when the node name is not a literal it can read', () => {
		expect(labels('nd(name).')).toEqual([]);
	});

	it('stands down for an unknown node name', () => {
		expect(labels("nd('nope').")).toEqual([]);
	});
});

describe('globals. and np.', () => {
	it('offers the patch’s global keys after globals.', () => {
		expect(labels('globals.')).toEqual(['default_ufreq', 'gain']);
		expect(labels('globals.ga'), 'partially typed').toEqual(['default_ufreq', 'gain']);
	});

	it('carries the global’s declared type as the detail', () => {
		expect(entriesFor(at('globals.')!, CAT)[0].detail).toBe('float');
	});

	it('offers the curated numpy surface after np.', () => {
		const np = labels('np.');
		expect(np).toContain('mean');
		expect(np).toContain('pi');
		expect(np).toContain('linspace');
	});

	// The one that a regex on `\.$` would get wrong: a member access on something else entirely.
	it('stands down on any other object’s attribute', () => {
		expect(labels('t.')).toEqual([]);
		expect(labels("nd('a').out.data.")).toEqual([]);
	});

	it('reads globals. as the right operand of an expression', () => {
		expect(labels("nd('a').out + globals.")).toEqual(['default_ufreq', 'gain']);
	});
});

/* The four names the evaluator injects (`backend/goofi-python/src/inproc/expr.rs`) are the entire scope, and
 * nothing else in the app tells the user they exist. Offering them at a variable position is both
 * the most useful completion here AND the observable proof of the "trick": our entries and Python's
 * own keywords/builtins are collected for the SAME cursor, so they can only arrive in one popup. */
describe('the injected scope', () => {
	it('offers nd / t / np / globals at a partially-typed name', () => {
		expect(labels('n')).toEqual(['nd', 't', 'np', 'globals']);
	});

	it('replaces the partially-typed name', () => {
		const c = at('nd');
		expect(c?.kind).toBe('scope');
		expect(c && [c.from, c.to]).toEqual([0, 2]);
	});

	// A property name is not a scope position — `x.nd` must not offer the scope.
	it('stands down after a dot and inside a string', () => {
		expect(labels('t.n')).toEqual([]);
		expect(labels("'n'", 2)).toEqual([]);
	});
});
