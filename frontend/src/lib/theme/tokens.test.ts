import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { contrastRatio, relativeLuminance } from './contrast';

const css = readFileSync(fileURLToPath(new URL('../../app.css', import.meta.url)), 'utf8');

/** Resolve a `--token` to its #rrggbb value from the :root block (one level of var() alias). */
function token(name: string): string {
	const re = new RegExp(`--${name}\\s*:\\s*([^;]+);`);
	const m = re.exec(css);
	if (!m) throw new Error(`token --${name} not found`);
	const v = m[1].trim();
	const alias = /^var\(--([a-z0-9-]+)\)$/i.exec(v);
	return alias ? token(alias[1]) : v;
}

describe('elevation ladder', () => {
	const steps = ['bg', 'surface-1', 'surface-2', 'surface-3', 'surface-4'].map(token);
	it('is monotonically lighter', () => {
		for (let i = 1; i < steps.length; i++) {
			expect(relativeLuminance(steps[i])).toBeGreaterThan(relativeLuminance(steps[i - 1]));
		}
	});
	it('steps each ≥ 1.08:1 (a real visible increment, unlike the old flat ladder)', () => {
		for (let i = 1; i < steps.length; i++) {
			expect(contrastRatio(steps[i], steps[i - 1])).toBeGreaterThanOrEqual(1.08);
		}
	});
});

describe('text tiers on --bg', () => {
	const bg = token('bg');
	it('--text ≥ 7:1', () => expect(contrastRatio(token('text'), bg)).toBeGreaterThanOrEqual(7));
	it('--text-dim ≥ 4.5:1', () => expect(contrastRatio(token('text-dim'), bg)).toBeGreaterThanOrEqual(4.5));
	it('--text-muted ≥ 3:1 on --bg and on --surface-2', () => {
		expect(contrastRatio(token('text-muted'), bg)).toBeGreaterThanOrEqual(3);
		expect(contrastRatio(token('text-muted'), token('surface-2'))).toBeGreaterThanOrEqual(3);
	});
	// F widened the ladder to --surface-3/-4. --surface-3 is the lightest surface TEXT sits on:
	// --text-muted clears 3:1 there (~3.08:1). --surface-4 is the ladder's top rung and the palette's
	// worst case — no rule paints it since Chip's neutral tone became a ghost, and it is kept
	// deliberately as the reference these assertions measure against (see app.css). It is a
	// control-FILL surface, NOT a text background, so --text-muted (2.53:1 there) is deliberately
	// NOT asserted on it, only the two stronger tiers a label overlapping such a control must use.
	// This encodes the ladder contract so phase M can't silently regress it.
	it('stronger tiers stay legible on the widened surfaces', () => {
		expect(contrastRatio(token('text'), token('surface-4'))).toBeGreaterThanOrEqual(4.5);
		expect(contrastRatio(token('text-dim'), token('surface-4'))).toBeGreaterThanOrEqual(3);
		expect(contrastRatio(token('text-muted'), token('surface-3'))).toBeGreaterThanOrEqual(3);
	});
});

describe('borders read as a light hairline, never an inverted groove', () => {
	it('--border and --border-strong are lighter than --surface-4', () => {
		const s4 = relativeLuminance(token('surface-4'));
		expect(relativeLuminance(token('border'))).toBeGreaterThan(s4);
		expect(relativeLuminance(token('border-strong'))).toBeGreaterThan(s4);
	});
});

describe('semantic + dtype colours stay legible on the lightest surface', () => {
	const s4 = token('surface-4');
	for (const t of ['success', 'warning', 'danger', 'dtype-array', 'dtype-string', 'dtype-table']) {
		it(`--${t} ≥ 3:1 on --surface-4`, () => expect(contrastRatio(token(t), s4)).toBeGreaterThanOrEqual(3));
	}
});

describe('ink on a filled semantic surface is one answer, and AA-legible', () => {
	// `--on-accent` / `--on-danger` are the ONLY answer to "what colour is text on a filled accent
	// or danger surface" — the `$lib/ui` primitives read them too. `--on-danger` was #ffffff, which
	// on --danger (#f06080) is 3.14:1: a WCAG AA failure shipping on the Toast and the proto banner,
	// both at --fs-small, while <Button variant="danger"> on the identical fill rendered near-black.
	for (const t of ['accent', 'danger']) {
		it(`--on-${t} clears AA (4.5:1) on --${t}`, () =>
			expect(contrastRatio(token(`on-${t}`), token(t))).toBeGreaterThanOrEqual(4.5));
	}
});

describe('scale tokens exist and are ordered', () => {
	function px(name: string): number {
		const m = new RegExp(`--${name}\\s*:\\s*([0-9.]+)rem`).exec(css);
		if (!m) throw new Error(`--${name} not a rem token`);
		return parseFloat(m[1]);
	}
	it('spacing is an 8-step ascending rem scale', () => {
		const steps = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => px(`space-${n}`));
		for (let i = 1; i < steps.length; i++) expect(steps[i]).toBeGreaterThan(steps[i - 1]);
	});
	it('type is a 5-step ascending rem scale', () => {
		const steps = ['micro', 'small', 'body', 'strong', 'title'].map((r) => px(`fs-${r}`));
		for (let i = 1; i < steps.length; i++) expect(steps[i]).toBeGreaterThan(steps[i - 1]);
	});
	it('--radius-lg is deleted', () => expect(css).not.toMatch(/--radius-lg\s*:/));
	it('--font-sans is deleted (collapsed to --font-mono)', () => expect(css).not.toMatch(/--font-sans\s*:/));
});

describe('the success green is differentiated from the accent (D-M1)', () => {
	// --success and --accent were byte-identical #50d0a0, so a selected+healthy node's accent ring
	// and its success health dot read as one colour. M nudges --success to a distinct green.
	it('--success is not the same colour as --accent', () => {
		expect(token('success')).not.toBe(token('accent'));
	});
	it('--success and --accent are perceptually separated (>1.1:1)', () => {
		expect(contrastRatio(token('success'), token('accent'))).toBeGreaterThan(1.1);
	});
});

describe('category colour system is gone', () => {
	it('no --cat-* tokens remain in app.css', () => {
		expect(css).not.toMatch(/--cat-[a-z]+\s*:/);
	});
});

/* The syntax-highlighting family (X, D-X2). Two asks that look opposed and are not: the COLOURS are
   conventional at full strength (code is read by colour, so the app's reduced-saliency brief stops at
   the editor's edge), and their LOCATION is still app.css — a CodeMirror theme object carrying hex
   would be the one palette in the app outside the SSOT.
   The expression editor sits on --surface-1 (app.css's `input` background, which the editor host
   keeps), so that is the background every --syn-* is measured against. The floor is AA 4.5:1, not the
   3:1 the semantic/dtype inks get: those are marks and washes, this is body text being read. */
describe('syntax colours clear the AA floor on the editor background (D-X2)', () => {
	const SYN = [
		'keyword',
		'name',
		'function',
		'literal',
		'type',
		'operator',
		'comment',
		'string',
		'punct'
	];
	const editorBg = token('surface-1');
	for (const t of SYN) {
		it(`--syn-${t} ≥ 4.5:1 on --surface-1`, () =>
			expect(contrastRatio(token(`syn-${t}`), editorBg)).toBeGreaterThanOrEqual(4.5));
	}
	// A conventional scheme's whole value is that the roles are TOLD APART at a glance; two rungs
	// resolving to one hex would silently collapse two of them.
	it('names nine distinct inks', () => {
		expect(new Set(SYN.map((t) => token(`syn-${t}`))).size).toBe(SYN.length);
	});
});
