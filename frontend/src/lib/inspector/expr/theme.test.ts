import { describe, it, expect } from 'vitest';
import { EXPR_HIGHLIGHT_SPEC, EXPR_THEME_SPEC } from './theme';

/* The editor's styling has to be CSS-in-JS — CodeMirror generates the class names, and the completion
 * popup is mounted in `document.body` where no Svelte-scoped rule can reach it — which puts it outside
 * `theme/styleDrift.test.ts`'s sweep of `lib/**\/*.svelte` + `app.css`. This is that sweep, over the
 * two spec objects, so the same rules bind here:
 *
 *   · D-X2's "conventional in VALUE, centralized in LOCATION" — no colour literal may appear, because
 *     every ink has a name in app.css;
 *   · the spacing/type vocabulary — the same four properties styleDrift guards, with the same
 *     documented 16px exemption (an absolute device threshold, not a rung);
 *   · D-R7's single coarse-pointer idiom.
 *
 * Scoped to styleDrift's own property family on purpose rather than a stricter one: it exempts
 * `width`/`height` as structural geometry, and that boundary should not move just because a rule
 * crossed a file extension.
 */

/** Every `prop: value` leaf in a (possibly nested) StyleModule spec, with the selector path it sits
 *  under. `tag` is skipped: in a highlight spec it holds Lezer `Tag` objects — a graph, not a
 *  declaration — and walking into it does not terminate. */
function leaves(spec: object, path = ''): { path: string; prop: string; value: string }[] {
	const out: { path: string; prop: string; value: string }[] = [];
	for (const [key, value] of Object.entries(spec)) {
		if (key === 'tag') continue;
		if (value && typeof value === 'object') out.push(...leaves(value, path ? `${path} ${key}` : key));
		else out.push({ path, prop: key, value: String(value) });
	}
	return out;
}

/** Numeric literals in a value, `var()` references removed — styleDrift's `literals()`. */
function literals(value: string): string[] {
	let v = value;
	for (let prev = ''; v !== prev; ) {
		prev = v;
		v = v.replace(/var\(--[a-z0-9-]+\)/g, ' ');
	}
	return v.match(/-?\d*\.?\d+(?:px|rem|em|%|ch|vw|vh|pt)?/g) ?? [];
}

const THEME = leaves(EXPR_THEME_SPEC);
const COARSE = '(hover: none) and (pointer: coarse)';

describe('the editor theme speaks the token ladder', () => {
	it('names no colour literal — every ink has a name in app.css (D-X2)', () => {
		const offenders = [...THEME, ...leaves(EXPR_HIGHLIGHT_SPEC)]
			.filter((l) => /#[0-9a-f]{3,8}\b|\brgba?\(|\bhsla?\(/i.test(l.value))
			.map((l) => `${l.path} { ${l.prop}: ${l.value} }`);
		expect(offenders).toEqual([]);
	});

	it('routes every syntax colour through a --syn-* token', () => {
		for (const l of leaves(EXPR_HIGHLIGHT_SPEC).filter((l) => l.prop === 'color')) {
			expect(l.value, l.path).toMatch(/^var\(--(syn-[a-z]+|danger)\)$/);
		}
	});

	it('sizes type and spacing through --fs-* / --space-*, never a raw literal', () => {
		// The one exemption styleDrift also grants, for the same reason: iOS force-zooms a focused
		// control under 16px, which is a device threshold no scale rung can express.
		const PROPS = /^(fontSize|gap|rowGap|columnGap|padding|margin|transition|transitionDuration)/;
		const offenders = THEME.filter(
			(l) => PROPS.test(l.prop) && literals(l.value).some((n) => n !== '0' && n !== '16px')
		).map((l) => `${l.path} { ${l.prop}: ${l.value} }`);
		expect(offenders).toEqual([]);
	});

	it('gates every pointer-dependent rule on the single coarse idiom (D-R7)', () => {
		const queries = Object.keys(EXPR_THEME_SPEC).filter((k) => k.startsWith('@media'));
		expect(queries.length, 'there is a coarse door to check').toBeGreaterThan(0);
		for (const q of queries) expect(q).toBe(`@media ${COARSE}`);
	});

	/* Each guard above only earns its line if it fires on the defect it claims to catch — the same
	   discipline styleDrift's fixtures follow, and the reason `leaves()` and `literals()` are shared
	   with the assertions rather than restated. */
	it('spots a hex ink and a raw spacing literal', () => {
		const hex = leaves({ '.x': { color: '#ffffff', background: 'rgba(0,0,0,.5)' } });
		expect(hex.filter((l) => /#[0-9a-f]{3,8}\b|\brgba?\(/i.test(l.value))).toHaveLength(2);
		expect(literals('var(--space-4) 7px'), 'a token contributes nothing to read').toEqual(['7px']);
		expect(literals('var(--space-2) var(--space-4)')).toEqual([]);
	});

	// D-X8: the popup rows are tap targets, and a `--hit` floor is what makes them ones.
	it('floors the completion rows at --hit under coarse (D-X8)', () => {
		const coarse = leaves(
			(EXPR_THEME_SPEC as Record<string, object>)[`@media ${COARSE}`],
			`@media ${COARSE}`
		);
		expect(coarse.find((l) => l.prop === 'minHeight')?.value).toBe('var(--hit)');
	});
});
