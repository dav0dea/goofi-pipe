import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ICONS } from './icons';
import { CHROME_ICONS } from 'tatami';

/* The vendored-geometry guard.
 *
 * `icons.ts` is DATA, not logic: Lucide's published path geometry, copied in so the app carries no
 * npm dependency for twenty-odd icons. Data drifts the same way code does — an icon pasted with a
 * baked `stroke="#fff"`, one drawn from memory on a 16-box grid, one vendored and never used —
 * and none of that shows up in a typecheck. This is what reads the bytes.
 *
 * There are TWO tables, because there are two owners: the panel chrome vendors the glyphs it draws
 * and the app vendors its own, and one renderer merges them. Everything below judges the merge,
 * except the two rules that are about ownership — sorted within a table, and in one table only.
 *
 * `Icon.svelte` cannot mount in vitest, so the testable half deliberately lives here: the geometry
 * is a plain string table, and the component is the 12 lines that wrap it in one <svg>.
 */

/** Both trees an icon can be rendered from: the app's, and the panel package's own chrome. */
const ROOTS = [
	fileURLToPath(new URL('../..', import.meta.url)),
	fileURLToPath(new URL('../../../packages/tatami/src', import.meta.url))
];

/** The SVG elements Lucide's geometry is drawn from. Anything else is not a vendored icon. */
const DRAW = /^(path|circle|rect|line|polyline|polygon|ellipse)$/;

/** The merged table, as the renderer resolves it. */
const ALL: Record<string, string> = { ...CHROME_ICONS, ...ICONS };
const names = Object.keys(ALL);

/** Every `.ts`/`.svelte` under `src/`, except this guard and the table it reads. */
function sourceFiles(dir: string): string[] {
	const out: string[] = [];
	for (const entry of readdirSync(dir)) {
		const p = join(dir, entry);
		if (statSync(p).isDirectory()) out.push(...sourceFiles(p));
		else if (/\.(ts|svelte)$/.test(entry) && !/^icons\.(ts|test\.ts)$/.test(entry)) out.push(p);
	}
	return out;
}

const allSource = (): string =>
	ROOTS.flatMap((r) => sourceFiles(r))
		.map((p) => readFileSync(p, 'utf8'))
		.join('\n');

/** Every icon name the source RENDERS. There are three shapes an icon is named in — the
 *  component's plain attribute (`<Icon name="x" />`), an EXPRESSION in that attribute
 *  (`name={copied ? 'check' : 'copy'}`) and a menu/panel record (`icon: 'undo-2'`, whose value is
 *  an expression just as often) — and each spells the icon as a literal, so this is answerable from
 *  the source. Read through those shapes rather than a bare substring: `'x'` and `'copy'` occur all
 *  over a codebase for reasons that have nothing to do with an icon, and a scan they satisfy proves
 *  nothing. The attribute is read off `<Icon` specifically, because `name="…"` is an attribute half
 *  the form controls in the app also carry. */
function rendered(): Set<string> {
	const src = allSource();
	const used = new Set<string>();
	for (const [, tag] of src.matchAll(/<Icon\b([^>]*)>/g)) {
		for (const [, lit] of tag.matchAll(/\bname="([\w-]+)"/g)) used.add(lit);
		for (const [, expr] of tag.matchAll(/\bname=\{([^}]*)\}/g))
			for (const [, lit] of expr.matchAll(/'([\w-]+)'/g)) used.add(lit);
	}
	// A record field is bounded by its comma; every quoted literal in it is an icon name.
	for (const [, inRecord] of src.matchAll(/\bicon:\s*([^,\n]+)/g))
		for (const [, lit] of inRecord.matchAll(/'([\w-]+)'/g)) used.add(lit);
	return used;
}

describe('vendored Lucide geometry', () => {
	it('gives every name real drawing geometry', () => {
		expect(names.length, 'the table is not empty').toBeGreaterThan(0);
		const empty = names.filter((n) => !/<(\w+)[^>]*\/>/.test(ALL[n]));
		expect(empty, 'every icon resolves to at least one drawing element').toEqual([]);
	});

	it('draws with Lucide elements only', () => {
		const offenders: string[] = [];
		for (const n of names) {
			// The whole string must be self-closing drawing tags back to back — nothing else can be in
			// there, which is also what makes the `{@html}` in `Icon.svelte` safe by construction.
			const rest = ALL[n].replace(/<(\w+)((?:\s+[\w-]+="[^"]*")*)\s*\/>/g, (_, tag: string) =>
				DRAW.test(tag) ? '' : `<${tag}>`
			);
			if (rest !== '') offenders.push(`${n}: ${rest}`);
		}
		expect(offenders).toEqual([]);
	});

	/* The whole point of vendoring the geometry rather than a finished <svg>: paint is
	   `Icon.svelte`'s (stroke: currentColor), so an icon inherits the colour of the control it sits
	   in and can never introduce one of its own — which is the defect this icon system replaced. */
	it('bakes in no paint of its own — colour comes from the control', () => {
		const offenders = names.filter((n) => /\b(fill|stroke|style|class|color)=/.test(ALL[n]));
		expect(offenders, 'no icon carries its own fill/stroke/style').toEqual([]);
	});

	/* Lucide's grid is 24×24. A coordinate outside it is the tell of an icon drawn from memory at
	   another scale, which reads as a different weight beside the real ones. */
	it('stays on the 24-box Lucide grid', () => {
		const offenders: string[] = [];
		for (const n of names)
			for (const [, num] of ALL[n].matchAll(/(-?\d*\.\d+|-?\d+)/g))
				if (Math.abs(parseFloat(num)) > 24) offenders.push(`${n}: ${num}`);
		expect(offenders).toEqual([]);
	});

	it('names icons the way Lucide does — kebab-case, unique, sorted', () => {
		expect(names.filter((n) => !/^[a-z0-9]+(-[a-z0-9]+)*$/.test(n))).toEqual([]);
		for (const [who, table] of [
			['the app', ICONS],
			['the chrome', CHROME_ICONS]
		] as const) {
			const own = Object.keys(table);
			expect([...own].sort(), `${who}'s table is sorted`).toEqual(own);
		}
	});

	/* One glyph, one owner. A name in both tables is the same geometry vendored twice, and the
	   merge hides it: the app's copy wins, the chrome's copy is never drawn, and the day one of
	   them is corrected the other is not. */
	it('vendors each glyph in exactly one table', () => {
		const both = Object.keys(ICONS).filter((n) => n in CHROME_ICONS);
		expect(both, 'the app and the chrome each own their own glyphs').toEqual([]);
	});

	/* A vendored icon nothing renders is dead weight in the bundle, and the reason a hand-vendored
	   pack rots: the pack grows, the UI does not. There are three shapes an icon can be named in —
	   the component's plain attribute (`name="x"`), an EXPRESSION in that attribute
	   (`name={copied ? 'check' : 'copy'}`) and a menu/panel record (`icon: 'undo-2'`, whose value is
	   an expression just as often) — and each spells the icon as a literal, so this is answerable
	   from the source. Read through those shapes rather than a bare substring: `'x'` and `'copy'`
	   occur all over a codebase for reasons that have nothing to do with an icon, and a scan they
	   satisfy proves nothing. */
	it('vendors nothing the app does not render', () => {
		expect(names.filter((n) => !rendered().has(n))).toEqual([]);
	});

	/* The other direction, and the one that used to be a typecheck: `Icon` takes a plain string
	   now, because half the names reach it from a consumer's own table and a union cannot enumerate
	   those. A typo therefore draws NOTHING — silently, in the one place a glyph was the whole
	   content of the control. This reads the same source the guard above does and asks whether the
	   merge can answer it, which the union never covered for a record field anyway. */
	it('renders nothing it has no geometry for', () => {
		expect([...rendered()].filter((n) => !(n in ALL))).toEqual([]);
	});
});
