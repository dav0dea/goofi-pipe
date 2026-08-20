import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ICONS } from './icons';
import { CHROME_ICONS } from 'panelty';

/* The vendored-geometry guard.
 *
 * `icons.ts` is DATA, not logic: Lucide's published path geometry, copied in so the app carries no
 * npm dependency for twenty-odd icons. Data drifts the same way code does — an icon pasted with a
 * baked `stroke="#fff"`, one drawn from memory on a 16-box grid, one vendored and never used —
 * and none of that shows up in a typecheck. This is what reads the bytes.
 *
 * There are TWO tables, because there are two owners: `panelty` vendors the glyphs its chrome
 * draws, this app vendors its own, and one renderer merges them at runtime. What is judged here is
 * THIS app's table — the package's own repo holds its half to the same rules — plus the two
 * questions that are about the pair: that no glyph is vendored twice, and that every name this
 * app's source renders resolves in the merge.
 *
 * `Icon.svelte` cannot mount in vitest, so the testable half deliberately lives here: the geometry
 * is a plain string table, and the component is the 12 lines that wrap it in one <svg>.
 */

const SRC = fileURLToPath(new URL('../..', import.meta.url));

/** The SVG elements Lucide's geometry is drawn from. Anything else is not a vendored icon. */
const DRAW = /^(path|circle|rect|line|polyline|polygon|ellipse)$/;

/** The merged table, as the renderer resolves it — only the resolution question reads this. */
const ALL: Record<string, string> = { ...CHROME_ICONS, ...ICONS };
/** This app's own geometry: what the shape rules below are about. */
const names = Object.keys(ICONS);

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
	sourceFiles(SRC)
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
		const empty = names.filter((n) => !/<(\w+)[^>]*\/>/.test(ICONS[n as keyof typeof ICONS]));
		expect(empty, 'every icon resolves to at least one drawing element').toEqual([]);
	});

	it('draws with Lucide elements only', () => {
		const offenders: string[] = [];
		for (const n of names) {
			// The whole string must be self-closing drawing tags back to back — nothing else can be in
			// there, which is also what makes the `{@html}` in `Icon.svelte` safe by construction.
			const rest = ICONS[n as keyof typeof ICONS].replace(/<(\w+)((?:\s+[\w-]+="[^"]*")*)\s*\/>/g, (_, tag: string) =>
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
		const offenders = names.filter((n) => /\b(fill|stroke|style|class|color)=/.test(ICONS[n as keyof typeof ICONS]));
		expect(offenders, 'no icon carries its own fill/stroke/style').toEqual([]);
	});

	/* Lucide's grid is 24×24. A coordinate outside it is the tell of an icon drawn from memory at
	   another scale, which reads as a different weight beside the real ones. */
	it('stays on the 24-box Lucide grid', () => {
		const offenders: string[] = [];
		for (const n of names)
			for (const [, num] of ICONS[n as keyof typeof ICONS].matchAll(/(-?\d*\.\d+|-?\d+)/g))
				if (Math.abs(parseFloat(num)) > 24) offenders.push(`${n}: ${num}`);
		expect(offenders).toEqual([]);
	});

	it('names icons the way Lucide does — kebab-case, unique, sorted', () => {
		expect(names.filter((n) => !/^[a-z0-9]+(-[a-z0-9]+)*$/.test(n))).toEqual([]);
		expect([...names].sort(), 'the table is sorted').toEqual(names);
	});

	/* One glyph, one owner — across a package boundary, which is where this stops being obvious. A
	   name in both tables is the same geometry vendored twice, and the merge hides it: this app's
	   copy wins, the chrome's is never drawn, and the day one of them is corrected the other is
	   not. It fails HERE when the package adds a glyph this app already had. */
	it('vendors each glyph in exactly one table', () => {
		const both = names.filter((n) => n in CHROME_ICONS);
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
