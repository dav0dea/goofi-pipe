import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/* The style-vocabulary guard.
 *
 * `tokens.test.ts` proves the ladder in `app.css` is well-formed; this proves the CSS that
 * consumes it actually SPEAKS it. Every spacing and type value must resolve through
 * `--space-*` / `--fs-*`, so a density or type-scale change is one edit in `app.css` rather than
 * a codebase-wide hunt. Sub-project M swept the four properties that carry the vocabulary —
 * `font-size`, `gap`, `padding`, `margin` — and this test is what stops them drifting back.
 *
 * `app.css` itself is scanned, not just the components: the north star calls it the CENTRAL
 * styling module, and leaving it outside the guard is how the app's most-inherited control
 * padding (`input, select, textarea`) sat there as a raw `4px 8px`.
 *
 * Deliberately NOT scanned: `width`/`height`/`inset`/`border-*` (structural geometry, not
 * vocabulary — a 22px tab ＋ or a 1px hairline is a shape, not a spacing rung), and breakpoints
 * (`@container (max-width: 240px)` in `ui/Field.svelte` is a layout threshold; it names the
 * width at which a Field's paired controls stop fitting, which no spacing rung can express).
 */

const LIB = fileURLToPath(new URL('..', import.meta.url));
const APP_CSS = fileURLToPath(new URL('../../app.css', import.meta.url));

/* Files whose raw **px** are exempt, each with the reason it stays in raw units. The exemption is
   px-scoped on purpose: every one of these is a fixed-px coordinate system, and none of them has
   any business bypassing a rung that is not a px at all — a `--dur-*` duration, say. */
const ALLOW_FILE: Record<string, string> = {
	// The node canvas is a fixed-px coordinate system (--node-w/-header/-u/-viewer in app.css),
	// mirrored in editor/nodeMetrics.ts + snap.ts on the connector hot path. Its paddings and type
	// sizes are measured against those px; rem-ifying them desyncs the snap grid the moment the
	// responsive html clamp or the coarse-pointer floor moves the rem.
	'editor/GoofiNode.svelte': 'frozen node-canvas geometry (mirrored in nodeMetrics.ts/snap.ts)',
	'editor/BoundaryNode.svelte': 'frozen node-canvas geometry (mirrored in nodeMetrics.ts/snap.ts)',
	'editor/PlacementPreview.svelte': 'frozen node-canvas geometry (mirrors GoofiNode exactly)',
	'viewers/SlotViewer.svelte': 'frozen node-canvas geometry (sits inside a --node-u slot row)',
	// A dev-only diagnostic overlay, deliberately as small and dense as it can be drawn — it is
	// sized against the frames it reports on, not against the app's type scale.
	'editor/PerfHud.svelte': 'dev diagnostic overlay, sized to be unobtrusive, not to the scale'
};

/** Narrower exemptions: `file` omitted means the reason is not one file's, it is the value's. */
const ALLOW_VALUE: { file?: string; prop?: RegExp; value: RegExp; why: string }[] = [
	{
		prop: /^font-size$/,
		value: /^16px$/,
		why: 'iOS force-zooms a focused control below 16px — an absolute device threshold, not a scale rung, so it is the one type literal that is right in any file (R raised it at five more sites whose own class rule out-specified the app.css floor)'
	},
	{
		file: 'panels/ConsolePanel.svelte',
		prop: /^padding$/,
		value: /^2px$/,
		why: 'row vertical padding is mirrored by `PAD = 4` in the script, which estimates row heights in px before the ResizeObserver measures them — a rem would make the estimate wrong at every root size but 14'
	},
	{
		file: 'viewers/StringViewer.svelte',
		value: /em$/,
		why: 'rendered-markdown prose scale: headings/lists size RELATIVE to the viewer root, which is itself on --fs-*; absolute rungs would flatten the document hierarchy'
	},
	{
		file: 'app.css',
		prop: /^font-size$/,
		value: /^(11px|14px|0\.9vw|0\.3vh)$/,
		why: 'the responsive root clamp IS the rem every --fs-*/--space-* rung is measured in — a rung cannot express the size of its own unit'
	},
	{
		file: 'app.css',
		prop: /^transition-duration$/,
		value: /^0\.01$/,
		why: 'the prefers-reduced-motion kill switch: 0.01ms is "off", not a rung — routing it through --dur-fast would restore the motion it exists to remove'
	}
];

/** Geometric invariants that are not spacing at all: nothing, a hairline, full, a pill, a half. */
const INVARIANT = new Set(['0', '1px', '100%', '999px', '50%']);

const PROPS = [
	'font-size',
	'gap',
	'row-gap',
	'column-gap',
	'padding',
	'margin',
	// Motion is vocabulary too: `--dur-fast`/`--dur-slow` are the whole ladder, and every
	// `lib/ui` primitive already speaks it. `animation`/`animation-duration` stay OUT — the three
	// keyframed spinners/flashes F blessed are their own timings, not transitions.
	'transition',
	'transition-duration',
	...['padding', 'margin'].flatMap((p) => [
		`${p}-top`,
		`${p}-right`,
		`${p}-bottom`,
		`${p}-left`,
		`${p}-inline`,
		`${p}-block`,
		`${p}-inline-start`,
		`${p}-inline-end`,
		`${p}-block-start`,
		`${p}-block-end`
	])
].join('|');

function svelteFiles(dir: string): string[] {
	const out: string[] = [];
	for (const entry of readdirSync(dir)) {
		const p = join(dir, entry);
		if (statSync(p).isDirectory()) out.push(...svelteFiles(p));
		else if (entry.endsWith('.svelte')) out.push(p);
	}
	return out.sort();
}

/** Comments stripped, so a commented-out rule never trips the scan. */
function stripComments(css: string): string {
	return css.replace(/\/\*[\s\S]*?\*\//g, ' ');
}

/** Every `<style>` block's CSS. */
function styleCss(src: string): string {
	return stripComments(
		[...src.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map((m) => m[1]).join('\n')
	);
}

/** Everything the guard scans: the central module, then every component `<style>`. */
function sources(): { rel: string; css: string }[] {
	return [
		{ rel: 'app.css', css: stripComments(readFileSync(APP_CSS, 'utf8')) },
		...svelteFiles(LIB).map((p) => ({
			rel: p.slice(LIB.length),
			css: styleCss(readFileSync(p, 'utf8'))
		}))
	];
}

/** The raw numeric literals in one declaration value, with `var(--token)` references removed. */
function literals(value: string): string[] {
	let v = value;
	for (let prev = ''; v !== prev; ) {
		prev = v;
		v = v.replace(/var\(--[a-z0-9-]+\)/g, ' ');
	}
	return v.match(/-?\d*\.?\d+(?:px|rem|em|%|ch|vw|vh|pt)?/g) ?? [];
}

function drift(): string[] {
	const found: string[] = [];
	for (const { rel, css } of sources()) {
		const pxExempt = rel in ALLOW_FILE;
		const allowed = ALLOW_VALUE.filter((a) => a.file === undefined || a.file === rel);
		const decl = new RegExp(`(?:^|[;{}])\\s*(${PROPS})\\s*:\\s*([^;}]+)`, 'g');
		for (const m of css.matchAll(decl)) {
			const raw = literals(m[2]).filter(
				(n) =>
					!INVARIANT.has(n) &&
					!(pxExempt && n.endsWith('px')) &&
					!allowed.some((a) => (a.prop?.test(m[1]) ?? true) && a.value.test(n))
			);
			if (raw.length) found.push(`${rel}  ${m[1]}: ${m[2].trim()}`);
		}
	}
	return found;
}

/** Every `@media` prelude in `css` that asks about the pointer at all. */
function pointerQueries(css: string): string[] {
	return [...css.matchAll(/@media([^{]+)\{/g)]
		.map((m) => m[1].replace(/\s+/g, ' ').trim())
		.filter((q) => /pointer/.test(q));
}

/** Every `:global()` selector in `css` that names a `$lib/ui` primitive's own `.ui-*` class. */
function uiReachIns(css: string): string[] {
	return [...css.matchAll(/:global\([^)]*\.ui-[\w-]+/g)].map((m) => `${m[0]}…)`);
}

describe('style vocabulary', () => {
	it('sizes type and spacing through --fs-* / --space-*, never a raw literal', () => {
		expect(drift()).toEqual([]);
	});

	it('exempts nothing it cannot justify in one line', () => {
		for (const why of Object.values(ALLOW_FILE)) expect(why.length).toBeGreaterThan(0);
		for (const a of ALLOW_VALUE) expect(a.why.length).toBeGreaterThan(0);
	});

	it('scans the central module and the components it claims to (no silently empty sweep)', () => {
		const scanned = sources();
		expect(scanned.length).toBeGreaterThan(40);
		const app = scanned.find((s) => s.rel === 'app.css');
		expect(app, 'app.css is in the sweep, not exempt from it').toBeDefined();
		expect(app!.css).toContain('--space-1');
	});

	/* The sharpest single measure of "is a primitive a primitive": a consumer that reaches into
	   `$lib/ui`'s own `.ui-*` class to restyle it is not composing the primitive, it is patching it.
	   Worse, it patches it from a TIE: a fully-`:global()` selector gets no Svelte scope class, so
	   `.ui-icon-btn.vs-cog` scores exactly what the primitive's own `.ui-icon-btn.svelte-xxx` does —
	   and the two rules land in different built CSS chunks, where the winner is decided by Vite's
	   emitted <link> order rather than by anything in the source. Consumers state their own class
	   under a scoped ancestor (`.row :global(.console-copy-btn)`, which the scope class carries above
	   the tie) or set a documented custom-property hook; both win by construction.

	   The match is POSITION-INDEPENDENT inside the `:global()` argument, because the hazard is: class
	   order in a compound selector is semantically void (`:global(.vs-cog.ui-icon-btn)` is the same
	   selector, the same (0,2,0) and the same cross-chunk tie as the primitive-class-first spelling
	   M-12 deleted), and a leading tag qualifier is an established idiom here (`PanelHeader.svelte`
	   pins `:global(button.content-btn)` precisely to clear such a tie). Anchoring `.ui-` to the head
	   of the argument would have caught only the one spelling that was already gone. */
	it('never restyles a `.ui-*` primitive class from outside $lib/ui', () => {
		const offenders = sources()
			.filter((s) => !s.rel.startsWith('ui/'))
			.flatMap((s) => uiReachIns(s.css).map((hit) => `${s.rel}  ${hit}`));
		expect(offenders).toEqual([]);
	});

	/* The guard above is only worth its line if it fires on the shapes a reach-in actually takes.
	   These five are the realistic ones; the last three are what a head-anchored `.ui-` missed. */
	it('spots a `.ui-*` reach-in wherever it sits inside the `:global()` argument', () => {
		// The primitive class is interpolated so this fixture list is not itself a source hit for
		// the very spelling the guard exists to keep out of the tree.
		const UI = '.ui-icon-btn';
		for (const css of [
			`.vs-anchor :global(${UI}) { color: red; }`,
			`:global(${UI}.vs-cog) { padding: 0; }`,
			`:global(.vs-cog${UI}) { padding: 0; }`, // byte-equivalent CSS to the line above
			`:global(button${UI}) { padding: 0; }`, // the tag-qualifier idiom PanelHeader uses
			'.row :global(.foo .ui-btn) { color: red; }'
		])
			expect(uiReachIns(css), css).toHaveLength(1);

		// And stays quiet on what is NOT a reach-in: a scoped `.ui-*` (Svelte stamps it, no tie), and
		// a `:global()` naming someone else's class — including one that merely starts `.u`.
		for (const css of [
			'.ui-btn { color: red; }',
			':global(.uplot, .uplot *) { color: red; }',
			'.md :global(pre code) { color: red; }',
			'.tab :global(.close) { width: 0; }'
		])
			expect(uiReachIns(css), css).toEqual([]);
	});

	/* D-R7: ONE coarse-pointer idiom, `(hover: none) and (pointer: coarse)`.
	 *
	 * `app.css` states the rationale where it raises `--hit`: the floor is for a real touch device,
	 * so fine-pointer desktop chrome keeps its natural, compact geometry. A rule gated on
	 * `(pointer: coarse)` ALONE also fires on a hover-capable touchscreen laptop — where the box it
	 * grows was never inflated, because app.css's floor did not fire. IconButton shipped both
	 * spellings in one file for one concern (C18): its density floor two-clause, its `::after` hit
	 * rect one-clause, so on such a machine the invisible hit rect grew to 44px around a 20px box and
	 * adjacent header icons' targets overlapped.
	 *
	 * `any-pointer: coarse` is rejected outright: it matches when ANY attached device is coarse, so a
	 * desktop with a drawing tablet plugged in would wear phone chrome. */
	it('gates every pointer-dependent rule on the single coarse idiom (D-R7)', () => {
		const offenders = sources().flatMap((s) =>
			pointerQueries(s.css)
				.filter((q) => q !== '(hover: none) and (pointer: coarse)')
				.map((q) => `${s.rel}  @media ${q}`)
		);
		expect(offenders).toEqual([]);
	});

	// The guard above only earns its line if it fires on the spellings that actually drift in.
	it('spots the one-clause and any-pointer spellings the idiom replaces', () => {
		for (const css of [
			'@media (pointer: coarse) { .a { inset: 0; } }',
			'@media (any-pointer: coarse) { .a { inset: 0; } }',
			'@media (pointer: coarse) and (hover: hover) { .a { inset: 0; } }'
		])
			expect(pointerQueries(css).filter((q) => q !== '(hover: none) and (pointer: coarse)'), css)
				.toHaveLength(1);

		// …and stays quiet on the idiom itself, however it is wrapped, and on a query about
		// something else entirely.
		expect(pointerQueries('@media (hover: none) and (pointer: coarse) {\n\t.a { inset: 0; }\n}')).toEqual(
			['(hover: none) and (pointer: coarse)']
		);
		expect(pointerQueries('@media (prefers-reduced-motion: reduce) { .a { inset: 0; } }')).toEqual([]);
	});

	// F stripped every gradient: the surface ladder carries elevation now, and a gradient reads as
	// a different material sitting on the flat palette. Nothing but this stops one coming back.
	it('paints surfaces flat — no gradients anywhere (C4)', () => {
		const offenders = sources()
			.filter((s) => /gradient\(/.test(s.css))
			.map((s) => s.rel);
		expect(offenders).toEqual([]);
	});
});
