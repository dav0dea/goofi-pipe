import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/* The read-the-CSS-from-disk guard.
 *
 * Two families live here because they ask the same question of the same bytes: what does the
 * stylesheet SAY, before any of it is mounted. `describe('style vocabulary')` guards the token
 * ladder; `describe('coarse-pointer doors')` guards R's success criteria §5.2 and §5.4 — the two
 * that are enumerable over the source rather than reachable by a hand-written case.
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

/** The same components, whole — markup and script included. The door guards need to see which
 *  element a class is actually ON, and which files bind a right-click. */
function components(): { rel: string; src: string }[] {
	return svelteFiles(LIB).map((p) => ({ rel: p.slice(LIB.length), src: readFileSync(p, 'utf8') }));
}

/** The one coarse-pointer idiom (D-R7), spelled once so every guard below asks it the same way. */
const COARSE = '(hover: none) and (pointer: coarse)';

/**
 * Every rule in `css`, with the `@media` prelude it sits under (`null` at top level).
 *
 * Deliberately shallow about `@container`: its prelude falls away and its rules read as
 * top-level. That is the conservative answer for both door guards — a container-scoped rule
 * still out-specifies `app.css`'s floor, and still is not a coarse door.
 */
function rules(css: string): { sel: string; body: string; media: string | null }[] {
	const out: { sel: string; body: string; media: string | null }[] = [];
	const decl = /([^{}]+)\{([^{}]*)\}/g;
	for (let i = 0; i < css.length; ) {
		const at = css.indexOf('@media', i);
		const head = css.slice(i, at === -1 ? css.length : at);
		for (const m of head.matchAll(decl))
			out.push({ sel: m[1].replace(/\s+/g, ' ').trim(), body: m[2], media: null });
		if (at === -1) break;
		const open = css.indexOf('{', at);
		if (open === -1) break;
		let depth = 1;
		let j = open + 1;
		for (; j < css.length && depth > 0; j++) {
			if (css[j] === '{') depth++;
			else if (css[j] === '}') depth--;
		}
		const prelude = css.slice(at + '@media'.length, open).replace(/\s+/g, ' ').trim();
		// Recurse, so a nested query is attributed to ITSELF and cannot inherit a coarse prelude
		// it does not have.
		for (const r of rules(css.slice(open + 1, j - 1)))
			out.push({ ...r, media: r.media ?? prelude });
		i = j;
	}
	return out;
}

/** The last compound of each comma-separated branch, `:global()` unwrapped — the part of a
 *  selector that names what the rule actually paints. */
function keyCompounds(sel: string): string[] {
	return sel.split(',').map(
		(one) =>
			one
				.replace(/:global\(([^)]*)\)/g, '$1')
				.trim()
				.split(/\s*[>+~]\s*|\s+/)
				.filter(Boolean)
				.at(-1) ?? ''
	);
}

const classesIn = (compound: string): string[] =>
	[...compound.matchAll(/\.([\w-]+)/g)].map((m) => m[1]);

/** Does this rule body actually leave its target VISIBLE? The value half of `hoverReveals`'
 *  property test: a coarse rule that only recolours or resizes NAMES a class without resting it
 *  open, and counting that as a door made the scan satisfiable by any rule that mentions the
 *  target. `opacity: var(--token)` reads as a reveal — a token cannot be proven to be zero. */
function revealsAtRest(body: string): boolean {
	for (const [, prop, value] of body.matchAll(/(?:^|;)\s*(opacity|visibility)\s*:\s*([^;]+)/g)) {
		if (prop === 'opacity' ? parseFloat(value) !== 0 : !/hidden|collapse/.test(value)) return true;
	}
	return false;
}

/** Every class named by a coarse-gated rule that rests it open — i.e. everything the file gives a
 *  coarse resting form to. */
function coarseTargets(css: string): Set<string> {
	const out = new Set<string>();
	for (const r of rules(css))
		if (r.media === COARSE && revealsAtRest(r.body))
			for (const c of keyCompounds(r.sel).flatMap(classesIn)) out.add(c);
	return out;
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
				.filter((q) => q !== COARSE)
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
			expect(pointerQueries(css).filter((q) => q !== COARSE), css)
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

	/* `--warning` is this app's FAULT ink: `TopBar.svelte` paints it on a lost control connection,
	   three lines above the perf HUD it hosts. The HUD painted the same amber whenever the drop
	   counter moved — but a "drop" is latest-wins coalescing, which `frames.ts` states outright is
	   free ("no viewer is starved and no queue grows") and which the HUD's own tooltip calls
	   "latest-wins coalesced frames". Healthy 2-3 node patches therefore sat amber in every screen
	   capture. One status cluster cannot carry two contradictory meanings for one token, and there
	   is no threshold that turns a designed behaviour into a fault — so the counters read in the
	   neutral ink, and this is the only thing keeping them there. */
	it('never paints a fault ink on the frame counters (coalescing is by design, not a fault)', () => {
		const hud = sources().find((s) => s.rel === 'editor/PerfHud.svelte');
		expect(hud, 'the HUD is still where this guard looks for it').toBeDefined();
		expect(hud!.css.match(/--(?:warning|danger)\b/g) ?? []).toEqual([]);
	});
});

/* ------------------------------------------------------------------------------------------- */

/**
 * R's success criterion §5.2 — *no interaction exists solely behind hover or right-click* — as a
 * TEST rather than a claim. Both halves are enumerable over the source, which is worth more than
 * any number of hand-written cases: a case proves the six doors R built still open; a scan proves
 * a seventh cannot appear without one.
 *
 * `touch-hover-doors.spec.ts` is the other half of this pair and stays: it proves the doors named
 * here actually open under real touch. This file proves the inventory is closed.
 */

/** A hover-gated reveal: a rule that turns `opacity`/`visibility` on from behind `:hover`.
 *
 *  Only those two properties count. A background or colour change on hover is *feedback* — every
 *  button in the tree has one, and a device with no hover loses nothing by not seeing it. What a
 *  coarse pointer cannot survive is a control or a label that is not THERE until hovered. */
function hoverReveals(css: string): { sel: string; targets: string[] }[] {
	return rules(css)
		.filter(
			(r) =>
				r.media !== COARSE &&
				/:hover/.test(r.sel) &&
				/(?:^|;)\s*(opacity|visibility)\s*:/.test(r.body)
		)
		.map((r) => ({ sel: r.sel, targets: keyCompounds(r.sel).flatMap(classesIn) }));
}

/** Does this file bind a right-click? Both spellings count: the Svelte attribute, and
 *  `addEventListener('contextmenu', …)` — which is not a stylistic choice here but the only way
 *  to reach an element the app does not author, such as the SvelteFlow pane. */
function bindsRightClick(src: string): boolean {
	return /\boncontextmenu\b|addEventListener\(\s*['"]contextmenu/.test(src);
}

/** Reveals that deliberately have no coarse resting form, each with the reason it is a decision
 *  and not an oversight. `target` is the class the reveal lands on. */
const HOVER_ONLY_OK: { file: string; target: string; why: string }[] = [
	{
		file: 'workspace/Panel.svelte',
		target: 'corner',
		why: 'split/join by corner is a fine-pointer power-user gesture R declined to grow — a --hit grip drops a 44px split-drag triangle over every panel corner, including the one the editor’s zoom cluster sits in — and D-R5 gives touch the header long-press menu as its door instead'
	},
	{
		file: 'panels/NodeEditorPanel.svelte',
		target: 'inspector-toggle',
		why: 'it RESTS at opacity .5 — visible and tappable with no hover; the hover only brightens an affordance that is already there, and R-Task 8 removes it outright while the pane it opens is up'
	}
];

/** Classes a file puts on a real `<input>`/`<select>`/`<textarea>`, from the markup — the only
 *  place that link is written down. A rule naming one of these out-specifies `app.css`'s
 *  `input, select, textarea` floor, which scores (0,0,1) and loses to anything. */
function controlClasses(src: string): Set<string> {
	const markup = src.replace(/<script[\s\S]*?<\/script>/g, ' ').replace(/<style[\s\S]*?<\/style>/g, ' ');
	const out = new Set<string>();
	for (const [, , attrs] of markup.matchAll(/<(input|select|textarea)\b([^>]*)>/g)) {
		for (const [, list] of attrs.matchAll(/\bclass=["']([^"']*)["']/g))
			for (const token of list.split(/\s+/)) if (token && !token.includes('{')) out.add(token);
		for (const [, name] of attrs.matchAll(/\bclass:([\w-]+)/g)) out.add(name);
	}
	return out;
}

/** What a `font-size` rule lands on, in the vocabulary the floor is written in: the type
 *  selectors it names plus whichever of this file's control classes it names. Empty ⇒ the rule
 *  does not paint a form control and cannot defeat the floor. */
function fontTargets(sel: string, classes: Set<string>): string[] {
	const out = new Set<string>();
	for (const key of keyCompounds(sel)) {
		for (const type of ['input', 'select', 'textarea'])
			if (new RegExp(`(^|[^-\\w])${type}\\b`).test(key)) out.add(type);
		for (const c of classesIn(key)) if (classes.has(c)) out.add(c);
	}
	return [...out];
}

/** The largest px literal in a declaration value, or 0 when it names none. `16px` and
 *  `var(--viewer-kind-fs, 16px)` both answer 16; a bare `var(--fs-small)` answers 0, which is the
 *  honest answer — a rem rung cannot be proven to clear an absolute device threshold. */
function maxPx(value: string): number {
	return [...value.matchAll(/(\d*\.?\d+)px/g)].reduce((a, m) => Math.max(a, parseFloat(m[1])), 0);
}

/** iOS force-zooms the page when a control under this size takes focus. */
const FOCUS_ZOOM_FLOOR = 16;

describe('coarse-pointer doors', () => {
	it('gives every hover-revealed control a coarse resting form (§5.2)', () => {
		const offenders: string[] = [];
		for (const { rel, css } of sources()) {
			const answered = coarseTargets(css);
			for (const { sel, targets } of hoverReveals(css)) {
				if (targets.some((t) => answered.has(t))) continue;
				if (targets.some((t) => HOVER_ONLY_OK.some((e) => e.file === rel && e.target === t)))
					continue;
				offenders.push(`${rel}  ${sel}`);
			}
		}
		expect(offenders).toEqual([]);
	});

	/* PanelHeader's split/join menu is the tree's only right-click, and R-Task 7 gave it a long
	   press. This keeps that a rule rather than a coincidence. */
	it('never leaves a right-click as the only door (§5.2)', () => {
		const offenders = components()
			.filter((c) => bindsRightClick(c.src) && !/createLongPress/.test(c.src))
			.map((c) => c.rel);
		expect(offenders).toEqual([]);
	});

	/**
	 * §5.4, pinned globally rather than site by site. `app.css` floors `input, select, textarea` at
	 * 16px under the coarse idiom — at (0,0,1), which ANY product class rule out-specifies, which
	 * is how this failed at seven sites. So: a file that sets `font-size` on one of its own form
	 * controls must restate a ≥16px size for that control under coarse. `touch-hit-floor.spec.ts`
	 * measures four of these in the running app; this is what stops an eighth appearing.
	 */
	it('never lets a control’s own rule drop it under the focus-zoom floor (§5.4)', () => {
		const offenders: string[] = [];
		for (const { rel, src } of [
			{ rel: 'app.css', src: readFileSync(APP_CSS, 'utf8') },
			...components()
		]) {
			const classes = controlClasses(src);
			const css = rel === 'app.css' ? stripComments(src) : styleCss(src);
			const floored = new Set<string>();
			const declared = new Map<string, string>();
			for (const r of rules(css)) {
				for (const [, value] of r.body.matchAll(/(?:^|;)\s*font-size\s*:\s*([^;]+)/g)) {
					for (const t of fontTargets(r.sel, classes)) {
						if (r.media === COARSE && maxPx(value) >= FOCUS_ZOOM_FLOOR) floored.add(t);
						else if (r.media !== COARSE) declared.set(t, `${r.sel} { font-size: ${value.trim()} }`);
					}
				}
			}
			// De-duplicated by rule: `input.name` names two targets and is still one defect.
			for (const [t, where] of declared)
				if (!floored.has(t) && !offenders.includes(`${rel}  ${where}`))
					offenders.push(`${rel}  ${where}`);
		}
		expect(offenders).toEqual([]);
	});

	it('exempts nothing it cannot justify in one line', () => {
		for (const e of HOVER_ONLY_OK) expect(e.why.length, `${e.file} .${e.target}`).toBeGreaterThan(0);
	});

	/* Both guards only earn their lines if they fire on the defect they claim to catch — so each
	   is shown red against a fixture of exactly that shape, and quiet against the fixed one. */
	it('spots a reveal whose coarse door is missing', () => {
		const reveal = '.row:hover :global(.copy) { opacity: 1; }';
		const door = `@media ${COARSE} { .row :global(.copy) { opacity: 1; } }`;
		expect(hoverReveals(reveal).map((r) => r.targets)).toEqual([['copy']]);
		expect(coarseTargets(reveal).has('copy'), 'undoored').toBe(false);
		expect(coarseTargets(reveal + door).has('copy'), 'doored').toBe(true);
		// A hover that only recolours is feedback, not a reveal, and needs no door.
		expect(hoverReveals('.btn:hover { background: red; }')).toEqual([]);
		// …and the mirror image: a coarse rule that only recolours or resizes NAMES the class
		// without resting it open, so it is not a door either. Without this the scan passes on
		// any coarse rule that happens to mention the target.
		const cosmetic = `@media ${COARSE} { .row :global(.copy) { color: red; font-size: 16px; } }`;
		expect(coarseTargets(reveal + cosmetic).has('copy'), 'cosmetic-only').toBe(false);
		expect(
			coarseTargets(`@media ${COARSE} { .copy { visibility: visible; } }`).has('copy'),
			'visibility is the other half of the pair'
		).toBe(true);
	});

	/* The right-click guard reads SOURCE, not CSS, so its fixture is a source snippet: both
	   spellings of "this file binds a right-click" have to register, or a handler wired the way
	   the SvelteFlow pane must be wired is invisible to it. */
	it('sees a right-click bound imperatively, not just as an attribute', () => {
		expect(bindsRightClick('<div oncontextmenu={menu}></div>'), 'attribute').toBe(true);
		expect(bindsRightClick("el.addEventListener('contextmenu', menu);"), 'imperative').toBe(true);
		expect(bindsRightClick("el.addEventListener('pointerdown', down);"), 'unrelated').toBe(false);
	});

	it('spots a control that keeps its small type under coarse', () => {
		const markup = '<input class="name" />';
		const small = '<style>input.name { font-size: var(--fs-small); }</style>';
		const raised = `<style>input.name { font-size: var(--fs-small); } @media ${COARSE} { input.name { font-size: 16px; } }</style>`;
		const scan = (src: string): string[] => {
			const classes = controlClasses(src);
			const floored = new Set<string>();
			const declared = new Set<string>();
			for (const r of rules(styleCss(src)))
				for (const [, v] of r.body.matchAll(/(?:^|;)\s*font-size\s*:\s*([^;]+)/g))
					for (const t of fontTargets(r.sel, classes))
						if (r.media === COARSE && maxPx(v) >= FOCUS_ZOOM_FLOOR) floored.add(t);
						else if (r.media !== COARSE) declared.add(t);
			return [...declared].filter((t) => !floored.has(t));
		};
		expect(scan(markup + small), 'unfloored').toEqual(['input', 'name']);
		expect(scan(markup + raised), 'floored').toEqual([]);
		// A rem rung under coarse is NOT a floor: it cannot be proven to clear an absolute threshold.
		expect(
			scan(markup + raised.replace('font-size: 16px', 'font-size: var(--fs-body)')),
			'a token is not a floor'
		).toEqual(['input', 'name']);
	});
});
