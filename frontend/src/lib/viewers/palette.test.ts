import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SERIES } from './palette';

const VIEWERS = fileURLToPath(new URL('.', import.meta.url));

/** Every viewer source, minus the module that is allowed to spell these values out. */
function consumers(): { rel: string; src: string }[] {
	return readdirSync(VIEWERS)
		.filter((f) => (f.endsWith('.svelte') || f.endsWith('.ts')) && f !== 'palette.ts')
		.filter((f) => !f.endsWith('.test.ts'))
		.sort()
		.map((f) => ({ rel: f, src: readFileSync(join(VIEWERS, f), 'utf8') }));
}

describe('canvas series palette', () => {
	it('SERIES is one non-empty list of distinct #rrggbb', () => {
		expect(SERIES.length).toBeGreaterThan(0);
		for (const c of SERIES) expect(c).toMatch(/^#[0-9a-fA-F]{6}$/);
		// A repeat would silently give two series the same line — the palette's whole job is to
		// tell them apart. Its LENGTH is deliberately unpinned: with one exported const and its
		// importers, the 8-vs-7 drift is impossible by construction, and pinning it would only
		// fail the next deliberate extension.
		expect(new Set(SERIES).size, 'no two series share a colour').toBe(SERIES.length);
	});
});

describe('canvas chrome', () => {
	/* A canvas context parses `font` as a standalone CSS value, with no element to resolve custom
	   properties against, so `ctx.font = '11px var(--font-mono)'` is SILENTLY discarded and the
	   context keeps the UA's 10px sans-serif. It is the one place in the tree where a `var(--…)`
	   reference does nothing and says nothing — everywhere else a non-CSS `var()` sits in an
	   element-attached `style=`/SVG attribute, where substitution resolves. */
	it('never writes a CSS custom property into ctx.font', () => {
		const offenders = consumers().flatMap(({ rel, src }) =>
			[...src.matchAll(/\.font\s*=\s*([^;\n]+)/g)]
				.filter((m) => m[1].includes('var('))
				.map((m) => `${rel}  ${m[0].trim()}`)
		);
		expect(offenders).toEqual([]);
	});

	/* palette.ts exists because a canvas copy drifted before (the 8-vs-7 series list). Tick labels
	   are the same shape of value: every viewer that draws them reads the font and the ink from
	   there, rather than re-typing a shortened font stack or re-keying the same rgba. */
	it('states the tick-label font and ink once, in palette.ts', () => {
		const dup = consumers()
			.filter(({ src }) => /ui-monospace|rgba\(208, ?208, ?208/.test(src))
			.map(({ rel }) => rel);
		expect(dup).toEqual([]);
	});
});
