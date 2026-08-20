import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

/* The read-the-source guard for ONE mistake, because it is invisible in every other way.
 *
 * Every app-wide store here is a LAZY singleton: `notify()`/`flash()`/… construct the instance on
 * first call. Calling one for the first time from inside a `$derived` therefore creates its `$state`
 * signals inside that derived's tracking scope, and writes from outside never re-run it. The store
 * is set, the surface is dead, and nothing fails: `notify.test.ts` passes against the store while
 * the toast it feeds shows nothing (which is exactly what shipped — a failed save AND the undo/redo
 * replay failure were both silent). Whether a given site is live depends only on which component
 * happens to mount first, so it is not reviewable case by case.
 *
 * The rule is therefore positional and mechanical: reach the singleton at component-script top
 * level (`const n = notify();`), the way every other consumer already does, and read fields off
 * that in the derived. `$effect` is deliberately NOT scanned — an effect is a reaction that re-runs
 * on a write like any other, so constructing there is merely unidiomatic, not broken.
 */

/** Both source trees the app is built from: its own, and the panel package it composes — whose
 *  `workspace()` is one of the accessors this guards, and whose components call it. */
const ROOTS = [
	fileURLToPath(new URL('../..', import.meta.url)),
	fileURLToPath(new URL('../../../packages/tatami/src', import.meta.url))
];

/** Every `x.svelte` / `x.svelte.ts` / `x.ts` under a tree, tests excluded. */
function sources(dir: string, out: string[] = []): string[] {
	for (const entry of readdirSync(dir)) {
		const path = join(dir, entry);
		if (statSync(path).isDirectory()) sources(path, out);
		else if (/\.(svelte|ts)$/.test(path) && !path.endsWith('.test.ts')) out.push(path);
	}
	return out;
}

/** The lazy-singleton accessors, read from the stores themselves so a new one is covered the day it
 *  is written rather than the day someone remembers to list it here. */
function accessors(files: string[]): string[] {
	const found = new Set<string>();
	for (const file of files) {
		const src = readFileSync(file, 'utf8');
		for (const m of src.matchAll(/export function (\w+)\(\)[^{]*\{\s*if \(!(\w+)\) \2 = new/g)) {
			found.add(m[1]);
		}
	}
	return [...found];
}

/** The text of the rune's argument list starting at `open` (a `(`), read with balanced parens so a
 *  nested call or an arrow body cannot end it early. */
function argsAt(src: string, open: number): string {
	let depth = 0;
	for (let i = open; i < src.length; i++) {
		if (src[i] === '(') depth++;
		else if (src[i] === ')' && --depth === 0) return src.slice(open, i + 1);
	}
	return src.slice(open);
}

describe('lazy store singletons are never constructed inside a $derived', () => {
	const files = ROOTS.flatMap((r) => sources(r));

	it('finds the accessors it is meant to be guarding', () => {
		// A scan that quietly matches nothing buys the confidence without the cover.
		expect(accessors(files)).toEqual(
			expect.arrayContaining(['notify', 'flash', 'graph', 'history', 'ui', 'workspace'])
		);
	});

	it('no $derived calls one', () => {
		const names = accessors(files);
		const offenders: string[] = [];
		for (const file of files) {
			const src = readFileSync(file, 'utf8');
			for (const m of src.matchAll(/\$derived(\.by)?\(/g)) {
				const args = argsAt(src, m.index + m[0].length - 1);
				for (const name of names) {
					if (!new RegExp(`\\b${name}\\(\\)`).test(args)) continue;
					const line = src.slice(0, m.index).split('\n').length;
					offenders.push(`${relative(ROOTS[0], file)}:${line} reads ${name}() inside a $derived`);
				}
			}
		}
		expect(offenders, 'hoist it: `const s = store();` above, then read `s.field` in the derived').toEqual(
			[]
		);
	});
});
