import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

/**
 * Every primitive the barrel exports has a consumer in the app.
 *
 * This is what the 56-test `/dev/ui` gallery sweep used to be for: a primitive added to the barrel
 * and used nowhere escaped coverage. The stronger question is cheaper to ask — an export nothing
 * imports is dead code, which "delete before adding" says to remove rather than to sample.
 */
const SRC = join(__dirname, '../..');

function sources(dir: string, out: string[] = []): string[] {
	for (const name of readdirSync(dir)) {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) sources(p, out);
		else if (/\.(ts|svelte)$/.test(name) && !/\.test\.ts$/.test(name)) out.push(p);
	}
	return out;
}

/** The value names `$lib/ui` re-exports; types carry no runtime consumer to look for. */
function exportedNames(): string[] {
	const barrel = readFileSync(join(__dirname, 'index.ts'), 'utf8');
	const names: string[] = [];
	for (const line of barrel.split('\n')) {
		const m = line.match(/^export \{(.+)\} from/);
		if (!m) continue;
		for (const part of m[1].split(',')) {
			const t = part.trim();
			if (!t || t.startsWith('type ')) continue;
			names.push((t.split(/\s+as\s+/).pop() ?? t).trim());
		}
	}
	return names;
}

describe('the $lib/ui barrel', () => {
	it('exports nothing the app does not use', () => {
		const files = sources(SRC).filter((f) => !f.endsWith(join('lib', 'ui', 'index.ts')));
		const bodies = files.map((f) => ({ f, text: readFileSync(f, 'utf8') }));
		const orphans = exportedNames().filter(
			(name) =>
				!bodies.some(
					({ f, text }) =>
						!f.includes(join('lib', 'ui')) && new RegExp(`\\b${name}\\b`).test(text)
				)
		);
		expect(orphans, 'a primitive nothing imports is dead code, not a primitive').toEqual([]);
	});
});
