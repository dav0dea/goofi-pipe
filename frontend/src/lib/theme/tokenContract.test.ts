/**
 * The panel system's styling hand-over, held to completeness.
 *
 * The package draws every rule through `var(--tatami-x, var(--tatami-x-default))`. Two tiers, two
 * owners: the `-default` is the package's own shipped look, and `--tatami-x` is the slot this app
 * fills in one block in `app.css`. Nothing enforces either side at runtime — an unmapped token
 * simply falls through to the package's default, which is the same colour today and drifts the day
 * either palette moves. The failure is silent, invisible in review, and looks exactly like a theme
 * that works.
 *
 * So it is read off the bytes: what the package READS, what its own stylesheet DEFAULTS, and what
 * the app MAPS have to be one set. The scan is deliberately whole-directory rather than a list —
 * a token added in a new component is covered the moment it is written.
 */
import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const PKG = fileURLToPath(new URL('../workspace', import.meta.url));
const TOKENS = fileURLToPath(new URL('../workspace/ui/tokens.css', import.meta.url));
const APP_CSS = fileURLToPath(new URL('../../app.css', import.meta.url));

/**
 * The per-instance group: a hook a consumer sets on ONE strip or ONE button, not a theme token.
 * Each states its fallback inline against a shared-base token, so it needs no `-default` tier and
 * an app that never sets it still gets a styled control.
 */
const HOOKS = new Set([
	'tab-surface',
	'tab-body',
	'tab-fs',
	'tab-pad',
	'tab-align',
	'btn-ink',
	'icon-btn-size'
]);

function sources(dir: string): string[] {
	const out: string[] = [];
	for (const entry of readdirSync(dir)) {
		const p = join(dir, entry);
		if (statSync(p).isDirectory()) out.push(...sources(p));
		else if (/\.(svelte|css)$/.test(entry) && entry !== 'tokens.css') out.push(p);
	}
	return out;
}

const packageCss = (): string =>
	sources(PKG)
		.map((p) => readFileSync(p, 'utf8'))
		.join('\n');

/** Every `--tatami-x` the package READS, `-default` tier excluded — the contract itself. */
function read(): Set<string> {
	const out = new Set<string>();
	for (const [, name] of packageCss().matchAll(/var\(\s*--tatami-([a-z0-9-]+)/g))
		if (!name.endsWith('-default')) out.add(name);
	return out;
}

/** Every `--tatami-x…: ` one stylesheet DECLARES. */
function declared(file: string, suffix = ''): Set<string> {
	const src = readFileSync(file, 'utf8');
	const out = new Set<string>();
	for (const [, name] of src.matchAll(/^\t--tatami-([a-z0-9-]+)\s*:/gm))
		if (suffix ? name.endsWith(suffix) : !name.endsWith('-default'))
			out.add(suffix ? name.slice(0, -suffix.length) : name);
	return out;
}

describe('the tatami styling contract', () => {
	const themed = (): string[] => [...read()].filter((n) => !HOOKS.has(n)).sort();

	it('reads a contract at all, and every read is one of the three groups', () => {
		const names = [...read()];
		expect(names.length, 'the package draws through tokens').toBeGreaterThan(20);
		// Shared base, `panel-*`, `tab-*` — nothing else, so a fourth group cannot appear unnoticed.
		const stray = names.filter((n) => /^(panel|tab)$/.test(n) || n.endsWith('-'));
		expect(stray, 'no half-named token').toEqual([]);
	});

	it('ships a default for every themed token it reads', () => {
		const missing = themed().filter((n) => !declared(TOKENS, '-default').has(n));
		expect(missing, 'every token the package reads has a shipped default').toEqual([]);
	});

	it('defaults nothing it does not read — a stale default is a value nobody can see', () => {
		const dead = [...declared(TOKENS, '-default')].filter((n) => !read().has(n)).sort();
		expect(dead).toEqual([]);
	});

	it('gives the per-instance hooks no default tier — their fallback is inline', () => {
		const overreach = [...HOOKS].filter((n) => declared(TOKENS, '-default').has(n)).sort();
		expect(overreach, 'a hook falls back to a shared token, not to a tier of its own').toEqual([]);
	});

	it('is mapped by this app, completely, and only where the package asks', () => {
		const mapped = declared(APP_CSS);
		const unmapped = themed().filter((n) => !mapped.has(n));
		expect(unmapped, 'app.css maps every themed token the package reads').toEqual([]);
		const spurious = [...mapped].filter((n) => !read().has(n)).sort();
		expect(spurious, 'and maps nothing the package does not read').toEqual([]);
	});
});
