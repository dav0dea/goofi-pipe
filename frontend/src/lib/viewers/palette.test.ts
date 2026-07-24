import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { DTYPE_INK, SERIES } from './palette';

const css = readFileSync(fileURLToPath(new URL('../../app.css', import.meta.url)), 'utf8');
function cssToken(name: string): string {
	const m = new RegExp(`--${name}\\s*:\\s*(#[0-9a-fA-F]{6})`).exec(css);
	if (!m) throw new Error(`--${name} not found`);
	return m[1].toLowerCase();
}

describe('canvas palette agrees with the CSS dtype tokens', () => {
	it('DTYPE_INK matches --dtype-* (canvas cannot read var(), so this pins the two together)', () => {
		expect(DTYPE_INK.array.toLowerCase()).toBe(cssToken('dtype-array'));
		expect(DTYPE_INK.string.toLowerCase()).toBe(cssToken('dtype-string'));
		expect(DTYPE_INK.table.toLowerCase()).toBe(cssToken('dtype-table'));
	});
	it('SERIES is a single non-empty list of #rrggbb (no 8-vs-7 drift)', () => {
		expect(SERIES.length).toBeGreaterThan(0);
		for (const c of SERIES) expect(c).toMatch(/^#[0-9a-fA-F]{6}$/);
	});
});
